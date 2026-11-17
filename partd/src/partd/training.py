from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import save_artifact
from .common import error_result, identifier, now, read_json, resolve_path, sha256, source_fingerprint, versions, write_json
from .data import load_csv, write_predictions
from .metrics import DIRECTIONS, calculate_metrics


def build_pipeline(model, seed, preprocessing):
    if not isinstance(model, dict) or set(model) - {"name", "params"}:
        raise ValueError("model accepts only name and params; hyperparameters belong inside params")
    params = dict(model.get("params", {}))
    if "random_state" in params and params["random_state"] != seed:
        raise ValueError("set random_state through seed; conflicting model parameter")
    params["random_state"] = seed
    if model["name"] == "logistic_regression":
        params.setdefault("max_iter", 1000)
        estimator = LogisticRegression(**params)
    elif model["name"] == "random_forest":
        params.setdefault("n_estimators", 100)
        params.setdefault("n_jobs", 1)
        estimator = RandomForestClassifier(**params)
    else:
        raise ValueError("model.name must be logistic_regression or random_forest")
    if preprocessing not in {"numeric", "passthrough"}:
        raise ValueError("preprocessing must be numeric or passthrough")
    steps = []
    if preprocessing == "numeric":
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
        if model["name"] == "logistic_regression":
            steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def train(request_path):
    """One experiment only. No search, test-set evaluation or final selection."""
    started = time.monotonic()
    directory = None
    context = {}
    stage = "request"
    try:
        request_path = Path(request_path).resolve()
        request = read_json(request_path)
        if not isinstance(request, dict) or request.get("schema_version") != 1:
            raise ValueError("request.schema_version must be 1")
        allowed = {"schema_version", "task_id", "run_id", "experiment_id", "data", "model", "metrics", "seed", "preprocessing", "output_dir"}
        if set(request) - allowed:
            raise ValueError(f"unknown request fields: {sorted(set(request) - allowed)}")
        context = {name: identifier(request[name], name) for name in ("task_id", "run_id", "experiment_id")}
        root = resolve_path(request["output_dir"], request_path.parent)
        experiment_dir = root / context["task_id"] / context["run_id"] / context["experiment_id"]
        experiment_dir.mkdir(parents=True, exist_ok=False)
        directory = experiment_dir
        write_json(directory / "request.json", request)
        stage = "data_preparation"
        data = request["data"]
        input_source = data.get("source", "numeric_csv")
        prepared = None
        if input_source == "partb":
            from .partb_adapter import prepare_training
            preprocessing = request.get("preprocessing", "partb")
            if preprocessing != "partb":
                raise ValueError("Part B inputs require preprocessing=partb; D must not fit another numeric preprocessor")
            prepared = prepare_training(data, request_path.parent, directory)
            training, validation = prepared["train"], prepared["validation"]
            target, id_column = prepared["target"], prepared["id_column"]
            provenance = prepared["provenance"]
        elif input_source == "numeric_csv":
            allowed_data = {"source", "train_csv", "validation_csv", "target_column", "id_column", "feature_columns"}
            if set(data) - allowed_data:
                raise ValueError(f"unknown data fields: {sorted(set(data) - allowed_data)}; designated test data is not accepted here")
            train_path = resolve_path(data["train_csv"], request_path.parent)
            val_path = resolve_path(data["validation_csv"], request_path.parent)
            if train_path == val_path:
                raise ValueError("training and validation must be separate files")
            target = data.get("target_column", "label")
            id_column = data.get("id_column", "sample_id")
            training = load_csv(train_path, target_column=target, id_column=id_column, features=data.get("feature_columns"))
            validation = load_csv(val_path, target_column=target, id_column=id_column, features=training.features)
            provenance = {"source": input_source, "train_sha256": sha256(train_path), "validation_sha256": sha256(val_path),
                          "n_train": len(training.ids), "n_validation": len(validation.ids)}
            preprocessing = request.get("preprocessing", "numeric")
        else:
            raise ValueError("data.source must be numeric_csv or partb")
        if set(training.ids) & set(validation.ids):
            raise ValueError("training and validation sample IDs overlap")
        if len(set(training.y)) < 2:
            raise ValueError("training requires at least two classes")
        if set(validation.y) - set(training.y):
            raise ValueError("validation contains labels absent from training")
        metric_spec = request.get("metrics", {"primary": prepared["metric"] if prepared else "accuracy", "additional": ["f1_macro"]})
        stage = "model_configuration"
        if not isinstance(metric_spec, dict) or set(metric_spec) - {"primary", "additional"}:
            raise ValueError("metrics accepts only primary and additional")
        if not isinstance(metric_spec.get("additional", []), list):
            raise ValueError("metrics.additional must be a list")
        primary = metric_spec["primary"]
        names = list(dict.fromkeys([primary, *metric_spec.get("additional", [])]))
        if any(name not in DIRECTIONS for name in names):
            raise ValueError(f"unsupported metrics: {names}; supported: {sorted(DIRECTIONS)}")
        seed = request.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        pipeline = build_pipeline(request["model"], seed, "passthrough" if prepared else preprocessing)
        if preprocessing == "passthrough" and (np.isnan(training.X).any() or np.isnan(validation.X).any()):
            raise ValueError("passthrough inputs must already have missing values handled by B")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stage = "training"
            pipeline.fit(training.X, training.y)
        stage = "validation"
        predicted = pipeline.predict(validation.X)
        probabilities = pipeline.predict_proba(validation.X)
        classes = pipeline.classes_.tolist()
        metrics = calculate_metrics(validation.y, predicted, probabilities, classes, names)
        metadata = {**context, "features": training.features, "classes": classes,
                    "target_column": target, "id_column": id_column, "seed": seed,
                    "preprocessing": preprocessing, "model": request["model"], "created_at": now(),
                    "partd_version": "0.2.0", "partd_source_sha256": source_fingerprint(),
                    "input_source": input_source, "data": provenance}
        if prepared:
            metadata["partb"] = prepared["partb"]
        stage = "artifact_save"
        artifact = save_artifact(directory, pipeline, metadata,
                                 preprocessor=prepared["processor"] if prepared else None,
                                 task_config=prepared["config"] if prepared else None)
        write_predictions(directory / "validation_predictions.csv", validation.ids, predicted, probabilities, validation.y)
        result = {"event": "partd.result", "schema_version": 1, "command": "train", "status": "success", **context,
                  "metrics": metrics, "primary_metric": {"name": primary, "direction": DIRECTIONS[primary], "value": metrics[primary]},
                  "artifact": artifact, "data": metadata["data"], "versions": versions(include_partb=bool(prepared)),
                  "input_source": input_source,
                  "partd_source_sha256": metadata["partd_source_sha256"],
                  "warnings": [{"category": warning.category.__name__, "message": str(warning.message)} for warning in caught],
                  "data_warnings": prepared["warnings"] if prepared else [],
                  "validation_predictions_path": str((directory / "validation_predictions.csv").resolve()),
                  "elapsed_seconds": time.monotonic() - started}
    except Exception as exc:
        result = error_result("train", exc, stage=stage, **context)
    if directory is not None:
        result["result_path"] = str((directory / "result.json").resolve())
        write_json(directory / "result.json", result)
    return result
