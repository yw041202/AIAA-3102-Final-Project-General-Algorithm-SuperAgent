"""Small synthetic data for interface acceptance, never a course result."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .common import write_json


def create_demo_inputs(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261115)
    for split, count, labeled in (("train", 180, True), ("validation", 60, True), ("prediction", 30, False)):
        X = rng.normal(size=(count, 6))
        y = np.where(X[:, 0] + 0.7 * X[:, 1] - 0.4 * X[:, 2] + rng.normal(scale=0.4, size=count) > 0,
                     "positive", "negative")
        X[rng.random(X.shape) < 0.04] = np.nan
        with (directory / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", *[f"feature_{i}" for i in range(6)], *(["label"] if labeled else [])])
            for index in range(count):
                writer.writerow([f"{split}_{index:04d}", *["" if np.isnan(value) else float(value) for value in X[index]],
                                 *([y[index]] if labeled else [])])
    request = {"schema_version": 1, "task_id": "synthetic_demo", "run_id": "development",
               "experiment_id": "logistic_direct", "data": {"train_csv": "train.csv", "validation_csv": "validation.csv",
               "id_column": "sample_id", "target_column": "label"},
               "model": {"name": "logistic_regression", "params": {"C": 1.0}}, "preprocessing": "numeric", "seed": 42,
               "metrics": {"primary": "accuracy", "additional": ["f1_macro", "balanced_accuracy", "roc_auc", "log_loss"]},
               "output_dir": "artifacts"}
    write_json(directory / "request.json", request)
    request_c = {**request, "experiment_id": "logistic_via_c"}
    write_json(directory / "request_via_c.json", request_c)
    write_json(directory / "DATA_NOTICE.json", {"synthetic": True, "purpose": "development interface acceptance only",
                                               "n_train": 180, "n_validation": 60, "n_prediction": 30, "seed": 20261115})
    return directory / "request.json"


def create_partb_inputs(directory, partb_root, modality="tabular"):
    """Raw categorical/numeric or paired text fixtures, using B's real APIs later."""
    if modality not in {"tabular", "text"}:
        raise ValueError("fixture modality must be tabular or text")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261117)
    features = ["category", "amount"] if modality == "tabular" else ["left", "right"]
    for split, count, labeled in (("train", 120, True), ("val", 40, True), ("prediction", 20, False)):
        with (directory / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", *features, *(["label"] if labeled else [])])
            for i in range(count):
                label = i % 2
                if modality == "tabular":
                    values = ["red" if label else "blue", float(rng.normal(3 if label else -3, 1))]
                    if i % 13 == 0:
                        values[1] = ""
                    if split != "train" and i == 0:
                        values[0] = "unseen_category"
                else:
                    values = ["bright good title" if label else "dark bad title", "dark bad title" if label else "bright good title"]
                    if split != "train":
                        values[0] += " validationonly"
                writer.writerow([f"{split}:{i:04d}", *values, *([label] if labeled else [])])
    config = {"modality": modality, "format": "csv", "feature_columns": features,
              "id_column": "sample_id", "label_column": "label", "allowed_labels": [0, 1], "metric": "accuracy",
              "splits": {"train": {"path": "train.csv"}, "val": {"path": "val.csv"},
                         "prediction": {"path": "prediction.csv"}, "test": {"path": "must_not_read_test.csv"}}}
    write_json(directory / "config.json", config)
    options = {"numeric_columns": ["amount"]} if modality == "tabular" else {"combine": "difference", "ngram_range": [1, 1]}
    request = {"schema_version": 1, "task_id": f"synthetic_b_{modality}", "run_id": "development",
               "experiment_id": "direct", "data": {"source": "partb", "config_path": "config.json", "data_root": ".",
               "partb_root": str(Path(partb_root).resolve()), "preprocess_options": options},
               "model": {"name": "logistic_regression", "params": {"C": 1.0}}, "preprocessing": "partb", "seed": 42,
               "metrics": {"primary": "accuracy", "additional": ["f1_macro", "roc_auc", "log_loss"]}, "output_dir": "artifacts"}
    write_json(directory / "request.json", request)
    write_json(directory / "request_via_c.json", {**request, "experiment_id": "via_c"})
    write_json(directory / "DATA_NOTICE.json", {"synthetic": True, "modality": modality, "seed": 20261117,
                                               "purpose": "B/D/C interface acceptance; no course test data", "counts": [120, 40, 20]})
    return directory / "request.json"
