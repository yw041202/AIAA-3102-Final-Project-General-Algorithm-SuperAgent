"""Frozen pretrained image features and reusable classification heads.

Torch is imported lazily. Existing tabular/text paths need no torch installation.
No optimizer, augmentation or fine-tuning is implemented in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np

from .common import identifier, now, read_json, resolve_path, sha256, source_fingerprint, versions, write_json
from .data import write_predictions
from .metrics import DIRECTIONS, calculate_metrics
from .partb_adapter import code_fingerprint, import_partb

BACKBONES = {
    "resnet18": {"filename": "resnet18-f37072fd.pth", "sha256_prefix": "f37072fd", "dimension": 512,
        "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"},
    "efficientnet_b0": {"filename": "efficientnet_b0_rwightman-7f5810bc.pth", "sha256_prefix": "7f5810bc", "dimension": 1280,
        "sha256": "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934"},
}


def vision_versions():
    import torch
    import torchvision
    return {**versions(include_partb=True), "torch": torch.__version__, "torchvision": torchvision.__version__}


def request_context(request):
    if request.get("schema_version") != 1:
        raise ValueError("vision request requires schema_version=1")
    return {key: identifier(request[key], key) for key in ("task_id", "run_id", "experiment_id")}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_encoder(name, weight_path, *, threads=4, preprocessing="official"):
    import torch
    from torch import nn
    from torchvision import models, transforms
    from torchvision.transforms import InterpolationMode
    if name not in BACKBONES:
        raise ValueError(f"unsupported frozen backbone: {name}")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    weight_path = Path(weight_path).resolve()
    digest = sha256(weight_path)
    if digest != BACKBONES[name]["sha256"]:
        raise ValueError("pretrained weights do not match the pinned official SHA256")
    constructor, weight_enum = ((models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1)
        if name == "resnet18" else (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1))
    model = constructor(weights=None)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if name == "resnet18":
        model.fc = nn.Identity()
    else:
        model.classifier = nn.Identity()
    model.requires_grad_(False)
    model.eval()
    if preprocessing == "official":
        transform = weight_enum.transforms()
    elif preprocessing == "resize224":
        transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    else:
        raise ValueError("preprocessing must be official or resize224")
    metadata = {"backbone": name, "weights": f"{type(weight_enum).__name__}.IMAGENET1K_V1",
        "source_url": weight_enum.url, "weight_sha256": digest,
        "feature_dimension": BACKBONES[name]["dimension"], "preprocessing": preprocessing,
        "transform": repr(transform), "rgb_conversion": "PIL EXIF transpose then RGB",
        "frozen": True, "eval_mode": True, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": "cpu", "threads": threads}
    return model, transform, metadata


def parameter_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def describe_split(split):
    """Ordered image/ID/label identity; reads development inputs only."""
    digest = hashlib.sha256()
    image_hashes = []
    for sample_id, path in zip(split.sample_ids, split.x.iloc[:, 0]):
        image_hash = sha256(path)
        image_hashes.append(image_hash)
        digest.update(json_digest([str(sample_id), image_hash]).encode())
    labels, counts = np.unique(split.y, return_counts=True)
    return {"rows": len(split.x), "class_counts": dict(zip(labels.astype(str).tolist(), counts.tolist())),
        "image_content_sha256": digest.hexdigest(),
        "ids_sha256": json_digest(split.sample_ids.astype(str).tolist()),
        "labels_sha256": json_digest(split.y.astype(str).tolist()),
        "image_hashes": image_hashes}


def development_description(train, val):
    if set(train.sample_ids) & set(val.sample_ids):
        raise ValueError("training and validation sample IDs overlap")
    descriptions = {s.name: describe_split(s) for s in (train, val)}
    if set(descriptions["train"]["image_hashes"]) & set(descriptions["val"]["image_hashes"]):
        raise ValueError("identical image bytes occur in training and validation")
    for value in descriptions.values():
        del value["image_hashes"]
    return descriptions


def encode_paths(model, transform, paths, *, batch_size, progress_path=None, split=None):
    import torch
    from PIL import Image, ImageOps
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs = []
    begin = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
            features = model(torch.stack(tensors)).cpu().numpy().astype(np.float32, copy=False)
            if features.ndim != 2 or not np.isfinite(features).all():
                raise ValueError("invalid frozen image features")
            outputs.append(features)
            if progress_path:
                write_json(progress_path, {"split": split, "processed": min(start + batch_size, len(paths)),
                    "total": len(paths), "elapsed_seconds": time.perf_counter() - begin})
    return np.concatenate(outputs), time.perf_counter() - begin


def extract_features(request_path):
    begin = time.perf_counter()
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    context = request_context(request)
    output = resolve_path(request["output_dir"], request_path.parent)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", request)
    module = import_partb(request["partb_root"])
    config_path = resolve_path(request["config_path"], request_path.parent)
    data_root = resolve_path(request["data_root"], request_path.parent)
    loader = module.DataModule(config_path, data_root)
    if loader.modality != "image":
        raise ValueError("vision extraction requires an image task")
    train, val = loader.load_development()
    limit = request.get("max_per_class")
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("max_per_class must be positive")
        selected = []
        for split in (train, val):
            indices = np.concatenate([np.flatnonzero(split.y == label)[:limit] for label in np.unique(split.y)])
            indices.sort()
            selected.append(module.DataSplit(split.name, split.x.iloc[indices].reset_index(drop=True), split.y[indices], split.sample_ids[indices]))
        train, val = selected
    descriptions = development_description(train, val)
    spec = request["encoder"]
    model, transform, encoder_meta = load_encoder(spec["name"], spec["weight_path"],
        threads=request.get("threads", 4), preprocessing=spec.get("preprocessing", "official"))
    before = parameter_digest(model)
    datasets, file_hashes = {}, {}
    for split in (train, val):
        paths = split.x.iloc[:, 0].astype(str).tolist()
        X, elapsed = encode_paths(model, transform, paths, batch_size=request.get("batch_size", 16),
            progress_path=output / "progress.json", split=split.name)
        if X.shape[1] != encoder_meta["feature_dimension"]:
            raise ValueError("encoder feature dimension differs from contract")
        np.save(output / f"{split.name}_features.npy", X, allow_pickle=False)
        np.save(output / f"{split.name}_labels.npy", split.y.astype(str), allow_pickle=False)
        write_json(output / f"{split.name}_ids.json", split.sample_ids.astype(str).tolist())
        for suffix in ("features.npy", "labels.npy", "ids.json"):
            filename = f"{split.name}_{suffix}"
            file_hashes[filename] = sha256(output / filename)
        datasets[split.name] = {**descriptions[split.name], "shape": list(X.shape), "encoding_seconds": elapsed}
    after = parameter_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen backbone state changed during extraction")
    metadata = {"schema_version": 1, "kind": "frozen_image_features", **context, "created_at": now(),
        "encoder": encoder_meta, "weight_path": str(Path(spec["weight_path"]).resolve()),
        "config": read_json(config_path), "config_sha256": sha256(config_path), "data_root": str(data_root),
        "features": loader.features, "target_column": loader.config.get("label_column", "label"),
        "id_column": loader.config.get("id_column", "sample_id"), "partb_sha256": code_fingerprint(module),
        "datasets": datasets, "files": file_hashes, "versions": vision_versions(),
        "source_sha256": source_fingerprint(), "backbone_state_before": before, "backbone_state_after": after,
        "batch_size": request.get("batch_size", 16), "sampled_for_smoke_only": limit is not None,
        "test_set_used": False, "elapsed_seconds": time.perf_counter() - begin}
    metadata["cache_key"] = json_digest({"encoder": encoder_meta, "versions": metadata["versions"], "datasets": {
        key: {k: v for k, v in value.items() if k != "encoding_seconds"} for key, value in datasets.items()}, "partb": metadata["partb_sha256"]})
    write_json(output / "features.json", metadata)
    result = {"event": "partd.result", "schema_version": 1, "command": "vision-extract", "status": "success", **context,
        "feature_directory": str(output), "metadata": metadata}
    write_json(output / "result.json", result)
    return result


def read_feature_cache(directory):
    directory = Path(directory)
    metadata = read_json(directory / "features.json")
    if metadata.get("kind") != "frozen_image_features" or metadata["versions"] != vision_versions():
        raise ValueError("invalid feature cache or incompatible environment")
    for name, expected in metadata["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"feature cache checksum mismatch: {name}")
    if metadata["source_sha256"] != source_fingerprint():
        raise ValueError("D source changed since feature extraction")
    return metadata


def fit_head(request_path):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    begin = time.perf_counter()
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    context = request_context(request)
    directory = resolve_path(request["output_dir"], request_path.parent)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", request)
    feature_dir = resolve_path(request["feature_directory"], request_path.parent)
    features = read_feature_cache(feature_dir)
    if features["task_id"] != context["task_id"] or features["run_id"] != context["run_id"]:
        raise ValueError("feature task_id/run_id differ from head request")
    X_train = np.load(feature_dir / "train_features.npy", allow_pickle=False)
    X_val = np.load(feature_dir / "val_features.npy", allow_pickle=False)
    y_train = np.load(feature_dir / "train_labels.npy", allow_pickle=False)
    y_val = np.load(feature_dir / "val_labels.npy", allow_pickle=False)
    ids = read_json(feature_dir / "val_ids.json")
    head = request["head"]
    if head.get("name") not in {"logistic_regression", "linear_svm"}:
        raise ValueError("frozen image experiments support logistic_regression and linear_svm")
    seed = request.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid seed")
    scaling = request.get("feature_scaling", "standard")
    if scaling not in {"standard", "l2", "none"}:
        raise ValueError("feature_scaling must be standard, l2 or none")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    params = dict(head.get("params", {}))
    if params.get("random_state", seed) != seed:
        raise ValueError("set random_state through seed")
    params["random_state"] = seed
    calibrate = params.pop("calibrate", False)
    if type(calibrate) is not bool or (calibrate and head["name"] != "linear_svm"):
        raise ValueError("calibrate is a boolean option for linear_svm only")
    if head["name"] == "logistic_regression":
        params.setdefault("max_iter", 1000)
        classifier = LogisticRegression(**params)
    else:
        params.setdefault("max_iter", 10000)
        params.setdefault("dual", "auto")
        classifier = LinearSVC(**params)
        if calibrate:
            classifier = CalibratedClassifierCV(classifier, method="sigmoid", cv=3, n_jobs=1)
    spec = request.get("metrics", {"primary": features["config"]["metric"], "additional": ["f1_macro", "balanced_accuracy"]})
    if not isinstance(spec, dict) or set(spec) - {"primary", "additional"} or not isinstance(spec.get("additional", []), list):
        raise ValueError("metrics accepts primary and an additional list")
    primary = spec["primary"]
    names = list(dict.fromkeys([primary, *spec.get("additional", [])]))
    if any(name not in DIRECTIONS for name in names):
        raise ValueError("unsupported image validation metric")
    if "log_loss" in names and not hasattr(classifier, "predict_proba"):
        raise ValueError("log_loss requires LR or linear_svm with params.calibrate=true")
    if "roc_auc" in names and len(np.unique(y_train)) > 2 and not hasattr(classifier, "predict_proba"):
        raise ValueError("multiclass roc_auc requires LR or calibrated linear_svm")
    steps = []
    if scaling == "standard": steps.append(("feature_scaler", StandardScaler()))
    if scaling == "l2": steps.append(("feature_normalizer", Normalizer()))
    pipeline = Pipeline(steps + [("head", classifier)])
    fit_begin = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_begin
    validation_begin = time.perf_counter()
    predicted = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
    scores = pipeline.decision_function(X_val) if probabilities is None and hasattr(pipeline, "decision_function") else None
    metrics = calculate_metrics(y_val, predicted, probabilities, pipeline.classes_.tolist(), names, decision_scores=scores)
    validation_seconds = time.perf_counter() - validation_begin
    write_predictions(directory / "validation_predictions.csv", ids, predicted, probabilities, truth=y_val)
    joblib.dump(pipeline, directory / "head.joblib", compress=3)
    weight_source = Path(features["weight_path"])
    weight_target = directory / "backbone.pth"
    # Hard links are actual self-contained files, not links requiring the source path.
    try:
        os.link(weight_source, weight_target)
    except OSError:
        shutil.copyfile(weight_source, weight_target)
    write_json(directory / "task_config.json", features["config"])
    manifest = {"schema_version": 1, "kind": "frozen_vision_classifier", **context,
        "created_at": now(), "encoder": features["encoder"], "head": head, "feature_scaling": scaling,
        "classes": pipeline.classes_.tolist(), "features": features["features"],
        "target_column": features["target_column"], "id_column": features["id_column"],
        "partb_sha256": features["partb_sha256"], "source_sha256": source_fingerprint(), "seed": seed,
        "versions": vision_versions(), "feature_cache_key": features["cache_key"],
        "data": features["datasets"], "frozen_backbone_unchanged": features["backbone_state_before"] == features["backbone_state_after"],
        "files": {name: sha256(directory / name) for name in ("head.joblib", "backbone.pth", "task_config.json")}}
    write_json(directory / "image_artifact.json", manifest)
    result = {"event": "partd.result", "schema_version": 1, "command": "vision-fit", "status": "success", **context,
        "metrics": metrics, "primary_metric": {"name": primary, "direction": DIRECTIONS[primary], "value": metrics[primary]},
        "artifact": {"directory": str(directory), "manifest_path": str(directory / "image_artifact.json")},
        "encoder": features["encoder"], "head": head, "feature_scaling": scaling,
        "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - begin,
        "training": {"fit_seconds": fit_seconds, "validation_seconds": validation_seconds,
            "has_probabilities": probabilities is not None,
            "convergence_warning": any(w.category.__name__ == "ConvergenceWarning" for w in caught)},
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        "validation_predictions_path": str(directory / "validation_predictions.csv"), "test_set_used": False}
    write_json(directory / "result.json", result)
    return result


def load_image_artifact(directory, partb_root=None):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "image_artifact.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "frozen_vision_classifier":
        raise ValueError("unsupported image artifact")
    if manifest["versions"] != vision_versions():
        raise ValueError("image artifact environment differs from training")
    if manifest["source_sha256"] != source_fingerprint():
        raise ValueError("image artifact D source differs from training")
    module = import_partb(partb_root)
    if manifest["partb_sha256"] != code_fingerprint(module):
        raise ValueError("image artifact B source differs from training")
    if set(manifest["files"]) != {"head.joblib", "backbone.pth", "task_config.json"}:
        raise ValueError("incomplete image artifact")
    for filename, expected in manifest["files"].items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"image artifact checksum mismatch: {filename}")
    pipeline = joblib.load(directory / "head.joblib")
    if pipeline.classes_.tolist() != manifest["classes"]:
        raise ValueError("saved class mapping differs")
    return module, pipeline, manifest


def predict_images(artifact_dir, data_root, output, *, split="test", partb_root=None, batch_size=16, include_probabilities=False):
    begin = time.perf_counter()
    directory, output = Path(artifact_dir).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".metadata.json").exists():
        raise FileExistsError("image prediction output exists")
    module, pipeline, manifest = load_image_artifact(directory, partb_root)
    loader = module.DataModule(directory / "task_config.json", data_root)
    data = loader.load_prediction(split=split)  # Inputs and IDs only; no targets or fitting.
    if loader.features != manifest["features"] or loader.modality != "image":
        raise ValueError("prediction modality/features differ from training")
    encoder, transform, meta = load_encoder(manifest["encoder"]["backbone"], directory / "backbone.pth",
        threads=manifest["encoder"]["threads"], preprocessing=manifest["encoder"]["preprocessing"])
    if meta != manifest["encoder"]:
        raise ValueError("image preprocessing differs from training")
    X, encoding_seconds = encode_paths(encoder, transform, data.x.iloc[:, 0].astype(str).tolist(), batch_size=batch_size)
    predicted = pipeline.predict(X)
    if include_probabilities and not hasattr(pipeline, "predict_proba"):
        raise ValueError("this image head has no probabilities; use LR or a training-only calibrated SVM")
    probabilities = pipeline.predict_proba(X) if include_probabilities else None
    write_predictions(output, data.sample_ids.astype(str).tolist(), predicted, probabilities)
    result = {"event": "partd.result", "schema_version": 1, "command": "vision-predict", "status": "success",
        "task_id": manifest["task_id"], "run_id": manifest["run_id"], "source_experiment_id": manifest["experiment_id"],
        "n_predictions": len(predicted), "output_path": str(output), "classes": manifest["classes"],
        "probability_columns": {f"probability_{i}": label for i, label in enumerate(manifest["classes"])} if include_probabilities else {},
        "labels_loaded": False, "fit_called": False, "split": split,
        "encoding_seconds": encoding_seconds, "elapsed_seconds": time.perf_counter() - begin,
        "versions": vision_versions(), "artifact_manifest_sha256": sha256(directory / "image_artifact.json")}
    write_json(output.with_suffix(output.suffix + ".metadata.json"), result)
    return result
