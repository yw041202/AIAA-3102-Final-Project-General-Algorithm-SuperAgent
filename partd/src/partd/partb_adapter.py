"""B's public API adapter. Only development data may enter training."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from .common import read_json, resolve_path, sha256, source_fingerprint, write_json
from .data import Dataset


def import_partb(root=None):
    root = root or os.environ.get("PARTB_ROOT")
    expected = None
    if root:
        path = Path(root).expanduser().resolve()
        package = path if (path / "__init__.py").is_file() else path / "partb"
        if not (package / "preprocessing.py").is_file():
            raise ValueError("partb_root must be a Part B package directory or its repository root")
        expected = package.resolve()
        sys.path.insert(0, str(package.parent))
    try:
        module = importlib.import_module("partb")
    except ImportError as exc:
        raise ImportError("install Part B in this Python environment or supply partb_root / PARTB_ROOT") from exc
    if expected and Path(module.__file__).resolve().parent != expected:
        raise ImportError("a different Part B is already imported; start a new process with the requested partb_root")
    return module


def code_fingerprint(module):
    digest = hashlib.sha256()
    for name in ("__init__.py", "data.py", "preprocessing.py"):
        digest.update(name.encode("utf-8"))
        digest.update((Path(module.__file__).parent / name).read_bytes())
    return digest.hexdigest()


def frame_fingerprint(split):
    import pandas as pd
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(split.x, index=False).to_numpy().tobytes())
    digest.update(json.dumps(split.sample_ids.astype(str).tolist(), ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def validate_matrix(matrix):
    if matrix.ndim != 2:
        raise ValueError("this milestone supports 2-D tabular/text features; image training is a later milestone")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError("B returned non-finite features after preprocessing")


def prepare_training(data, base, directory, model_name=None):
    allowed = {"source", "config_path", "data_root", "partb_root", "preprocess_options"}
    if set(data) - allowed:
        raise ValueError(f"unknown Part B data fields: {sorted(set(data) - allowed)}")
    root = str(resolve_path(data["partb_root"], base)) if data.get("partb_root") else None
    module = import_partb(root)
    config_path = resolve_path(data["config_path"], base)
    config = read_json(config_path)
    loader = module.DataModule(config_path, resolve_path(data["data_root"], base))
    if loader.modality not in {"tabular", "text"}:
        raise ValueError("Part D V0.2 supports B tabular/text training; image training remains a later milestone")
    training, validation = loader.load_development()
    options = data.get("preprocess_options", {})
    if not isinstance(options, dict):
        raise ValueError("preprocess_options must be an object")
    if model_name in {"catboost", "hist_gradient_boosting"} and loader.modality != "tabular":
        raise ValueError("native categorical boosting requires tabular data")
    backend = {"catboost": "catboost", "hist_gradient_boosting": "hist"}.get(model_name)
    if options.get("analyzer") == "word_char": backend = "word_char"
    if backend:
        from .advanced_preprocessing import AdvancedPreprocessor
        if (backend == "word_char") != (loader.modality == "text"):
            raise ValueError("candidate preprocessing backend/modality mismatch")
        processor = AdvancedPreprocessor(loader.modality, options, backend).fit(training)
    else:
        processor = module.Preprocessor(loader.modality, options).fit(training)
    X_train, X_val = processor.transform(training.x), processor.transform(validation.x)
    if backend not in {"catboost", "hist"}:
        validate_matrix(X_train)
        validate_matrix(X_val)
    train = Dataset(X_train, training.y, training.sample_ids.astype(str).tolist(), list(training.x.columns))
    val = Dataset(X_val, validation.y, validation.sample_ids.astype(str).tolist(), list(validation.x.columns))
    audit = loader.audit()
    write_json(Path(directory) / "data_audit.json", audit)
    provenance = {"source": "partb", "config_sha256": sha256(config_path),
                  "n_train": len(train.ids), "n_validation": len(val.ids),
                  "train_sha256": frame_fingerprint(training), "validation_sha256": frame_fingerprint(validation),
                  "train_labels_sha256": hashlib.sha256(json.dumps(training.y.tolist(), ensure_ascii=False).encode()).hexdigest(),
                  "validation_labels_sha256": hashlib.sha256(json.dumps(validation.y.tolist(), ensure_ascii=False).encode()).hexdigest(),
                  "feature_shape": list(X_train.shape), "sparse": sparse.issparse(X_train),
                  "split_policy": audit["split_policy"]}
    return {"train": train, "validation": val, "processor": processor, "config": config,
            "target": config.get("label_column", "label"), "id_column": config.get("id_column", "sample_id"),
            "model_parameters": processor.model_parameters() if backend else {},
            "provenance": provenance, "partb": {"modality": loader.modality, "source_sha256": code_fingerprint(module),
            "preprocess_options": processor.config}, "warnings": audit["warnings"], "metric": config["metric"]}


def load_preprocessor(directory, manifest, root=None):
    module = import_partb(root)
    if code_fingerprint(module) != manifest["partb"]["source_sha256"]:
        raise ValueError("Part B source differs from the training artifact; use the recorded version")
    if manifest["partb"]["preprocess_options"].get("d_backend") and manifest["partd_source_sha256"] != source_fingerprint():
        raise ValueError("advanced preprocessing D source differs from training")
    processor = module.Preprocessor.load(Path(directory) / "preprocessor.joblib")
    if (processor.columns != manifest["features"] or processor.config != manifest["partb"]["preprocess_options"]
            or processor.modality != manifest["partb"]["modality"]):
        raise ValueError("saved B preprocessing does not match artifact metadata")
    return module, processor


def prepare_prediction(directory, manifest, *, data_root, config_path=None, split="test", partb_root=None):
    module, processor = load_preprocessor(directory, manifest, partb_root)
    path = Path(config_path).resolve() if config_path else Path(directory) / "task_config.json"
    loader = module.DataModule(path, data_root)
    if loader.modality != manifest["partb"]["modality"] or loader.features != manifest["features"]:
        raise ValueError("prediction configuration modality/features differ from the saved artifact")
    data = loader.load_prediction(split=split)  # no labels, train loading or fitting
    matrix = processor.transform(data.x)
    if manifest["partb"]["preprocess_options"].get("d_backend") not in {"catboost", "hist"}:
        validate_matrix(matrix)
    return Dataset(matrix, None, data.sample_ids.astype(str).tolist(), list(data.x.columns)), frame_fingerprint(data)
