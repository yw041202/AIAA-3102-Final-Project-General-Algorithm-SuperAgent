"""Persist the fitted pipeline, including preprocessing; verify before loading."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import joblib

from .common import read_json, sha256, versions, write_json


def save_artifact(directory, pipeline, metadata, *, preprocessor=None, task_config=None):
    directory = Path(directory)
    model_path = directory / "model.joblib"
    temporary = directory / "model.joblib.tmp"
    joblib.dump({"schema_version": 1, "pipeline": pipeline, "metadata": metadata}, temporary, compress=3)
    temporary.replace(model_path)
    manifest = {"schema_version": 2 if preprocessor is not None else 1, **metadata,
                "versions": versions(include_partb=preprocessor is not None), "model_file": "model.joblib",
                "model_sha256": sha256(model_path)}
    if preprocessor is not None:
        preprocessor.save(directory / "preprocessor.joblib")
        write_json(directory / "task_config.json", task_config)
        manifest["files"] = {name: sha256(directory / name) for name in
                             ("preprocessor.joblib", "preprocessor.joblib.json", "task_config.json")}
    if metadata["model"]["name"] == "catboost":
        manifest["optional_dependencies"] = {"catboost": importlib.metadata.version("catboost")}
    write_json(directory / "artifact.json", manifest)
    result = {"directory": str(directory.resolve()), "model_path": str(model_path.resolve()),
              "manifest_path": str((directory / "artifact.json").resolve()), "model_sha256": manifest["model_sha256"]}
    if preprocessor is not None:
        result["preprocessor_path"] = str((directory / "preprocessor.joblib").resolve())
        result["task_config_path"] = str((directory / "task_config.json").resolve())
    return result


def load_artifact(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "artifact.json")
    if manifest.get("schema_version") not in {1, 2} or manifest.get("model_file") != "model.joblib":
        raise ValueError("unsupported or invalid artifact manifest")
    current = versions(include_partb=manifest.get("input_source") == "partb")
    if manifest["versions"] != current:
        raise ValueError(f"artifact environment differs: expected {manifest['versions']}, current {current}")
    for name, expected in manifest.get("optional_dependencies", {}).items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"artifact dependency differs: {name}")
    if manifest.get("input_source") == "partb":
        expected_files = {"preprocessor.joblib", "preprocessor.joblib.json", "task_config.json"}
        if manifest.get("schema_version") != 2 or set(manifest.get("files", {})) != expected_files:
            raise ValueError("incomplete Part B artifact manifest")
        for name in expected_files:
            if sha256(directory / name) != manifest["files"][name]:
                raise ValueError(f"artifact checksum mismatch: {name}")
    model_path = directory / "model.joblib"
    if sha256(model_path) != manifest["model_sha256"]:
        raise ValueError("model checksum mismatch")
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("unsupported model bundle")
    keys = ["features", "classes", "id_column", "target_column", "task_id", "run_id", "experiment_id"]
    if "input_source" in bundle["metadata"]:
        keys.append("input_source")
    for name in keys:
        if bundle["metadata"][name] != manifest[name]:
            raise ValueError(f"artifact metadata mismatch: {name}")
    return bundle["pipeline"], manifest
