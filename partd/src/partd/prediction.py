from __future__ import annotations

from pathlib import Path

from .artifacts import load_artifact
from .common import sha256, write_json
from .data import load_csv, write_predictions


def predict(artifact_dir, input_csv, output_csv, *, include_probabilities=False,
            data_root=None, config_path=None, split="test", partb_root=None):
    """Load a fitted pipeline and predict. Never fit or accept true labels."""
    artifact_dir, output_csv = (Path(p).resolve() for p in (artifact_dir, output_csv))
    input_csv = Path(input_csv).resolve() if input_csv is not None else None
    metadata_path = output_csv.with_suffix(output_csv.suffix + ".metadata.json")
    if output_csv.exists() or metadata_path.exists():
        raise FileExistsError(f"prediction output already exists: {output_csv}")
    if output_csv == input_csv:
        raise ValueError("prediction output cannot overwrite its input")
    pipeline, manifest = load_artifact(artifact_dir)
    source = manifest.get("input_source", "numeric_csv")
    if source == "partb":
        if input_csv is not None or data_root is None:
            raise ValueError("Part B artifacts require --data-root, optional --config/--split, and no --input")
        from .partb_adapter import prepare_prediction
        dataset, input_hash = prepare_prediction(artifact_dir, manifest, data_root=data_root,
                                                 config_path=config_path, split=split, partb_root=partb_root)
    else:
        if input_csv is None or data_root is not None or config_path is not None or partb_root is not None:
            raise ValueError("numeric CSV artifacts require --input and no Part B prediction options")
        dataset = load_csv(input_csv, target_column=manifest["target_column"], id_column=manifest["id_column"],
                           features=manifest["features"], labeled=False)
        input_hash = sha256(input_csv)
    predicted = pipeline.predict(dataset.X)
    probabilities = pipeline.predict_proba(dataset.X) if include_probabilities else None
    write_predictions(output_csv, dataset.ids, predicted, probabilities)
    result = {"event": "partd.result", "schema_version": 1, "command": "predict", "status": "success",
              "n_predictions": len(dataset.ids), "output_path": str(output_csv),
              "metadata_path": str(metadata_path), "source_experiment_id": manifest["experiment_id"],
              "classes": manifest["classes"], "probability_columns": {f"probability_{i}": label for i, label in enumerate(manifest["classes"])} if include_probabilities else {},
              "input_sha256": input_hash, "input_source": source,
              "input_hash_kind": "features_and_ids" if source == "partb" else "file_bytes",
              "split": split if source == "partb" else None,
              "preprocessor_sha256": manifest.get("files", {}).get("preprocessor.joblib"),
              "partb_source_sha256": manifest.get("partb", {}).get("source_sha256"),
              "model_sha256": manifest["model_sha256"], "versions": manifest["versions"]}
    write_json(metadata_path, result)
    return result
