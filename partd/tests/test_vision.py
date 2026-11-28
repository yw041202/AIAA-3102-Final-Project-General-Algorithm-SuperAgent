"""Frozen-image integration checks using local official weights, never downloads."""
import csv
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from PIL import Image

from partd.c_executor import execute_via_c
from partd.common import read_json, write_json
from partd.vision import BACKBONES, extract_features, fit_head, load_image_artifact, read_feature_cache


@pytest.mark.parametrize("backbone,grayscale", [("resnet18", False), ("efficientnet_b0", True)])
def test_frozen_backbone_portable_heads_and_cache_guards(tmp_path_factory, monkeypatch, backbone, grayscale):
    weight_root = os.environ.get("VISION_WEIGHTS_ROOT")
    if not weight_root:
        pytest.skip("set VISION_WEIGHTS_ROOT to existing official pretrained files")
    root = tmp_path_factory.mktemp("v")
    source = root / "raw"
    source.mkdir()
    rng = np.random.default_rng(9)
    for split, per_class in (("train", 8), ("val", 3)):
        with (source / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "image_path", "label"])
            for label in range(3):
                for index in range(per_class):
                    array = rng.integers(0, 35, size=(32, 32) if grayscale else (32, 32, 3), dtype=np.uint8)
                    if grayscale: array[:, 5 + label * 6:10 + label * 6] = 200
                    else: array[:, :, label] = 200
                    filename = f"{split}-{label}-{index}.png"
                    Image.fromarray(array).save(source / filename)
                    writer.writerow([f"{split}:{label}:{index}", filename, f"class_{label}"])
    config = {"modality": "image", "format": "image_csv", "feature_columns": ["image_path"],
        "id_column": "sample_id", "label_column": "label", "metric": "accuracy",
        "splits": {"train": {"path": "train.csv"}, "val": {"path": "val.csv"}, "test": {"path": "must-not-read.csv"}}}
    write_json(source / "config.json", config)
    request = {"schema_version": 1, "task_id": "synthetic_image", "run_id": "test", "experiment_id": "features",
        "config_path": str(source / "config.json"), "data_root": str(source), "partb_root": os.environ["PARTB_ROOT"],
        "encoder": {"name": backbone, "weight_path": str(Path(weight_root) / BACKBONES[backbone]["filename"])},
        "threads": 2, "batch_size": 4, "output_dir": str(root / "features")}
    write_json(root / "extract.json", request)
    from partb import DataModule
    original = DataModule._read
    def guard(self, split, with_labels):
        if split == "test": raise AssertionError("test access during development")
        return original(self, split, with_labels)
    monkeypatch.setattr(DataModule, "_read", guard)
    result = extract_features(root / "extract.json")
    metadata = result["metadata"]
    assert metadata["encoder"]["trainable_parameters"] == 0
    assert metadata["backbone_state_before"] == metadata["backbone_state_after"]
    assert metadata["datasets"]["train"]["shape"] == [24, BACKBONES[backbone]["dimension"]]
    prediction_root = root / "prediction_only"
    prediction_root.mkdir()
    with (source / "val.csv").open(encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    with (prediction_root / "val.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    for r in records: shutil.copyfile(source / r["image_path"], prediction_root / r["image_path"])
    for name in ("logistic_regression", "linear_svm"):
        training = {"schema_version": 1, "task_id": "synthetic_image", "run_id": "test", "experiment_id": name,
            "feature_directory": str(root / "features"), "head": {"name": name, "params": {"C": .1}},
            "feature_scaling": "standard", "output_dir": str(root / name)}
        write_json(root / f"{name}.json", training)
        trained = fit_head(root / f"{name}.json")
        artifact = Path(trained["artifact"]["directory"])
        _, pipeline, manifest = load_image_artifact(artifact, os.environ["PARTB_ROOT"])
        X = np.load(root / "features/train_features.npy")
        np.testing.assert_allclose(pipeline.named_steps["feature_scaler"].mean_, X.astype(float).mean(axis=0))
        portable = root / (name + "_portable")
        shutil.copytree(artifact, portable)
        out = root / f"{name}_pred.csv"
        restored = execute_via_c(["vision-predict", "--artifact", str(portable), "--data-root", str(prediction_root),
            "--split", "val", "--partb-root", os.environ["PARTB_ROOT"], "--batch-size", "4", "--output", str(out)],
            partc_root=os.environ["PARTC_ROOT"], workdir=root / (name + "_c"), task_id="synthetic_image", run_id="test",
            timeout_seconds=120, python=sys.executable)
        assert restored["status"] == "success", restored
        with Path(trained["validation_predictions_path"]).open(encoding="utf-8") as f:
            before = list(csv.DictReader(f))
        with out.open(encoding="utf-8") as f:
            after = list(csv.DictReader(f))
        assert [(r["sample_id"],r["prediction"]) for r in before] == [(r["sample_id"],r["prediction"]) for r in after]
        assert restored["partd_result"]["labels_loaded"] is False
        assert restored["partd_result"]["fit_called"] is False
        with (portable / "head.joblib").open("ab") as f: f.write(b"changed")
        with pytest.raises(ValueError, match="checksum"):
            load_image_artifact(portable, os.environ["PARTB_ROOT"])
    with (root / "features/val_features.npy").open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_feature_cache(root / "features")
