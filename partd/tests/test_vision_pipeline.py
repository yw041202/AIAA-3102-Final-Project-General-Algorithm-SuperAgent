import csv
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from partd import train
from partd.c_executor import execute_via_c
from partd.common import read_json, write_json


@pytest.fixture
def image_task(tmp_path_factory):
    if not os.environ.get("VISION_WEIGHTS_ROOT"):
        pytest.skip("set VISION_WEIGHTS_ROOT to existing official checkpoints")
    spec = importlib.util.spec_from_file_location("vision_demo", Path(__file__).parents[1] / "examples/vision_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    root = tmp_path_factory.mktemp("p")
    config = demo.create_images(root / "raw")
    request = {"schema_version": 1, "task_id": "unknown_image_task", "run_id": "test", "experiment_id": "first",
        "data": {"source": "partb", "config_path": str(config), "data_root": str(root / "raw"), "partb_root": os.environ["PARTB_ROOT"]},
        "model": {"name": "frozen_vision", "params": {}}, "preprocessing": "partb", "seed": 42,
        "metrics": {"primary": "log_loss", "additional": ["accuracy", "roc_auc"]},
        "vision": {"weights_root": os.environ["VISION_WEIGHTS_ROOT"], "cache_dir": str(root / "cache"), "threads": 2, "batch_size": 4},
        "output_dir": str(root / "artifacts")}
    write_json(root / "request.json", request)
    return root, request


def test_unified_c_pipeline_cache_metric_prediction_and_input_changes(image_task):
    root, request = image_task
    def execute():
        path = root / f"{request['experiment_id']}.json"
        write_json(path, request)
        outer = execute_via_c(["train", "--request", str(path)], partc_root=os.environ["PARTC_ROOT"],
            workdir=root / f"c-{request['experiment_id']}", task_id=request["task_id"], run_id=request["run_id"],
            timeout_seconds=120, python=sys.executable)
        return outer
    first = execute()
    assert first["status"] == "success", first
    first = first["partd_result"]
    assert first["primary_metric"]["direction"] == "minimize" and first["cache"]["hit"] is False
    assert first["test_set_used"] is False
    request["experiment_id"] = "svm"
    request["model"]["params"]["head"] = {"name": "linear_svm", "params": {"C": .1}}
    request["metrics"] = {"primary": "balanced_accuracy", "additional": []}
    second = execute()["partd_result"]
    assert second["status"] == "success" and second["cache"]["hit"] is True
    assert first["cache"]["directory"] == second["cache"]["directory"]
    assert second["primary_metric"]["name"] == "balanced_accuracy"
    preserved = read_json(Path(second["result_path"]))
    duplicate = execute()
    assert duplicate["status"] == "error"
    assert read_json(Path(second["result_path"])) == preserved
    request["experiment_id"] = "invalid_metric"
    request["metrics"] = {"primary": "log_loss", "additional": []}
    failed = execute()["partd_result"]
    assert failed["status"] == "error" and failed["stage"] == "training"
    assert "requires LR" in failed["error"]["message"] and failed["candidate"]
    # Change assignments while preserving class counts: old features must not be reused.
    train_manifest = root / "raw/train.csv"
    with train_manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["label"], rows[-1]["label"] = rows[-1]["label"], rows[0]["label"]
    with train_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["sample_id", "image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    request["experiment_id"] = "changed_labels"
    request["metrics"] = {"primary": "accuracy", "additional": []}
    changed = execute()["partd_result"]
    assert changed["status"] == "success" and changed["cache"]["hit"] is False
    assert changed["cache"]["identity_sha256"] != first["cache"]["identity_sha256"]
    # A changed image, even at the same filename/ID, also invalidates the cache.
    image = root / "raw/val-0-0.png"
    pixels = np.asarray(Image.open(image)).copy()
    pixels[0, 0] ^= 255
    Image.fromarray(pixels).save(image)
    request["experiment_id"] = "changed_pixels"
    changed_again = execute()["partd_result"]
    assert changed_again["status"] == "success" and changed_again["cache"]["hit"] is False
    assert changed_again["cache"]["identity_sha256"] != changed["cache"]["identity_sha256"]


def test_image_overlap_and_unknown_options_fail_without_test_access(image_task):
    root, request = image_task
    request["vision"]["max_per_class"] = 1
    write_json(root / "request.json", request)
    invalid = train(root / "request.json")
    assert invalid["status"] == "error" and "vision accepts" in invalid["error"]["message"]
    del request["vision"]["max_per_class"]
    request["experiment_id"] = "overlap"
    (root / "raw/val-0-0.png").write_bytes((root / "raw/train-0-0.png").read_bytes())
    write_json(root / "request.json", request)
    invalid = train(root / "request.json")
    assert invalid["status"] == "error" and "identical image bytes" in invalid["error"]["message"]


def test_policy_driven_demo_new_process_predicts_without_training_or_labels(tmp_path_factory):
    if not os.environ.get("VISION_WEIGHTS_ROOT"):
        pytest.skip("set VISION_WEIGHTS_ROOT")
    spec = importlib.util.spec_from_file_location("vision_demo_acceptance", Path(__file__).parents[1] / "examples/vision_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.run(tmp_path_factory.mktemp("demo") / "run", os.environ["VISION_WEIGHTS_ROOT"],
        os.environ["PARTB_ROOT"], os.environ["PARTC_ROOT"])
    assert summary["cache_hits"] == [False, True]
    assert summary["predictions_identical"] and summary["candidate_count"] == 2
    assert not summary["prediction_labels_loaded"] and not summary["prediction_fit_called"]
