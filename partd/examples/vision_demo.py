"""Offline D/B/C pipeline demonstration using generated images, never teacher data.

The deterministic policy illustrates the contract A can consume. It is not an
official SuperAgent run and does not invoke an LLM or evaluate a test set.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from partd.c_executor import execute_via_c
from partd.common import read_json, write_json
from partd.vision_policy import suggest


def create_images(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(27)
    for split, count in (("train", 6), ("val", 3)):
        with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "image_path", "label"])
            for label in range(3):
                for index in range(count):
                    pixels = rng.integers(0, 40, (28, 28), dtype=np.uint8)
                    pixels[4 + label * 7:8 + label * 7] = 210
                    name = f"{split}-{label}-{index}.png"
                    Image.fromarray(pixels).save(root / name)
                    writer.writerow([f"{split}:{label}:{index}", name, f"shape_{label}"])
    config = {"modality": "image", "format": "image_csv", "feature_columns": ["image_path"],
        "id_column": "sample_id", "label_column": "label", "metric": "accuracy",
        "splits": {"train": {"path": "train.csv"}, "val": {"path": "val.csv"},
                   "test": {"path": "never-created-test.csv"}}}
    write_json(root / "config.json", config)
    return root / "config.json"


def run(output, weights_root, partb_root, partc_root, backbone="resnet18"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    config = create_images(raw)
    history = []
    policy = {"schema_version": 1, "primary_metric": "accuracy", "history": history,
              "policy": {"backbones": [backbone], "max_experiments": 2}}
    while True:
        decision = suggest(policy)
        write_json(output / f"decision-{len(history):02}.json", decision)
        if decision["action"] == "stop":
            break
        experiment = f"candidate-{len(history):02}"
        request = {"schema_version": 1, "task_id": "generated_images", "run_id": "demo", "experiment_id": experiment,
            "data": {"source": "partb", "config_path": str(config), "data_root": str(raw), "partb_root": str(partb_root)},
            "model": decision["model"], "preprocessing": "partb", "seed": 42,
            "metrics": {"primary": "accuracy", "additional": ["f1_macro"]},
            "vision": {"weights_root": str(weights_root), "cache_dir": str(output / "cache"), "threads": 2, "batch_size": 4},
            "output_dir": str(output / "artifacts")}
        path = output / f"request-{experiment}.json"
        write_json(path, request)
        executed = execute_via_c(["train", "--request", str(path)], partc_root=partc_root,
            workdir=output / f"execution-{experiment}", task_id="generated_images", run_id="demo",
            timeout_seconds=120, python=sys.executable)
        if executed["status"] != "success":
            raise RuntimeError(executed)
        history.append(executed["partd_result"])
    # Demonstration caller locks its validation-only choice before prediction.
    best = max(history, key=lambda row: row["primary_metric"]["value"])
    write_json(output / "selection.json", {"experiment_id": best["experiment_id"], "metric": best["primary_metric"],
        "official_result": False, "test_set_used": False})
    portable = output / "portable_model"
    shutil.copytree(best["artifact"]["directory"], portable)
    prediction_root = output / "prediction_only"
    prediction_root.mkdir()
    with (raw / "val.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (prediction_root / "val.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["sample_id", "image_path"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        shutil.copyfile(raw / row["image_path"], prediction_root / row["image_path"])
    predicted = execute_via_c(["predict", "--artifact", str(portable), "--data-root", str(prediction_root),
        "--partb-root", str(partb_root), "--split", "val", "--output", str(output / "predictions.csv")],
        partc_root=partc_root, workdir=output / "prediction_execution", task_id="generated_images", run_id="demo",
        timeout_seconds=120, python=sys.executable)
    if predicted["status"] != "success":
        raise RuntimeError(predicted)
    with Path(best["validation_predictions_path"]).open(encoding="utf-8") as handle:
        expected = [(r["sample_id"], r["prediction"]) for r in csv.DictReader(handle)]
    with (output / "predictions.csv").open(encoding="utf-8") as handle:
        actual = [(r["sample_id"], r["prediction"]) for r in csv.DictReader(handle)]
    if expected != actual:
        raise AssertionError("new-process predictions differ")
    summary = {"status": "success", "official_result": False, "test_set_used": False,
        "candidate_count": len(history), "cache_hits": [r["cache"]["hit"] for r in history],
        "predictions_identical": True, "prediction_labels_loaded": predicted["partd_result"]["labels_loaded"],
        "prediction_fit_called": predicted["partd_result"]["fit_called"],
        "source_sha256": best["partd_source_sha256"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--partb-root", required=True)
    parser.add_argument("--partc-root", required=True)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    args = parser.parse_args()
    import json
    print(json.dumps(run(Path(args.output).resolve(), Path(args.weights_root).resolve(),
        Path(args.partb_root).resolve(), Path(args.partc_root).resolve(), args.backbone), ensure_ascii=True))
