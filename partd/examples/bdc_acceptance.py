"""Formal Part D CLI + original B + original C; synthetic train/val/prediction."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from partd.c_executor import execute_via_c
from partd.common import read_json, source_fingerprint, versions, write_json
from partd.demo_data import create_partb_inputs


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cli(arguments):
    process = subprocess.run([sys.executable, str(ROOT / "run_partd.py"), *arguments],
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--partb-root", required=True)
    parser.add_argument("--partc-root", required=True)
    args = parser.parse_args()
    b_root, c_root = str(Path(args.partb_root).resolve()), str(Path(args.partc_root).resolve())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for modality in ("tabular", "text"):
        folder = output / modality
        request_path = create_partb_inputs(folder, b_root, modality)
        request = read_json(request_path)
        def c_run(arguments, label):
            return execute_via_c(arguments, partc_root=c_root, workdir=folder / label,
                                 task_id=request["task_id"], run_id=request["run_id"])
        direct = cli(["train", "--request", str(request_path)])
        c_train = c_run(["train", "--request", str(folder / "request_via_c.json")], "c_train")
        assert c_train["status"] == "success", c_train
        assert direct["metrics"] == c_train["partd_result"]["metrics"]
        artifact = direct["artifact"]["directory"]
        c_artifact = c_train["partd_result"]["artifact"]["directory"]
        def prediction_args(selected_artifact, split, name):
            return ["predict", "--artifact", str(selected_artifact), "--data-root", str(folder), "--split", split,
                    "--partb-root", b_root, "--output", str(folder / name), "--include-probabilities"]
        reloaded = cli(prediction_args(artifact, "val", "validation_reloaded.csv"))
        c_validation = c_run(prediction_args(c_artifact, "val", "validation_c.csv"), "c_validation")
        assert c_validation["status"] == "success", c_validation
        before, after = rows(direct["validation_predictions_path"]), rows(reloaded["output_path"])
        c_rows = rows(c_validation["partd_result"]["output_path"])
        assert after == c_rows
        assert len(before) == len(after) == 40
        for a, b in zip(before, after):
            assert a["sample_id"] == b["sample_id"] and a["prediction"] == b["prediction"]
            for field in ("probability_0", "probability_1"):
                assert abs(float(a[field]) - float(b[field])) < 1e-12
        accuracy = sum(row["true_label"] == row["prediction"] for row in before) / 40
        assert accuracy == direct["metrics"]["accuracy"]
        bad = read_json(request_path)
        bad["experiment_id"] = "expected_error"
        bad["data"]["preprocess_options"]["wrong_option"] = True
        write_json(folder / "bad_request.json", bad)
        failure = c_run(["train", "--request", str(folder / "bad_request.json")], "c_expected_error")
        assert failure["execution"]["status"] == "runtime_error" and failure["execution"]["exit_code"] == 2
        assert failure["partd_result"]["stage"] == "data_preparation"
        portable = folder / "portable"
        shutil.copytree(artifact, portable)
        (folder / "train.csv").rename(folder / "train.unavailable")
        (folder / "val.csv").rename(folder / "val.unavailable")
        prediction = cli(prediction_args(portable, "prediction", "predictions_direct.csv"))
        c_prediction = c_run(prediction_args(portable, "prediction", "predictions_c.csv"), "c_prediction")
        assert c_prediction["status"] == "success", c_prediction
        assert rows(prediction["output_path"]) == rows(c_prediction["partd_result"]["output_path"])
        assert prediction["n_predictions"] == 20
        assert not (folder / "must_not_read_test.csv").exists()
        manifest = read_json(portable / "artifact.json")
        records.append({"modality": modality, "status": "success", "metrics": direct["metrics"],
                        "n_train": 120, "n_validation": 40, "n_predictions": 20,
                        "sparse_features": direct["data"]["sparse"], "raw_input_prediction": True,
                        "reload_labels_and_probabilities_match": True, "direct_and_c_match": True,
                        "prediction_without_training_or_validation_files": True,
                        "c_train_status": c_train["execution"]["status"], "c_predict_status": c_prediction["execution"]["status"],
                        "c_expected_error_status": failure["execution"]["status"], "c_expected_error_exit_code": 2,
                        "b_source_sha256": manifest["partb"]["source_sha256"],
                        "c_executor_sha256": c_train["executor_sha256"], "artifact_directory": str(portable)})
    summary = {"status": "success", "partd_version": "0.2.0", "partd_source_sha256": source_fingerprint(),
               "versions": versions(include_partb=True), "synthetic_data_only": True, "live_llm_calls": 0,
               "course_test_files_accessed": False, "independent_subprocesses": 14, "records": records,
               "pending": ["A main-loop integration on 2026-11-19", "B/A cross-review", "official task runs", "C Windows timeout cleanup"]}
    write_json(output / "acceptance_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
