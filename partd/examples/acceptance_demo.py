"""One command: independent-process train/load/predict, then real C+D execution."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from partd.c_executor import execute_via_c
from partd.common import read_json, versions, write_json
from partd.demo_data import create_demo_inputs


def run_cli(arguments):
    result = subprocess.run([sys.executable, str(ROOT / "run_partd.py"), *arguments],
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError(f"D subprocess failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New directory; existing results are not overwritten")
    parser.add_argument("--partc-root", required=True, help="Part C directory containing src/")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    request_path = create_demo_inputs(output)

    training = run_cli(["train", "--request", str(request_path)])
    artifact = training["artifact"]["directory"]
    prediction = run_cli(["predict", "--artifact", artifact, "--input", str(output / "prediction.csv"),
                          "--output", str(output / "predictions_direct.csv"), "--include-probabilities"])
    validation = run_cli(["predict", "--artifact", artifact, "--input", str(output / "validation.csv"),
                          "--output", str(output / "validation_reloaded.csv"), "--include-probabilities"])
    before, after = rows(training["validation_predictions_path"]), rows(validation["output_path"])
    assert len(before) == len(after) == 60
    assert all(a["sample_id"] == b["sample_id"] and a["prediction"] == b["prediction"] for a, b in zip(before, after))
    probability_columns = [name for name in after[0] if name.startswith("probability_")]
    assert all(abs(float(a[name]) - float(b[name])) < 1e-12 for a, b in zip(before, after) for name in probability_columns)
    independent_accuracy = sum(row["true_label"] == row["prediction"] for row in before) / len(before)
    assert independent_accuracy == training["metrics"]["accuracy"]

    c_train = execute_via_c(["train", "--request", str(output / "request_via_c.json")],
                            partc_root=args.partc_root, workdir=output / "c_train",
                            task_id="synthetic_demo", run_id="development")
    assert c_train["status"] == "success", c_train
    c_artifact = c_train["partd_result"]["artifact"]["directory"]
    c_predict = execute_via_c(["predict", "--artifact", c_artifact, "--input", str(output / "prediction.csv"),
                              "--output", str(output / "predictions_via_c.csv"), "--include-probabilities"],
                             partc_root=args.partc_root, workdir=output / "c_predict",
                             task_id="synthetic_demo", run_id="development")
    assert c_predict["status"] == "success", c_predict
    assert rows(prediction["output_path"]) == rows(c_predict["partd_result"]["output_path"])

    bad_request = read_json(request_path)
    bad_request["experiment_id"] = "invalid_method"
    bad_request["model"]["name"] = "unsupported_method"
    write_json(output / "request_invalid.json", bad_request)
    c_failure = execute_via_c(["train", "--request", str(output / "request_invalid.json")],
                              partc_root=args.partc_root, workdir=output / "c_expected_failure",
                              task_id="synthetic_demo", run_id="development")
    assert c_failure["status"] == "error"
    assert c_failure["execution"]["status"] == "runtime_error"
    assert c_failure["execution"]["exit_code"] == 2
    assert c_failure["partd_result"]["error"]["type"] == "ValueError"

    summary = {"status": "success", "synthetic_data_only": True, "live_llm_calls": 0,
               "versions": versions(), "metrics": training["metrics"], "independently_recomputed_accuracy": independent_accuracy,
               "n_predictions": prediction["n_predictions"], "reload_prediction_match": True,
               "reload_probability_match": True, "direct_and_c_prediction_match": True,
               "c_train_status": c_train["execution"]["status"], "c_predict_status": c_predict["execution"]["status"],
               "c_expected_failure_status": c_failure["execution"]["status"],
               "c_expected_failure_exit_code": c_failure["execution"]["exit_code"],
               "c_executor_sha256": c_train["executor_sha256"],
               "c_executor_reference_commit": "877625a2216d2a6bd5410e860ccf8006baea5748",
               "c_executor_matches_reference_sha256": c_train["executor_sha256"] == "0f7409c77e984a4c3e17a2db218d6d72f122eb1a09f61a6576aa17d192603b3e",
               "partd_source_sha256": training["partd_source_sha256"],
               "independent_subprocesses": 6, "artifact_directory": artifact,
               "pending": ["real A/B integration", "official task data/metrics/output format", "C Windows timeout cleanup"]}
    write_json(output / "acceptance_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
