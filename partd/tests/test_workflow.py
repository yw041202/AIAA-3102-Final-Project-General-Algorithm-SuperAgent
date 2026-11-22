import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from partd import predict, train
from partd.artifacts import load_artifact
from partd.c_executor import execute_via_c
from partd.common import read_json, write_json
from partd.data import load_csv
from partd.metrics import calculate_metrics


def modify_request(demo, update):
    request = read_json(demo)
    update(request)
    write_json(demo, request)


def test_independent_process_predict_matches_training_and_manual_metric(demo):
    result = train(demo)
    assert result["status"] == "success", result
    runner = Path(__file__).resolve().parents[1] / "run_partd.py"
    output = demo.parent / "new_process.csv"
    process = subprocess.run([sys.executable, str(runner), "predict", "--artifact", result["artifact"]["directory"],
                              "--input", str(demo.parent / "validation.csv"), "--output", str(output), "--include-probabilities"],
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout)["n_predictions"] == 60
    with open(result["validation_predictions_path"], encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    with output.open(encoding="utf-8") as handle:
        restored = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in original] == [row["sample_id"] for row in restored]
    assert [row["prediction"] for row in original] == [row["prediction"] for row in restored]
    assert [row["probability_1"] for row in original] == [row["probability_1"] for row in restored]
    assert sum(row["prediction"] == row["true_label"] for row in original) / 60 == result["metrics"]["accuracy"]


def test_preprocessing_fits_training_only(demo):
    result = train(demo)
    pipeline, manifest = load_artifact(result["artifact"]["directory"])
    training = load_csv(demo.parent / "train.csv")
    # Independently calculate the training medians and imputed means.
    medians = np.nanmedian(training.X, axis=0)
    np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, medians)
    imputed = np.where(np.isnan(training.X), medians, training.X)
    np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, imputed.mean(axis=0))
    assert manifest["data"]["n_train"] == 180


def test_seed_reproducibility_and_configurable_model(demo):
    modify_request(demo, lambda r: r.update(model={"name": "random_forest", "params": {"n_estimators": 12, "max_depth": 3}}))
    first = train(demo)
    modify_request(demo, lambda r: r.update(experiment_id="second_seeded_run"))
    second = train(demo)
    assert first["status"] == second["status"] == "success"
    assert first["metrics"] == second["metrics"]
    assert Path(first["validation_predictions_path"]).read_bytes() == Path(second["validation_predictions_path"]).read_bytes()


def test_metric_values_and_direction(demo):
    y = np.asarray(["a", "a", "b", "b"])
    predictions = np.asarray(["a", "b", "b", "b"])
    probabilities = np.asarray([[0.9, 0.1], [0.3, 0.7], [0.1, 0.9], [0.2, 0.8]])
    values = calculate_metrics(y, predictions, probabilities, ["a", "b"], ["accuracy", "f1_macro"])
    assert values["accuracy"] == 0.75
    assert values["f1_macro"] == pytest.approx(((2 / 3) + (4 / 5)) / 2)
    modify_request(demo, lambda r: r.update(metrics={"primary": "log_loss", "additional": []}))
    result = train(demo)
    assert result["primary_metric"]["direction"] == "minimize"


def test_multiclass_probability_metric(demo):
    for split in ("train", "validation"):
        path = demo.parent / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for index, row in enumerate(rows):
            row["label"] = ["red", "green", "blue"][index % 3]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    result = train(demo)
    assert result["status"] == "success", result
    assert len(load_artifact(result["artifact"]["directory"])[1]["classes"]) == 3
    assert 0 <= result["metrics"]["roc_auc"] <= 1


@pytest.mark.parametrize("field,value", [("model", {"name": "unknown"}), ("metrics", {"primary": "unknown"}), ("seed", -1)])
def test_bad_config_has_persisted_error(demo, field, value):
    modify_request(demo, lambda r: r.update({field: value}))
    result = train(demo)
    assert result["status"] == "error"
    assert read_json(result["result_path"])["error"] == result["error"]


def test_no_designated_test_data_accepted(demo):
    modify_request(demo, lambda r: r["data"].update(test_csv="prediction.csv"))
    result = train(demo)
    assert result["status"] == "error"
    assert "test data" in result["error"]["message"]


def test_overlap_rejected(demo):
    path = demo.parent / "validation.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("validation_0000", "train_0000"), encoding="utf-8")
    result = train(demo)
    assert result["status"] == "error"
    assert "overlap" in result["error"]["message"]


def test_artifact_tampering_rejected_before_load(demo):
    result = train(demo)
    with Path(result["artifact"]["model_path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(result["artifact"]["directory"])


def test_no_overwriting_prior_experiment(demo):
    first = train(demo)
    content = Path(first["result_path"]).read_bytes()
    second = train(demo)
    assert second["status"] == "error"
    assert second["error"]["type"] == "FileExistsError"
    assert Path(first["result_path"]).read_bytes() == content


def test_portable_artifact_predicts_without_training_data(demo):
    result = train(demo)
    portable = demo.parent / "portable"
    shutil.copytree(result["artifact"]["directory"], portable)
    (demo.parent / "train.csv").rename(demo.parent / "train.unavailable")
    (demo.parent / "validation.csv").rename(demo.parent / "validation.unavailable")
    predicted = predict(portable, demo.parent / "prediction.csv", demo.parent / "portable_predictions.csv", include_probabilities=True)
    assert predicted["n_predictions"] == 30
    assert predicted["probability_columns"] == {"probability_0": "negative", "probability_1": "positive"}


def test_prediction_uses_feature_names_instead_of_csv_column_order(demo):
    result = train(demo)
    first = predict(result["artifact"]["directory"], demo.parent / "prediction.csv", demo.parent / "ordered.csv")
    path = demo.parent / "prediction.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reversed(reader.fieldnames)), list(reader)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    second = predict(result["artifact"]["directory"], path, demo.parent / "reordered.csv")
    assert Path(first["output_path"]).read_bytes() == Path(second["output_path"]).read_bytes()


def test_missing_prediction_feature_and_duplicate_ids_rejected(demo):
    result = train(demo)
    path = demo.parent / "prediction.csv"
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace("feature_0", "wrong_feature"), encoding="utf-8")
    with pytest.raises(ValueError, match="missing feature"):
        predict(result["artifact"]["directory"], path, demo.parent / "invalid.csv")
    path.write_text(content.replace("prediction_0001", "prediction_0000"), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        predict(result["artifact"]["directory"], path, demo.parent / "invalid.csv")


def test_passthrough_ready_features(demo):
    for split in ("train", "validation"):
        path = demo.parent / f"{split}.csv"
        text = path.read_text(encoding="utf-8")
        # Demo blanks are missing numeric cells; use B-like already-filled data.
        while ",," in text:
            text = text.replace(",,", ",0,")
        path.write_text(text, encoding="utf-8")
    modify_request(demo, lambda r: r.update(preprocessing="passthrough"))
    result = train(demo)
    assert result["status"] == "success", result
    pipeline, _ = load_artifact(result["artifact"]["directory"])
    assert "imputer" not in pipeline.named_steps


def test_c_executor_training_and_error_feedback(demo):
    root = os.environ.get("PARTC_ROOT")
    if not root:
        pytest.skip("set PARTC_ROOT to run real C+D integration")
    response = execute_via_c(["train", "--request", str(demo)], partc_root=root,
                             workdir=demo.parent / "c_success", task_id="synthetic_demo", run_id="development")
    assert response["status"] == "success", response
    assert response["partd_result"]["metrics"]["accuracy"] >= 0.5
    assert Path(response["execution_log_path"]).is_file()
    modify_request(demo, lambda r: r.update(experiment_id="bad", model={"name": "unknown"}))
    failed = execute_via_c(["train", "--request", str(demo)], partc_root=root,
                           workdir=demo.parent / "c_failure", task_id="synthetic_demo", run_id="development")
    assert failed["execution"]["status"] == "runtime_error"
    assert failed["execution"]["exit_code"] == 2
    assert failed["partd_result"]["status"] == "error"


def test_c_and_d_identifiers_cannot_silently_diverge(demo):
    with pytest.raises(ValueError, match="must match"):
        execute_via_c(["train", "--request", str(demo)], workdir=demo.parent / "wrong_ids",
                       task_id="different_task", run_id="development")


@pytest.mark.parametrize("field,value", [("model", {"name": "logistic_regression", "C": 99}),
                                         ("metrics", {"primary": "accuracy", "additionals": ["f1_macro"]})])
def test_misplaced_parameters_are_not_silently_ignored(demo, field, value):
    modify_request(demo, lambda r: r.update({field: value}))
    result = train(demo)
    assert result["status"] == "error"
    assert result["stage"] == "model_configuration"
