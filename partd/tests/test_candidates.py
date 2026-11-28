import csv
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from partd import predict, train
from partd.artifacts import load_artifact
from partd.candidates import build_candidate_pipeline
from partd.c_executor import execute_via_c
from partd.common import read_json, write_json
from partd.demo_data import create_partb_inputs
from partd.metrics import calculate_metrics


@pytest.mark.parametrize("name,params", [
    ("linear_svm", {}), ("rbf_svm", {}), ("complement_nb", {"alpha": 0.5}),
    ("mlp", {"hidden_layer_sizes": [16], "max_iter": 30}),
    ("gradient_boosting", {"n_estimators": 15}),
])
def test_sparse_b_candidate_save_load_and_c_new_process(tmp_path, name, params):
    b_root = os.environ["PARTB_ROOT"]
    request_path = create_partb_inputs(tmp_path / "b", b_root, "text")
    request = read_json(request_path)
    request["model"] = {"name": name, "params": params}
    request["metrics"] = {"primary": "accuracy", "additional": ["f1_macro", "roc_auc"]}
    if name == "complement_nb":
        request["data"]["preprocess_options"]["combine"] = "concat"
    write_json(request_path, request)
    response = execute_via_c(["train", "--request", str(request_path)],
        partc_root=os.environ["PARTC_ROOT"], workdir=tmp_path / "ct",
        task_id=request["task_id"], run_id=request["run_id"])
    result = response["partd_result"]
    assert response["status"] == "success", response
    artifact = result["artifact"]["directory"]
    pipeline, manifest = load_artifact(artifact)
    assert manifest["data"]["sparse"]
    output = tmp_path / "loaded.csv"
    restored = execute_via_c(["predict", "--artifact", artifact, "--data-root", str(request_path.parent),
        "--split", "val", "--partb-root", b_root, "--output", str(output)],
        partc_root=os.environ["PARTC_ROOT"], workdir=tmp_path / "cp",
        task_id=request["task_id"], run_id=request["run_id"])
    assert restored["status"] == "success", restored
    with Path(result["validation_predictions_path"]).open(encoding="utf-8") as f:
        original = list(csv.DictReader(f))
    with output.open(encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert [r["prediction"] for r in original] == [r["prediction"] for r in loaded]
    assert sum(r["true_label"] == r["prediction"] for r in original) / len(original) == result["metrics"]["accuracy"]
    if name in {"linear_svm", "rbf_svm"}:
        with pytest.raises(ValueError, match="no probability"):
            predict(artifact, None, tmp_path / "bad.csv", data_root=request_path.parent,
                    split="val", partb_root=b_root, include_probabilities=True)


def test_nb_rejects_signed_features_and_svm_margin_auc():
    X = sparse.csr_matrix([[1., -1.], [0., 2.], [2., 0.], [1., 1.]])
    y = np.asarray([0, 1, 0, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        build_candidate_pipeline({"name": "complement_nb"}, 42, "passthrough").fit(X, y)
    result = calculate_metrics(y, y, None, [0, 1], ["accuracy", "roc_auc"], decision_scores=np.array([-2, 1, -1, 2]))
    assert result == {"accuracy": 1.0, "roc_auc": 1.0}


def test_calibration_and_model_transforms_fit_train_only(demo):
    request = read_json(demo)
    request["model"] = {"name": "linear_svm", "params": {"calibrate": True}}
    write_json(demo, request)
    result = train(demo)
    assert result["status"] == "success", result
    assert result["training"]["has_probabilities"]
    request.update(experiment_id="svd_mlp", model={"name": "mlp", "params": {
        "svd_components": 3, "feature_scaling": "maxabs", "hidden_layer_sizes": [8], "max_iter": 20}})
    write_json(demo, request)
    result = train(demo)
    assert result["status"] == "success", result
    pipeline, manifest = load_artifact(result["artifact"]["directory"])
    assert pipeline.named_steps["svd"].n_features_in_ == 6
    assert manifest["data"]["n_train"] == 180


def test_uncalibrated_svm_log_loss_fails_before_training(demo):
    request = read_json(demo)
    request["model"] = {"name": "linear_svm"}
    write_json(demo, request)
    result = train(demo)
    assert result["status"] == "error"
    assert result["stage"] == "model_configuration"
    assert "requires probabilities" in result["error"]["message"]
