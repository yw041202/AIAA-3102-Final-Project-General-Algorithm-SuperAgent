import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from partd import predict, train
from partd.artifacts import load_artifact
from partd.c_executor import execute_via_c
from partd.common import read_json, write_json
from partd.demo_data import create_partb_inputs
from partd.partb_adapter import import_partb, load_preprocessor


@pytest.fixture(params=["tabular", "text"])
def b_demo(request, tmp_path_factory):
    root = os.environ.get("PARTB_ROOT")
    if not root:
        pytest.skip("set PARTB_ROOT for B integration tests")
    return create_partb_inputs(tmp_path_factory.mktemp("b") / "d", root, request.param)


def test_formal_b_train_cli_and_sparse_artifact(b_demo):
    runner = Path(__file__).resolve().parents[1] / "run_partd.py"
    child = subprocess.run([sys.executable, str(runner), "train", "--request", str(b_demo)],
                           capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout)
    assert result["input_source"] == "partb"
    assert result["data"]["sparse"] is True
    pipeline, manifest = load_artifact(result["artifact"]["directory"])
    assert list(pipeline.named_steps) == ["classifier"]  # no second D scaler/imputer
    assert manifest["schema_version"] == 2
    assert manifest["classes"] == [0, 1]  # B's numeric labels keep their meaning/type
    module, processor = load_preprocessor(result["artifact"]["directory"], manifest, os.environ["PARTB_ROOT"])
    assert processor.fit_record["source_split"] == "train"
    assert processor.fit_record["rows"] == 120
    if manifest["partb"]["modality"] == "text":
        assert "validationonly" not in processor.transformer.vocabulary_
    assert set(manifest["files"]) == {"preprocessor.joblib", "preprocessor.joblib.json", "task_config.json"}


def test_relocated_b_artifact_predicts_raw_input_without_training_or_refit(b_demo):
    result = train(b_demo)
    assert result["status"] == "success", result
    folder = b_demo.parent
    portable = folder / "p"
    shutil.copytree(result["artifact"]["directory"], portable)
    (folder / "train.csv").rename(folder / "train.unavailable")
    runner = Path(__file__).resolve().parents[1] / "run_partd.py"
    child = subprocess.run([sys.executable, str(runner), "predict", "--artifact", str(portable), "--data-root", str(folder),
                           "--split", "val", "--partb-root", os.environ["PARTB_ROOT"], "--output", str(folder / "restored.csv"),
                           "--include-probabilities"], capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout)["n_predictions"] == 40
    with Path(result["validation_predictions_path"]).open(encoding="utf-8") as handle:
        before = list(csv.DictReader(handle))
    with (folder / "restored.csv").open(encoding="utf-8") as handle:
        after = list(csv.DictReader(handle))
    assert [r["sample_id"] for r in before] == [r["sample_id"] for r in after]
    assert [r["prediction"] for r in before] == [r["prediction"] for r in after]
    np.testing.assert_allclose([float(r["probability_1"]) for r in before], [float(r["probability_1"]) for r in after], rtol=0, atol=1e-12)
    (folder / "val.csv").rename(folder / "val.unavailable")
    output = predict(portable, None, folder / "new.csv", data_root=folder, split="prediction",
                     partb_root=os.environ["PARTB_ROOT"], include_probabilities=True)
    assert output["n_predictions"] == 20
    assert output["probability_columns"] == {"probability_0": 0, "probability_1": 1}


def test_partb_state_tampering_is_detected(b_demo):
    result = train(b_demo)
    with Path(result["artifact"]["preprocessor_path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        predict(result["artifact"]["directory"], None, b_demo.parent / "bad.csv", data_root=b_demo.parent,
                split="prediction", partb_root=os.environ["PARTB_ROOT"])


def test_b_errors_have_context_and_phase(b_demo):
    request = read_json(b_demo)
    request["data"]["preprocess_options"]["wrong_option"] = True
    write_json(b_demo, request)
    result = train(b_demo)
    assert result["status"] == "error"
    assert result["stage"] == "data_preparation"
    assert result["experiment_id"] == request["experiment_id"]
    assert read_json(result["result_path"])["error"]["type"] == "ValueError"


def test_training_does_not_open_test_and_prediction_does_not_fit(b_demo, monkeypatch):
    module = import_partb(os.environ["PARTB_ROOT"])
    original_read = module.DataModule._read
    def only_development(self, name, with_labels):
        assert name in {"train", "val"}, "training attempted to access another split"
        return original_read(self, name, with_labels)
    monkeypatch.setattr(module.DataModule, "_read", only_development)
    result = train(b_demo)
    assert result["status"] == "success", result
    monkeypatch.setattr(module.DataModule, "_read", original_read)
    def no_fit(*args, **kwargs):
        raise AssertionError("prediction tried to refit preprocessing")
    monkeypatch.setattr(module.Preprocessor, "fit", no_fit)
    output = predict(result["artifact"]["directory"], None, b_demo.parent / "p.csv", data_root=b_demo.parent,
                     split="prediction", partb_root=os.environ["PARTB_ROOT"])
    assert output["status"] == "success"


def test_c_launches_real_b_d_training_and_prediction(b_demo):
    c_root = os.environ.get("PARTC_ROOT")
    if not c_root:
        pytest.skip("set PARTC_ROOT for B/D/C integration")
    request = read_json(b_demo)
    result = execute_via_c(["train", "--request", str(b_demo)], partc_root=c_root, workdir=b_demo.parent / "ct",
                           task_id=request["task_id"], run_id=request["run_id"])
    assert result["status"] == "success", result
    artifact = result["partd_result"]["artifact"]["directory"]
    output = execute_via_c(["predict", "--artifact", artifact, "--data-root", str(b_demo.parent), "--split", "prediction",
                           "--partb-root", os.environ["PARTB_ROOT"], "--output", str(b_demo.parent / "cp.csv")],
                          partc_root=c_root, workdir=b_demo.parent / "cp", task_id=request["task_id"], run_id=request["run_id"])
    assert output["status"] == "success", output
    assert output["partd_result"]["n_predictions"] == 20


def test_old_numeric_v01_artifact_still_loads(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    old = root / "evidence" / "demo-run"
    if not old.exists():
        pytest.skip("legacy evidence fixture not present")
    output = predict(old / "artifacts/synthetic_demo/development/logistic_direct", old / "prediction.csv",
                     tmp_path_factory.mktemp("old") / "p.csv")
    assert output["n_predictions"] == 30


def test_b_npy_input_preserves_labels_and_predicts_without_label_file(tmp_path_factory):
    b_root = os.environ.get("PARTB_ROOT")
    if not b_root:
        pytest.skip("set PARTB_ROOT for NPY integration")
    request_path = create_partb_inputs(tmp_path_factory.mktemp("npy") / "d", b_root)
    folder = request_path.parent
    config = read_json(folder / "config.json")
    config.update(format="npy", trusted_pickle=True)
    for split in ("train", "val", "prediction"):
        with (folder / f"{split}.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        np.save(folder / f"{split}.npy", np.asarray([[r["category"], r["amount"]] for r in rows], dtype=object))
        config["splits"][split] = {"features": f"{split}.npy", "labels": f"y_{split}.npy"}
        if split != "prediction":
            np.save(folder / f"y_{split}.npy", np.asarray([int(r["label"]) for r in rows]))
    config["splits"]["test"] = {"features": "absent_test.npy", "labels": "absent_test_labels.npy"}
    write_json(folder / "config.json", config)
    result = train(request_path)
    assert result["status"] == "success", result
    assert load_artifact(result["artifact"]["directory"])[1]["classes"] == [0, 1]
    output = predict(result["artifact"]["directory"], None, folder / "npy-p.csv", data_root=folder,
                     split="prediction", partb_root=b_root)
    assert output["n_predictions"] == 20
    assert not (folder / "y_prediction.npy").exists()


def test_b_text_concat_options_are_forwarded(tmp_path_factory):
    b_root = os.environ.get("PARTB_ROOT")
    if not b_root:
        pytest.skip("set PARTB_ROOT for text integration")
    request_path = create_partb_inputs(tmp_path_factory.mktemp("concat") / "d", b_root, "text")
    request = read_json(request_path)
    request["data"]["preprocess_options"] = {"combine": "concat", "max_features": 20}
    write_json(request_path, request)
    result = train(request_path)
    assert result["status"] == "success", result
    _, manifest = load_artifact(result["artifact"]["directory"])
    _, processor = load_preprocessor(result["artifact"]["directory"], manifest, b_root)
    assert processor.combine == "concat"
    assert result["data"]["feature_shape"][1] == 2 * len(processor.transformer.vocabulary_)
