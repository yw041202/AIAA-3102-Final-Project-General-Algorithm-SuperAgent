import csv, os, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from partd import train
from partd.c_executor import execute_via_c
from partd.common import read_json, write_json

@pytest.mark.parametrize("model", ["catboost", "hist_gradient_boosting", "linear_svm"])
def test_advanced_train_only_portable_c_prediction(tmp_path_factory, model):
    if not os.environ.get("PARTB_ROOT") or not os.environ.get("PARTC_ROOT"):
        pytest.skip("set PARTB_ROOT and PARTC_ROOT for integration checks")
    if model == "catboost":
        pytest.importorskip("catboost", reason="install requirements-native-cpu.txt for native boosting checks")
    root = tmp_path_factory.mktemp("a")
    text = model == "linear_svm"
    rng = np.random.default_rng(28)
    frames = {}
    for split, n in (("train", 90), ("val", 18)):
        labels = np.arange(n) % 3
        frame = pd.DataFrame({"sample_id": [f"{split}:{i}" for i in range(n)], "label": [f"c{y}" for y in labels]})
        if text:
            frame["text"] = [f"shared message category{y} {'apple cherry pear'.split()[y]} sample {i}" for i,y in enumerate(labels)]
        else:
            frame["category"] = [f"type{y}" for y in labels]
            frame["numeric"] = labels + rng.normal(0,.1,n)
            frame.loc[0, "numeric"] = np.nan
            if split == "val": frame.loc[0, "category"] = "new-category"
        frame.to_csv(root / f"{split}.csv", index=False)
        frames[split] = frame
    features = ["text"] if text else ["category", "numeric"]
    config = {"modality": "text" if text else "tabular", "format": "csv", "feature_columns": features,
        "label_column": "label", "id_column": "sample_id", "metric": "accuracy",
        "splits": {"train": {"path": "train.csv"}, "val": {"path": "val.csv"}, "test": {"path": "must-not-open.csv"}}}
    write_json(root / "config.json", config)
    params = {"iterations": 30} if model == "catboost" else {"max_iter": 30} if model == "hist_gradient_boosting" else {"C": .1}
    options = {"analyzer": "word_char", "combine": "concat", "min_df": 1, "max_features": 500} if text else {"numeric_columns": ["numeric"], "encoding": "native"}
    request = {"schema_version": 1, "task_id": "generic", "run_id": "check", "experiment_id": model,
        "data": {"source": "partb", "config_path": str(root / "config.json"), "data_root": str(root),
                 "partb_root": os.environ["PARTB_ROOT"], "preprocess_options": options},
        "model": {"name": model, "params": params}, "preprocessing": "partb",
        "metrics": {"primary": "accuracy", "additional": ["f1_macro"]}, "output_dir": str(root / "artifacts")}
    path = root / "request.json"
    write_json(path, request)
    result = train(path)
    assert result["status"] == "success", result
    artifact = Path(result["artifact"]["directory"])
    processor = joblib.load(artifact / "preprocessor.joblib")
    assert processor.fit_record["rows"] == 90
    assert processor.columns == features
    only = root / "prediction"
    only.mkdir()
    frames["val"].drop(columns="label").to_csv(only / "val.csv", index=False)
    response = execute_via_c(["predict", "--artifact", str(artifact), "--data-root", str(only),
        "--split", "val", "--partb-root", os.environ["PARTB_ROOT"], "--output", str(root / "predicted.csv")],
        partc_root=os.environ["PARTC_ROOT"], workdir=root / "execution", task_id="generic", run_id="check", python=sys.executable)
    assert response["status"] == "success", response
    a = pd.read_csv(result["validation_predictions_path"], dtype=str)
    b = pd.read_csv(root / "predicted.csv", dtype=str)
    assert a[["sample_id", "prediction"]].equals(b)
    assert set(b.prediction) <= {"c0", "c1", "c2"}
    if not text:
        request["experiment_id"] += "-bad-encoding"
        request["data"]["preprocess_options"]["encoding"] = "onehot"
        write_json(path, request)
        rejected = train(path)
        assert rejected["status"] == "error" and "encoding=native" in rejected["error"]["message"]
