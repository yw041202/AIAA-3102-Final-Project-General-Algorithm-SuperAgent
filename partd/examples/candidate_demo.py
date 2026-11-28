"""Generated native-tabular and joint-text examples through real C workers."""
import argparse
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from partd.c_executor import execute_via_c
from partd.common import write_json, source_fingerprint


def run(output, partb_root, partc_root):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(28)
    results = []
    for modality in ("tabular", "text"):
        raw = root / modality
        raw.mkdir()
        for split, count in (("train", 90), ("val", 18)):
            labels = np.arange(count) % 3
            frame = pd.DataFrame({"sample_id": [f"{split}:{i}" for i in range(count)], "label": [f"class_{i}" for i in labels]})
            if modality == "tabular":
                frame["kind"] = [f"kind_{i}" for i in labels]
                frame["value"] = labels + rng.normal(0, .1, count)
                frame.loc[0, "value"] = np.nan
            else:
                frame["text"] = [f"shared note {'apple cherry pear'.split()[i]} category{i}" for i in labels]
            frame.to_csv(raw / f"{split}.csv", index=False)
        features = ["kind", "value"] if modality == "tabular" else ["text"]
        config = {"modality": modality, "format": "csv", "feature_columns": features, "label_column": "label", "id_column": "sample_id",
            "metric": "accuracy", "splits": {"train": {"path": "train.csv"}, "val": {"path": "val.csv"}, "test": {"path": "unused-test.csv"}}}
        write_json(raw / "config.json", config)
        models = [("catboost", {"iterations": 30}), ("hist_gradient_boosting", {"max_iter": 30})] if modality == "tabular" else [("linear_svm", {"C": .1})]
        options = {"encoding": "native", "numeric_columns": ["value"]} if modality == "tabular" else {
            "analyzer": "word_char", "combine": "concat", "min_df": 1, "max_features": 500}
        prediction_root = raw / "prediction_only"
        prediction_root.mkdir()
        pd.read_csv(raw / "val.csv").drop(columns="label").to_csv(prediction_root / "val.csv", index=False)
        for name, params in models:
            task = "generated_" + modality
            request = {"schema_version": 1, "task_id": task, "run_id": "demo", "experiment_id": name, "seed": 42,
                "data": {"source": "partb", "config_path": str(raw / "config.json"), "data_root": str(raw),
                         "partb_root": str(partb_root), "preprocess_options": options},
                "model": {"name": name, "params": params}, "preprocessing": "partb", "metrics": {"primary": "accuracy", "additional": ["f1_macro"]},
                "output_dir": str(root / "artifacts")}
            path = raw / (name + ".json")
            write_json(path, request)
            response = execute_via_c(["train", "--request", str(path)], partc_root=partc_root, workdir=raw / (name + "_train"),
                task_id=task, run_id="demo", timeout_seconds=120, python=sys.executable)
            if response["status"] != "success": raise RuntimeError(response)
            trained = response["partd_result"]
            copied = root / "portable_models" / name
            shutil.copytree(trained["artifact"]["directory"], copied)
            output = raw / (name + "_predictions.csv")
            prediction = execute_via_c(["predict", "--artifact", str(copied), "--data-root", str(prediction_root), "--split", "val",
                "--partb-root", str(partb_root), "--output", str(output)], partc_root=partc_root, workdir=raw / (name + "_predict"),
                task_id=task, run_id="demo", timeout_seconds=120, python=sys.executable)
            if prediction["status"] != "success": raise RuntimeError(prediction)
            expected = pd.read_csv(trained["validation_predictions_path"], dtype=str)[["sample_id", "prediction"]]
            if not expected.equals(pd.read_csv(output, dtype=str)): raise ValueError("portable predictions differ")
            results.append({"model": name, "status": "success", "metrics": trained["metrics"], "predictions_identical": True,
                            "prediction_inputs_have_labels": False, "prediction_inputs_have_training_files": False})
    summary = {"status": "success", "source_sha256": source_fingerprint(), "official_result": False, "paid_llm_calls": 0, "teacher_test_used": False, "results": results}
    write_json(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partb-root", required=True)
    parser.add_argument("--partc-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, Path(args.partb_root).resolve(), Path(args.partc_root).resolve()), ensure_ascii=True))
