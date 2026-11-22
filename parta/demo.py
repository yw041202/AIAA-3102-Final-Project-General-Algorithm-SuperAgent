"""Synthetic inputs and a deterministic planner for offline integration only."""
import csv
import json
from pathlib import Path

from .agent import write_json


class OfflinePlanner:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        history = context["history"]
        if not history:
            model = {"name": "logistic_regression", "params": {"C": 1.0}}
            reason = "Establish a linear baseline for the observed development data"
        elif len(history) == 1:
            model = {"name": "random_forest", "params": {"n_estimators": 20, "max_depth": 4}}
            reason = f"Compare a nonlinear candidate after baseline feedback: {history[-1]['status']}"
        else:
            return {"request_status": "success", "output_text": json.dumps({"action": "stop", "reason": "Offline fixture complete"})}
        return {"request_status": "success", "output_text": json.dumps({"action": "experiment", "reason": reason,
                "model": model, "preprocess_options": {}})}

    def summary(self):
        return {"offline": True, "live_calls": 0, "spent_rmb": "0"}


def make_fixture(directory, modality="tabular", metric="accuracy"):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    features = ["color", "size"] if modality == "tabular" else ["text"]
    for split, count in (("train", 40), ("val", 16), ("test", 12)):
        with (directory / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", *features, "label"])
            for i in range(count):
                values = ["red" if i % 2 else "blue", str(i % 3)] if modality == "tabular" else ["positive good" if i % 2 else "negative bad"]
                writer.writerow([f"{split}_{i}", *values, i % 2 if split != "test" else "DO_NOT_READ_TEST_LABEL"])
    write_json(directory / "data.json", {"modality": modality, "format": "csv", "feature_columns": features,
               "label_column": "label", "id_column": "id", "allowed_labels": [0, 1], "metric": metric,
               "splits": {split: {"path": f"{split}.csv"} for split in ("train", "val", "test")}})
    write_json(directory / "task.json", {"task_id": f"synthetic_{modality}", "run_id": "offline_001",
               "data_config": "data.json", "data_root": ".", "output_root": "runs", "max_steps": 4, "patience": 3})
    return directory / "task.json"
