"""A minimal D-side consumer, for interface acceptance rather than final results.

Run after installing partb, or run from the repository root with:
python -m partb.examples.train_validation_demo ...
"""

import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from partb import DataModule, Preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    module = DataModule(args.config, args.data_root)
    train, val = module.load_development()
    options = {"flatten": True, "size": [8, 8]} if module.modality == "image" else {}
    processor = Preprocessor(module.modality, options).fit(train)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(processor.transform(train.x), train.y)
    predictions = model.predict(processor.transform(val.x))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    processor.save(output / "preprocessor.joblib")
    result = {"status": "success", "split": "val", "metric": "accuracy",
              "accuracy": float(accuracy_score(val.y, predictions)), "rows": len(predictions),
              "purpose": "B/D interface acceptance only", "test_accessed": False}
    (output / "integration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
