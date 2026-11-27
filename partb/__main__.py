"""Standalone B acceptance run; all scores belong to development validation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import sparse

from . import DataModule, Preprocessor


def save_matrix(matrix, path):
    if sparse.issparse(matrix):
        sparse.save_npz(path.with_suffix(".npz"), matrix)
    else:
        np.save(path.with_suffix(".npy"), matrix)


def main():
    parser = argparse.ArgumentParser(description="Audit and verify B using train/val only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocess-config", help="JSON file of Agent-selected preprocessing options")
    parser.add_argument("--reload-state", help=argparse.SUPPRESS)
    args = parser.parse_args()
    module = DataModule(args.config, args.data_root)
    train, val = module.load_development()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Image verification uses a small batch; decoding all data is batched below.
    probe = val.x.iloc[:8] if module.modality == "image" else val.x
    if args.reload_state:
        processor = Preprocessor.load(args.reload_state)
        save_matrix(processor.transform(probe), output / "reloaded")
        return
    options = {}
    if args.preprocess_config:
        options = json.loads(Path(args.preprocess_config).read_text(encoding="utf-8"))
    processor = Preprocessor(module.modality, options).fit(train)
    state = output / "preprocessor.joblib"
    processor.save(state)
    expected = processor.transform(probe)
    audit = module.audit()
    if module.modality == "image":
        for split in (train, val):
            for _ in processor.iter_batches(split):
                pass
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, "-m", "partb", "--config", str(module.config_path),
               "--data-root", str(module.root), "--output", str(output), "--reload-state", str(state)]
    subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, check=True)
    if sparse.issparse(expected):
        actual = sparse.load_npz(output / "reloaded.npz")
        equal = expected.shape == actual.shape and (expected != actual).nnz == 0
    else:
        actual = np.load(output / "reloaded.npy")
        equal = np.array_equal(expected, actual)
    if not equal:
        raise AssertionError("Preprocessing changed after reload in a new process")
    result = {"status": "success", "modality": module.modality,
              "train_rows": len(train.x), "val_rows": len(val.x),
              "validation_probe_shape": list(expected.shape), "new_process_reload_equal": equal,
              "test_accessed": False, "state_path": str(state), "audit_path": str(output / "audit.json")}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
