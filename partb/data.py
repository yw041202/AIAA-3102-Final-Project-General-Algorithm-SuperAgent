"""Config-driven loading, explicit split management and development-only audits."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    name: str
    x: pd.DataFrame
    y: np.ndarray | None
    sample_ids: np.ndarray


class DataModule:
    def __init__(self, config_path, data_root):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        allowed_fields = {"modality", "format", "feature_columns", "label_column", "id_column",
                          "allowed_labels", "metric", "splits", "validation_split", "trusted_pickle", "image_root"}
        unknown = set(self.config) - allowed_fields
        if unknown:
            raise ValueError(f"Unknown task configuration fields: {sorted(unknown)}")
        self.root = Path(data_root).resolve()
        self.modality = self.config["modality"]
        if self.modality not in {"tabular", "text", "image"}:
            raise ValueError("modality must be tabular, text or image")
        self.features = self.config["feature_columns"]
        if (not isinstance(self.features, list) or not self.features
                or not all(isinstance(c, str) and c.strip() for c in self.features)
                or len(set(self.features)) != len(self.features)):
            raise ValueError("feature_columns must be nonempty and unique")
        if self.config.get("label_column") in self.features:
            raise ValueError("The label column cannot be an input feature")
        identifier = self.config.get("id_column")
        if identifier is not None and (identifier in self.features or identifier == self.config.get("label_column")):
            raise ValueError("The ID column must be separate from features and labels")
        source_key = "features" if self.config["format"] == "npy" else "path"
        sources = [self.root / spec[source_key] for spec in self.config["splits"].values()]
        if len({p.resolve() for p in sources}) != len(sources):
            raise ValueError("Different splits must not reference the same input file or folder")
        for i, path in enumerate(sources):
            for other in sources[i + 1:]:
                if path.exists() and other.exists() and path.samefile(other):
                    raise ValueError("Different splits must not reference the same physical source")
        self._development = None

    def _read(self, name, with_labels):
        spec = self.config["splits"][name]
        file_format = self.config["format"]
        if file_format == "npy":
            # Pickled arrays must be explicitly trusted in the task configuration.
            raw = np.load(self.root / spec["features"],
                          allow_pickle=self.config.get("trusted_pickle", False))
            if raw.ndim != 2 or raw.shape[1] != len(self.features):
                raise ValueError("Feature array shape does not match feature_columns")
            x = pd.DataFrame(raw, columns=self.features)
            y = np.load(self.root / spec["labels"], allow_pickle=False) if with_labels else None
            ids = np.array([f"{name}:{i}" for i in range(len(x))])
        elif file_format in {"csv", "image_csv"}:
            columns = list(self.features)
            id_column = self.config.get("id_column")
            if id_column:
                columns.append(id_column)
            if with_labels:
                columns.append(self.config["label_column"])
            # Read only requested fields; test targets never enter prediction data.
            # Preserve categorical codes such as "001"; numeric conversion is explicit.
            text_types = {c: str for c in self.features}
            if id_column:
                text_types[id_column] = str
            frame = pd.read_csv(self.root / spec["path"], usecols=columns,
                                dtype=text_types, keep_default_na=False)
            x = frame[self.features].copy()
            y = frame[self.config["label_column"]].to_numpy() if with_labels else None
            ids = frame[id_column].astype(str).to_numpy() if id_column else np.array(
                [f"{name}:{i}" for i in range(len(x))])
        elif file_format == "image_folder":
            folder = self.root / spec["path"]
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)
            x = pd.DataFrame({self.features[0]: [str(p.resolve()) for p in paths]})
            ids = np.array([p.relative_to(folder).as_posix() for p in paths])
            y = np.array([p.parent.name for p in paths]) if with_labels else None
            if with_labels and any(p.parent.parent != folder for p in paths):
                raise ValueError("Labeled image folders must use split/class_name/image layout")
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        if self.modality == "image":
            if len(self.features) != 1:
                raise ValueError("Image data requires exactly one path column")
            base = self.root / self.config.get("image_root", ".")
            x[self.features[0]] = x[self.features[0]].map(lambda p: str((base / str(p)).resolve()))
        if not len(x) or len(set(ids)) != len(ids) or any(not str(i).strip() for i in ids):
            raise ValueError("A split must be nonempty with unique sample IDs")
        if y is not None:
            if y.ndim != 1 or len(y) != len(x):
                raise ValueError("Labels must be one-dimensional and match feature rows")
            if pd.isna(y).any() or any(str(v).strip() == "" for v in y):
                raise ValueError("Missing labels are not supported")
            expected = self.config.get("allowed_labels")
            if expected is not None and not set(y).issubset(set(expected)):
                raise ValueError("Unexpected target labels")
        return DataSplit(name, x, y, ids)

    def load_development(self):
        """Return train/val only. Generated splits require an explicit policy."""
        if self._development is not None:
            return self._development
        train = self._read("train", True)
        if "val" in self.config["splits"]:
            val = self._read("val", True)
        else:
            policy = self.config["validation_split"]
            train_idx, val_idx = train_test_split(
                np.arange(len(train.x)), test_size=policy["fraction"],
                random_state=policy["seed"], stratify=train.y)
            source = train
            train = self._subset(source, train_idx, "train")
            val = self._subset(source, val_idx, "val")
        if len(np.unique(train.y)) < 2:
            raise ValueError("Training data must contain at least two classes")
        if not set(val.y).issubset(set(train.y)):
            raise ValueError("Validation labels contain a class absent from training")
        self._development = train, val
        return self._development

    @staticmethod
    def _subset(source, indices, name):
        return DataSplit(name, source.x.iloc[indices].reset_index(drop=True),
                         source.y[indices], source.sample_ids[indices])

    def load_prediction(self, split="test"):
        """Load inputs without targets, only after the Agent selects a solution."""
        return self._read(split, False)

    def load_final_evaluation(self, *, final_evaluation=False):
        """Explicit final-stage access. The caller must keep results out of search."""
        if not final_evaluation:
            raise PermissionError("Test labels require final_evaluation=True")
        return self._read("test", True)

    def audit(self):
        """Development audit; never opens test files or changes official splits."""
        train, val = self.load_development()
        summaries = {}
        hashes = {}
        for split in (train, val):
            canonical = split.x.fillna("<MISSING>").astype(str)
            hashes[split.name] = pd.util.hash_pandas_object(canonical, index=False)
            labels, counts = np.unique(split.y, return_counts=True)
            summaries[split.name] = {
                "rows": len(split.x), "columns": list(split.x.columns),
                "label_counts": {str(k): int(v) for k, v in zip(labels, counts)},
                "missing_counts": {c: int(split.x[c].map(is_missing).sum()) for c in self.features},
                "unique_counts": {c: int(canonical[c].nunique()) for c in self.features},
                "duplicate_input_rows": int(hashes[split.name].duplicated().sum()),
                "sample_ids": split.sample_ids.tolist(),
                "input_fingerprint": hashlib.sha256(hashes[split.name].to_numpy().tobytes()).hexdigest(),
            }
        overlap = len(set(hashes["train"]) & set(hashes["val"]))
        return {"modality": self.modality, "metric": self.config["metric"],
                "split_policy": "official" if "val" in self.config["splits"] else self.config["validation_split"],
                "splits": summaries, "train_val_shared_unique_inputs": overlap,
                "warnings": ["Shared input rows detected; official splits were preserved."] if overlap else []}


def is_missing(value):
    return bool(pd.isna(value)) or str(value).strip().lower() in {"", "nan", "none", "null", "?"}
