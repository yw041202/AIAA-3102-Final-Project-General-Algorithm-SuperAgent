"""Train-only fitted preprocessing with portable state and batched images."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import DataSplit, is_missing


class Preprocessor:
    def __init__(self, modality, config=None):
        if modality not in {"tabular", "text", "image"}:
            raise ValueError("Unsupported modality")
        self.modality = modality
        self.config = deepcopy(dict(config or {}))
        allowed_options = {
            "tabular": {"encoding", "numeric_columns"},
            "text": {"combine", "analyzer", "ngram_range", "min_df", "max_features", "lowercase"},
            "image": {"size", "mean", "std", "flatten"},
        }
        unknown = set(self.config) - allowed_options[modality]
        if unknown:
            raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
        self.fitted = False

    def fit(self, train: DataSplit):
        if train.name != "train":
            raise ValueError("Preprocessing may only be fitted on the training split")
        if self.fitted:
            raise ValueError("Create a new Preprocessor for each experiment")
        self.columns = list(train.x.columns)
        self.fit_record = {
            "source_split": train.name, "rows": len(train.x),
            "sample_ids_sha256": hashlib.sha256(
                json.dumps(train.sample_ids.tolist()).encode()).hexdigest(),
            "columns": self.columns, "modality": self.modality, "config": self.config,
        }
        if self.modality == "tabular":
            self.numeric = self.config.get("numeric_columns", [])
            if not isinstance(self.numeric, list) or len(set(self.numeric)) != len(self.numeric):
                raise ValueError("numeric_columns must be a list of unique column names")
            if not set(self.numeric).issubset(self.columns):
                raise ValueError("Unknown numeric column")
            self.categorical = [c for c in self.columns if c not in self.numeric]
            encoding = self.config.get("encoding", "onehot")
            if encoding not in {"onehot", "ordinal"}:
                raise ValueError("encoding must be onehot or ordinal")
            encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
            if encoding == "ordinal":
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                         dtype=np.float32)
            steps = []
            if self.categorical:
                steps.append(("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                    ("encode", encoder),
                ]), self.categorical))
            if self.numeric:
                steps.append(("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                ]), self.numeric))
            self.transformer = ColumnTransformer(steps, sparse_threshold=1.0)
            self.transformer.fit(self._tabular_frame(train.x))
        elif self.modality == "text":
            self.combine = self.config.get("combine", "concat")
            if self.combine not in {"concat", "difference"}:
                raise ValueError("combine must be concat or difference")
            if self.combine == "difference" and len(self.columns) != 2:
                raise ValueError("difference requires exactly two ordered text columns")
            self.transformer = TfidfVectorizer(
                analyzer=self.config.get("analyzer", "word"),
                ngram_range=tuple(self.config.get("ngram_range", [1, 2])),
                min_df=self.config.get("min_df", 1),
                max_features=self.config.get("max_features", 30000),
                lowercase=self.config.get("lowercase", True), dtype=np.float32,
            )
            frame = self._text_frame(train.x)
            corpus = [text for column in self.columns for text in frame[column]]
            self.transformer.fit(corpus)
        else:
            if len(self.columns) != 1:
                raise ValueError("Images require one path column")
            self.size = tuple(self.config.get("size", [64, 64]))
            if len(self.size) != 2 or any(not isinstance(n, int) or n <= 0 for n in self.size):
                raise ValueError("Image size must be positive [height, width]")
            self.mean = np.asarray(self.config.get("mean", [0, 0, 0]), dtype=np.float32).reshape(3, 1, 1)
            self.std = np.asarray(self.config.get("std", [1, 1, 1]), dtype=np.float32).reshape(3, 1, 1)
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
                raise ValueError("Image normalization requires finite mean and positive std")
        self.fitted = True
        return self

    def _validate(self, frame):
        if not self.fitted:
            raise RuntimeError("Call fit(train) first")
        if list(frame.columns) != self.columns:
            raise ValueError("Input feature columns/order differ from fitted preprocessing")

    def _tabular_frame(self, frame):
        result = frame.copy()
        for column in self.categorical:
            result[column] = result[column].map(lambda value: np.nan if is_missing(value) else str(value))
        for column in self.numeric:
            result[column] = pd.to_numeric(result[column].map(
                lambda value: np.nan if is_missing(value) else value), errors="raise")
            if np.isinf(result[column].to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite numeric values in {column}")
        return result

    @staticmethod
    def _text_frame(frame):
        return frame.fillna("").astype(str)

    def transform(self, frame):
        """Images return a batch of float32 NCHW tensors; text returns CSR."""
        self._validate(frame)
        if self.modality == "tabular":
            return self.transformer.transform(self._tabular_frame(frame)).astype(np.float32)
        if self.modality == "text":
            text = self._text_frame(frame)
            matrices = [self.transformer.transform(text[c]) for c in self.columns]
            if self.combine == "difference":
                return (matrices[0] - matrices[1]).tocsr()
            return sparse.hstack(matrices, format="csr")
        images = []
        for path in frame[self.columns[0]]:
            try:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    resized = rgb.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
                    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
                    images.append((array - self.mean) / self.std)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot decode image {path}: {exc}") from exc
        if not images:
            if self.config.get("flatten", False):
                return np.empty((0, 3 * self.size[0] * self.size[1]), dtype=np.float32)
            return np.empty((0, 3, *self.size), dtype=np.float32)
        result = np.stack(images)
        if self.config.get("flatten", False):
            result = result.reshape(len(result), -1)
        return result

    def iter_batches(self, split, batch_size=32):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(split.x), batch_size):
            stop = start + batch_size
            yield (self.transform(split.x.iloc[start:stop]),
                   None if split.y is None else split.y[start:stop],
                   split.sample_ids[start:stop])

    def save(self, path):
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(self.fit_record, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path):
        """Only load artifacts created by trusted project code (joblib uses pickle)."""
        result = joblib.load(path)
        if not isinstance(result, Preprocessor) or not result.fitted:
            raise ValueError("Invalid preprocessing artifact")
        return result
