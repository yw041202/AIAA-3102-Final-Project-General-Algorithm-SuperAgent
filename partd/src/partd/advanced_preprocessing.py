"""Optional D adapters over B's raw frames; train-only state is saved with models."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OrdinalEncoder

from partb import Preprocessor
from partb.data import is_missing


class AdvancedPreprocessor(Preprocessor):
    def __init__(self, modality, config, backend):
        self.modality, self.config, self.backend = modality, deepcopy(config), backend
        self.config["d_backend"] = backend
        self.fitted = False

    def fit(self, train):
        if train.name != "train" or self.fitted:
            raise ValueError("create a new processor and fit training data only")
        self.columns = list(train.x.columns)
        self.fit_record = {"source_split": "train", "rows": len(train.x), "columns": self.columns,
            "sample_ids_sha256": hashlib.sha256(json.dumps(train.sample_ids.tolist()).encode()).hexdigest(),
            "config": self.config, "modality": self.modality}
        if self.backend == "word_char":
            allowed = {"d_backend", "analyzer", "combine", "max_features", "min_df", "lowercase", "char_weight"}
            if set(self.config) - allowed: raise ValueError("unsupported combined TF-IDF settings")
            self.parts = []
            for analyzer, ngrams in (("word", [1, 2]), ("char", [3, 5])):
                options = {k: v for k, v in self.config.items() if k in {"combine", "max_features", "min_df", "lowercase"}}
                options.update(analyzer=analyzer, ngram_range=ngrams)
                self.parts.append(Preprocessor("text", options).fit(train))
            self.weight = self.config.get("char_weight", 1.0)
            if type(self.weight) not in {int, float} or not 0 < self.weight < float("inf"):
                raise ValueError("char_weight must be finite and positive")
            self.normalizer = Normalizer()
        else:
            if set(self.config) - {"numeric_columns", "encoding", "d_backend"}:
                raise ValueError("native tabular processor accepts numeric_columns and encoding only")
            if self.config.get("encoding", "native") != "native":
                raise ValueError("native boosting requires encoding=native, not onehot/ordinal preprocessing")
            self.numeric = self.config.get("numeric_columns", [])
            if (not isinstance(self.numeric, list) or not all(isinstance(c, str) for c in self.numeric)
                    or len(set(self.numeric)) != len(self.numeric) or not set(self.numeric) <= set(self.columns)):
                raise ValueError("invalid numeric columns")
            self.categorical = [c for c in self.columns if c not in self.numeric]
            if self.backend == "hist":
                steps = []
                if self.categorical:
                    steps.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64), self.categorical))
                if self.numeric: steps.append(("num", "passthrough", self.numeric))
                self.transformer = ColumnTransformer(steps, sparse_threshold=0)
                self.transformer.fit(self._native_frame(train.x))
                if self.categorical and any(len(c) > 255 for c in self.transformer.named_transformers_["cat"].categories_):
                    raise ValueError("hist native categorical columns must have at most 255 categories")
            elif self.backend != "catboost": raise ValueError("unsupported advanced backend")
        self.fitted = True
        return self

    def _native_frame(self, frame):
        result = frame.copy()
        for column in self.categorical:
            result[column] = result[column].map(lambda v: "__MISSING__" if is_missing(v) else str(v))
        for column in self.numeric:
            result[column] = pd.to_numeric(result[column].map(lambda v: np.nan if is_missing(v) else v), errors="raise").astype(float)
            if np.isinf(result[column].to_numpy()).any(): raise ValueError("numeric inputs contain infinity")
        return result

    def transform(self, frame):
        self._validate(frame)
        if self.backend == "word_char":
            parts = [p.transform(frame) for p in self.parts]
            return self.normalizer.transform(sparse.hstack([parts[0], parts[1] * self.weight], format="csr"))
        frame = self._native_frame(frame)
        return frame if self.backend == "catboost" else self.transformer.transform(frame)

    def model_parameters(self):
        if self.backend == "catboost":
            return {"cat_features": [self.columns.index(c) for c in self.categorical]}
        if self.backend == "hist":
            return {"categorical_features": list(range(len(self.categorical))) if self.categorical else None}
        return {}
