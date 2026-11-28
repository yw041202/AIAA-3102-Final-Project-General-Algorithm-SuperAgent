"""Optional CatBoost classifier with sklearn-style one-dimensional labels."""
import numpy as np
from catboost import CatBoostClassifier

class FlatCatBoostClassifier(CatBoostClassifier):
    def predict(self, data, prediction_type="Class", **kwargs):
        values = super().predict(data, prediction_type=prediction_type, **kwargs)
        return np.asarray(values).reshape(-1) if prediction_type == "Class" else values
