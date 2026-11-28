"""Reusable candidates; final selection remains the calling Agent's responsibility."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC

MODEL_NAMES = ("logistic_regression", "random_forest", "linear_svm", "rbf_svm",
               "complement_nb", "mlp", "gradient_boosting", "catboost", "hist_gradient_boosting")


class NonNegativeFeatures(TransformerMixin, BaseEstimator):
    """Reject incompatible NB inputs without clipping away their meaning."""
    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        values = X.data if sparse.issparse(X) else np.asarray(X)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("complement_nb requires finite nonnegative features; use nonnegative TF-IDF concat, not signed difference or standardized numeric features")
        return X


def build_candidate_pipeline(model, seed, preprocessing):
    if not isinstance(model, dict) or set(model) - {"name", "params"}:
        raise ValueError("model accepts only name and params; hyperparameters belong inside params")
    if model.get("name") not in MODEL_NAMES:
        raise ValueError(f"model.name must be one of {', '.join(MODEL_NAMES)}")
    if preprocessing not in {"numeric", "passthrough"}:
        raise ValueError("preprocessing must be numeric or passthrough")
    if not isinstance(model.get("params", {}), dict):
        raise ValueError("model.params must be an object")
    params = dict(model.get("params", {}))
    name = model["name"]
    if "random_state" in params and params["random_state"] != seed:
        raise ValueError("set random_state through seed; conflicting model parameter")
    if name != "complement_nb":
        params["random_state"] = seed
    elif "random_state" in params:
        raise ValueError("complement_nb is deterministic and does not accept random_state")

    scaling = params.pop("feature_scaling", "auto")
    if scaling not in {"auto", "none", "standard", "maxabs"}:
        raise ValueError("feature_scaling must be auto, none, standard or maxabs")
    components = params.pop("svd_components", None)
    if components is not None and (type(components) is not int or components < 1):
        raise ValueError("svd_components must be a positive integer")
    if name == "complement_nb" and (components is not None or scaling == "standard"):
        raise ValueError("complement_nb cannot use signed SVD or standard-scaled features")
    if name in {"catboost", "hist_gradient_boosting"} and (components is not None or scaling not in {"auto", "none"}):
        raise ValueError("native boosting cannot use generic scaling or SVD; preserve categorical codes")
    calibrate = params.pop("calibrate", False)
    if type(calibrate) is not bool or (calibrate and name != "linear_svm"):
        raise ValueError("calibrate is a boolean option for linear_svm only")

    if name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        classifier = LogisticRegression(**params)
    elif name == "random_forest":
        params.setdefault("n_estimators", 100)
        params.setdefault("n_jobs", 1)
        classifier = RandomForestClassifier(**params)
    elif name == "linear_svm":
        params.setdefault("dual", "auto")
        params.setdefault("max_iter", 10000)
        classifier = LinearSVC(**params)
        if calibrate:
            classifier = CalibratedClassifierCV(classifier, method="sigmoid", cv=3, n_jobs=1)
    elif name == "rbf_svm":
        params.setdefault("kernel", "rbf")
        if params["kernel"] != "rbf":
            raise ValueError("rbf_svm requires kernel=rbf")
        params.setdefault("cache_size", 256)
        classifier = SVC(**params)
    elif name == "complement_nb":
        classifier = ComplementNB(**params)
    elif name == "mlp":
        params.setdefault("hidden_layer_sizes", (64,))
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        params.setdefault("early_stopping", True)
        params.setdefault("max_iter", 200)
        classifier = MLPClassifier(**params)
    elif name == "gradient_boosting":
        classifier = GradientBoostingClassifier(**params)
    elif name == "hist_gradient_boosting":
        params.setdefault("early_stopping", True)
        params.setdefault("max_iter", 300)
        classifier = HistGradientBoostingClassifier(**params)
    else:
        from .catboost_adapter import FlatCatBoostClassifier
        params.pop("random_state", None)
        params.setdefault("random_seed", seed)
        if params["random_seed"] != seed: raise ValueError("set CatBoost random_seed through seed")
        params.setdefault("iterations", 500)
        params.setdefault("learning_rate", .05)
        params.setdefault("depth", 6)
        params.setdefault("thread_count", 4)
        params.setdefault("allow_writing_files", False)
        params.setdefault("verbose", False)
        classifier = FlatCatBoostClassifier(**params)

    steps = []
    if preprocessing == "numeric":
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    if name == "complement_nb":
        steps.append(("nonnegative", NonNegativeFeatures()))
    if components is not None:
        steps.append(("svd", TruncatedSVD(n_components=components, random_state=seed)))
    if scaling == "auto":
        scaling = "standard" if preprocessing == "numeric" and name in {
            "logistic_regression", "linear_svm", "rbf_svm", "mlp"} else "none"
    if scaling == "standard":
        steps.append(("scaler", StandardScaler(with_mean=preprocessing == "numeric" or components is not None)))
    elif scaling == "maxabs":
        steps.append(("scaler", MaxAbsScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)
