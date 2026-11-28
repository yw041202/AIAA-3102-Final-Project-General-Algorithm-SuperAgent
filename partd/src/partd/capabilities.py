"""Machine-readable candidate contract; the Agent owns planning and selection."""
from copy import deepcopy
from .candidates import MODEL_NAMES
from .metrics import DIRECTIONS


def capabilities(modality=None):
    if modality not in {None, "tabular", "text", "image"}:
        raise ValueError("unsupported modality")
    definitions = {
        "logistic_regression": (["tabular", "text"], "baseline", ["C", "max_iter", "solver", "class_weight", "feature_scaling", "svd_components"], "always"),
        "random_forest": (["tabular", "text"], "baseline", ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight", "n_jobs"], "always"),
        "linear_svm": (["tabular", "text"], "priority", ["C", "max_iter", "tol", "class_weight", "calibrate", "feature_scaling", "svd_components"], "when calibrated"),
        "rbf_svm": (["tabular", "text"], "priority", ["C", "gamma", "cache_size", "probability", "class_weight", "feature_scaling", "svd_components"], "when params.probability=true"),
        "complement_nb": (["tabular", "text"], "fast_backup", ["alpha", "fit_prior", "norm", "feature_scaling"], "always"),
        "mlp": (["tabular", "text"], "optional", ["hidden_layer_sizes", "alpha", "max_iter", "early_stopping", "n_iter_no_change", "batch_size", "solver", "learning_rate_init", "feature_scaling", "svd_components"], "always"),
        "gradient_boosting": (["tabular", "text"], "priority", ["n_estimators", "max_depth", "learning_rate", "min_samples_leaf", "subsample"], "always"),
        "catboost": (["tabular"], "priority", ["iterations", "depth", "learning_rate", "l2_leaf_reg", "thread_count", "max_ctr_complexity", "boosting_type", "random_strength"], "always"),
        "hist_gradient_boosting": (["tabular"], "priority", ["max_iter", "learning_rate", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "early_stopping", "n_iter_no_change", "class_weight"], "always"),
    }
    models = []
    for name in MODEL_NAMES:
        modalities, priority, parameters, probability = definitions[name]
        record = {"name": name, "modalities": modalities, "search_priority": priority,
            "planner_parameter_fields": parameters, "probabilities": probability,
            "requires_nonnegative_features": name == "complement_nb",
            "raw_native_tabular": name in {"catboost", "hist_gradient_boosting"},
            "optional_dependency": "catboost==1.2.8" if name == "catboost" else None}
        if record["raw_native_tabular"]:
            record["constraints"] = ["preprocessing=partb; encoding=native; explicit numeric_columns",
                "categorical positions are derived by D; do not apply generic scaling or SVD"]
            if name == "hist_gradient_boosting":
                record["constraints"].append("at most 255 categories per categorical column")
        elif name == "complement_nb":
            record["constraints"] = ["nonnegative inputs; no signed text difference, standard scaling or SVD"]
        else:
            record["constraints"] = ["fit preprocessing on training data only; retain its state for prediction"]
        if name == "rbf_svm": record["resource_note"] = "bound sample size, cache_size and timeout; compare with linear_svm"
        if name == "mlp": record["resource_note"] = "optional exploration; inspect convergence, memory and seed variability"
        models.append(record)
    models.append({"name": "frozen_vision", "modalities": ["image"], "search_priority": "priority",
        "details_command": "vision-capabilities", "policy_command": "vision-suggest",
        "constraints": ["raw B image inputs; pre-provision pinned weights; keep backbone frozen",
                        "D owns weight-specific transforms; do not apply B's 64px preprocessor first"]})
    return deepcopy({"schema_version": 1, "selection_owner": "A", "task_identity_lookup": False,
        "models": [m for m in models if modality is None or modality in m["modalities"]], "metrics": DIRECTIONS,
        "text_preprocessing": {"analyzer": ["word", "char", "word_char"], "combine": ["concat", "difference"],
            "joint_fields": ["max_features", "min_df", "lowercase", "char_weight"],
            "joint_constraints": "word_char uses fixed word [1,2]/char [3,5]; difference requires two ordered columns"},
        "runtime_path_fields": "A supplies actual paths from deployment config; model-generated paths are not needed",
        "strategy_document": "ABC_HANDOFF.md"})
