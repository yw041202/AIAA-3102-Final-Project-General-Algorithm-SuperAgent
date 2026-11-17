from __future__ import annotations

import math
from sklearn import metrics as skm

DIRECTIONS = {"accuracy": "maximize", "balanced_accuracy": "maximize", "f1_macro": "maximize",
              "precision_macro": "maximize", "recall_macro": "maximize", "roc_auc": "maximize", "log_loss": "minimize"}


def calculate_metrics(y, predicted, probabilities, classes, names):
    values = {}
    for name in names:
        if name not in DIRECTIONS:
            raise ValueError(f"unsupported metric {name!r}; supported: {sorted(DIRECTIONS)}")
        if name == "accuracy":
            value = skm.accuracy_score(y, predicted)
        elif name == "balanced_accuracy":
            value = skm.balanced_accuracy_score(y, predicted)
        elif name in {"f1_macro", "precision_macro", "recall_macro"}:
            function = {"f1_macro": skm.f1_score, "precision_macro": skm.precision_score, "recall_macro": skm.recall_score}[name]
            value = function(y, predicted, average="macro", zero_division=0)
        elif name == "log_loss":
            value = skm.log_loss(y, probabilities, labels=classes)
        elif len(classes) == 2:
            value = skm.roc_auc_score(y == classes[1], probabilities[:, 1])
        else:
            value = skm.roc_auc_score(y, probabilities, labels=classes, multi_class="ovr", average="macro")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} is undefined on this validation split")
        values[name] = value
    return values
