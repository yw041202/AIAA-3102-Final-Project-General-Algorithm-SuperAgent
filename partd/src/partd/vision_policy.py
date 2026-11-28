"""Dataset-independent image search suggestions. Never trains or exports a model."""
from __future__ import annotations

import math

from .metrics import DIRECTIONS
from .vision import BACKBONES, json_digest
from .vision_pipeline import validate_candidate


def capabilities():
    return {"schema_version": 1, "model_name": "frozen_vision", "modality": "image",
        "backbones": BACKBONES, "heads": ["logistic_regression", "linear_svm"],
        "preprocessing": ["official", "resize224"], "feature_scaling": ["l2", "standard", "none"],
        "metrics": DIRECTIONS, "probabilities": "LR, or linear_svm with training-only calibrate=true",
        "device": "cpu", "fine_tuning": False, "downloads_during_execution": False,
        "selection_owner": "A", "same_run_feature_cache": True,
        "feedback_fields": ["primary_metric", "training", "warnings", "cache", "elapsed_seconds", "error", "stage"]}


def suggest(request):
    """Propose one next candidate from validation/resource feedback, or a stop.

    This is an optional search prior for A, not a replacement for its planner.
    History is a list of D train results; C-only failures need candidate/status.
    """
    if not isinstance(request, dict) or set(request) - {"schema_version", "history", "primary_metric", "policy", "elapsed_seconds"}:
        raise ValueError("vision-suggest accepts schema_version/history/primary_metric/policy/elapsed_seconds")
    if request.get("schema_version") != 1:
        raise ValueError("policy request requires schema_version=1")
    metric = request.get("primary_metric", "accuracy")
    if metric not in DIRECTIONS:
        raise ValueError("unsupported policy metric")
    policy = request.get("policy", {})
    if not isinstance(policy, dict) or set(policy) - {"backbones", "max_experiments", "patience", "min_gain", "time_budget_seconds", "slowdown_factor"}:
        raise ValueError("unsupported image policy settings")
    backbones = policy.get("backbones", list(BACKBONES))
    if not isinstance(backbones, list) or not backbones or len(set(backbones)) != len(backbones) or any(b not in BACKBONES for b in backbones):
        raise ValueError("policy backbones must be a nonempty unique supported list")
    maximum, patience = policy.get("max_experiments", 8), policy.get("patience", 3)
    gain, slowdown = policy.get("min_gain", .001), policy.get("slowdown_factor", 2.0)
    if any(type(x) is not int or x < 1 for x in (maximum, patience)):
        raise ValueError("max_experiments and patience must be positive integers")
    if any(type(x) not in {int, float} or not math.isfinite(x) for x in (gain, slowdown)) or gain < 0 or slowdown <= 1:
        raise ValueError("min_gain must be nonnegative and slowdown_factor must exceed one")
    elapsed, limit = request.get("elapsed_seconds", 0), policy.get("time_budget_seconds")
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    if limit is not None and (type(limit) not in {int, float} or not math.isfinite(limit) or limit <= 0):
        raise ValueError("time_budget_seconds must be finite and positive")
    history = request.get("history", [])
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    good, seen, stale = [], set(), 0
    direction = 1 if DIRECTIONS[metric] == "maximize" else -1
    best_utility = None
    for row in history:
        if not isinstance(row, dict) or row.get("status") not in {"success", "error"}:
            raise ValueError("history entries need status=success/error")
        candidate = validate_candidate(row["candidate"])
        seen.add(json_digest(candidate))
        if row["status"] == "success":
            primary = row["primary_metric"]
            score = primary["value"]
            if primary["name"] != metric or primary["direction"] != DIRECTIONS[metric] or type(score) not in {int, float} or not math.isfinite(score):
                raise ValueError("history has an inconsistent primary metric")
            good.append((direction * score, candidate, row))
            if best_utility is None or direction * score > best_utility + gain:
                best_utility, stale = direction * score, 0
            else:
                stale += 1
        else:
            stale += 1

    def stop(reason):
        return {"action": "stop", "reason": reason, "attempts": len(history),
                "selection_owner": "A", "test_set_used": False}

    if len(history) >= maximum:
        return stop("candidate budget reached")
    if limit is not None and elapsed >= limit:
        return stop("wall-clock budget reached; retain successful candidates")
    # Compare basic head/backbone choices before patience-based local stopping.
    calibrate = metric in {"log_loss", "roc_auc"}
    baseline = []
    for head in ("logistic_regression", "linear_svm"):
        for backbone in backbones:
            params = {"C": 1.0}
            if head == "linear_svm" and calibrate:
                params["calibrate"] = True
            baseline.append(validate_candidate({"backbone": backbone,
                "head": {"name": head, "params": params}, "feature_scaling": "l2"}))

    def propose(candidate, reason):
        return {"action": "experiment", "reason": reason,
            "model": {"name": "frozen_vision", "params": candidate},
            "preprocess_options": {}, "selection_owner": "A", "test_set_used": False}

    if stale >= patience and not good:
        return stop("repeated failures; inspect C/D feedback before another run")
    for candidate in baseline:
        if json_digest(candidate) not in seen:
            return propose(candidate, "compare frozen backbone/head baselines with reusable L2 features")
    if stale >= patience:
        return stop("validation improvement below min_gain for patience attempts")
    if not good:
        return stop("no successful image baseline")

    # Stable order resolves ties; no dataset names or development scores are embedded.
    good.sort(key=lambda x: x[0], reverse=True)
    best = good[0][1]
    center = best["head"]["params"].get("C", 1.0)
    for c_value in (center / 10, center * 10):
        if not .0001 <= c_value <= 1000:
            continue
        candidate = {**best, "head": {**best["head"], "params": {**best["head"]["params"], "C": c_value}}}
        # Prune growth only when the same branch became slower AND worse at a larger C.
        def family(config):
            return json_digest({**config, "head": {**config["head"], "params": {
                k: v for k, v in config["head"]["params"].items() if k != "C"}}})
        branch = [x for x in good if family(x[1]) == family(best)]
        blocked = False
        for utility, config, row in branch:
            for lower_u, lower_c, lower_row in branch:
                c1, c0 = config["head"]["params"].get("C", 1.0), lower_c["head"]["params"].get("C", 1.0)
                t1, t0 = row.get("training", {}).get("fit_seconds", 0), lower_row.get("training", {}).get("fit_seconds", 0)
                if c_value > c1 > c0 and utility < lower_u - gain and t0 > 0 and t1 > slowdown * t0:
                    blocked = True
        if not blocked and json_digest(candidate) not in seen:
            return propose(candidate, "refine regularization around the best validation candidate; reuse its encoder cache")
    alternative = {**best, "feature_scaling": "standard" if best["feature_scaling"] != "standard" else "l2"}
    if json_digest(alternative) not in seen:
        return propose(alternative, "compare a train-only scaling alternative for the leading branch")
    return stop("local refinement exhausted; A may expand the search explicitly")
