import copy

import pytest

from partd.vision_policy import capabilities, suggest
from partd.vision_pipeline import validate_candidate


def completed(candidate, score=.8, metric="accuracy", **kwargs):
    return {"status": "success", "candidate": candidate,
        "primary_metric": {"name": metric, "direction": "minimize" if metric == "log_loss" else "maximize", "value": score}, **kwargs}


def test_policy_covers_baselines_then_refines_and_stops():
    request = {"schema_version": 1, "history": [], "policy": {"max_experiments": 6, "patience": 5}}
    decisions = []
    for i in range(6):
        decision = suggest(request)
        assert decision["action"] == "experiment"
        candidate = decision["model"]["params"]
        assert candidate not in decisions
        decisions.append(candidate)
        request["history"].append(completed(candidate, .7 + i * .01))
    assert len({(c["backbone"], c["head"]["name"]) for c in decisions[:4]}) == 4
    assert suggest(request)["action"] == "stop"
    assert capabilities()["fine_tuning"] is False


def test_minimize_probability_policy_and_resource_stop():
    request = {"schema_version": 1, "primary_metric": "log_loss", "history": [], "policy": {"backbones": ["resnet18"]}}
    first = suggest(request)["model"]["params"]
    request["history"].append(completed(first, .5, "log_loss"))
    second = suggest(request)["model"]["params"]
    assert second["head"]["params"]["calibrate"] is True
    request["history"].append(completed(second, .4, "log_loss"))
    assert suggest(request)["model"]["params"]["head"]["name"] == "linear_svm"
    request.update(elapsed_seconds=11, policy={"time_budget_seconds": 10})
    assert suggest(request)["action"] == "stop"


def test_patience_and_failure_feedback_do_not_select_a_model():
    request = {"schema_version": 1, "history": [], "policy": {"backbones": ["resnet18"], "patience": 1}}
    first = suggest(request)["model"]["params"]
    request["history"].append(completed(first))
    second = suggest(request)["model"]["params"]
    request["history"].append(completed(second, .799))
    stopped = suggest(request)
    assert stopped["action"] == "stop" and "artifact" not in stopped
    request["history"] = [{"status": "error", "candidate": first}]
    assert suggest(request)["action"] == "stop"


@pytest.mark.parametrize("change", [{"head": {"name": "random_forest"}}, {"backbone": "dataset_answer"},
    {"head": {"name": "linear_svm", "params": {"C": -1}}}, {"training_data": "test.csv"},
    {"head": {"name": "logistic_regression", "params": {"calibrate": True}}}])
def test_candidate_rejects_invalid_capabilities(change):
    with pytest.raises(ValueError):
        validate_candidate(change)


def test_policy_rejects_mixed_metrics_and_does_not_mutate_feedback():
    request = {"schema_version": 1, "history": [completed(validate_candidate({}))]}
    original = copy.deepcopy(request)
    suggest(request)
    assert request == original
    request["history"][0]["primary_metric"]["name"] = "f1_macro"
    with pytest.raises(ValueError, match="inconsistent"):
        suggest(request)


def test_slow_and_worse_growth_is_pruned_only_for_the_same_head_settings():
    request = {"schema_version": 1, "history": [], "policy": {"max_experiments": 20, "patience": 10}}
    for i in range(4):
        candidate = suggest(request)["model"]["params"]
        request["history"].append(completed(candidate, .9 if i == 0 else .7, training={"fit_seconds": 1}))
    leading = request["history"][0]["candidate"]
    for c, score, seconds in ((.1, .85, .5), (5, .8, 5)):
        candidate = copy.deepcopy(leading)
        candidate["head"]["params"]["C"] = c
        request["history"].append(completed(candidate, score, training={"fit_seconds": seconds}))
    assert suggest(request)["model"]["params"]["feature_scaling"] == "standard"
    # A different solver limit is a separate branch, so its cost cannot prune C.
    request["history"][-1]["candidate"]["head"]["params"]["max_iter"] = 2000
    assert suggest(request)["model"]["params"]["head"]["params"]["C"] == 10
