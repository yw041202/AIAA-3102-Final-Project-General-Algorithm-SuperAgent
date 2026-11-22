import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parta.agent import load_task, parse_decision, run, write_json
from parta.demo import OfflinePlanner, make_fixture
from partc import ChatMessage, PythonExecutor
from partc.llm_client import estimate_input_tokens


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Nested model artifacts otherwise exceed legacy Windows MAX_PATH.
    return tmp_path_factory.mktemp("a")


@pytest.mark.parametrize("modality,metric", [("tabular", "accuracy"), ("text", "log_loss")])
def test_real_bcd_closed_loop(tmp_path, modality, metric):
    config = load_task(make_fixture(tmp_path / "inputs", modality, metric))
    planner = OfflinePlanner()
    result = run(config, planner=planner, offline=True)
    assert result["status"] == "success", result
    assert len(result["history"]) == 2
    assert all(row["status"] == "success" for row in result["history"])
    values = [row["metric"]["value"] for row in result["history"]]
    assert result["selected"]["metric"]["value"] == (min(values) if metric == "log_loss" else max(values))
    assert result["prediction"]["n_predictions"] == 12
    assert result["budget"]["live_calls"] == 0
    assert "test" not in planner.contexts[0]["data"]["splits"]
    assert "DO_NOT_READ_TEST_LABEL" not in json.dumps(planner.contexts)
    assert Path(result["final_model"], "preprocessor.joblib").is_file()
    with pytest.raises(FileExistsError):
        run(config, planner=OfflinePlanner(), offline=True)


class SequencePlanner:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.contexts = []

    def propose(self, context):
        self.contexts.append(json.loads(json.dumps(context)))
        return next(self.responses)

    def summary(self):
        return {"spent_rmb": "0", "offline": True}


def experiment(params=None):
    return {"request_status": "success", "output_text": json.dumps({"action": "experiment", "reason": "test",
            "model": {"name": "logistic_regression", "params": params or {}}, "preprocess_options": {}})}


def test_real_training_failure_is_fed_back_and_repaired(tmp_path):
    config = load_task(make_fixture(tmp_path / "inputs"))
    planner = SequencePlanner([experiment({"C": -1}), experiment({"C": 1}),
        {"request_status": "success", "output_text": '{"action":"stop","reason":"repair succeeded"}'}])
    result = run(config, planner=planner, offline=True)
    assert result["status"] == "success", result
    assert result["history"][0]["status"] == "error"
    assert planner.contexts[1]["history"][0]["status"] == "error"
    assert result["selected"]["experiment_id"] == "experiment_002"


@pytest.mark.parametrize("status", ["budget_exceeded", "success_uncertain", "error"])
def test_failed_call_stops_without_execution(tmp_path, status):
    config = load_task(make_fixture(tmp_path / "inputs"))
    planner = SequencePlanner([{"request_status": status, "output_text": "ignored"}])
    def forbidden(*args, **kwargs):
        pytest.fail("No experiment permitted after unsuccessful LLM call")
    result = run(config, planner=planner, executor=forbidden, offline=True)
    assert result["status"] == "failed"
    assert result["stop_reason"] == status
    assert len(planner.contexts) == 1


def test_invalid_decisions_are_bounded(tmp_path):
    config = load_task(make_fixture(tmp_path / "inputs"))
    config["patience"] = 2
    planner = SequencePlanner([{"request_status": "success", "output_text": "not JSON"}] * 2)
    result = run(config, planner=planner, offline=True)
    assert result["stop_reason"] == "patience"
    assert len(result["history"]) == 2
    assert planner.contexts[1]["history"][0]["status"] == "error"


def test_planner_cannot_change_data_or_metric():
    candidate = json.loads(experiment()["output_text"])
    candidate["data_root"] = "test"
    with pytest.raises(ValueError):
        parse_decision(json.dumps(candidate), "tabular")


def test_no_candidate_stop_is_failure(tmp_path):
    config = load_task(make_fixture(tmp_path / "inputs"))
    planner = SequencePlanner([{"request_status": "success", "output_text": '{"action":"stop","reason":"cannot proceed"}'}])
    assert run(config, planner=planner, offline=True)["status"] == "failed"


def test_real_executor_timeout(tmp_path):
    result = PythonExecutor(cwd=str(tmp_path), timeout_seconds=0.3).execute_code("import time; time.sleep(30)")
    assert result.status == "timeout"
    assert result.exit_code is not None


def test_cjk_reservation_is_conservative():
    text = "中文预算验证" * 100
    assert estimate_input_tokens([ChatMessage.user(text)]) >= len(text.encode("utf-8"))


@pytest.mark.parametrize("budget", ["21", "0", "NaN", "Infinity"])
def test_bad_budget_rejected(tmp_path, budget):
    path = make_fixture(tmp_path / "inputs")
    config = json.loads(path.read_text())
    config["budget_rmb"] = budget
    write_json(path, config)
    with pytest.raises(ValueError):
        load_task(path)
