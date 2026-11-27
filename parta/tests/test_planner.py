import json
from decimal import Decimal
from pathlib import Path

import pytest

from parta.agent import LLMPlanner, load_task
from parta.demo import make_fixture
from partc.llm_client import MockBackend


def test_user_key_alias_does_not_reach_training_environment(monkeypatch, tmp_path):
    from partc.config import load_config, sanitize_environ
    monkeypatch.setenv("AIAA3102_API_KEY", "dummy-project-secret")
    monkeypatch.setenv("ZZZ_API_KEY", "dummy-legacy-secret")
    assert load_config(str(tmp_path / "absent.env")).api_key == "dummy-project-secret"
    clean = sanitize_environ()
    assert "AIAA3102_API_KEY" not in clean
    assert "ZZZ_API_KEY" not in clean


def test_real_c_client_accounts_for_planning_and_persists_budget(tmp_path_factory, monkeypatch):
    folder = tmp_path_factory.mktemp("p")
    config = load_task(make_fixture(folder / "d"))
    monkeypatch.setattr("partc.llm_client.SDKBackend", lambda cfg: MockBackend(
        content='{"action":"stop","reason":"done"}', prompt_tokens=20, completion_tokens=10))
    planner = LLMPlanner(config, folder / "r1", str(folder / "no-env"))
    result = planner.propose({"data": {}, "history": []})
    assert result["request_status"] == "success"
    assert Decimal(planner.summary()["spent_rmb"]) > 0
    second = LLMPlanner(config, folder / "r2", str(folder / "no-env"))
    assert second.summary()["spent_rmb"] == planner.summary()["spent_rmb"]
    assert (folder / "r1" / "llm_logs" / "llm.jsonl").is_file()


def test_actual_c_budget_rejection_makes_no_backend_call(tmp_path_factory, monkeypatch):
    folder = tmp_path_factory.mktemp("p")
    config = load_task(make_fixture(folder / "d"))
    config["budget_rmb"] = "0.00000001"
    class NeverCalled:
        def complete(self, *args):
            pytest.fail("Budget rejection must precede network access")
    monkeypatch.setattr("partc.llm_client.SDKBackend", lambda cfg: NeverCalled())
    planner = LLMPlanner(config, folder / "r", str(folder / "no-env"))
    assert planner.propose({"history": []})["request_status"] == "budget_exceeded"
    assert planner.summary()["spent_rmb"] == "0"


def test_damaged_ledger_blocks_backend_initialization(tmp_path_factory, monkeypatch):
    from partc import BudgetStateError
    folder = tmp_path_factory.mktemp("badledger")
    config = load_task(make_fixture(folder / "d"))
    path = Path(config['output_root']) / 'budget' / 'ledger.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{broken', encoding='utf-8')
    def forbidden(*args, **kwargs):
        pytest.fail('No backend may be initialized with an invalid ledger')
    monkeypatch.setattr('partc.llm_client.SDKBackend', forbidden)
    with pytest.raises(BudgetStateError):
        LLMPlanner(config, folder / 'run', str(folder / 'absent.env'))
    assert path.read_text() == '{broken'
