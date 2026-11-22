"""End-to-end Mock Agent demo test (fully offline)."""

from __future__ import annotations

from decimal import Decimal

import mock_agent


def test_mock_agent_demo(tmp_path):
    summary = mock_agent.run(workdir=str(tmp_path), model="deepseek-v4.1-flash-1017", budget_rmb="20")

    # LLM calls happened and were recorded
    assert summary["n_llm_calls"] >= 2
    assert Decimal(summary["llm_cost_rmb"]) > 0
    assert Decimal(summary["budget_rmb"]) == Decimal("20")

    # execution succeeded through the real code executor
    assert summary["execution_status"] == "success"
    assert summary["execution_exit_code"] == 0

    # budget-exceeded path was demonstrated
    assert summary["budget_exceeded_demo_triggered"] is True

    # summaries are internally consistent
    assert summary["llm_summary"]["n_llm_calls"] == summary["n_llm_calls"]
    assert summary["budget_summary"]["spent_rmb"] == summary["llm_cost_rmb"]


def test_mock_agent_persists_ledger(tmp_path):
    summary = mock_agent.run(workdir=str(tmp_path), budget_rmb="20")
    task_id = summary["task_id"]
    # re-open the persisted ledger and confirm the cost survives a restart
    import os

    from partc.budget_manager import BudgetManager

    ledger = os.path.join(str(tmp_path), ".partc_state", "demo", "state", "ledger.json")
    bm = BudgetManager(ledger, "20")
    assert bm.has_task(task_id)
    assert bm.summary(task_id)["spent_rmb"] == summary["llm_cost_rmb"]