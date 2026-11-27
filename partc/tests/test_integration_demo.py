"""Simulated A/C/D integration acceptance test (offline)."""

from __future__ import annotations

from decimal import Decimal

import integration_demo


def test_integration_demo_flow(tmp_path):
    summary = integration_demo.run(workdir=str(tmp_path), model="deepseek-v4.1-flash-1017", budget_rmb="20")

    # C executed the training script successfully and returned structured status
    assert summary["execution_status"] == "success"
    assert summary["exit_code"] == 0

    # the trainer emitted a metric that the planner could parse
    assert summary["accuracy"] is not None
    assert summary["accuracy"] >= 0.9

    # execution log paths were produced and are referenced in the result
    for key in ("stdout_path", "stderr_path"):
        assert summary[key]

    # the planner's follow-up LLM call went through the formal client + budget
    assert summary["llm_request_status"] == "success"
    assert Decimal(summary["llm_cost_rmb"]) > 0
    assert summary["llm_output"]

    # task-level summaries are consistent
    assert summary["budget_summary"]["task_id"] == summary["task_id"]
    assert summary["exec_summary"]["n_executions"] == 1
    assert summary["llm_summary"]["n_llm_calls"] >= 1


def test_integration_demo_persists_and_recovers(tmp_path):
    summary = integration_demo.run(workdir=str(tmp_path), budget_rmb="20")

    import os

    from partc.budget_manager import BudgetManager

    ledger = os.path.join(str(tmp_path), ".partc_state", "integration", "state", "ledger.json")
    bm = BudgetManager(ledger, "20")
    assert bm.has_task(summary["task_id"])
    assert bm.summary(summary["task_id"])["spent_rmb"] == summary["budget_summary"]["spent_rmb"]