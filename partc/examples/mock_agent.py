"""C5 — Mock Agent integration demo (fully offline).

Simulates the loop a real SuperAgent would drive, using only the Part C
toolkit with a deterministic mock LLM backend and the real code executor —
no network access, no API key, no cost.

The demo:
1. sends LLM requests (mock) and records token usage and cost;
2. executes a controlled Python snippet through the code executor;
3. runs a follow-up "reflection" LLM call;
4. demonstrates that budget-exceeded calls are rejected before any request;
5. prints a task-level cost & execution summary.
"""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partc.config import PartCConfig  # noqa: E402
from partc.budget_manager import BudgetManager  # noqa: E402
from partc.llm_client import MockBackend, ZhizengzengLLMClient  # noqa: E402
from partc.logger import StructuredLogger  # noqa: E402
from partc.models import ModelRegistry  # noqa: E402
from partc.code_executor import PythonExecutor  # noqa: E402
from partc.schemas import ChatMessage  # noqa: E402


def run(
    *,
    workdir: Optional[str] = None,
    model: str = "deepseek-v4.1-flash-1017",
    budget_rmb: str = "20",
    log_dir: Optional[str] = None,
    state_dir: Optional[str] = None,
) -> dict:
    workdir = workdir or os.getcwd()
    log_dir = log_dir or os.path.join(workdir, ".partc_state", "demo", "logs")
    state_dir = state_dir or os.path.join(workdir, ".partc_state", "demo", "state")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    cfg = PartCConfig(
        model=model,
        budget_rmb=budget_rmb,
        log_dir=log_dir,
        state_dir=state_dir,
        timeout_seconds=30.0,
        max_retries=2,
        retry_backoff_base_seconds=0.0,
    )
    registry = ModelRegistry.load_default()
    budget = BudgetManager(os.path.join(state_dir, "ledger.json"), default_budget_rmb=budget_rmb)
    logger = StructuredLogger(log_dir)

    mock = MockBackend(
        content='PLAN: (1) load data (2) train a small classifier (3) evaluate. ACTION: run experiments.',
        completion_tokens=24,
    )
    client = ZhizengzengLLMClient(cfg, budget_manager=budget, logger=logger, model_registry=registry, backend=mock)

    task_id = "mock_task_001"
    run_id = "mock_run_001"

    # 1. planning call
    plan = client.chat(
        [ChatMessage.system("You are an AutoML agent."), ChatMessage.user("Plan the next steps.")],
        task_id=task_id, run_id=run_id, max_tokens=64,
    )

    # 2. execute a controlled simple snippet (the part D would run a real training script)
    executor = PythonExecutor(timeout_seconds=10, cwd=workdir, output_dir=os.path.join(workdir, ".partc_state", "demo", "out"))
    code = (
        "scores = {'baseline': 0.81, 'candidate': 0.83}\n"
        "for k, v in scores.items():\n"
        "    print(f'{k}: {v:.2f}')\n"
    )
    exec_result = executor.execute_code(code, task_id=task_id, run_id=run_id)
    logger.log_execution(exec_result)

    # 3. reflection call, informed (in outline) by the execution result
    reflection = client.chat(
        [ChatMessage.user(f"Execution status: {exec_result.status}. Choose next action.")],
        task_id=task_id, run_id=run_id, max_tokens=64,
    )

    # 4. demonstrate budget-exceeded rejection on a separate tiny-budget task
    tiny_task = "mock_task_tiny_budget"
    budget.ensure_task(tiny_task, budget_rmb="0.00000001")
    rejected = client.chat(
        [ChatMessage.user("this must not be sent")],
        task_id=tiny_task, run_id=run_id, max_tokens=64,
    )
    budget_exceeded_demo = rejected.request_status == "budget_exceeded"

    llm_summary = logger.llm_summary(task_id)
    exec_summary = logger.execution_summary(task_id)
    budget_summary = budget.summary(task_id)

    summary = {
        "task_id": task_id,
        "n_llm_calls": llm_summary["n_llm_calls"],
        "llm_cost_rmb": budget_summary["spent_rmb"],
        "budget_rmb": budget_summary["budget_rmb"],
        "available_rmb": budget_summary["available_rmb"],
        "execution_status": exec_result.status,
        "execution_exit_code": exec_result.exit_code,
        "budget_exceeded_demo_triggered": budget_exceeded_demo,
        "plan_status": plan.request_status,
        "reflection_status": reflection.request_status,
        "llm_summary": llm_summary,
        "exec_summary": exec_summary,
        "budget_summary": budget_summary,
    }
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())