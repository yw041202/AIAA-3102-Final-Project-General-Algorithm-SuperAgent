"""Simulated A/C/D integration acceptance demo (fully offline).

This is **not** a real SuperAgent and performs **no** autonomous experiment
planning.  It only verifies that the Part C public interface is sufficient for
agent A (planning) and member D (training) to drive a tool-call loop through C.

Flow simulated here:

1. ``MockPlanner`` (stands in for A) produces one fixed experiment request.
2. C's ``PythonExecutor`` runs a controlled, self-contained training script
   (a tiny nearest-centroid classifier on synthetic 2-D data — no sklearn,
   no numpy, no network), printing one JSON metric line to stdout.
3. ``ExecResult`` returns structured status / exit code / stdout / log paths.
4. ``MockPlanner`` reads the structured result and issues a follow-up LLM call
   through the formal C client.
5. C reports the task-level cost and execution-log summary.

Everything runs offline via ``MockBackend``; no API key, no billable call.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partc.config import PartCConfig  # noqa: E402
from partc.budget_manager import BudgetManager  # noqa: E402
from partc.code_executor import PythonExecutor  # noqa: E402
from partc.llm_client import MockBackend, ZhizengzengLLMClient  # noqa: E402
from partc.logger import StructuredLogger  # noqa: E402
from partc.models import ModelRegistry  # noqa: E402
from partc.schemas import ChatMessage, ExecResult  # noqa: E402
from partc.errors import ModelComplianceError  # noqa: E402

# A pure-stdlib "training script", the kind D would provide.  Trains a
# nearest-centroid classifier on two synthetic Gaussian-ish clusters and emits
# a single JSON metrics line — the contract C/D rely on for feedback.
TRAIN_SCRIPT = """
import json, math, random
random.seed(0)

def gen(n, cx, cy):
    return [(cx + random.uniform(-1, 1), cy + random.uniform(-1, 1)) for _ in range(n)]

train = [(p, 0) for p in gen(50, 0.0, 0.0)] + [(p, 1) for p in gen(50, 3.0, 3.0)]
test  = [(p, 0) for p in gen(30, 0.0, 0.0)] + [(p, 1) for p in gen(30, 3.0, 3.0)]

def centroid(rows):
    n = len(rows)
    return (sum(p[0] for p in rows) / n, sum(p[1] for p in rows) / n)

c0 = centroid([p for p, l in train if l == 0])
c1 = centroid([p for p, l in train if l == 1])

def d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

correct = sum(1 for p, l in test if (d(p, c0) < d(p, c1)) == (l == 0))
print(json.dumps({"accuracy": round(correct / len(test), 4),
                  "model_type": "nearest_centroid",
                  "n_train": len(train), "n_test": len(test)}))
"""


class MockPlanner:
    """Stand-in for member A's planner: emits one request, reads the result."""

    def __init__(self, model: str = "deepseek-v4.1-flash-1017") -> None:
        self.model = model
        self.requests: list[Dict[str, Any]] = []

    def next_experiment(self) -> Dict[str, Any]:
        req = {
            "experiment_id": "exp_001",
            "objective": "classify two synthetic clusters; report accuracy",
            "train_script": TRAIN_SCRIPT,
            "model": self.model,
        }
        self.requests.append(req)
        return req

    @staticmethod
    def read_metric(exec_result: ExecResult) -> Optional[float]:
        """Parse the JSON metric line from the executor's structured stdout."""
        if exec_result.status != "success" or not exec_result.stdout:
            return None
        for line in exec_result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return float(json.loads(line)["accuracy"])
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        return None


def run(
    *,
    workdir: Optional[str] = None,
    model: str = "deepseek-v4.1-flash-1017",
    budget_rmb: str = "20",
    log_dir: Optional[str] = None,
    state_dir: Optional[str] = None,
) -> Dict[str, Any]:
    workdir = workdir or os.getcwd()
    log_dir = log_dir or os.path.join(workdir, ".partc_state", "integration", "logs")
    state_dir = state_dir or os.path.join(workdir, ".partc_state", "integration", "state")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    cfg = PartCConfig(
        model=model, budget_rmb=budget_rmb, log_dir=log_dir, state_dir=state_dir,
        timeout_seconds=30.0, max_retries=2, retry_backoff_base_seconds=0.0,
    )
    registry = ModelRegistry.load_default()
    budget = BudgetManager(os.path.join(state_dir, "ledger.json"), default_budget_rmb=budget_rmb)
    logger = StructuredLogger(log_dir)
    client = ZhizengzengLLMClient(
        cfg, budget_manager=budget, logger=logger, model_registry=registry,
        backend=MockBackend(content="NEXT ACTION: keep best model; report metrics.", completion_tokens=12),
    )

    task_id = "integration_task_001"
    run_id = "integration_run_001"
    planner = MockPlanner(model=model)

    # 1. planner (A) provides an experiment request
    req = planner.next_experiment()

    # 2. C executes the training script (D's controlled entry point)
    executor = PythonExecutor(
        timeout_seconds=15, cwd=workdir,
        output_dir=os.path.join(workdir, ".partc_state", "integration", "out"),
        sanitize_env=True,
    )
    exec_result = executor.execute_code(req["train_script"], task_id=task_id, run_id=run_id)
    logger.log_execution(exec_result)

    # 3. planner reads the structured result
    accuracy = planner.read_metric(exec_result)

    # 4. planner issues a follow-up LLM call through the formal C client
    llm_result = client.chat(
        [
            ChatMessage.system("You plan AutoML experiments from structured results."),
            ChatMessage.user(
                f"Experiment {req['experiment_id']} finished: status={exec_result.status}, "
                f"accuracy={accuracy if accuracy is not None else 'n/a'}."
            ),
        ],
        model=req["model"], task_id=task_id, run_id=run_id, max_tokens=64,
    )

    # 5. task-level cost + execution summary (what A reads to decide next)
    summary = {
        "task_id": task_id,
        "experiment_id": req["experiment_id"],
        "execution_status": exec_result.status,
        "exit_code": exec_result.exit_code,
        "accuracy": accuracy,
        "stdout_path": exec_result.stdout_path,
        "stderr_path": exec_result.stderr_path,
        "llm_request_status": llm_result.request_status,
        "llm_output": llm_result.output_text,
        "llm_cost_rmb": llm_result.calculated_cost_rmb,
        "budget_summary": budget.summary(task_id),
        "llm_summary": logger.llm_summary(task_id),
        "exec_summary": logger.execution_summary(task_id),
    }
    return summary


def _count_exceeded_demo(client: ZhizengzengLLMClient, task_id: str) -> bool:
    # verify a non-compliant model is rejected through the same public path
    try:
        client.chat([ChatMessage.user("x")], model="gpt-4-turbo", task_id=task_id)
        return False
    except ModelComplianceError:
        return True


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())