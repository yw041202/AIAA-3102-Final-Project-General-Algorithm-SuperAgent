"""Optional adapter using C's unmodified public executor and logger APIs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import identifier, read_json, sha256, write_json


def execute_via_c(arguments, *, partc_root=None, workdir, task_id, run_id, timeout_seconds=120, python=None):
    identifier(task_id, "task_id")
    identifier(run_id, "run_id")
    if arguments and arguments[0] == "train" and "--request" in arguments:
        request = read_json(arguments[arguments.index("--request") + 1])
        if request.get("task_id") != task_id or request.get("run_id") != run_id:
            raise ValueError("C task_id/run_id must match the D training request")
    if partc_root is not None:
        source = Path(partc_root).resolve() / "src"
        if not (source / "partc" / "__init__.py").is_file():
            raise ValueError("partc_root must point to the directory containing Part C's src/ folder")
        sys.path.insert(0, str(source))
    from partc import PythonExecutor, StructuredLogger
    import partc.code_executor as executor_module

    if partc_root is not None and Path(executor_module.__file__).resolve().parent.parent != source:
        raise ImportError("another Part C package is already imported; start a new process with the requested partc_root")

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(str(workdir / "logs"))
    executor = PythonExecutor(python=python or sys.executable, timeout_seconds=timeout_seconds,
                              cwd=str(workdir), output_dir=str(workdir / "stdout_stderr"), sanitize_env=True)
    # Arguments must contain absolute request/artifact/data/output paths: C changes cwd.
    result = executor.execute_file(str(Path(__file__).with_name("runner.py").resolve()),
                                   args=list(arguments), task_id=task_id, run_id=run_id)
    logger.log_execution(result)
    payload = None
    for line in reversed((result.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("event") == "partd.result":
            payload = candidate
            break
    combined = {"execution": result.to_dict(), "partd_result": payload,
                "status": "success" if result.status == "success" and payload and payload.get("status") == "success" else "error",
                "execution_summary": logger.execution_summary(task_id),
                "executor_source": str(Path(executor_module.__file__).resolve()),
                "executor_sha256": sha256(executor_module.__file__), "python": python or sys.executable,
                "execution_log_path": str(workdir / "logs" / "execution.jsonl")}
    write_json(workdir / f"{result.execution_id}.json", combined)
    return combined
