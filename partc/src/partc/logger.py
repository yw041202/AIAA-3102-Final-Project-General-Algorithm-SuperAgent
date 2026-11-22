"""C4 — Structured JSONL logging and error handling.

Every LLM call and code execution is appended as one JSON object per line
(JSONL), so logs are machine-readable by the upper agent layer and can be
replayed after a restart.

Secret safety: records are sanitised before being written — the API key value
and any key- / token-named fields are redacted, and errors are written fully
(failures and retries are never dropped).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from .schemas import ExecResult, LLMResult

_REDACT_MARKERS = ("secret", "password", "authorization", "apikey", "api_key", "credential", "bearer")
_SAFE_COUNT_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "input_tokens",
    "output_tokens",
}
_REDACTED = "<redacted>"


def _is_sensitive_key(name: str) -> bool:
    lower = str(name).lower()
    if lower in _SAFE_COUNT_FIELDS:
        return False
    if any(m in lower for m in _REDACT_MARKERS):
        return True
    if "token" in lower:
        return True
    return False


class StructuredLogger:
    """Append-only JSONL logger scoped to a log directory."""

    def __init__(self, log_dir: str, api_key: Optional[str] = None) -> None:
        self.log_dir = log_dir
        self._secrets: List[str] = [api_key] if api_key else []
        os.makedirs(self.log_dir, exist_ok=True)

    # -- paths ------------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.log_dir, name)

    # -- low-level ----------------------------------------------------------
    def _sanitize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self.redact(record, self._secrets)

    @staticmethod
    def redact(record: Any, secrets: Optional[Iterable[str]] = None) -> Any:
        """Return a copy with sensitive values removed (recursive)."""
        secret_values = {s for s in (secrets or ()) if s}
        if isinstance(record, dict):
            out: Dict[str, Any] = {}
            for k, v in record.items():
                if _is_sensitive_key(k):
                    out[k] = _REDACTED
                else:
                    out[k] = StructuredLogger.redact(v, secret_values)
            return out
        if isinstance(record, list):
            return [StructuredLogger.redact(v, secret_values) for v in record]
        if isinstance(record, str):
            for s in secret_values:
                record = record.replace(s, _REDACTED)
        return record

    def _append(self, name: str, record: Dict[str, Any]) -> str:
        path = self._path(name)
        line = json.dumps(self._sanitize(record), ensure_ascii=False, default=str)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return path

    # -- LLM ---------------------------------------------------------------
    def log_llm(self, result: LLMResult) -> str:
        record = {
            "event": "llm_call",
            "task_id": result.task_id,
            "run_id": result.run_id,
            "call_id": result.call_id,
            "timestamp": result.timestamp,
            "provider": result.provider,
            "requested_model_id": result.requested_model_id,
            "resolved_model_version": result.resolved_model_version,
            "request_status": result.request_status,
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "calculated_cost": result.calculated_cost_rmb,
            "cumulative_cost": result.cumulative_cost_rmb,
            "pricing_version": result.pricing_version,
            "error_type": result.error_type,
            "retry_count": result.retry_count,
            "error_message": result.error_message,
            "elapsed_seconds": result.elapsed_seconds,
            "extra": result.extra,
        }
        return self._append("llm.jsonl", record)

    def log_execution(self, result: ExecResult) -> str:
        record = {
            "event": "code_execution",
            "execution_id": result.execution_id,
            "task_id": result.task_id,
            "run_id": result.run_id,
            "script_ref": result.script_ref,
            "status": result.status,
            "exit_code": result.exit_code,
            "elapsed_seconds": result.elapsed_seconds,
            "stdout_path": result.stdout_path,
            "stderr_path": result.stderr_path,
            "error_type": result.error_type,
            "error_details": result.error_details,
            "timestamp": result.timestamp,
        }
        return self._append("execution.jsonl", record)

    def log_event(self, kind: str, record: Dict[str, Any]) -> str:
        """Generic event logger (e.g. budget events, agent decisions)."""
        event = dict(record)
        event["event"] = kind
        return self._append(f"{kind}.jsonl", event)

    # -- reading back --------------------------------------------------------
    def _read(self, name: str) -> List[Dict[str, Any]]:
        path = self._path(name)
        if not os.path.isfile(path):
            return []
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({"raw": line, "error": "unparseable log line"})
        return records

    def llm_records(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        recs = self._read("llm.jsonl")
        return [r for r in recs if task_id is None or r.get("task_id") == task_id]

    def execution_records(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        recs = self._read("execution.jsonl")
        return [r for r in recs if task_id is None or r.get("task_id") == task_id]

    def llm_summary(self, task_id: str) -> Dict[str, Any]:
        recs = self.llm_records(task_id)
        return {
            "task_id": task_id,
            "n_llm_calls": len(recs),
            "n_success": sum(1 for r in recs if r.get("request_status") in ("success", "success_uncertain")),
            "n_error": sum(1 for r in recs if r.get("request_status") == "error"),
            "n_budget_exceeded": sum(1 for r in recs if r.get("request_status") == "budget_exceeded"),
            "prompt_tokens": _sum_int(recs, "prompt_tokens"),
            "completion_tokens": _sum_int(recs, "completion_tokens"),
        }

    def execution_summary(self, task_id: str) -> Dict[str, Any]:
        recs = self.execution_records(task_id)
        return {
            "task_id": task_id,
            "n_executions": len(recs),
            "n_success": sum(1 for r in recs if r.get("status") == "success"),
            "n_runtime_error": sum(1 for r in recs if r.get("status") == "runtime_error"),
            "n_timeout": sum(1 for r in recs if r.get("status") == "timeout"),
            "n_execution_error": sum(1 for r in recs if r.get("status") == "execution_error"),
        }


def _sum_int(records: List[Dict[str, Any]], key: str) -> Optional[int]:
    total: Optional[int] = None
    for r in records:
        v = r.get(key)
        if isinstance(v, int):
            total = (total or 0) + v
    return total