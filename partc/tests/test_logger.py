"""Structured logger tests."""

from __future__ import annotations

import os

from partc.logger import StructuredLogger
from partc.schemas import ChatMessage, ExecResult, LLMResult, TokenUsage


def _llm(tmp_path, task_id="t1", request_status="success", **kw):
    r = LLMResult(task_id=task_id, run_id="r1", request_status=request_status, requested_model_id="m", resolved_model_version="v", output_text="hi", usage=TokenUsage(prompt_tokens=100, completion_tokens=50), **kw)
    return r


def test_log_llm_writes_required_fields(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"), api_key="sk-secret")
    log.log_llm(_llm(tmp_path))
    recs = log.llm_records("t1")
    assert len(recs) == 1
    rec = recs[0]
    for field in (
        "task_id", "run_id", "call_id", "timestamp", "provider",
        "requested_model_id", "resolved_model_version", "request_status",
        "prompt_tokens", "completion_tokens", "calculated_cost",
        "cumulative_cost", "pricing_version", "error_type", "retry_count",
    ):
        assert field in rec


def test_log_llm_does_not_leak_api_key(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"), api_key="sk-SUPERSECRET-1234")
    log.log_llm(_llm(tmp_path, error_message="tried sk-SUPERSECRET-1234 as key"))
    rec = log.llm_records("t1")[0]
    blob = __import__("json").dumps(rec)
    assert "SUPERSECRET" not in blob


def test_log_execution_writes(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"))
    log.log_execution(ExecResult(task_id="t1", run_id="r1", status="success", exit_code=0, stdout="ok", stderr=""))
    recs = log.execution_records("t1")
    assert len(recs) == 1
    assert recs[0]["status"] == "success"
    assert recs[0]["exit_code"] == 0


def test_llm_summary_counts(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"))
    log.log_llm(_llm(tmp_path))
    log.log_llm(_llm(tmp_path, request_status="error", error_type="api_timeout"))
    log.log_llm(_llm(tmp_path, request_status="budget_exceeded", error_type="budget_exceeded"))
    s = log.llm_summary("t1")
    assert s["n_llm_calls"] == 3
    assert s["n_success"] == 1
    assert s["n_error"] == 1
    assert s["n_budget_exceeded"] == 1
    assert s["prompt_tokens"] == 300
    assert s["completion_tokens"] == 150


def test_execution_summary_counts(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"))
    log.log_execution(ExecResult(task_id="t1", status="success"))
    log.log_execution(ExecResult(task_id="t1", status="runtime_error"))
    log.log_execution(ExecResult(task_id="t1", status="timeout"))
    s = log.execution_summary("t1")
    assert s["n_executions"] == 3
    assert s["n_success"] == 1
    assert s["n_runtime_error"] == 1
    assert s["n_timeout"] == 1


def test_filter_by_task(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"))
    log.log_llm(_llm(tmp_path, task_id="a"))
    log.log_llm(_llm(tmp_path, task_id="b"))
    assert len(log.llm_records("a")) == 1
    assert len(log.llm_records("b")) == 1
    assert len(log.llm_records()) == 2


def test_redact_recursive(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"), api_key="secret123")
    dirty = {"api_key": "secret123", "nested": {"token": "secret123"}, "list": ["secret123 ok"], "keep": "value"}
    clean = log.redact(dirty, secrets=["secret123"])
    assert clean["api_key"] == "<redacted>"
    assert clean["nested"]["token"] == "<redacted>"
    assert "secret123" not in clean["list"][0]
    assert clean["keep"] == "value"


def test_log_is_jsonl(tmp_path):
    log = StructuredLogger(str(tmp_path / "logs"), api_key="k")
    log.log_llm(_llm(tmp_path))
    log.log_llm(_llm(tmp_path))
    path = os.path.join(str(tmp_path / "logs"), "llm.jsonl")
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        __import__("json").loads(line)