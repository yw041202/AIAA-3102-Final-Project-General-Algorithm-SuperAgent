"""LLM client tests — all offline via MockBackend."""

from __future__ import annotations

from decimal import Decimal

import pytest

from partc.budget_manager import BudgetManager
from partc.config import PartCConfig
from partc.errors import (
    BackendAuthError,
    BackendBadRequestError,
    BackendTimeoutError,
    ModelComplianceError,
)
from partc.llm_client import (
    MockBackend,
    ZhizengzengLLMClient,
    parse_usage,
)
from partc.logger import StructuredLogger
from partc.models import ModelRegistry
from partc.schemas import ChatMessage

MSGS = [ChatMessage.system("you are helpful"), ChatMessage.user("hello")]


def test_mock_normal_response(client):
    result = client.chat(MSGS, task_id="t1", run_id="r1")
    assert result.ok is True
    assert result.request_status == "success"
    assert result.output_text == "mock-ok"
    assert result.usage.completion_tokens == 20
    assert result.calculated_cost_rmb is not None
    assert Decimal(result.calculated_cost_rmb) > 0
    assert result.cumulative_cost_rmb == result.calculated_cost_rmb
    assert result.requested_model_id == "deepseek-v4.1-flash-1017"
    assert result.resolved_model_version == "DeepSeek-V4.1-Flash-1017"


def test_noncompliant_model_raises(client):
    with pytest.raises(ModelComplianceError):
        client.chat(MSGS, model="gpt-4-turbo", task_id="t1")


def test_api_timeout_is_retried_and_marked_uncertain(make_client, cfg, budget, logger):
    calls = {"n": 0}

    def script(model, messages, max_tokens):
        calls["n"] += 1
        if calls["n"] < 3:
            raise BackendTimeoutError("slow")
        return {"id": "x", "model": model, "content": "ok", "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    client = make_client(backend=MockBackend(script=script))
    result = client.chat(MSGS, task_id="t1")
    assert result.request_status == "success"
    assert result.retry_count == 2  # two retries before success


def test_timeout_all_retries_exhausted_holds_reservation(make_client, cfg, budget, logger):
    client = make_client(backend=MockBackend(script=BackendTimeoutError("slow")))
    result = client.chat(MSGS, task_id="t1")
    assert result.request_status == "error"
    assert result.error_type == "api_timeout"
    # reservation must be held (fail-closed), not released
    assert budget.available("t1") < Decimal("20")


def test_auth_error_releases_reservation(make_client, cfg, budget, logger):
    client = make_client(backend=MockBackend(script=BackendAuthError("bad key", status_code=401)))
    result = client.chat(MSGS, task_id="t1")
    assert result.request_status == "error"
    assert result.error_type == "auth_error"
    # not retryable, reservation released fully
    assert budget.available("t1") == Decimal("20")


def test_bad_request_releases_and_not_retried(make_client, budget, logger):
    client = make_client(backend=MockBackend(script=BackendBadRequestError("bad", status_code=400)))
    result = client.chat(MSGS, task_id="t1")
    assert result.request_status == "error"
    assert result.error_type == "bad_request"
    assert result.retry_count == 0
    assert budget.available("t1") == Decimal("20")


def test_unknown_usage_is_uncertain(make_client, budget, logger):
    client = make_client(backend=MockBackend(script=lambda m, msgs, mt: {
        "id": "x", "model": m, "content": "ok", "finish_reason": "stop", "usage": {}}))
    result = client.chat(MSGS, task_id="t1")
    assert result.request_status == "success_uncertain"
    assert result.error_type == "usage_unknown"
    assert result.calculated_cost_rmb is None
    assert budget.available("t1") < Decimal("20")  # reservation still held


def test_budget_exceeded_rejects_without_call(tmp_path):
    # tiny budget so even one call cannot fit
    bm = BudgetManager(str(tmp_path / "ledger.json"), "0.00000001")
    log = StructuredLogger(str(tmp_path / "logs"))
    cfg = PartCConfig(model="deepseek-v4.1-flash-1017", budget_rmb="20",
                      state_dir=str(tmp_path / "state"), log_dir=str(tmp_path / "logs"),
                      retry_backoff_base_seconds=0.0)
    backend = MockBackend()
    client = ZhizengzengLLMClient(cfg, budget_manager=bm, logger=log, backend=backend)
    result = client.chat(MSGS, task_id="t1")
    assert result.request_status == "budget_exceeded"
    assert result.error_type == "budget_exceeded"
    assert backend.calls == []  # no network call was made


def test_parse_usage_no_double_count_reasoning():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 40},
        "prompt_tokens_details": {"cached_tokens": 60},
    }
    tu = parse_usage(usage)
    assert tu.prompt_tokens == 100
    assert tu.completion_tokens == 50  # reasoning already included, not added
    assert tu.extra["reasoning_tokens"] == 40
    assert tu.extra["cached_tokens"] == 60


def test_parse_usage_missing_is_incomplete():
    assert parse_usage({}).is_complete is False
    assert parse_usage({"total_tokens": 10}).is_complete is False
    assert parse_usage({"prompt_tokens": 1, "completion_tokens": 2}).is_complete is True


def test_alias_sends_canonical_model_id(client):
    res = client.chat(MSGS, model="deepseek-flash", task_id="t1")
    assert res.requested_model_id == "deepseek-flash"  # what the caller asked for
    assert res.resolved_model_version == "DeepSeek-V4.1-Flash-1017"
    sent = client.backend.calls[-1]["model"]  # what went on the wire
    assert sent == "deepseek-v4.1-flash-1017"


def test_context_exceeding_tiers_fails_closed(make_client):
    from partc.models import ModelRegistry

    registry = ModelRegistry.from_dict({
        "compliance": {"providers_allowed": ["deepseek", "qwen"], "release_date_cutoff": "2026-10-21"},
        "models": [{
            "provider": "qwen",
            "requested_model_id": "qwen-tiny-tier",
            "underlying_model_version": "Tiny",
            "official_release_date": "2026-01-01",
            "release_date_confirmed": True,
            "pricing": {
                "currency": "CNY", "unit": "per_1m_tokens",
                "input_per_1m_tokens": "1.0", "output_per_1m_tokens": "1.0",
                "billing_mode": "tier",
                "tiers": [{"label": "<=100", "max_input_tokens": 100,
                           "input_per_1m_tokens": "1.0", "output_per_1m_tokens": "1.0"}],
            },
            "verified": True, "allowed": True,
        }],
    })
    client = make_client(mock=True, registry=registry)
    with pytest.raises(ModelComplianceError):
        client.chat([ChatMessage.user("x" * 1000)], model="qwen-tiny-tier", task_id="t1")