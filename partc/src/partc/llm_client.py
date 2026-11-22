"""C1 — Zhizengzeng (智增增) LLM client.

A thin, structured client over the Zhizengzeng OpenAI-compatible Chat
Completions endpoint.  The upper agent layer (A) never sees the underlying
SDK: it passes messages and receives a :class:`~partc.schemas.LLMResult`.

Responsibilities handled here, transparently for the caller:

* model compliance validation (Qwen/DeepSeek, released <= 2026-10-21, official
  pricing) — non-compliant models raise :class:`ModelComplianceError`;
* budget pre-check + reservation, actual-cost settlement, and fail-closed
  handling of unknown usage;
* timeout + bounded retries with backoff;
* structured, sanitised JSONL logging.

Live vs mock
------------
The real backend is used only when constructed without a mock; it requires
``ZZZ_API_KEY`` and performs network calls.  Mock / offline tests inject a
:class:`MockBackend` and never touch the network or import the ``openai`` SDK.
"""

from __future__ import annotations

import os
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Union

from .config import PartCConfig, normalize_base_url
from .errors import (
    BackendAuthError,
    BackendBadRequestError,
    BackendError,
    BackendNetworkError,
    BackendRateLimitError,
    BackendServerError,
    BackendTimeoutError,
    BudgetExceededError,
    ConfigurationError,
    ModelComplianceError,
)
from .logger import StructuredLogger
from .models import ModelRegistry
from .budget_manager import BudgetManager, CostCalculator
from .schemas import ChatMessage, LLMResult, TokenUsage, new_id

Message = Union[ChatMessage, Dict[str, str]]

_DEFAULT_TASK_ID = "default_task"


def estimate_input_tokens(messages: List[Message]) -> int:
    """Conservative UTF-8 byte estimate plus message framing for text requests.

    This is not a provider tokenizer; actual usage still controls settlement.
    Byte counting avoids the old characters/4 underestimate for CJK text.
    """
    total = 256
    for m in messages:
        if isinstance(m, ChatMessage):
            content = m.content
        elif isinstance(m, dict):
            content = m.get("content", "")
        else:
            content = str(m)
        total += max(1, len(str(content).encode("utf-8"))) + 64
    return max(total, 1)


class SDKBackend:
    """Real Zhizengzeng backend using the ``openai`` SDK (lazily imported)."""

    def __init__(self, config: PartCConfig) -> None:
        if not config.has_api_key():
            raise ConfigurationError("ZZZ_API_KEY is required for live (non-mock) LLM calls")
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("the 'openai' package is required for live LLM calls") from exc
        self._openai = openai
        self._disable_thinking = config.extra.get("PARTC_DISABLE_THINKING", "false").lower() == "true"
        transport = {}
        if config.extra.get("PARTC_TRUST_ENV", "true").lower() == "false":
            import httpx
            transport["http_client"] = httpx.Client(trust_env=False)
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=normalize_base_url(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=0,  # retries are handled by the client, not the SDK
            **transport,
        )

    def complete(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: Optional[float], timeout: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self._disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except self._openai.APITimeoutError as exc:
            raise BackendTimeoutError(str(exc)) from exc
        except self._openai.APIConnectionError as exc:
            raise BackendNetworkError(str(exc)) from exc
        except self._openai.RateLimitError as exc:
            raise BackendRateLimitError(str(exc)) from exc
        except self._openai.AuthenticationError as exc:
            raise BackendAuthError(str(exc), status_code=getattr(exc, "status_code", 401)) from exc
        except self._openai.BadRequestError as exc:
            raise BackendBadRequestError(str(exc), status_code=getattr(exc, "status_code", 400)) from exc
        except self._openai.APIStatusError as exc:
            code = getattr(exc, "status_code", 0)
            if code >= 500:
                raise BackendServerError(str(exc), status_code=code) from exc
            if code == 429:
                raise BackendRateLimitError(str(exc), status_code=code) from exc
            raise BackendError(str(exc), status_code=code) from exc
        except Exception as exc:
            raise BackendNetworkError(str(exc)) from exc

        return _parse_openai_response(resp)


def _parse_openai_response(resp: Any) -> Dict[str, Any]:
    try:
        choice = resp.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
    except (AttributeError, IndexError) as exc:
        raise BackendServerError(f"malformed API response: {exc}") from exc
    usage = getattr(resp, "usage", None)
    return {
        "id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "content": content,
        "finish_reason": finish_reason,
        "usage": _usage_to_dict(usage),
    }


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    d: Dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, field, None)
        if v is not None:
            d[field] = int(v)
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        d["prompt_tokens_details"] = {
            "cached_tokens": getattr(ptd, "cached_tokens", None),
        }
    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None:
        d["completion_tokens_details"] = {
            "reasoning_tokens": getattr(ctd, "reasoning_tokens", None),
        }
    return d


class MockBackend:
    """Deterministic offline backend for tests and demos.

    Returns a fixed response with explicit token usage.  By default it echoes a
    short reply; ``script`` (a callable, or a fixed exception) can be supplied
    to simulate errors, timeouts or custom sequences.
    """

    def __init__(
        self,
        *,
        content: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: int = 20,
        script: Optional[Any] = None,
    ) -> None:
        self.content = content or "mock-ok"
        self.completion_tokens = completion_tokens
        self._script = script
        self.calls: List[Dict[str, Any]] = []
        self.prompt_tokens = prompt_tokens

    def complete(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: Optional[float], timeout: float) -> Dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        if callable(self._script):
            result = self._script(model, messages, max_tokens)
            if isinstance(result, Exception):
                raise result
            if result is not None:
                return result
        elif isinstance(self._script, Exception):
            raise self._script
        # deterministic prompt token count (derived from input) unless overridden
        pt = self.prompt_tokens if self.prompt_tokens is not None else estimate_input_tokens([ChatMessage(m["role"], m["content"]) for m in messages])
        ct = min(self.completion_tokens, max_tokens if max_tokens else self.completion_tokens)
        return {
            "id": new_id("chatcmpl"),
            "model": model,
            "content": self.content,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }


def parse_usage(raw_usage: Optional[Dict[str, Any]]) -> TokenUsage:
    """Parse an OpenAI-compatible ``usage`` object without double-counting.

    Reasoning tokens (``completion_tokens_details.reasoning_tokens``) are
    assumed to already be included in ``completion_tokens`` (DeepSeek-style),
    so they are recorded in ``extra`` only.  If the usage is missing or lacks
    prompt/completion counts, the returned :class:`TokenUsage` is incomplete and
    the client will treat the call as cost-uncertain rather than zero.
    """
    if not raw_usage or not isinstance(raw_usage, dict):
        return TokenUsage()
    extra: Dict[str, Any] = {}
    ptd = raw_usage.get("prompt_tokens_details")
    if isinstance(ptd, dict) and ptd.get("cached_tokens") is not None:
        extra["cached_tokens"] = ptd.get("cached_tokens")
    ctd = raw_usage.get("completion_tokens_details")
    if isinstance(ctd, dict) and ctd.get("reasoning_tokens") is not None:
        extra["reasoning_tokens"] = ctd.get("reasoning_tokens")
    return TokenUsage(
        prompt_tokens=_to_int(raw_usage.get("prompt_tokens")),
        completion_tokens=_to_int(raw_usage.get("completion_tokens")),
        total_tokens=_to_int(raw_usage.get("total_tokens")),
        extra=extra,
    )


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ZhizengzengLLMClient:
    def __init__(
        self,
        config: PartCConfig,
        *,
        budget_manager: Optional[BudgetManager] = None,
        logger: Optional[StructuredLogger] = None,
        model_registry: Optional[ModelRegistry] = None,
        backend: Optional[Any] = None,
        mock: bool = False,
    ) -> None:
        self.config = config
        self.model_registry = model_registry or ModelRegistry.load_default()
        self.budget_manager = budget_manager or BudgetManager(
            os.path.join(config.state_dir, "ledger.json"),
            default_budget_rmb=config.budget_rmb,
        )
        self.logger = logger or StructuredLogger(config.log_dir, api_key=config.api_key)
        if backend is not None:
            self.backend = backend
        elif mock:
            self.backend = MockBackend()
        else:
            self.backend = SDKBackend(config)

    # -- main entry point -----------------------------------------------------
    def chat(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        task_id: Optional[str] = None,
        run_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> LLMResult:
        task_id = task_id or _DEFAULT_TASK_ID
        run_id = run_id or new_id("run")
        call_id = new_id("call")

        # 1. compliance (raises ModelComplianceError for non-compliant ids)
        model_id = model or self.config.model
        info = self.model_registry.require_compliant(model_id)
        wire_model_id = info.requested_model_id  # canonical id actually sent to the gateway

        # 2. reserve upper-bound cost (context-tier aware)
        openai_messages = [m.to_openai() if isinstance(m, ChatMessage) else dict(m) for m in messages]
        prompt_estimate = estimate_input_tokens(messages)
        p_in, p_out, tier_label = info.resolve_pricing(prompt_estimate)
        if p_in is None or p_out is None:
            raise ModelComplianceError(
                f"cannot price {wire_model_id!r}: ~{prompt_estimate} input tokens exceed all recorded context tiers",
                model_id=model_id, reasons=[tier_label],
            )
        calculator = CostCalculator(p_in, p_out)
        reserve_rmb = calculator.estimate_upper_bound(prompt_estimate, max_tokens)
        try:
            self.budget_manager.reserve(task_id, call_id, wire_model_id, tier_label, reserve_rmb)
        except BudgetExceededError as exc:
            result = self._result(
                task_id=task_id,
                run_id=run_id,
                call_id=call_id,
                model_id=model_id,
                info=info,
                pricing_version=tier_label,
                status="budget_exceeded",
                error_type="budget_exceeded",
                error_message=str(exc),
            )
            self.logger.log_llm(result)
            return result

        # 3. call with bounded retries
        max_attempts = 1 + (self.config.max_retries if retries is None else retries)
        backoff = self.config.retry_backoff_base_seconds
        timeout_value = timeout if timeout is not None else self.config.timeout_seconds

        raw: Optional[Dict[str, Any]] = None
        last_error: Optional[BackendError] = None
        retry_count = 0
        for attempt in range(max_attempts):
            try:
                raw = self.backend.complete(wire_model_id, openai_messages, max_tokens, temperature, timeout_value)
                break
            except BackendError as exc:
                last_error = exc
                if not exc.retryable or attempt == max_attempts - 1:
                    break
                retry_count += 1
                if backoff > 0:
                    time.sleep(backoff * (2 ** (attempt - 1)) if attempt >= 1 else backoff)

        if raw is None:
            return self._failure_result(
                task_id=task_id, run_id=run_id, call_id=call_id,
                model_id=model_id, info=info, error=last_error, retry_count=retry_count,
            )

        # 4. parse output + usage
        output_text = raw.get("content")
        usage = parse_usage(raw.get("usage"))
        elapsed = None  # elapsed handled by caller-level timing if needed; recorded from raw if present

        result = self._result(
            task_id=task_id, run_id=run_id, call_id=call_id,
            model_id=model_id, info=info, output_text=output_text, usage=usage,
            retry_count=retry_count,
            extra={"finish_reason": raw.get("finish_reason"), "provider_model": raw.get("model"), "provider_response_id": raw.get("id")},
        )

        # 5. settle or mark uncertain (context-tier aware)
        if usage.is_complete:
            assert usage.prompt_tokens is not None and usage.completion_tokens is not None
            p_in_s, p_out_s, settle_tier = info.resolve_pricing(usage.prompt_tokens)
            if p_in_s is None or p_out_s is None:
                self.budget_manager.mark_uncertain(
                    task_id, call_id,
                    input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
                    error_type="pricing_tier_unresolved",
                )
                result.request_status = "success_uncertain"
                result.error_type = "pricing_tier_unresolved"
                result.error_message = "actual input tokens exceed recorded context tiers; cost held as uncertain"
                result.pricing_version = settle_tier
            else:
                cost = CostCalculator(p_in_s, p_out_s).cost(usage.prompt_tokens, usage.completion_tokens)
                self.budget_manager.settle(task_id, call_id, usage.prompt_tokens, usage.completion_tokens, cost)
                result.request_status = "success"
                result.calculated_cost_rmb = str(cost)
                result.pricing_version = settle_tier
        else:
            # fail-closed: do not guess zero; keep the reservation held
            self.budget_manager.mark_uncertain(
                task_id, call_id,
                input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
                error_type="usage_unknown",
            )
            result.request_status = "success_uncertain"
            result.error_type = "usage_unknown"
            result.error_message = "response missing reliable token usage; cost held as uncertain"

        result.cumulative_cost_rmb = str(self.budget_manager.cumulative_cost(task_id))
        self.logger.log_llm(result)
        return result

    # -- helpers ---------------------------------------------------------------
    def _result(
        self,
        *,
        task_id: str,
        run_id: str,
        call_id: str,
        model_id: str,
        info,
        status: str = "success",
        output_text: Optional[str] = None,
        usage: Optional[TokenUsage] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        pricing_version: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        return LLMResult(
            task_id=task_id,
            run_id=run_id,
            call_id=call_id,
            provider=("zhizengzeng" if urlparse(self.config.base_url).hostname == "api.zhizengzeng.com"
                      else urlparse(self.config.base_url).hostname or "openai-compatible"),
            requested_model_id=model_id,
            resolved_model_version=info.underlying_model_version,
            request_status=status,
            output_text=output_text,
            usage=usage or TokenUsage(),
            currency=info.pricing.get("currency", "CNY"),
            pricing_version=pricing_version or info.pricing_version,
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count,
            extra=extra or {},
        )

    def _failure_result(
        self,
        *,
        task_id: str,
        run_id: str,
        call_id: str,
        model_id: str,
        info,
        error: Optional[BackendError],
        retry_count: int,
    ) -> LLMResult:
        error_type = error.error_type if error else "unknown"
        error_message = str(error) if error else "unknown error"
        # ambiguous failures (may have been charged before a lost response) hold
        # the reservation; deterministic pre-send failures release it.
        if error_type in ("api_timeout", "rate_limit", "network_error", "server_error"):
            self.budget_manager.mark_uncertain(task_id, call_id, error_type=error_type)
        else:
            self.budget_manager.release(task_id, call_id, error_type=error_type)

        result = self._result(
            task_id=task_id, run_id=run_id, call_id=call_id,
            model_id=model_id, info=info, status="error",
            error_type=error_type, error_message=error_message, retry_count=retry_count,
        )
        self.logger.log_llm(result)
        return result
