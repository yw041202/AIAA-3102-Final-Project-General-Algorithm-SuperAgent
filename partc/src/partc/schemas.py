"""Public data structures returned by the Part C toolkit.

Everything that crosses the boundary between Part C and the rest of the
SuperAgent is a plain dataclass (or an explicitly documented dict), never a
raw SDK object.  All cost values are ``Decimal`` strings when serialised to
keep exact decimal arithmetic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Return a unique id like ``<prefix>_<uuid4-hex-short>``."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_openai(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls("assistant", content)


@dataclass
class TokenUsage:
    """Parsed token usage.

    ``prompt_tokens`` / ``completion_tokens`` are the authoritative billing
    counts.  Reasoning tokens (if reported separately) are assumed to already
    be included in ``completion_tokens`` and are NOT added again; they are
    recorded only in ``extra`` for auditability.
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# request_status values:
#   success         — call succeeded, output and usage parsed
#   success_uncertain — call succeeded but token usage is missing/unreliable
#   budget_exceeded — rejected before any network call
#   error           — call failed (see error_type)
_KNOWN_STATUSES = ("success", "success_uncertain", "budget_exceeded", "error")


@dataclass
class LLMResult:
    """Structured result of a single LLM call."""

    task_id: str
    run_id: str
    call_id: str = field(default_factory=lambda: new_id("call"))
    request_status: str = "success"
    provider: str = "zhizengzeng"
    requested_model_id: Optional[str] = None
    resolved_model_version: Optional[str] = None
    output_text: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    calculated_cost_rmb: Optional[str] = None  # Decimal string; None if unknown
    cumulative_cost_rmb: Optional[str] = None  # task cumulative after this call
    currency: str = "CNY"
    pricing_version: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    elapsed_seconds: Optional[float] = None
    timestamp: str = field(default_factory=_now_iso)
    # extra provider fields preserved for audit (e.g. cached tokens, finish_reason)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return self.request_status in ("success", "success_uncertain")


# execution status values
EXEC_SUCCESS = "success"
EXEC_RUNTIME_ERROR = "runtime_error"
EXEC_TIMEOUT = "timeout"
EXEC_EXECUTION_ERROR = "execution_error"
EXEC_CANCELLED = "cancelled"


@dataclass
class ExecResult:
    """Structured result of a code execution."""

    execution_id: str = field(default_factory=lambda: new_id("exec"))
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    script_ref: Optional[str] = None  # path, or "<code_string>" marker
    status: str = EXEC_SUCCESS
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    timeout_seconds: Optional[float] = None
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    timestamp: str = field(default_factory=_now_iso)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)