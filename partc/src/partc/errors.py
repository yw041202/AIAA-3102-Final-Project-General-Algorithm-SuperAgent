"""Exception hierarchy for the Part C toolkit.

All Part C exceptions derive from :class:`PartCError` so that upper layers
(agent A) can catch one root type and still distinguish specific failures via
the ``error_type`` / ``kind`` attributes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PartCError(Exception):
    """Base class for all Part C errors."""

    kind = "partc_error"

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(PartCError):
    """Invalid or missing configuration (e.g. missing API key in live mode)."""

    kind = "configuration_error"


class ModelComplianceError(PartCError):
    """A requested model is not compliant with the course restrictions."""

    kind = "model_compliance_error"

    def __init__(self, message: str, *, model_id: str, reasons: list[str], **kw: Any) -> None:
        super().__init__(message, **kw)
        self.model_id = model_id
        self.reasons = reasons


class BudgetExceededError(PartCError):
    """A task's remaining budget is insufficient to reserve a call."""

    kind = "budget_exceeded"

    def __init__(
        self,
        message: str,
        *,
        task_id: str,
        call_id: str,
        required_rmb: str,
        available_rmb: str,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.task_id = task_id
        self.call_id = call_id
        self.required_rmb = required_rmb
        self.available_rmb = available_rmb


class BackendError(Exception):
    """An LLM backend failure, carrying a stable ``error_type`` taxonomy.

    ``retryable`` says whether an automatic retry is appropriate: timeouts,
    rate-limit and network errors are retryable; authentication and validation
    errors are not (retrying cannot fix them).
    """

    error_type = "unknown"
    retryable = False

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BackendTimeoutError(BackendError):
    error_type = "api_timeout"
    retryable = True


class BackendRateLimitError(BackendError):
    error_type = "rate_limit"
    retryable = True


class BackendNetworkError(BackendError):
    error_type = "network_error"
    retryable = True


class BackendServerError(BackendError):
    error_type = "server_error"
    retryable = True


class BackendAuthError(BackendError):
    error_type = "auth_error"
    retryable = False


class BackendBadRequestError(BackendError):
    error_type = "bad_request"
    retryable = False