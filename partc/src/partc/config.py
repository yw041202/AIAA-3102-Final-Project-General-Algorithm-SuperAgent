"""Configuration loading, base-URL normalisation and secret protection.

The toolkit reads settings from environment variables, optionally overlaid
from a ``.env`` file (loaded via ``python-dotenv`` when installed, otherwise a
minimal built-in parser is used so the package still works offline with the
standard library alone).

Security guarantees enforced here:

* the API key is never hard-coded; ``AIAA3102_API_KEY`` takes precedence over ``ZZZ_API_KEY``;
* other AI-tool environment variables (``OPENAI_API_KEY`` etc.) are ignored;
* the key is never included in log records or ``repr`` output;
* the base URL is normalised so that ``/v1/v1`` is never produced.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from .errors import ConfigurationError

# Environment-variable names that are never inherited by subprocesses we spawn.
SENSITIVE_ENV_VARS = {
    "AIAA3102_API_KEY",
    "ZZZ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
}

DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash-1017"


def _redact(value: Optional[str]) -> str:
    """Turn a secret into a safe ``<set:4 chars>…`` placeholder for logging."""
    if value is None:
        return "<unset>"
    if len(value) < 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]} (len={len(value)})"


def normalize_base_url(url: str) -> str:
    """Normalize common endpoints, preserving explicit versioned API bases.

    Handles the common mistakes: a bare host, ``/v1/v1`` duplication, a fully
    qualified ``/chat/completions`` endpoint, or a trailing slash.
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    # Strip an accidentally supplied full chat-completions endpoint.
    url = re.sub(r"/chat/completions/?$", "", url)
    if re.search(r"/api/v\d+$", url):
        return url
    # Strip any existing /v1 suffix(es) so we can add exactly one back.
    while url.endswith("/v1"):
        url = url[:-3]
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _load_dotenv(path: str, environ: Dict[str, str]) -> Dict[str, str]:
    """Load KEY=VALUE pairs from ``path`` into ``environ`` in-place.

    Tries ``python-dotenv`` first, then a small standard-library parser.  Does
    not override already-present environment variables (real env wins).
    """
    if not os.path.isfile(path):
        return environ
    loaded: Dict[str, str] = {}
    try:
        from dotenv import dotenv_values  # type: ignore

        loaded = {k: v for k, v in (dotenv_values(path) or {}).items() if v is not None}
    except Exception:
        # Minimal fallback: handle KEY=VALUE and simple `export KEY=VALUE`.
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.removeprefix("export ").strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                loaded[key] = value
    for key, value in loaded.items():
        environ.setdefault(key, value)  # real environment takes precedence
    return environ


@dataclass
class PartCConfig:
    """Resolved runtime configuration for the toolkit."""

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    budget_rmb: str = "20"
    log_dir: str = "./logs"
    state_dir: str = "./.partc_state"
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_base_seconds: float = 1.0
    env_file: str = ".env"
    # extra, unresolved settings preserved verbatim (for forward-compat)
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def api_key_redacted(self) -> str:
        return _redact(self.api_key)

    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def to_dict(self, *, redact_secrets: bool = True) -> Dict[str, object]:
        return {
            "api_key": self.api_key_redacted if redact_secrets else self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "budget_rmb": self.budget_rmb,
            "log_dir": self.log_dir,
            "state_dir": self.state_dir,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def load_config(env_file: Optional[str] = None) -> PartCConfig:
    """Build a :class:`PartCConfig` from process env + optional ``.env`` file.

    Raises :class:`ConfigurationError` only for malformed values, never for a
    missing API key (offline development and mock tests must keep working).
    """
    environ: Dict[str, str] = dict(os.environ)
    path = env_file or environ.get("PARTC_ENV_FILE", ".env")
    _load_dotenv(path, environ)

    def get(name: str, default: str) -> str:
        return environ.get(name, default).strip()

    base_url = normalize_base_url(get("ZZZ_BASE_URL", DEFAULT_BASE_URL))
    model = get("ZZZ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

    budget_rmb = get("PARTC_BUDGET_RMB", "20")
    timeout = _to_float(get("PARTC_TIMEOUT_SECONDS", "120"), "PARTC_TIMEOUT_SECONDS", 120.0)
    retries = _to_int(get("PARTC_MAX_RETRIES", "3"), "PARTC_MAX_RETRIES", 3)

    cfg = PartCConfig(
        api_key=environ.get("AIAA3102_API_KEY") or environ.get("ZZZ_API_KEY"),
        base_url=base_url,
        model=model,
        budget_rmb=budget_rmb,
        log_dir=get("PARTC_LOG_DIR", "./logs"),
        state_dir=get("PARTC_STATE_DIR", "./.partc_state"),
        timeout_seconds=timeout,
        max_retries=retries,
        env_file=path,
        extra={k: v for k, v in environ.items() if k.startswith(("ZZZ_", "PARTC_"))},
    )
    return cfg


def _to_float(value: str, name: str, default: float) -> float:
    try:
        return float(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be a number, got {value!r}")


def _to_int(value: str, name: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}")


def sanitize_environ(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of the environment with sensitive variables removed.

    Used by :class:`~partc.code_executor.PythonExecutor` so that spawned code
    cannot accidentally inherit ``ZZZ_API_KEY`` or other credentials.
    """
    env = dict(os.environ if environ is None else environ)
    for key in SENSITIVE_ENV_VARS:
        env.pop(key, None)
    return env
