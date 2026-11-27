"""Part C — Tooling for the AIAA3102 General Algorithm SuperAgent.

This package provides the "C" responsibilities of the project:

* C1: LLM client for the Zhizengzeng (智增增) unified API gateway
* C2: Token & budget management (RMB 20 / task, exact decimal accounting)
* C3: Python code executor
* C4: Structured JSONL logging & error handling

All code is importable as a normal Python package and does not depend on the
A/B/D modules existing.
"""

from __future__ import annotations

from .schemas import (
    EXEC_CANCELLED,
    EXEC_EXECUTION_ERROR,
    EXEC_RUNTIME_ERROR,
    EXEC_SUCCESS,
    EXEC_TIMEOUT,
    ChatMessage,
    ExecResult,
    LLMResult,
    TokenUsage,
    new_id,
)
from .errors import (
    BudgetExceededError,
    BudgetStateError,
    ConfigurationError,
    ModelComplianceError,
    PartCError,
)
from .config import PartCConfig, load_config
from .models import ModelInfo, ModelRegistry
from .budget_manager import BudgetManager, CostCalculator
from .logger import StructuredLogger
from .code_executor import PythonExecutor
from .llm_client import ZhizengzengLLMClient

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # schemas
    "ChatMessage",
    "LLMResult",
    "TokenUsage",
    "ExecResult",
    "new_id",
    "EXEC_SUCCESS",
    "EXEC_RUNTIME_ERROR",
    "EXEC_TIMEOUT",
    "EXEC_EXECUTION_ERROR",
    "EXEC_CANCELLED",
    # errors
    "PartCError",
    "ConfigurationError",
    "ModelComplianceError",
    "BudgetExceededError",
    "BudgetStateError",
    # modules
    "PartCConfig",
    "load_config",
    "ModelInfo",
    "ModelRegistry",
    "BudgetManager",
    "CostCalculator",
    "StructuredLogger",
    "PythonExecutor",
    "ZhizengzengLLMClient",
]
