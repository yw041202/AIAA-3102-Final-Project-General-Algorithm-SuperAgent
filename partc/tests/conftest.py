"""Pytest fixtures for the Part C toolkit.

Makes ``src/`` importable without an installed package, so tests run against
the working tree directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
for _p in (SRC, EXAMPLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402

from partc.config import PartCConfig  # noqa: E402
from partc.models import ModelRegistry  # noqa: E402
from partc.budget_manager import BudgetManager  # noqa: E402
from partc.logger import StructuredLogger  # noqa: E402
from partc.llm_client import ZhizengzengLLMClient, MockBackend  # noqa: E402


@pytest.fixture
def cfg(tmp_path) -> PartCConfig:
    return PartCConfig(
        api_key=None,
        base_url="https://api.zhizengzeng.com/v1",
        model="deepseek-v4.1-flash-1017",
        budget_rmb="20",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
        timeout_seconds=30.0,
        max_retries=3,
        retry_backoff_base_seconds=0.0,
    )


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.load_default()


@pytest.fixture
def budget(tmp_path) -> BudgetManager:
    return BudgetManager(str(tmp_path / "ledger.json"), default_budget_rmb="20")


@pytest.fixture
def logger(tmp_path) -> StructuredLogger:
    return StructuredLogger(str(tmp_path / "logs"))


@pytest.fixture
def make_client(cfg, budget, logger):
    def _make(backend=None, mock=False, registry=None):
        return ZhizengzengLLMClient(
            cfg,
            budget_manager=budget,
            logger=logger,
            model_registry=registry or ModelRegistry.load_default(),
            backend=backend,
            mock=mock,
        )

    return _make


@pytest.fixture
def client(make_client):
    return make_client(mock=True)