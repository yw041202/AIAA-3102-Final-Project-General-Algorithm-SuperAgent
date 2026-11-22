"""Zhizengzeng live smoke test — explicitly triggered, NOT run by default.

Usage::

    python examples/live_smoke_test.py --live [--model deepseek-v4.1-flash-1017]

This makes ONE real, billable request through the Zhizengzeng gateway to
verify a real model responds and that token usage is reported.  It prints the
response and usage on stdout and writes a log record.

Without ``--live`` it does nothing except print the configuration (no cost).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partc.config import load_config, PartCConfig  # noqa: E402
from partc.budget_manager import BudgetManager  # noqa: E402
from partc.llm_client import ZhizengzengLLMClient  # noqa: E402
from partc.logger import StructuredLogger  # noqa: E402
from partc.models import ModelRegistry  # noqa: E402
from partc.schemas import ChatMessage, new_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Zhizengzeng live smoke test (billable).")
    parser.add_argument("--live", action="store_true", help="actually perform a billable API call")
    parser.add_argument("--model", default=None, help="model id (must be course-compliant)")
    parser.add_argument("--prompt", default="Reply with exactly the word: OK")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config()
    if not args.live:
        print("Dry run (no --live): no request will be made.")
        print("Config  :", json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if not cfg.has_api_key():
        print("ERROR: ZZZ_API_KEY is not set. Set it in .env or the environment.", file=sys.stderr)
        return 2

    model = args.model or cfg.model
    registry = ModelRegistry.load_default()
    info, reasons = registry.validate(model)
    if reasons:
        print(f"ERROR: model {model!r} is not course-compliant: {reasons}", file=sys.stderr)
        return 2

    budget = BudgetManager(os.path.join(cfg.state_dir, "ledger.json"), default_budget_rmb=cfg.budget_rmb)
    logger = StructuredLogger(cfg.log_dir, api_key=cfg.api_key)
    client = ZhizengzengLLMClient(cfg, budget_manager=budget, logger=logger, model_registry=registry)

    task_id = "live_smoke_test"
    run_id = new_id("live_run")
    result = client.chat(
        [ChatMessage.user(args.prompt)],
        model=model,
        task_id=task_id,
        run_id=run_id,
        max_tokens=args.max_tokens,
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())