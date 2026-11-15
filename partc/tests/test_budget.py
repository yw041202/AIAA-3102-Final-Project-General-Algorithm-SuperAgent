"""Budget manager tests: exact decimal cost, reserve/settle, boundaries,
restart recovery, unknown-usage handling."""

from __future__ import annotations

from decimal import Decimal

import pytest

from partc.budget_manager import BudgetManager, CostCalculator
from partc.errors import BudgetExceededError

P_IN = Decimal("0.000002")  # 2 RMB / 1M
P_OUT = Decimal("0.000008")  # 8 RMB / 1M


def test_single_call_cost():
    calc = CostCalculator(P_IN, P_OUT)
    assert calc.cost(1_000_000, 0) == Decimal("2")
    assert calc.cost(0, 1_000_000) == Decimal("8")
    assert calc.cost(500_000, 250_000) == Decimal("1") + Decimal("2")


def test_cost_is_exact_decimal_no_float_drift():
    calc = CostCalculator(P_IN, P_OUT)
    total = sum((calc.cost(1, 1) for _ in range(10)), Decimal("0"))
    assert total == Decimal("0.00001") * 10
    assert total == Decimal("0.0001")


def test_reserve_and_settle(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("t1")
    bm.reserve("t1", "c1", "deepseek-v4.1-flash-1017", "peak", Decimal("0.02"))
    assert bm.summary("t1")["held_reserved_rmb"] == "0.02"
    bm.settle("t1", "c1", 1000, 1000, Decimal("0.01"))
    s = bm.summary("t1")
    assert s["spent_rmb"] == "0.01"
    assert s["held_reserved_rmb"] == "0"
    assert s["available_rmb"] == "19.99"


def test_budget_exactly_used(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("t1")
    bm.reserve("t1", "c1", "m", "peak", Decimal("20"))
    bm.settle("t1", "c1", 1_000_000, 2_000_000, Decimal("20"))
    s = bm.summary("t1")
    assert s["available_rmb"] == "0"
    assert not s["over_budget"]


def test_budget_insufficient_rejects(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("t1")
    bm.reserve("t1", "c1", "m", "peak", Decimal("19"))
    bm.settle("t1", "c1", 1, 1, Decimal("19"))
    with pytest.raises(BudgetExceededError):
        bm.reserve("t1", "c2", "m", "peak", Decimal("1.00000001"))


def test_tasks_are_independent(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("a")
    bm.ensure_task("b")
    bm.reserve("a", "c1", "m", "peak", Decimal("15"))
    # task b still has full budget
    assert bm.available("b") == Decimal("20")


def test_unknown_usage_holds_reservation(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("t1")
    bm.reserve("t1", "c1", "m", "peak", Decimal("3"))
    bm.mark_uncertain("t1", "c1", error_type="usage_unknown")
    s = bm.summary("t1")
    assert s["held_reserved_rmb"] == "3"
    assert s["spent_rmb"] == "0"
    assert s["uncertain"] == 1


def test_release_frees_reservation(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("t1")
    bm.reserve("t1", "c1", "m", "peak", Decimal("5"))
    bm.release("t1", "c1", error_type="bad_request")
    s = bm.summary("t1")
    assert s["held_reserved_rmb"] == "0"
    assert s["available_rmb"] == "20"


def test_restart_recovery(tmp_path):
    path = str(tmp_path / "ledger.json")
    bm = BudgetManager(path, "20")
    bm.ensure_task("t1")
    bm.reserve("t1", "c1", "m", "peak", Decimal("2"))
    bm.settle("t1", "c1", 1000, 1000, Decimal("1.5"))

    bm2 = BudgetManager(path, "20")
    s = bm2.summary("t1")
    assert s["spent_rmb"] == "1.5"
    assert s["calls"] == 1


def test_cumulative_cost(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"), "20")
    bm.ensure_task("t1")
    for i in range(3):
        bm.reserve("t1", f"c{i}", "m", "peak", Decimal("1"))
        bm.settle("t1", f"c{i}", 100, 100, Decimal("0.5"))
    assert bm.cumulative_cost("t1") == Decimal("1.5")


def test_default_budget_is_20(tmp_path):
    bm = BudgetManager(str(tmp_path / "ledger.json"))
    assert bm.ensure_task("t1").budget_rmb == Decimal("20")