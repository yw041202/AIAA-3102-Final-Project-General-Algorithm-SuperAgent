"""C2 — Token & budget management.

Course rule: a single evaluation task may spend at most **RMB 20** on LLM
usage.  Cost is computed from *official* standard input/output token prices
(as of 2026-10-21) multiplied by *actual* token consumption.  Cache discounts,
platform promotions and free credits are deliberately excluded.

Design notes
------------
* All money is handled with :class:`decimal.Decimal` (never binary floats) to
  avoid cumulative rounding error.
* Cost is per task (``task_id``); each task has an independent ledger.
* A call first *reserves* a conservative upper-bound cost; on success it is
  *settled* to the real cost; on a failure that was never charged it is
  *released*; on a success with missing usage it is marked *uncertain* and the
  reservation is kept held (never silently treated as zero).
* The ledger is persisted to JSON so a restart restores state.
* V1 assumes a single task is driven serially; a per-process lock protects the
  in-memory ledger from interleaved mutations.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .errors import BudgetExceededError
from .schemas import _now_iso

_ONE_CENT = Decimal("0.00000001")


def _fmt(d: Decimal) -> str:
    q = d.quantize(_ONE_CENT)
    if q == 0:
        return "0"
    s = format(q, "f")  # fixed-point, never scientific notation
    s = s.rstrip("0").rstrip(".")
    return s


class CostCalculator:
    """Compute cost in RMB from prices and token counts (exact decimal)."""

    def __init__(self, input_price_per_token: Decimal, output_price_per_token: Decimal) -> None:
        self.input_price = input_price_per_token
        self.output_price = output_price_per_token

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Return exact cost for the given token counts."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return self.input_price * Decimal(input_tokens) + self.output_price * Decimal(output_tokens)

    def estimate_upper_bound(self, input_tokens: int, max_output_tokens: int) -> Decimal:
        """Upper-bound cost for an outgoing call with a given max output."""
        return self.cost(input_tokens, max_output_tokens)


@dataclass
class LedgerEntry:
    call_id: str
    model: str
    pricing_version: str
    status: str  # reserved | settled | uncertain | released
    timestamp: str = field(default_factory=_now_iso)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reserved_rmb: Decimal = Decimal("0")
    cost_rmb: Optional[Decimal] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "model": self.model,
            "pricing_version": self.pricing_version,
            "status": self.status,
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reserved_rmb": _fmt(self.reserved_rmb),
            "cost_rmb": None if self.cost_rmb is None else _fmt(self.cost_rmb),
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        cost = data.get("cost_rmb")
        return cls(
            call_id=data["call_id"],
            model=data.get("model", ""),
            pricing_version=data.get("pricing_version", ""),
            status=data["status"],
            timestamp=data.get("timestamp", ""),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            reserved_rmb=Decimal(str(data.get("reserved_rmb", "0"))),
            cost_rmb=None if cost is None else Decimal(str(cost)),
            error_type=data.get("error_type"),
        )


@dataclass
class TaskLedger:
    task_id: str
    budget_rmb: Decimal
    entries: List[LedgerEntry] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    @property
    def spent_rmb(self) -> Decimal:
        total = Decimal("0")
        for e in self.entries:
            if e.status == "settled":
                total += e.cost_rmb or Decimal("0")
        return total

    @property
    def held_rmb(self) -> Decimal:
        total = Decimal("0")
        for e in self.entries:
            if e.status in ("reserved", "uncertain"):
                total += e.reserved_rmb
        return total

    @property
    def available_rmb(self) -> Decimal:
        return self.budget_rmb - self.spent_rmb - self.held_rmb

    def summary(self) -> Dict[str, Any]:
        n_calls = len([e for e in self.entries if e.status != "released"])
        return {
            "task_id": self.task_id,
            "budget_rmb": _fmt(self.budget_rmb),
            "spent_rmb": _fmt(self.spent_rmb),
            "held_reserved_rmb": _fmt(self.held_rmb),
            "available_rmb": _fmt(self.available_rmb),
            "calls": n_calls,
            "settled": sum(1 for e in self.entries if e.status == "settled"),
            "uncertain": sum(1 for e in self.entries if e.status == "uncertain"),
            "errors": sum(1 for e in self.entries if e.error_type),
            "over_budget": self.available_rmb < 0,
        }


class BudgetManager:
    """Per-task, persistent, exact-decimal budget accounting."""

    def __init__(self, state_path: str, default_budget_rmb: Optional[str] = None) -> None:
        self.state_path = state_path
        self.default_budget_rmb = Decimal(str(default_budget_rmb or "20"))
        self._tasks: Dict[str, TaskLedger] = {}
        self._lock = threading.RLock()
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        tasks = data.get("tasks", {})
        for task_id, raw in tasks.items():
            ledger = TaskLedger(
                task_id=task_id,
                budget_rmb=Decimal(str(raw.get("budget_rmb", self.default_budget_rmb))),
                entries=[LedgerEntry.from_dict(e) for e in raw.get("entries", [])],
                created_at=raw.get("created_at", ""),
            )
            self._tasks[task_id] = ledger

    def _save(self) -> None:
        parent = os.path.dirname(self.state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "version": 1,
            "tasks": {
                tid: {
                    "task_id": t.task_id,
                    "budget_rmb": _fmt(t.budget_rmb),
                    "created_at": t.created_at,
                    "entries": [e.to_dict() for e in t.entries],
                }
                for tid, t in self._tasks.items()
            },
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    # -- task lifecycle ---------------------------------------------------
    def ensure_task(self, task_id: str, budget_rmb: Optional[str] = None) -> TaskLedger:
        with self._lock:
            if task_id not in self._tasks:
                budget = Decimal(str(budget_rmb or self.default_budget_rmb))
                self._tasks[task_id] = TaskLedger(task_id=task_id, budget_rmb=budget)
                self._save()
            return self._tasks[task_id]

    def has_task(self, task_id: str) -> bool:
        return task_id in self._tasks

    def get_task(self, task_id: str) -> Optional[TaskLedger]:
        return self._tasks.get(task_id)

    # -- accounting -------------------------------------------------------
    def available(self, task_id: str) -> Decimal:
        ledger = self.ensure_task(task_id)
        with self._lock:
            return ledger.available_rmb

    def reserve(self, task_id: str, call_id: str, model: str, pricing_version: str, reserve_rmb: Decimal) -> None:
        """Reserve an upper-bound cost before a call.  Raises BudgetExceededError."""
        with self._lock:
            ledger = self.ensure_task(task_id)
            reserve = Decimal(reserve_rmb)
            if reserve < 0:
                raise ValueError("reserve amount must be non-negative")
            if ledger.available_rmb < reserve:
                raise BudgetExceededError(
                    f"task {task_id!r} budget exceeded for call {call_id}",
                    task_id=task_id,
                    call_id=call_id,
                    required_rmb=_fmt(reserve),
                    available_rmb=_fmt(ledger.available_rmb),
                )
            entry = LedgerEntry(
                call_id=call_id,
                model=model,
                pricing_version=pricing_version,
                status="reserved",
                reserved_rmb=reserve,
            )
            ledger.entries.append(entry)
            self._save()

    def settle(self, task_id: str, call_id: str, input_tokens: int, output_tokens: int, cost_rmb: Decimal) -> None:
        """Replace a reservation with the actual settled cost."""
        with self._lock:
            ledger = self.ensure_task(task_id)
            entry = self._find(ledger, call_id)
            entry.status = "settled"
            entry.input_tokens = input_tokens
            entry.output_tokens = output_tokens
            entry.cost_rmb = Decimal(cost_rmb)
            entry.reserved_rmb = Decimal("0")
            self._save()

    def mark_uncertain(self, task_id: str, call_id: str, *, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None, error_type: Optional[str] = None) -> None:
        """Keep the reservation held because usage could not be verified."""
        with self._lock:
            ledger = self.ensure_task(task_id)
            entry = self._find(ledger, call_id)
            entry.status = "uncertain"
            entry.input_tokens = input_tokens
            entry.output_tokens = output_tokens
            entry.error_type = error_type
            # reserved_rmb is intentionally left in place (fail-closed)
            self._save()

    def release(self, task_id: str, call_id: str, error_type: Optional[str] = None) -> None:
        """Void a reservation for a call that was never charged."""
        with self._lock:
            ledger = self.ensure_task(task_id)
            entry = self._find(ledger, call_id)
            entry.status = "released"
            entry.reserved_rmb = Decimal("0")
            entry.cost_rmb = Decimal("0")
            entry.error_type = error_type
            self._save()

    def record_error_free(self, task_id: str, call_id: str, model: str, pricing_version: str, error_type: str) -> None:
        """Record a call that failed before incurring any known charge."""
        with self._lock:
            ledger = self.ensure_task(task_id)
            entry = LedgerEntry(
                call_id=call_id, model=model, pricing_version=pricing_version,
                status="released", reserved_rmb=Decimal("0"), cost_rmb=Decimal("0"),
                error_type=error_type,
            )
            ledger.entries.append(entry)
            self._save()

    def _find(self, ledger: TaskLedger, call_id: str) -> LedgerEntry:
        for e in reversed(ledger.entries):
            if e.call_id == call_id:
                return e
        raise KeyError(f"no ledger entry for call_id {call_id!r}")

    def summary(self, task_id: str) -> Dict[str, Any]:
        ledger = self.ensure_task(task_id)
        with self._lock:
            return ledger.summary()

    def cumulative_cost(self, task_id: str) -> Decimal:
        with self._lock:
            return self.ensure_task(task_id).spent_rmb

    def task_ids(self) -> List[str]:
        return sorted(self._tasks.keys())