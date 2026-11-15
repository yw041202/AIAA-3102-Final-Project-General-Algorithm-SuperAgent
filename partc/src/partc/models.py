"""Model registry / allowlist and course-compliance validation.

A model is *compliant* only when it satisfies all of the course restrictions:

1. the provider is Qwen or DeepSeek (inferred from the registry, not the name);
2. it was officially released on or before ``release_date_cutoff``
   (2026-10-21);
3. it has an official standard token price that has been verified against an
   official source (no fabricated pricing).

Models that cannot be verified are *kept as candidates* but are never enabled
in formal (billed) mode — ``validate`` fails closed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .errors import ModelComplianceError

ONE_MILLION = Decimal("1000000")


def _parse_date(value: str) -> date:
    """Parse ``YYYY-MM-DD`` or ``YYYY-MM`` / ``YYYY`` (assumed day 1)."""
    value = (value or "").strip()
    parts = value.split("-")
    try:
        if len(parts) == 3:
            y, m, d = (int(p) for p in parts)
            return date(y, m, d)
        if len(parts) == 2:
            y, m = (int(p) for p in parts)
            return date(y, m, 1)
        if len(parts) == 1 and parts[0].isdigit():
            return date(int(parts[0]), 1, 1)
    except (ValueError, TypeError):
        pass
    raise ValueError(f"unparseable date: {value!r}")


@dataclass
class ModelInfo:
    """One entry of the model registry."""

    provider: str
    requested_model_id: str
    underlying_model_version: Optional[str]
    official_release_date: Optional[str]
    release_date_confirmed: bool
    pricing: Dict[str, Any]
    verified: bool
    allowed: bool
    aliases: List[str] = field(default_factory=list)
    tiers: List[Dict[str, Any]] = field(default_factory=list)
    official_release_source: Optional[str] = None
    pricing_source: Optional[str] = None
    pricing_reference_date: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(
            provider=str(data.get("provider", "")).lower(),
            requested_model_id=str(data.get("requested_model_id", "")),
            underlying_model_version=data.get("underlying_model_version"),
            official_release_date=data.get("official_release_date"),
            release_date_confirmed=bool(data.get("release_date_confirmed", False)),
            pricing=data.get("pricing") or {},
            verified=bool(data.get("verified", False)),
            allowed=bool(data.get("allowed", False)),
            aliases=[str(a) for a in (data.get("aliases") or [])],
            tiers=[dict(t) for t in ((data.get("pricing") or {}).get("tiers") or []) if isinstance(t, dict)],
            official_release_source=data.get("official_release_source"),
            pricing_source=data.get("pricing_source"),
            pricing_reference_date=data.get("pricing_reference_date"),
            notes=data.get("notes"),
        )

    @property
    def pricing_version(self) -> str:
        return str(self.pricing.get("billing_mode") or self.pricing_reference_date or "unknown")

    def input_price_per_token(self) -> Optional[Decimal]:
        return _price_per_token(self.pricing.get("input_per_1m_tokens"))

    def output_price_per_token(self) -> Optional[Decimal]:
        return _price_per_token(self.pricing.get("output_per_1m_tokens"))

    def resolve_pricing(self, input_tokens: int) -> Tuple[Optional[Decimal], Optional[Decimal], str]:
        """Return ``(input_price, output_price, tier_label)`` for an input size.

        Tiered models pick the smallest context tier that covers ``input_tokens``.
        If the model has no tiers, flat pricing is returned.  If the input
        exceeds every recorded tier, both prices are ``None`` so the caller can
        fail closed rather than fabricate a price.
        """
        tiers = sorted(
            (t for t in self.tiers if t.get("max_input_tokens") is not None),
            key=lambda t: int(t["max_input_tokens"]),
        )
        if tiers:
            for t in tiers:
                if input_tokens <= int(t["max_input_tokens"]):
                    p_in = _price_per_token(t.get("input_per_1m_tokens"))
                    p_out = _price_per_token(t.get("output_per_1m_tokens"))
                    return p_in, p_out, str(t.get("label") or self.pricing_version)
            return None, None, "context_exceeds_all_tiers"
        return self.input_price_per_token(), self.output_price_per_token(), self.pricing_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "requested_model_id": self.requested_model_id,
            "underlying_model_version": self.underlying_model_version,
            "official_release_date": self.official_release_date,
            "release_date_confirmed": self.release_date_confirmed,
            "pricing": self.pricing,
            "verified": self.verified,
            "allowed": self.allowed,
            "aliases": self.aliases,
            "pricing_version": self.pricing_version,
        }


def _price_per_token(per_million: Any) -> Optional[Decimal]:
    if per_million is None or per_million == "":
        return None
    try:
        return Decimal(str(per_million)) / ONE_MILLION
    except Exception:
        return None


class ModelRegistry:
    """In-memory model allowlist with compliance validation."""

    def __init__(self, models: List[ModelInfo], compliance: Optional[Dict[str, Any]] = None):
        self.compliance = compliance or {}
        self.release_date_cutoff = self.compliance.get("release_date_cutoff", "2026-10-21")
        self.providers_allowed = set(self.compliance.get("providers_allowed", []) or ["deepseek", "qwen"])
        self._by_id: Dict[str, ModelInfo] = {}
        self._by_alias: Dict[str, str] = {}
        for m in models:
            self._register(m)

    @classmethod
    def from_file(cls, path: str) -> "ModelRegistry":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRegistry":
        compliance = data.get("compliance", {}) or {}
        models = [ModelInfo.from_dict(m) for m in (data.get("models") or [])]
        return cls(models, compliance)

    @classmethod
    def load_default(cls, registry_path: Optional[str] = None) -> "ModelRegistry":
        """Load the bundled registry, or an explicit path, or fall back to empty."""
        if registry_path and os.path.isfile(registry_path):
            return cls.from_file(registry_path)
        bundled = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "model_registry.json",
        )
        if os.path.isfile(bundled):
            return cls.from_file(bundled)
        return cls([])

    def _register(self, model: ModelInfo) -> None:
        self._by_id[model.requested_model_id] = model
        for alias in model.aliases:
            self._by_alias[alias] = model.requested_model_id

    def resolve(self, model_id: str) -> Optional[ModelInfo]:
        if not model_id:
            return None
        key = model_id.strip()
        if key in self._by_id:
            return self._by_id[key]
        if key in self._by_alias:
            return self._by_id[self._by_alias[key]]
        # tolerate case / whitespace differences
        folded = {k.lower(): v for k, v in self._by_id.items()}
        if key.lower() in folded:
            return self._by_id[folded[key.lower()]]
        return None

    def all_model_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def validate(self, model_id: str) -> Tuple[Optional[ModelInfo], List[str]]:
        """Return ``(model_info, reasons)``; empty ``reasons`` means compliant."""
        reasons: List[str] = []
        info = self.resolve(model_id)
        if info is None:
            return None, [f"unknown model id {model_id!r} (not in allowlist)"]

        if info.provider not in self.providers_allowed:
            reasons.append(f"provider {info.provider!r} is not in the course allowlist {sorted(self.providers_allowed)}")

        try:
            cutoff = _parse_date(self.release_date_cutoff)
            if info.official_release_date:
                released = _parse_date(info.official_release_date)
                if released > cutoff:
                    reasons.append(
                        f"released {info.official_release_date} after cutoff {self.release_date_cutoff}"
                    )
            else:
                reasons.append("no official_release_date recorded")
        except ValueError as exc:
            reasons.append(str(exc))

        if not info.official_release_date or not info.release_date_confirmed:
            reasons.append("official release date is not confirmed")

        if info.input_price_per_token() is None or info.output_price_per_token() is None:
            reasons.append("no verified official standard input/output pricing")

        if not info.verified:
            reasons.append("model pricing/release not verified (unverified)")

        if not info.allowed:
            reasons.append("model not enabled (allowed=false)")

        return info, reasons

    def require_compliant(self, model_id: str) -> ModelInfo:
        info, reasons = self.validate(model_id)
        if reasons:
            name = info.requested_model_id if info else model_id
            raise ModelComplianceError(
                f"model {name!r} is not course-compliant",
                model_id=model_id,
                reasons=reasons,
            )
        assert info is not None
        return info

    def get_pricing(self, model_id: str) -> Tuple[Decimal, Decimal]:
        """Return ``(input_price_per_token, output_price_per_token)`` for a
        compliant model, in CNY per token."""
        info = self.require_compliant(model_id)
        p_in, p_out = info.input_price_per_token(), info.output_price_per_token()
        assert p_in is not None and p_out is not None
        return p_in, p_out

    def compliant_model_ids(self) -> List[str]:
        """Model ids that currently pass :meth:`validate` (for convenience)."""
        return [mid for mid in self._by_id if not self.validate(mid)[1]]