"""Model registry / compliance validation tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from partc.errors import ModelComplianceError
from partc.models import ModelRegistry


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load_default()


def test_compliant_models_pass(registry):
    for model_id in (
        "deepseek-v4.1-flash-1017",
        "qwen3.7-plus-2026-05-26",
        "qwen3.7-flash-2026-07-15",
        "qwen3.6-plus-2026-04-02",
    ):
        info, reasons = registry.validate(model_id)
        assert reasons == [], f"{model_id} should be compliant, got {reasons}"


def test_release_date_unconfirmed_fails_closed(registry):
    # deepseek-v4-pro-0813 has verified pricing but an inferred release date
    info, reasons = registry.validate("deepseek-v4-pro-0813")
    assert info is not None
    assert any("not confirmed" in r for r in reasons)


def test_noncompliant_released_after_cutoff_rejected(registry):
    info, reasons = registry.validate("qwen4-max-preview")
    assert info is not None
    assert any("after cutoff" in r for r in reasons)


def test_unknown_model_rejected(registry):
    info, reasons = registry.validate("gpt-4-turbo")
    assert info is None
    assert any("unknown model id" in r for r in reasons)


def test_unverified_candidate_rejected(registry):
    info, reasons = registry.validate("qwen3.8-flash")
    assert info is not None
    assert any("not enabled" in r or "unverified" in r or "not confirmed" in r for r in reasons)


def test_alias_resolves_to_same_model(registry):
    canonical = registry.resolve("deepseek-v4.1-flash-1017")
    via_alias = registry.resolve("deepseek-flash")
    assert canonical is via_alias and canonical is not None
    assert canonical.underlying_model_version == "DeepSeek-V4.1-Flash-1017"


def test_pricing_positive_and_exact(registry):
    p_in, p_out = registry.get_pricing("deepseek-v4.1-flash-1017")
    assert p_in == Decimal("0.000002")  # 2.0 RMB / 1M tokens
    assert p_out == Decimal("0.000008")  # 8.0 RMB / 1M tokens
    assert p_in > 0 and p_out > 0


def test_require_compliant_raises(registry):
    with pytest.raises(ModelComplianceError):
        registry.require_compliant("gpt-4-turbo")
    with pytest.raises(ModelComplianceError):
        registry.require_compliant("qwen4-max-preview")


def test_all_providers_are_qwen_or_deepseek(registry):
    for model_id in registry.all_model_ids():
        info = registry.resolve(model_id)
        assert info.provider in ("qwen", "deepseek")


def test_no_null_pricing_for_allowed_models(registry):
    for model_id in registry.all_model_ids():
        info = registry.resolve(model_id)
        if info and info.allowed:
            assert info.input_price_per_token() is not None
            assert info.output_price_per_token() is not None


def test_qwen_plus_context_tier_selection(registry):
    info = registry.resolve("qwen3.7-plus-2026-05-26")
    # <=256K tier
    p_in, p_out, label = info.resolve_pricing(1000)
    assert (p_in, p_out) == (Decimal("0.000002"), Decimal("0.000008"))
    assert label == "<=256K"
    # >256K => <=1M tier (higher price)
    p_in, p_out, label = info.resolve_pricing(300_000)
    assert (p_in, p_out) == (Decimal("0.000006"), Decimal("0.000024"))
    assert label == "<=1M"
    # exceeds every recorded tier => fail-closed (None prices)
    p_in, p_out, label = info.resolve_pricing(2_000_000)
    assert p_in is None and p_out is None


def test_qwen_flash_context_tier_selection(registry):
    info = registry.resolve("qwen3.7-flash-2026-07-15")
    p_in, p_out, _ = info.resolve_pricing(5_000)
    assert (p_in, p_out) == (Decimal("0.0000002"), Decimal("0.0000008"))  # <=32K
    p_in, p_out, _ = info.resolve_pricing(300_000)
    assert (p_in, p_out) == (Decimal("0.0000012"), Decimal("0.0000048"))  # <=1M


def test_deepseek_flat_pricing_no_tier(registry):
    info = registry.resolve("deepseek-v4.1-flash-1017")
    p_in, p_out, label = info.resolve_pricing(1_000_000)
    assert (p_in, p_out) == (Decimal("0.000002"), Decimal("0.000008"))
    assert label == "peak"