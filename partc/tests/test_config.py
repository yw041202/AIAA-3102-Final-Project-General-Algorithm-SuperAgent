"""Config loading, base-URL normalisation and API-key protection."""

from __future__ import annotations

import os

from partc.config import (
    load_config,
    normalize_base_url,
    sanitize_environ,
)
from partc.errors import ConfigurationError


def test_normalize_base_url_keeps_single_v1():
    assert normalize_base_url("https://api.zhizengzeng.com/v1") == "https://api.zhizengzeng.com/v1"


def test_normalize_base_url_adds_v1():
    assert normalize_base_url("https://api.zhizengzeng.com") == "https://api.zhizengzeng.com/v1"


def test_normalize_base_url_deduplicates_v1():
    assert normalize_base_url("https://api.zhizengzeng.com/v1/v1") == "https://api.zhizengzeng.com/v1"


def test_normalize_base_url_strips_chat_completions():
    assert normalize_base_url("https://api.zhizengzeng.com/v1/chat/completions") == "https://api.zhizengzeng.com/v1"


def test_normalize_base_url_trailing_slash():
    assert normalize_base_url("https://api.zhizengzeng.com/v1/") == "https://api.zhizengzeng.com/v1"


def test_normalize_adds_scheme():
    assert normalize_base_url("api.zhizengzeng.com") == "https://api.zhizengzeng.com/v1"


def test_load_config_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZZZ_API_KEY", "sk-test-1234567890")
    monkeypatch.setenv("ZZZ_BASE_URL", "https://api.zhizengzeng.com/v1/v1")
    monkeypatch.setenv("ZZZ_MODEL", "deepseek-v4.1-flash-1017")
    monkeypatch.setenv("PARTC_BUDGET_RMB", "20")
    cfg = load_config(env_file=str(tmp_path / "does-not-exist.env"))
    assert cfg.api_key == "sk-test-1234567890"
    assert cfg.base_url == "https://api.zhizengzeng.com/v1"
    assert cfg.model == "deepseek-v4.1-flash-1017"
    assert cfg.budget_rmb == "20"


def test_load_config_ignores_unrelated_keys(monkeypatch, tmp_path):
    # other AI tools' keys must be ignored
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-be-ignored")
    monkeypatch.delenv("ZZZ_API_KEY", raising=False)
    cfg = load_config(env_file=str(tmp_path / "nope.env"))
    assert cfg.api_key in (None, "")


def test_load_config_bad_number_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTC_TIMEOUT_SECONDS", "abc")
    import pytest

    with pytest.raises(ConfigurationError):
        load_config(env_file=str(tmp_path / "nope.env"))


def test_api_key_redacted():
    from partc.config import PartCConfig

    cfg = PartCConfig(api_key="sk-super-secret-value-12345")
    assert "super-secret" not in cfg.api_key_redacted
    assert "2345" in cfg.api_key_redacted  # tail kept for identification


def test_to_dict_redacts_secrets():
    from partc.config import PartCConfig

    cfg = PartCConfig(api_key="sk-secret-abcdefgh")
    d = cfg.to_dict()
    assert d["api_key"] != "sk-secret-abcdefgh"
    assert d["api_key"].startswith("sk-s")


def test_sanitize_environ_removes_zzz_key(monkeypatch):
    monkeypatch.setenv("ZZZ_API_KEY", "secret")
    monkeypatch.setenv("KEEP_ME", "keep")
    env = sanitize_environ()
    assert "ZZZ_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert env.get("KEEP_ME") == "keep"


def test_school_key_precedence_and_secret_free_repr(monkeypatch, tmp_path):
    monkeypatch.setenv("AIAA3102_API_KEY", "fake-school-credential")
    monkeypatch.setenv("ZZZ_API_KEY", "fake-legacy-credential")
    cfg = load_config(str(tmp_path / "absent.env"))
    assert cfg.api_key == "fake-school-credential"
    assert "ZZZ_API_KEY" not in cfg.extra
    assert "AIAA3102_API_KEY" not in cfg.extra
    assert "fake-school-credential" not in repr(cfg)
    assert "fake-legacy-credential" not in repr(cfg)
    clean = sanitize_environ()
    assert "AIAA3102_API_KEY" not in clean
    assert "ZZZ_API_KEY" not in clean


def test_direct_config_extra_cannot_leak_in_repr():
    from partc.config import PartCConfig
    cfg = PartCConfig(api_key="fake-main-secret", extra={"ZZZ_API_KEY": "fake-extra-secret"})
    assert "fake-main-secret" not in repr(cfg)
    assert "fake-extra-secret" not in repr(cfg)
