from decimal import Decimal
from types import SimpleNamespace

from partc import PartCConfig, ModelRegistry
from partc.config import normalize_base_url
from partc.llm_client import SDKBackend


def test_hkust_v2_base_is_preserved():
    assert normalize_base_url("https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions") == "https://aigc-api.hkust-gz.edu.cn/api/v2"


def test_hkust_model_uses_course_off_peak_rates():
    model = ModelRegistry.load_default().require_compliant("DeepSeek-V4.1-Flash")
    assert model.underlying_model_version == "DeepSeek-V4.1-Flash-1017"
    assert model.input_price_per_token() == Decimal("0.000001")
    assert model.output_price_per_token() == Decimal("0.000004")


def test_school_thinking_flag_reaches_sdk(monkeypatch):
    import openai
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")], usage=None)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    backend = SDKBackend(PartCConfig(api_key="dummy", base_url="https://aigc-api.hkust-gz.edu.cn/api/v2", extra={"PARTC_DISABLE_THINKING": "true"}))
    backend.complete("DeepSeek-V4.1-Flash", [], 128, 0, 60)
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_school_direct_transport_is_explicit_and_verified(monkeypatch):
    import httpx
    import openai
    recorded = {}
    sentinel = object()
    def transport(**kwargs):
        recorded.update(kwargs)
        return sentinel
    def sdk(**kwargs):
        assert kwargs["http_client"] is sentinel
        assert kwargs["base_url"] == "https://aigc-api.hkust-gz.edu.cn/api/v2"
        return SimpleNamespace()
    monkeypatch.setattr(httpx, "Client", transport)
    monkeypatch.setattr(openai, "OpenAI", sdk)
    SDKBackend(PartCConfig(api_key="dummy", base_url="https://aigc-api.hkust-gz.edu.cn/api/v2", extra={"PARTC_TRUST_ENV": "false"}))
    assert recorded == {"trust_env": False}  # verify=False must never be supplied.
