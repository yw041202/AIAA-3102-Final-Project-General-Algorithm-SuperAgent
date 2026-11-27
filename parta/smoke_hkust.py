"""One bounded, logged API call using the existing C client. Run from repo root."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from partc import load_config, ZhizengzengLLMClient, ChatMessage


def main():
    cfg = load_config("parta/configs/hkust.env.example")
    cfg.budget_rmb = "0.10"
    cfg.state_dir = "runs/hkust_smoke/budget"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = Path("runs/hkust_smoke") / stamp
    cfg.log_dir = str(output / "logs")
    client = ZhizengzengLLMClient(cfg)
    result = client.chat([ChatMessage.user('Return only this JSON object: {"status":"ok","answer":2}')],
                         task_id="hkust_connectivity", run_id=stamp, max_tokens=128, temperature=0, retries=0)
    record = result.to_dict()
    if result.usage.is_complete:
        record["estimated_platform_cost_rmb"] = str((Decimal(result.usage.prompt_tokens) * Decimal("0.5")
            + Decimal(result.usage.completion_tokens) * Decimal("2")) / Decimal(1000000))
    record["base_url"] = cfg.base_url
    record["is_course_task_result"] = False
    record["platform_cost_note"] = "Estimate from school published rates, not observed account debit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: record.get(k) for k in ("request_status", "output_text", "usage", "calculated_cost_rmb",
        "estimated_platform_cost_rmb", "requested_model_id", "error_type", "error_message")}, ensure_ascii=True))
    print("Evidence: " + str(output / "result.json"))
    return 0 if result.request_status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
