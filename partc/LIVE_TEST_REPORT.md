# Live Test Report — Zhizengzeng Real API Smoke Test

**执行日期**: 2026-11-14（由用户亲自执行）
**命令**: `python examples/live_smoke_test.py --live`
**状态**: ✅ 成功（`request_status: success`）

> 本次为**一次真实、计费的请求**（约 ¥0.000186）。报告记录真实返回，未编造任何日志；
> 其中可被本地日志/账本佐证的字段均已核对，无法由本地文件佐证的字段标注为“来自终端输出”。

---

## 1. 测试目的

验证 Part C 的**真实 API 路径**（`SDKBackend` + `openai` SDK）端到端可用：
合规校验 → 预算预留 → 真实请求 → `usage` 解析 → 按峰值价结算 → JSONL 落盘 → 账本持久化。

## 2. 实际执行命令

```bash
python examples/live_smoke_test.py --live
```

（未 `--live` 时为 dry-run，不发起请求；真实请求必须显式传入 `--live`。）

## 3. 模型与 API 配置（不含密钥）

| 项 | 值 |
|---|---|
| 网关 Base URL | `https://api.zhizengzeng.com/v1`（归一化为单个 `/v1`） |
| requested_model_id | `deepseek-v4.1-flash-1017` |
| 密钥 | `ZZZ_API_KEY`（从 `.env` / 环境读取，未记录、未打印） |
| 默认任务预算 | RMB 20 |
| 计费口径 | 峰值（`peak`）、输入按 cache-miss 全价 |

## 4. 真实返回的关键字段

| 字段 | 值 | 来源 |
|---|---|---|
| `request_status` | `success` | 终端 + `logs/llm.jsonl` |
| `provider` | `zhizengzeng` | 终端 + 日志 |
| `requested_model_id` | `deepseek-v4.1-flash-1017` | 终端 + 日志 |
| `resolved_model_version` | `DeepSeek-V4.1-Flash-1017` | 终端 + 日志（**本地注册表映射，非后端独立验证**） |
| `prompt_tokens` | 37 | 终端 + 日志 |
| `completion_tokens` | 14 | 终端 + 日志 |
| `total_tokens` | 51（=37+14） | 终端（`usage.total_tokens`） |
| `reasoning_tokens` | 11 | 终端（`usage.extra`，来自 `completion_tokens_details`） |
| `cached_tokens` | 0 | 终端（`usage.extra`，来自 `prompt_tokens_details`） |
| `calculated_cost_rmb` | `0.000186` | 终端 + 日志 + 账本 |
| `pricing_version` | `peak` | 终端 + 日志 + 账本 |
| `retry_count` | 0 | 终端 + 日志 |
| `finish_reason` | `stop` | 终端 + 日志（`extra.finish_reason`） |
| `output_text` | `OK` | 终端（输出文本不持久化到日志） |
| `provider_response_id` | `chatcmpl-97403a20-…-6f6aeda70546` | 日志（`extra.provider_response_id`） |

## 5. Token 与费用计算核对

```
输入  37 × ¥2 / 1,000,000 = 0.000074
输出  14 × ¥8 / 1,000,000 = 0.000112
合计                      = 0.000186 RMB  ✅ 与返回一致
```

- 输入按**峰值 + cache-miss** 价 ¥2/1M；输出按**峰值** ¥8/1M。
- **Reasoning tokens（11）已计入 completion_tokens（14），不重复计费**。
  `cached_tokens=0`，故本次无缓存命中折扣（本工具口径也本就不计缓存折扣）。
- 本地账本 `.partc_state/ledger.json` 中该次 `status="settled"`、
  `input_tokens=37`、`output_tokens=14`、`cost_rmb=0.000186`，与返回完全一致。

## 6. 已验证的功能

- ✅ 真实 `openai` SDK 调用（非 Mock）成功、网关返回正常。
- ✅ 合规模型 `deepseek-v4.1-flash-1017` 通过 `ModelComplianceError` 门禁。
- ✅ 预算 `reserve → settle` 按真实 usage 结算，费用为精确 `Decimal`。
- ✅ `usage` 解析：`prompt/completion/total` 及 `reasoning/cached` 明细（后者仅入 `extra`，不重复计费）。
- ✅ 峰值价（`pricing_version=peak`）正确应用。
- ✅ JSONL 结构化日志落盘、账本持久化、`finish_reason`/响应 ID 留档。
- ✅ 密钥未出现在终端输出、日志或配置转储中。

## 7. 尚未验证 / 无法仅凭此次证明的事项

1. **后端真实模型权重版本**：`resolved_model_version` 来自**本地注册表映射**，本次测试
   **不能**独立证明智增增后端实际部署的模型权重版本与官方 `DeepSeek-V4.1-Flash-1017`
   完全一致。日志中的 `extra.provider_model = deepseek-v4.1-flash-1017` 只是网关回显了
   请求的模型串，不等于对后端版本的官方认证。
2. **平台实际扣费**：`calculated_cost_rmb` 是本地按官方价计算，本次未核对智增增账单。
3. **缓存命中 / 峰谷价 / 长上下文分档**：本次 `cached_tokens=0`、输入很小，未覆盖
   Qwen 上下文分档、>256K 长上下文、off-peak、cache-hit 等路径。
4. **失败/重试/超时/限流**：本次一次成功，未触发错误分支。
5. **其余合规模型（Qwen 系列）的真实可用性**：本次仅测了默认 DeepSeek 模型。
6. **官方价格与发布日期的再度认证**：一次成功调用不改变 `MODEL_COMPLIANCE.md` 中
   待课程确认的口径（峰/谷价取定、历史价格、汇率）。

## 8. 测试证据来源说明

- **`logs/llm.jsonl`**：含一次匹配记录（`task_id=live_smoke_test`、模型、`success`、
  `prompt_tokens=37`、`completion_tokens=14`、`calculated_cost=0.000186`、
  `pricing_version=peak`、`retry_count=0`、`extra.finish_reason=stop`）。
- **`.partc_state/ledger.json`**：含 `live_smoke_test` 一条 `settled` 记录，token 与费用一致。
- `total_tokens` / `reasoning_tokens` / `cached_tokens` / `output_text` 四个字段**未持久化**
  到当前 `llm.jsonl`（`log_llm` 仅落 `prompt/completion`），证据来自用户提供的真实终端输出。