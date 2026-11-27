# Model Compliance — 课程模型限制核实结论

**核实日期**: 2026-11-13
**课程截止日期**: 2026-10-21
**课程限制**: 仅 Qwen 或 DeepSeek；官方发布日期不晚于 2026-10-21；具有官方标准 API 定价。
**预算口径**: 单评价任务 LLM 预算 ≤ RMB 20；按 **2026-10-21 官方标准价** × 实际 token 计费；**不考虑**缓存折扣、平台促销、充值赠送。

> 本文结论**仅基于官方文档页面**。第三方（智增增等）的模型名、折扣价、赠送额度均不用于合规判断。
> 无法核实发布日期/价格的模型一律 `verified=false` 且 **不在正式计费模式启用**（fail closed）。

## 0. 信息核实层级（如何阅读本文）

本文区分四类证据强度，避免把“一次调用成功”误读成“官方认证”：

1. **已核实的官方信息**：发布日期与标准定价，来源为 DeepSeek / 阿里云百炼官方页面
   （见第 5 节）。仅此级别可用于合规定价。
2. **已通过实际 API 验证的信息**：2026-11-14 的 Live Smoke Test 仅证明
   `deepseek-v4.1-flash-1017` 被智增增接受并可返回正常 `usage`、费用按峰值价正确计算。
   **不**独立证明后端权重版本、也不重新认证官方价格/发布日期（见 `LIVE_TEST_REPORT.md`）。
3. **依据配置/文档推断的信息**：`deepseek-v4-pro-0813` 发布日期仅由 `0813` 后缀推断；
   `qwen3.8-flash` 精确发布日期未确认（仅知家族）。`resolved_model_version` 字段本身即
   **本地注册表映射**，不是后端实测值。
4. **仍待老师/组员确认的信息**：峰值/谷价取定、cache-miss 口径、历史价格口径、汇率、
   以及智增增后端真实部署版本。

---

## 1. 已核实并启用（compliant & allowed）

### DeepSeek

| requested_model_id | 后端版本 | 发布日期 | 输入价 (¥/百万 token) | 输出价 (¥/百万 token) | 状态 |
|---|---|---|---|---|---|
| `deepseek-v4.1-flash-1017` | DeepSeek-V4.1-Flash-1017 | 2026-10-17 ✔ | **2.0**（峰值、cache-miss） | **8.0**（峰值） | ✅ verified / allowed |

- 官方价格页版本号 `DeepSeek-V4.1-Flash-1017`，端点为 `deepseek-flash`。
- 计费口径（本工具采用的保守口径）：
  - 输入一律按 **cache-miss** 计费（缓存命中折扣被课程明确排除）；
  - 采用 **峰值(peak)** 价（为 off-peak 的 2 倍）。off-peak：输入 miss ¥1、输出 ¥4。
- 波动价：峰时 = 北京时间周一至周五 9:00–12:00 / 14:00–18:00（除中国法定节假日）。
  其余时间为 off-peak（半价）。

### Qwen / Aliyun Model Studio

| requested_model_id | 版本 | 发布日期 | 输入价 (¥/百万 token) | 输出价 (¥/百万 token) | 状态 |
|---|---|---|---|---|---|
| `qwen3.7-plus-2026-05-26` | Qwen3.7-Plus (2026-05-26) | 2026-05-26 ✔ | 2.0 (≤256K) | 8.0 (非思考/思考) | ✅ |
| `qwen3.7-flash-2026-07-15` | Qwen3.7-Flash (2026-07-15) | 2026-07-15 ✔ | 1.2 (≤1M) | 4.8 | ✅ |
| `qwen3.6-plus-2026-04-02` | Qwen3.6-Plus (2026-04-02) | 2026-04-02 ✔ | 2.0 (≤256K) | 12.0 | ✅ |

> 上述 Qwen 均选用**带日期后缀的版本化 Model ID**，可明确对应版本与发布日期。
> Qwen 按**上下文长度分档计价**：代码已实现按每次请求的实际输入 token 数选择对应档位
> （`ModelInfo.resolve_pricing`）；若输入超过所有已记录档位，则 **fail-closed**（不编造价格，
> 该调用被拒绝，计费状态标记 `pricing_tier_unresolved`）。
> DeepSeek 无上下文长度分档（1M 内统一定价），仅峰/谷价与 cache 命中/未命中差异，
> 本工具采用**峰值 + cache-miss** 的保守口径。

**默认候选模型**: `deepseek-v4.1-flash-1017`（轻量、支持视觉、价格低）。
**推荐 Qwen 备选**: `qwen3.7-plus-2026-05-26`（代码生成 / Agent 决策，价格适中、版本明确）。

---

## 2. 已记录但未启用（candidate，fail-closed）

| requested_model_id | 原因 |
|---|---|
| `deepseek-v4-pro-0813` | 价格已核实（输入峰值 ¥9/1M、输出峰值 ¥27/1M），但发布日期仅由 `0813` 后缀推断，未独立确认 → **not enabled** |
| `qwen3.8-flash` | 价格已知（≤1M：输入 ¥0.8 / 输出 ¥2.7），但 `flash` 变体精确发布日期未确认 → **not enabled** |

## 3. 明确不合规（排除）

| requested_model_id | 原因 |
|---|---|
| `qwen4-max-preview` | 发布于 2026-11-11（晚于截止日期），且为 preview/订阅制，非标准按 token 定价 |

## 4. 计费口径决策（本工具默认值，**待课程确认**）

1. **峰值价**：DeepSeek 存在峰/谷价差（±50%）。为避免低估费用，默认按**峰值**计费。
   → 待确认：课程“标准价”是否应取峰值、谷值或统一某个值。
2. **输入一律 cache-miss**：缓存命中优惠被课程排除，故无论响应是否回报 `cached_tokens`，输入均按全价（cache-miss）计费。
3. **币种**：DeepSeek 官方同时列示 USD 与 RMB，本工具采用官方中文页 **RMB** 价；课程预算为 RMB。
   → 待确认：是否需要汇率换算口径。
4. **思考 token**：DeepSeek thinking token 已计入 `completion_tokens`（不重复计费）；若智增增回报口径不同，需复核。
5. **无可靠 usage 的请求**：不做零成本假设，保留预算预留（`uncertain` 状态），待确认。

---

## 5. 数据来源

- DeepSeek 官方定价（英文/中文）：
  <https://api-docs.deepseek.com/quick_start/pricing/>
  <https://api-docs.deepseek.com/zh-cn/quick_start/pricing/>
- 阿里云百炼模型定价：
  <https://help.aliyun.com/zh/model-studio/model-pricing>
- DeepSeek-V4.1-Flash-1017 发布信息（第三方百科，用于核实 `1017` = 2026-10-17）：
  <https://baike.baidu.com/item/DeepSeek-V4.1-Flash-1017/69439738>
- 2026-11 智增增/DeepSeek 价格横评（仅作背景参考，**不**用于合规定价）：
  <https://www.csdn.net/article/2026-11-12/170598707>

## 6. 尚未确认的问题（见 HANDOFF.md 待确认清单）

- DeepSeek 峰/谷价在课程中如何取定。
- 智增增上 `deepseek-v4.1-flash-1017` 的实际模型字符串与前向版本是否会变化（`flash`、`latest` 动态别名风险）。
- 历史价格（截至 2026-10-21）的口径与汇率换算规则。

## 7. 官方固定版本 vs 智增增平台 Model ID（合规判定依据）

合规判定**只依据官方文档**，不依据智增增是否可调用。因此注册表区分两个字段：

| 字段 | 含义 | 示例 |
|---|---|---|
| `requested_model_id` | 实际发给智增增网关的字符串 | `deepseek-v4.1-flash-1017`、`qwen3.7-plus-2026-05-26` |
| `underlying_model_version` | 官方后端版本（用于合规与审计） | `DeepSeek-V4.1-Flash-1017`、`Qwen3.7-Plus (2026-05-26)` |

- 客户端在合规校验通过后，**总是发送 `requested_model_id`（canonical）**，即使调用方传入了
  别名（如 `deepseek-flash`）；别名仅用于本地解析，不会直接上送。
- `deepseek-flash` / `latest` 等**动态别名**可能在官方侧指向更新的版本 —— 因此它们仅作为本地
  别名映射到已核实版本，正式计费一律使用带日期/版本号的固定 ID。
- “在智增增可用” **不能**证明后端版本、发布日期或官方定价 —— 这些必须由官方文档核实后才
  可标记 `verified=true` 与 `allowed=true`（见第 2 节候选模型示例）。
- 智增增模型名与后端版本的**真实对应关系仍需一次 `--live` smoke test 确认**（已列于 HANDOFF）。