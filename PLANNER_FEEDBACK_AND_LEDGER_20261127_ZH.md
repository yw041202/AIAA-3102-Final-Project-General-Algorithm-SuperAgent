# PR #3：配置反馈修复与账本来源核对

2026-11-27。本轮按 B 的配置生成问题、C 的账本来源问题依次处理。在本地集成工作树完成修改与验证，经用户确认后，以独立修复提交更新 PR #3。基础代码对应 PR #3 的 `ac92053` 文件内容；不将本地合并审查的 Git 合并历史直接推入 PR。

## 1. B 的配置生成与错误恢复问题

[B 的原评论](https://github.com/yw041202/AIAA-3102-Final-Project-General-Algorithm-SuperAgent/pull/3#issuecomment-7212137232)报告：Headlines 三轮均将文本参数包在 `preprocess_options.text` 中，第三轮还出现 `mindf` 拼写；简短错误未帮助模型纠正。Coupon 还曾输出 Python `None`。

修改位于 [parta/decisions.py](parta/decisions.py)、[parta/agent.py](parta/agent.py) 和 [回归测试](parta/tests/test_decisions.py)：

- 将允许字段和结构校验集中到同一模块，规划提示从该契约生成。根据表格或文本模态附上完整可解析的 JSON 示例，包括 `model`、`preprocess_options`、`null`、布尔值与停止动作。
- 明确 `preprocess_options` 必须平铺，不能再嵌套 `text`、`tabular` 或 `tfidf`，并提示 `min_df` 的正确拼写。示例说明为语法示范，由模型根据数据和反馈选择配置。
- 校验错误包含 `code`、字段路径、未知字段、缺失字段、允许字段和修正提示。它们保存到实验历史，并完整送入下一轮规划。例如 `preprocess_options.text.mindf` 会提示移除包装层，并在确有该意图时使用 `min_df`。
- 严格拒绝 Python 字面量、Markdown 围栏、重复 JSON 键、NaN/Infinity 等；不自动改写或执行非法配置。参数取值仍由 B/D 检查。

保留 `max_steps=6`、`patience=3`、客户端重试为 0，以及 B 的严格接口。没有通过增加轮数或人工指定模型配置来绕过失败。

### 测试与真实任务结果

- 新增针对性测试 **18 passed**：覆盖 B 的嵌套结构和拼写问题、非 JSON 输出、提示示例与解析器一致性、提示实际经过 C 客户端，以及“错误反馈 → 下一轮纠正 → 真实 B/C/D 训练与预测”的闭环。
- 全套 **184 passed，0 failed / 0 errors / 0 skipped，232.07 秒**。测试进程带学校和旧平台两种假密钥，使用隔离的测试账本。
- CPython **3.11.5** 与全部 **35** 个锁定包匹配，`pip check` 通过。
- 学校 API 自主运行：`task_id=headlines`、新 `run_id=feedback27`，沿用原 `runs/released/budget/ledger.json`。模型 `DeepSeek-V4.1-Flash`，本轮源码指纹 `d068c461d6943cc7cd6c3de4c8ef4135b8cccfc498ed1e424dc9095aae927231`。
- **6 次规划调用、6 个成功训练实验、0 次配置校验失败**；达到原定轮数后停止。本轮已结算费用 **¥0.013985**，无新增不确定调用。
- 按验证集最优选择 `experiment_004`：验证集 accuracy **0.6829787234**。选定并保存模型后，产出 **3,232** 条测试预测；最后一次性计算测试 accuracy **0.6615099010**，未用于规划或选择。
- 独立进程重新加载最终模型和预处理器，预测 CSV 与原输出逐字节相同。模型、日志、请求、结果、预测和测试 XML 已归档至 [本轮证据](review-evidence/feedback-20261127)。

此次运行全过程未修改代码或人工调整候选配置。它是修复后的工程验证，`official_result=false`；单次随机规划不能证明准确率提升。Coupon 本轮只做了离线回归，没有重新调用付费 API；历史 Coupon 结果仍属于原源码版本。

## 2. C 的账本差异与额外调用来源

[C 的原评审](https://github.com/yw041202/AIAA-3102-Final-Project-General-Algorithm-SuperAgent/pull/3#pullrequestreview-6070729439)询问：为什么 11 月 22 日两任务各 6 次调用，而 11 月 26 日兼容性核验中分别为 10 次和 8 次？

原因是：11 月 22 日快照之后，11 月 26 日集成核验以 `run_id=audit26` 继续使用同一累计账本。后来三项修复的兼容性检查只是读取该账本，并没有自行产生这些模型调用。兼容性 JSON 的顶层 `released` 是任务输出目录的分组名称，不是账本条目的状态。

| 任务 / 来源 | 新增调用（已结算 / 不确定） | 新增已结算费用 | 新增不确定预留 |
|---|---:|---:|---:|
| Coupon：11/22 `live_001` | 6（6 / 0） | ¥0.013216 | ¥0 |
| Coupon：11/26 `audit26` | 4（3 / 1） | ¥0.006029 | ¥0.014576 |
| Headlines：11/22 `live_001` | 6（6 / 0） | ¥0.009767 | ¥0 |
| Headlines：11/26 `audit26` | 2（1 / 1） | ¥0.001105 | ¥0.010817 |
| Headlines：本轮 11/27 `feedback27` | 6（6 / 0） | ¥0.013985 | ¥0 |

因此 11/26 的 Coupon 为 **10 次 = 9 已结算 + 1 不确定**，Headlines 为 **8 次 = 7 已结算 + 1 不确定**。本轮结束后 Coupon 不变，Headlines 为 **14 次 = 13 已结算 + 1 不确定**。

11/26 的两个不确定条目均对应 `server_error`，没有 token usage，不能按免费调用处理。原预留完整保留；预留不是实际已结算费用。客户端日志的 `retry_count=0` 表示客户端没有自动重试，不代表可以推断服务端内部尝试次数。

| 当前本地任务账本 | 已结算费用 | 不确定预留 |
|---|---:|---:|
| Coupon | ¥0.019245 | ¥0.014576 |
| Headlines | ¥0.024857 | ¥0.010817 |
| 两任务合计 | **¥0.044102** | **¥0.025393** |

学校连通性测试另有独立账本：已结算 **¥0.000059**、不确定预留 **¥0.000888**。它不计入上述两任务行，另附 `connectivity-ledger.json`。含该测试的本地已结算总额为 **¥0.044161**，不确定预留共 **¥0.026281**。这些数字仅覆盖已取得本地原始账本的调用，不代表全组所有成员的费用已汇总齐全。

### 核验方式和 B 的待核验费用

[reconciliation.json](review-evidence/feedback-20261127/reconciliation.json)列出每一条调用的 task ID、run ID、call ID、来源日志、token、费用和状态。[verify_ledger.py](review-evidence/feedback-20261127/verify_ledger.py)使用标准库即可重放核验：

```powershell
python review-evidence/feedback-20261127/verify_ledger.py
```

核验确认：11/22 条目到 11/26、再到本轮结束均仅追加；历史条目全部字段和任务额度未改变；原始日志与账本按 call ID 一一对应、无遗漏；11/26 汇总与已发布的兼容性记录相符；所有已结算费用按这些记录所用的课程价格映射（每百万输入 / 输出 token 为 ¥1 / ¥4）独立重算一致。此处核验固定历史映射，不主张重新核实了官方历史价表。

B 另报告 Headlines **3 次、¥0.003537**。用户确认暂时没有原始日志，因此单列为 **组员报告、待凭证核验**，不伪造 call ID 或 token，不并入已核验账本。待取得原始日志后再核对任务、调用 ID、价格版本、token 和是否重复，追加团队汇总。上述本地费用与 B 报告费用仍分别列示。

## 组员可核验的内容

- B：检查平铺契约与 B 接口一致；运行新增测试，查看新 Headlines 运行的 6 个请求及实际训练结果。
- C：运行账本核验脚本，按 `audit26` 原始日志核对新增的 4 次 / 2 次调用；确认不确定预留未消失。
- D：解压证据目录中的 `headlines-feedback27/final_model.zip`，按锁定环境及 B 原始数据重新预测，核对 `verification.json` 中 CSV 的 SHA-256。

本轮修复与证据已获用户授权发布至 PR #3，供组员交叉验收。PR 保持待审阅状态，不自动合并。
