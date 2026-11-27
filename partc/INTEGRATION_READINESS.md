# Integration Readiness — Part C V1

Part C 是否已具备交付 A 计划 / D 训练成员的资格评估。结论基于四项验收：公开 API、
模型合规、可靠性、模拟 A/C/D 集成。本文件**不宣称**自治 Agent 已实现；它只评估
“C 层对外接口是否足以支撑 A/D 接入”。

---

## 1. 验收结论（tldr）

| 验收项 | 结果 |
|---|---|
| 公开 API 可独立构造/调用 | ✅ 通过 |
| 预算单任务查询 | ✅ 通过 |
| 执行器独立运行脚本 | ✅ 通过 |
| 结构化结果可直接消费 | ✅ 通过 |
| 错误状态可识别 | ✅ 通过 |
| 无外部文件/隐式路径依赖 | ✅ 通过 |
| 新 Python 进程可 `import partc` | ✅ 通过 |
| 模拟 A/C/D 集成（离线） | ✅ 通过 |
| 多次失败计费、未知 usage 保守处理 | ✅ 通过 |
| 重启预算恢复、超时清场、密钥不继承 | ✅ 通过 |
| 真实 smoke test 走正式 C 客户/预算路径 | ✅ 通过；且已于 2026-11-14 实际执行成功（见 LIVE_TEST_REPORT.md） |

**结论：Part C V1 已具备与 A/D 交接集成的条件。** 遗留问题为“后端真实版本确认”与
“课程口径确认”（见第 6 节），**不阻塞** A/D 并行开发。

---

## 2. 公开 API 验收（Task 2）

在包目录之外（`/tmp`）以全新 Python 进程验证：无隐式路径、无 `.env`、无本机 state
依赖即可完成以下全部操作。

| 检查项 | 做法 | 结果 |
|---|---|---|
| 独立构造 LLM 客户端 | `ZhizengzengLLMClient(cfg, budget_manager=…, logger=…, model_registry=…, backend=MockBackend(...))` | ✅ |
| 独立构造预算并查询 | `BudgetManager(ledger, "20")` → `reserve/settle` → `summary(task_id)` | ✅ task_id 对应、spent/available 一致 |
| 执行器独立跑脚本 | `PythonExecutor(...).execute_code(script, task_id=…)` → `ExecResult` | ✅ status/exit_code/stdout 结构化 |
| 结构化结果直接可用 | `LLMResult` / `ExecResult` / `budget.summary()` 均返回 dataclass/dict | ✅ |
| 错误状态可识别 | `budget_exceeded`、`error`、`pricing_tier_unresolved`；非合规模型抛 `ModelComplianceError` | ✅ |
| 包导入（新进程） | `PYTHONPATH=src python -c "import partc"` | ✅ `StructuredLogger` 已导出 |

---

## 3. 模拟 A/C/D 集成验收（Task 3）

`examples/integration_demo.py`（离线、无真实 Agent）流程：

1. `MockPlanner`（模拟 A）产出一次实验请求；
2. C 的 `PythonExecutor` 执行纯标准库训练脚本（最近邻中心分类，合成二维数据）；打印
   一行 JSON 指标；
3. `ExecResult` 返回结构化状态 / 退出码 / stdout / 日志路径；
4. `MockPlanner.read_metric` 解析出 `accuracy`；
5. 通过正式 C 客户端发起一次后续 LLM 调用（MockBackend）；
6. C 返回任务级成本 + 执行日志汇总。

| 断言 | 结果 |
|---|---|
| `execution_status == "success"`、`exit_code == 0` | ✅ |
| `accuracy >= 0.9`（实际 1.0） | ✅ |
| `stdout_path` / `stderr_path` 非空 | ✅ |
| `llm_request_status == "success"`、cost > 0、输出非空 | ✅ |
| budget/llm/exec 三个汇总 task_id 一致 | ✅ |

覆盖测试：`tests/test_integration_demo.py`（`test_integration_demo_flow`、
`test_integration_demo_persists_and_recovers`）。

---

## 4. 可靠性验收（Task 4）

| 检查项 | 结论 |
|---|---|
| 多次失败计费 | 重试仅在最终一次结算扣费；**失败/重试的 token 不重复计费**，仅按最后一次可用 usage 结算，无 usage 时标记 `uncertain` 保留预留。 |
| 未知 usage 保守处理 | 无可靠 usage 时不按 0 计费，保留预算预留（`success_uncertain` / `uncertain`），待确认。 |
| 重启预算恢复 | `BudgetManager` 从 `ledger.json` 恢复；第二次构造同一账本读取到相同 `spent_rmb`。 |
| 执行器超时清场 | `start_new_session=True` + `os.killpg` 杀整棵进程组；不残留子进程。 |
| 子进程不继承密钥 | 执行器 `sanitize_env=True` 剥离 `ZZZ_API_KEY` 等密钥后再 spawn。 |
| JSONL 可解析 | `logs/llm.jsonl`、`logs/execution.jsonl` 每行合法 JSON；密钥字段已脱敏（token 计数字段保留）。 |
| 外部导入 | 全部入口函数不依赖 A/B/D 模块存在；`PYTHONPATH=src` 新进程可导入。 |
| 真实 smoke test 走正式路径 | `examples/live_smoke_test.py --live` 构造 `ZhizengzengLLMClient`（含预算/日志/注册表），不绕过 SDK 直发。 |

> **明确保留的告警**：`PythonExecutor` 不是安全沙箱（macOS subprocess）。V1 不部署
> Docker；不可信/陌生代码不能视为安全执行，仅可信受控入口可用。

---

## 5. 模型与定价验收（Task 1）

见 `MODEL_COMPLIANCE.md`。要点：

- `qwen37-flash-2026-07-15` 为笔误，规范写法 `qwen3.7-flash-2026-07-15`（注册表已用规范名）。
- DeepSeek `¥2/1M` 输入（峰值、cache-miss）、`¥8/1M` 输出（峰值）为**标准价峰值**，来源为官方
  定价页（见 MODEL_COMPLIANCE 第 5 节）。
- Qwen 上下文分档计价已实现（`ModelInfo.resolve_pricing`），超档 fail-closed。
- 官方固定版本 ID 与智增增平台 Model ID 已区分（`requested_model_id` vs
  `underlying_model_version`，见 MODEL_COMPLIANCE 第 7 节）；“平台可调用”不等于“后端已核实”。

---

## 6. 遗留项（不阻塞离线并行开发）

- **真实验证（已部分完成）**：2026-11-14 Live Smoke Test 已确认网关接受
  `deepseek-v4.1-flash-1017`、`usage` 字段（含 `reasoning/cached_tokens`）正确解析、费用按
  峰值价计算。**遗留**：智增增**后端真实权重版本**仍未独立验证（`resolved_model_version`
  是本地注册表映射）。
- **课程口径确认**：峰值/谷价取定、汇率换算、历史价格依据，见 HANDOFF「尚需全组确认」。
- **安全执行环境**：不可信代码的沙箱化不在 V1（不部署 Docker / 不承诺任意代码安全）。