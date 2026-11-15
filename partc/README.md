# Part C — Toolkit for the AIAA3102 General Algorithm SuperAgent

Part C 是 SuperAgent 的「LLM 访问 / 预算 / 执行 / 日志」层。它向上层的 agent
planning（**A**）、data（**B**）与 model training（**D**）提供统一、合规、可计费、
可复现的调用入口，本身是一个普通、可导入的 Python 包，**不依赖** A/B/D 模块存在。

## 职责边界

Part C **负责**：

| #  | 模块                     | 职责                                                         |
|----|--------------------------|-------------------------------------------------------------|
| C1 | `partc.llm_client`       | Zhizengzeng LLM 客户端（OpenAI 兼容；真实 SDK / 离线 Mock 双后端） |
| C2 | `partc.budget_manager`   | 每任务 RMB 预算、`Decimal` 精确计费、上下文分档/峰值价、账本持久化 |
| C3 | `partc.code_executor`    | Python 子进程执行器（文件 / 代码串、超时、环境脱敏）            |
| C4 | `partc.logger`           | 结构化 JSONL 日志 + 任务统计 + 密钥脱敏                        |
| —  | `partc.models`           | 模型合规白名单 / 注册表（fail-closed）                         |
| —  | `partc.config`           | `.env` + 环境配置、Base URL 归一化、密钥保护                   |

Part C **不负责**：实验规划（A）、数据处理（B）、模型训练与评估（D）、不可信代码的
沙箱化执行、以及真实 Agent 的自治决策。

## 代码结构

```text
partc/
├── src/partc/                 # 包源码
│   ├── __init__.py            # 顶层导出（含 StructuredLogger）
│   ├── config.py              # load_config / PartCConfig / Base URL / 密钥保护
│   ├── models.py              # ModelRegistry + ModelInfo（合规校验、分档计价）
│   ├── errors.py              # 异常层级（PartCError → ... / BackendError → ...）
│   ├── schemas.py             # ChatMessage / LLMResult / TokenUsage / ExecResult
│   ├── budget_manager.py      # CostCalculator + BudgetManager（账本持久化）
│   ├── llm_client.py          # ZhizengzengLLMClient + SDKBackend / MockBackend
│   ├── code_executor.py       # PythonExecutor
│   └── logger.py              # StructuredLogger（JSONL）
├── configs/model_registry.json  # 合规白名单 + 已核实定价
├── tests/                       # pytest 离线套件（68 项）
├── examples/mock_agent.py       # C5 离线 Mock Demo
├── examples/integration_demo.py # 模拟 A/C/D 集成 Demo
├── examples/live_smoke_test.py  # 真实 API smoke test（仅 --live）
├── README.md / HANDOFF.md / MODEL_COMPLIANCE.md
├── LIVE_TEST_REPORT.md / INTEGRATION_READINESS.md / PART_C_FINAL_SUMMARY.md
├── pyproject.toml / .env.example / .gitignore
```

## 环境安装

要求 Python **3.9+**。

```bash
cd partc
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # 安装 openai + python-dotenv
pip install -e ".[test]"    # 追加 pytest
```

运行时依赖仅 `openai`（真实调用）与 `python-dotenv`（配置）；其余全部标准库。
**离线开发、Mock Demo 与全部测试无需 `openai`**（它在构造真实后端时才被惰性导入）。

## Zhizengzeng API 配置

```bash
cp .env.example .env
# 编辑 .env，填入 ZZZ_API_KEY=
```

```env
ZZZ_API_KEY=your_key_here
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
ZZZ_MODEL=deepseek-v4.1-flash-1017
PARTC_BUDGET_RMB=20
PARTC_LOG_DIR=./logs
PARTC_STATE_DIR=./.partc_state
PARTC_TIMEOUT_SECONDS=120
PARTC_MAX_RETRIES=3
```

代码内强制：

* Base URL **归一化为单个 `/v1`**（不会出现 `/v1/v1`，也不带 `/chat/completions`）；
* 密钥**只**从 `ZZZ_API_KEY` 读取（忽略 `OPENAI_API_KEY` 等其他工具的变量）；
* 密钥**永不记录**（配置转储与日志中都脱敏）；
* `.env` 已加入 `.gitignore`，密钥绝不硬编码。

## 合规模型配置

Part C 只允许 **Qwen / DeepSeek、官方发布时间 ≤ 2026-10-21、有官方标准 API 定价** 的模型；
未核实的模型一律 fail-closed（不可在计费模式启用）。

当前**已启用**（`allowed=true`）：

| requested_model_id | 输入价 (¥/1M) | 输出价 (¥/1M) | 说明 |
|---|---|---|---|
| `deepseek-v4.1-flash-1017`（默认） | 2.0（峰值/cache-miss） | 8.0（峰值） | 默认候选 |
| `qwen3.7-plus-2026-05-26` | 2.0 (≤256K) / 6.0 (≤1M) | 8.0 / 24.0 | 上下文分档 |
| `qwen3.7-flash-2026-07-15` | 0.2 / 0.6 / 1.2（≤32K/≤256K/≤1M） | 0.8 / 2.4 / 4.8 | 上下文分档 |
| `qwen3.6-plus-2026-04-02` | 2.0 (≤256K) / 8.0 (≤1M) | 12.0 / 48.0 | 上下文分档 |

候选（未启用）：`deepseek-v4-pro-0813`、`qwen3.8-flash`；不合规示例：`qwen4-max-preview`。
详情与来源见 `MODEL_COMPLIANCE.md`。

切换模型：改 `.env` 的 `ZZZ_MODEL`，或在 `chat(model=...)` 传入上面的 `requested_model_id`。
**非白名单/未核实模型会抛 `ModelComplianceError`**，不会静默退回默认模型。

## Mock 模式 vs 真实 API 模式

| | Mock 模式（默认，离线） | 真实 API（显式） |
|---|---|---|
| 后端 | `MockBackend`（确定性、零网络、零费用） | `SDKBackend`（`openai` SDK） |
| 触发 | `mock=True` 或注入 `MockBackend` | 不传 mock、配置 `ZZZ_API_KEY` |
| 示例 | `examples/mock_agent.py`、`integration_demo.py` | `examples/live_smoke_test.py --live` |
| 费用 | 无（但仍按计价器演练） | 真实计费，需显式 `--live` |

## 使用示例

### 1. LLM Client（C1）

```python
from partc import (
    load_config, ModelRegistry, BudgetManager, StructuredLogger,
    ZhizengzengLLMClient, ChatMessage,
)

cfg = load_config()
registry = ModelRegistry.load_default()
budget = BudgetManager(".partc_state/ledger.json", default_budget_rmb=cfg.budget_rmb)
logger = StructuredLogger(cfg.log_dir, api_key=cfg.api_key)

client = ZhizengzengLLMClient(cfg, budget_manager=budget, logger=logger, model_registry=registry)

result = client.chat(
    [ChatMessage.system("You are an AutoML agent."),
     ChatMessage.user("Plan the next experiment.")],
    model="deepseek-v4.1-flash-1017",   # 合规白名单 ID（可省略，用 ZZZ_MODEL）
    max_tokens=1024,
    task_id="task_001",
    run_id="run_001",
)

print(result.request_status)          # success | success_uncertain | budget_exceeded | error
print(result.output_text)             # 模型返回文本
print(result.usage.prompt_tokens, result.usage.completion_tokens)
print(result.calculated_cost_rmb)     # 本次费用（Decimal 字符串，None=未知）
print(result.cumulative_cost_rmb)     # 任务累计费用
print(result.error_type)              # 出错时的稳定错误码（None=正常）
print(budget.summary("task_001"))     # 任务预算汇总
```

错误处理约定：

* 模型不合规 → 抛 `ModelComplianceError`（硬错误）；
* 预算不足 → 返回 `request_status="budget_exceeded"`（不抛异常、不发请求）；
* API/网络/超时/限流 → 内部重试，最终失败返回 `request_status="error"` + `error_type`。

### 2. PythonExecutor（C3）

```python
from partc import PythonExecutor

ex = PythonExecutor(timeout_seconds=10, cwd=".", sanitize_env=True)

r = ex.execute_code("print('hello')", task_id="t1")
# r.status in {"success","runtime_error","timeout","execution_error"}
# r.exit_code / r.stdout / r.stderr
# r.stdout_path / r.stderr_path   （产物已落盘，路径返回在结果中）
```

注意：macOS `subprocess` **不是安全沙箱**。`shell=True` 永不用；`ZZZ_API_KEY` 等敏感
环境变量在 spawn 前被剥离；超时用进程组（`os.killpg`）整体击杀，不残留子进程。

### 3. BudgetManager（C2）

```python
from partc import BudgetManager

bm = BudgetManager(".partc_state/ledger.json", default_budget_rmb="20")
bm.ensure_task("task_001", budget_rmb="20")

# 生命周期由 LLMClient 自动驱动；也可手动调用：
bm.reserve("task_001", "call_1", "deepseek-v4.1-flash-1017", "peak", "0.02")
bm.settle("task_001", "call_1", input_tokens=37, output_tokens=14, cost_rmb="0.000186")
# bm.release(...)  /  bm.mark_uncertain(...)

print(bm.summary("task_001"))           # spent / held / available / over_budget
print(bm.cumulative_cost("task_001"))   # 已结算累计
```

账本保存在 `state_dir/ledger.json`，重启后自动恢复。

## 运行示例

```bash
# 全离线 Mock Agent Demo（打印任务汇总 JSON）
python examples/mock_agent.py

# 模拟 A/C/D 集成 Demo（离线，验证 C 的公开 API）
python examples/integration_demo.py

# Live Smoke Test：dry-run（不发请求）与真实请求
python examples/live_smoke_test.py               # dry run
python examples/live_smoke_test.py --live        # 真实、计费、一次请求（需 ZZZ_API_KEY）
```

## 测试结果

```bash
pytest            # 全离线，无 API 费用 — 68 passed
```

| 验收 | 结果 |
|---|---|
| `pytest` | ✅ 68 passed（0 失败，全离线） |
| Mock Agent Demo | ✅ 2 次 LLM、费用 ¥0.000452、执行 success、budget_exceeded 演示触发 |
| 模拟 A/C/D 集成 Demo | ✅ 执行 success/exit 0、accuracy 1.0、LLM success、费用 ¥0.000166 |
| Zhizengzeng Live Smoke Test | ✅ success、37/14 tokens、费用 ¥0.000186（2026-11-14，见 LIVE_TEST_REPORT.md） |

## 已知限制

1. **执行器不是沙箱**：不可信/陌生代码不应视为安全执行；V1 不部署 Docker。
2. **后端真实权重版本未独立验证**：Live Test 的 `resolved_model_version` 来自本地注册表；
   智增增后端部署版本需进一步核实（但已确认网关接受该模型并返回正常）。
3. **真实 API 仅验证了默认 DeepSeek 模型**：Qwen 系列与失败/重试/长上下文路径未走真实调用。
4. **计费口径待课程确认**：采用峰值价 + 输入 cache-miss 的保守口径；峰/谷价取定、
   历史价格与汇率换算以课程/组员裁定为准（见 HANDOFF.md）。
5. **单任务串行假设**：`BudgetManager` 用进程内锁保护，V1 假设单任务被串行驱动。