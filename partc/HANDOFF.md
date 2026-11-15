# HANDOFF — Part C V1 交接清单

Part C 已完成、可独立使用，并已通过一次真实 API smoke test。本文给出 A、D 接入的最小
可运行示例，并列出尚需全组确认的问题（区分**必须确认**与**可后续优化**）。

---

## 当前状态（更新至 2026-11-14）

- ✅ **真实 API 已验证**：用户于 2026-11-14 执行 `python examples/live_smoke_test.py --live`
  成功（`request_status=success`，`deepseek-v4.1-flash-1017`，37/14 tokens，费用 ¥0.000186）。
  详见 `LIVE_TEST_REPORT.md`。
- ✅ **LLM `usage` 主要字段已获真实响应验证**：`prompt_tokens` / `completion_tokens` /
  `total_tokens` / `reasoning_tokens` / `cached_tokens` / `finish_reason` 均已确认解析；
  reasoning 计入 completion、不重复计费。
- ✅ **68 项离线 pytest 通过**（本轮已重新运行确认）；Mock Demo 与模拟 A/C/D 集成 Demo 均通过。
- ✅ **模拟 A/C/D 集成已完成**：`examples/integration_demo.py`（离线）。
- ⚠️ **后端真实权重版本仍需进一步验证**：`resolved_model_version` 为本地注册表映射，
  不等于独立认证智增增后台部署版本。
- ⚠️ **历史价格口径 / 峰谷价取定 / 汇率仍需课程确认**（见下文）。
- ⚠️ **Code Executor 不是不可信代码的安全沙箱**（macOS subprocess，V1 不部署 Docker）。
- 🗑️ 已无「缺少 `ZZZ_API_KEY`」这一阻塞项（真实密钥已可用并验证）。

---

## 对外接口约定（速览）

```python
from partc import (
    load_config, ModelRegistry, BudgetManager, StructuredLogger,
    ZhizengzengLLMClient, PythonExecutor, ChatMessage,
)

cfg = load_config()                                  # 读 .env / 环境变量
reg = ModelRegistry.load_default()                   # 合规白名单
budget = BudgetManager(".partc_state/ledger.json", default_budget_rmb=cfg.budget_rmb)
logger = StructuredLogger(cfg.log_dir, api_key=cfg.api_key)
client = ZhizengzengLLMClient(cfg, budget_manager=budget, logger=logger, model_registry=reg)

res = client.chat([...messages...], model=..., max_tokens=..., task_id=..., run_id=...)
# res: LLMResult
#   res.request_status ∈ {success, success_uncertain, budget_exceeded, error}
#   res.output_text / res.usage.{prompt_tokens,completion_tokens,total_tokens,extra}
#   res.calculated_cost_rmb / res.cumulative_cost_rmb
#   res.pricing_version / res.error_type / res.retry_count / res.resolved_model_version
```

要点：

- 模型不合规 **抛 `ModelComplianceError`**（硬错误）；别名解析为固定版本 ID 上送。
- 预算不足返回 `request_status="budget_exceeded"`（不抛异常、不发请求）。
- API/网络/超时/限流内部重试后失败返回 `request_status="error"` + `error_type`。
- 账本按 `task_id` 独立，从 `state_dir/ledger.json` 重启恢复。
- Qwen 按上下文长度分档计价；超档 fail-closed（`pricing_tier_unresolved`）。
- 执行器返回 `ExecResult.status ∈ {success, runtime_error, timeout, execution_error}`。

---

## 与 A 的对接（最小可运行示例）

```python
from partc import (
    load_config, ModelRegistry, BudgetManager, StructuredLogger,
    ZhizengzengLLMClient, ChatMessage,
)
from partc.errors import ModelComplianceError

cfg = load_config()
registry = ModelRegistry.load_default()
budget = BudgetManager(".partc_state/ledger.json", default_budget_rmb=cfg.budget_rmb)
logger = StructuredLogger(cfg.log_dir, api_key=cfg.api_key)
client = ZhizengzengLLMClient(cfg, budget_manager=budget,
                              logger=logger, model_registry=registry)

TASK_ID = "task_001"

def ask_llm(prompt: str, run_id: str):
    # 模型不合规是硬错误：由上层决定是否终止，不应静默换模型
    try:
        res = client.chat(
            [ChatMessage.system("You are an AutoML agent."),
             ChatMessage.user(prompt)],
            model="deepseek-v4.1-flash-1017",   # 或 cfg.model（合规白名单 ID）
            max_tokens=1024,
            task_id=TASK_ID,
            run_id=run_id,
        )
    except ModelComplianceError as exc:
        raise   # 交给上层：记录并停止，rather than fallback

    if res.request_status == "budget_exceeded":
        # 停止当前任务；读取剩余预算决定下一步
        s = budget.summary(TASK_ID)          # {"available_rmb": "...", "spent_rmb": "...", ...}
        raise SystemExit(f"budget exceeded, remaining={s['available_rmb']}")

    if res.request_status == "error":
        # 按稳定错误码分类（api_timeout / rate_limit / network_error / server_error / auth_error ...）
        return {"error": res.error_type}

    # 成功：结构化消费
    return {
        "output": res.output_text,
        "prompt_tokens": res.usage.prompt_tokens,
        "completion_tokens": res.usage.completion_tokens,
        "cost_rmb": res.calculated_cost_rmb,      # 本次费用（None=不确定）
        "cumulative_rmb": res.cumulative_cost_rmb,  # 任务累计
        "pricing": res.pricing_version,           # 如 "peak" 或分档标签
    }

# 任务级费用与日志（供 A 账务/汇报）
total_cost = budget.cumulative_cost(TASK_ID)   # Decimal
logs       = logger.llm_records(TASK_ID)       # list[dict]（已脱敏）
summary    = logger.llm_summary(TASK_ID)       # n_llm_calls / n_success / 合计 token
```

---

## 与 D 的对接（最小可运行示例）

```python
from partc import PythonExecutor

ex = PythonExecutor(
    timeout_seconds=120,             # 每个脚本的运行上限
    cwd="runs/task_001",             # 工作目录（可用 allowed_roots 限制范围）
    output_dir="runs/task_001/out",  # stdout/stderr 落盘位置
    sanitize_env=True,               # 剥离 ZZZ_API_KEY 等，子进程不继承密钥
)

# 方式一：运行已存在的脚本文件
r = ex.execute_file(
    "training/train.py",
    args=["--data", "data.csv"],
    task_id="task_001",
    run_id="run_001",
)

# 方式二：直接运行受控代码字符串
# r = ex.execute_code(code, task_id="task_001", run_id="run_001")

print(r.status)       # success | runtime_error | timeout | execution_error
print(r.exit_code)     # 0 或非 0（timeout/execution_error 时可能为 None）
print(r.stdout)        # 截断后的 stdout 文本（≤1MiB）
print(r.stderr)        # 截断后的 stderr 文本
print(r.stdout_path)   # 完整 stdout 文件路径
print(r.stderr_path)   # 完整 stderr 文件路径
print(r.elapsed_seconds, r.timeout_seconds)

# 契约：D 的训练脚本把指标（如 accuracy）以 JSON 行打印到 stdout，
#       A/D 从 r.stdout 解析（见 examples/integration_demo.py 的 read_metric）。
```

安全说明：执行器**不是沙箱**，`shell=True` 永不用、超时按进程组整体击杀、
敏感环境变量剥离；但**不可信/陌生代码不能被当作安全执行**，需全组约定“可信受控入口”
与“任意/Agent 生成代码”的分级（见下文）。

---

## 尚需全组确认的问题

### 必须确认（影响计费/合规正确性）

1. **DeepSeek 峰/谷价取定**：当前默认按**峰值**（保守）。课程“标准价”是否以峰值计？
2. **缓存命中是否计费**：当前输入一律按 **cache-miss** 全价（缓存折扣被排除）。请复核。
3. **历史价格口径**：以 2026-10-21 官方标准价为基准；若课程要求历史某一时点价格，需再校准。
4. **币种与汇率**：课程以 RMB 计；DeepSeek 官方同时列示 USD/RMB，是否需汇率换算。
5. **后端真实部署版本**：智增增上 `deepseek-v4.1-flash-1017` 的实际后端版本需一次核验
   （例如核对输出行为 / 平台说明）；Live Test 仅证明“可用并回显该 ID”。
6. **不可信代码安全策略**：Agent 或 D 生成的代码如何分级（可信入口 vs 任意代码）；
   V1 不提供沙箱，也不宜在无沙箱时运行任意代码。

### 可后续优化（不阻塞集成）

1. Task/Run ID 生命周期统一规则（当前 `task_id` 默认 `default_task`，`run_id` 自动生成）。
2. 是否需要 `finish_reason`、raw usage、结构体 JSON 输出 / tool-calling 等额外元数据。
3. 训练产物目录与命名约定（当前 `output_dir`/`cwd` 可配置但无统一规范）。
4. 长时间训练的任务级资源/超时策略；QoS 或排队机制。
5. 是否需要在失败（`error`）与不确定（`success_uncertain`）状态上叠加人工确认流程。

> 本工具从未把 Mock 结果描述为真实 API 结果。真实计费与验证仅由 `--live` 显式触发。