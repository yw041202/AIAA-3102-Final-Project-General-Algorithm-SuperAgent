# 给 A 的整合入口（V0.2，2026-11-17）

本地的 B→D→C 已通过；A 预计 11 月 19 日接入自主决策。此文只说明工具调用，不代替 A 的实验规划或最终选型。

## 安装与目录

团队仓库预期包含 partb/、partc/、partd/。使用 C 实际启动训练的同一解释器安装三者，或传源码根目录。D 已固定 scikit-learn 1.4.2；新产物记录全部实际依赖版本。依赖不一致时预测会明确拒绝。

```powershell
python -m pip install -e ./partb
python -m pip install -e ./partc
python -m pip install -e ./partd
```

本机无需安装即可复现，参见 README 的 bdc_acceptance.py 命令。D 的数据/产物路径都支持请求相对路径；交给 C 的命令行路径请使用绝对路径。

## 发起实验

A 为每个新实验写一份 JSON，字段见 HANDOFF 和 schemas/train_request.schema.json。其中：

- data.source=partb，config_path 指向 B 配置，data_root 指向原始数据；preprocess_options 由 A 决定。
- preprocessing=partb，防止第二次拟合 D 的数值处理。
- model.name/params、seed、metrics 由 A 给出；同一实验目录不覆盖，重试换新的 experiment_id。
- task_id 是同一任务的稳定编号，run_id 是运行编号，experiment_id 是候选编号；任务名称不用于切换专用答案。

```python
from pathlib import Path
from partd.c_executor import execute_via_c

request_path = Path("runs/request_001.json").resolve()  # A 已生成的实际请求
response = execute_via_c(
    ["train", "--request", str(request_path)],
    workdir=Path("runs/execution_001").resolve(),
    task_id="task_001", run_id="run_001",  # 必须与请求一致
    # 若 C 未安装：partc_root=Path("partc").resolve()
)
if response["status"] == "success":
    result = response["partd_result"]
    metric = result["primary_metric"]  # name/value/direction
    candidate = result["artifact"]["directory"]
    # 交给 A 的自主控制器决定继续、停止及选择；D 不自动挑最终模型。
else:
    failure = response["partd_result"] or response["execution"]
    # 交给 A 的失败处理；有效 D 错误含 stage、编号、error.type/message。
```

也可直接调用 `partd.train(request_path)`，或 CLI `python -m partd train --request ...`。直接调用者承担执行、超时与日志责任。

## A 选择候选后进行预测

选定的候选目录包含 B 处理器、任务配置和 D 模型。只传该目录和预测数据根目录，不需要训练/验证文件：

```python
response = execute_via_c(
    ["predict", "--artifact", str(Path(selected_candidate).resolve()),
     "--data-root", str(Path(prediction_data_root).resolve()),
     "--split", "test", "--output", str(Path("runs/predictions.csv").resolve()),
     "--include-probabilities"],
    workdir=Path("runs/prediction_execution").resolve(),
    task_id="task_001", run_id="run_001",
)
```

默认使用候选内的 task_config.json。新数据位置/文件名不同可提供 --config，但模态和特征列/顺序须匹配。B 未安装时传 --partb-root。预测不 fit、不读标签、不返回测试评分；按公告要求检查输出列名、顺序和标签类型。D 默认 sample_id/prediction，概率列编号映射见同名 metadata JSON。

## 返回和日志

- C 执行成功和 D status=success 同时满足，response.status 才为 success。
- 非法配置、数据或训练失败：D 写结构化错误，退出码 2；C 返回 runtime_error。
- 适配器显式写一次 C execution 日志，A 不要对相同 execution_id 重复记录。
- D 每个候选写 request/result/manifest、验证预测和 B 审计，含版本与代码指纹。
- C 原版 Windows 超时清理问题仍由 C 跟进；本次不声称已修复。

## 整合日建议验收

1. A 自己生成两份不同的候选配置，实际调用 D，按验证结果反馈继续或停止。
2. 确认失败记录能进入 A 的后续规划，不能中途由人替系统决定。
3. A 自动给出最终候选编号并触发预测；保存决策与产物引用。
4. B 独立核对样本 ID、标签、划分和处理状态，重算一个指标并恢复预测。
5. 官方任务指标与输出格式按公告落实；模拟例子的固定请求不算正式自主实验。
