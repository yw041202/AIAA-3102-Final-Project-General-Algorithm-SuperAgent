# Part D 训练验证与独立预测

提供数值、B 表格/文本与可选冻结图像的单候选实验工具，完成训练、指定验证指标、模型保存加载及独立预测，并接入 C 执行器。A 决定实验配置、下一步动作和最终候选。

图像入口沿用 train/predict，model.name=frozen_vision，可选 ResNet18 或 EfficientNet-B0，以及 LR/SVM 分类头。特征缓存、反馈、可调用搜索建议、权重来源和 A/B/C 对接见 [图像交付说明](VISION.md)。当前 A 版本还需接入图像分支；D/B/C 验收不等于完整 Agent 自主运行。

入口：[图像接口与策略](VISION.md)、[通用交接](HANDOFF.md)、[本次验收](ACCEPTANCE_VISION.md)。原 [A 整合说明](A_QUICKSTART.md)、[节点清单](MILESTONES_20261117.md) 和 [V0.2 验收记录](ACCEPTANCE_20261117.md) 保留为历史记录。

## 完整 B/D/C 复现

先在本目录准备环境及固定版本的 B/C。它们尚未合并 main 时也可以独立复现；团队已提供匹配版本时可复用。以下命令无需 API 密钥，不调用收费模型。

```powershell
python -m pip install -r requirements-partb.txt
python -m pip install pytest
git clone --branch partb/data-processing https://github.com/azhao258/AIAA-3102-Final-Project-General-Algorithm-SuperAgent.git .deps/partb
git -C .deps/partb checkout --detach 1fbb9c51da9dfa256d1b4078b6b24ed86a032ba1
git clone --branch feat/partc-v1 https://github.com/yw041202/AIAA-3102-Final-Project-General-Algorithm-SuperAgent.git .deps/partc
git -C .deps/partc checkout --detach 877625a2216d2a6bd5410e860ccf8006baea5748
```

然后运行，输出目录必须尚不存在：

```powershell
$dPython = 'python'
$env:PYTHONIOENCODING = 'utf-8'
& $dPython examples/bdc_acceptance.py --output runs/my-bdc-demo --partb-root .deps/partb --partc-root .deps/partc/partc
```

该示例真实调用 B、C，表格和文本各完成训练、验证、保存、新进程恢复、原始输入独立预测及预期失败，共 14 个独立子进程。预测前移走模拟训练/验证文件，证明不依赖它们或重新 fit。

B/C 快照分别来自 1fbb9c51da9dfa256d1b4078b6b24ed86a032ba1、877625a2216d2a6bd5410e860ccf8006baea5748，源码未修改。B 根目录参数接受含 partb 的仓库，也接受包目录；C 参数指向含 src 的目录。团队合并后使用安装的 B/C 或实际源码位置。

## 兼容：数值 CSV 复现

历史验收环境为 Python 3.12.4，命令中的 python 应指向自己的训练环境。依赖见 requirements.txt；无需安装 D 包即可从源码启动。

在本目录执行：

```powershell
$dPython = 'python'
$env:PYTHONIOENCODING = 'utf-8'
& $dPython examples/acceptance_demo.py --output runs/my-first-demo --partc-root .deps/partc/partc
```

`--output` 必须是尚不存在的目录，以保护已有记录。再次运行请换目录名。

这条命令会生成 180 条训练、60 条验证、30 条无标签预测输入，依次启动六个独立子进程：直接训练、直接预测、加载后重算验证预测、通过 C 训练、通过 C 预测，以及一次预期失败的 C 调用。最后核对指标、标签、概率、错误状态和执行日志，生成 `acceptance_summary.json`。

C 参考提交为 877625a2216d2a6bd5410e860ccf8006baea5748，源码未改动。上面的克隆目录对应 .deps/partc/partc；团队合并后可将 --partc-root 改为相邻 ../partc。该参数必须指向含 src 的目录。

如果另建环境，使用 Python 3.10–3.12，安装依赖并选择相同解释器运行 D 和 C：

```powershell
python -m pip install -r requirements-partb.txt
python -m pip install pytest
# B 应安装在同一解释器，或通过 partb_root / --partb-root / PARTB_ROOT 定位
python -m pip install -e .deps/partb/partb
# 可选：安装为包后支持 partd 命令或 python -m partd
python -m pip install -e . --no-deps --no-build-isolation
```

## 分步使用

运行上述 Demo 后，已有请求、输入和候选模型。以下示例创建新的预测文件：

```powershell
& $dPython run_partd.py predict --artifact runs/my-first-demo/artifacts/synthetic_demo/development/logistic_direct --input runs/my-first-demo/prediction.csv --output runs/my-first-demo/predictions-again.csv --include-probabilities
```

训练调用为 `run_partd.py train --request <请求JSON路径>`。每次新实验应使用新的 `experiment_id`；相同实验目录已存在时明确失败，不覆盖旧模型或旧结果。具体接口见 [HANDOFF.md](HANDOFF.md)。

程序标准输出最后一行是可解析 JSON，包含 `event="partd.result"`。成功退出码为 0，配置或运行失败为 2。JSON 使用 ASCII 转义传输中文路径，避免 Windows 重定向编码导致路径损坏；读取 JSON 后仍得到原始中文。

## 支持的能力

- 表格/文本候选：`logistic_regression`、`random_forest`；图像候选 `frozen_vision` 的骨干、LR/SVM 头及参数由请求指定。本次图像交付不包含另行开发的表格/文本模型扩展。
- 指标：accuracy、balanced_accuracy、f1_macro、precision_macro、recall_macro、roc_auc、log_loss。主指标可配置，结果标明 maximize/minimize。ROC AUC 和 log loss 使用概率；二分类正类为保存的 `classes[1]`。
- 默认 numeric 预处理：仅在训练部分拟合的中位数填补；逻辑回归另有标准化。拟合后的处理和分类器作为同一个 Pipeline 保存。
- partb 模式：调用 B 的加载与 train-only 预处理，保留 CSR，不重复拟合 D 数值处理；保存 B 处理器、来源、任务配置和 D 模型，独立预测仅 load/transform。
- passthrough 模式：兼容外部已处理的数值 CSV；完整 B 原始输入预测请使用 partb 模式。
- 严格区分训练/验证输入，不接受训练请求中的 test_csv；检查重复及跨划分重叠的样本 ID。
- 保存输入哈希、运行/实验编号、配置、依赖版本、D 源码指纹、训练警告和指标；失败尽可能保存结构化记录。
- 新进程加载后预测，不需要训练集、验证集或 C/LLM API。
- 模型文件哈希核对、按列名恢复特征顺序、类别概率列映射、已有输出防覆盖。

## 每次训练的产物

```text
output_dir/task_id/run_id/experiment_id/
  request.json                 原始实验请求
  result.json                  指标、状态、警告、产物引用
  artifact.json                类别/特征、版本、模型哈希、训练来源
  model.joblib                 拟合后的完整预处理与分类器
  validation_predictions.csv   验证集的ID、真值、预测和概率
  preprocessor.joblib          partb 模式：B 的完整处理状态
  preprocessor.joblib.json     partb 模式：拟合来源
  task_config.json             partb 模式：输入结构配置
  data_audit.json              partb 模式：训练/验证审计
```

预测生成 CSV 和同名 `.metadata.json`，记录类别概率映射、模型/输入哈希和预测数量。交付时可以整体移动实验目录，预测不依赖原训练文件位置。

模型使用 joblib 保存，只加载可信来源产物。哈希用于发现损坏，不代表安全认证。加载时检查记录的 Python/NumPy/scikit-learn/joblib；partb 模式另检查 pandas/SciPy/Pillow、B 源码指纹、处理器和配置哈希。旧 V0.1 数值产物仍可加载。

预处理训练/预测一致性和模型持久化依据：[scikit-learn 数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)、[模型持久化说明](https://scikit-learn.org/stable/model_persistence.html)。实现与验收使用本机 1.4.2。

## C 的接入范围

`src/partd/c_executor.py` 调用 C 原有的 `PythonExecutor.execute_file`，使用绝对入口路径及参数列表，再显式调用 `StructuredLogger.log_execution`。返回值同时保留 C 的执行结果与解析出的 D 实验结果。C 日志 ID 与 D 请求 ID 不一致时拒绝启动。

- 已覆盖：训练成功、预测成功、训练配置失败的非零退出与结构化错误。
- 未修改 C 的计费模块，也不调用 LLM。
- 超时与子进程清理由实际部署的 C 执行器负责，需在目标环境交叉验收。骨干提取与最终原图预测都应给足超时。

## 测试

完整历史模型兼容性验收使用 Python 3.12.4 和记录的固定依赖；其他支持版本可训练新模型，但历史产物的版本检查会拒绝不匹配的环境。干净检出先创建测试临时目录的父目录：

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PARTB_ROOT = (Resolve-Path .deps/partb).Path
$env:PARTC_ROOT = (Resolve-Path .deps/partc/partc).Path
& $dPython -m pytest -q -p no:cacheprovider --basetemp runs/pytest-check
```

使用项目内短临时路径，避免 Windows 路径长度限制。V0.2 本机验收为 35 项通过、无跳过，覆盖原功能、B 的 CSV/NPY/CSR、原始输入预测、训练隔离、无重新拟合、状态损坏，以及 B/D/C 的成功和失败。不设置 PARTB_ROOT/PARTC_ROOT 时相应联调测试跳过；正式验收均已设置。

## 当前边界

- B 公开接口已对齐；图像使用原图路径。请求约束见 schemas/train_request.schema.json。
- 图像通过 frozen_vision 候选接入统一 train/predict，A 的图像规划、反馈与最终完整自主运行仍待对接。
- 课程数据未包含在交付中；正式自主运行及官方输出格式仍按公告落实。
- D 可返回搜索建议但不替 A 进行最终选型；候选模型保存不表示已被选为最终模型。
- 当前验收是本机源码环境和独立进程验证，不等于 11 月 27 日的跨机器干净环境验收。

Git 分支只保留源码、契约、精简验证摘要和 V0.1 的小型兼容性测试模型；完整运行产物由 Demo 写到已忽略的 runs/。历史摘要中的本机目录已移除，分支中的生成文件均来自模拟数据。

详见 [最新验收](ACCEPTANCE_20261117.md)、[历史 V0.1](ACCEPTANCE.md)、[交接](HANDOFF.md) 和 [AI 声明](AI_USAGE.md)。
