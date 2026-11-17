# Part D V0.2 验收：B 数据接入、完整预测与 C 执行

日期：2026-11-17（北京时间）。本轮真实执行代码、使用模拟输入；无收费 LLM 调用，无课程测试集。

## 结果

- 35 项测试全部通过，0 失败、0 跳过，包含误放参数/拼错字段拒绝，避免 A 的实验配置被静默忽略。
- 正式训练/预测入口接入真实 B 和 C；表格、文本两条路径共 14 个独立子进程全部完成预期行为。
- 保存前后的验证预测、类别概率及样本顺序一致；直接执行与 C 执行结果一致。
- 完整候选目录复制后，移走模拟训练/验证文件，仍能在新进程处理 20 条原始预测输入。
- 无效 B 参数由 D 转为带阶段和编号的错误，C 返回 runtime_error、退出码 2；这是预期失败验证。
- 另覆盖了 NPY 类别输入、无标签 NPY 预测、文本拼接/差分与旧 V0.1 数值产物兼容。
- 构建并安装 V0.2 wheel 到独立目标目录，在源码目录外、Python 隔离模式中导入安装包；两种模态的直接预测及 C 子进程预测均与源码版逐项一致。复用本机数值依赖，不称为干净环境验收。
- JSON Schema 自身合法，六份新旧有效请求均通过契约校验；运行时另由 D/B 检查语义。

这些是本地模块与集成验收，不是 A 的自主流程、正式课程结果或另一台电脑的干净环境验收。

## 固定版本

| 项目 | 版本 |
| --- | --- |
| D | 0.2.0 |
| D 运行源码 SHA256 | 02faffd46aac7a6a3cce21249627d8a660b45de5c76392d9bb8b73151fb0b170 |
| B 参考提交 | 1fbb9c51da9dfa256d1b4078b6b24ed86a032ba1 |
| B 三个核心文件组合指纹 | 0dad32bb325c4d26aedf44d61643acf37724ac75f1fdb3833f0f55280548d160 |
| C 参考提交 | 877625a2216d2a6bd5410e860ccf8006baea5748 |
| 实际 C 执行器 SHA256 | 0f7409c77e984a4c3e17a2db218d6d72f122eb1a09f61a6576aa17d192603b3e |
| 环境 | Windows，Python 3.12.4，NumPy 1.26.4，scikit-learn 1.4.2，joblib 1.4.2，pandas 2.2.2，SciPy 1.13.1，Pillow 10.3.0 |

B/C 源码未改动。以上验收在发布本 PR 之前于本地完成，未调用真实模型 API。

## 数据与检查

表格、文本分别使用 120 条训练、40 条验证、20 条无标签预测输入。表格含类别、缺失数值和验证新类别；文本为有序双列，验证专用词不会进入训练词表。B 只拟合训练部分，D 不再添加第二个填补器或缩放器，CSR 不转为大 dense 数组。

每个模态验证以下链路：

1. 直接 CLI 训练，以及 C 启动 CLI 训练；核对验证指标。
2. 新进程加载模型和 B 状态，重新处理原始验证输入，核对保存前后的标签和概率（阈值 1e-12）；独立统计 accuracy。
3. C 再运行相同验证预测，结果与直接调用一致。
4. 无效预处理参数返回 data_preparation 阶段错误，保留失败候选记录和执行日志。
5. 复制完整候选目录，移走训练和验证文件；直接调用与 C 分别对原始无标签预测输入生成 20 条结果，逐项一致。

模拟输入易分类，accuracy 为 1.0，仅证明链路，不作性能承诺。课程 test 文件在示例中不存在；训练隔离还通过了只允许 train/val 读取的针对性检查，预测禁止重新 fit 的检查也通过。

## 复现

工作目录为 project/partd，output 必须是新目录：

```powershell
$dPython = 'python'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PARTB_ROOT = (Resolve-Path .deps/partb).Path
$env:PARTC_ROOT = (Resolve-Path .deps/partc/partc).Path
New-Item -ItemType Directory -Force runs | Out-Null
& $dPython -m pytest -q -p no:cacheprovider --basetemp runs/reproduce-tests
& $dPython examples/bdc_acceptance.py --output runs/reproduce-bdc --partb-root $env:PARTB_ROOT --partc-root $env:PARTC_ROOT
```

历史完整结果位于原本机 runs/bdc-20261117-final。分支只提交 evidence/bdc-20261117/acceptance_summary.json、evidence/test-results-20261117.xml 与 evidence/installation-check-20261117.json；原本机目录字段已移除。完整模型、处理器、预测与日志可运行 Demo 重新生成，写入已忽略的 runs/。

## 待办

- A 的真实自主调度、最终选择和导出调用，按用户计划于 11 月 19 日整合。
- B/A 人工交叉复现、对字段和预测格式的确认。
- 真实课程数据与正式任务指标/提交格式；完整最终版本的自主运行和报告结果。
- 图像训练留待 11 月 24 日阶段；本版明确报错，不暗中展平或删样本。
- C 原版 Windows 超时清理及既有预算边界由 C 跟进；本轮覆盖正常执行和进程非零退出。

本地节点状态见 MILESTONES_20261117.md，给 A 的调用方式见 A_QUICKSTART.md。
