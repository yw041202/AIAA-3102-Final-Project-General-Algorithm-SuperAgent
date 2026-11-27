# A 模块交付与真实任务交叉验收

运行及核验日期：2026-11-22。A 的决策、调度、反馈、停止、自动选型和模型导出已实现；两项老师发布的任务均已完成真实自主运行，等待组员交叉验收。本文件不代表团队已完成正式提交。

## 结果

| 任务 | 自动选中的模型 | 验证 accuracy | 测试 accuracy | 测试预测数 | 课程核算费用 |
| --- | --- | ---: | ---: | ---: | ---: |
| Coupon | Random Forest，400 棵树，experiment_006 | 73.8424% | 75.1281% | 2537 | ¥0.013216 |
| Headlines | 词级 TF-IDF difference + Logistic Regression，C=2，experiment_003 | 67.8723% | 65.8416% | 3232 | ¥0.009767 |

模型均由验证准确率自动选择，之后锁定选择并生成测试预测，最后才计算测试准确率。测试指标未反馈给规划器，也未据此追加实验。两次运行期间没有人工修改运行代码、指定下一候选或选择模型。

- Coupon：6 次规划，前两次分别因时间字段无法转数值、包含不支持的预处理字段失败。Agent 收到错误后自行修正，后四次训练成功；达到 6 轮上限结束。
- Headlines：6 次规划，第一次使用了不支持的预处理字段，后五次训练成功；连续三个候选未超过第 3 轮的验证分数，按 patience 结束。
- 规划器：学校 HPC AIGC 的 `DeepSeek-V4.1-Flash`。完整记录中保留模型版本映射、逐次 token、费用、失败和决策理由。LLM 的理由是候选假设，不能直接当成任务事实或科学结论。
- 两个任务均返回 `status: success`。`official_result: false` 表示尚未完成团队最终验收与冻结，不表示运行失败。

## 独立核验与计费

完整 A/B/C/D 测试：**142 passed，0 failed，0 skipped，295.36 秒**。同一环境中新进程重新加载两个保存模型，分别生成 2537 / 3232 条预测，与原始 CSV 逐字节一致。模型压缩包逐文件与这些已核验模型比较，内容完全相同。

| 项目 | 输入 token | 输出 token | 调用数 | 课程费用 |
| --- | ---: | ---: | ---: | ---: |
| Coupon | 10068 | 787 | 6 | ¥0.013216 |
| Headlines | 5987 | 945 | 6 | ¥0.009767 |
| 合计 | 16055 | 1732 | 12 | ¥0.022983 |

按老师通知允许的 DeepSeek 非高峰口径，输入每百万 token 1 元，输出 4 元，逐条重新计算并与日志和账本一致。独立连通性测试另有一次成功调用 ¥0.000059，以及一次网络错误的未知用量预留 ¥0.000888，均保留在独立账本中。已知课程核算总额 ¥0.023042，加保留预留共 ¥0.023930，低于用户授权的总验证预算 50 元。以上不是学校账户实际扣款；未读取账户账单。两个任务各自的持久预算上限为 20 元。

仍须 C 保存并确认截至 2026-10-21 的官方价格证据。当前网页核验和模型映射说明见 [学校 API 文档](HKUST_API_ZH.md)，不得把当前网页当成历史快照。

## 可交叉核验的证据

- [机器核验结果](evidence/released-20261122/verification.json)、[142 项测试 XML](evidence/released-20261122/test-results.xml)、[文件 SHA-256 清单](evidence/released-20261122/SHA256SUMS.json)。
- Coupon：[完整摘要](evidence/released-20261122/coupon/summary.json)、[预测 CSV](evidence/released-20261122/coupon/predictions.csv)、[模型及预处理 ZIP](evidence/released-20261122/coupon/final_model.zip)。
- Headlines：[完整摘要](evidence/released-20261122/headlines/summary.json)、[预测 CSV](evidence/released-20261122/headlines/predictions.csv)、[模型及预处理 ZIP](evidence/released-20261122/headlines/final_model.zip)。
- 每项任务目录还包含 decisions.jsonl、requests/、results/、execution/、llm_logs/ 和最终评估。失败候选没有删除。
- [任务账本](evidence/released-20261122/task-budget-ledger.json)、[连通性账本](evidence/released-20261122/connectivity-budget-ledger.json)。原始数据与 API key 不在提交中。

同一生产源代码指纹：`98a391de36c6a588a1b5a31f561ae029ce94ced830c72e772c23f308eac5494d`。该指纹覆盖 A/B/C/D 的 Python 源码（不包括测试），配置另保存在运行证据中。`.gitattributes` 固定相应换行符，以保持新检出源文件与模型记录的字节指纹一致；证据文件也保持原始字节。

运行记录中的绝对路径是原始运行环境信息，不要求组员创建同名路径；复现时通过 `--data-root` 指向本机老师数据。

## 组员独立复现

先按根 README 安装依赖。保存模型记录的环境为 Python 3.11.5、NumPy 1.26.4、scikit-learn 1.4.2、joblib 1.4.2、pandas 3.0.6、SciPy 1.17.1、Pillow 12.3.0。D 严格检查环境与来源，不应为方便复现而关闭检查。

以下从仓库根目录运行，两个 `$data` 值请替换为老师数据在自己机器上的路径。只复现预测无需 API，也不会新增模型费用。

```powershell
Expand-Archive -LiteralPath parta/evidence/released-20261122/coupon/final_model.zip -DestinationPath runs/review-coupon
$data = '../final_project_tasks/final_project_tasks/in_vehicle_coupon_recommendation'
.\.venv\Scripts\python.exe -m partd predict --artifact runs/review-coupon/final_model --data-root $data --output runs/review-coupon/predictions.csv
(Get-FileHash runs/review-coupon/predictions.csv).Hash -eq (Get-FileHash parta/evidence/released-20261122/coupon/predictions.csv).Hash

Expand-Archive -LiteralPath parta/evidence/released-20261122/headlines/final_model.zip -DestinationPath runs/review-headlines
$data = '../final_project_tasks/final_project_tasks/headlines_ctr_clean'
.\.venv\Scripts\python.exe -m partd predict --artifact runs/review-headlines/final_model --data-root $data --output runs/review-headlines/predictions.csv
(Get-FileHash runs/review-headlines/predictions.csv).Hash -eq (Get-FileHash parta/evidence/released-20261122/headlines/predictions.csv).Hash
```

两个比较预期均为 `True`。目录应使用尚不存在的新目录。完整自主重跑命令见 [README](README.md)，需 API；再次运行必须更换 run_id，同时保留 task_id、output_root 及累计预算账本。LLM 决策有随机性，重跑不保证选择同一模型或得到相同分数。

## 本次 PR 的交叉验收分工

请每位验收人在 PR 留下「使用的 commit、环境、执行命令、实际结果、通过/阻塞及原因」，不要只回复“看过了”。

| 验收人 | 独立执行任务 | 预期证据 |
| --- | --- | --- |
| B / @azhao258 | 对照老师说明确认两项数据配置、划分、标签及预测顺序；检查预处理只在 train 拟合、选择前不加载测试集；核验 B 源码没有被集成改写 | 数据配置核对结果、测试隔离检查、样本数 2537 / 3232、发现的问题 |
| C / @ywu044 | 复核学校 API 适配、模型映射与课程日期价格依据；根据全部 12 条日志独立重算 token 和费用；检查未知调用预留、无隐式重试、预算停止及 Windows 超时测试 | 两项费用与账本一致、模型/价格依据、相关测试结果及 API 改动意见 |
| D / @Capry26 | 按上述命令在新进程加载两份模型并核对预测哈希；检查选择由验证指标驱动、测试分数仅在锁定后计算；按 README 启动自主流程或离线 demo 检查无需人工决策 | 两次预测比较 True、输出数量、最终实验 006 / 003、主流程验收意见 |

## 还需全组共同完成

1. 完成交叉验收，解决阻塞问题并确定最后提交的统一代码版本；如果修改影响运行的代码或配置，须用新版本完整自主重跑报告中的结果，累计费用不能清零。不得基于已经看到的测试分数选择改动。
2. D 与 B 确认图像训练能力的剩余缺口。当前仅支持表格/文本与 LR/RF，图像输入会明确拒绝；这会限制未见任务的覆盖范围。
3. C 补齐历史定价证据。全组确认课程正式截止时间、最终交付格式，归档代码、模型、日志及费用。
4. 每人基于自己的技术工作共同完成 report.pdf、presentation 与 AI 工具使用声明，并能解释实现及实验。A 部分可使用本文件、README 中的流程图和决策记录。

本分支整合 B 分支与 C/D 已有 PR 的固定版本，不修改他们的远端分支、不代替组员批准或合并原 PR。完整来源 SHA 见 [初始检查记录](REVIEW_AND_PLAN_ZH.md)。
