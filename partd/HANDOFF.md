# Part D 交接说明

状态：已有 B 表格/文本接口；新增图像统一 train/predict、冻结特征缓存及搜索建议。图像请求将 model.name 设为 frozen_vision，其完整配置、返回字段、策略与 A/B/C 接入步骤见 [VISION.md](VISION.md)。A 的图像分支尚待接入。下文原有表格/文本契约保持兼容。

## 推荐：A → D → B

```json
{
  "schema_version": 1,
  "task_id": "task_001", "run_id": "run_001", "experiment_id": "exp_001",
  "data": {
    "source": "partb", "config_path": "task_config.json", "data_root": "raw_data",
    "partb_root": "../partb",
    "preprocess_options": {"encoding": "onehot", "numeric_columns": []}
  },
  "preprocessing": "partb",
  "model": {"name": "logistic_regression", "params": {"C": 1.0}},
  "seed": 42,
  "metrics": {"primary": "accuracy", "additional": ["f1_macro", "log_loss"]},
  "output_dir": "artifacts"
}
```

路径相对于请求 JSON。partb_root 可省略，使用当前解释器安装的 B 或 PARTB_ROOT；显式参数接受 B 包目录或仓库根目录。preprocess_options 交给 B，每个实验新建处理器且只 fit(train)。数组/CSR 直接用于训练，不重复拟合 D 的数值处理。若 metrics 省略，主指标取 B 配置的 metric；正式实验由 A 明确传入公告指标。

每个候选保存 model.joblib、preprocessor.joblib、拟合来源 JSON、task_config.json、manifest 与审计。标签保持原类型和含义，classes 标明概率列映射。请求约束见 schemas/train_request.schema.json；D/B 另执行语义检查。

原始输入预测：

```powershell
python run_partd.py predict --artifact <候选目录> --data-root <原始数据根目录> --partb-root <B目录> --split test --output <新预测.csv> --include-probabilities
```

默认使用保存的任务配置，可用 --config 覆盖数据路径配置，但模态和特征顺序必须一致。--split 默认为 test；开发验收可用 val/prediction。课程 test 仅在 A 选定最终方案后使用。预测仅 load_prediction/transform，不读标签、不加载训练集、不 fit，不返回测试性能指标。

候选目录可整体移动；仍需可导入匹配版本的 B 及依赖。partb 模式用 --data-root，不用 --input；数值 CSV 继续用 --input。错误包含 type/message，训练失败另包含 stage 和有效的实验编号。

## 兼容入口：数值 CSV 实验请求

```json
{
  "schema_version": 1,
  "task_id": "task_001",
  "run_id": "run_001",
  "experiment_id": "exp_001",
  "data": {
    "train_csv": "train.csv",
    "validation_csv": "validation.csv",
    "id_column": "sample_id",
    "target_column": "label",
    "feature_columns": ["feature_0", "feature_1"]
  },
  "model": {"name": "logistic_regression", "params": {"C": 1.0}},
  "preprocessing": "numeric",
  "seed": 42,
  "metrics": {"primary": "accuracy", "additional": ["f1_macro"]},
  "output_dir": "artifacts"
}
```

- 请求内相对路径相对于该 JSON 文件的位置解释，与程序启动目录无关。
- task_id、run_id、experiment_id 用于追溯，不用于选择任务专用算法。它们应为适合目录名称的唯一标识。
- 不接受 test_csv。训练结果中的所有指标都来自 validation_csv。
- 主指标不存在、无法计算或名称不支持时返回明确失败，不虚构数值。
- 同一实验目录不重复使用。A 如需重试，应保留原失败记录并给新尝试新的 experiment_id。
- model 支持 logistic_regression / random_forest，参数遵循相应 scikit-learn 1.4.2 估计器接口。随机性由 seed 控制，冲突的 random_state 被拒绝。

## 兼容的数值 CSV 数据边界

第一版是 UTF-8 数值 CSV；训练与验证必须有全局不重叠的 sample_id、非空 label 和数值特征。ID 作为字符串保存，禁止缺失或重复。标签也按原始字符串保存，避免重新编码后无法还原。概率列与 classes 的对应关系在产物和预测 metadata 中给出。

特征允许空值或 NaN，不允许无穷值和普通文本。列名唯一，预测时按保存的特征名取列，即使 CSV 列顺序改变也能保持一致。feature_columns 可省略，届时从训练文件中排除 ID 和 label 后推导。

numeric 模式拟合数值填补及必要缩放；passthrough 用于已处理数值 CSV。B 的表格/文本使用上面的 partb 模式；图像使用 VISION.md 中的 frozen_vision 候选。

## D → A：实验结果

可直接调用 `partd.train(request_path)`，或启动 `run_partd.py train --request ...`。结果含：

| 字段 | 含义 |
| --- | --- |
| event / schema_version | 固定为 partd.result / 1 |
| status | success 或 error |
| task_id / run_id / experiment_id | 输入请求的追溯编号（有效配置时） |
| metrics | 指标名到浮点数的映射 |
| primary_metric | name、value、direction（maximize/minimize） |
| artifact | 候选模型目录、model/manifest 路径及哈希 |
| data | 训练/验证样本量与输入文件哈希 |
| warnings | 训练期间的警告，可供 A 判断是否继续改进 |
| result_path | 落盘实验记录；输入解析早期失败时可能不存在 |
| error | 失败的类型和说明 |

CLI 成功返回 0，失败返回 2，输出一行 JSON。命令行参数本身不合法时由 argparse 输出标准错误帮助，C 仍能读取非零退出和 stderr。

D 为每个候选保存可复现产物。A 根据内部验证和资源自主选出最终候选，再将所选产物目录传入预测入口；D 不根据测试分数或人工判断替 A 选择。

## 数值 CSV 独立预测

```powershell
python run_partd.py predict --artifact <候选目录> --input <无标签输入.csv> --output <新预测文件.csv> --include-probabilities
```

输入只需要 ID 和保存的特征列，不需要标签、训练/验证数据、LLM、C 或 API 密钥。默认输出 sample_id、prediction；可选 probability_0、probability_1 等，类别对应关系位于 `.metadata.json`。输出顺序与输入一致。输出路径存在时拒绝覆盖。

产物中含已拟合的完整 numeric Pipeline，可移动后加载。候选目录中 artifact.json 和 model.joblib 必须成对保留；使用与训练记录一致的依赖环境。

## C → D：执行适配

```python
from partd.c_executor import execute_via_c

response = execute_via_c(
    ["train", "--request", "/absolute/path/request.json"],
    partc_root="/absolute/path/partc",  # 若 C 已安装可省略
    workdir="/absolute/path/execution_001",
    task_id="task_001",
    run_id="run_001",
    timeout_seconds=120,
)
# response["execution"]：C 的结构化执行结果
# response["partd_result"]：从 stdout 解析出的 D 结果，若未输出则为 None
# response["status"]：只有执行和 D 都成功才为 success
```

请求路径和预测的输入/产物/输出路径应传绝对路径。C 改变工作目录时也不会造成路径歧义。适配器使用当前 Python，或通过 python 参数显式指定训练解释器。

本适配器负责显式写入一次 C execution 日志，A 不应再把同一次执行重复记账。A 若直接调用 C 而跳过此适配器，应自行承担日志记录与 D 结果解析。

适配器不修改 C 源码。实际部署的 C 提交和目标平台应由 C 统一锁定并验收，特别是超时后的进程清理。

## 待全组交叉验收

1. 是否采用此请求/返回格式；A/B 若有现有接口，优先协商适配，避免重复设计。
2. 本版已按 B 公共接口完成标签/样本 ID 和处理状态打包；请 B 交叉复现并确认。
3. 官方指标、预测列名、概率及类别顺序，按老师任务公告落实。
4. A 的最终候选选择与导出调用时机，以及 C 的超时/资源和统一运行平台。
