# Part B 交接说明

## 公共接口

```python
from partb import DataModule, DataSplit, Preprocessor

data = DataModule(config_path, data_root)
train, val = data.load_development()
audit = data.audit()

# options 由 A 的实验方案提供。每次实验新建处理器。
processor = Preprocessor(data.modality, options).fit(train)
X_train = processor.transform(train.x)
X_val = processor.transform(val.x)
y_train, y_val = train.y, val.y
processor.save(artifact_path)
```

DataSplit 字段为 name、x（DataFrame）、y（一维 ndarray 或 None）、sample_ids（一维字符串数组）。每一行 x/y/ID 对齐。B 不重排官方样本，标签保持原始含义。D 如果编码标签，需独立保存其标签映射。图像文件夹标签为字符串。

## A 提供的预处理参数

| modality | options 示例 | 输出 |
| --- | --- | --- |
| tabular | `{"encoding": "onehot", "numeric_columns": []}` | 数组或稀疏矩阵 |
| tabular | `{"encoding": "ordinal", "numeric_columns": ["age"]}` | 数组；age 必须是可解析数值 |
| text | `{"analyzer": "word", "ngram_range": [1, 2], "max_features": 30000, "combine": "concat"}` | CSR；按配置列顺序拼接 |
| text | `{"analyzer": "char", "ngram_range": [3, 5], "combine": "difference"}` | CSR；两列输入为 A−B |
| image | `{"size": [64, 64], "mean": [0,0,0], "std": [1,1,1], "flatten": false}` | float32 NCHW |

文本额外支持 min_df、lowercase。图像 size 是 [高, 宽]，先除以 255，再执行 (x−mean)/std。均值/标准差是配置指定常数，本实现不从验证或测试数据估计。课程优惠券配置中 21 列均按类别输入，元数据没有数值列。

## D 的训练、保存与预测

训练代码使用 X_train/y_train，验证代码使用 X_val/y_val。不要将大稀疏文本矩阵直接转为 dense。图像优先按批处理。

```python
for images, labels, sample_ids in processor.iter_batches(train, batch_size=32):
    # images 为 numpy 数组；D 可按自己的训练框架转为张量。
    pass
```

D 保存每个候选模型时，同时记录该实验的 preprocessor.joblib 路径及配置。A 选定最终候选后，D 执行以下流程。

```python
processor = Preprocessor.load(selected_preprocessor_path)
prediction_data = data.load_prediction()
X_test = processor.transform(prediction_data.x)  # 图像请使用 iter_batches
predictions = selected_model.predict(X_test)
# 依照 prediction_data.sample_ids 和任务规定写出预测；由 D 管理输出格式。
```

新进程必须可 import partb，且具有匹配依赖。先在执行器的 Python 环境执行 `python -m pip install -e ./partb`。状态保存包含拟合后的词表、编码器、填充值/缩放器，及其列顺序；切换实验方案必须重新 fit 新实例。

## C 的执行器接入

B 是可导入的数据包。C 可以启动 D 的训练脚本，由该脚本调用 B。交接前统一解释器及依赖。脚本、配置、数据根目录和产物目录都建议传入绝对路径，避免 cwd 差异。

本模块不直接打印数据或指标；CLI 和训练示例最后打印一行 JSON。D 可沿用 C 的 stdout 指标契约，将 split=val 与 accuracy 一起返回。A/D 仍需调用 C 的执行日志记录函数。

## 错误与边界

加载及预处理 API 保留 Python 异常，由 D/A 的调用边界捕获并转换为实验失败记录。常见为 FileNotFoundError/OSError（文件）、ValueError（配置/形状/标签/图片解码）、PermissionError（测试标签入口）、RuntimeError（未 fit）。失败信息应包含阶段和实验 ID，禁止静默删除错误样本导致标签错位。

必须等待团队确认的项目是 A 实际请求字段、D 图像训练输入及完整模型产物约定、C 运行环境和 Windows 超时修复。当前 B 接口可独立使用，尚未与他们的实际实现完成验收。
