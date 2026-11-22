# Part B 数据模块

提供表格、单列或多列文本、图像的数据接入、开发集检查、划分管理、可配置预处理及状态保存/加载。所有实现按配置中的数据结构工作。课程任务名称只用于选择配置文件。

## 环境与运行

Python 3.10+，CPU 即可。从仓库根目录运行以下命令。

```powershell
python -m pip install -e ./partb
python -m unittest discover -s partb/tests -v

python -m partb --config partb/configs/coupon.json --data-root "C:/Users/83946/Desktop/py4ai/new_project/final_project_tasks/in_vehicle_coupon_recommendation" --output partb/outputs/coupon
python -m partb --config partb/configs/headlines.json --data-root "C:/Users/83946/Desktop/py4ai/new_project/final_project_tasks/headlines_ctr_clean" --output partb/outputs/headlines
```

组员将 `--data-root` 改成自己的数据目录。配置内路径相对于 data_root；程序将根目录解析为绝对路径。安装到 C 执行器实际使用的 Python 环境后，D 可以在其他工作目录 `import partb`。

每次运行输出 `audit.json`、`preprocessor.joblib`、其 `.json` 拟合来源记录、`verification.json` 和新进程转换结果。`verification.json` 应显示 success、new_process_reload_equal=true、test_accessed=false。输出文件位于已忽略的 outputs 目录，原始数据不复制进代码仓库。

命令使用示例默认预处理。Agent 可通过 `--preprocess-config options.json` 或 Python 构造参数提供另一组配置。每个实验使用独立输出目录，避免覆盖候选方案。

## 支持范围

| 类型 | 接入方式 | 预处理与输出 |
| --- | --- | --- |
| 表格 | NPY 特征/标签文件，或带表头 CSV | 类别缺失填充，onehot 或 ordinal 编码，未知类别处理；显式声明的数值列采用训练中位数填充与标准化；输出 float32 数组或稀疏矩阵 |
| 文本 | CSV 的一个或多个有序文本列 | 共享训练词表的词级/字符级 TF-IDF；多列拼接，或两列 A−B；输出 CSR 稀疏矩阵 |
| 图像 | CSV 图片路径清单，或 split/class/image 目录 | EXIF 方向校正、RGB 转换、尺寸调整、除以 255、可配置均值/标准差；默认 float32 NCHW，支持分批或展平 |

文本默认保持标点、大小写原始内容，是否 lowercase 由向量器配置控制。A/B 顺序与原始标签保持对应。文本空值变为空字符串，完全空词表会给出可定位的异常。图像预处理为确定性基础流程，后续可以与 D 协商增加训练专用增强。当前不包含预训练特征提取、视频、DICOM 或压缩包自动识别。

## 图像模拟验收

目前课程已发布任务没有图像数据，以下输入仅用于接口验证。

```powershell
python partb/examples/make_image_fixture.py --output partb/outputs/image_fixture
python -m partb --config partb/configs/image_manifest.example.json --data-root partb/outputs/image_fixture --output partb/outputs/image_manifest
python -m partb --config partb/configs/image_folder.example.json --data-root partb/outputs/image_fixture --output partb/outputs/image_folder
```

CSV 清单使用 image_path、label、sample_id。图片路径相对于配置的 data_root/image_root。文件夹模式中，训练和验证使用 `train/class_name/image.png`、`val/class_name/image.png`，标签为类别目录名。无标签预测目录也可递归读取，输出顺序为路径排序，sample_ids 保存相对路径。

## 划分、数据检查与评估

- 存在官方 val 时直接保留。没有 val 时必须显式配置 `validation_split`，例如 `{"fraction": 0.2, "seed": 42}`，采用分层划分。类别样本过少无法分层时返回错误，由调用方处理。
- 未知任务配置字段、未知预处理参数、重复划分来源，以及 ID/标签混入特征的配置会在处理前被拒绝。路径检查包括规范化路径和已存在文件的同一物理来源检查。
- `load_development()` 仅加载 train/val。`audit()` 检查行数、标签分布、缺失标记、唯一值、重复输入和训练/验证重叠，同时记录样本归属与输入指纹。完整审计包含样本 ID，A 可仅将摘要字段发送给 LLM。
- 缺失标记当前包括实际空值、空字符串、nan、none、null、?。这是一项通用默认约定，新任务若把这些字符串用作有效类别，需要调整约定并重新验证。
- 预处理只接受 name=train 的 DataSplit 拟合；验证/预测仅 transform。该约束防止意外误用，调用方仍应遵守测试集隔离。
- `load_prediction()` 返回测试输入，y=None；CSV 的目标列不会进入特征。仅当 Agent 已选定最终方案后调用。
- `load_final_evaluation(final_evaluation=True)` 提供显式最终评估入口。测试结果必须隔离于搜索循环。此入口是流程约定，不是操作系统访问控制。
- 官方划分中的重复样本只报告，保持数据原有行数及顺序。图像审计的重复检测基于路径，CLI 会分批解码检查文件可用性；未实现图片像素级去重或近重复检测。
- NPY pickle 仅对课程已知可信配置显式启用；joblib 状态也只加载团队可信产物。

## 验收与交接

`HANDOFF.md` 包含 A/D 调用示例、C 集成注意事项、错误规则。`VALIDATION.md` 记录本次实测结果和待团队联调项。

```powershell
python -m partb.examples.train_validation_demo --config partb/configs/coupon.json --data-root "C:/Users/83946/Desktop/py4ai/new_project/final_project_tasks/in_vehicle_coupon_recommendation" --output partb/outputs/coupon/integration
```

该示例是一个临时 D 接口消费者，用线性分类器确认特征可用于训练和验证。它没有自主搜索，其得分不能作为最终 SuperAgent 的正式报告结果。

## AI 工具使用记录草稿

B 模块使用 AI 辅助完成接口设计、代码实现、测试和文档整理。组员应复核数据处理约定、划分来源、代码与验证记录，并在小组最终 AI 使用声明中如实合并。当前运行验证未调用收费 LLM API。
