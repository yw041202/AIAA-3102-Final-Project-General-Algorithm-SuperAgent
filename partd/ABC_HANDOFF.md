# D 模块统一对接说明

本文件合并表格、文本和冻结图像的对接事项，按 A、B、C 分工。D 提供单候选训练、验证、完整保存加载、独立预测及能力发现；最终规划、停止和选型由 A 完成。本 PR 的源码改动限定在 D。

接口详见 [分类候选](CANDIDATES.md)、[图像流水线](VISION.md)、[通用请求返回](HANDOFF.md)。开发成绩不等于正式完整自主 Agent 成绩，最终报告须由同一冻结版本自主运行产生。

## A 规划调度与最终交付

### 能力和请求

```powershell
python partd/run_partd.py capabilities
python partd/run_partd.py capabilities --modality tabular
python partd/run_partd.py capabilities --modality text
python partd/run_partd.py vision-capabilities
```

通用入口返回模型名、模态、建议参数字段、概率能力、依赖和限制，不导入 CatBoost/Torch 或启动训练；实际依赖由 C 确认。图像可调用建议仍为 vision-suggest。

1. 同步修改 parta/decisions.py 的 MODELS、PREPROCESS、decision_contract，以及 agent.py 的模态判断和请求生成。保留 LR/RF；优先开放线性/RBF SVM、GB、CatBoost、HGB；NB 为非负输入的快速备选，MLP 为可选探索。
2. 表格原生模型使用 encoding=native，numeric_columns 明确数值列，类别位置由 D 自动生成。原生类别不做通用缩放或 SVD。
3. 文本开放 analyzer=word_char 和 char_weight。联合特征可传 combine、max_features、min_df、lowercase；两个分支固定为 word [1,2]、char [3,5]，不要再附带 ngram_range。difference 要求两个有顺序的文本列，不能交给 NB。
4. 图像开放 image 分支，model.name=frozen_vision；params 包含 backbone、head、feature_scaling、preprocessing。vision 中的 weights_root、cache_dir、batch_size、threads 来自启动配置；data.preprocess_options 留空。LLM 不需要猜本机目录。
5. 主指标按公告传入。SVM 默认无概率；log_loss、多分类 roc_auc 需要训练集内部校准或其他有概率的模型。

示例：[原生表格](examples/native_candidate_request.json)、[联合文本](examples/joint_text_request.json)、[图像](examples/vision_request.json)。请求内相对路径从 JSON 所在目录解析，传给 C 的命令行路径使用绝对路径。

### 搜索和反馈

先按模态、样本量、类别/数值列、稀疏性及指标能力筛选适用候选，覆盖少量基线，再围绕验证领先配置调参。词级、字符级、联合特征都作为候选比较。策略不按任务名、数据集名或开发最佳参数查找答案。

将 primary_metric 的名称、方向、值，以及 training、warnings、data_warnings、error/stage、elapsed_seconds、图像 cache 信息加入历史，送回下一轮。区分首次提特征、缓存命中与分类头拟合。超时、收敛、内存和参数错误按原因重试或换方法，使用新的 experiment_id，保留失败记录。

结合候选范围配置原 max_steps=6、patience=3，保留候选数、耐心、时间/资源与持久 LLM 预算边界。领先配置的种子复核规则在正式运行前确定。表格/文本搜索规则是供 A 实现的建议；D 不自动接管规划器。vision-suggest 是已实现的图像建议接口，A 仍掌握最终选择。

### 导出和最终评价

使用 train 返回的 artifact.directory，不能猜目录。普通分类包使用 artifact.json；图像包使用 image_artifact.json，返回的图像模型目录位于实验目录的 model/。A 的报告读取须兼容两种包。整体复制返回目录后，统一 predict 自动识别。

先写 selection.json，再导出、预测，之后才进行公告规定的最终 test 评价。测试分数不回流规划，不据此重新挑模型。新任务重新训练分类器；可复用通用视觉骨干，不能把开发任务分类头当作新标签任务答案。

## B 数据与预处理

### 确认原始输入

现有 DataModule 公共 API 已能供 D 使用，本次没有修改 B。核对 config 的 modality、format、feature_columns 顺序、label_column、id_column、splits、metric；训练/验证 ID 不重叠，标签语义一致。独立预测只需要原始输入与 ID，不需要标签或训练清单。

- 表格明确数值列与类别列。D 为 CatBoost 保留类别字符串，为 HGB 拟合 train-only ordinal 映射并标记类别列；HGB 每列最多 255 类，未知类别映射为缺失。不要先 one-hot、标准化或降维类别列后，再按原生类别交接。
- 文本保留原文和列顺序。两个标题的 difference 与单文本 concat 语义不同。D 的 word_char 组合两个 B TF-IDF 处理器，词表只在训练集拟合，权重与状态一起保存。NB 只能接非负特征。
- 图像提供原图路径、标签、ID。D 做 EXIF/RGB 及权重指定的 224px 变换，B 不提前应用默认 64×64 处理。确保 image_csv/image_folder 路径在新预测根目录可解析。

### 状态归属与交叉验收

D 已实现原生类别和联合文本适配器，是 B Preprocessor 的 D 子类；A 可先通过 D 统一入口调用。若 B 后续接管，保持等价的 train-only 拟合、变换和保存语义，再统一版本重训，避免重复拟合两套处理器。模型专用缩放/SVD、图像权重变换和缓存由 D 保存。

B 用固定小输入独立复算指定指标，再在无标签、无训练文件的新根目录启动新进程预测。核对数量、ID 顺序、原标签、概率列与 classes 映射及保存前后的一致性。B 源码变化可能触发校验失效，需要与 D 协调重生成产物。

## C 环境权重与执行记录

### 统一依赖和部署

1. 基础科学计算依赖按全组锁定，当前 D 使用 sklearn 1.4.2。原生类别增加 [requirements-native-cpu.txt](requirements-native-cpu.txt) 的 CatBoost 1.2.8 及依赖；其余普通分类器来自 sklearn。
2. 图像增加 [requirements-vision-cpu.txt](requirements-vision-cpu.txt) 的 torch 2.6.0+cpu、torchvision 0.21.0+cpu。CPU 即可。部署前在 weights_root 放好固定 ResNet18、EfficientNet-B0 ImageNet 权重并核验完整 SHA256，来源和使用方式见 VISION.md；训练预测不联网下载。
3. 本机开发为 Python 3.12.4，团队记录环境为 3.11.5；由 C 确定并验证唯一环境和锁文件，D 在该环境重训。保留版本、源码和文件哈希校验。PyArrow 只用于本轮外部数据准备，不是 CSV/NPY 训练预测必需项。

### 执行和日志

适配器继续调用 C 的 PythonExecutor.execute_file 和 StructuredLogger，无需因这些模型改公共接口。使用绝对请求/模型/输入/输出路径，task_id、run_id 与请求匹配。stdout 最后一条 partd.result 是 D 载荷；C 非零退出、超时或缺失载荷都按失败处理。

给首次图像提特征及最终原图预测足够超时，按实际规模测量，控制 CPU 线程与并发。缓存命中不能当作首次成本。保留 stdout/stderr、执行 ID、错误分类，并在目标系统验证超时后的整个子进程树清理。新实验不覆盖旧输出，未完成的 .building 缓存不能当作成功缓存。

D 适配器已记一次 execution 日志，A 不重复记录同一次执行。分类器拟合不调用收费 LLM；A 规划调用仍由原账本精确记 token 和费用，已花费与不确定预留保持累积，不重置预算。

## 全组验收顺序

1. C 准备统一环境及权重，B 确认数据契约；运行生成数据的 [候选示例](examples/candidate_demo.py) 和 [图像示例](examples/vision_demo.py)，复核训练、验证、保存、加载与独立预测。
2. A 接入能力、合法请求和反馈，在未参与开发的新任务上检验一次启动后的选择、停止、导出和预测，期间不人工改代码、决定下一实验或挑最终模型。
3. 冻结最终代码和依赖，重新生成对应缓存及模型。本次 D 指纹变化后，旧图像缓存及高级预处理产物不能冒充本版本产物，不移除校验来复用。
4. 以同一最终版本完成老师任务的完整自主运行，先锁定方案再做最终测试，提交对应模型、预处理、预测、日志和报告。
