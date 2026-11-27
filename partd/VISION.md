# 图像训练与预测交付

本模块为 A 提供可复用的图像候选实验工具：接收 B 的原始图片路径，以冻结的通用预训练骨干提取特征，训练分类头，在验证集计算指定指标，保存完整方案，再在新进程中从原图预测。新任务重新训练分类头；代码不根据任务名、数据集名或开发实验成绩查找答案。

这次交付包含 D 的流水线和可调用的搜索建议。A 当前集成版本仍需开放图像分支、传递配置并消费反馈，才能完成全 Agent 的自主图像运行。开发分数不能直接作为课程正式自主运行结果。

## 数据流与职责

```text
A 选择一个候选配置
  → C 执行 train --request
  → B DataModule.load_development 提供 train/val 原图路径、标签和 ID
  → D 核验并复用特征缓存，或冻结骨干提取特征
  → D 在训练特征上拟合分类头及必要缩放
  → D 返回验证指标、耗时、警告、缓存信息和完整候选目录
  → A 根据历史反馈决定继续、停止及最终候选
  → A 先落盘 selection，再复制所选目录
  → C 执行 predict，由 B 读取无标签输入，D 加载保存方案并预测
```

D 每次 train 只做一个候选。缓存复用和预处理由 D 承担；最终选择、搜索预算和正式 test 评价由 A 承担。预测不加载训练数据、标签或训练缓存，也不重新拟合处理器。

## 支持的方案

| 部件 | 支持内容 |
| --- | --- |
| 冻结骨干 | ResNet18（512 维）、EfficientNet-B0（1280 维） |
| 分类头 | LogisticRegression、LinearSVC；SVM 可用训练集内部三折校准得到概率 |
| 特征处理 | l2、standard、none；standard 只拟合训练特征 |
| 图像处理 | official、resize224；本轮真实数据实验采用 official |
| 指标 | accuracy、balanced_accuracy、f1_macro、precision_macro、recall_macro、roc_auc、log_loss |
| 标签 | 二分类或多分类，保留标签含义及类别顺序 |
| 输入 | B 的 image_csv / image_folder；灰度图转换为 RGB |

log_loss 和多分类 roc_auc 需要 LR 或校准 SVM。二分类 roc_auc 也支持未校准 SVM 的 decision score。概率列顺序写入预测 metadata。

原图直接交给 D 的骨干专用变换，不先经过 B 的默认 64×64 图像预处理。official 使用明确权重枚举的变换：EXIF 纠正、RGB、短边 256、中心裁剪 224、ImageNet 均值与标准差；ResNet 为 bilinear，EfficientNet 为 bicubic。resize224 是保留全图的独立候选，不能和 official 缓存混用，尚未在这两份开发数据上比较。

## 统一实验接口

与已有表格、文本使用相同的入口和结果外壳：

```powershell
python partd/run_partd.py train --request <请求的绝对路径.json>
python partd/run_partd.py predict --artifact <所选模型目录> --data-root <原图根目录> --partb-root <B目录> --split val --output <新的预测.csv>
```

完整请求模板见 [examples/vision_request.json](examples/vision_request.json)，约束见 [schemas/train_request.schema.json](schemas/train_request.schema.json)。模板路径相对于 JSON 所在目录解析，使用前换成实际数据路径。C 改变工作目录时，传给命令行的请求、模型、数据及输出路径用绝对路径。

model.name 设为 frozen_vision，params 包含 backbone、head、feature_scaling、preprocessing。head.params 开放 C、max_iter、tol、class_weight，SVM 另支持 calibrate。vision 包含本机部署设置 weights_root、cache_dir、batch_size、threads；这些目录由启动配置传入，不要求 LLM 猜测本机路径。data.preprocess_options 留空。

task_id、run_id、experiment_id 只用于追溯。每次实验使用新的 experiment_id。统一入口不接受用于抽样试跑的 max_per_class；正式 train 默认使用 B 提供的完整开发集。

| 返回字段 | A 的用途 |
| --- | --- |
| status、error、stage | 区分配置、数据、缓存和训练失败；同一次失败保留记录 |
| candidate | 规范化后的候选配置，便于去重、重试与反馈 |
| primary_metric、metrics | 使用公告指标及 maximize/minimize 方向比较候选 |
| training.fit_seconds、validation_seconds | 判断新增计算是否值得；这两项不含骨干提取 |
| training.convergence_warning、warnings | 收敛不足时考虑 max_iter、正则强度及缩放 |
| cache.hit、seconds、identity_sha256 | 区分首次提特征和缓存命中；elapsed_seconds 是本次整个候选耗时 |
| artifact.directory、kind | 导出时整体复制该目录；kind=frozen_vision_classifier |
| validation_predictions_path | 独立复算指标、排查类别或数量问题 |

CLI 最后一行沿用 event=partd.result；成功退出 0，失败退出 2。C 未产生 D 结果的超时或启动错误，A 应结合 C execution 状态处理，并把原候选和 status=error 记入历史。不要把超时当成已训练出可比较的模型。

训练产物位于 output_dir/task_id/run_id/experiment_id/。其中 result.json 是统一结果，model/ 是可移动的预测方案，包含 image_artifact.json、backbone.pth、head.joblib、task_config.json 和验证证据。预测命令的 --artifact 指向 model/，或者 A 复制它后的 final_model/。

现有数值及 B 表格/文本接口保持可用。图像产物使用 image_artifact.json，而不是旧 model.joblib 格式；A 的导出可以复制整个目录，额外的清单检查或报告读取不能假定所有模型都有 artifact.json。

## 缓存与失败恢复

同一 task/run 的多个候选在骨干、权重、图像处理及运行设置相同时复用 train/val 特征。分类头、C、缩放和指定指标的变化不需要再次运行骨干。

缓存身份覆盖有序图片内容、样本 ID、完整标签分配、配置、完整权重 SHA256、图像处理、batch/thread 设置、依赖版本、B 和 D 源码指纹。即使只是原路径下图片改变，或标签重排而类别数量不变，也会生成不同缓存。命中后再次核对 features/labels/IDs 文件哈希。不同 run 使用独立缓存，正式运行不借用人工开发阶段的分类器或搜索历史。

提取先写入 .building 目录，完成核验后才重命名为可命中缓存。进程被 C 终止的半成品不会被误用。损坏的完整缓存明确报错，不静默跳过校验。A 在单个 task/run 内顺序执行候选；本版本不支持并发写入同一缓存键。旧失败记录和半成品可供排查，清理应在运行结束后统一进行。

## 可调用的搜索策略

```powershell
python partd/run_partd.py vision-capabilities
python partd/run_partd.py vision-suggest --request <策略请求的绝对路径.json>
```

[examples/vision_policy.json](examples/vision_policy.json) 给出策略请求。history 放 D 已返回的结果，primary_metric 与公告一致，elapsed_seconds 由 A 统计。工具只返回 experiment 建议或 stop 原因，不启动训练、不导出最终模型。A 可以将它作为 planner 的默认建议，也可以根据能力和反馈生成其他合法候选。

策略已实现为代码，并有离线测试：

1. 先比较可用骨干的 LR 与 SVM 基线，默认 C=1、l2。资源不足或只部署一个权重时，backbones 显式限制可用范围。
2. 基线比较后围绕验证表现较好的分支尝试 C/10、C×10，再考虑 train-only standard 缩放。所有比较使用相同主指标和方向，不绑定开发数据集名称。
3. 配置去重。每个失败也计入候选预算；重复失败且没有成功候选时，建议停止检查环境或输入。
4. 已完成基础覆盖后，连续 patience 次未超过 min_gain 的提升则建议停止；也支持候选总数及整体时间预算。默认 8 次、patience=3、min_gain=0.001，均可配置，这些是搜索先验而非准确率承诺。
5. 同一分支增大 C 后，若验证变差且拟合时间超过较小 C 的 slowdown_factor 倍，不建议继续向更大 C 扩张。停止慢分支的决定依据运行记录；工具本身不会杀进程。
6. log_loss / roc_auc 策略中的 SVM 建议启用训练集内部校准。缺失概率的非法请求明确失败，不伪造概率。

这些策略不能替代 A 的完整决策闭环。A 还应把收敛警告、C 的超时/资源错误和缓存命中情况加入下一次上下文；内存问题可减小 batch_size，收敛问题可调整 C、max_iter 或缩放，并用新 experiment_id 留下重试轨迹。到达预算后从成功候选中自主选型；如果全失败则返回失败说明。

## 权重来源与环境

老师邮件已明确允许通用预训练视觉权重，并要求注明模型、来源、使用方式，以及用最终完整自主运行报告结果。本模块仅使用 ImageNet 预训练权重作冻结特征提取，不进行微调、数据增强或视觉优化器更新。

| 骨干 | 固定权重 | 官方下载地址 |
| --- | --- | --- |
| ResNet18 | ResNet18_Weights.IMAGENET1K_V1 | https://download.pytorch.org/models/resnet18-f37072fd.pth |
| EfficientNet-B0 | EfficientNet_B0_Weights.IMAGENET1K_V1 | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth |

完整 SHA256 固定在 vision.py，加载时核验。模型创建 weights=None，checkpoint 使用 weights_only=True。全参数 requires_grad=False、eval、inference_mode，并核对提取前后状态哈希。train/predict 都不下载权重。

可先在独立环境安装 B/D 科学计算依赖，再安装 [requirements-vision-cpu.txt](requirements-vision-cpu.txt) 的 torch 2.6.0+cpu 与 torchvision 0.21.0+cpu。权重合计约 65 MiB，部署前放入 weights_root。可选的准备工具仅在显式 --download 时联网：

```powershell
python partd/examples/prepare_vision_weights.py --weights-root <权重目录> --download
python partd/examples/prepare_vision_weights.py --weights-root <权重目录>
```

第二条只做本地校验。权重安装属于部署准备，不能在候选实验中临时依赖网络。

本机验证环境为 Python 3.12.4、NumPy 1.26.4、scikit-learn 1.4.2、joblib 1.4.2、pandas 2.2.2、SciPy 1.13.1、Pillow 10.3.0。团队记录的 Python 3.11.5 环境需由 C 安装验证并重新训练最终候选；不要跳过依赖、B/D 源码或哈希校验来加载旧环境的模型。依赖锁定后再冻结正式提交版本。

D 的 .gitattributes 将运行时 Python 源码固定为 LF 换行，避免 Git 在不同系统检出时仅因换行变化产生不同的 D 源码指纹。B 源码仍需使用全组记录的相同检出版本和字节。

CPU 可运行，不要求独立显卡。开发机约 16 GB 内存；两份小型开发任务完整冻结提特征约累计 10 分钟，完成任务的采样进程内存峰值约 751 MiB，不能当作所有任务的上限。大图仍会逐批处理，但完整特征矩阵和分类器拟合会占内存。float32 特征矩阵约需 样本数×512或1280×4 字节，另留图片、权重、Python 和训练开销。每个候选当前保存一份完整骨干，磁盘预算还应按候选数量预留；8 个候选建议至少另留 1 GB，原始数据单独计算。

## A B C 对接与验收

| 成员 | 需要完成的动作 |
| --- | --- |
| A | 开放 image 分支；在 parta/decisions.py 的 MODELS、PREPROCESS 和 decision_contract 中加入 frozen_vision 契约，并对齐 agent.py 的请求生成；从任务配置读取主指标；传入部署路径；将 D/C 反馈加入下一轮；保持选择落盘后才预测及最终评价；报告兼容 image_artifact.json |
| B | 确认原图路径、标签、sample_id、split 与 metric；保持 DataModule 公共接口；图像不提前做 64px 缩放；复核无标签新输入格式 |
| C | 在统一环境安装 CPU 视觉依赖，预置两份固定权重；给提特征和最终原图预测设置合适超时；记录启动/失败/资源情况；验证目标系统超时清理；保留原有 API 预算及日志 |
| D | 交付单候选流水线、缓存、分类头与指标、完整模型包、统一预测入口、能力和策略工具、示例与测试；环境统一后重新生成最终产物 |

运行 [examples/vision_demo.py](examples/vision_demo.py) 可自动生成小型三分类灰度图，通过真实 C 进程训练两个候选，验证缓存命中、保存复制、选择落盘及从无标签图片的新进程预测。此示例是 D/B/C 接口验收，不是收费 LLM 自主运行：

```powershell
python partd/examples/vision_demo.py --output <新的目录> --weights-root <权重目录> --partb-root <B目录> --partc-root <C目录>
```

测试时设置 PARTB_ROOT、PARTC_ROOT、VISION_WEIGHTS_ROOT，再运行 partd/tests。未提供权重的视觉集成测试会跳过，正式验收需确认没有跳过。验证包含 RGB/灰度、多分类、两个骨干和分类头、train-only 缩放、指标方向、标签/图片变更导致缓存失效、损坏拒绝、重复输出保护及无标签独立预测。

本地开发参考成绩与范围见 [evidence/vision-development.json](evidence/vision-development.json)。这些结果帮助验证候选可用性，不能在新任务直接复用其分类头。全组下一项验收应在未用于开发的新图像任务上，从一次启动完成 A 的选择、实验反馈、停止、导出和预测；期间不人工改代码、决定下一实验或挑最终模型。正式 test 仅用于最终评价。
