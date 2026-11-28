# 表格与文本候选说明

在图像 D 版本上补齐通用候选扩展，并新增 CatBoost、HistGradientBoosting 和词级与字符级联合 TF-IDF。A/B/C 的统一对接事项见 [ABC_HANDOFF.md](ABC_HANDOFF.md)。本次源码改动限定在 D。模型比较是开发实验，完整自主 SuperAgent 成绩仍须由最终统一版本重新运行。

## 模型与输入

| 模型名称 | 输入要求 | 本地实现 |
| --- | --- | --- |
| logistic_regression | 数值矩阵、稀疏文本 | 原有接口，参数可配置 |
| random_forest | 已处理表格或数值矩阵 | 原有接口 |
| linear_svm | 数值矩阵、稀疏文本 | 可选训练集内部概率校准 |
| rbf_svm | 已处理数值特征 | 保留 CSR；缩放属于可配置候选 |
| complement_nb | 非负特征 | 明确拒绝带负数的 difference 特征，不靠截断掩盖 |
| mlp | 已处理数值/稀疏特征 | 可选 train-only 缩放和 SVD，需观察收敛及种子波动 |
| gradient_boosting | 已处理表格或数值矩阵 | 现有本地候选 |
| catboost | B 提供的原始类别与数值列 | D 原生适配器；不先 one-hot；缺失类别映射为固定标记，数值缺失由模型处理 |
| hist_gradient_boosting | 原始类别与数值列 | D 只在训练集拟合 ordinal 映射，明确传入类别列位置；每个类别列最多 255 类，未知类别映射为缺失 |

numeric_columns 必须由数据契约明确给出。CatBoost 保留其余列为类别字符串；HistGradientBoosting 把类别编码视为类别，避免误当连续顺序。自定义适配器是 B Preprocessor 的 D 子类，随 preprocessor.joblib 保存，预测无需原训练文件。

## 原生表格请求

统一 train 请求保持现有结构：data.source=partb，preprocessing=partb。示例中的 numeric_field 需换成实际数值列；全类别数据可传空列表。

```json
{
  "model": {"name": "catboost", "params": {"iterations": 500, "depth": 6, "learning_rate": 0.05}},
  "data": {"source": "partb", "config_path": "config.json", "data_root": "raw",
           "preprocess_options": {"encoding": "native", "numeric_columns": ["numeric_field"]}},
  "preprocessing": "partb"
}
```

同样的 data 设置可用于 hist_gradient_boosting；模型专用类别索引由 D 的处理器生成，不要求 LLM 手工猜测。seed 控制随机性，CatBoost 的冲突 random_seed 会被拒绝。CatBoost 默认 CPU、4 线程、无训练过程文件写入和静默输出。

## 文本联合特征

普通 analyzer=word/char 保持 B 原接口。analyzer=word_char 时由 D 组合两个 B 处理器：word 的 ngram_range=[1,2]，char 的 ngram_range=[3,5]，各自只在训练数据上拟合。max_features 是每个分支的上限；char_weight 控制字符分支权重，组合后进行每行 L2 归一化。

```json
{
  "analyzer": "word_char", "combine": "difference", "max_features": 50000,
  "min_df": 3, "lowercase": true, "char_weight": 1.0
}
```

difference 要求两个有顺序的文本列；单文本用 concat。变换器和两个词表完整保存，预测仅 transform，不重新建词表。

## 模型保存与环境

入口仍为 run_partd.py train / predict，C 执行器适配保持原接口。保存模型、处理器、任务配置、类序、验证预测、训练警告与耗时，加载核验完整文件哈希。CatBoost 另核验其版本；原生类别和联合文本处理另核验 D 源码指纹。

本机为 Python 3.12.4、scikit-learn 1.4.2、NumPy 1.26.4、pandas 2.2.2、SciPy 1.13.1、joblib 1.4.2、CatBoost 1.2.8。CatBoost 依赖见 requirements-native-cpu.txt。PyArrow 17.0.0 仅用于读取本轮公开数据的 Parquet 原文件，不是 CSV/NPY 模型推理依赖。实验环境继承现有科学计算包，验收针对实际使用路径，不宣称全局继承包均已做依赖一致性验收。

团队正式环境需由 C 锁定后重新训练。新 D 源码指纹也会使旧图像特征缓存及图像产物的严格检查失效；正式统一提交前重新生成对应产物，不能为复用旧模型而移除校验。

## 开发实验与发布验收

精简结果见 [CLASSIFICATION_RESULTS.md](CLASSIFICATION_RESULTS.md)，发布验收另记录当前源码版本。历史 94 次开发实验使用的 D 指纹为 c3597b6e17feb97d5a79089d968d7584b9229ed82bbea285f2f5375b3657966b；发布增加能力入口和非法配置保护，历史分数不冒充发布版本的完整自主成绩。


完整 runs/round2/report.html、model-comparison.csv、consolidated.json、audit.json 和运行日志保留在 D 的本地实验目录；仓库仅提供上述精简结果与证据。教师数据仅复制 train/val，测试文件不在开发输入目录。Adult/AG News 从原训练部分固定去重抽取 10000/2000；Dry Bean 在策略冻结后才准备，外部测试均在选择锁定后进行一次评价，不据测试分数继续调参。

相关实现位于 src/partd/candidates.py、advanced_preprocessing.py、catboost_adapter.py 和 partb_adapter.py。A 后续需开放模型与预处理契约并消费反馈；当前开发驱动不是 A 的 LLM 规划器。
