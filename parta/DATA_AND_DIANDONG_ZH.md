# 正式数据接入与 Diandong Cloud 检查

检查日期：2026-11-22。

**后续更正：** 用户确认密钥属于学校 HPC AIGC 服务，而不是 Diandong。已在正确服务调用成功。下文数据检查仍有效；Diandong 认证失败与操作建议仅保留为历史记录，请按 [学校 API 接入记录](HKUST_API_ZH.md) 配置，无需因此前错误平台返回而更换学校密钥。

## 数据可用性

实际数据目录比用户提供的解压目录多嵌套一层：

```text
../final_project_tasks/final_project_tasks/
  in_vehicle_coupon_recommendation/
  headlines_ctr_clean/
```

保持原始数据位置，不复制进 Git 仓库。`__MACOSX` 是压缩包附带的元数据目录，不参与加载。

| 任务 | 训练/验证样本数（实测） | 默认预处理后的列数（实测） | 测试样本数（任务说明） |
| --- | --- | --- | --- |
| 优惠券接受预测 | 8117 / 2030 | 111 | 2537 |
| 标题点击率比较 | 8301 / 940 | 60000，稀疏矩阵 | 3232 |

已检查文件是否存在，使用 B 的 DataModule 加载 train/val、生成审计，并在 train 上拟合预处理后转换 train/val。测试集内容和标签没有用于本次检查。优惠券存在训练/验证相同特征组合，沿用官方划分并保留警告，不人工删除或重划分。

数据包内的两份 README 均明确要求最终报告 test accuracy。配置已保存为：

- `parta/configs/coupon.released.json`
- `parta/configs/headlines.released.json`

两者启用 `final_evaluation: true`。A 完成全部自主实验、锁定候选并生成预测之后，才允许读取测试标签并输出 `final_evaluation.json`。测试结果不回流给规划器。默认示例和离线 demo 仍不读取测试标签。

新增评估检查覆盖：必须有选型和对应预测、样本 ID/顺序一致、禁止重复评估、accuracy 计算。与规划器及 C 客户端相关测试合计 18 项通过，记录见 `runs/data-api-check.xml`。真实数据训练成绩和测试成绩尚未产生。

## 已确认的平台信息

- 官网：[HKUST Diandong Cloud](https://hkust.ddycloud.com/)
- [API Key 管理页](https://hkust.ddycloud.com/console/aiToken/apiKey)
- [Token 页面](https://hkust.ddycloud.com/console/aiToken)
- [使用日志](https://hkust.ddycloud.com/console/aiToken/usingLogs)
- 模型 API Base URL：`https://hkust.ddycloud.com/v1`

依据：官网公开 API Key 页面脚本 `/_nuxt/apiKey.440125df.js` 将接口地址设置为 `location.origin + "/v1"`；复制密钥逻辑为 `"sk-" + 完整令牌内容`；令牌详情展示“可使用模型”和 OpenAI API 类型。首页主要介绍 GPU 租赁，但本项目当前使用模型 API 入口，不需要为这两个基础分类任务先购买 GPU 实例。

公开配置示例在 `parta/configs/diandong.env.example`。沿用 C 的 ZZZ_BASE_URL 配置变量名，但地址已指向 Diandong；实际调用日志会记录真实服务域名。API key 从 AIAA3102_API_KEY 读取，不写入配置文件。

## 当前认证状态

使用用户级环境变量查询 `/v1/models`，平台返回“无效的令牌”。检查未发现首尾空白、额外引号或 Bearer 前缀。变量缺少官网复制逻辑中的 `sk-`；仅在内存中补齐后再次查询，仍被拒绝。

没有进行任何模型生成请求；本次模型 token 消耗为 0。无法在认证失败时确定账号实际可用模型、令牌额度或模型映射，不据公共网页猜测账号权限。

请用户登录 API Key 管理页：

1. 确认使用老师对应的账号，目标令牌状态为“正常”，未过期且额度可用。
2. 使用密钥旁边的复制按钮获取完整密钥；不要复制 ID、令牌名称或遮蔽后的显示文字。
3. 在 Windows 用户环境变量中更新 AIAA3102_API_KEY，无需向聊天发送密钥。
4. 告知已更新。助手会重新读取用户级变量，查询可使用模型，再核实合规版本并开始小额验证。

若复制现有令牌仍失败，可以在平台允许且符合老师账号安排的情况下新增令牌，或请课程助教核对权限；不需要先充值或购买 GPU。

## 认证通过后的运行方式

从仓库根目录执行；模型名称要先根据账号模型列表确认，不能直接假定 C 原平台的名称在 Diandong 可用。

```powershell
$env:AIAA3102_API_KEY = [Environment]::GetEnvironmentVariable('AIAA3102_API_KEY', 'User')
$env:ZZZ_MODEL = '<已核实可用且符合课程要求的模型 ID>'
.\.venv\Scripts\python.exe run_superagent.py run --config parta/configs/coupon.released.json --env-file parta/configs/diandong.env.example
.\.venv\Scripts\python.exe run_superagent.py run --config parta/configs/headlines.released.json --env-file parta/configs/diandong.env.example
```

两项任务各自最多按官方 token 价格计人民币 20 元，使用同一个持久账本跨运行累计。用户授权总验证预算为 50 元，预留部分用于连通性检查；正式课程任务仍遵循单任务 20 元。平台实际扣费与课程官方价格核算需分别核对。
