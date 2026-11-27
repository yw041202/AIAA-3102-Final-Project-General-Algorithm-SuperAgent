# 学校 HPC AIGC API：配置与实测

2026-11-22 更新：用户的 AIAA3102_API_KEY 属于学校 HPC AIGC 服务，**不属于 Diandong Cloud**。此前 Diandong 返回的“无效令牌”不能证明该 key 在学校服务无效；本次已在正确服务上调用成功，无需更换该密钥。

## 已核实的接入信息

- [学校 API 调用说明](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-usage/)
- [学校模型名称与后端映射](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-model-list/)
- Base URL：`https://aigc-api.hkust-gz.edu.cn/api/v2`
- 完整生成端点：`https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions`
- 请求模型名：`DeepSeek-V4.1-Flash`
- 学校文档映射的后端版本：`DeepSeek-V4.1-Flash-1017`
- 密钥：Windows 用户环境变量 `AIAA3102_API_KEY`，原样使用，不添加 Diandong 的前缀。

选择依据：该模型在学校明确可用，且后端版本对应 [DeepSeek 官方 2026-10-17 更新](https://api-docs.deepseek.com/updates/)，符合课程发布截止日期。文档允许以 `chat_template_kwargs: {enable_thinking: false}` 关闭思考，本次结构化连通性测试使用此配置。

## 成功调用记录

返回内容：`{"status":"ok","answer":2}`。

| 项目 | 本次结果 |
| --- | --- |
| 输入 token | 19 |
| 输出 token | 10 |
| 总 token | 29 |
| 课程核算价格（每百万 token） | 输入 1 元、输出 4 元 |
| 课程核算费用 | 0.000059 元 |
| 按学校公开单价估算的平台费用 | 0.0000295 元；未读取账户实际扣费 |

原始返回记录：`runs/hkust_smoke/20261122T094950428427/result.json`。C 的逐次调用日志在同目录 `logs/llm.jsonl`；独立测试账本在 `runs/hkust_smoke/budget/ledger.json`。连通性测试预算上限 0.10 元，计入用户授权的总验证预算，不属于课程任务的模型成绩。

首次尝试发生 TLS EOF 连接错误，未获得用量；C 按不确定调用保留预算预留，记录未删除。随后无密钥诊断发现直连可以建立 HTTPS，于是仅对学校客户端设置 `PARTC_TRUST_ENV=false` 后调用成功。TLS 证书验证保持启用，未改变系统代理配置。环境变量中没有显式 HTTP_PROXY，不据此断言具体代理来源。

## 课程计费口径

根据用户提供的老师通知，DeepSeek 可在任何时段按官方非高峰价格核算，且不计缓存优惠、平台促销和免费额度。[官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)当前显示非高峰输入/输出为每百万 token 1/4 元，学校公开价为 0.5/2 元。程序分别记录课程核算和平台估算，不混用。

学校专用模型条目已加入 C 注册表，采用 `course_off_peak_20261021` 计费标记。当前官网核验日期为 2026-11-22；正式报告仍应保留截至 10 月 21 日的定价依据，当前网页不等于历史存档。其他平台原有注册表条目保持原状。

## 已完成的代码适配

1. 保留 `/api/v2`，避免 C 自动追加 `/v1` 形成错误路径。
2. 使用学校文档中的模型名，记录对应后端版本。
3. 按学校参数格式关闭思考。
4. 学校专用配置启用不继承代理的 HTTP 客户端，证书校验不变。
5. 两项正式任务配置固定为上述模型；模型调用仍需使用学校 env 配置。

## 运行命令

从仓库根目录执行，密钥不写入文件：

```powershell
$env:AIAA3102_API_KEY = [Environment]::GetEnvironmentVariable('AIAA3102_API_KEY', 'User')
.\.venv\Scripts\python.exe -m parta.smoke_hkust
```

两项数据任务已完成完整自主运行，命令为：

```powershell
.\.venv\Scripts\python.exe run_superagent.py run --config parta/configs/coupon.released.json --env-file parta/configs/hkust.env.example
.\.venv\Scripts\python.exe run_superagent.py run --config parta/configs/headlines.released.json --env-file parta/configs/hkust.env.example
```

完整结果与交叉验收见 [LIVE_RESULTS_ZH.md](LIVE_RESULTS_ZH.md)。已有 live_001 目录不能覆盖；再次运行应使用新的 run_id，保留 task_id、output_root 和账本。任务各自预算仍不超过 20 元；不要通过改 task_id、移走账本等方式清零费用。Windows 当前进程若已有旧 ZZZ_BASE_URL/ZZZ_MODEL 配置，它们会优先于 env 文件，运行前应确认没有指向旧平台。
