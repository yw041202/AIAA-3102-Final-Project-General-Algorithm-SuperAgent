# 本地集成所用 B/C 版本

本交付包主要包含 D 的源码、说明和模拟验收证据。B/C 保持队友原有源码，不复制为 D 内部实现；团队整合时安装其包或传入源码根目录。

| 模块 | 固定版本与源代码 |
| --- | --- |
| B | [azhao258 的提交 1fbb9c5](https://github.com/azhao258/AIAA-3102-Final-Project-General-Algorithm-SuperAgent/tree/1fbb9c51da9dfa256d1b4078b6b24ed86a032ba1/partb) |
| C | [yw041202 的提交 877625a](https://github.com/yw041202/AIAA-3102-Final-Project-General-Algorithm-SuperAgent/tree/877625a2216d2a6bd5410e860ccf8006baea5748/partc) |

README 提供固定提交的独立克隆命令，放入本目录已忽略的 .deps/partb、.deps/partc。B 的参数可指向仓库或包目录；C 指向含 src 的包目录。也可使用同一解释器已安装的匹配版本，避免重复克隆。

模型记录 B 源码字节指纹；恢复旧模型时使用匹配快照/安装包，不在保存状态与恢复之间重新格式化或更换 B 源码。依赖以 requirements-partb.txt 和每个 artifact.json 的实际版本为准。

本轮没有调用收费 API，C 的调用仅使用 PythonExecutor/StructuredLogger。C 的 Windows 超时清理和计费边界待 C 后续修复，本包未修改它们。
