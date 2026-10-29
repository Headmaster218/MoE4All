---
kind: reference
status: current
scope: submission-protocol
last_verified: 2026-10-20
verified_commit: working-tree
---

# 社区报告封装与提交协议

## 类型与路线

| 类型 | 扩展名 | payload schema | Git 归档路径 |
| --- | --- | --- | --- |
| Benchmark | `.m4b` | `benchmark/1` | `benchmark-database/v1/YYYY-MM/<id>.m4b` |
| 错误报告 | `.m4e` | `error/1` | `error-database/v1/YYYY-MM/<id>.m4e` |

两类文件共用 `MOE4ALL-SUBMISSION/1` UTF-8 文本封装。三平台向导都支持 GitHub PR 或 Cloudflare 中转：可使用已登录 gh 时直接提交；工具缺失、未授权或直传失败时使用浏览器确认的中转，不自动安装工具。PR 创建后即公开，不等合并才公开；不创建 Issue、不自动合并。

## 封装

```text
MOE4ALL-SUBMISSION/1
kind: benchmark
schema: benchmark/1
id: <unique-id>
created-at: 2026-10-19T01:02:03.004Z
engine: 1.0.3
encoding: gzip+base64
decoded-bytes: <原始 JSON 字节数>
compressed-bytes: <gzip 字节数>
sha256: <原始 JSON 字节的 SHA-256，小写十六进制>
summary-count: <摘要行数>
summary-001: <从 payload 重新生成的单行摘要>

-----BEGIN MOE4ALL PAYLOAD-----
<每行 76 个 ASCII 字符的标准 Base64，最后一行可更短>
-----END MOE4ALL PAYLOAD-----
```

将 payload 序列化为紧凑 UTF-8 JSON，对确切字节计算 SHA-256，然后压缩为一个 gzip member、标准 Base64 编码并按 76 字符换行。摘要可读，payload 无损，但不是加密。

生产者生成 LF、不带 UTF-8 BOM；解码器也接受 CRLF 和裸 CR。完整文件与解压 JSON **分别不超过 128 KiB**。gzip 必须完整、CRC 正确，不接受多个 member 或尾随压缩数据。长度、hash、kind、schema、身份和摘要都需一致。

header 名称使用 ASCII 字母、数字和连字符，顺序不作为接收条件；重复 header、标记后有内容、非规范 Base64 都拒绝。JSON 拒绝重复／保留字段、NaN、Infinity、不安全整数、未配对代理字符和超过 48 层的嵌套。UTC 接受 Z 和 +00:00；目录月份取原始报告 UTC 时间，不取上传时间。

## 摘要与校验

摘要是 payload 的确定性投影，不是第二份事实来源。Benchmark 包含模型、架构、体积、设备、VRAM、profile 和各上下文结果；错误报告包含失败类型、模型／设备、描述与最终错误。吞吐量、GiB 保留两位小数，使用 binary64 ties-to-even 格式化。

控制字符转为空格、连续 Unicode White_Space 合并；U+FEFF 不属于 Rust 的空白字符。截断按 code point 计数，**达到上限即追加 ...**：描述／错误摘要上限 240，engine header 上限 80，最终摘要行上限 500。截断不改变 payload。

共享向量位于 `scripts/fixtures/submission-vectors.json`，Rust、独立 Python 和网站校验对应边界。decode 允许检查尚未补充描述的 fatal 报告；verify 和提交校验要求完整、可公开的报告。

```text
infr submission verify <file.m4b-or-m4e>
infr submission decode <file> [--output decoded.json]
python3 scripts/moe4all_submission.py verify <file>
python3 scripts/moe4all_submission.py decode <file> --output decoded.json
python3 scripts/validate-benchmark-submission.py <file.m4b>
python3 scripts/validate-error-submission.py <file.m4e>
```

浏览器和 Worker 都重新校验完整封装、gzip、JSON、schema、身份、摘要及隐私；浏览器通过不代表服务端可以跳过校验。浏览器使用可检查输入消耗的严格 gzip 解码器，不仅依赖无法暴露 member 消耗量的 DecompressionStream。

## Payload

Benchmark 保留模型、硬件、配置、上下文与原始测量。自动 suite 与三平台手动结果都使用 .m4b；手动结果须明确 prefill 或 decode，不上传旧裸 JSON。每份最多 64 个不同上下文，每上下文最多 64 种不同并发，每并发最多 100 轮；需包含 overall、三个计分段和与并发一致的 lane 矩阵。128 KiB 总限额同时生效，不保证最大矩阵都能容纳。

自动 suite、直接 `infr bench` 和三平台手动向导都在 warmup 前调用同一硬件 collector。报告保留 RAM 容量／预算与可用量快照、所选 GPU 的 Vulkan/PCI 身份、PCIe 物理父链与可取得的固件插槽标识，以及模型所在盘的介质信息和有界顺序／随机读取探测。`pcie_endpoint_current_*`（兼容别名 `pcie_current_*`）只描述显卡端点到紧邻上游端口的当前协商链路；`pcie_endpoint_max_*` 是显卡端点能力，**不是主板插槽、交换芯片上行口或整条主机路径能力**。`pcie_path` 逐级记录端点和上游 PCI bridge，`pcie_physical_slot` 只保留 ACPI/sysfs 与 SMBIOS 可核对的固件插槽归属；插槽名里的宽度最多是 connector hint，不是协商宽度。只有每一级当前链路属性都可用时，`pcie_end_to_end_current_*` 才给出全路径最窄链路，否则必须为 null 并保留 `pcie_end_to_end_unknown_reason`，不得拿显卡支持值或插槽名称冒充实际物理链路。平台不公开、权限不足或设备无法唯一匹配的字段同样保留 probe status/reason，不推测。PCIe 理论带宽不同于实测速度，buffered 文件读取也不能当作纯物理 SSD 吞吐量；统一内存设备的 RAM/VRAM 不能相加。

错误报告描述非空、最多 4096 code point，可在向导返回修改。诊断最多 220 行启动日志、12 行最近日志，两部分合计最多 32 KiB，附必要环境、最终错误及错误链。禁止速度表、prompt、response、图片、密钥、主机名、用户名、序列号及未脱敏绝对路径；隐私声明不能代替内容检查。原始滚动证据不整体嵌入报告。

## Git 与 CI

每个 PR 只能新增一份不可变报告，不改旧数据或混入代码。文件名等于 ID，全库唯一；更正结果使用新 ID。CI 按 merge-base 判定变更，读取 NUL 分隔路径，严格检查目录、月份、类型、身份与 payload。

纯报告 PR 只运行独立 Python 校验，不触发模型下载、推理或重型构建。维护者手动审核；Git 直提不进入 D1，也不受 Cloudflare 来源额度约束。

## 中转与浏览器回执

默认 API 为 https://api.moe4all.com/v1，确认页为 https://moe4all.com/community/upload.html。正式入口使用 HTTPS；显式配置 loopback 开发地址可用 HTTP，不接受其他明文来源。

向导绑定类型、ID、大小与整个文件 SHA-256，创建 15 分钟会话，并发送最多 8 行、每行 500 字符的摘要。浏览器将它标为客户端提供、尚未校验。用户主动同意公开并完成 Turnstile 后，向导发送确切原文件字节。

Cloudflare 先返回回执，再异步创建 PR；复核 PR head 原件后立即删除云端正文，不等合并。128 MiB 缓冲区满时返回可重试 503，不淘汰未过期报告。最多重试 72 小时；每来源滚动 24 小时最多成功接收 50 份新报告、创建 100 个会话，重复提交不重复消耗成功报告额度。

| 回执状态 | 含义 |
| --- | --- |
| forwarding | 已接收，后台创建或重试 PR |
| pending_review | PR 原件复核完成，等待维护者审核 |
| published | main 原件已复核、归档；错误归档不等于修复 |
| rejected | PR 关闭且未合并 |
| delivery_failed | 转发过期、正文清理；重新导出新 ID 再尝试 |

向导只保存私密回执并打开网站，不实现查询界面。使用 URL fragment 交接独立回执凭证，网页读取后清除地址栏敏感参数并复核返回身份。无法自动打开时，在网页导入 .private-receipt.json。凭证不进入报告、Git 或普通日志。

重复提交的 file_sha256 指向最初原件，submitted_file_sha256 绑定本次文件，不能因换行差异误判损坏。HTTP 200 不代表 PR 已建立，应展示真实状态。429／503 有限重试且遵守 Retry-After；过长等待保留原件、稍后重试，不提前重发。上传失败不改变已成功 Benchmark 的退出结果。

## 榜单与验收边界

网站按定时 CI 或手动操作，从 Git main 的 .m4b 生成全量精简静态数据库；浏览器获取最新已部署版本并本地筛选。**不要求引擎逐次通知，也不是网页每次实时扫描 Git。** .m4e 不上榜。

本轮验证本地协议、三平台上传逻辑、真实 Windows CLI 与 loopback HTTP、浏览器回执交接和合成 Git 数据的索引生成。GitHub 写操作使用 fixture，没有创建真实 PR、部署云端或占用 GPU。非 macOS 主机的 Zsh 检查不替代原生 macOS 安装验收。
