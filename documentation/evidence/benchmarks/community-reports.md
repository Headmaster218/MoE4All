---
kind: benchmark
status: historical
scope: community-performance-and-compatibility
run_date: mixed; survey-cutoff-2026-09-24
commit: mixed-or-unreported
evidence: historical-sample
---

# 社区性能与稳定性记录

本页汇总截至 2026-09-24 的社区性能、回退与兼容性证据。它们不是本机受控 benchmark；硬件、模型、量化、上下文、软件版本或测试口径缺失时，不用于跨 GPU 排名。完整调研原件保存在本机忽略的 `benchmark-data/artifacts/MoE4All_用户实机性能全量调研_2026-09-24.docx`，不随文档发布。

## 主要观察

- **专家驻留比例是关键变量。** Ryzen AI Max+ 395 上 Qwen3.6 35B 全驻留报告为 51.4 tok/s；同机仅约 24% expert cache 覆盖时只有 1.3 tok/s，并观察到单核分页、NVMe 低利用。容量足够与否不能只看模型能否启动。
- **Prefill 调度曾造成数量级差异。** RX 7900 XTX 的 Qwen3.8 长提示 pp4096 对照中，layer-major 为 20.57 tok/s、chunk-major 为 229.09 tok/s（11.14 倍）；相应 PR 样本由约 11.44–13.34 提至 169.44–205 tok/s。此为固定测试路径的历史 A/B，不代表所有提示/版本。
- **Intel 既有成功样本，也有明确性能回退。** Arc A770 上 v0.6 样本约 30–32 tok/s；Issue #41 同机跨 v0.6/v0.7 对照中，四类 workload 回退约 12.0%–24.6%，但空闲 RAM 从 13.1 增至 29.6 GiB、进程工作集从 39.1 降至 21.8 GB，模型加载约快 3 秒。吞吐与资源占用的取舍需同时记录。
- **显存不足后落入 RAM/SSD 会跨越交互性门槛。** DeepSeek V4 Flash Q4 在 144–147 GB 模型、128 GB UMA 环境报告约 0.018–0.02 tok/s，属于容量越界和外存路径验证，不是可用 decode 基准。
- **跨硬件平台的数据不等价。** NVIDIA 公开数值大多缺命令、量化或上下文；Intel 有较完整固定负载对照但样本少；AMD 样本最多。报告将主观样本分为 A/B/C 级，不应把社区反馈写成当前发行版保证。

## 固定条件性能对照与回退

下表优先列出有明确硬件和对照条件的记录。Issue/PR 编号用于回查原始讨论；这些记录的实现版本可能早于 0.9.0。

| 平台与硬件 | 模型与测试条件 | 结果 | 证据与限制 |
| --- | --- | --- | --- |
| RX 7900 XTX 24 GB；64 GB DDR5-5800；Ryzen 7 7800X3D | Qwen3.8 Flash Next UD-Q2_K_XL，73.45 GB，ctx 131K，Q8 KV，ubatch 1024，pp4096 路径 A/B | layer-major 20.57；chunk-major 229.09 tok/s，约 11.14 倍。PR 同类样本约 11.44–13.34 提至 169.44–205 tok/s | Issue #4 / PR #10；历史修复对照，不是 0.9.0 测量 |
| RX 7700 XT 12 GB；64 GB RAM | Ornith 1.5 35B-A3B，Q8 KV | serve 131K：41.5–42.7；run 131K：54.7；run 8K：50.4；Prefill 约 225–252 tok/s。调优后 Prefill 66→250、Decode 26→43 tok/s | PR #21；原始运行参数和重复数不完整 |
| Ryzen AI Max+ 395；128 GB LPDDR5X-8000 | Qwen3.6 35B UD-Q5_K_M，全驻留，Linux/RADV | MoE4All tg128 51.4、pp512 695.3 tok/s；llama.cpp ROCm tg128 约 63–85 tok/s | Issue #30；社区固定设备记录 |
| 同上 | Qwen3.8 Flash Next，约 74 GB，全驻留，Linux/RADV | MoE4All tg128 19.1、pp512 411.7；llama.cpp ROCm pp512 351.6 tok/s；开启 MTP 后 Decode 约 33 tok/s | Issue #30；配置细节见原报告 |
| 同上 | Qwen3.6 35B-A3B，23.77 GB expert，仅约 24% expert cache 覆盖 | tg64 约 1.3 tok/s；同机全驻留时 tg128 51.4 tok/s | Issue #31；分页/驻留差异，不是纯 GPU 算力对照 |
| 同上 | DeepSeek V4 Flash Q2，submit split 对照 | split=16 约 0.5；split=4096 约 1.8 tok/s | Issue #29/#30；约 3.6 倍差异 |
| Arc A770 15.9 GB；64 GB DDR4-3200 | Ornith 1.5 35B Q4_K_M，F16 KV，ctx 96K，四类任务 | v0.6：约 29.2–31.5；v0.7：约 22.7–25.7 tok/s，回退 12.0%–24.6% | Issue #41；v0.7 同时降低工作集与空闲 RAM 占用，需将内存收益和速度回退并列看待 |
| Radeon Pro R9700 31.9 GB；128 GB RAM | Qwen3.8 UD-Q4_K_XL，111 GB，ctx 64K，RAM budget 对照 | budget 55%：RAM 约 80 GB、加载约 1.3 GB/s；58%：full RAM store，RAM 127.7/128 GB、NVMe 100%，加载约 10 分钟 | Issue #34；启动/加载与系统压力数据，不是 Decode benchmark |

## 其他社区自报

以下数据有一定设备/模型信息，但缺少完整命令、版本、量化或重复样本，属于 B 或 B− 级历史样本：

| 硬件 | 报告模型/条件 | 自报结果 | 备注 |
| --- | --- | ---: | --- |
| RX 9070 XT + 32 GB DDR4 | Qwen3.8 Flash Q2 | 3–5 tok/s | 内存/显存紧张，可能触发外存路径 |
| RX 7700 XT + 64 GB RAM | Ornith 1.5 35B-A3B，OpenCode | 最高约 40 tok/s | 上下文和完整配置未给 |
| Ryzen AI Max+ 395，128 GB UMA | Qwen3.8 Flash，权重占用 80 GB+ | 27 tok/s | 参数未给 |
| RX 9070 + 32 GB DDR5 | Qwen3.8 27B IQ3_XS，108K | 约 30 tok/s | 报告显存占用约 14.8 GB |
| RTX 3090 Ti + 64 GB RAM | Qwen3.8 Flash Next | 约 29 tok/s | 模型量化、context 和命令未给 |
| 2× RTX 3080 20 GB + 32 GB DDR4 | Qwen3.8 | 约 5 tok/s | 报告称最初发生交换，之后调参；不可解释为双卡上限 |
| RTX 4090 48 GB | Qwen3.8 / Qwen3.6 35B | 约 40+ / 120 tok/s | 参数和上下文未给 |

原调查还包含 RX 6900 XT、其他 APU/UMA、NVIDIA 与非 MoE4All 引擎的记录。由于信息缺失或运行路径不同，它们保留在全量原件中，不混入上表作发行版性能承诺。

## 兼容性与稳定性观察

| 设备/问题 | 记录 | 截点状态 | 边界 |
| --- | --- | --- | --- |
| RX 6900 XT / RDNA2 | MoE arena 启动失败；host-visible heap 约 256 MiB，而完整 Prefill layer 需约 481.5 MiB | Issue #3 开放 | 明确的设备资源兼容边界，不是 Decode 回退 |
| Radeon Pro VII / Wave64 | Wave64 路径不受支持 | Issue #14 开放 | 架构兼容性问题 |
| RX 580 / Wave64 | 请求 Wave64 支持 | Issue #43 开放 | 未形成速度证据 |
| AI Max+ 395 / 128 GB UMA | 默认预算可令大模型负载造成系统严重卡顿；报告可运行设置为 RAM 50g + VRAM 15g | Issue #28 开放 | 资源预算与系统可用性风险 |
| RX 7900 XTX / Vision | v0.7 视觉请求崩溃报告 | Issue #42 开放 | 功能/稳定性问题；不可与文本吞吐混算 |
| RX 7900 XTX / ReBAR | 开启 BIOS Above 4G Decoding 与 ReBAR 后可运行 Qwen3.8 Q2 | Issue #9 已关闭 | 用户环境经验，不等于所有主板的要求 |

状态仅对应原报告 2026-09-24 截点，不代表此后 Issue 状态。后来是否已由代码修复，需按对应 tag 和同设备实测另行确认。

## 调研方法与边界

调研覆盖 5 个近期 B 站视频的主评论及楼中楼，并检查 GitHub 仓库截点的 28 个 Issue、16 个 PR 及 74 条讨论评论。页面累计显示 660 条评论，本次可读取 628 条不同评论；差额 32 条来自楼中楼计数与可读列表不一致，未对缺失内容作猜测。原件将样本按 A/B/C 级区分，并排除维护者目标值、理论带宽估算和缺少单位的数字。

完整逐条材料、全部 Issues/PR 清单、评论覆盖核验和未纳入主表的数据保留在本机原始调研文档中。本页是提炼后的结果；社区自报不能替代固定版本、本机同口径复测。
