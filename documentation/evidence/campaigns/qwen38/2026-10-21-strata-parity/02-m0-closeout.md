# MoE4All M0 收尾报告

日期：2026-10-20  
计划书：`/home/qiankun/Documents/Codex/2026-10-19/new-chat/outputs/moe4all-strata-parity-plan.md`  
仓库：`/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All`  
证据根目录：`/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/benchmark-data/artifacts/strata-parity-m0-20261019/`

## A. 当前状态与提交信息

| 项目 | 值 |
| --- | --- |
| 分支 | `perf/strata-parity` |
| 提交 | `fb7c5784908e8cfe6754d32826a70ffa4ed902db` |
| 二进制 | `infr 1.0.3` |
| SHA-256 | `8ba3f1e4dbe85557607f0c86f0220097331c5dbce27193bb3df6359abd18e81e` |
| 实际 CPU | Intel i7-10700K |
| 计划 CPU | Ryzen 5 5600X |
| GPU | 2 x RX 7900 XTX；可用 Vulkan0 / Vulkan2 |
| 系统 | Ubuntu 22.04.5，Linux 6.8，RADV Mesa 23.2.1 |

状态：**正确性基线通过，性能基线按时间盒冻结。**

未提交内容已备份：

- `work/moe4all-m0-uncommitted.patch`，SHA-256 `f3cf5e6d3cf3af10e12c6579f57b39388982c74718af38dde51c6be026c864b8`
- `work/moe4all-m0-untracked.tgz`，SHA-256 `2c480fddfd453fb37caed2ac91f69e72fc9a0b68b6968c4c6aba655dbe9ec58d`

## B. M0-11 口径复核结果

| 项目 | 1.0 发布记录 | 当前 M0 |
| --- | --- | --- |
| 二进制 | `infr 1.0.0` | `infr 1.0.3` |
| 源提交 | `fabf3e614` | `fb7c578` |
| 系统 | Windows 11 | Ubuntu 22.04 / RADV |
| CPU | Ryzen 5 5600X | i7-10700K |
| GPU | 单卡 7900 XTX | 单卡 Vulkan2 |
| 模型 | Flash-Next AD-4.27bpw-Q4_K_M-M64，33 分片 | 同一模型目录和 33 分片 |
| KV | Q8 | Q8 |
| MTP | 关 | 关 |
| CPU miss | 关 | 关 |
| UBATCH | 4096 起步，可回退 | 4096 |
| 预热 | 1 轮 | 1 轮 |
| 输出长度 | 512 token | 32 token |
| 统计口径 | 完整区间，多槽位矩阵 | 固定唯一提示，单槽位 |

结论：**不是同一 A/B。**

- 当前不是 1.0 发布二进制的同源、同系统复现。
- 原来的 G0“30K 50.44 / 150K 44.16”必须重写为本机 1.0.3 基线。
- 如果要判定回归，必须单独构建 `fabf3e614` 二进制，在相同 Linux 主机、模型、prompt 和输出长度下做对照。

M0-12 已完成该对跑：

- `fabf3e614` 默认 release 构建，源提交一致。
- 旧二进制显示 `infr 0.11.0`，SHA-256 `82750cb29f46fc319dc2d768391c94b8f6e98a776fe99ab8265e05dfc14e32eb`。
- 30K 旧版预填 207.42/207.45/207.45，输出 42.70/42.71/42.38。
- 30K 当前版预填 207.36/207.48/207.47，输出 42.62/42.70/42.72。
- 两者 30K 哈希均相同。
- 判定：没有发现 1.0.3 预填回归；约 207 token/s 是本机环境/发布口径的共同结果。

## C. 4K/30K 最小基线与预热协议结果

预热协议：

- 1 轮丢弃预热。
- 同一固定提示重复测量。
- 页面缓存保持预热，不主动清空。
- GPU 性能档位未能固定：当前为 `auto`，设置 high 需要 sudo。
- 单卡 Vulkan2、单路、贪心、MTP 关闭、KV VRAM、CPU miss 关闭。

结果：

| 场景 | 预填三轮 | 输出三轮 | TTFT 三轮 | 哈希 |
| --- | ---: | ---: | ---: | --- |
| 4K，32 token | 214.1 / 214.9 / 214.7 | 40.08 / 39.97 / 40.04 | 18.4 / 18.3 / 18.3 s | 稳定 |
| 30K，32 token | 207.4 / 207.5 / 207.5 | 42.62 / 42.70 / 42.72 | 144.3 / 144.2 / 144.2 s | 稳定 |

30K 固定输入三轮哈希：

`bfccea41ad5424cee5ab0fc28542d7c82f591383f8902619e0c45b4b57b2f76f`

结论：

- 发布模型单路最小配置的 30K 正确性通过。
- 预填约 207 token/s，远低于发布记录的 1,414.75 token/s，但口径不同，不能直接判为回归。
- 输出约 42.7 token/s，与发布记录 50.44 token/s 接近，但仍低约 15%。

## D. 30K 哈希分叉隔离结果

### 发布模型

- 单卡、单路、KV VRAM、MTP 关闭、自动 GPU 热专家缓存。
- 1 轮预热 + 3 轮测量。
- 三轮输出哈希完全一致。
- 结论：最小发布模型配置不复现分叉。

### IQ3_XXS 模型

- 相同 30K 固定提示、单路、MTP 关闭。
- 三次哈希从第 56 个字符后分叉。
- token embedding dtype 是 `Iq3S`。
- 当前 `embed_gather_spv` 没有 `Iq3S` 实现。
- 结论：分叉和 GPU embedding 路径属于 IQ3_XXS 模型格式差异，不是发布模型 30K 基线的通用现象。

未完成：

- top-1/top-2 logit margin 未记录。当前服务响应只提供 timings、usage 和文本，没有逐 token logits margin；此项明确为**证据不足**。
- top-1/top-2 margin 需要在 runner 增加诊断输出，不属于本次 M0 架构改造范围。

## E. 并发和 MTP 缺 GPU embedding 定性

### 两路并发

- 发布模型 token embedding 为 `Q8_0`，GPU embed gather 支持，因此两路预填可以启动。
- 4K 两路请求均完成。
- 两路相同输入和 seed 的输出哈希不同：
  - lane 0：`5d347c7d...ee903`
  - lane 1：`d87f92b7...20d0a5`
- 同一 lane 在预热轮和测量轮内保持一致，说明是跨 lane 调度/归约不确定性，不是随机采样。
- 当前路径不支持“相同输入跨 lane bit-identical”；这是并发确定性问题，不是缺失 GPU embedding。

### MTP

- IQ3_XXS 的 `Iq3S` token embedding 没有 GPU gather，所以 MTP verify 失败。
- 发布模型 token embedding 为 `Q8_0`，具备 GPU gather。
- 当前发布模型目录没有 MTP sidecar，所以无法运行 MTP K=3 基线。
- 结论：MTP 不是 1.0 发布基线的一部分；它应移到后续前置实现任务，而不是继续阻塞 M0。

## F. M0-7 容量预算审计表

1.0 发布模型：

- 完整分片：88.034 GiB
- routed 专家张量：47.168 GiB
- 非专家张量：40.855 GiB
- 256K 单会话 KV：4.642 GiB
- 256K 双会话 KV：9.285 GiB

审计结论：

- “全部专家锁页”和“完整非专家权重”没有重复计数，它们是互斥的 tensor 集合：47.168 + 40.855 = 88.023 GiB。
- 全专家 + 双 256K KV = 56.453 GiB。
- 再加完整非专家权重的最坏值 = 97.528 GiB。
- 62 GiB 物理内存下，保留完整专家锁页后只剩约 5.5 GiB 给运行时、非专家和系统，不可行。
- 48 GiB 进程预算更不可行。

| 类别 | 内容 | 256K 双会话预算 |
| --- | --- | ---: |
| 必须常驻 | KV + 运行时 + 热专家 | 约 30.5 到 38.5 GiB |
| 可分页 | 其余专家 | 约 23 到 31 GiB |
| 可流式 | 非专家权重、冷专家 | 不全量常驻 |
| 预留余量 | OS、驱动、碎片、GC、scratch | 至少 8 GiB |

计划已删除“全部专家保存在锁页内存”，改为“热专家锁页 + 冷专家分页/流式”。

## G. M0-8/M0-9/M0-10 补强结果

### M0-8 CPU 专家

- 当前硬件：i7-10700K，不是 Ryzen 5 5600X；结论只对当前硬件成立。
- 发布模型真实权重探针：
  - 1 miss：p50 约 135 到 139 微秒。
  - 2 miss：p50 约 226 到 228 微秒。
- IQ3_XXS 合成工作集：
  - p50 114.7 微秒，p90 121.5 微秒，约 14.18 GiB/s。
- 四行 MTP CPU miss 被 `geometry` 明确拒绝。
- 测试 `m0_four_row_cpu_miss_is_explicitly_unsupported` 通过。
- 结论：单行 CPU 专家值得继续；四行和多路合并仍未实现。

### M0-9 双卡通信

- 发布模型宽 residual：`(1 + 4) * 2560 * 4 = 51.2 KiB/行`。
- 四行 residual：约 204.8 KiB。
- P2P 256 KiB：约 1.88 GiB/s，单次约 137 微秒。
- 主机中转 256 KiB：约 278 微秒。
- 30K 单 token 约 23.4 毫秒；一次 P2P 边界约 0.6%。
- 每个窗口只在层切分边界传 1 次 residual，不是每层 1 次。
- 结论：低于 20% 层时间阈值。若传完整 118 MB 状态，成本会显著超阈值。

### M0-10 回滚

- 118,038,528 字节状态。
- 1,000 次快照加恢复：约 680 毫秒。
- 每轮约 0.680 毫秒。
- 1,000 次后污染并恢复，字节完全一致。
- 结论：复制层可信；完整 KV 重写、窗口状态污染和调度同步仍未验证。

## H. 计划书修改摘要

已更新计划书：

- M0 状态改为“未通过/被阻断”。
- 加入 M0-11。
- 主要模型改为发布模型 33 分片；IQ3_XXS 降为格式诊断模型。
- 加入 1.0 发布模型 4K/30K 三轮结果。
- 加入并发跨 lane 哈希不同的事实。
- 把 MTP 从 M0 基线移到前置实现任务。
- 将全专家锁页改为热专家锁页 + 冷专家分页/流式。
- 更新容量表为发布模型 47.168 GiB 专家、9.285 GiB 双 256K KV。
- 加入“150K 预填瓶颈未定位则不重建 150K 基线”。
- 保留 `INFR_STRATA_PARITY` 一键回退设计。

## I. 当前入口任务下一步

1. 开始 M1a：为 segmented KV 增长增加主机回退。
2. 两个并发会话共享预算，VRAM 预留/claim 失败时退回主机。
3. 不移动已有 KV 页，不设计淘汰算法。
4. 验收两个 32K 会话增长成功，单 lane 输出哈希与 G0 一致。
5. 跨 lane bit-identical 继续作为独立优化项。

## J. 证据文件路径清单

新增关键文件：

- `m0-11-scope-review.md`
- `m0-7-budget-release-model.jsonl`
- `m0-8-release-model-probe.log`
- `m0-8-four-row-capability.log`
- `m0-9-dual-gpu.log`
- `m0-10-rollback.log`
- `release-model-4k-r3-warm1/`
- `release-model-30k-r3-warm1-v2/`
- `release-model-4k-p2/`
- `fabf3e614-4k-r3-warm1/`
- `fabf3e614-30k-r3-warm1/`
- `m0-12-fabf3e614-binary.sha256`
- `m0-12-fabf3e614-build.txt`

旧文件：

- `m0-8-cpu-experts.log`
- `m0-8-real-model-probe-failed.log`
- `hash-30k-r3-fixed-v2/`
- `hash-4k-r3-fixed/`
- `baseline-150k-r3-v2/`
- `m0-matrix-summary.json`

所有性能数字必须与对应目录中的 `manifest.json`、server 日志和请求输出一起读取。
