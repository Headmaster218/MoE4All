# MoE4All IQ3_XXS 驻留与 Vulkan0 Prefill 预算

日期：2026-10-21  
目标模型：SC117 Qwen3.8-Flash-Next IQ3_XXS  
硬件：2 x RX 7900 XTX，Vulkan0/Vulkan2；62 GiB 可用主机内存

## Vulkan0 Prefill 迁移结果

命令口径：

- 模型：SC117 IQ3_XXS 两分片
- 设备：Vulkan0
- 上下文：30K
- 生成长度：32 token
- 预热：丢弃 1 轮
- 测量：3 轮
- 正确性路径：`kernels.vulkan.decode_gpu_resident=false`
- 提示：`iq3-divergence-trace-30k-r3`

| 轮次 | 预填 token/s | decode token/s | 输出 hash |
| --- | ---: | ---: | --- |
| 1 | 401.34 | 31.98 | `400972cd...` |
| 2 | 403.79 | 31.89 | `400972cd...` |
| 3 | 402.99 | 32.05 | `400972cd...` |

预填三轮极差约 `0.61%`，低于 5% 波动门。输出 hash 三轮一致。

注意：

- 这里的 hash 是 32-token 路径 hash，不能与 128-token 的 `49f384ee...` 直接比较。
- decode 约 32 tok/s 只在暖机后的 32-token 短跑成立，不作为完整 128-token decode G0。
- Vulkan0 prefill 约 `402.7 tok/s`，明显高于 Vulkan2 同路径的约 `220-237 tok/s`。

## IQ3_XXS 固定容量

| 项目 | GiB |
| --- | ---: |
| routed experts | 39.966 |
| 非专家张量 | 30.270 |
| 256K 双会话 KV | 9.285 |
| runtime/scratch | 5.200 |
| 安全预留 | 8.000 |

## 主机内存预算

256K 双会话、全部 expert 都放主机：

`39.966 + 9.285 + 5.2 + 8 = 62.451 GiB`

这已经比 62 GiB 可用内存多约 `0.45 GiB`，还没有计算非专家页缓存和碎片。

因此“256K 双会话 + 全 expert 主机常驻 + 非专家页缓存”不可行。

推荐方向：24 GiB hot expert 留在显存，其余 expert 放主机：

`15.966 + 9.285 + 5.2 + 8 = 38.451 GiB`

剩余约 `23.549 GiB` 可用于非专家页缓存、分配器碎片和系统波动。

## 显存 Hot Expert 三档

假设每个 4K prefill chunk 最坏需要接触全部冷 expert，主机到卡带宽 `5.2 GiB/s`：

| hot expert | 冷 expert 残余 | 纯搬运时间 | 理想 prefill 上限 |
| ---: | ---: | ---: | ---: |
| 16 GiB | 23.966 GiB | 4.61 s | 888 token/s |
| 24 GiB | 15.966 GiB | 3.07 s | 1,334 token/s |
| 32 GiB | 7.966 GiB | 1.53 s | 2,675 token/s |

但 32 GiB hot 档在 48 GiB 名义显存里加上 9.285 GiB KV 和 5.2 GiB runtime 后，只剩约 `1.515 GiB`，没有可靠余量，不能作为安全默认。

用当前 Vulkan0 实测效率修正：

- 全 paged 冷专家 39.966 GiB，理论 7.69 s；实测 30K prefill 402.7 token/s，对应 4K chunk 约 10.17 s。
- 当前搬运效率约为理想的 `75.6%`。
- 按这个效率，16 GiB hot 约需 6.10 s/chunk，上限约 671 token/s。
- 24 GiB hot 约需 4.07 s/chunk，上限约 1,007 token/s。
- 32 GiB hot 约需 2.03 s/chunk，上限约 2,021 token/s，仍低于 2,500 token/s。

结论：

- 只靠“热 expert + 冷 expert 流式”，IQ3_XXS 在 62 GiB RAM、48 GiB VRAM 下不能达到 2,500 token/s。
- 24 GiB hot 是容量上最平衡的一档，但预计只能把 prefill 推到约 1,000 token/s 量级，除非加入 CPU 计算、多卡同时驻留或更强压缩。
- 要达到 2,500 token/s，4K chunk 的 1.6 秒预算只允许搬运约 `8.32 GiB`。IQ3 expert 至少需要常驻约 `31.65 GiB`，这已经接近 full expert，当前两卡显存和 runtime 余量不够。

## 判定

1. Vulkan0 值得作为新的 prefill-lane 候选，但完整 G0-prefill 仍应使用相同路径做一次正式冻结。
2. IQ3_XXS 的 decode 仍然没有可交付性能路径；`P-hit-first-determinism` 继续是 decode 主线。
3. M2 不能只做“更大 ring”或“更多 stream”。容量预算已经证明，必须减少跨主机边界的 expert 字节数，或把相当一部分 expert 计算搬到 CPU/第二张卡。
4. 驻留预算诊断不受 decode 阻塞，可以继续进行；M2 实现仍需要额外门禁。

## 证据

- `benchmark-data/artifacts/strata-parity-iq3-correctness-20261021/30k-vulkan0-prefill-g0-warm1-r3/`
- `benchmark-data/artifacts/strata-parity-m1b-20261020/prefill-bandwidth-probe-vulkan0.log`
- `benchmark-data/artifacts/strata-parity-m1b-20261020/m2-p0-budget.md`
- `benchmark-data/artifacts/strata-parity-iq3-correctness-20261020/30k-smallm-static-g0-r2-128/`
- `benchmark-data/artifacts/strata-parity-iq3-correctness-20261020/30k-batched-static-g0-r2-128/`

