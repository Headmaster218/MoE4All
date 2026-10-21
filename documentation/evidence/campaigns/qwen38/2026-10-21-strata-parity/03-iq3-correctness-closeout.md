# MoE4All IQ3_XXS 30K 正确性结案报告

日期：2026-10-20  
仓库：`/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All`  
分支：`perf/strata-parity`  
HEAD：`dca2cc4 perf(moe): expose Iq3S hit-first opt-in`  
二进制：`target/release/infr`  
SHA-256：`b4ed21345dc1089d03bbc69dcc0e5eb6a8fb5d7776131b4b66d7ffa8383409fe`

## 结论

IQ3_XXS 的 30K 静默分叉根因已定位：

- 不是 embedding。
- 不是 KV。
- 不是 MoE 路由。
- 不是 gate 权重。
- 不是 expert slot 内容。
- 不是 IQ3S small-m GEMV 的数学实现。
- 根因在 GPU-resident decode 的 hit-first 主 journal/conditional 编排。

当前状态：

- 正确性 G0 已建立：resident controller + `hit_first=false`，完整 30K 128-token 两轮输出 hash 一致。
- 性能 G0 未建立：正确路径只有 `0.34/0.60 tok/s`，其他可测正确路径也远低于 20 tok/s。
- hit-first 速度快，但不正确，只能作为性能参考或独立诊断，不能进入默认或验收。

## 关键 A/B

| 路径 | hidden 一致性 | 文本 hash | decode 速度 | 判定 |
| --- | --- | --- | --- | --- |
| resident hit-first 默认 | 第 2 层 `ffn_raw` 从第 1 个 decode 位置起不同 | 不稳定 | 快 | 不通过正确性 |
| resident + `hit_first=false` | 1152/1152 一致 | 128-token 两轮 `49f384ee...` | 0.34/0.60 tok/s | 正确性通过，性能不通过 |
| static small-m | 1152/1152 一致 | 128-token 两轮 `49f384ee...` | 0.34/0.60 tok/s | 正确性通过，性能不通过 |
| static batched MMQ | 8-token 前缀两轮一致 | `13a3fc68...` | 冷/热约 0.2/7.0 tok/s | 正确性前缀通过，性能不通过，完整 128 未跑完 |

## 根因证据

第 2 层、`pos=29894..29899`：

- MoE 输入整行 hash 相同。
- top-10 expert id 和顺序相同。
- 10 个 gate weight 完全相同。
- `hit_mask=1023`，没有 miss。
- gate/up/down 每个命中 slot 的 64 点分层内容 hash 完全相同。
- 物理 slot id 不同。
- `ffn_raw` 在第 1-4、6 个位置不同，第 5 个位置偶然相同。

这说明：

- slot 内容正确。
- hit-first 的 masked/reordered 主执行路径会让同一逻辑计算产生不同结果。
- miss resume 单独修复不足；实验补丁没有改善 hidden，已撤销。

## 默认行为

- 含 Iq3S expert bank 的 resident graph 自动关闭 hit-first，其他量化格式保持原默认。
- 正确性兜底提交：`2296f6b fix(moe): disable hit-first for Iq3S resident decode`。
- 诊断 opt-in 提交：`dca2cc4 perf(moe): expose Iq3S hit-first opt-in`。
- opt-in 配置：`kernels.vulkan.decode_gpu_resident_iq3s_hit_first=true`。
- opt-in 只能用于复现性能或继续修 hit-first，不能默认开启。

## 证据目录

根目录：

`/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/benchmark-data/artifacts/strata-parity-iq3-correctness-20261020/`

关键子目录：

- `30k-token-trace-gpu-r3/`
- `30k-token-trace-host-argmax-sametag-r3/`
- `30k-layer-stage-trace-host-argmax-r3/`
- `30k-layer-moe-route-trace-v2-r3/`
- `30k-layer6-moe-weight-trace-r3/`
- `30k-layer6-moe-slot-trace-r2/`
- `30k-layer6-moe-slotsample-trace-r2/`
- `30k-smallm-static-slots-ab-r2/`
- `30k-batched-moe-static-prefix-ab-r2/`
- `30k-resident-no-fastlut-prefix-ab-r2/`
- `30k-resident-fastlut-no-hitfirst-prefix-ab-r2/`
- `30k-resident-no-hitfirst-only-prefix-ab-r2/`
- `30k-iq3s-no-hitfirst-default-r2/`
- `30k-smallm-static-g0-r2-128/`

## 下一步

1. 冻结本报告作为 IQ3_XXS 正确性 G0，不把它当性能 G0。
2. 下一阶段只推进可测的 Vulkan0 prefill 迁移和驻留预算。
3. `P-hit-first-determinism` 独立立项，范围为 hit-first 主 journal/conditional 的执行顺序和内存依赖，预计 2-5 天。
4. M2 实现继续受 decode/正确性门约束；M2 驻留预算、路由统计和容量诊断不受约束。

## 2026-10-21 补充

已完成：

- static batched 128-token 两轮：hash 一致 `f7c0f88b...`，但该 hash 与 small-m G0 `49f384ee...` 不同；decode `0.72/1.25 tok/s`，不满足临时性能门。
- Vulkan0 30K prefill 三轮：`401.34/403.79/402.99 tok/s`，32-token hash 三轮一致 `400972cd...`。
- IQ3_XXS 驻留与压缩预算表：见 `moe4all-iq3-budget-vulkan0-20261021.md`。
- `P-hit-first-determinism` 独立分支和首个实现提交：`perf/hit-first-determinism`, `2d771ef`。

首个 hit-first 实现提交未通过 hidden 一致性，不能替代 `hit_first=false` 正确性兜底。它只证明初次 all-hit 标志不是唯一分叉点。

当前结论：没有 `G0-decode-temp`。P-hit-first 获得 2 天硬时间盒；失败则转 M2 最小实现，并把 M2 目标降级为 prefill 从约 401 提升到约 1,000 token/s。

