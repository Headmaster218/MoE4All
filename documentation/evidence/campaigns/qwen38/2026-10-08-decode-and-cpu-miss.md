---
kind: campaign
status: historical
scope: qwen38-decode-cpu-miss
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# 普通 Decode、精确 QSA 与 CPU miss 专项

## 范围与采用结果

2026-10-08，RX 7900 XTX、Flash-Next AD 4.27bpw、Vulkan、MTP off。提炼同日 Decode optimization 与 CPU miss feasibility 两份累积报告，保留实验编号和否决结论；各轮有自己的 binary/cache/shape，不把所有收益相加成一个升级幅度。本阶段未新测 MTP、90K、错峰到达或 24 小时稳定性。

最终 Decode 组合 `e1d6b0abd`+`08ca51a5f`：
真单 slot 30K 两轮 39.6362/38.8876，150K 36.8906/37.1766 tok/s。
3 configured/2 active、150K overlap off 为 45.8209/45.6614，auto 为 46.6976/46.8521，配对 +1.91%/+2.61%，平均约 +2.26%。18 requests 中 6 warmup、12 measured，输出哈希一致；单 slot 实际 ubatch 4096，三 slot 双活跃 3584。不能据此宣称 45/90 tok/s 目标全部达到。

## E1..E16 决策保留

| 实验 | 方向 | 结果与取舍 |
| --- | --- | --- |
| E1 | mapped expert streaming stores | memcpy 约 14.3..14.6 GiB/s，无稳定端到端收益，否决 |
| E2 | non-imported tail 可复用 host staging | WDDM import/分配限制，长三路约 -9%，否决 |
| E3 | HC launch fusion | 保留 scalar Q8 HC injection；multirow 2 ULP 差异未过严格 parity，不采用 |
| E4 | IQ2_S shared-expert output grouping | 后被 E5 替代；早期并发 no-op 不作收益证据 |
| E5 | 去掉 IQ2_S workgroup grid staging | 保留 immutable IQ codebooks/NR4，384 条件精确 parity |
| E6 | Q8_0 RM4 投影 | cold HC 回退，否决 |
| E7 | cohort-sized Q8 特化 | 无模型端到端收益，否决 |
| E8 | Wave32 tree reduction | 精确但 occupancy/端到端不利，否决 |
| E9 | Gate/Up 与 Down promotion overlap | 保留 eligible deep 2/3-row；浅 context 不一律启用 |
| E10 | paged IQ2_S wide fetch | 768 条件精确但更慢，否决 |
| E11 | shallow Q8 四输出 | 24 条件精确，收益不稳定，否决 |
| E12 | QSA prepare fusion | eligible BF16 出现输出 hash 分歧且无速度收益；默认 off，不复活 |
| E13 | shallow IQ4_NL WG32 | scalar cold +0.94%、warm -1.15%，三路约 -2.2%，否决 |
| E14 | HC up + mix fusion | 精确 parity；反序 1024-token on/off 均值仅 +0.12%，否决 |
| E15 | exact integer QSA prefix scans | 159 条件精确；最终限于测过的 Windows RDNA3 deep scalar/top512 |
| E16 | register/subgroup chronological sort | 159 条件精确、barrier 更少仍回退：39.8372/39.1045 → 38.7867/38.2823，均值 -2.37%，否决 |

E15 早期 scalar 30K +3.11%、150K +3.21%，但 warm fixed-prefix 控制只有 +0.65% 且一轮负值，因此没有据早期 screen 对全部设备/cohort 默认开启。自动 gate 按最终矩阵收窄。数值正确是必要条件，不是收益证明。

## 时间线瓶颈

每 cohort 在排除 warmup 后采24个steady Decode units、分6窗口；这是三个cohort的有界refresh，不是新六格trace矩阵。无dropped event或query unavailable警告。单位均ms，除wall P90外为P50。

| shape | step wall P50 | wall P90 | Main envelope | mapped copy | prepare outside Main | encode outside Main |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30K、1/1 slot | 27.27 | 34.68 | 14.75 | 0.98 | 0.65 | 1.54 |
| 30K、1/3 slot | 27.47 | 31.76 | 14.82 | 0.97 | 0.64 | 1.56 |
| 150K、3/3 slot | 63.76 | 74.20 | 30.85 | 9.44 | 6.95 | 2.51 |

长三路 expert prepare 约 10.2 ms，其中 mapped copy 约 9.44，outside Main mapped 约 6.72；平均 DMA 343.42 MiB/13.09 ms，重叠约 4.77 ms。这些分类有嵌套和 overlap，不能相加或相减独立 P50 来预测提升。profiler 开销约 2.13/2.20/3.59 ms，clean 速度不能直接用 trace wall 代替。Main envelope包含依赖等待，不等于纯kernel计算；pending也不是全部driver开销。此次shape/sampling与原trace不同，DMA字节减少不能未经matched A/B就归因于优化。

优先方向由证据支持为 host mapped tail、router-to-hit 等依赖交接及 long-context main work；仅看 GPU 功耗/利用率不足以判定唯一瓶颈。分析方法见[时间线手册](../../../development/performance/decode-timeline.md)。

## 独立的 GPU router-to-hit ABBA

`07c7b0e97` 缩短 paged Decode router-to-hit handoff。3 slots、30K/150K、1/2/3 active、Q8、greedy/no-think；每 arm/cell 四轮 512-token 请求，第一轮排除，old/new/new/old，warm prefix；全部输出 hash 一致。

| context / active | old tok/s | new tok/s | 配对变化 |
| --- | ---: | ---: | ---: |
| 30K / 1 | 40.718 | 43.228 | +6.165% |
| 30K / 2 | 60.530 | 63.446 | +4.817% |
| 30K / 3 | 69.701 | 71.026 | +1.901% |
| 150K / 1 | 38.429 | 40.568 | +5.566% |
| 150K / 2 | 49.496 | 49.923 | +0.863%，接近噪声 |
| 150K / 3 | 45.659 | 46.989 | +2.912%，old 漂移约 3.2% |

真正单 slot ABBA 40.6132 → 43.1349，+6.209%。三 slot arena 约 16.30 GiB、25471 expert blocks、14.23 GiB runtime claims，请求 ubatch 4096 实际 fallback 3584。不要将这一表作为下段 CPU offload 提升。

## CPU miss 尝试与最终判断

完整 RAM、4 个物理 worker、packed F32 input、IQ2 Gate/Up+IQ4_NL Down 的单 miss。孤立 FFN P50 约 117 µs，未达到约 70..80 µs full/40 µs compute 的目标。早期 Rayon/SpinPool 与 Host DMA 关闭控制更慢，promotion 也存在启动时机成本。

| context | GPU before | CPU reference | CPU host result | CPU poll | host+poll | GPU after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30K | 44.0364 | 42.8167 | 44.0853 | 41.6371 | 43.5102 | 43.9839 |
| 150K | 41.0289 | 未测此项 | 40.9979 | 未测此项 | 未测此项 | 40.9703 |

host result 30K 约 +0.17% 是噪声级，150K 持平；因此默认 off。报告早期说“CPU 尚未提交”是当时状态，后续 `a1f0a8e3f`/`cd44a8215` 纳入实验 task pool/offload，`cba30dd50` 提供向导入口。代码支持 1..3 miss，不意味着该报告已验证 2/3 miss 的收益。

原始报告为忽略目录中的 `decode-optimization-20261008.md` 与 `cpu-miss-feasibility-20261008.md`；失败 shader/patch/EXE 不进入当前生产实现。历史测试曾临时把 KV quota 提到 48 GiB，只是当时条件；新测试统一遵守 30 GiB 根目录缓存规则。
