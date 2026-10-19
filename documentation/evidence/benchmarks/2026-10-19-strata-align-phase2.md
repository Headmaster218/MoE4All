---
kind: benchmark
status: current
scope: strata-alignment phase 2 QSA decode
run_date: 2026-10-19
commit: perf/strata-align
evidence: measured
---

# 2026-10-19 Strata 对齐 Phase 2：QSA decode 热点

## 结论

Phase 2 只保留两项数学不变的 QSA decode 优化：

- Qwen3.8 Q8 K/V gather 改为每线程一次处理 4 个 code，减少重复 scale load 和线程数。
- 单行 QSA score 默认改为 32-block / 1024-thread workgroup，dispatch grid 缩小 4 倍。

F16 block key 试验被否决：虽然 32K prefill 从 748.6 提升到 921.4 tok/s，但 decode
从 22.88 降到 19.20 tok/s，并且完整输出发生变化，不能进入稳定路径。

## 32K 单发

模型：Qwen3.8-Flash-Next-GSQ-RCO-abliterated IQ3_XXS。

配置：`INFR_PIPELINE=Vulkan0,Vulkan2`、26/22、262144 context、Q8 K/V、
`INFR_KV_OVERFLOW_VRAM_MB=0`、`-u256`、128 token 输出。

- 最终默认路径：prefill **721.5 tok/s**，decode **23.12 tok/s**。
- 同轮旧 decode-tile 对照为 22.2-22.9 tok/s；提升主要来自 score dispatch
  grid 缩小，QSA gather ×4 为独立的正收益。
- QSA score tile 16 没有收益；tile 32 与基线输出逐字节一致。

## 双 lane 回归

同一 32K prompt，双 lane、262144 context、26/22、`-u256`：

- 每 lane decode：20.59 / 20.66 tok/s。
- cohort predicted wall：6.218 s。
- 聚合 decode：**41.17 tok/s**。

两条 lane 均正常完成，无请求失败、无状态污染。

## 正确性

`cargo test -p infr-vulkan --test qsa_parity -- --nocapture`：
3 passed。Q8 gather ×4 与参考 K/V 对拍通过；tile 32 的完整 128-token 输出
与旧 tile 逐字节一致。

## 未采用项

- 4 lane 只在短 prompt 上得到 61.47 tok/s；32K prompt 的 4 路 prefill 需要额外
  25.2 MiB MoE activation scratch，不能在不牺牲 256K 能力的前提下容纳。
- F16 block key 因 decode 回退和输出漂移撤销。
- QSA block cache 仍为 F32 host tier；逐算子聚合显示 score 仍是最大单点，
  下一步若要继续必须解决 host-tier block key 的系统级放置或无损压缩问题。
