---
kind: benchmark
status: current
scope: strata-alignment phase 0 and phase 1
run_date: 2026-10-19
commit: perf/strata-align
evidence: measured
---

# 2026-10-19 Strata 对齐：Phase 0 基线与 Phase 1 状态驻留

## 范围

模型：Qwen3.8-Flash-Next-GSQ-RCO-abliterated IQ3_XXS。

配置：两路 `INFR_PIPELINE=Vulkan0,Vulkan2`、26/22 层切分、Q8 K/V、
`INFR_KV_OVERFLOW_VRAM_MB=0`、`-u256`、262144 context、单请求、128 token 输出。

Phase 0 是不改代码的基线。Phase 1 将 gated-DeltaNet conv/S state 与 PLE history
放到 `RecurrentState` device-local 类别，并在每张卡为这类固定状态预留 64 MiB；
该预留从层切分容量先扣除，如果 26/22 不再放得下会改写 layer band。

## Phase 0 基线

| 目标 | 实际 prompt | Prefill | Prefill 速度 | Decode 128 token |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 30,980 | 46.048 s | 672.8 tok/s | 12.753 s / 10.04 tok/s |
| 64K | 62,655 | 119.140 s | 525.9 tok/s | 13.607 s / 9.41 tok/s |
| 200K | 197,512 | 653.844 s | 302.1 tok/s | 18.823 s / 6.8 tok/s |

32K 第一次 fresh 请求为 11.786 s / 10.86 tok/s，第二次同前缀请求
只复用了 3 token，因此记为 10.04 tok/s 的稳态复测。

200K 第二遍同前缀测试被主动中止，避免继续消耗 11 分钟；以上是第一遍 fresh
请求的完整结果。

## Phase 1 结果

| 32K 指标 | 基线 | Phase 1 | 变化 |
| --- | ---: | ---: | ---: |
| Fresh decode | 10.86 tok/s | **22.40 tok/s** | +106.2% |
| 稳态复测 decode | 10.04 tok/s | **17.16 tok/s** | +70.9% |
| Prefill | 672.8 tok/s | **747.1 tok/s** | +11.0% |

Phase 1 的 32K 输入为 30,980 token，prefill 41.469 s；fresh 输出 128 token
耗时 5.715 s，稳态复测耗时 7.457 s。

启动日志仍选择 `26/22`，说明 64 MiB per-card reserve 在加入固定状态预留后
仍能保持原层边界，没有 OOM 或强制降级。

### 64K 深度

| 64K 指标 | Phase 0 | Phase 1 | 变化 |
| --- | ---: | ---: | ---: |
| Fresh decode | 9.90 tok/s | **18.06 tok/s** | +82.4% |
| 稳态复测 decode | 9.41 tok/s | **15.27 tok/s** | +62.3% |
| Prefill | 525.9 tok/s | **563.8 tok/s** | +7.2% |

实际输入 62,655 token。fresh prefill 111.135 s，输出 128 token 用 7.086 s；
稳态复测 prefill 110.761 s，输出 128 token 用 8.385 s。

### 32K 双 lane

`--parallel 2`、262144 context、32K prompt、每 lane 128 token：

- lane A：7.219 s，17.73 tok/s
- lane B：7.267 s，17.73 tok/s
- 按两 lane 的较长输出时间计算，聚合吞吐 **35.23 tok/s**，未达到 40 tok/s
- cohort ratio：1.619
- 并发结束后单 lane：5.883 s，21.76 tok/s

两条 lane 都正常完成，没有失败请求或状态污染。该结果比 Phase 0 的 1.748
cohort ratio 略低，但单 lane 稳态已明显提高；双 lane 尚未冲破 40 t/s，
State 驻留解决了 PCIe 主损失，但没有消除 dispatch 和双 deck 调度上限。

## 放置证据

Phase 0 启动日志：

```text
0 of 219 KV buffers in VRAM; remaining 219 (337.7 MiB) in SYSTEM RAM
```

Phase 1 启动日志：

```text
0 of 146 KV buffers in VRAM; remaining 146 (225.1 MiB) in SYSTEM RAM
```

减少的 73 个 KV buffer 和 112.6 MiB 正是从 KV host tier 移出的 recurrent state
与 PLE state。说明 Phase 1 不只是改了标签，实际放置已经改变。

## 结论

Phase 1 达到并超过 15-20 tok/s 的第一阶段目标。状态原本每次 decode 都经
PCIe 读取，现在固定在 device-local；这解释了 fresh decode 的 2.06x 和稳态
复测的 1.71x 提升。下一步仍应单独优化 prefill 路径和 dispatch 数，不应把本结果
误读成已经达到 Strata 的 47-50 tok/s。

`v6-state-resident` 标记在 Phase 1 代码提交 `89eab002` 上。双 lane 32K 聚合为
35.23 tok/s，因此该 tag 表示状态驻留里程碑，不表示双 lane 已经达到 40 tok/s。
