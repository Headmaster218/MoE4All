# Qwen3.5-122B Trace 与阶段数据

[首页](../README.md) / [Reference](README.md) / Qwen 122B trace

## Cold-start → 2K Decode

- Date：2026-08-24。
- Code：`7eb9f0d`。
- Model：`Qwen3.5-122B-A10B-APEX-I-Quality.gguf`。
- VRAM budget：20 GiB。
- Bounded RAM：51 GiB（54.76 GB displayed）。
- KV：Q8_0/Q8_0，context 4096。
- 70 prompt + 2048 Decode，cold process。

### Endpoint

| Item | Value |
|---|---:|
| Wall time | 193.139 s |
| Server Decode | 11.2 tok/s |
| Final 5-s windows | mostly 11.7～12.5 tok/s |
| Output | correct normal prose，finish=`length` |

### Geometry

| Item | Value |
|---|---:|
| Expert payload | 72.36 GB |
| GPU pool | 15.53 GB / 7910 slots |
| RAM tier | 54.76 GB |
| Pool 0 | 1.7 MB blocks；4613 GPU / 16272 RAM |
| Pool 1 | 2.2 MB blocks；1647 GPU / 5811 RAM |
| Pool 2 | 2.6 MB blocks；1646 GPU / 5811 RAM |

### Ordered access

| Phase | Calls | Accesses | GPU hit |
|---|---:|---:|---:|
| Startup warmup | 96 | 2,304 | 11.849% |
| Request Prefill route | 48 | 79,488 | 79.506% |
| Decode | 98,304 | 2,359,296 | 69.573% |
| Total | 98,448 | 2,441,088 | 69.842% |

每个 Decode call 为 24 records（8 Experts × 3 roles），48 layers/token；identity：

```text
2048 * 48 * 8 * 3 = 2,359,296
```

无 backward call id、无 sequence discontinuity。

### Cache/Tier

| Item | Value |
|---|---:|
| Request GPU misses | 734,155 |
| Request RAM conditional hits | 709,162 / 734,155 = 96.596% |
| SSD reads | 24,993 blocks |
| SSD / all request accesses | 1.025% |
| SSD traffic | about 50.227 GB |
| SSD blocks/generated token | 12.20（含 request Prefill route） |

Decode GPU hit windows：

| Generated tokens | GPU hit |
|---:|---:|
| 0–127 | 72.125% |
| 128–511 | 71.248% |
| 512–1023 | 70.309% |
| 1024–1535 | 67.428% |
| 1536–2047 | 69.088% |

没有随 warmup 单调升到 90%；路由 working set 持续变化。

## Role/pool 分布

三 roles hit 几乎一致：Down 69.844%、Gate 69.840%、Up 69.842%。这支持“不必为 role
写固定 quota”的结论。

| Pool | Accesses | Hit rate |
|---|---:|---:|
| 0 | 1,423,968 | 74.711% |
| 1 | 508,560 | 61.683% |
| 2 | 508,560 | 64.367% |

pool 差异来自容量比例、block size 和层分布；也是有限 Host import 不能 first-come 的原因。

## Shared fusion ABBA

条件：tg256、Q8 KV、15.58 GB mapped ReBAR、54.76 GB RAM、3/5 reps。

| Model | Off A/B | On A/B | 结论 |
|---|---:|---:|---|
| 122B | 17.0 / 17.3 | 19.2 / 19.2 | shared fusion +11.5～12.9% |
| 35B guard | 85.7 / 86.5 | 90.0 / 90.6 | 通用路径未退步，约 +4.7～5.7% |

## Host DMA 阶段

| Path | tg256 | Notes |
|---|---:|---|
| CPU/旧 host path after fusion | 19.2 | bounded RAM |
| Host DMA, first-come import | 22.4 | finite import coverage 偏向早期 pool |
| Host DMA, proportional | **23.2** | 3 reps，22.7～23.5 |

最终：13.97 GiB GPU expert arena、45 GiB bounded RAM；driver import 28.99/45.00 GiB；
三 pool 覆盖 62.6%/67.7%/64.9%。

23.2 是短 tg256 性能验收，不能替代 cold 2K trace 的 SSD-backed 11.2。

---

[Reference](README.md) · [Qwen 122B 模型页](../models/qwen35-122b.md) ·
[Trace/simulation](../experiments/trace-simulation.md)
