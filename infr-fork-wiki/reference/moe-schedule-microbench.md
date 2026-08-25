# Decode MoE 计算与搬运微基准

[首页](../README.md) / [Reference](README.md) / MoE schedule microbench

## Scope

focused `infr-vulkan` integration test，使用 production paged expert kernels、
`MoePagerSession`、full/inclusive RAM promotion 与 `FileBlockIo`，不编译/运行 CLI、模型图、
KV、Attention 或 server。

- 35B pools：Q5_K 0.688 MiB、Q6_K 0.820 MiB / matrix。
- 122B pools：IQ4_XS 1.594 MiB、Q5_K 2.063 MiB、Q6_K 2.461 MiB。
- compute：shared + routed，2～9 total experts。
- transfer：1～8 experts，UGD/UG/D，RAM/SSD。
- SSD：disjoint first-touch 1 GiB windows，排除 cached rerun。
- incremental compile 2.66 s；5 filtered runs 合计 14.1 s。

## Full UGD compute（µs）

| Model/pool | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 35B Q5_K | 18.6 | 19.8 | 23.4 | 23.4 | 26.2 | 28.3 | 30.8 | 32.7 |
| 35B Q6_K | 32.4 | 28.0 | 31.7 | 36.3 | 34.0 | 37.6 | 40.0 | 44.4 |
| 122B IQ4_XS | 34.5 | 45.7 | 57.6 | 72.2 | 86.4 | 95.8 | 108.0 | 120.0 |
| 122B Q5_K | 26.2 | 33.8 | 39.0 | 45.9 | 52.9 | 59.6 | 64.5 | 68.9 |
| 122B Q6_K | 29.5 | 36.2 | 45.6 | 53.6 | 62.8 | 72.3 | 78.0 | 87.9 |

非严格单调的小波动来自 microsecond 级 GPU dispatch/timestamp noise。结论是完整 9-expert
compute 仍不超过 120 µs；IQ4_XS 虽小，kernel 最慢。

## Transfer fit

Median 拟合 `T = fixed + bytes / bandwidth`：

| Tier | Fixed | Sustained bandwidth | R² |
|---|---:|---:|---:|
| 35B full RAM | about 0～3 µs | 14.8 GiB/s | >=0.9996 |
| 122B inclusive RAM | 14～51 µs | 14.3～14.7 GiB/s | >=0.9977 |
| SSD UGD | 0.31～1.08 ms | 3.69～4.29 GiB/s | >=0.9908 |
| SSD split UG/D | 0.50～2.06 ms | 3.66～5.39 GiB/s | 0.8867～0.9816 |

## One/eight expert UGD medians

| Model/pool | RAM 1 | RAM 8 | SSD 1 | SSD 8 |
|---|---:|---:|---:|---:|
| 35B Q5_K | 136 µs | 1.088 ms | — | — |
| 35B Q6_K | 164 µs | 1.297 ms | — | — |
| 122B IQ4_XS | 353 µs | 2.616 ms | 1.689 ms | 9.514 ms |
| 122B Q5_K | 442 µs | 3.317 ms | 2.753 ms | 14.236 ms |
| 122B Q6_K | 523 µs | 3.992 ms | 2.553 ms | 15.240 ms |

## UG → D split costs

- first submit：25～50 µs；
- record/submit hand-off：20～50 µs；
- resident UGD 拆成 UG+D：额外 54～118 µs；
- 35B routed UG compute：9～22 µs；
- 122B routed UG compute：14～70 µs；
- RAM D promotion：35B 45～431 µs；122B 109～1348 µs；
- SSD D promotion：122B 0.94～4.92 ms。

UG compute 不足以 cover 一个 D promotion。CPU 可以先 record 后续 D，而不必阻塞 wait UG；
但 D promotion/submit 若在 UG 完成前未到 queue，GPU 仍 bubble。

## Scheduler simulation conclusion

对 wait-all、resident-first all-UGD、split-Down 比较：

- resident-first 在多 miss/122B 场景常可隐藏数百 µs；
- split-Down 从 RAM 通常慢约 0.1～0.75 ms；
- split-Down 从 SSD 通常慢约 0.6～3.1 ms；
- 少数 noisy positive endpoint 小于 SSD run variance；
- 等“一半 misses”再加第三 compute segment 也无法藏住完整 promotion。

最终通用策略：shared + resident UGD 先算；同时批量 promote all miss UGD；再算 miss UGD。

## 与 Host DMA 微基准的区别

本表是 production Pager path 的 RAM/SSD promotion，约 14～15 GiB/s。独立 Vulkan imported
host copy 对 20～40 MiB payload 可到 23～25 GiB/s。后者证明数据通路有优化空间，不会让
每个 1-expert transfer 自动达到 25 GiB/s；固定 submit、region count、未 import suffix 和
queue dependency 仍存在。

---

[Reference](README.md) · [MoE scheduling](../architecture/moe-scheduling.md) ·
[Host DMA](../architecture/host-dma.md)
