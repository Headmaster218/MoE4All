# DeepSeek V4 缓存与模拟数据

[首页](../README.md) / [Reference](README.md) / DeepSeek V4 data

## Final measured run

- Code：performance head `f2fb30f`，compatibility `7935cf8`。
- Model：DeepSeek-V4-Flash-0731-AD-MXFP4，4 GGUF shards。
- Workload：tg128，ctx 16K，22 GiB VRAM budget，512 MiB reserve，40 GiB RAM。
- Submit：16；profile run 4.2 tok/s，adjacent unprofiled 4.3 tok/s。

| Item | Value |
|---|---:|
| Expert payload | 147.17 GB / 33,024 blocks |
| Block | 4.25 MiB |
| VRAM arena | 12.67 GiB / 3,052 blocks |
| RAM | 40 GiB / 9,637 blocks |
| GPU hits | 66,585 / 99,072 = 67.21% |
| RAM hits conditional | 20,018 / 32,487 = 61.62% |
| Combined hit | 87.41% |
| SSD demand | 12,469 blocks / 51.75 GiB / 128 tokens |
| SSD/token | 97.41 blocks / 0.404 GiB |
| Host→ReBAR | 134.83 GiB / 128 = 1.053 GiB/token |
| Distinct trace blocks | 14,217 |

## Cost model

在 12.67 GiB VRAM + 40 GiB RAM、4.3 tok/s 校准：

```text
token_ms = 72.57
         + 0.2396 * MXFP4_GPU_miss_blocks
         + 1.0170 * MXFP4_RAM_miss_blocks
```

`0.2396 ms` 对应 4.25 MiB / 18.6 GB/s RAM→ReBAR。模型未显式表达所有 nonlinear
overlap，因此只用于 trace 内容量比较。

## MXFP4 predicted Decode tok/s

| VRAM expert | RAM 45 | 47 | 50 | 60 | 75 | 100 | 110 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.67 GiB | 4.58 | 4.74 | 4.93 | 7.50 | 7.50 | 7.50 | 7.50 |
| 13.47 GiB | 4.63 | 4.78 | 4.98 | 7.61 | 7.61 | 7.61 | 7.61 |
| 14.00 GiB | 4.65 | 4.81 | 5.01 | 7.67 | 7.67 | 7.67 | 7.67 |
| 14.50 GiB | 4.68 | 4.84 | 5.04 | 7.75 | 7.75 | 7.75 | 7.75 |
| 15.00 GiB | 4.70 | 4.86 | 5.06 | 7.80 | 7.80 | 7.80 | 7.80 |

RAM total hit：45/47/50/60 GiB 分别 89.24%、90.13%、91.20%、100%（仅此 trace）。

## IQ3_M size-effect estimate

假设 expert bytes 和两项 transfer cost 都 ×0.873，compute 固定：

| VRAM expert | RAM 45 | 47 | 50 | 60 | 75 | 100 | 110 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.67 GiB | 5.61 | 5.94 | 6.36 | 8.21 | 8.21 | 8.21 | 8.21 |
| 13.47 GiB | 5.66 | 6.00 | 6.43 | 8.32 | 8.32 | 8.32 | 8.32 |
| 14.00 GiB | 5.69 | 6.03 | 6.46 | 8.37 | 8.37 | 8.37 | 8.37 |
| 14.50 GiB | 5.71 | 6.06 | 6.50 | 8.44 | 8.44 | 8.44 | 8.44 |
| 15.00 GiB | 5.75 | 6.10 | 6.54 | 8.51 | 8.51 | 8.51 | 8.51 |

RAM total hit：91.78%、93.25%、94.85%、100%。这是 size-effect，不包含 IQ3_M kernel
差异和量化改变 router 的影响。

## 为什么 60 GiB 后平台

128-token trace 的 distinct working set 约：

- MXFP4：59.0 GiB；
- IQ3_M size assumption：51.5 GiB。

所以 60/75/100/110 GiB 对这个固定短 trace 都是 100% RAM hit。完整 MXFP4 expert payload
仍有 147.17 GB，新 conversation 可以访问 trace 未出现 blocks。

## VRAM capacity status

- 12.67 GiB：final measured configuration。
- 13.47 GiB：成功加载过。
- 14.00 GiB：当前 load order 在最后约 414 MiB allocation 附近失败。
- 14.50/15.00 GiB：假设未来 post-load expansion 的理论点。

即使 12.67→15.00 GiB 且 SSD miss 消失，模拟只多约 0.30 tok/s。RAM capacity、block bytes
和 Host transfer 是更大的杠杆。

##  retained endpoint improvements

| Optimization | Before | After |
|---|---:|---:|
| HyperConnection Sinkhorn cache-hot | about 8.0 | 8.7 tok/s |
| All-role batch copy calls | 11,008 | 5,504 |
| All-role aggregate push | about 5.9 | 6.9 GB/s |
| MXFP4 complete-block endpoint | 3.9 | 4.3 tok/s |
| F32 GEMV `16384x24`, 688 dispatches | 79.5 ms | 23.7 ms |

---

[Reference](README.md) · [DeepSeek 模型页](../models/deepseek-v4-flash.md) ·
[Trace/simulation](../experiments/trace-simulation.md)
