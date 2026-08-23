# DeepSeek V4 Flash / RX 7900 XTX closeout

Date: 2026-08-24

Branch: `feat/deepseek-v4-flash`

Feature baseline: `b007730` (`feat(deepseek4): add FP8 KV and MXFP4 indexer`)

Performance head: `f2fb30f` (`perf(vulkan): vectorize decode f32 GEMV`)

Compatibility closeout: `7935cf8` (`fix(pager): preserve full-host down overlap`)

Model: `DeepSeek-V4-Flash-0731-AD-MXFP4`, four GGUF shards

Machine: Radeon RX 7900 XTX 24 GiB, Windows Vulkan

## 1. Final status

DeepSeek V4 Flash now loads and generates through the native `infr` graph with the shipped
compressed-attention topology, FP8 KV, MXFP4 indexer state and paged MXFP4 experts. DSpark is not
implemented. This performance campaign is closed at about **4.3 decode tokens/s** for the measured
40 GiB bounded-RAM / SSD configuration. The original 17--25 tokens/s target is not reachable with
that storage hierarchy: the remaining dominant cost is expert-cache miss traffic, not one missing
compute kernel.

The retained policy is an **inclusive Host shadow**. Every VRAM-resident expert keeps immutable RAM
bytes when the RAM budget permits. This duplicates the VRAM working set inside the RAM budget, but
turns VRAM eviction into metadata only. Reading an evicted expert back from mapped ReBAR was measured
at only about 44 MB/s, so a partially exclusive hierarchy would either stall on GPU reads or rebuild
victims from SSD.

No benchmark was run after `7935cf8`: that commit only restores the pre-existing complete-Host-Store
Down-overlap path while leaving the measured bounded-RAM/SSD DeepSeek path unchanged.

## 2. Reproducible final run

The cache trace came from this command shape:

```text
infr bench -p 0 -n 128 -r 1 --ctx 16k --dev Vulkan0 \
  --set device.vram_budget=22g \
  --set device.vram_reserve=512m \
  --set paging.dram=40g \
  DeepSeek-V4-Flash-0731-AD-MXFP4-00001-of-00004.gguf
```

Environment: `INFR_PAGER_PROFILE=1`, `INFR_PAGER_TRACE=<path>`,
`INFR_SUBMIT_DISPATCHES=16`. The profiling run reported 4.2 t/s; the adjacent unprofiled final run
reported 4.3 t/s.

Final measured cache state:

| Item | Value |
|---|---:|
| Expert payload | 147.17 GB, 33,024 blocks |
| One MXFP4 expert-role block | 4.25 MiB |
| VRAM expert arena | 12.67 GiB, 3,052 blocks |
| Inclusive RAM budget | 40 GiB, 9,637 blocks |
| GPU hits | 66,585 / 99,072 = 67.21% |
| RAM hits, conditional on GPU miss | 20,018 / 32,487 = 61.62% |
| Combined VRAM-or-RAM hit rate | 87.41% |
| SSD demand | 12,469 blocks = 51.75 GiB / 128 tokens |
| SSD demand per token | 97.41 blocks = 0.404 GiB |
| Host-to-ReBAR traffic | 134.83 GiB / 128 tokens = 1.053 GiB/token |

The trace contains exactly 99,072 accesses over 128 tokens, or 774 block accesses/token, and 14,217
distinct block ids. The raw CSV is archived as
[`deepseek-v4-flash-gpu-access-20260823.zip`](deepseek-v4-flash-gpu-access-20260823.zip); inside it is
`gpu-access.csv` with columns `block_id,hit,evicted`.

- Raw CSV size: 957,327 bytes
- Raw CSV SHA-256: `7e76123915d37f1c3294dc22586c48ee84015e52e41ddee2a387e64212ca0ed8`
- ZIP size: 230,592 bytes
- ZIP SHA-256: `cee06ddc744492ace1fd07e7f1de107e1153d98791253f33067183ddba02d594`
- Original ignored working path: `target/perf/dsv4-opt/gpu-access.csv`
- Source profile log: `target/perf/dsv4-opt/cache-trace-run.log`

## 3. Commit audit and impact on other models

| Commit | Result | Scope outside DeepSeek V4 |
|---|---|---|
| `b007730` | Added the full V4 FP8-KV/MXFP4-index execution path. | Existing models do not emit the new ops; defaults leave their graphs unchanged. |
| `4343c03` | Fixed automatic cache sizing and reduced resident-weight arena fragmentation. | The 1.5 GiB load-driver reserve is V4-only. Adaptive 64/128/256 MiB resident-BDA blocks are generic and can reduce allocations/fragmentation for other Vulkan models; the first block remains 64 MiB for small models. |
| `99e6e40` | Added the one-token, `hc=4` parallel Sinkhorn kernel. A-B-B-A moved about 8.0 to 8.7 t/s in the short cache-hot test. | Only graphs emitting gated HyperConnect with exactly one row and `hc=4` select it; currently V4-specific. |
| `082e1eb` | Opened independent Windows file handles so concurrent positioned reads can really overlap. Standalone fanout 1 vs 4 was within noise. | Generic to Windows SSD-backed block I/O; no effect on fully resident/full-Host-Store models beyond a few open handles. |
| `f3af338` | Added concurrent bounded-host promotions and parallel RAM/SSD-to-ReBAR work. | Generic to models using the bounded inclusive RAM/SSD tier. Full Host Store and resident models do not enter it. |
| `074e388` | Batched Gate+Up promotion decisions for one shared size pool. | Generic to non-fused paged MoE using the bounded tier. Correctness/LRU order remains serial; only independent byte movement is parallel. |
| `f6ca35c` | Extended the bounded-tier batch to Gate+Up+Down. In the final sequence it reduced copy calls from 11,008 to 5,504 and raised the observed aggregate push bandwidth from about 5.9 to 6.9 GB/s; the one-run endpoint moved 3.9 to 4.0 t/s. | Originally also selected the complete Host Store path and suppressed its useful Down overlap. `7935cf8` now confines the three-role batch to bounded RAM/SSD, protecting Qwen/Ling full-store Decode. |
| `495fc9a` | Decodes one complete MXFP4 block per GEMV tile, sharing its scale/address work. Final `tg128` moved 3.9 to 4.3 t/s. | Applies to any Vulkan MXFP4 decode matrix, not to Q/K/IQ/F16 formats. |
| `f2fb30f` | Vec4-load F32 decode GEMV cut the profiled `16384x24` total from 79.5 to 23.7 ms for the same 688 dispatches (about 3.35x kernel speedup). End-to-end remained 4.3 t/s because paging dominated. | Generic to Vulkan one-row F32 linears whose input width is divisible by four; parity is checked at reassociation tolerance. Other shapes retain their old kernels. |
| `7935cf8` | Restored Down-copy/Up+Gate overlap for the complete Host Store while preserving V4's bounded-tier batch. | Compatibility fix specifically preventing a Qwen/Ling-style full-store regression. |

The new architecture ops and DSV4 shaders are therefore isolated. The changes with intentional
cross-model reach are the resident-BDA allocator, Windows block I/O, bounded host pager, MXFP4
decode and one-row F32 GEMV. Of those, only the pager scheduling had an obvious incompatible
performance interaction, and it is gated by `7935cf8`.

## 4. Retained optimizations and rejected experiments

Retained because they had a real isolated or end-to-end benefit:

- V4-safe automatic VRAM planning and adaptive resident-weight packing.
- Decode-specialized HyperConnect Sinkhorn.
- Concurrent bounded-host promotion and all-role batching.
- Complete-block MXFP4 dequantization.
- Vec4 F32 decode GEMV.

Measured experiments not retained as performance changes:

| Experiment | Observation | Decision |
|---|---|---|
| Wider MXFP4 `dqblk` mapping | 6.8 vs 6.9 t/s in A/B | No benefit. |
| Unlimited submit batching | 6.8 vs 6.9 t/s; other runs regressed | Keep split/16. |
| DP4A MXFP4 variants | Small early gain, but 5.9--6.3 t/s in later comparable runs | Inferior to shared block decode. |
| F16 HyperConnect temporaries | 7.3--7.5 vs 8.2--8.3 t/s F32 | Negative. |
| Split HyperConnect into four dispatches | 8.2 vs 8.3--8.4 t/s | Dispatch cost cancels parallelism. |
| Non-temporal ReBAR writes | 2.7--2.8 t/s and unstable | No stable gain. |
| Windows per-piece handle fanout alone | 2.7--2.9 t/s | Not sufficient without request-level batching. |
| Eight-way request fanout | 3.7 vs 3.8 t/s | Too much overhead. |
| Drop inclusive GPU shadows | 3.9 vs 4.0 t/s, RAM conditional hit 60.6% vs 61.6% | Negative and makes eviction recovery expensive. |
| Flat outer parallel copy | 3.9 vs 3.9 t/s | Neutral. |

Numbers from different `tgN`, cache sizes or profiling modes are not compared as speedups. In
particular, the short cache-hot 7--16 t/s results are diagnostic and are not the final SSD-backed
throughput.

## 5. Full-shadow simulation

The simulator repeats the recorded route trace to steady state and runs independent exact LRUs for
VRAM and inclusive RAM. It is calibrated to 4.3 t/s at 12.67 GiB VRAM + 40 GiB RAM:

```text
token_ms = 72.57
         + 0.2396 * MXFP4_GPU_miss_blocks
         + 1.0170 * MXFP4_RAM_miss_blocks
```

`0.2396 ms` is one 4.25 MiB block at the separately measured 18.6 GB/s RAM-to-ReBAR rate. For the
IQ3_M estimate, block bytes and both transfer terms are multiplied by 0.873 while compute/kernel
time stays fixed. It is therefore a conservative **size-effect estimate**, not an IQ3_M kernel
benchmark.

### MXFP4 predicted decode tokens/s

| VRAM expert cache | RAM 45 GiB | 47 GiB | 50 GiB | 60 GiB | 75 GiB | 100 GiB | 110 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.67 GiB | 4.58 | 4.74 | 4.93 | 7.50 | 7.50 | 7.50 | 7.50 |
| 13.47 GiB | 4.63 | 4.78 | 4.98 | 7.61 | 7.61 | 7.61 | 7.61 |
| 14.00 GiB | 4.65 | 4.81 | 5.01 | 7.67 | 7.67 | 7.67 | 7.67 |
| 14.50 GiB | 4.68 | 4.84 | 5.04 | 7.75 | 7.75 | 7.75 | 7.75 |
| 15.00 GiB | 4.70 | 4.86 | 5.06 | 7.80 | 7.80 | 7.80 | 7.80 |

RAM total hit rates are 89.24%, 90.13%, 91.20% and 100% at 45, 47, 50 and
60-or-more GiB respectively.

### IQ3_M size-effect estimate, predicted decode tokens/s

| VRAM expert cache | RAM 45 GiB | 47 GiB | 50 GiB | 60 GiB | 75 GiB | 100 GiB | 110 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.67 GiB | 5.61 | 5.94 | 6.36 | 8.21 | 8.21 | 8.21 | 8.21 |
| 13.47 GiB | 5.66 | 6.00 | 6.43 | 8.32 | 8.32 | 8.32 | 8.32 |
| 14.00 GiB | 5.69 | 6.03 | 6.46 | 8.37 | 8.37 | 8.37 | 8.37 |
| 14.50 GiB | 5.71 | 6.06 | 6.50 | 8.44 | 8.44 | 8.44 | 8.44 |
| 15.00 GiB | 5.75 | 6.10 | 6.54 | 8.51 | 8.51 | 8.51 | 8.51 |

RAM total hit rates are 91.78%, 93.25%, 94.85% and 100% at 45, 47, 50 and
60-or-more GiB respectively.

The 60 GiB plateau is a limit of this trace, not proof that 60 GiB contains the model. The trace's
distinct working set is about 59.0 GiB in MXFP4 and 51.5 GiB at the assumed IQ3_M size, while the
complete MXFP4 expert payload is 147.17 GB. New routes in a longer or different conversation can
use the extra 75/100/110 GiB capacity; this 128-token trace cannot quantify them.

The current 12.67 GiB VRAM arena is conservative. 13.47 GiB has loaded successfully. 14 GiB failed
near the final 414 MiB allocation under the current load order; 14.5--15 GiB are theoretical
post-load expansion points. Even the theoretical 12.67-to-15 GiB increase is only about 0.30 t/s
once SSD misses disappear, so RAM capacity and smaller expert blocks are the larger levers.

## 6. Closeout decision

- Keep the inclusive full shadow; do not reintroduce GPU-to-RAM eviction copies.
- Treat 13.47 GiB as the largest already demonstrated VRAM expert arena; larger rows above are
  simulations, not configuration recommendations.
- Keep the archived route trace as the baseline for future cache-policy simulations.
- Resume model-performance work on Ling. DeepSeek V4 remains supported, but further speed work
  should wait for a materially different storage configuration or a longer representative route
  trace rather than more one-off kernel tuning.
