# 原始证据索引

[首页](../README.md) / [Reference](README.md) / Evidence index

## 说明

Wiki 与原 `infr` 仓库物理分离。下列路径只是事实溯源位置，不是 Wiki 的内部导航依赖。
关键结论和表格已经复制/重写进本目录；即使原 target 清理，Wiki 仍可阅读。

源仓库根：`D:\AIinfr\infr`。

## Qwen 35B

| Source path | 用途 |
|---|---|
| `docs/perf/qwen36-rx7900xtx-optimization-history-20260819.md` | commit-by-commit 性能主记录 |
| `docs/perf/q8-kv-m8.md` | Q8 KV 初始瓶颈归因 |
| `docs/perf/qwen36-hd256-tuning.md` | split-K 与 hd256 参数决定 |
| `TWO_POOL_DECODE_CACHE_BENCHMARK.md` | 六池→Q5/Q6 global pool 与 ratio A/B |
| `report.md` | 120-case Q8/F16/depth/cache 矩阵 |
| `target/perf/current-full-matrix-20260820/results.csv` | 矩阵 raw result |
| `target/perf/two-pool-cache-20260819/` | pool/rate/raw logs |
| `target/perf/hw-bottleneck-20260818/report.md` | CPU/GPU utilization 与瓶颈归因 |
| `target/perf/current-prefill-matrix-354c0c3-20260819/` | Host Store/CPU push Prefill matrix |

## Unified VRAM / Embedding

| Source path | 用途 |
|---|---|
| `docs/unified-vram-acceptance-20260822.md` | 第一阶段 unified arena 验收 |
| `docs/unified-vram-elastic-acceptance-20260824.md` | fully elastic 验收 |
| `target/perf/unified-vram-api-results.json` | Chat/Embedding API result |
| `target/perf/unified-vram-embedding-parity.json` | llama.cpp parity |
| `target/perf/unified-vram-20260824-033627.stderr.log` | demand load/loan/restore 运行日志 |
| `target/perf/unified-vram-20260824-040332.stderr.log` | split-budget final smoke |
| `scripts/bench-embedding-parity.ps1` | parity runner |

## GUI

| Source path | 用途 |
|---|---|
| `crates/infr-gui/README.md` | 当前 GUI 功能与安全边界 |
| `crates/infr-gui/src/worker.rs` | child lifecycle、log normalization、rate parsing |
| `crates/infr-gui/src/catalog.rs` | model catalog 与 memory estimate |
| `crates/infr-gui/Start-INFR-GUI.ps1` | Windows launcher |

## DeepSeek V4

| Source path | 用途 |
|---|---|
| `docs/perf/deepseek-v4-flash-rx7900xtx-closeout-20260824.md` | campaign closeout 主记录 |
| `docs/deepseek.md` | architecture/correctness 研究 |
| `target/perf/dsv4-opt/cache-trace-run.log` | final profile source |
| `target/perf/dsv4-opt/gpu-access.csv` | raw route trace working copy |
| `docs/perf/deepseek-v4-flash-gpu-access-20260823.zip` | archived trace |

DeepSeek archive checksums：

- raw CSV SHA-256：`7e76123915d37f1c3294dc22586c48ee84015e52e41ddee2a387e64212ca0ed8`
- ZIP SHA-256：`cee06ddc744492ace1fd07e7f1de107e1153d98791253f33067183ddba02d594`

## Qwen 122B

| Source path | 用途 |
|---|---|
| `target/perf/qwen35-122b-cold-2048-20260824-112345.md` | cold→2K trace report |
| `target/perf/qwen35-122b-cold-2048-20260824-112345.analysis.json` | aggregate hit/miss analysis |
| `target/perf/qwen35-122b-cold-2048-20260824-112345.pager.csv` | 107,776,445-byte ordered trace |
| `target/perf/qwen35-122b-cold-2048-20260824-112345.pager.zip` | compressed trace |
| `target/perf/shared-slot-ab-20260824/` | shared fusion 35B/122B ABBA |
| `target/perf/moe-schedule-cost-final2-20260824-203548/report.md` | focused scheduling microbench |

Qwen trace checksums：

- CSV SHA-256：`9E6067B8EE60F8B7BBF8F1AD0B018B25BBC87D01DD3421FC886B114EB98FD51B`
- ZIP SHA-256：`B19D7F0767B3FD82F2E4DC12679C273AA316B6122B1227D55026F0C2203AA618`

## PCIe / DMA

| Source path | 用途 |
|---|---|
| `crates/infr-vulkan/tests/external_host_dma.rs` | ordinary RAM import、capacity、H2D/D2H probe |
| `target/perf/pcie-queue-duplex-20260824-220900.log` | queue family 单向/双向 matrix |
| `target/perf/pcie-duplex-d3d12.cpp` | D3D12 对照探索 |
| `crates/infr-vulkan/src/lib.rs` | proportional Host import 与 fallback |

## Source commit evidence

Git 范围：

```text
git log upstream/main..HEAD
merge-base = d7f320e7b8936fd6e1860115c5dd579c4572a27f
head at first Wiki build = 311ed4c
```

完整摘要见 [commit map](commit-map.md)。

---

[Reference](README.md) · [Benchmark method](benchmark-method.md) ·
[Commit map](commit-map.md)
