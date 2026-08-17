# Qwen3.6 hd256 FlashAttention tuning

This note records the M5 parameter sweep performed after the hd256 BM16
FlashAttention path and its activation-memory fix landed.  The target was a
Radeon RX 7900 XTX using the Windows Vulkan driver.  The model family was
Qwen3.6-35B-A3B with f16 KV, a 512-token prefill, ubatch 512, and synthetic KV
depth.  Submit splitting was disabled so watchdog fences did not distort the
comparison.

## Split-K sweep

The APEX-I-Balanced model was first swept with two repetitions per point:

| KV depth | automatic | 1 split | 2 splits | 4 splits |
|---:|---:|---:|---:|---:|
| 100k | 329.3 t/s | 329.4 t/s | 344.3 t/s | 330.2 t/s |
| 200k | 228.4 t/s | 224.6 t/s | 225.9 t/s | 229.1 t/s |

The 100k result suggested a possible two-split win, but whole-model runs are
noisy because expert paging dominates wall time.  Cross-model confirmation did
not establish a safe global default: IQ4_NL_XL improved only 409.1 to 413.3
t/s at 100k, while the Q4_K_XL pair was unstable and regressed.  At 200k,
automatic and four-split results were effectively tied for all three models.

Per-op GPU profiling isolated the attention kernel from expert-paging noise:

| KV depth | comparison | hd256 attention time | result |
|---:|---|---:|---:|
| 100k | automatic vs 2 splits | 71.4 vs 65.3 ms | 2 splits 8.5% faster |
| 200k | automatic vs 4 splits | 124.1 vs 124.4 ms | tied |

`flash_splits` is a global override rather than a depth- and device-specific
policy.  Forcing the isolated 100k winner would therefore also change shapes
and devices for which it was not a winner.  M5 keeps the automatic policy and
retains `kernels.vulkan.flash_splits` / `INFR_FLASH_SPLITS` as an explicit A/B
override.

## Tile and address-path decisions

- Keep the dedicated hd256 BM16 tile.  It fits the Windows driver's 32 KiB
  shared-memory limit and is the production geometry validated by M2/M3.
- Do not add an unmeasured hd256 BM8 or BM32 variant merely to create another
  switch.  A new tile is justified only by an isolated kernel result.
- Keep `kv.coopmat_bda` off by default.  The current option applies only to
  hd128 FlashAttention; hd256 continues to use bound K/V buffers.  The existing
  hd128 RDNA3 measurement also shows the BDA cooperative-matrix load regressing
  to about 0.8x because it does not produce an efficient scalar base address.

Raw logs are under
`target/perf/qwen36-moe-m5-tuning-20260817` in the benchmark workspace and are
intentionally not versioned.
