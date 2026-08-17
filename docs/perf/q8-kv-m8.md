# M8: Q8 KV decode profile

Date: 2026-08-17  
Device: AMD Radeon RX 7900 XTX (Windows Vulkan)  
Models: Qwen3.6-35B-A3B APEX-I-Balanced, UD-Q4_K_XL, UD-IQ4_NL_XL

## Method

- `infr bench`, Vulkan0, `ubatch=512`, `ctx=210k`.
- Synthetic KV depths 0, 100k, and 200k; the deep cases do not prefill from zero.
- Expert cache is 10 GiB at 0/100k and 8 GiB at 200k.
- K and V use the same format in each A/B: either f16 or Q8_0.
- Per-op runs use 40 decode tokens and `INFR_PROF_OPS=1`.
- Throughput runs disable profiling and use 60 decode tokens, 3 reps.

Raw logs are under `target/perf/m8-q8-kv-profile-20260817/` (ignored build output).

## Stable decode throughput

| Model | Depth | f16 tok/s | Q8 tok/s | Q8 / f16 |
|---|---:|---:|---:|---:|
| APEX-I-Balanced | 100k | 22.8 | 22.7 | 99.6% |
| APEX-I-Balanced | 200k | 18.7 | 17.1 | 91.4% |
| UD-Q4_K_XL | 100k | 24.3 | 23.7 | 97.5% |
| UD-Q4_K_XL | 200k | 22.3 | 20.0 | 89.7% |
| UD-IQ4_NL_XL | 100k | 24.4 | 23.9 | 98.0% |
| UD-IQ4_NL_XL | 200k | 22.2 | 20.0 | 90.1% |

Q8_0 uses 34 bytes per 32 elements (1.0625 B/element), versus 2 B/element for
f16, so the cache retains a 46.9% byte saving.

## Device-time attribution

The coupled planar-Q8 path does not materialize an f16 cache and has no separate
dequant dispatch. It decodes in `attn_partial_q8_bda`; consequently the Q8 decode
cost and the attention cost are the same timestamp bucket.

Across all three models the attention kernels are effectively model-independent:

| Depth | f16 `attn_decode_hd256` | Q8 `attn_partial_q8_bda` | Q8 / f16 |
|---|---:|---:|---:|
| 100k | 406.8 ms | 535.0 ms | 131.5% |
| 200k | 807.5 ms | 1.04 s | 128.8% |

These totals cover 40 tokens and 10 attention layers (400 dispatches). The Q8
write path is negligible: `store_q8_f16` is about 0.8 ms and `store_q8` below
0.1 ms for the same run. Pager and expert kernels are not the Q8 regression.

## M9 decision

Proceed with a narrow planar-Q8 attention optimization. The existing layout is
already planar and inline-dequantized, but each vec4 decoder independently loads
and unpacks the scale shared by eight vec4s in a 32-element block. Preserve the
layout, f16 path, non-Q8 paths, BDA/bound variants, and Linux subgroup behavior.
The acceptance target remains at least 90% of f16 decode at long context while
retaining the Q8 cache-size advantage.
