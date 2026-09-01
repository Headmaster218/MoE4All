# Decode Q5/Q6 Global Cache Pool

Date: 2026-08-20

Branch: `rgp-deep-optimization`

Target: APEX-I Balanced, Q8 KV, synthetic depth 200K, 7 GiB expert cache, RX 7900 XTX.

## Final design

The former six `(role, size)` caches have been replaced by two logical caches, one for Q5_K and
one for Q6_K. Each logical cache owns exactly one global `Pager`, LRU list and free-slot space.
Layer and role no longer constrain placement: any Gate, Up or Down block of the matching physical
size may occupy any slot in that logical pool.

Windows AMD cannot reliably map one ReBAR allocation larger than about 3 GiB, so logical storage
is backed by several physical arenas:

| Logical pool | Global slots | Physical arenas |
|---|---:|---:|
| Q5_K / 0.7 MiB | 5,637 | 3.00 GiB + 803.7 MiB |
| Q6_K / 0.9 MiB | 4,013 | 3.00 GiB + 220.7 MiB |

Arena boundaries are invisible to allocation and eviction. The LUT contains each resident block's
final 64-bit GPU address, so a shader can use a slot from any arena without a layer-to-arena mapping.

One shared batch epoch is opened before resolving Gate, Up and Down for a Router result. Every block
touched in that epoch is ineligible for eviction until the whole routed set has been resolved. Thus a
Down miss cannot evict this round's Gate/Up, and later Down misses cannot evict an earlier required
Down. Prefill/Decode mode changes clear and rebuild residency metadata so Decode regains the complete
arena rather than inheriting Prefill placement.

The final policy is plain global LRU. The experimental 8:7 Gate+Up/Down retention and paired
Gate+Up eviction were removed: they increased misses and reduced throughput.

## End-to-end benchmark

Common command shape: 1,000 Decode tokens, synthetic 200K KV, Q8 K/V, `ubatch=4096`, unlimited
submit, 7 GiB expert cache. Model loading occurs once per measurement and is excluded from t/s.

### Policy exploration

| Variant | Runs (t/s) | Mean | Final misses | Versus six-pool misses |
|---|---:|---:|---:|---:|
| Old six fixed role/size pools | 41.7, 41.3 | 41.50 | 283,067 | baseline |
| Global Q5/Q6 + 8:7 retention/pair eviction | 37.6, 37.7 | 37.65 | 299,264 | +5.72% |
| Global Q5/Q6 + plain LRU | 39.2, 40.6 | 39.90 | 283,428 | +0.13% |

The 8:7 policy was a real regression and was removed. Plain global LRU preserves the old miss
profile to within 0.13% while allowing every layer/role to use every slot.

### Down-only retention follow-up

Because the first 8:7 experiment also paired Gate+Up eviction, a second implementation isolated
the ratio question: Gate, Up and Down remained independent blocks; only Down received a soft
occupancy cap, and crossing the cap evicted exactly one oldest Down. The tested weights below mean
`Down:Gate:Up`; weight 8 is exact plain-LRU control.

| Weight | 500-token symmetric runs (t/s) | Mean | Final misses |
|---|---:|---:|---:|
| 8:8:8 | 40.9, 40.8 | 40.85 | 106,101 |
| 7:8:8 | 41.0, 41.3 | 41.15 | 106,382 |
| 6:8:8 | 40.9, 40.7 | 40.80 | 108,368 |
| 5:8:8 | 38.7, 39.5 | 39.10 | 114,750 |

The apparent 500-token advantage at weight 7 did not survive the full confirmation:

| Weight | 1,000-token crossed runs (t/s) | Mean | Final misses |
|---|---:|---:|---:|
| 8:8:8 | 40.6, 40.7 | 40.65 | 283,428 |
| 7:8:8 | 40.6, 40.5 | 40.55 | 283,858 |

Weight 7 is effectively neutral/slightly negative; weights 6 and 5 increasingly churn Down.
Therefore the ratio code was removed rather than left on the hot path. The final implementation
does not bind the three matrices: each remains independently addressable and evictable under one
global LRU, with only current-Router epoch protection coupling their lifetime temporarily.

### Interleaved confirmation

The server changed performance state during the run, so paired measurements in the same state are
more informative than comparing all samples as one population:

| Server state | Global pool | Old six pools | Delta |
|---|---:|---:|---:|
| Higher-throughput pair | 40.8 | 40.9 | -0.24% |
| Lower-throughput pair | 38.4 | 38.8 | -1.03% |

An intervening global-pool run measured 37.8 t/s, but its miss counters were bit-for-bit identical
to every other final global run. The immediately following old executable also fell to 38.8 t/s.
This confirms substantial system/GPU run-to-run variation rather than nondeterministic eviction.
The measured fixed end-to-end cost of the global layout is approximately 0-1% in paired runs.

## Pager profile

For an equal 960,000 cache lookup sample:

| Layout | Lookup time | Eviction time | Victim scan |
|---|---:|---:|---:|
| Old six pools | 370.8 ms | 73.9 ms | 1.0 step average |
| Global Q5/Q6 pools | 284.7 ms | 62.4 ms | 1.0 step average |

The larger global LRU itself is not a CPU bottleneck. Lookup and eviction bookkeeping became faster;
the small end-to-end cost is therefore in the new absolute-address LUT/GPU fixed path or ordinary
measurement noise, not in a long victim scan.

## Validation

- `cargo test -p infr-core pager --lib`: 50 passed.
- `cargo test -p infr-vulkan pager --lib`: 10 passed, 1 hardware-only teardown test ignored.
- Vulkan parity was checked for single-expert GEMV, multi-expert GEMV and paged MMQ address formats.
- Final pure-LRU APEX-I Balanced Prefill -> Decode smoke passed at 52.5 t/s for 16 Decode tokens.
- Logs confirm the full 23.57 GB CPU expert payload remains CPU-only; GPU-visible Host payload is zero.

Raw logs, CSV files, the old six-pool executable and experimental executables are under
`target/perf/two-pool-cache-20260819/`.
