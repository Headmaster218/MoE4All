# Qwen3.6 MoE / RX 7900 XTX optimization record

Date: 2026-08-19  
Branch: `rgp-deep-optimization`  
Measured implementation: `354c0c3` (`vulkan:cpu-push-MoE-host-store-into-ReBAR-pools`)  
Primary model: Qwen3.6-35B-A3B APEX-I-Balanced  
Machine: Radeon RX 7900 XTX 24 GiB, Ryzen 5 5600X, Windows Vulkan

## 1. Scope and measurement rules

This document reconstructs the optimization work from the first repeatable synthetic-context
baseline through the current layer-granular Prefill memory architecture. The first four commits in
the table below were already present as benchmark/pager infrastructure; the performance work after
`16fbee2` was developed and measured in this optimization series.

Rules used when interpreting the numbers:

- `--synthetic-depth` materializes benchmark KV state directly. A 250K test does not execute a
  250K-token Prefill first.
- Prefill and Decode are separate projects. A kernel or scheduling change is not assumed to help
  both.
- Only like-for-like A/B results are treated as a speedup. Results with different cache sizes,
  declared context windows, KV formats, batch sizes, or profiling enabled are labelled separately.
- Short one-repetition results are diagnostic. Three- or five-repetition runs and A-B-B-A runs are
  preferred for final conclusions.
- Diagnostic profiling is not required by the production path and can be fully disabled.

## 2. Current architecture

### Host memory

- The complete 23,571,988,480-byte MoE expert payload exists once in physical host RAM.
- Model loading arranges it permanently as `Layer -> Role -> Expert`, split into 14 chunks only at
  complete layer boundaries.
- Every expert retains an offset index for Decode.
- Runtime Prefill performs no GGUF reread, gather, expert packing, or role reordering.
- The full Host store is CPU-owned and is not exposed as a complete GPU-visible/shared-VRAM mirror.

### GPU memory and execution

- Windows AMD rejected one mapped allocation spanning the complete logical cache. The cache is
  therefore six `(role, expert-size)` ReBAR pools.
- Each pool is `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` and mapped once.
- Prefill interprets the pools as fixed resident whole layers plus A/B whole-layer streaming lanes.
- Decode reuses the complete same arena as expert-level LRU slots and pushes one indexed expert at
  a time from the same Host store.
- Prefill performs direct CPU writes into final ReBAR locations. There is no full staging mirror and
  no GPU `copy_buffer` upload.

At 200K, 7 GiB cache, batch 4096:

- 8/40 layers are resident; 32 are streamed.
- 17.63 GiB is pushed at 19.18 GB/s in about 0.99 s.
- All 32 next-layer pushes start while GPU work is live.
- 13/32 pushes finish before that individual compute segment finishes. A/B overlap is real, but
  one-layer prefetch depth cannot hide every layer whose upload window is longer than its compute
  window.

## 3. Current Prefill matrix

Command shape: `infr bench MODEL -p BATCH -n 0 --synthetic-depth DEPTH -r 3 -u BATCH`, Q8 K/V,
Vulkan0, `INFR_SUBMIT_DISPATCHES=0`. The benchmark itself performs its normal discarded warm
repetition before the three timed repetitions.

Cache is fixed within each depth column and selected so the 8192 batch also fits safely:

| Synthetic KV depth | Declared context | Expert cache | Resident layers |
|---:|---:|---:|---:|
| 0 | 16K | 8 GiB | 9/40 |
| 64K | 80K | 8 GiB | 9/40 |
| 128K | 140K | 7 GiB | 8/40 |
| 250K | 260K | 2500 MiB | 0/40 |

Throughput in tokens/s:

| Batch | 0 | 64K | 128K | 250K |
|---:|---:|---:|---:|---:|
| 512 | 432.1 | 409.5 | 335.6 | 225.5 |
| 1024 | 830.7 | 681.5 | 506.2 | 315.3 |
| 2048 | 1499.7 | 972.5 | 626.3 | 376.5 |
| 4096 | 2654.1 | 1170.1 | 717.8 | 409.3 |
| 8192 | 3762.1 | 1037.8 | 762.8 | 437.0 |

Interpretation:

- At depth 0, increasing the microbatch strongly amortizes fixed per-layer paging/submission cost;
  8192 reaches 3762.1 t/s.
- At 64K, 4096 is the measured optimum. Moving to 8192 adds enough causal-attention and activation
  work to fall from 1170.1 to 1037.8 t/s.
- At 128K and 250K, the existing-KV attention term dominates more of the run. Larger batches still
  improve throughput, but with sharply diminishing returns.
- The 250K column is a valid pure A/B streaming result: the minimum practical cache leaves zero
  resident layers, while the 8192 batch still fits and reaches 437.0 t/s.
- The columns are operational capacity points, not a cache-isolated context sweep. In particular,
  250K necessarily has less expert cache than the shallower columns.

### Exact 4K cross-checks

The declared context window changes memory placement even at synthetic depth 0, so two exact
configuration cross-checks were run after the matrix:

| Depth | Context | Cache | Current `354c0c3` | Relevant predecessor | Change |
|---:|---:|---:|---:|---:|---:|
| 0 | 8K | 8 GiB | 2977.8 t/s (3 reps) | layer staging 2672.5 | +11.4% |
| 200K | 210K | 7 GiB | 515.5 t/s (3 reps) | GPU-pull Host store 492.6 | +4.6% |

Against the older pre-Host-store layer-staging result at 200K (522.3 t/s), the current 515.5 t/s is
effectively flat (-1.3%). The CPU-push change recovered the GPU-pull regression; it did not create a
material new long-context speedup because long attention and the remaining per-layer windows are
already dominant.

## 4. Commit-by-commit record

| Commit | Change | Result / reason retained |
|---|---|---|
| `ebf5b79` | Opt-in pager profiling; correct explicit submit-cap handling | Measurement infrastructure. Made cache lookup, copy, upload, wait, submit and LRU costs visible without production overhead. |
| `8c49710` | O(1) LRU promotion | Removed repeated list scans on hits. Later profiles showed promotion scans at zero; cold victim selection remained separate. |
| `898ff91` | Correct benchmark KV-format reporting | Correctness/reproducibility only; prevents f16/Q8 results being mislabelled. |
| `16fbee2` | Synthetic context depth | Enabled 100K/200K/250K tests without spending hours generating the prefix. This is the received repeatable baseline. |
| `dbc51fe` | Independent hd256 BM16 FlashAttention Prefill and combine path | Main early Prefill win. Balanced pp512 rose from about 164 to 320 t/s at 100K and 89 to 229 t/s at 200K. It removed the old large score-matrix path and used an hd256 geometry that fits the Windows 32 KiB shared-memory limit. |
| `9bef28d` | Reclaim hd256 FA activation reserve | Corrected capacity accounting after the score matrix disappeared. Measured peak was about 540 MiB; final reserve 548 MiB. It primarily returned usable VRAM/context rather than changing throughput. |
| `95b8ffa` | Record hd256 tuning decision | Documentation: keep BM16 and automatic split policy; do not force an unproven global tile/split. |
| `447cd50` | Strided DeltaNet Decode | Removed CopyStrided-style work on the supported path. End-to-end gains were small/mixed (roughly 0-2% in stable deep cases), but the isolated path is cheaper and compatibility fallbacks remain. |
| `276d9c8` | IQ4_NL partial-tile mask | Correctness and forward compatibility for ragged expert tiles. No broad throughput claim. |
| `0ffdefd` | Subgroup expert Decode for paged quants | Kernel-level coverage for paged quant formats. End-to-end 100K A/B was flat for APEX/Q4_K_XL and about +1% for IQ4_NL_XL, so it was not counted as a major whole-model win. |
| `3c9523a` | Q8 KV Decode profile | Established that Q8 attention, not KV writes or pager work, caused the long-context Q8 deficit. Q8 used 46.9% fewer KV bytes but was initially about 9-10% slower than f16 at 200K. |
| `46c0b88` | Dedicated hd256 Q8 Decode path | Scale/block-aware Q8 specialization. At 200K it improved APEX 17.1 -> 18.4, Q4_K_XL 20.0 -> 21.9, IQ4_NL_XL 19.8 -> 21.5 t/s; 100K was essentially flat. |
| `a73d43a` | hd256 register-O FlashAttention Prefill | Large long-context win by keeping output accumulators in registers and avoiding score/output traffic. Balanced 200K 226.8 -> 275.2 (+21.3%); three-model gains were about 20-23% at 200K. |
| `ff69e83` | Recycle transient recorder resources | Removed repeated Vulkan buffer/descriptor/recorder allocation from the hot loop. The associated Decode restructuring was the largest Decode step: Balanced reached 44.6 t/s at 100K and 32.8 at 200K, from roughly low-20s beforehand. |
| `5a33e58` | Prefer generic hd256 Decode on RDNA3 | The nominally specialized kernel was slower on RDNA3. Short crossover: generic 56.2 vs specialized 54.0 at 100K, 42.8 vs 41.9 at 200K. Final 3-rep: 47.9/33.9 t/s. |
| `02e0bfb` | Split hd256 Prefill output across four lanes | Restored output-dimension occupancy. Relative to the preceding Prefill baseline, Balanced pp512 moved roughly 487 -> 509 at d0, 367 -> 417 at 100K and 279 -> 300 at 200K. |
| `d72f60f` | Cluster Q8 QK work by quant block | Reduced repeated scale/code handling and changed the depth split from 32 to 8. 200K A-B-B-A: 31.55 -> 33.95 t/s (+7.6%). |
| `e6b6137` | Wider Q8 hd256 workgroups | Increased workgroup occupancy/parallel reduction. LS64 -> LS128: 34.0 -> 34.7 t/s (+2.1%). |
| `633638b` | More Q8 Decode wave parallelism | LS128 -> LS256: about 34.35 -> 35.0 t/s (+1.9%). |
| `84fb844` | Packed fp16 Q8 QK accumulation | Reduced register/bandwidth pressure in QK. 35.1 -> 36.35 t/s (+3.6%). |
| `0b37574` | Pack Q8 value dequantization | Reduced PV-side value traffic/conversion cost. 36.3 -> 37.2 t/s (+2.5%). |
| `15eb5f7` | Widen Q8 hd256 Decode chunks to 1024 | Halved combine work while retaining pass-1 occupancy after the D8/LS changes. 37.3 -> 39.65 t/s (+6.3%). The complete retained Q8 kernel sequence improved about 31.55 -> 39.65 (+25.7%) at 200K. |
| `b8cbc52` | Configurable multi-slot pager upload pipeline | Allowed host preparation, upload and GPU work to overlap instead of serializing each expert batch. Four/eight slots were productive; odd 3/5-slot layouts were much worse. |
| `6afab3a` | Amortize cold-batch victim scans | Replaced repeated cold victim walks with batch bookkeeping. It addresses the earlier 114-128 average victim-scan steps without changing hit semantics. |
| `3651e29` | Batch paged expert staging copies | Copies independent expert blocks in parallel and removes per-expert scheduling overhead. At 5 GiB cache, Q8 pp512 reached 412.3 t/s at 200K; at 16 GiB and d0 it reached 823.1 t/s. |
| `ce97e4f` | Layer-major MoE Host store and Prefill A/B layout | Introduced load-time layer-contiguous storage, resident whole layers, A/B lanes, and separate Decode expert LRU. It removed Prefill expert LRU decisions, but its GPU-pull/staging implementation regressed d0 2672.5 -> 2438.9 and 200K 522.3 -> 492.6 at batch 4096. The architecture was retained because it enabled direct contiguous transfer. |
| `354c0c3` | CPU-push the unique Host store into mapped ReBAR pools | Removed the full GPU-visible Host allocation and GPU copy path. Exact d0 batch-4096 improved 2672.5 -> 2977.8 (+11.4%). At 200K it recovered GPU-pull 492.6 -> 515.5, effectively tying the older 522.3 staging result. Physical full expert payload is now one Host copy. |

## 5. Major baseline-to-milestone comparisons

### Received f16 baseline to `02e0bfb`

These are matched model/batch/cache/depth measurements (Balanced, batch 512):

| Operation | Depth | `16fbee2` | `02e0bfb` | Change |
|---|---:|---:|---:|---:|
| Prefill | 0 | 437.3 | 509.4 | +16.5% |
| Prefill | 100K | 164.3 | 416.8 | +153.7% |
| Prefill | 200K | 89.1 | 299.8 | +236.5% |
| Decode | 0 | 13.8 | 31.6 | +129.0% |
| Decode | 100K | 20.5 | 47.7 | +132.7% |
| Decode | 200K | 18.3 | 33.6 | +83.6% |

The deep Prefill jump is mostly hd256 FlashAttention plus register-O; the Decode jump is mostly
recorder/resource reuse and removal of per-layer CPU/GPU synchronization, not one attention tile.

### Pager/Prefill data-path progression

The following rows are not all one cache configuration; they show the measured capability reached
at each architecture step, with the relevant configuration stated:

| Stage | Workload | Result | Main limiting factor / conclusion |
|---|---|---:|---|
| Original expert pager | pp512 d200K, Q8 | about 301 t/s | Host copy, victim scans and serialized upload gaps. |
| Multi-slot + batched staging | pp512 d200K, 5 GiB | 412.3 | Much better overlap; still copies/gathers expert payload. |
| Large expert cache | pp512 d0, 16 GiB | 823.1 | Hit rate/cache capacity strongly controls shallow Prefill. |
| Layer mode, old staging | pp4096 d0 / d200K | 2672.5 / 522.3 | Layer mode itself is near-neutral at long context, but removes expert-LRU orchestration. |
| Layer Host store, GPU pull | pp4096 d0 / d200K | 2438.9 / 492.6 | Extra GPU-visible Host/staging path regressed both. |
| Unique Host + CPU ReBAR push | pp4096 d0 / d200K | 2977.8 / 515.5 | Strong d0 recovery/win; long context remains attention-dominated. |

## 6. Experiments that were rejected or left non-default

These attempts matter because repeating them would spend time rediscovering the same constraint.

| Experiment | Result | Why rejected / next condition that could change the result |
|---|---:|---|
| Dynamic/fixed submit-cap search (64-768) | Best candidates only +0.34% at 100K and +0.47% at 200K | Below run drift and the 1% allowed tuning overhead. Keep `unlimited` on discrete GPU with existing TDR protection. |
| Adaptive benchmark warmup | 28.6 -> 41.4 -> 60.0 t/s across variants, still 25-28% spread | It warmed pager/cache state rather than converging. Reverted. |
| Forced FA split-K | Isolated 100K kernel: split-2 +8.5%; whole model/cross-model inconsistent; 200K tied | Global override could regress other depths/models. Keep automatic policy. |
| hd128 cooperative-matrix BDA | About 0.8x in the measured path | Scalar base-address generation/load path was inefficient. Remains opt-in. |
| Prefill KV BDA at hd256 | 100K 402.7 -> 404.0; 200K 286.1 -> 284.7 | Noise/slight regression. Bound path remains default. |
| Direct planar-Q8 hd256 Prefill FA | -6.3% at 100K, -8.9% at 200K | GQA/query tiles repeatedly decoded the same compact KV. Existing one-time Q8->f16 prepass was only 0.62% of device time. A future design must share decode across GQA heads/tiles. |
| Q8 Decode chunk 512 -> 1024 before D8 changes | 31.85 -> 30.9 (-3.0%) | Wider serial pass-1 hurt before enough wave/block parallelism existed. The same chunk became +6.3% only after D8/LS/QK/PV changes. |
| Q8 scale-once shared decode | 31.45 -> 29.9 (-4.9%) | Sharing the scale reduced independent lane work/occupancy more than it saved loads. |
| Combine one/two tiles per head | ntile4 39.3; ntile2 38.3; ntile1 35.4 | Less duplicated exp/max work could not compensate for lost occupancy. |
| Larger Q8 chunks after 1024 | 1536 flat; 2048 -4.1%; 4096 -7.7% | Combine shrank, but pass-1 serial QK/PV grew and workgroup latency hiding fell. |
| Combine-128 specialization | 34.75 -> 30.5 (-12.2%) | Too little output/head parallelism. |
| Parallel-32 combine rewrite | 34.95 -> 35.0 | No measurable whole-model gain; not worth extra path. |
| Raw-f16 Q8 scale | 37.3 -> 36.85 (-1.2%) | Conversion/register behavior outweighed the smaller scalar representation. |
| f16 PV weight | 37.2 -> 36.35 (-2.3%) | Reduced precision/width did not map to a faster instruction/data path. |
| Packed Q8 unpack rewrite | 37.05 -> 33.05 (-10.8%) | Extra unpack/shuffle work and compiler register behavior were worse than bitfield extraction. |
| Extra ring slots without geometry control | Slots 2 about 344, 3 about 289, 4 about 376, 5 about 284 t/s | More slots are not monotonically better; lane reuse and allocation geometry matter. Eight slots with a 3 GiB ring reached about 392 t/s in that sweep. |
| Smaller host-copy chunks | d200K: 4096-byte setting 390.2; 512 323.2; 256 289.4 | Fine-grained jobs add scheduler/copy overhead. Batch independent experts or copy complete contiguous banks instead. |
| Layer mode versus expert LRU before Host redesign | pp512 d200K roughly 415 vs 418-420; pp4096 528.3 vs 522.3 | Layer granularity alone was not a speed win. Its value was enabling fixed resident addresses and contiguous A/B transfer. |
| Third ReBAR lane | Not implemented | A/B completes before current compute in only 40.6% of streamed 200K layers. A third maximum-layer lane would add about 660 MiB and permit deeper prefetch, but exceeds the confirmed A/B design and needs a measured benefit before adoption. |

## 7. Hardware diagnosis and why the successful changes worked

Earlier low-overhead/RGP measurements showed two distinct regimes:

- Prefill d0 used about 38% GPU 3D and roughly one CPU logical thread. Host memcpy was already
  about 18.8 GB/s one-way (at least about 37.6 GB/s host DRAM traffic counting read+write). The GPU
  was starved by copies, paging decisions and synchronization. This is why batching copies,
  recorder reuse, resident layers and direct ReBAR push help shallow/big-batch Prefill.
- Prefill d200K reached about 92% Memory Unit busy inside long FA windows, even though whole-run GPU
  use was only about 53%. It has both saturated attention windows and gaps. Removing Host-path work
  helps, but cannot produce the same percentage gain as d0 unless attention is also improved.
- Decode was primarily a latency/bubble problem: about one CPU logical thread, roughly 19% GPU-busy
  samples in selected windows, and low wall-clock PCIe utilization. Recorder/resource reuse and
  fewer synchronization points therefore produced much larger gains than bandwidth-only changes.
- Q8 Decode was a separate inner-kernel problem after the bubbles shrank. D8 block clustering,
  wider waves, packed fp16 QK/PV and chunk 1024 were cumulative because they improved occupancy and
  reuse together; applying the final chunk in isolation was slower.

## 8. Final assessment

The series achieved three durable outcomes:

1. Long-context hd256 attention no longer falls through the old score-matrix path; f16 Prefill and
   Decode improved by roughly 2-3x at the deepest original baseline points.
2. Q8 KV Decode retained its 46.9% KV byte saving while the dedicated kernel chain improved about
   25.7% at 200K and reached/cleared the original 90%-of-f16 acceptance target.
3. Prefill and Decode now use deliberately different MoE cache granularities. Prefill gets
   load-time layer-contiguous resident/A/B streaming; Decode retains expert-level LRU. The complete
   expert payload exists once in Host RAM and is directly addressable by both paths.

The current main remaining Prefill limit is not Host layout or a hidden second 20+ GiB copy. At
large context it is long-attention work plus per-layer A/B windows that are not always fully hidden.
The next major change should therefore be justified by a timeline-level result: either a deeper
prefetch design with measured benefit per added MiB, or a genuinely new long-context FA
decomposition. More small copy-chunk, submit-cap, or combine-tile sweeps are unlikely to move the
whole model materially.

## 9. Primary artifacts

- `target/perf/current-prefill-matrix-354c0c3-20260819/results.csv`
- `target/perf/current-prefill-matrix-354c0c3-20260819/*.log`
- `target/perf/qwen36-moe-20260817-155522-16fbee2/report.md`
- `target/perf/final-3model-02e0bfb/report.md`
- `target/perf/hw-bottleneck-20260818/report.md`
- `target/perf/dispatch-cap-study-20260818/report.md`
- `target/perf/q8-fa-ab-20260818/report.md`
- `target/perf/q8-chunk-combine-feasibility-20260818/feasibility-summary.md`
- `target/perf/moe-layer-stream-final-20260818/`
- `target/perf/moe-host-store-20260819/`
- `target/perf/moe-cpu-push-20260819/`

