# Validated models & performance

Everything below is **validated on an AMD Radeon RX 7900 XTX** (RDNA3, 24 GB,
Vulkan / RADV): correctness is checked against the CPU reference implementation
(the `gpu_seam_matches_cpu_*` tests generate token-for-token on both and
compare) and throughput is measured against the system `llama.cpp` build with
`infr compare`.

**Throughput vs llama.cpp** — ratios are `infr / llama.cpp` (**>1.0 = infr is
faster**); r=3, 2026-08-02 snapshot (infr `691c0dc`, every model×quant in the
local cache, oracle `llama-bench` **`c629da5`** Vulkan build on every row).
Hardware: **AMD Radeon RX 7900 XTX** (RDNA3, 24 GB, Vulkan / RADV, Mesa).
`pp512` = 512-token prefill throughput, `tg128` = 128-token decode throughput,
`tg64@d4096` = decode at 4096 KV depth, `pp4@d4096` = short-turn prefill at 4096
KV depth (the multi-turn serve shape).

> ### Read the prefill columns as one significant figure
>
> The two **decode** columns are reproducible; the two **prefill** columns are
> not, and by much more than this table used to admit. Running the whole sweep
> TWICE against the same infr binary (`691c0dc`) and near-identical llama.cpp
> absolutes gives, per column:
>
> | column       | mean abs Δ between runs | worst row | rows moving >5% |
> | ------------ | ----------------------- | --------- | --------------- |
> | `tg128`      | 0.8%                    | 3.0%      | 0 / 35          |
> | `tg64@d4096` | 0.7%                    | 2.2%      | 0 / 35          |
> | `pp512`      | 6.8%                    | **34.5%** | 10 / 35         |
> | `pp4@d4096`  | 7.7%                    | **31.7%** | **19 / 35**     |
>
> That is the SAME binary measured twice, so it is pure run-to-run variance, not
> a code change. The cause is known and documented as an open issue: infr's
> default prefill is nondeterministic in its tier/chunk choice, so a short
> prefill can land on a different kernel tier between runs. It bites hardest on
> the small models (whole prefill is short) and on the IQ3_S MoE.
>
> Practical consequences: **a prefill cell's second decimal is meaningless**, a
> prefill difference under ~10% between two rows or two snapshots is not a
> result, and the per-column win counts below move by several rows run to run
> (`pp512` was 26/35 on the immediately preceding run and is 34/35 here, same
> binary). Decode cells can be read as written. To get a stable prefill number,
> pin the chunk with `-u/--ubatch` and repeat the run.

> **The oracle moved, so these ratios are NOT comparable to the 2026-07-13 table
> they replace.** That snapshot was taken against `llama-bench` **b9957**; this
> one against **`00f5442`**, three weeks of upstream Vulkan work later. Most
> ratios fell, and almost none of that is infr slowing down — an infr-vs-infr
> A/B across the same commit range (`2b3a943` → HEAD, same GPU, alternated
> order, 30 s cooldowns) has infr FASTER on every metric it moved: `pp512`
> +9–19%, `tg128` +0.3–4.7%, `tg64@d4096` +0.6–3.7%, `pp4@d4096` +5%. Where a
> row lost ground, llama.cpp gained more than we did. Read a drop here as "they
> closed the gap", not "we regressed" — and if you need the latter question
> answered, run the infr-vs-infr comparison, not this table.

| Model                 | Quant       | pp512     | tg128     | tg64@d4096 | pp4@d4096 |
| --------------------- | ----------- | --------- | --------- | ---------- | --------- |
| Qwen3-0.6B            | Q2_K        | **1.16×** | **1.41×** | **1.23×**  | **2.01×** |
| Qwen3-0.6B            | IQ4_XS      | **1.12×** | **1.16×** | **1.13×**  | **1.91×** |
| Qwen3-0.6B            | Q4_0        | **1.12×** | **1.30×** | **1.17×**  | **1.89×** |
| Qwen3-0.6B            | Q4_K_M      | **1.15×** | **1.20×** | **1.14×**  | **2.04×** |
| Qwen3-0.6B            | Q5_K_M      | **1.17×** | **1.25×** | **1.16×**  | **1.85×** |
| Qwen3-0.6B            | Q6_K¹       | **1.18×** | **1.06×** | **1.03×**  | **1.83×** |
| Qwen3-0.6B            | Q8_0        | **1.19×** | **1.12×** | **1.08×**  | **1.86×** |
| Qwen3-0.6B            | BF16⁸       | **1.06×** | 0.98×     | 0.98×      | **1.79×** |
| Qwen3.5-0.8B          | Q4_K_M      | **1.37×** | **1.11×** | **1.07×**  | **1.64×** |
| Gemma-3-1B            | Q2_K        | **1.05×** | **1.04×** | 0.97×      | **1.01×** |
| Gemma-3-1B            | Q4_K_M      | 0.96×     | **1.17×** | **1.08×**  | **1.10×** |
| Gemma-3-1B            | Q8_0        | **1.32×** | **1.13×** | **1.06×**  | **1.04×** |
| Llama-3.2-1B          | Q4_K_M      | **1.04×** | **1.08×** | 0.95×      | **1.15×** |
| Llama-3.2-1B          | Q8_0        | **1.04×** | 0.97×     | 0.88×      | **1.05×** |
| Qwen3-1.7B            | Q4_K_M      | **1.15×** | **1.12×** | **1.10×**  | **1.65×** |
| Qwen3.5-4B (MTP)²     | Q4_K_M      | **1.29×** | **1.01×** | **1.01×**  | **1.58×** |
| Qwen3.5-4B (MTP)²     | UD-Q4_K_XL  | **1.27×** | **1.02×** | **1.02×**  | **1.60×** |
| Gemma-4-E2B           | Q4_K_M      | **1.15×** | **1.06×** | 0.98×      | **1.10×** |
| Qwen3-8B              | Q4_K_M      | **1.39×** | **1.02×** | **1.00×**  | **1.07×** |
| Ornith-1.0-9B         | Q4_K_M      | **1.39×** | **1.04×** | **1.04×**  | **1.27×** |
| Qwen3.5-9B            | Q4_K_M      | **1.39×** | **1.05×** | **1.05×**  | **1.23×** |
| Qwen3.5-9B (MTP)²     | Q4_K_M      | **1.41×** | **1.00×** | **1.01×**  | **1.23×** |
| Qwen3.5-9B (MTP)²     | UD-Q4_K_XL  | **1.39×** | **1.00×** | **1.00×**  | **1.20×** |
| Gemma-3-12B           | Q4_K_M      | **1.32×** | **1.13×** | **1.13×**  | **1.54×** |
| Gemma-4-12B           | Q4_K_M      | **1.35×** | **1.12×** | **1.10×**  | **1.46×** |
| Qwen3-14B             | Q2_K³       | **1.24×** | 0.90×     | 0.84×      | **1.02×** |
| Qwen3-14B             | Q4_K_M      | **1.23×** | **1.01×** | 0.94×      | 0.97×     |
| Qwen3-14B             | Q8_0        | **1.18×** | 0.99×     | 0.94×      | 0.88×     |
| Gemma-4-26B-A4B (MoE) | UD-Q4_K_M⁹  | **1.15×** | **1.05×** | **1.06×**  | **1.35×** |
| Qwen3.6-27B           | Q4_K_M      | **1.27×** | **1.02×** | 0.99×      | **1.07×** |
| Qwen3-30B-A3B (MoE)   | Q4_K_M⁹     | **1.10×** | 0.96×     | 0.91×      | 0.85×     |
| Gemma-4-31B           | UD-Q5_K_XL⁴ | **1.08×** | **1.03×** | **1.04×**  | **1.06×** |
| Ornith-1.0-35B        | Q4_K_M⁵     | **1.04×** | **1.03×** | **1.02×**  | **1.18×** |
| Qwen3.6-35B-A3B (MoE) | UD-IQ3_S⁶   | **1.17×** | 0.91×     | 0.91×      | **1.02×** |
| Qwen3.6-35B-A3B (MoE) | UD-Q4_K_M   | **1.19×** | **1.00×** | **1.00×**  | **1.30×** |

**Column by column.** `pp4@d4096` — multi-turn ingest, the shape a coding agent
actually runs — is the strongest column at **32 of 35** rows and up to
**2.04×**, and the small-model rows (1.8×–2.0× on Qwen3-0.6B) are the clearest
wins here. `pp512` reads 34 of 35 this run. Both are prefill, so both carry the
variance in the box above: treat the counts as "most rows" and the values as one
significant figure.

Decode is the half to quote precisely. `tg128` wins **29 of 35** and
`tg64@d4096` **24 of 35**, both reproducible to under 3% run to run.
`tg64@d4096` is the softer of the two — 11 rows below 1.0, worst Qwen3-14B Q2_K
at **0.84×** — and the at-depth softness is the clearest remaining signal in the
table, because it is the one that survives re-measurement.

The losses concentrate on **Qwen3-14B and the larger MoEs**, not spread evenly,
which makes them a tractable target rather than a broad deficit.

¹ **Q6_K now decodes on the int8 tier too** (`f82d74e` + `de987d7`). It was the
last format still unpacking its `ql`/`qh` bit-planes **byte-at-a-time** (8
scalar `rb()` loads per 32-element sub-block, where every other k-quant already
read aligned u32s and masked in-register) — and it was the only format badly
LOSING at decode (Qwen3-14B-Q6_K: 44.3 int8 vs 58.9 f32 t/s, **−25%**). Those
two facts were the same fact. A word-parallel `wdec` rewrite (funnel-shifted
`ru32u` word loads — Q6_K's 210-byte stride is 2 mod 4, so it needs the stitch —
plus a SWAR `q−32` rebias) is **bit-identical** to the old byte loop and
inverted the result: decode 44.3 → **64.3 t/s**, now BEATING f32's 58.4; prefill
`pp4@d4096` 137.9 → **183.6** (+34%). Unpack ALU, not memory bandwidth, was the
wall.

² **MTP speculative decode is currently DISABLED — see "MTP is parked" below.**
These rows are the models' ORDINARY (non-speculative) numbers, which is how the
MTP-head GGUFs now run. `INFR_MTP=1` is ignored with a warning; the `mtp128`
column is no longer measured.

These four rows' `tg64@d4096` cells were a GPU device-lost in the raw sweep and
are re-measured post-`8513358`: 35821b6's capacity gate on the `nonfa` coopmat
prefill tier (which reads K in whole 256-row tiles, so it touches
`ceil(kv_len/256)*256` rows) had no catcher for a **non-SWA** model — `split_ok`
only covered the SWA `ring_past` case — so the op fell through to the scalar
`attention_kv` at 3591 rows × 3591 kv and hung the GPU. MTP's un-chunked
whole-prompt verify is the only shape that reliably lands `kv_len` within one
tile-pad of the cache's row capacity.

³ **The int8-activation decode tier.** Quantizing the _activations_ to int8 and
integer-dotting them against the raw weight codes (`dotPacked4x8AccSatEXT`, the
`mmvq` shape) avoids dequantizing weights to f32 at all. On AMD the tier is now
default-on for **Q2_K, Q4_K, Q6_K, Q4_0, Q5_0, Q5_1, IQ4_NL**; **ordinary
prefill takes it for every integer dtype** (all 12). This row (Qwen3-14B Q2_K)
is what it bought at 2 bits: tg128 0.74× → 0.81×, tg64@d4096 0.72× → 0.78×, and
`pp4@d4096` 0.98× → a win. (The table's current values for this row,
0.85×/0.81×/1.49×, also include the later wide-rmsnorm lift — footnote ⁸.)

The single most useful thing learned here: **int8's value is row-count
dependent, and the two directions are independent policies.** The cost of the
tier is a per-dispatch activation-quantize pass; the benefit is the unpack ALU
it saves. At m=1 (decode) the quantize is dead weight amortized over one row, so
a dtype with a cheap unpack (Q8*0 — at 8 bits the stored byte already IS the
dp4a operand) \_loses*. At m≥3 (prefill) it amortizes hard and every integer
dtype wins, by +21% to +67%. So a dtype can lose decode and win prefill by a
mile, and infr ships two separate policy sets to say so —
`mmv_int8_decode_dtypes` (m=1) and `mrow_int8_prefill_dtypes` (m≥3), in
`crates/infr-vulkan/src/adapter.rs`. Conflating them is what used to keep
Q3_K/Q5_K/Q6_K's large prefill wins unreachable: they were tied to an
off-by-default decode tier.

Every entry is **measured on infr's own kernels**, not inherited from
llama.cpp's table — the two engines have different kernel overheads, so a win on
one does not imply a win on the other. (llama.cpp's `ggml_vk_should_use_mmvq`
returns true for every quant on AMD at `k >= 2048`, carving out only Q6_K and
Q8_0, so taking this trade is parity with the oracle, not a quality regression.)

**Q3_K stays OFF at decode** — and this is an accuracy result, not a perf one.
Flipping it broke `gpu_seam_matches_cpu_qwen3_q2k` into **degenerate** output
(`<think>` repeated to the token limit against the oracle's coherent answer).
Cause: **GGUFs are mixed** — unsloth's Qwen3-0.6B-**Q2_K** file carries Q3*K
tensors — so a "Q3_K" flip silently moved a 0.6B model's layers to int8, where
accumulated quantization error is worst, and it fell off a coherence cliff. The
cliff was then isolated to the \_decode* side specifically: the same test run
PREFILL-int8-only stays coherent and matches the CPU oracle token-for-token,
while DECODE-int8-only reproduces the divergence exactly. So Q3_K's prefill win
ships and its decode tier does not. **Q5_K** is off at decode on a plain
throughput call (−1.4% decode, +45% prefill); its accuracy was never in
question. Re-attempting Q3_K decode needs the accuracy question answered
(per-tensor-role gating? a size floor?), not a re-measure.

⁴ Gemma-4-31B (21.9 GiB weights on the 24 GB card) runs **fully resident,
including at depth**, after two placement slices: try-resident-first dense
placement (`e2c0694` — honest activation reserve + a phantom +1.6 GiB accounting
fix) and **window-sized ring KV for sliding-window layers** (`35821b6`,
llama.cpp-parity: 50 of its 60 layers are SWA with a 1024 window, so their
caches are 2048-row rings instead of full-context — @8k that's 0.44 GiB instead
of 5.5). The d4096 row went 0.08× → 0.90× (28 vs 31 t/s). The same slice also
reuses empty KV slots instead of forking a duplicate (`f74556c` — was silently
wasting a full KV per session, 6.25 GiB on a 14B), and lifted the gemma-family
multi-turn rows (12B `pp4@d4096` 1.40× → 1.66×: less dead KV to re-scan).

This row's `pp4@d4096` was the table's worst loss at 0.84×; it is now **1.32×**,
a win. That came from Q5*K's ordinary-prefill int8 tier (footnote ³) — this is a
Q5_K_XL file, and Q5_K's prefill win (+45%) was previously unreachable because
it was gated behind an off-by-default \_decode* tier. Splitting the two policies
banked it. Its **decode** was then closed too (0.91×/0.92× → **0.98×/1.00×**) by
the wide rmsnorm (footnote ⁸) — this model, at 21.9 GiB on a 24 GB card and 57%
of its GPU time in one Q5_K GEMV, is where that kernel was found. The fix added
zero allocations, so it still loads fully resident (peak 23.14 / 23.98 GiB).

⁵ **The DeltaNet scan: chunking was the bug, not the fix** (`0a5d366`).
Ornith-35B `pp512` was 0.89× — its scan ran **31.7 ms per 512 tokens against
llama.cpp's 6.8 ms**, 4.6× slower, and that one kernel was the whole loss.
Reading llama.cpp's `gated_delta_net.comp` showed its "fused" GDN is **not
chunked at all**: it is the plain token-serial recurrence with the state shard
held in **registers**.

Counting FLOPs kills the chunked premise outright — the chunked delta rule costs
~420M FMA/layer against ~402M for the plain recurrence, so it **saves no
arithmetic**. It only shortens the dependency chain, and it pays for that with
LDS-resident state, runtime trip counts that block unrolling, ~96 workgroup
barriers, and only 256 workgroups (~2.7 per CU on a 96-CU part — nothing in
flight to hide latency). It sustained **0.76 TFLOP/s against llama.cpp's 3.5**.

The fix was to go _simpler_, not more fused: single-subgroup workgroups (zero
barriers; the kd-contractions become one `subgroupAdd`), state in registers, and
all transcendentals hoisted out of the serial loop into a flat gates pass.
**31.7 → 8.4 ms.** `pp512` 0.89× → **1.03×**, and every one of Ornith-35B's four
metrics is now a win. It also lifted the other DeltaNet models (Ornith-9B, the
Qwen3.5/3.6 family). Decode is untouched (`rows == 1` still routes to the old
sequential kernel). Gated on `kd == 128`; `INFR_DN_CHUNK_SCAN=1` restores the
old path. The path no longer needs coopmat at all, so non-coopmat GPUs get the
fast kernel too. Nulls: LDS-staging the k̂/q̂ tiles **regressed** it to 51.4 ms
(occupancy collapse), and the bandwidth theory was simply wrong — cutting
traffic 4× bought 6%, so it was latency-bound all along.

⁶ Grid i-quant (IQ1–IQ3) row: the grid-perf slice closed both structural gaps
`618cd3b` left behind (that commit fixed the device-lost TDR — dynamically
indexed GLSL `const` codebook tables lowered to ~1 MB of per-invocation scratch
by RADV/ACO — by staging the grids through `shared` memory): a grid-aware
`dqblk` amortizes the per-32-group scale/sign/qh decode and grid gathers that
the per-element `dq()` re-derived (decode 0.50× → 0.89×, tg128 75 → 134 t/s),
and IQ2_S/IQ3_S — this file's expert pair — got batched dp4a mmq expert GEMMs
(shared-LUT grid staging feeds the int8 dot; prefill 0.03× → 0.91×, pp512 75 →
2575 t/s). The other five grid formats keep the id-GEMV prefill fallback (no
shipped MoE GGUF uses them for expert banks — see `MOE_MMQ_DTYPES`'s exclusions
doc).

**Prefill is now a WIN** (`c7c3f50`): `pp512` 0.90× → **1.16×**. The gap was
codebook _staging_, not bandwidth — the discriminator is that this file's Q4*K_M
sibling (same architecture, arithmetic experts) runs `expert_gateup` in 46 ms
while moving **1.76× more weight bytes**. Two causes, in places nobody had
looked: IQ2_S's scale nibble covers 16 elements, so its mmq k-loop ran at
`BLK=16` — the \_only* expert kernel doing k/16 passes where
Q4_K/Q6_K/IQ3_S/IQ4_XS all do k/32, i.e. double the barriers, scale staging and
activation loads for identical dp4a work. Merging it to `BLK=32` is
bit-identical (the two halves provably share one sub-block index and one scale
byte at a 32-aligned start, and the partial sums fold in the old loop's exact
summation order — proved against a host dequant reference by `grid_mmq_parity`)
and took `expert_gateup` 82.4 → **65.8 ms**. IQ3_S's down-projection also joined
the subgroup id-GEMV band (42 → 34.8 ms).

**Decode is still a loss** (0.93× / 0.94×, up from 0.89× / 0.90×) and the
remaining lever is _quantified but deliberately not taken_: ablating the
codebook staging entirely measures `native_idm_iq2s` 49.8 → **23.3 ms** and
`native_idm_iq3s` 42.0 → **18.9 ms** — i.e. **~50 ms of a 505 ms decode is pure
per-workgroup re-staging of the codebook into LDS**, which is essentially the
whole residual gap. The fix is to make the codebook **L2-resident in a buffer**
instead of re-staged by every workgroup (this also frees 8 KB of LDS per
workgroup, so it should beat the ablation). That needs a new SSBO binding across
every grid GEMV variant and re-validation of all 7 grid dtypes — a campaign of
its own, not a slice. Null: an SG tier for IQ2_S gate/up **regressed** hard
(`native_idm_iq2s` 49.8 → 117.2 ms — 8 KB of LDS on a single-wave workgroup
collapses occupancy).

⁷ **The legacy 32-block quants now have an int8 dp4a GEMV**, not just a dp4a
GEMM. The dp4a _GEMM_ (`native_gemm_mmq_*`) has covered ~17 dtypes for a while,
but the dp4a _GEMV_ (`native_mmv_mrow.comp`) covered only the six k-quants +
IQ4*XS — so every non-k-quant integer file fell to the f32 dequant path at
decode AND at small-m prefill, which is exactly why this Q8_0 row was one of the
table's three `pp4@d4096` LOSSES. Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/IQ4_NL now have
`wdec` arms (the mmq unpack, word-parallelized: aligned/funnel-shifted u32 loads
— every `_0`-family stride is 2 mod 4 — nibble masks, SWAR zero-point rebias, a
4-bit→4-byte-lane `qh` spread, and Q4_1/Q5_1's additive min folded through the
ones-dot against `sact`). Measured on Qwen3-14B (7900 XTX), int8 vs the f32 GEMV
that shipped before, **ordinary prefill** (`pp4@d4096`): Q4_0 **+66.9%**, Q5_0
**+64.0%**, Q5_1 **+42.2%**, Q4_1 **+32.9%**, Q8_0 **+28.8%** (128 → 158 t/s —
this row: 0.92× → **1.18×**), IQ4_NL **+20.7%**. **Decode** (`tg64`) is a
separate policy and splits: Q5_0 **+16.8%**, Q4_0 **+10.5%**, IQ4_NL **+6.3%**,
Q5_1 **+6.1%** are default-ON; **Q8_0 −4.2%, and Q4_1 a wash, are default-OFF**
(prefill-only). Q8_0's decode loss is structural, not a wart to fix — at 8 bits
the stored byte already IS the dp4a operand, so there is no unpack ALU to save,
and decode is weight-bandwidth bound while the int8 route still pays the
`quant_q8` bubble (llama.cpp excludes Q8_0 from mmvq off old GCN for the same
reason). Hence this row's `tg64@d4096` stays 0.97×: the fix is a prefill fix.
Guards: `mmv_mrow_legacy_formats` (each `wdec` vs a from-scratch host reference,
f64-accumulated), `mmv_row1_bit_identical` (m=1 decode ≡ row 0 of the m≥3
dispatch, exact `to_bits()`), and all 13 `gpu_seam_matches_cpu*\*` (two of which
load an IQ4_NL and a Q8_0 model, so the decode flips face the CPU oracle).

⁸ **The decode "bandwidth wall" was mostly a norm kernel running on one
workgroup** (`2b3a943`) — the highest-leverage fix in this table, and it is not
a GEMV. `rmsnorm` dispatches one workgroup **per row**, so at decode
(`rows == 1`) the entire dispatch was a _single_ 256-thread workgroup — 8
wave32s on **one WGP out of 48** — reducing a 21 KB row. Pure latency with
nothing in flight: **12.7 µs per dispatch, against ~1.2 µs for `add` over the
same vector** (which fans out to `dim/64` workgroups). At 241 dispatches per
token that was **8.9% of all decode GPU time**. A whole-row reduction cannot be
split across workgroups without a second dispatch, so the fix keeps the single
workgroup but restores memory-level parallelism _inside_ it: **1024 threads ×
vec4 loads** = 4× the waves and 4× the bytes per request. **12.7 → 4.0 µs.**
Gated to `rows == 1 && dim >= 2048`; the 256-thread build's SPIR-V is
byte-identical, so the change is purely additive.

This is **not model-specific** — it lifts every model with hidden ≥ 2048, and it
is what turned the entire 8B–27B decode band from losses into wins (Qwen3-8B
`tg128` 0.96× → 1.02×, Qwen3-14B Q4_K_M 0.97× → 1.02×, Gemma-3-12B 1.02× →
1.12×, Gemma-4-26B MoE 1.01× → 1.13×).

It also corrects a story this README told for a long time. "Decode is
weight-bandwidth bound" was **measured but incomplete**: `native_q8_0` runs the
_same_ `native_gemv` kernel at the _same_ m=1 shape and reaches **863 GB/s = 90%
of the card's ~960 GB/s peak** — that is the real wall — while Q5*K, at 57% of
all GPU time, sat at **737 GB/s (77%)**. Same kernel, so the memory system was
never the difference. Null result from the same slice: the Q5_K \_ALU*
hypothesis was **falsified** — a SWAR rewrite of its 5-bit rebuild predicted
~22% and measured **2.3%** (ACO was already fusing the shift+mask into
`v_bfe_u32`); it shipped anyway because it is bit-identical and free. The
genuine residual is **VMEM instruction count** (a Q5_K sub-block issues 16 word
loads to Q8_0's 8, and a superblock's `qh` bytes get re-read ~3× by its
sub-blocks), which needs superblock-granular decode — left open. **BF16 decode**
(0.87×) is the one row none of this can help: it is the only non-integer weight
dtype, so there is no unpack ALU to save and no weight codes to integer-dot.

⁹ **MoE expert GEMMs: the waste was inside the tile, not in the routing**
(`6a33065`). The expert GEMMs are **72% of MoE prefill GPU time**. The suspicion
was that infr lacked llama.cpp's `mmid` row-packing (sort/gather rows by expert
so each expert gets one contiguous GEMM) — **it does not**: infr already packs
(`moe_bucket_count` → `_scan` → `_scatter`, expert id on `gl_WorkGroupID.y`),
and the whole packing pipeline costs **3.6% of GPU time**. There was nothing to
win there.

The real waste: a tile was only skipped _wholesale_ when its first row was past
the expert's segment. Inside a **partial** tile, rows past the segment end still
ran the full dp4a k-loop and had their results thrown away by the clipped store.
At 128 experts × top-8 that is ~32 routed rows in a BM=64 tile — **half of every
tile computing garbage**. A `live` row mask around the dp4a (staging and both
barriers stay unconditional, so no divergent barrier) drops it.

The instructive part is the **null result**: the obvious fix — shrink the tile
to BM=32 to match the ~32 real rows — is exactly **backwards**, measuring a
**15.4% LOSS** (BM=32/BN=64: 3054 → 2584). BM=64 at ~32 rows/expert gives
exactly one row tile per expert, so each expert's weight bank is staged **once**
— the floor. BM=32 adds a second row tile for any expert over 32 rows, and every
row tile re-stages the whole (much larger) weight bank. **The GEMM is
weight-staging bound, not math bound**; masking drops the dead math _without_
paying the re-stage. A BN=128 wide-N tile also ships, gated on `k <= 1024`: it
helps the shallow-k `down` proj (`expert_down` 56.7 → 50.0 ms) but **hurts** the
deep-k `gate`/`up` proj (`expert_gateup` 65.0 → 69.2 ms), so applying it
unconditionally would have been a wash that slowed the dominant op. `pp512`:
Qwen3-30B-A3B 0.95× → **1.09×**, Gemma-4-26B-A4B 1.07× → **1.15×**.

The MoE expert kernel floor (the id-indexed GEMV family every MoE model needs
for decode) now covers **every weight dtype the dense Vulkan path supports** —
all quants (Q\* incl. ternary Q2_0, K-quants, IQ\*, TQ\*, MXFP4/NVFP4, BF16)
plus F16/F32 float banks — so no expert-bank quant is rejected at load. On top
of that, the batched-MoE dp4a mmq prefill family covers Q4_0 / Q4_1 / Q5_0 /
Q5_1 / Q8_0 / Q2_0 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ4_NL / IQ4_XS / MXFP4
/ NVFP4 (`infr_core::tensor::MOE_MMQ_DTYPES` is the single source of truth both
the graph-build and adapter gates derive from; `moe_mmq_drift_test` guards the
kernel tables against drift, and its doc records the deliberate exclusions: grid
i-quants (IQ1–IQ3), ternary (TQ\*), and float banks prefill via the per-token
id-GEMV path).

**Where infr wins.** Decode is the trustworthy story, because it is the
reproducible one: `tg128` wins **29 of 35** rows and `tg64@d4096` **24 of 35**,
both with run-to-run variance under 3%. Prefill reads as a near-sweep this run
(`pp512` 34/35, `pp4@d4096` 32/35, peaking at **2.04×**) but see the
reproducibility box above before quoting those counts — the immediately
preceding run of the same binary gave 26/35 on `pp512`. The durable prefill
claim is the SHAPE, not the count: `pp4@d4096` (multi-turn ingest, what a coding
agent actually runs) is consistently the strongest column, roughly 1.5–2× on the
small and mid models across both runs.

**Where infr loses.** 21 losing cells of 140, and after the prefill columns
stabilised into wins this run they are almost entirely DECODE — which is the
half of the table you can trust. Two clusters:

- **Qwen3-14B, every quant** — the densest cluster and the one to fix next. Q2_K
  is worst (`tg64@d4096` **0.84×**, `tg128` 0.90×), and Q4_K_M/Q8_0 lose the
  same column (0.94× both). Successive levers have lifted Q2_K (0.74×/0.72× →
  the int8 tier → the wide rmsnorm) without closing it. It remains the only row
  whose gap has never been root-caused to a named kernel; it deserves its own
  profile rather than another inherited hypothesis.
- **The larger MoEs at depth** — Qwen3-30B-A3B (`tg64@d4096` 0.91×, `tg128`
  0.96×, `pp4@d4096` 0.85×) and Qwen3.6-35B-A3B UD-IQ3_S (0.91× on both decode
  columns). The IQ3_S gap is **fully diagnosed and deliberately not fixed**: ~50
  ms of its decode is per-workgroup re-staging of the codebook into LDS
  (footnote ⁶). Making the codebook L2-resident should close essentially all of
  it, but it touches every grid GEMV variant and all 7 grid dtypes — a campaign,
  not a slice.

Note the small-model `pp512` cluster reported in the previous snapshot is GONE
this run (only Gemma-3-1B Q4_K_M at 0.96× remains, inside the noise band). That
cluster was prefill variance, not a real deficit — a caution against acting on a
single prefill run.

Two structural rows sit outside those clusters: **BF16** (`pp512` 0.94×, decode
~0.98–0.99×) is the only non-integer weight dtype in the table, so there is no
unpack ALU to save and no weight codes to integer-dot — out of reach of
everything that fixed the other rows. And **Llama-3.2-1B `tg64@d4096`**
(0.90×–0.94×) remains an isolated small-model decode-at-depth row whose other
columns are wins.

**A loss the table does not show.** `infr compare`'s deep-context turn shapes
(16k–32k KV, beyond this table's 4096) still lose on the MoE rows and get
**monotonically worse with depth** — Qwen3-30B-A3B `pg8192,512`: 0.88× @8k,
0.80× @16k, **0.74× @32k**. The published table tops out at d4096 and so
flatters us at exactly the shape a long-lived agent session actually reaches.
Untriaged; likely the most valuable open item here.

**DiffusionGemma** (`dg-step`) beats the reference fork at 1.23× (this sweep;
previously 1.18×).

**Ternary-Bonsai (Q2_0) — infr is the only engine that runs these on a GPU.**
llama.cpp merged the **Q2_0** weight dtype (GGML type 42) but shipped **no GPU
kernels for it**: there is not a single `q2_0` reference in its `ggml-vulkan/`
or `ggml-cuda/` trees, so every backend but CPU refuses to load these files.
infr runs Q2_0 natively on Vulkan (in-shader dequant + dp4a mmq — `ad89fb4`), so
the comparison below is **absolute throughput on different devices, not a
like-for- like ratio**: infr on the RX 7900 XTX vs llama.cpp on a Ryzen 9
9950X3D (16 threads, Release + `GGML_NATIVE`). r=3, 2026-07-12.

| Model (Prism ML) | Size    | infr pp512 | infr tg128 | llama.cpp pp512 | llama.cpp tg128 |
| ---------------- | ------- | ---------- | ---------- | --------------- | --------------- |
| Bonsai-1.7B      | 462 MiB | **6365**   | **594**    | 108.7 (CPU)     | 78.3 (CPU)      |
| Bonsai-4B        | 1.05 GB | **2756**   | **303**    | 41.9 (CPU)      | 33.9 (CPU)      |
| Bonsai-8B        | 2.15 GB | **1647**   | **212**    | 22.1 (CPU)      | 18.6 (CPU)      |

Use the **`Q2_0_g64`** files — despite the name they are the layout upstream
merged (64-elem / 18 B blocks, 2.25 bpw). The repos' plain `*-Q2_0.gguf` /
`*-PQ2_0.gguf` uploads predate the merge and use 128-elem / 34 B blocks (2.125
bpw); llama.cpp master rejects them too. Same scheme otherwise — one f16 scale
per 128 weights instead of per 64 — so they could be supported by a lossless
load-time repack if the format sticks around.

```bash
infr run prism-ml/Ternary-Bonsai-8B-gguf:Q2_0_g64 "What is the capital of France?"
```

**Llama-4-Scout** (109B-A17B, Q2_K, 37 GB) is deliberately absent from the table
above (its per-token small-m dispatch shape isn't comparable to the batched
pp/tg columns) but runs end to end on a 24 GB card via the paged expert cache
(`infr_vulkan::pager`). Prefill runs the batched bucket-scatter dp4a mmq
expert-GEMM pipeline against the pager arena with NO host round-trip at all:
each layer pre-stages its full expert set through a pipelined staging ring
(recorded ring→arena copies, fenced half rotation — CPU expert memcpys overlap
GPU execution) under a scan-resistant eviction policy, and every paged dispatch
reads a frozen per-layer LUT window from a tape instead of a live LUT. Decode
keeps the id-indexed small-m GEMV with at most ONE mapped-readback sync per
non-resident layer (fully-resident layers record straight through). Greedy
output is oracle-locked against llama.cpp (`cpu_llama4_scout_greedy`) AND
against the paged Vulkan path itself
(`gpu_seam_paged_moe_matches_scout_oracle`), token-for-token identical. Measured
(all 48 expert layers paged, per-role LRU caches of 312/312/238 experts — each
role's arena is one SSBO, capped at the device's 4 GiB binding range): `pp512`
**404 t/s** warm (r=3; pre-rework host-orchestration baseline: 189; llama.cpp's
CPU-offload hybrid: 136 — and past the ~363 t/s-equivalent GPU-busy ceiling the
old per-layer submit→readback→upload cadence measured, since staging now
overlaps compute), warm decode `tg64@d128` **~17 t/s** (baseline 14.2; llama.cpp
hybrid: 6.55 — decode stays upload-bound: a 24 GB budget can't hold the ~37 GB
decode working set, so ~350 MB/token still pages in). `INFR_CACHE` sizes the
pager's budget (see the MoE placement paragraph above); `INFR_PAGER_RING`
overrides the staging-ring size (default: budget/8 clamped to [256 MiB, 2 GiB]);
pure CPU stays available under `--dev cpu` / `-ngl 0`. Remaining follow-up:
splitting a role across several arena buffers to lift the 4 GiB per-role cache
cap.

**Also validated for correctness** (GPU seam vs CPU reference), beyond the perf
table: Qwen2-0.5B, Llama-3.2-1B, Gemma-4-12B (dense), and Qwen3-0.6B across
quant formats **Q4_K_M / Q5_K_M / Q6_K / Q4_0 / Q2_K / IQ4_XS / Q8_0 / BF16**
(each decoded on-device via hand-written SPIR-V, byte-identical to the CPU
dequant).

> Numbers are a snapshot and move with each perf slice; regenerate on your own
> hardware with `infr compare --sweep <model...>`. Results on other GPUs
> (NVIDIA, Intel Arc) and Apple Metal are wanted — please open an issue with
> your `infr bench` / `infr compare` output. Intel Arc testers: include one run
> with `INFR_DEBUG_COOPMAT=1` (the enumerated/chosen coopmat shapes), then A/B
> `INFR_CM_8X8=1` (opt-in 8x8x16 XMX prefill GEMM) against the default.
