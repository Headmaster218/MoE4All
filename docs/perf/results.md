# Validated models & performance

Everything below is **validated on an AMD Radeon RX 7900 XTX** (RDNA3, 24 GB,
Vulkan / RADV): correctness is checked against the CPU reference implementation
(the `gpu_seam_matches_cpu_*` tests generate token-for-token on both and
compare) and throughput is measured against a `llama.cpp` build with
`infr compare`.

**Throughput vs llama.cpp** — ratios are `infr / llama.cpp` (**>1.0 = infr is
faster**); r=3, **2026-08-03** snapshot. Provenance:

- **infr `2241e60`**, release build, every model×quant resolved from the local
  HF cache.
- **Oracle `llama-bench` b9833** (`build: c818263f2a (9833)`), Vulkan, release,
  on every row. **Not the `c629da5` the previous table cited** — see the oracle
  box below for why, and for what that does to comparability.
- **Hardware:** AMD Radeon RX 7900 XTX (RDNA3, 24 GB), Vulkan / RADV, **Mesa
  26.1.6**, host Ryzen 9 9950X3D. Exclusive use of the GPU; every measurement
  serial.

`pp512` = 512-token prefill throughput, `tg128` = 128-token decode throughput,
`tg64@d4096` = decode at 4096 KV depth, `pp4@d4096` = short-turn prefill at 4096
KV depth (the multi-turn serve shape).

> ### Caveat on the oracle: the SYSTEM `llama-bench` is broken on this box
>
> The distro `llama-cpp` package (b10182) links against ggml 0.17 while the
> installed `ggml-vulkan` is still 0.15.3, so `/usr/lib/libllama.so.0` dies with
> `undefined symbol: ggml_dsv4_hc_post` and **no system llama.cpp binary runs at
> all**. Rather than change the machine's packages, this sweep ran a cached
> **`llama-cpp-vulkan-b9833-1` release build against its own bundled libs**
> (`LD_LIBRARY_PATH` shim, passed to `infr compare --llama-bench`). It is a
> release build; the other cached builds print `warning: asserts enabled` and
> would flatter infr, so they were not used.
>
> Spot-check that b9833 is comparable to the `c629da5` the previous snapshot
> used, on the one shape both have a recorded value for (Qwen3-30B-A3B Q4_K_M,
> d8192): **`pp512` 1686.8 against 1692.9 recorded (−0.4%)**, **`tg128` 161.2
> against 165.1 (−2.4%)**. Close, but not zero — so treat cross-snapshot ratio
> diffs under ~3% as oracle drift, not as an infr change. To answer "did infr
> get faster", run an infr-vs-infr A/B, never a diff of two tables.

> ### How reproducible is each column?
>
> **The old warning in this box no longer holds.** The previous version warned
> that both prefill columns were untrustworthy: `pp512` 6.8% mean / **34.5%**
> worst between two runs of the same binary, `pp4@d4096` 7.7% / 31.7%. **That is
> fixed and the diagnosis in it was wrong.** Backlog **B6** established that the
> cause was not tier nondeterminism (the dispatched kernels were byte-identical
> across runs) but a **cold first timed rep** — `bench_vulkan`'s untimed warmup
> ran a different shape from the one about to be measured, so rep 1 paid that
> shape's one-time costs inside the measured window. `2241e60` warms the
> measured shape, and three independent measurements of variance on THIS tree
> now say:
>
> | column       | in-run rep spread¹ | ratio p2p, 4 full runs² | infr abs, 2 passes³ |
> | ------------ | ------------------ | ----------------------- | ------------------- |
> | `tg128`      | 0.5% / 3.4%        | 1.5% / **2.7%**         | 0.5% / 1.3%         |
> | `tg64@d4096` | 0.6% / 2.4%        | 1.4% / **2.2%**         | 0.6% / 1.7%         |
> | `pp512`      | 2.6% / 7.1%        | 1.6% / **3.4%**         | 0.6% / 3.5%         |
> | `pp4@d4096`  | 5.8% / 22.6%       | 7.4% / **16.0%**        | 2.0% / 10.0%        |
>
> ¹ mean / worst of `(max−min)/min` over the 3 reps `infr bench` reports on
> every line, across all 35 rows. ² mean / worst peak-to-peak of the
> **infr/llama ratio** over four complete `infr compare` runs of the six rows
> the old box named worst (Qwen3-0.6B Q4_K_M, Qwen3.5-0.8B, Gemma-3-1B Q4_K_M,
> Llama-3.2-1B Q4_K_M, Qwen3-1.7B, Qwen3.6-35B-A3B UD-IQ3_S). ³ mean / worst
> disagreement of infr's absolute t/s between the sweep pass and a separate
> infr-only pass ~90 minutes later, all 35 rows.
>
> **Three of the four columns are now reproducible and can be read as written.**
> `pp512` in particular: it is stable to ~1.5% run to run, so a `pp512` figure
> is a result, and the small-model `pp512` softness noted below is a real,
> repeatable deficit rather than the noise the old box blamed it on.
>
> **`pp4@d4096` is still soft, and only on the smallest models.** Every row with
> an in-run spread over 5% is ≤4B parameters (all eight Qwen3-0.6B quants,
> Qwen3.5-0.8B, all three Gemma-3-1B, Qwen3.5-4B UD-Q4_K_XL, Gemma-4-E2B); every
> row at 8B and above is under 5%, and the IQ3_S MoE that used to be the worst
> row in the table is now **2.6%** peak-to-peak. This is B6's residual and it is
> not tier choice: the metric times four tokens, ~3.7 ms of wall of which only
> ~2.8 ms is device time, so host record/submit/fence jitter is a quarter of the
> measurement. It is not an infr artefact either — **llama.cpp's own
> `pp4@d4096`** wobbles 3.5% mean / 8.9% worst over the same four runs, and the
> ratio compounds both. Read the small models' `pp4@d4096` as ±10%; read the ≥8B
> rows as written.

> ### Two rows do not run the same way the rest do
>
> `infr bench` now reports the chunk, the KV dtype and the final submit cap it
> used. No row in this sweep auto-quantized its KV cache, but two rows differ
> from the default and it changes what their at-depth cells mean:
>
> - **Qwen3.6-27B Q4_K_M — the submit splitter ARMS on both at-depth cells.**
>   The 4096-token depth prime is one 1633-dispatch forward that takes ~1.01 s,
>   just past the 1 s `SUBMIT_DANGER_NS` threshold, so
>   `VulkanBackend::observe_forward` latches a cap and every later forward in
>   that process — including the timed ones — splits every ~400 dispatches.
>   Reproduced in 3 of 3 processes (caps 392 / 401 / 403), so this row is
>   self-consistent, but its `tg64@d4096` and `pp4@d4096` are the only cells in
>   the table measured with a split submit structure. B6 recorded this splitter
>   as a latent hazard that had never fired; it fires here. (It also fires on
>   Qwen3-30B-A3B on some legs past d8192 — see the deep-context section — but
>   never at the d4096 this table uses.)
> - **Gemma-4-31B UD-Q5_K_XL runs a 256-row prefill chunk at depth**, not the
>   default 1024: at 21.9 GiB of weights on a 24 GB card the 1024-row chunk's
>   activation reserve does not fit, so dense placement drops to 256 and logs it
>   (footnote ⁴). Its `pp512`/`tg128` cells use the default 1024.

| Model                 | Quant       | pp512     | tg128     | tg64@d4096 | pp4@d4096 |
| --------------------- | ----------- | --------- | --------- | ---------- | --------- |
| Qwen3-0.6B            | Q2_K        | **1.20×** | **1.51×** | **1.27×**  | **2.19×** |
| Qwen3-0.6B            | IQ4_XS      | **1.14×** | **1.21×** | **1.17×**  | **1.86×** |
| Qwen3-0.6B            | Q4_0        | **1.20×** | **1.33×** | **1.20×**  | **1.91×** |
| Qwen3-0.6B            | Q4_K_M      | **1.20×** | **1.22×** | **1.16×**  | **1.95×** |
| Qwen3-0.6B            | Q5_K_M      | **1.17×** | **1.28×** | **1.20×**  | **1.90×** |
| Qwen3-0.6B            | Q6_K¹       | **1.20×** | **1.09×** | **1.08×**  | **1.72×** |
| Qwen3-0.6B            | Q8_0        | **1.23×** | **1.15×** | **1.12×**  | **1.91×** |
| Qwen3-0.6B            | BF16⁸       | **1.08×** | **1.00×** | 0.99×      | **1.83×** |
| Qwen3.5-0.8B          | Q4_K_M      | **1.36×** | **1.14×** | **1.09×**  | **1.75×** |
| Gemma-3-1B            | Q2_K        | **1.05×** | **1.07×** | **1.01×**  | **1.10×** |
| Gemma-3-1B            | Q4_K_M      | 0.95×     | **1.20×** | **1.11×**  | **1.05×** |
| Gemma-3-1B            | Q8_0        | **1.30×** | **1.14×** | **1.06×**  | **1.00×** |
| Llama-3.2-1B          | Q4_K_M      | **1.05×** | **1.07×** | 0.94×      | **1.18×** |
| Llama-3.2-1B          | Q8_0        | **1.04×** | 0.98×     | 0.88×      | **1.08×** |
| Qwen3-1.7B            | Q4_K_M      | **1.14×** | **1.15×** | **1.13×**  | **1.66×** |
| Qwen3.5-4B (MTP)²     | Q4_K_M      | **1.31×** | **1.02×** | **1.03×**  | **1.61×** |
| Qwen3.5-4B (MTP)²     | UD-Q4_K_XL  | **1.30×** | **1.02×** | **1.02×**  | **1.61×** |
| Gemma-4-E2B           | Q4_K_M      | **1.16×** | **1.06×** | 0.99×      | **1.05×** |
| Qwen3-8B              | Q4_K_M      | **1.42×** | **1.03×** | **1.02×**  | **1.22×** |
| Ornith-1.0-9B         | Q4_K_M      | **1.41×** | **1.04×** | **1.04×**  | **1.47×** |
| Qwen3.5-9B            | Q4_K_M      | **1.41×** | **1.06×** | **1.06×**  | **1.47×** |
| Qwen3.5-9B (MTP)²     | Q4_K_M      | **1.43×** | **1.01×** | **1.01×**  | **1.48×** |
| Qwen3.5-9B (MTP)²     | UD-Q4_K_XL  | **1.41×** | **1.01×** | **1.01×**  | **1.40×** |
| Gemma-3-12B           | Q4_K_M      | **1.34×** | **1.13×** | **1.14×**  | **1.55×** |
| Gemma-4-12B           | Q4_K_M      | **1.36×** | **1.13×** | **1.11×**  | **1.52×** |
| Qwen3-14B             | Q2_K³       | **1.26×** | 0.90×     | 0.90×      | **1.14×** |
| Qwen3-14B             | Q4_K_M      | **1.23×** | **1.02×** | **1.00×**  | **1.07×** |
| Qwen3-14B             | Q8_0⁷       | **1.16×** | 0.99×     | 0.98×      | 0.95×     |
| Gemma-4-26B-A4B (MoE) | UD-Q4_K_M⁹  | **1.18×** | **1.06×** | **1.07×**  | **1.37×** |
| Qwen3.6-27B           | Q4_K_M†     | **1.26×** | **1.03×** | 0.99×      | **1.16×** |
| Qwen3-30B-A3B (MoE)   | Q4_K_M⁹     | **1.11×** | **1.00×** | 0.96×      | **1.05×** |
| Gemma-4-31B           | UD-Q5_K_XL⁴ | **1.07×** | **1.03×** | **1.04×**  | **1.15×** |
| Ornith-1.0-35B        | Q4_K_M⁵     | **1.04×** | **1.04×** | **1.04×**  | **1.37×** |
| Qwen3.6-35B-A3B (MoE) | UD-IQ3_S⁶   | **1.18×** | 0.92×     | 0.92×      | **1.19×** |
| Qwen3.6-35B-A3B (MoE) | UD-Q4_K_M   | **1.20×** | **1.02×** | **1.00×**  | **1.40×** |

† at-depth cells run with the submit splitter armed — see the box above.

**`pp4@d4096` is not comparable to the pre-`2241e60` table.** That commit gave
`bench_vulkan` an untimed warm rep at the measured shape; before it, rep 1 was a
cold rep costing 1.8–3.5× steady state and was averaged in. The column's
absolutes moved by up to +26% from that alone (the IQ3_S MoE, 232 → 290 t/s), so
diffing this column against any earlier snapshot measures the bench fix, not the
engine. The other three columns are comparable modulo the oracle change.

**Column by column.** `pp4@d4096` — multi-turn ingest, the shape a coding agent
actually runs — is the strongest column at **34 of 35** rows and up to
**2.19×**, with the sub-1B rows (1.7×–2.2× on Qwen3-0.6B) the clearest wins.
`pp512` also reads **34 of 35**, peaking at **1.43×**, and unlike in previous
snapshots that count is stable: repeated runs move it by ~1.5%, not by eight
rows.

Decode: `tg128` wins **31 of 35** and `tg64@d4096` **26 of 35**. `tg64@d4096` is
still the softest column — 9 rows below 1.0 — and it is where the remaining work
is.

**What the decode-at-depth slices actually did.** The previous table warned its
decode-at-depth cells were "stale by 4–8%" because `attn_decode.comp` landed
after it. This table includes that kernel, and the correction is **not
family-wide**, exactly as backlog **B15** predicted. Per-row `tg64@d4096` change
against the previous snapshot: Qwen3-30B-A3B **0.91× → 0.96×** (its infr
absolute lands on 170.4 t/s, the figure B7 measured for the specialized kernel),
Qwen3-14B Q2_K 0.84× → 0.90×, Qwen3-14B Q4_K_M 0.94× → 1.00×; but Gemma-3-12B
1.13× → 1.14× and Gemma-4-12B 1.10× → 1.11×, i.e. inside noise. Averaged over
all 35 rows the column moved **+0.021×** — a couple of percent, concentrated on
Qwen and MoE rows. B15's measured 1.5% LOSS on gemma-3-12b at d32768 is beyond
this table's depth and is unaffected by these numbers.

The losses concentrate on **Qwen3-14B's Q2_K and Q8_0 files, the IQ3_S MoE, and
Llama-3.2-1B at depth** — Qwen3-14B Q4_K_M, which used to be in that cluster, is
now a win in all four columns.

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
wall. (Those are the slice's own A/B figures on Qwen3-14B, not this table's
Qwen3-0.6B row.)

² **MTP speculative decode is currently DISABLED** — see "MTP is parked" in
[mtp.md](../mtp.md). These rows are the models' ORDINARY (non-speculative)
numbers, which is how the MTP-head GGUFs now run. `INFR_MTP=1` is ignored with a
warning; the `mtp128` column is no longer measured.

³ **The int8-activation decode tier.** Quantizing the _activations_ to int8 and
integer-dotting them against the raw weight codes (`dotPacked4x8AccSatEXT`, the
`mmvq` shape) avoids dequantizing weights to f32 at all. On AMD the tier is now
default-on for **Q2_K, Q4_K, Q6_K, Q4_0, Q5_0, Q5_1, IQ4_NL**; **ordinary
prefill takes it for every integer dtype** (all 12). This row (Qwen3-14B Q2_K)
is what it bought at 2 bits: tg128 0.74× → 0.81×, tg64@d4096 0.72× → 0.78×, and
`pp4@d4096` 0.98× → a win. Successive levers since (the wide rmsnorm, footnote
⁸, and the decode-attention specialization) have carried it to the **0.90× /
0.90× / 1.14×** this table measures — better than the 0.84× floor the previous
snapshot recorded, and still the densest loss cluster in the table.

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
of 5.5). The d4096 row went 0.08× → 0.90× (28 vs 31 t/s) at that slice, and
reads **1.04×** here. The same slice also reuses empty KV slots instead of
forking a duplicate (`f74556c` — was silently wasting a full KV per session,
6.25 GiB on a 14B), and lifted the gemma-family multi-turn rows.

This row's `pp4@d4096` was once the table's worst loss at 0.84×; it is now
**1.15×**. That came from Q5*K's ordinary-prefill int8 tier (footnote ³) — this
is a Q5_K_XL file, and Q5_K's prefill win (+45%) was previously unreachable
because it was gated behind an off-by-default \_decode* tier. Splitting the two
policies banked it. Its **decode** was then closed too (0.91×/0.92× at the time
→ **1.03×/1.04×** here) by the wide rmsnorm (footnote ⁸) — this model, at 21.9
GiB on a 24 GB card and 57% of its GPU time in one Q5_K GEMV, is where that
kernel was found. Note this is also the row that drops to a **256-row prefill
chunk** at depth (see the box above): the fit is that tight.

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
**31.7 → 8.4 ms.** `pp512` 0.89× → **1.04×** here, and every one of Ornith-35B's
four metrics is a win. It also lifted the other DeltaNet models (Ornith-9B, the
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

**Prefill is a WIN** (`c7c3f50`): `pp512` 0.90× → **1.18×** here. The gap was
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

**Decode is still a loss** (**0.92× / 0.92×**) and the remaining lever is
_quantified but deliberately not taken_: ablating the codebook staging entirely
measures `native_idm_iq2s` 49.8 → **23.3 ms** and `native_idm_iq3s` 42.0 →
**18.9 ms** — i.e. **~50 ms of a 505 ms decode is pure per-workgroup re-staging
of the codebook into LDS**, which is essentially the whole residual gap. The fix
is to make the codebook **L2-resident in a buffer** instead of re-staged by
every workgroup (this also frees 8 KB of LDS per workgroup, so it should beat
the ablation). That needs a new SSBO binding across every grid GEMV variant and
re-validation of all 7 grid dtypes — a campaign of its own, not a slice. Null:
an SG tier for IQ2_S gate/up **regressed** hard (`native_idm_iq2s` 49.8 → 117.2
ms — 8 KB of LDS on a single-wave workgroup collapses occupancy).

This row is also the one B6 named as the table's worst variance offender, at
20.2% peak-to-peak on `pp4@d4096`. After the warm-rep fix it is **2.6%** over
four runs — the most reproducible large-model row in the sweep.

⁷ **The legacy 32-block quants now have an int8 dp4a GEMV**, not just a dp4a
GEMM. The dp4a _GEMM_ (`native_gemm_mmq_*`) has covered ~17 dtypes for a while,
but the dp4a _GEMV_ (`native_mmv_mrow.comp`) covered only the six k-quants +
IQ4\*XS — so every non-k-quant integer file fell to the f32 dequant path at
decode AND at small-m prefill, which is exactly why this Q8_0 row was one of the
table's `pp4@d4096` LOSSES. Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/IQ4_NL now have `wdec` arms
(the mmq unpack, word-parallelized: aligned/funnel-shifted u32 loads — every
`_0`-family stride is 2 mod 4 — nibble masks, SWAR zero-point rebias, a
4-bit→4-byte-lane `qh` spread, and Q4_1/Q5_1's additive min folded through the
ones-dot against `sact`). Measured on Qwen3-14B (7900 XTX), int8 vs the f32 GEMV
that shipped before, **ordinary prefill** (`pp4@d4096`): Q4_0 **+66.9%**, Q5_0
**+64.0%**, Q5_1 **+42.2%**, Q4_1 **+32.9%**, Q8_0 **+28.8%** (128 → 158 t/s),
IQ4_NL **+20.7%**. **Decode** (`tg64`) is a separate policy and splits: Q5_0
**+16.8%**, Q4_0 **+10.5%**, IQ4_NL **+6.3%**, Q5_1 **+6.1%** are default-ON;
**Q8_0 −4.2%, and Q4_1 a wash, are default-OFF** (prefill-only). Q8_0's decode
loss is structural, not a wart to fix — at 8 bits the stored byte already IS the
dp4a operand, so there is no unpack ALU to save, and decode is
weight-bandwidth-bound while the int8 route still pays the `quant_q8` bubble
(llama.cpp excludes Q8_0 from mmvq off old GCN for the same reason). Hence this
row's `tg64@d4096` sits at 0.98×: the fix was a prefill fix.

**Correction to what this footnote used to claim.** It read "this row: 0.92× →
1.18×" for `pp4@d4096`, which was the SLICE's own A/B (infr before vs infr
after), never a table cell — the table it sat next to said 0.88×, and B14 raised
exactly that disagreement. Measured here against b9833 the cell is **0.95×**:
still the table's only `pp4@d4096` loss, and still the only Qwen3-14B file
losing three of four columns. Guards for the kernels themselves are unchanged:
`mmv_mrow_legacy_formats` (each `wdec` vs a from-scratch host reference,
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
is what turned the entire 8B–27B decode band from losses into wins (`tg128` at
the slice: Qwen3-8B 0.96× → 1.02×, Qwen3-14B Q4_K_M 0.97× → 1.02×, Gemma-3-12B
1.02× → 1.12×, Gemma-4-26B MoE 1.01× → 1.13×; this table's current values for
those four are 1.03× / 1.02× / 1.13× / 1.06×).

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
is the one row none of this can help: it is the only non-integer weight dtype,
so there is no unpack ALU to save and no weight codes to integer-dot. It is no
longer a loss — this sweep reads `tg128` **1.00×** and `tg64@d4096` 0.99×,
against 0.98×/0.98× last time — but it is parity, not a win, and it will stay
there.

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
unconditionally would have been a wash that slowed the dominant op. `pp512` at
that slice: Qwen3-30B-A3B 0.95× → 1.09×, Gemma-4-26B-A4B 1.07× → 1.15×; this
sweep reads **1.11×** and **1.18×**.

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

**Where infr wins.** Prefill is now the strongest AND the reproducible half:
`pp512` **34 of 35** and `pp4@d4096` **34 of 35**, peaking at **2.19×**, and
both counts hold across repeated runs of the same binary (~1.5% ratio movement,
never a row-count swing). `pp4@d4096` — multi-turn ingest, what a coding agent
actually runs — is the strongest column in the table, roughly 1.5–2× on the
small and mid models. Decode is the narrower win: `tg128` **31 of 35**,
`tg64@d4096` **26 of 35**, both reproducible to under 3%.

**Where infr loses.** 15 losing cells of 140 (was 21), and 13 of the 15 are
DECODE. Three clusters:

- **Qwen3-14B Q2_K and Q8_0** — the densest cluster. Q2_K is worst (`tg128`
  0.90×, `tg64@d4096` **0.90×**), and Q8_0 loses three columns (0.99× / 0.98× /
  0.95×). Q2_K has been lifted by three successive levers (0.74×/0.72× → the
  int8 tier → the wide rmsnorm → the decode-attention specialization) without
  being closed, and it remains the only row whose gap has never been root-caused
  to a named kernel; it deserves its own profile rather than another inherited
  hypothesis. Q8_0's decode side is structural (footnote ⁷); its `pp4@d4096`
  0.95× is not, and is unexplained.
- **Qwen3.6-35B-A3B UD-IQ3_S** (0.92× on both decode columns). **Fully diagnosed
  and deliberately not fixed**: ~50 ms of its decode is per-workgroup re-staging
  of the codebook into LDS (footnote ⁶). Making the codebook L2-resident should
  close essentially all of it, but it touches every grid GEMV variant and all 7
  grid dtypes — a campaign, not a slice.
- **Small-model decode at depth.** Llama-3.2-1B is the worst cell in the whole
  table at `tg64@d4096` **0.88×** (Q8_0) / 0.94× (Q4_K_M), while both its other
  columns win. Reproducible: 0.92–0.94× on Q4_K_M across four runs. Gemma-4-E2B
  (0.99×) and Qwen3-0.6B BF16 (0.99×) sit just under parity in the same column.

**A prose claim the numbers overturn.** The previous snapshot said the
small-model `pp512` cluster "was prefill variance, not a real deficit". With
`pp512` now reproducible to ~1.5%, that is no longer defensible for the one row
left: **Gemma-3-1B Q4_K_M `pp512` measures 0.95× / 0.95× / 0.95× / 0.94× across
four independent runs.** It is a real, repeatable deficit — small, but a result,
not noise. Its Q2_K (1.05×) and Q8_0 (1.30×) siblings win the same column, so it
is dtype-specific, not architectural.

The other structural row is **BF16**, the only non-integer weight dtype in the
table, so there is no unpack ALU to save and no weight codes to integer-dot —
out of reach of everything that fixed the other rows. It has drifted up to
parity (`pp512` 1.08×, decode 1.00× / 0.99×) rather than being fixed.

**A loss the table does not show.** `infr compare`'s deep-context turn shapes
(8k–32k KV, beyond this table's 4096) still lose on the MoE rows and get
**monotonically worse with depth**. Re-measured 2026-08-03 on Qwen3-30B-A3B
Q4_K_M against the same b9833 oracle, r=3:

| depth | `pp512` infr/llama  | `tg128` infr/llama | `pg8192,512` (whole turn) |
| ----- | ------------------- | ------------------ | ------------------------- |
| 8192  | 1773.1 / 1686.8 t/s | 147.1 / 161.2 t/s  | 871.4 / 945.8 t/s         |
|       | **1.05×**           | 0.91×              | 0.92×                     |
| 16384 | 1138.8 / 1179.0     | 116.3 / 137.9      | 622.0 / 736.0             |
|       | 0.97×               | 0.84×              | 0.85×                     |
| 32768 | 669.9 / 735.3       | 71.3 / 110.4       | 401.6 / 517.6             |
|       | 0.91×               | **0.65×**          | **0.78×**                 |

The published table tops out at d4096 and so flatters us at exactly the shape a
long-lived agent session actually reaches. **It is decode, not prefill** —
`pp512` holds within 9% of parity all the way to 32k while `tg128` falls to
0.65×. `attn_partial` is 59% of decode GPU time at d32768.

These are better than the previous snapshot's 0.84× / 0.76× / 0.60× `tg128` and
0.878 / 0.787 / 0.727 whole-turn figures, and the decode-attention
specialization (`attn_decode.comp`) is why: it is bit-identical to the old
kernel, so it changed speed only. The shape of the problem is unchanged. Three
designs were measured against it and two are declined outright (GQA
head-grouping 1.87× slower, an LDS-staged K-tile 2.7× slower); backlog **B7**
carries the numbers, and **B15** records that the same specialization LOSES 1.5%
on gemma-3-12b at d32768, so it is not a free win everywhere.

Note that the submit splitter arms on four of these nine legs: **cap 269 on the
`pp512`/`tg128` legs at d32768**, and **342 / 222 on `pg8192,512` at d16384 /
d32768**. The other five legs report `submit unlimited`, as does every cell of
the main table except Qwen3.6-27B's. Those four rows are therefore measured with
a split submit structure — see the box at the top and backlog **B17**.

**DiffusionGemma** (`dg-step`, the in-step-parallel throughput) beats the
reference fork at **1.15×** (three runs: 1.16× / 1.14× / 1.16×; 1228 vs
1059–1074 tok/s). That is down from the 1.23× the previous snapshot recorded,
and the movement is entirely on the oracle's side — infr measured 1227–1228 in
all three runs. The DG oracle is the reference fork's own `llama-diffusion-cli`
(`~/Projects/mxaddict/llama.cpp-dg/build-vulkan`), reached via
`INFR_LLAMA_DIFFUSION_CLI` because the system copy is broken by the same package
mismatch as `llama-bench`.

**Ternary-Bonsai (Q2_0) — infr is the only engine that runs these on a GPU.**
llama.cpp merged the **Q2_0** weight dtype (GGML type 42) but shipped **no GPU
kernels for it**: there is not a single `q2_0` reference in its `ggml-vulkan/`
or `ggml-cuda/` trees, so every backend but CPU refuses to load these files.
infr runs Q2_0 natively on Vulkan (in-shader dequant + dp4a mmq — `ad89fb4`), so
the comparison below is **absolute throughput on different devices, not a
like-for-like ratio**: infr on the RX 7900 XTX vs llama.cpp on a Ryzen 9 9950X3D
(16 threads, Release + `GGML_NATIVE`). r=3, **2026-07-12 — NOT re-measured in
the 2026-08-03 sweep**, because its oracle is a CPU llama.cpp run rather than
the `llama-bench` shim.

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
cap. (These figures predate the 2026-08-03 sweep and were not re-measured.)

**Also validated for correctness** (GPU seam vs CPU reference), beyond the perf
table: Qwen2-0.5B, Llama-3.2-1B, Gemma-4-12B (dense), and Qwen3-0.6B across
quant formats **Q4_K_M / Q5_K_M / Q6_K / Q4_0 / Q2_K / IQ4_XS / Q8_0 / BF16**
(each decoded on-device via hand-written SPIR-V, byte-identical to the CPU
dequant).

## Weights that do not fit memory — the mmap/page-cache baseline

The phase-0 measurement for the tiered weight pager
(`docs/disk-streaming-plan.md`): what the GGUF mmap plus the OS page cache
deliver today when the weights do not fit the memory the process may use. It is
the bar that design has to beat, and every later phase re-runs this harness.

`scripts/paging-baseline.py`, CPU backend, Llama-3.2-1B-Instruct **F16 (2.48
GB)**, r=2, each run started cold (`posix_fadvise(DONTNEED)` on the model). The
squeeze is a cgroup-v2 `MemoryMax` rather than a bigger model — page cache
charged to the cgroup is reclaimed under the limit, so the access pattern is the
one a 60 GB model shows on a 48 GB host while the sweep still runs in minutes.
`majflt` is major faults (`ru_majflt`), `read` is what the process pulled off
the device (`ru_inblock`).

| MemoryMax | pp512 t/s | pp majflt | pp read | tg32 t/s | tg majflt | tg read |
| --------- | --------: | --------: | ------: | -------: | --------: | ------: |
| unlimited |      48.0 |     2 407 | 1.87 GB |    21.72 |     5 584 | 2.37 GB |
| 3 GB      |      46.9 |     2 406 | 1.87 GB |    22.48 |     5 870 | 2.37 GB |
| 2 GB      |      45.9 |     4 649 | 3.62 GB | **0.96** |   460 780 |  100 GB |
| 1.5 GB    |      46.6 |     5 061 | 3.73 GB | **0.67** |   419 517 |  153 GB |
| 1.2 GB    |      47.5 |     5 046 | 3.73 GB | **0.66** |   426 478 |  154 GB |

**Decode falls 23–33×** the moment the limit bites (22.5 → 0.96 → 0.67 t/s)
while **prefill is flat** (46.9 → 46.6, −0.6%). That split is the whole physics
of streaming weights: prefill amortizes one weight sweep over 512 tokens and
read only 3.7 GB for the entire run, decode pays a sweep per token.

**The page cache is doing worse than no cache at all.** At 1.5 GB it moved 153
GB for 32 tokens — 4.8 GB per token against a 2.48 GB model, i.e. **1.9× the
whole model per token**, where never caching anything would have read 1.0×. The
extra comes from evicting by recency against a cyclic sweep (every page dropped
just before its next use) plus 4 KiB fault granularity and readahead pulling in
neighbours that are evicted before they are read. It is not a small constant to
shave: it is the policy being wrong for the access pattern, which is the case
`Pager::schedule` already handles for VRAM.

**Not measured here:** the Metal path, and any model whose blob genuinely
exceeds host RAM — the largest local blob is Llama-4-Scout Q2_K at 36.8 GiB
against 60 GB of RAM, so the cgroup squeeze is what stands in for that case.
Both are gaps in the baseline, not results. The Vulkan path is measured below,
with a forced VRAM budget standing in for a card the model overflows.

### The host tier against that baseline (CPU backend)

Same harness, same model, `--dram 1g` adding a second arm that runs with
`paging.dram=1g` — a 0.81 GB arena of 1 MiB+ weight blocks read from the file by
`pread`, evicted by the cyclic-sweep policy instead of by recency:

| MemoryMax | mode | pp512 t/s | tg32 t/s | tg majflt | tg read |
| --------- | ---- | --------: | -------: | --------: | ------: |
| 2 GB      | mmap |      45.2 |     1.01 |   489 002 |   94 GB |
| 2 GB      | dram |      43.8 | **1.29** | **1 459** |   75 GB |
| 1.5 GB    | mmap |      47.8 |     0.63 |   448 882 |  153 GB |
| 1.5 GB    | dram |      44.2 | **1.30** | **2 136** |   75 GB |

**Decode is 1.28× faster at a 2 GB cap and 2.06× at 1.5 GB**, with **210–335×
fewer major faults** — the stalls moved from 4 KiB page faults inside kernel
loops to whole-block reads at the op boundary.

**The tier's read volume does not move with the cap** (75 GB at both, against
mmap's 94 → 153 GB as the cap tightens). That is the design claim, measured:
per-pass traffic is `model − resident`, and `resident` is what the budget says,
not what the kernel decided to keep. mmap gets _worse_ as memory gets tighter
because recency eviction against a cyclic sweep re-reads what it just dropped.

**Prefill costs 3–7.5%** (45.2 → 43.8, 47.8 → 44.2). Prefill was never the
problem — it amortizes one weight sweep over 512 tokens — so this is the tier's
extra copy showing up where there was nothing to fix. Nothing here is free; this
is what it costs.

Reproduce: `scripts/paging-baseline.py MODEL --limits 2G,1.5G --dram 1g`.

### The same tier under Vulkan dense streaming — now ahead of mmap

Same harness with `--dev Vulkan0 --cache 2g` (a forced 2 GB VRAM paging budget,
identical in both arms, which is what puts the run on the dense streaming path
at all). Qwen3-14B **Q8_0 (15.70 GB)**, `--prompt 64 --gen 8 --reps 1`. Every
row below is from the SAME binary; each `dram` row is paired with the `mmap` arm
of its own run, because the two alternate within one invocation:

| MemoryMax | mode      | pp64 t/s |  tg8 t/s | tg majflt | tg read |
| --------- | --------- | -------: | -------: | --------: | ------: |
| unlimited | mmap      |    110.7 |     1.74 |     6 091 |   15 GB |
| unlimited | dram `3g` |    110.7 |     1.75 |     1 568 |   15 GB |
| 8 GB      | mmap      |     11.3 |     0.18 |    66 108 |  232 GB |
| 8 GB      | dram `3g` |     15.6 |     0.24 |     1 653 |  174 GB |
| 8 GB      | dram `7g` | **25.0** | **0.39** |     1 728 |  110 GB |

**The tier beats the mmap it replaces by 2.17x on decode** and 2.21x on prefill
at a 7 GB arena under the cap, **1.41x** at a 3 GB one, and is at parity when
memory is plentiful (1.75 vs 1.74 — it does not regress the case it is not for).
Major faults are **38x** lower under the cap and read volume falls 232 → 110 GB.

**The budget is the biggest lever in the whole feature, and nothing sets it.**
The only difference between the last two rows is `paging.dram`: 3 GB → 7 GB is
worth **1.6x on its own** (0.24 → 0.39), because a bigger arena means fewer
bytes read per pass and this regime is bound by nothing else. `paging.dram` has
no auto-sizing — an unset budget disables the tier entirely — so a user who
guesses low gets a fraction of what is here. Backlog **B36**. Note the arena
must stay ANONYMOUS memory for a large budget to be safe: the kernel then
reclaims page cache in its favour, which is why a 7 GB arena under an 8 GB cap
does not thrash (major faults flat at ~1 700). The "double-caching halves the
budget" claim this document and the plan both used to carry was **wrong**, and a
`mincore` probe is what settled it.

**Two fixes got the tier from 0.79x to here, and measurement found both.**

_The reader, worth 1.29x/0.79x._ The tier sat at 0.79x of mmap on decode until
`FileBlockIo::read_block` stopped being a single `pread`. A drive delivers its
bandwidth on queue depth: measured on this NVMe over 16-128 MB blocks, one
positioned read sustains 1.2-1.5 GB/s while the device does 2.2 GB/s at depth
2-4 (8 and 16 buy nothing). A serial reader was therefore losing to the mapping
for a structural reason — the kernel issues readahead faults in parallel for
free. One block is now split across `IO_FANOUT` concurrent positioned reads.
**Read volume, fault counts and residency policy were all unchanged** across
that fix; only bandwidth moved, which is the signature a reader-only change
should leave and the reason to believe the gain is the one claimed.

_The admission doorkeeper, worth a further 10% of bytes read._ The arena sits
under a tier that only calls down on ITS misses, and on the first pass nothing
is resident above — so admitting on the FIRST miss filled the arena with exactly
the prefix the VRAM pager then keeps resident forever, blocks that never call
down again. `INFR_PAGER_STATS` showed it plainly:
`host0: hits=50 misses=9 slots=9`, 9 slots holding 9 blocks of which only 5
could ever be hit. Admission now needs a SECOND miss, which no block the tier
above keeps ever reaches. Useful hits per pass went 5 → 9, bytes read −10.5%,
decode 0.22 → 0.24 and prefill 13.2 → 15.6.

**Before that came a fix worth 1.6x, which the first measurement is what
found.** The tier originally pinned each block in its arena and memcpy'd it to
the pinned ring, and measured 0.83 t/s decode / 54.7 t/s prefill unlimited —
0.48x and 0.50x of mmap. The cost was structural: on CPU the arena _replaces_
the mapping (the kernels read the slot directly, no copy added), but on Vulkan
the bytes must reach the ring either way, so `disk → arena → ring` is one copy
more than `page-cache → ring`. `HostPager::fill` now admits a block only while a
slot is free and, once the arena is full, reads it **straight into the ring** —
which is also the right residency call, because under a cyclic sweep the block
that just missed is the one whose next use is furthest away. Decode went 0.83 →
1.36 and prefill 54.7 → 85.8; the concurrent reader and the doorkeeper took it
from there.

**What is left is reading fewer bytes, not overlapping the reads.** This regime
is I/O-bound by orders of magnitude — roughly 12 GB read per token against tens
of milliseconds of GPU compute — so prefetch, which hides a read behind compute,
has almost nothing to hide it behind. That is why the two fixes that worked were
a faster reader and a smarter admission rule, and why the remaining items are
about volume: auto-sizing the budget (above), and a chunked prefill that
re-reads the whole model once per chunk — `ceil(prompt / ubatch)` full sweeps,
which at the 1024-row default makes a 32k prompt 32 of them. Layer-major prefill
would read it once. Both are backlog **B36**.

_Corrected, because this document asserted otherwise._ The tier was said to be
held back by double-caching — a buffered `pread` leaving a page-cache copy of
what the arena already holds — with `posix_fadvise(DONTNEED)` ruled out because
`Gguf::open` maps the whole file. A `mincore` probe says both halves were wrong.
`DONTNEED` **does** reclaim mapped-but-untouched pages (65 536 → 0 in the
probe); only pages actually faulted THROUGH the mapping are pinned, and the tier
never touches paged ranges that way. And the fix is not needed anyway: an
anonymous arena already wins page-cache reclaim under a cap, which is what the 7
GB row above demonstrates.

**`paging.dram` is still off by default on the Vulkan path**, but the reason has
changed: the performance case is now made, and what is missing is coverage
(measured on one GPU, one drive, Linux only — the concurrent reader's speedup is
explicitly unverified on Windows, where a non-overlapped handle serializes
concurrent reads).

> Numbers are a snapshot and move with each perf slice; regenerate on your own
> hardware with `infr compare --sweep <model...>`. Results on other GPUs
> (NVIDIA, Intel Arc) and Apple Metal are wanted — please open an issue with
> your `infr bench` / `infr compare` output. Intel Arc testers: include one run
> with `INFR_DEBUG_COOPMAT=1` (the enumerated/chosen coopmat shapes), then A/B
> `INFR_CM_8X8=1` (opt-in 8x8x16 XMX prefill GEMM) against the default.
