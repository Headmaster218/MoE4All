use std::process::Command;

/// Compile GLSL compute shaders (which need features WGSL/naga can't express, e.g.
/// cooperative matrix) to SPIR-V via `glslc` at build time. Output to OUT_DIR.
fn main() {
    // INFR_PROFILE=1 at build time -> cfg(infr_profile) -> #[cfg_attr(infr_profile,
    // infr_prof::instrument)] annotations become live and inject profiling spans into every fn.
    // Default builds get NO cfg and zero profiling code. See docs/perf/playbook.md.
    println!("cargo:rerun-if-env-changed=INFR_PROFILE");
    println!("cargo:rustc-check-cfg=cfg(infr_profile)");
    if std::env::var("INFR_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=infr_profile");
    }
    println!("cargo:rerun-if-changed=shaders");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    gen_grids(&out);
    // (source stem, output stem, extra glslc defines)
    let builds: &[(&str, &str, &[&str])] = &[
        ("gemm_coopmat_tiled", "gemm_coopmat_tiled", &[]),
        ("gemm_warp", "gemm_warp", &[]),
        ("quant_q8", "quant_q8", &[]),
        ("gemm_proj", "gemm_proj", &[]),
        ("gemm_proj_warp", "gemm_proj_warp", &[]),
        ("attn_partial", "attn_partial", &[]),
        // PROBE (INFR_MROWS_ATTN=1): RB=4 rows-batched pass 1 — occupancy-hypothesis experiment.
        ("attn_partial_mrows", "attn_partial_mrows", &[]),
        (
            "attn_partial_mrows",
            "attn_partial_mrows_c256",
            &["-DSC_MAX=256u"],
        ),
        // A/B escape (INFR_NO_ATTN_HD=1): compile out the hd=256/512 fast paths so a regression
        // on those shapes is diagnosable against the general loops (f16 form-factors only).
        ("attn_partial", "attn_partial_nohd", &["-DNO_HD_SPEC"]),
        ("attn_partial", "attn_partial_dyn", &["-DUSE_PARAMS"]),
        (
            "attn_partial",
            "attn_partial_dyn_nohd",
            &["-DUSE_PARAMS", "-DNO_HD_SPEC"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac",
            &["-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_nohd",
            &["-DUSE_PARAMS", "-DSELF_CHUNK", "-DNO_HD_SPEC"],
        ),
        // Planar Q8_0 KV cache: coalesced split-K decode reading Q8 blocks (halves KV read traffic).
        // K and V decouple (-DKQ8 / -DVQ8) → 3 quant combos (kq8, vq8, both) for the STATIC split.
        ("attn_partial", "attn_partial_q8", &["-DKQ8", "-DVQ8"]),
        ("attn_partial", "attn_partial_kq8", &["-DKQ8"]),
        ("attn_partial", "attn_partial_vq8", &["-DVQ8"]),
        // Record-once (USE_PARAMS) is disabled for a Q8 cache (forced static), so only the coupled
        // variant is kept for the dyn/dynac form-factors (referenced but unreachable at runtime).
        (
            "attn_partial",
            "attn_partial_dyn_q8",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_q8",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        // KV-cache u64/BDA twins (#74, slice 2): `-DKV_BDA` reads the K/V cache by 64-bit device
        // address (kv_addr.glsl, wide KV2 b64 for the f16 vec4 loads / scalar kv_word for Q8) instead
        // of bound SSBOs at bindings 1/2 — the flash-decoding mirror of slice 1's scalar attention_kv.
        // The bound builds above stay compiled — attn_partial stays DUAL-ARM (its .comp also serves
        // the quantized-KV-prefill split read off the dequant f16 scratch, which is Activations with no
        // device address → bound). Production forks to these when the KV buffers report a device address
        // (see Recorder::attention_kv_split_at / _dynac_at and adapter.rs's `device_addr()` fork).
        ("attn_partial", "attn_partial_bda", &["-DKV_BDA"]),
        (
            "attn_partial",
            "attn_partial_nohd_bda",
            &["-DNO_HD_SPEC", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_kq8_bda",
            &["-DKQ8", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_vq8_bda",
            &["-DVQ8", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_q8_bda",
            &["-DKQ8", "-DVQ8", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_dyn_bda",
            &["-DUSE_PARAMS", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_dyn_nohd_bda",
            &["-DUSE_PARAMS", "-DNO_HD_SPEC", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_bda",
            &["-DUSE_PARAMS", "-DSELF_CHUNK", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_nohd_bda",
            &["-DUSE_PARAMS", "-DSELF_CHUNK", "-DNO_HD_SPEC", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_dynac_q8_bda",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS", "-DSELF_CHUNK", "-DKV_BDA"],
        ),
        (
            "attn_partial_mrows",
            "attn_partial_mrows_c256_bda",
            &["-DSC_MAX=256u", "-DKV_BDA"],
        ),
        // PROBE (B7 slice 1, tests/attn_ktile_probe.rs only — NOTHING in production dispatches
        // these): LDS-staged K-tile flash-decoding pass 1, one whole 128-dim dot per THREAD out of
        // shared memory, so attn_partial's per-key `subgroupAdd` disappears. Four configurations:
        // two workgroup/tile sizes (64 and 128 keys), a half-depth stage (KDW=32 → half the LDS,
        // two dim passes per tile) and a KPAD=0 build whose only purpose is to MEASURE the LDS
        // bank-conflict penalty the padding avoids.
        ("attn_ktile", "attn_ktile_w64", &["-DKWG=64"]),
        (
            "attn_ktile",
            "attn_ktile_w64_nopad",
            &["-DKWG=64", "-DKPAD=0"],
        ),
        ("attn_ktile", "attn_ktile_w128", &["-DKWG=128"]),
        (
            "attn_ktile",
            "attn_ktile_w64_dw32",
            &["-DKWG=64", "-DKDW=32"],
        ),
        // PROBE (B7 slice 2, tests/attn_dsplit_probe.rs only — NOTHING in production dispatches
        // these): attn_partial's hd==128 QK loop with the subgroup-reduction WIDTH as a build-time
        // constant. `-DDSW=w` makes w lanes cooperate on one key (128/w dims each, reduced with
        // subgroupClusteredAdd(., w)) so 32/w keys are in flight per wave; DSW=32 is the shipped
        // behaviour, DSW=1 the k-tile probe's per-thread dot, DSW=8 what llama.cpp picks for this
        // shape on RDNA3. `-DDWG=128` is the second axis: llama.cpp's 4-wave workgroup vs our 2.
        ("attn_partial_dsplit", "attn_dsplit_w1", &["-DDSW=1"]),
        ("attn_partial_dsplit", "attn_dsplit_w2", &["-DDSW=2"]),
        ("attn_partial_dsplit", "attn_dsplit_w4", &["-DDSW=4"]),
        ("attn_partial_dsplit", "attn_dsplit_w8", &["-DDSW=8"]),
        ("attn_partial_dsplit", "attn_dsplit_w16", &["-DDSW=16"]),
        ("attn_partial_dsplit", "attn_dsplit_w32", &["-DDSW=32"]),
        (
            "attn_partial_dsplit",
            "attn_dsplit_w1_wg128",
            &["-DDSW=1", "-DDWG=128"],
        ),
        (
            "attn_partial_dsplit",
            "attn_dsplit_w2_wg128",
            &["-DDSW=2", "-DDWG=128"],
        ),
        (
            "attn_partial_dsplit",
            "attn_dsplit_w4_wg128",
            &["-DDSW=4", "-DDWG=128"],
        ),
        (
            "attn_partial_dsplit",
            "attn_dsplit_w8_wg128",
            &["-DDSW=8", "-DDWG=128"],
        ),
        (
            "attn_partial_dsplit",
            "attn_dsplit_w16_wg128",
            &["-DDSW=16", "-DDWG=128"],
        ),
        (
            "attn_partial_dsplit",
            "attn_dsplit_w32_wg128",
            &["-DDSW=32", "-DDWG=128"],
        ),
        // B7 slice 3a — DECODE-ONLY specialization of the split-K pass 1 (`INFR_NO_ATTN_DECODE=1`
        // falls the whole thing back to `attn_partial`). Same kernel source in the two form-factors
        // production dispatches: the STATIC push-constant build (Recorder::attention_kv_split_impl,
        // used under INFR_SEAM_NO_REPLAY and by small-m callers at rows==1) and the record-once
        // SELF_CHUNK replay build (attention_kv_split_dynac_impl, what real decode runs). K/V are
        // always read by device address, so there is no bound-SSBO twin — the host gate requires it.
        ("attn_decode", "attn_decode", &[]),
        (
            "attn_decode",
            "attn_decode_dynac",
            &["-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        // B7 slice 3b — the same decode-only kernel widened to the shapes that were falling back,
        // as a FAMILY of build-time specializations (one kernel handling every axis at runtime is
        // just `attn_partial` again, and the whole win was deleting arms). Two axes:
        //   `-DSWA -DRING`  sliding-window layers (gemma3/gemma4 run SWA on most layers) INCLUDING
        //                   the ring cache those layers allocate (`window + ubatch` rows, position
        //                   j at row `j % rcap`). SWA always pairs with RING: the modulo is the
        //                   identity on a full-context cache, so one build covers both and the
        //                   host gate needs no cap reasoning.
        //   `-DDHD4=64/128` hd 256 / 512 (gemma4's per-layer head dims, qwen3.5) — `attn_partial`'s
        //                   dedicated hd4==64 / hd4==128 arms, ported with literal strides.
        ("attn_decode", "attn_decode_swa", &["-DSWA", "-DRING"]),
        (
            "attn_decode",
            "attn_decode_swa_dynac",
            &["-DSWA", "-DRING", "-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        ("attn_decode", "attn_decode_hd256", &["-DDHD4=64"]),
        (
            "attn_decode",
            "attn_decode_hd256_dynac",
            &["-DDHD4=64", "-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        (
            "attn_decode",
            "attn_decode_hd256_swa",
            &["-DDHD4=64", "-DSWA", "-DRING"],
        ),
        (
            "attn_decode",
            "attn_decode_hd256_swa_dynac",
            &[
                "-DDHD4=64",
                "-DSWA",
                "-DRING",
                "-DUSE_PARAMS",
                "-DSELF_CHUNK",
            ],
        ),
        ("attn_decode", "attn_decode_hd512", &["-DDHD4=128"]),
        (
            "attn_decode",
            "attn_decode_hd512_dynac",
            &["-DDHD4=128", "-DUSE_PARAMS", "-DSELF_CHUNK"],
        ),
        (
            "attn_decode",
            "attn_decode_hd512_swa",
            &["-DDHD4=128", "-DSWA", "-DRING"],
        ),
        (
            "attn_decode",
            "attn_decode_hd512_swa_dynac",
            &[
                "-DDHD4=128",
                "-DSWA",
                "-DRING",
                "-DUSE_PARAMS",
                "-DSELF_CHUNK",
            ],
        ),
        // Lever 1 (kv-decode-perf-levers #1): mainline low-bit KV decode reads the compact GGUF
        // block format INLINE (native_decode `dq()` on the cache via `dq_addr`=k_addr/v_addr), so
        // Q4/Q5 decode moves ¼ the traffic vs the dequant→f16 prepass + f16-scratch read. K and V
        // always share the cache dtype, so both sides route through one FMT (-DKMAINLINE -DVMAINLINE).
        // Always -DKV_BDA (the inline decode reads by device address). `_nohd` twin for
        // INFR_NO_ATTN_HD A/B (matches the f16 `attn_partial_nohd_bda`). Selected by
        // Recorder::attention_kv_split_impl's `kv_ml` arm; adapter.rs's `decode_inline` gate.
        (
            "attn_partial",
            "attn_partial_ml_q4_0_bda",
            &["-DKMAINLINE", "-DVMAINLINE", "-DFMT_Q4_0", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q4_0_nohd_bda",
            &[
                "-DKMAINLINE",
                "-DVMAINLINE",
                "-DFMT_Q4_0",
                "-DNO_HD_SPEC",
                "-DKV_BDA",
            ],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q4_1_bda",
            &["-DKMAINLINE", "-DVMAINLINE", "-DFMT_Q4_1", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q4_1_nohd_bda",
            &[
                "-DKMAINLINE",
                "-DVMAINLINE",
                "-DFMT_Q4_1",
                "-DNO_HD_SPEC",
                "-DKV_BDA",
            ],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q5_0_bda",
            &["-DKMAINLINE", "-DVMAINLINE", "-DFMT_Q5_0", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q5_0_nohd_bda",
            &[
                "-DKMAINLINE",
                "-DVMAINLINE",
                "-DFMT_Q5_0",
                "-DNO_HD_SPEC",
                "-DKV_BDA",
            ],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q5_1_bda",
            &["-DKMAINLINE", "-DVMAINLINE", "-DFMT_Q5_1", "-DKV_BDA"],
        ),
        (
            "attn_partial",
            "attn_partial_ml_q5_1_nohd_bda",
            &[
                "-DKMAINLINE",
                "-DVMAINLINE",
                "-DFMT_Q5_1",
                "-DNO_HD_SPEC",
                "-DKV_BDA",
            ],
        ),
        ("attn_qk", "attn_qk", &[]),
        ("attn_qk_warp", "attn_qk_warp", &[]),
        ("attn_flash", "attn_flash", &[]),
        ("attn_flash", "attn_flash_bm32", &["-DBM_TILE=32"]),
        ("attn_flash_partial", "attn_flash_partial", &[]),
        (
            "attn_flash_partial",
            "attn_flash_partial_bm32",
            &["-DBM_TILE=32"],
        ),
        // KV-cache u64/BDA twin (#74 slice 4 resurrection): `-DKV_COOPMAT_BDA` reads the K/V cache by
        // 64-bit device address (kv_addr.glsl `KvMat(k_addr)/(v_addr)` coopMatLoad base) instead of the
        // bound SSBOs at bindings 1/2 — which stay bound as INERT descriptors for Recorder::sync's
        // store→read hazard edge. OPT-IN (INFR_KV_COOPMAT_BDA=1), default OFF: on RADV/RDNA3 the coopmat
        // load gets no saddr from a buffer_reference base (~+33-40% code, ~0.8x prefill) so the bound
        // build stays the default; kept for silicon that addresses coopmat loads better. All new code is
        // under `#ifdef KV_COOPMAT_BDA`, so the base .spv above is byte-identical. Not combined with the
        // STAGE/DEQUANT builds (a separate, mutually-exclusive experimental flag family).
        (
            "attn_flash_partial",
            "attn_flash_partial_bda",
            &["-DKV_COOPMAT_BDA"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_bm32_bda",
            &["-DKV_COOPMAT_BDA", "-DBM_TILE=32"],
        ),
        // Levers 2 & 5 (kv-decode-perf-levers): shmem-staged coopMatLoad variants of the coopmat
        // prefill partial. `_stage` = f16 K/V staged into a 32-wide `KVstg` slab then coopMatLoad
        // from shmem (Lever 5, INFR_FLASH_STAGE). `_deq_<fmt>` = the quantized KV cache decoded to
        // f16 in the stage via native_decode's coalesced `dqblk`, skipping the dequant_kv_f16
        // prepass (Lever 2, INFR_FLASH_DEQUANT). Both keep the default path (above) untouched — the
        // staging code is entirely under `#ifdef STAGE`, so the base .spv is byte-identical. bm32
        // twins mirror the base tile pick (small-m split tiles / INFR_FLASH_BM=32 parity coverage).
        (
            "attn_flash_partial",
            "attn_flash_partial_stage",
            &["-DSTAGE"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_stage_bm32",
            &["-DSTAGE", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q4_0",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_0"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q4_0_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_0", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q4_1",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_1"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q4_1_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_1", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q5_0",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_0"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q5_0_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_0", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q5_1",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_1"],
        ),
        (
            "attn_flash_partial",
            "attn_flash_partial_deq_q5_1_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_1", "-DBM_TILE=32"],
        ),
        // Q8_0 is DEFERRED: infr's Q8_0 KV cache is PLANAR (codes[cap] then scales[cap/32]; see
        // store_q8 / dequant_q8_f16), NOT 34-byte interleaved GGUF Q8_0 blocks — so native_decode's
        // FMT_Q8_0 `dqblk` (which this shared substrate uses) would misread it. A Q8 dequant-in-flash
        // needs a bespoke planar decoder + a `cap` push constant; out of scope for the GGUF-block
        // substrate. iq4_nl (codebook gather) is likewise left on the prepass.
        ("attn_flash_warp", "attn_flash_warp", &[]),
        // BM=32 tile: 29056 B shared (vs 58112 B), fits NVIDIA (48 KB) / MoltenVK (32 KB) devices
        // whose maxComputeSharedMemorySize is under the 64 KB the default BM=64 tile needs.
        ("attn_flash_warp", "attn_flash_warp_bm32", &["-DBM_TILE=32"]),
        // KV-cache u64/BDA twins (#74 slice 4 resurrection): the hd==128 register-O DEFAULT prefill
        // kernel reading K/V by device address (INFR_KV_COOPMAT_BDA=1, default OFF). See the
        // attn_flash_partial_bda note above; base .spv byte-identical (all new code under the ifdef).
        (
            "attn_flash_warp",
            "attn_flash_warp_bda",
            &["-DKV_COOPMAT_BDA"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_bm32_bda",
            &["-DKV_COOPMAT_BDA", "-DBM_TILE=32"],
        ),
        // Lever 2 on the WARP kernel (kv-decode-perf-levers): dequant-in-flash that KEEPS the fast
        // register-tiled warp path (unlike attn_flash_partial's Lever-2 arm, which forfeits warp).
        // The quantized GGUF block cache is decoded to f16 in the QK/PV stage via native_decode's
        // coalesced `dqblk` (32-wide KVstg slab, +4096 B), skipping the dequant_kv_f16 prepass. The
        // staging is entirely under `#ifdef STAGE`, so the base warp .spv above stays byte-identical.
        // bm32 twins mirror the small-m deep-kv split tiles (rows 24..64) / INFR_FLASH_BM=32 parity.
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q4_0",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_0"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q4_0_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_0", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q4_1",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_1"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q4_1_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q4_1", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q5_0",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_0"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q5_0_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_0", "-DBM_TILE=32"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q5_1",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_1"],
        ),
        (
            "attn_flash_warp",
            "attn_flash_warp_deq_q5_1_bm32",
            &["-DSTAGE", "-DDEQUANT", "-DFMT_Q5_1", "-DBM_TILE=32"],
        ),
        ("attn_flash_reg", "attn_flash_reg", &[]),
        // BR=64 tile: 29440 B shared (vs 58880 B) for sub-64 KB shared devices (NVIDIA, MoltenVK).
        ("attn_flash_reg", "attn_flash_reg_br64", &["-DBR_TILE=64"]),
        // KV-cache u64/BDA twins (#74 slice 4 resurrection): K/V by device address (INFR_KV_COOPMAT_BDA
        // =1, default OFF). See the attn_flash_partial_bda note; base .spv byte-identical.
        (
            "attn_flash_reg",
            "attn_flash_reg_bda",
            &["-DKV_COOPMAT_BDA"],
        ),
        (
            "attn_flash_reg",
            "attn_flash_reg_br64_bda",
            &["-DKV_COOPMAT_BDA", "-DBR_TILE=64"],
        ),
        ("attn_flash_combine", "attn_flash_combine", &[]),
        ("attn_softmax", "attn_softmax", &[]),
        ("attn_pv", "attn_pv", &[]),
        ("attn_pv_warp", "attn_pv_warp", &[]),
        ("attn_pv_reduce", "attn_pv_reduce", &[]),
        // MLA attention (DeepSeek V2/V3): one workgroup per token, 128 lanes covering all heads.
        ("mla", "mla", &[]),
        // YaRN freq_factors variant: the internal q_pe rope DIVIDES its angle by ff[pair]
        // (`-DFREQ_FACTORS` adds the binding-5 ff buffer — see `Recorder::mla`).
        ("mla", "mla_ff", &["-DFREQ_FACTORS"]),
        ("rmsnorm", "rmsnorm", &[]),
        // Decode twin of `rmsnorm`: 1024 threads + vec4 loads in the single rows==1 workgroup, to
        // buy back the memory-level parallelism the 256-thread build lacks (see rmsnorm.comp).
        ("rmsnorm", "rmsnorm_wide", &["-DWIDE"]),
        // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate, Op::GatedRmsNorm) —
        // same reduction as `rmsnorm`, one extra buffer + the gate multiply on store.
        ("rmsnorm", "rmsnorm_gate", &["-DGATE"]),
        ("rmsnorm", "rmsnorm_add", &["-DADD"]),
        // f16-in/f16-out RMSNorm (llama4's post-rope weightless Q/K L2-norm, `Op::QkNorm` on the
        // f16 rope scratch — `w` stays f32).
        ("rmsnorm", "rmsnorm_f16", &["-DF16IO"]),
        ("softmax", "softmax", &[]),
        // DiffusionGemma denoise self-conditioning perf: scale read from a device buffer instead
        // of a push constant (see `Op::Softmax::scale_buf`'s doc + `Recorder::softmax_dyn`).
        ("softmax", "softmax_dyn", &["-DUSE_SCALE_BUF"]),
        ("deltanet", "deltanet", &[]),
        // Strided variant: q/k/v read from single convout buffer with offsets (env-gated).
        ("deltanet_strided", "deltanet_strided", &[]),
        ("deltanet_chunked", "deltanet_chunked", &[]),
        ("deltanet_prep", "deltanet_prep", &[]),
        ("deltanet_gates", "deltanet_gates", &[]),
        ("deltanet_scan", "deltanet_scan", &[]),
        // Token-serial prefill scan with the state column register-resident (norm + gates + seq).
        // Replaces the chunked prep/gates/scan trio at kd==128; see deltanet_seq.comp.
        ("deltanet_norm", "deltanet_norm", &[]),
        ("deltanet_gates_seq", "deltanet_gates_seq", &[]),
        ("deltanet_seq", "deltanet_seq", &[]),
        ("conv1d_silu", "conv1d_silu", &[]),
        ("conv1d_silu_par", "conv1d_silu_par", &[]),
        ("conv1d_shift", "conv1d_shift", &[]),
        ("copy_strided", "copy_strided", &[]),
        ("mul_sigmoid", "mul_sigmoid", &[]),
        ("add", "add", &[]),
        ("add_bias", "add_bias", &[]),
        ("mul_vec", "mul_vec", &[]),
        ("moe_shared_expert_add", "moe_shared_expert_add", &[]),
        ("add_scaled", "add_scaled", &[]),
        ("scale", "scale", &[]),
        ("softcap", "softcap", &[]),
        ("silu_mul", "silu_mul", &[]),
        ("silu_mul", "gelu_mul", &["-DGELU"]),
        ("silu_mul_fused", "silu_mul_fused", &[]),
        ("silu_mul_fused", "gelu_mul_fused", &["-DGELU"]),
        ("store_f16", "store_f16", &[]),
        ("store_f16", "store_f16_dyn", &["-DUSE_PARAMS"]),
        // Quantize a KV row into the Q8_0 cache. f32 source (V) + f16 source (un-fused roped K),
        // each with a record-once (pos from params) variant.
        ("store_q8", "store_q8", &[]),
        ("store_q8", "store_q8_dyn", &["-DUSE_PARAMS"]),
        ("store_q8", "store_q8_f16", &["-DSRC_F16"]),
        // Expand a Q8_0 KV prefix → f16 scratch so the f16 flash/non-FA prefill kernels can run.
        ("dequant_q8_f16", "dequant_q8_f16", &[]),
        (
            "store_q8",
            "store_q8_f16_dyn",
            &["-DSRC_F16", "-DUSE_PARAMS"],
        ),
        // Mainline low-bit KV quants (standard GGUF blocks): a quantize kernel per format (f32 V +
        // f16 K sources) and a dequant→f16 prefix expander per format (reuses native_decode `dq()`).
        // Static only — a quantized KV cache forces per-execute static decode (see decode_eligible).
        ("quant_kv", "quant_kv_q4_0", &["-DFMT_Q4_0"]),
        (
            "quant_kv",
            "quant_kv_q4_0_f16",
            &["-DFMT_Q4_0", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_q4_1", &["-DFMT_Q4_1"]),
        (
            "quant_kv",
            "quant_kv_q4_1_f16",
            &["-DFMT_Q4_1", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_q5_0", &["-DFMT_Q5_0"]),
        (
            "quant_kv",
            "quant_kv_q5_0_f16",
            &["-DFMT_Q5_0", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_q5_1", &["-DFMT_Q5_1"]),
        (
            "quant_kv",
            "quant_kv_q5_1_f16",
            &["-DFMT_Q5_1", "-DSRC_F16"],
        ),
        ("quant_kv", "quant_kv_iq4_nl", &["-DFMT_IQ4NL"]),
        (
            "quant_kv",
            "quant_kv_iq4_nl_f16",
            &["-DFMT_IQ4NL", "-DSRC_F16"],
        ),
        ("dequant_kv_f16", "dequant_kv_q4_0", &["-DFMT_Q4_0"]),
        ("dequant_kv_f16", "dequant_kv_q4_1", &["-DFMT_Q4_1"]),
        ("dequant_kv_f16", "dequant_kv_q5_0", &["-DFMT_Q5_0"]),
        ("dequant_kv_f16", "dequant_kv_q5_1", &["-DFMT_Q5_1"]),
        ("dequant_kv_f16", "dequant_kv_iq4_nl", &["-DFMT_IQ4NL"]),
        // Dense KV caches (f32/bf16): a cast-store per (dst, src) + the bf16→f16 read (native_decode
        // FMT_BF16). f32→f16 read reuses store_f16. K = f16 source, V = f32 source.
        ("store_kv_dense", "store_kv_f32", &["-DDST_F32"]),
        (
            "store_kv_dense",
            "store_kv_f32_from_f16",
            &["-DDST_F32", "-DSRC_F16"],
        ),
        ("store_kv_dense", "store_kv_bf16", &["-DDST_BF16"]),
        (
            "store_kv_dense",
            "store_kv_bf16_from_f16",
            &["-DDST_BF16", "-DSRC_F16"],
        ),
        ("dequant_kv_f16", "dequant_kv_bf16", &["-DFMT_BF16"]),
        // TurboQuant KV (WHT-rotated): quantize (f32 V + f16 K) + dequant→f16, per width.
        ("quant_turbo", "quant_turbo_t2", &["-DTURBO2"]),
        (
            "quant_turbo",
            "quant_turbo_t2_f16",
            &["-DTURBO2", "-DSRC_F16"],
        ),
        ("quant_turbo", "quant_turbo_t3", &["-DTURBO3"]),
        (
            "quant_turbo",
            "quant_turbo_t3_f16",
            &["-DTURBO3", "-DSRC_F16"],
        ),
        ("quant_turbo", "quant_turbo_t4", &["-DTURBO4"]),
        (
            "quant_turbo",
            "quant_turbo_t4_f16",
            &["-DTURBO4", "-DSRC_F16"],
        ),
        ("dequant_turbo_f16", "dequant_turbo_t2", &["-DTURBO2"]),
        ("dequant_turbo_f16", "dequant_turbo_t3", &["-DTURBO3"]),
        ("dequant_turbo_f16", "dequant_turbo_t4", &["-DTURBO4"]),
        ("rope", "rope", &[]),
        // YaRN freq_factors variant: the rotation angle is DIVIDED by ff[pair] (`-DFREQ_FACTORS`
        // adds the binding-2 ff buffer — see `Recorder::rope`).
        ("rope", "rope_ff", &["-DFREQ_FACTORS"]),
        ("rope", "rope_f16", &["-DOUT_F16"]),
        ("rope", "rope_f16_dyn", &["-DOUT_F16", "-DUSE_PARAMS"]),
        ("linear_f16", "linear_f16", &[]),
        // !caps.f16 fallback (no shaderFloat16 ext): reads the f16 weight buffer as packed u32 +
        // unpackHalf2x16 (core GLSL) instead of a float16_t SSBO read.
        ("linear_f16_noext", "linear_f16_noext", &[]),
        ("linear_bf16", "linear_bf16", &[]),
        // `linear_f32.comp` (the eager thread-per-output f32 GEMV) is GONE from this list: its sole
        // consumer was the now-deleted eager `VulkanBackend::linear` (linear.rs) — production's f32
        // Linear was ALREADY routing through the separate `linear_f32r` family below (rows>1 mrow/
        // vec4 tiles + a plain rows<=1 arm), never through `linear_f32.comp`. So unlike
        // linear_f16/linear_bf16 (whose `-DSTREAMED` twin IS the live production kernel), both of
        // `linear_f32`'s builds were dead — same as `e2b_proj` below, just not called out in the
        // audit by name.
        ("linear_f32r", "linear_f32r", &[]),
        ("linear_f32r", "linear_f32r_mrow8", &["-DMROW=8"]),
        (
            "linear_f32r",
            "linear_f32r_mrow4_v4",
            &["-DMROW=4", "-DVEC4"],
        ),
        (
            "linear_f32r",
            "linear_f32r_mrow8_v4",
            &["-DMROW=8", "-DVEC4"],
        ),
        ("e2b_gate", "e2b_gate", &[]),
        // e2b_proj (fused E2B per-layer proj GEMV+RMSNorm+Add) landed but was NEVER wired into
        // production — the E2B proj side runs unfused; only tests/weight_addr_parity.rs's
        // `e2b_proj_streamed_matches_resident` dispatched it (both the resident and streamed
        // builds), and that test is deleted along with it — there's no surviving entry point left
        // to exercise. Both builds and the shader itself (shaders/e2b_proj.comp) are gone too.
        ("matmul_f32", "matmul_f32", &[]),
        ("linear_res", "linear_res", &[]),
        ("attention", "attention", &[]),
        ("attn_combine", "attn_combine", &[]),
        ("attn_combine", "attn_combine_live", &["-DUSE_LIVE"]),
        ("attn_live", "attn_live", &[]),
        ("attention_kv", "attention_kv", &[]),
        ("attention_kv", "attention_kv_dyn", &["-DUSE_PARAMS"]),
        // Planar Q8_0 KV cache: scalar dequant-on-read variants. K/V decouple (-DKQ8 / -DVQ8) → 3
        // quant combos for the STATIC scalar path; coupled-only for the record-once (dead for Q8).
        ("attention_kv", "attention_kv_q8", &["-DKQ8", "-DVQ8"]),
        ("attention_kv", "attention_kv_kq8", &["-DKQ8"]),
        ("attention_kv", "attention_kv_vq8", &["-DVQ8"]),
        (
            "attention_kv",
            "attention_kv_dyn_q8",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS"],
        ),
        // KV-cache u64/BDA twins (#74, slice 1): `-DKV_BDA` reads the K/V cache by 64-bit device
        // address (kv_addr.glsl) instead of bound SSBOs at bindings 1/2 — the mirror of the weight
        // side's `-DSTREAMED`. The bound builds above stay compiled — attention_kv stays DUAL-ARM (its
        // .comp also serves the quantized-KV-prefill scalar read off the dequant f16 scratch, which is
        // Activations with no device address → bound). Production forks to these when the KV buffers
        // report a device address (see Recorder::attention_kv_at / adapter.rs's `device_addr()` fork).
        ("attention_kv", "attention_kv_bda", &["-DKV_BDA"]),
        (
            "attention_kv",
            "attention_kv_dyn_bda",
            &["-DUSE_PARAMS", "-DKV_BDA"],
        ),
        (
            "attention_kv",
            "attention_kv_q8_bda",
            &["-DKQ8", "-DVQ8", "-DKV_BDA"],
        ),
        (
            "attention_kv",
            "attention_kv_kq8_bda",
            &["-DKQ8", "-DKV_BDA"],
        ),
        (
            "attention_kv",
            "attention_kv_vq8_bda",
            &["-DVQ8", "-DKV_BDA"],
        ),
        (
            "attention_kv",
            "attention_kv_dyn_q8_bda",
            &["-DKQ8", "-DVQ8", "-DUSE_PARAMS", "-DKV_BDA"],
        ),
        ("qk_norm_rope", "qk_norm_rope", &[]),
        ("qk_norm_rope", "qk_norm_rope_dyn", &["-DUSE_PARAMS"]),
        ("qk_norm_rope", "qk_norm_rope_ff", &["-DFREQ_FACTORS"]),
        (
            "qk_norm_rope",
            "qk_norm_rope_dyn_ff",
            &["-DUSE_PARAMS", "-DFREQ_FACTORS"],
        ),
        // Interleaved q+g variant: reads query elements from strided buffer, eliminates per-head
        // CopyStrided dispatches (qwen35 attention layers).
        ("qk_norm_rope_interleaved", "qk_norm_rope_interleaved", &[]),
        (
            "qk_norm_rope_interleaved",
            "qk_norm_rope_interleaved_dyn",
            &["-DUSE_PARAMS"],
        ),
        // KV-cache u64/BDA store + dequant-read twins (#74, slice 3): `-DKV_BDA` writes the DEST cache
        // (store kernels) or reads the SOURCE cache (dequant) by 64-bit device address (kv_addr.glsl's
        // writable KvHalfW/KvF32W/KvU16W/KvByteW stores + the kv_word* reads) instead of the bound KV
        // SSBO — the KV descriptor slot disappears from the shader but stays BOUND at the recorder (the
        // dst at its write slot / the src at its read slot) purely so Recorder::sync's descriptor hazard
        // tracker keeps the store→read barrier a BDA access is invisible to. The store_f16 twin below
        // keeps its bound build because it ALSO writes a non-KV f16 scratch (the f32-cast dequant read,
        // Activations, no device address → bound); the store_q8/quant_kv/store_kv_dense/dequant_q8_f16/
        // dequant_kv_f16 kernels went BDA-only (slice 5): their base builds compile the (now
        // unconditional) BDA shader and their bound twins were deleted — always-KV-cache, no scratch
        // caller. See adapter.rs (the `device_addr()`-driven fork) and tests/kv_addr_parity.rs.
        ("store_f16", "store_f16_bda", &["-DKV_BDA"]),
        (
            "store_f16",
            "store_f16_dyn_bda",
            &["-DUSE_PARAMS", "-DKV_BDA"],
        ),
        ("rope", "rope_f16_bda", &["-DOUT_F16", "-DKV_BDA"]),
        (
            "rope",
            "rope_f16_dyn_bda",
            &["-DOUT_F16", "-DUSE_PARAMS", "-DKV_BDA"],
        ),
        ("qk_norm_rope", "qk_norm_rope_bda", &["-DKV_BDA"]),
        (
            "qk_norm_rope",
            "qk_norm_rope_dyn_bda",
            &["-DUSE_PARAMS", "-DKV_BDA"],
        ),
        (
            "qk_norm_rope",
            "qk_norm_rope_ff_bda",
            &["-DFREQ_FACTORS", "-DKV_BDA"],
        ),
        (
            "qk_norm_rope",
            "qk_norm_rope_dyn_ff_bda",
            &["-DUSE_PARAMS", "-DFREQ_FACTORS", "-DKV_BDA"],
        ),
        (
            "qk_norm_rope_interleaved",
            "qk_norm_rope_interleaved_bda",
            &["-DKV_BDA"],
        ),
        (
            "qk_norm_rope_interleaved",
            "qk_norm_rope_interleaved_dyn_bda",
            &["-DUSE_PARAMS", "-DKV_BDA"],
        ),
        // Native-block dequant GEMVs: one .spv per (quant format, residual) from one source.
        ("native_gemv", "native_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv", "native_bf16", &["-DFMT_BF16"]),
        (
            "native_gemv",
            "native_bf16_res",
            &["-DFMT_BF16", "-DUSE_RES"],
        ),
        (
            "native_gemv",
            "native_iq2xxs",
            &["-DFMT_IQ2XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv",
            "native_iq2xxs_res",
            &["-DFMT_IQ2XXS", "-DUSE_GRID", "-DUSE_RES"],
        ),
        (
            "native_gemv",
            "native_iq2xs",
            &["-DFMT_IQ2XS", "-DUSE_GRID"],
        ),
        (
            "native_gemv",
            "native_iq2xs_res",
            &["-DFMT_IQ2XS", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq2s", &["-DFMT_IQ2S", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq2s_res",
            &["-DFMT_IQ2S", "-DUSE_GRID", "-DUSE_RES"],
        ),
        (
            "native_gemv",
            "native_iq3xxs",
            &["-DFMT_IQ3XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv",
            "native_iq3xxs_res",
            &["-DFMT_IQ3XXS", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq3s", &["-DFMT_IQ3S", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq3s_res",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq1s", &["-DFMT_IQ1S", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq1s_res",
            &["-DFMT_IQ1S", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq1m", &["-DFMT_IQ1M", "-DUSE_GRID"]),
        (
            "native_gemv",
            "native_iq1m_res",
            &["-DFMT_IQ1M", "-DUSE_GRID", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq4nl", &["-DFMT_IQ4NL"]),
        (
            "native_gemv",
            "native_iq4nl_res",
            &["-DFMT_IQ4NL", "-DUSE_RES"],
        ),
        ("native_gemv", "native_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_gemv",
            "native_iq4xs_res",
            &["-DFMT_IQ4XS", "-DUSE_RES"],
        ),
        ("native_gemv", "native_mxfp4", &["-DFMT_MXFP4"]),
        (
            "native_gemv",
            "native_mxfp4_res",
            &["-DFMT_MXFP4", "-DUSE_RES"],
        ),
        ("native_gemv", "native_nvfp4", &["-DFMT_NVFP4"]),
        (
            "native_gemv",
            "native_nvfp4_res",
            &["-DFMT_NVFP4", "-DUSE_RES"],
        ),
        ("native_gemv", "native_tq1_0", &["-DFMT_TQ1_0"]),
        (
            "native_gemv",
            "native_tq1_0_res",
            &["-DFMT_TQ1_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_tq2_0", &["-DFMT_TQ2_0"]),
        (
            "native_gemv",
            "native_tq2_0_res",
            &["-DFMT_TQ2_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q2_0", &["-DFMT_Q2_0"]),
        (
            "native_gemv",
            "native_q2_0_res",
            &["-DFMT_Q2_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q4_0", &["-DFMT_Q4_0"]),
        (
            "native_gemv",
            "native_q4_0_res",
            &["-DFMT_Q4_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q4_1", &["-DFMT_Q4_1"]),
        (
            "native_gemv",
            "native_q4_1_res",
            &["-DFMT_Q4_1", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q5_0", &["-DFMT_Q5_0"]),
        (
            "native_gemv",
            "native_q5_0_res",
            &["-DFMT_Q5_0", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q5_1", &["-DFMT_Q5_1"]),
        (
            "native_gemv",
            "native_q5_1_res",
            &["-DFMT_Q5_1", "-DUSE_RES"],
        ),
        ("native_gemv", "native_q2k", &["-DFMT_Q2K"]),
        ("native_gemv", "native_q2k_res", &["-DFMT_Q2K", "-DUSE_RES"]),
        ("native_gemv", "native_q3k", &["-DFMT_Q3K"]),
        ("native_gemv", "native_q3k_res", &["-DFMT_Q3K", "-DUSE_RES"]),
        ("native_gemv", "native_q4k", &["-DFMT_Q4K"]),
        ("native_gemv", "native_q4k_res", &["-DFMT_Q4K", "-DUSE_RES"]),
        ("native_gemv", "native_q5k", &["-DFMT_Q5K"]),
        ("native_gemv", "native_q5k_res", &["-DFMT_Q5K", "-DUSE_RES"]),
        ("native_gemv", "native_q6k", &["-DFMT_Q6K"]),
        ("native_gemv", "native_q6k_res", &["-DFMT_Q6K", "-DUSE_RES"]),
        // Multi-output-row decode GEMV (RM rows/workgroup) — bit-identical to the RM=1 GEMV per
        // row, but RM× the in-flight weight streams per wave to saturate DRAM on the low-out_f
        // (out_f≈4096) projection GEMVs. K-quant dense/MoE-hot formats only.
        ("native_gemv_rm", "native_q4k_rm2", &["-DFMT_Q4K", "-DRM=2"]),
        (
            "native_gemv_rm",
            "native_q4k_rm2_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES"],
        ),
        ("native_gemv_rm", "native_q4k_rm4", &["-DFMT_Q4K", "-DRM=4"]),
        (
            "native_gemv_rm",
            "native_q4k_rm4_res",
            &["-DFMT_Q4K", "-DRM=4", "-DUSE_RES"],
        ),
        ("native_gemv_rm", "native_q6k_rm2", &["-DFMT_Q6K", "-DRM=2"]),
        (
            "native_gemv_rm",
            "native_q6k_rm2_res",
            &["-DFMT_Q6K", "-DRM=2", "-DUSE_RES"],
        ),
        ("native_gemv_rm", "native_q6k_rm4", &["-DFMT_Q6K", "-DRM=4"]),
        (
            "native_gemv_rm",
            "native_q6k_rm4_res",
            &["-DFMT_Q6K", "-DRM=4", "-DUSE_RES"],
        ),
        // ── Experimental RM kernel variants (env-gated, default OFF) ──────────────────────────
        // Subgroup shuffle tree-reduce (replaces shared-mem + 2 barriers per row)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_sg",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_SG"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_sg_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_SG"],
        ),
        // Double-buffered dequant (pre-load next dqblk during current dot product)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_dbuf",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_DBUF"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_dbuf_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_DBUF"],
        ),
        // 128-thread workgroup (2x memory requests in flight)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_wg128",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_WG128"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_wg128_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_WG128"],
        ),
        // Register-only reduce via subgroup ops (no shared memory at all)
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_reg",
            &["-DFMT_Q4K", "-DRM=2", "-DVARIANT_REG"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q4k_rm2_reg_res",
            &["-DFMT_Q4K", "-DRM=2", "-DUSE_RES", "-DVARIANT_REG"],
        ),
        // Q6K subgroup variant
        (
            "native_gemv_rm_v2",
            "native_q6k_rm2_sg",
            &["-DFMT_Q6K", "-DRM=2", "-DVARIANT_SG"],
        ),
        (
            "native_gemv_rm_v2",
            "native_q6k_rm2_sg_res",
            &["-DFMT_Q6K", "-DRM=2", "-DUSE_RES", "-DVARIANT_SG"],
        ),
        // Reassociation-tolerant subgroup + NUM_ROWS decode GEMV (wave32, subgroupAdd, no shared
        // reduce) for the latency-STARVED out_f≈2048-8192 Q6_K projections (ffn_down / o / attn_qkv).
        // NOT bit-identical to the tree GEMV (reordered accumulation); gated to that band only. Q6_K
        // ONLY — on Q4_K the tree/RM kernel already saturates and SG regressed at every measured shape
        // (decode_gemv_bw A/B), so no Q4_K SG build exists. NR ∈ {2,4,8} × {plain,res}.
        ("native_gemv_sg", "native_q6k_sg2", &["-DFMT_Q6K", "-DNR=2"]),
        (
            "native_gemv_sg",
            "native_q6k_sg2_res",
            &["-DFMT_Q6K", "-DNR=2", "-DUSE_RES"],
        ),
        ("native_gemv_sg", "native_q6k_sg4", &["-DFMT_Q6K", "-DNR=4"]),
        (
            "native_gemv_sg",
            "native_q6k_sg4_res",
            &["-DFMT_Q6K", "-DNR=4", "-DUSE_RES"],
        ),
        ("native_gemv_sg", "native_q6k_sg8", &["-DFMT_Q6K", "-DNR=8"]),
        (
            "native_gemv_sg",
            "native_q6k_sg8_res",
            &["-DFMT_Q6K", "-DNR=8", "-DUSE_RES"],
        ),
        // Multi-row GEMV (m = 2..8: spec verify / short suffix prefill) — mainstream dense
        // projection formats only; the rest fall back to the tiled GEMM.
        ("native_gemv_mrow", "native_mrow_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv_mrow", "native_mrow_bf16", &["-DFMT_BF16"]),
        ("native_gemv_mrow", "native_mrow_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemv_mrow", "native_mrow_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemv_mrow", "native_mrow_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemv_mrow", "native_mrow_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemv_mrow", "native_mrow_q2k", &["-DFMT_Q2K"]),
        ("native_gemv_mrow", "native_mrow_q3k", &["-DFMT_Q3K"]),
        ("native_gemv_mrow", "native_mrow_q4k", &["-DFMT_Q4K"]),
        ("native_gemv_mrow", "native_mrow_q5k", &["-DFMT_Q5K"]),
        ("native_gemv_mrow", "native_mrow_q6k", &["-DFMT_Q6K"]),
        ("native_gemv_mrow", "native_mrow_iq4nl", &["-DFMT_IQ4NL"]),
        ("native_gemv_mrow", "native_mrow_iq4xs", &["-DFMT_IQ4XS"]),
        ("native_gemv_mrow", "native_mrow_q2_0", &["-DFMT_Q2_0"]),
        (
            "native_gemv",
            "native_q8_0_res",
            &["-DFMT_Q8_0", "-DUSE_RES"],
        ),
        // Id-indexed native GEMVs for GPU-resident MoE decode (expert chosen from a GPU buffer): one
        // .spv per weight format experts can be stored in — the FULL dense-GEMV format set (plus
        // F16/F32 for float banks), so `moe_expert_dtype_ok` never rejects a dense-supported bank.
        ("native_gemv_id", "native_id_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv_id", "native_id_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemv_id", "native_id_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemv_id", "native_id_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemv_id", "native_id_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemv_id", "native_id_q2k", &["-DFMT_Q2K"]),
        ("native_gemv_id", "native_id_q3k", &["-DFMT_Q3K"]),
        ("native_gemv_id", "native_id_q4k", &["-DFMT_Q4K"]),
        ("native_gemv_id", "native_id_q5k", &["-DFMT_Q5K"]),
        ("native_gemv_id", "native_id_q6k", &["-DFMT_Q6K"]),
        // Codebook id-GEMV (native_decode.glsl's generic FMT_IQ4NL/FMT_IQ4XS dequant already
        // exists for the dense id-less GEMV; these are the id-indexed twins so GPU-resident MoE
        // decode can pick an IQ4_NL/IQ4_XS expert from a GPU buffer too — previously absent).
        ("native_gemv_id", "native_id_iq4nl", &["-DFMT_IQ4NL"]),
        ("native_gemv_id", "native_id_iq4xs", &["-DFMT_IQ4XS"]),
        // Full dense-parity id coverage (fp4 / ternary / grid i-quants / floats): every remaining
        // dtype the dense native GEMV decodes gets its id twin so `moe_expert_dtype_ok` accepts
        // everything dense accepts (field report: MXFP4_MOE expert banks used to be clean-rejected
        // at load). Grid formats add -DUSE_GRID exactly like their dense builds (the codebook LUTs
        // are compile-time consts from native_grids.glsl — no extra binding). F16/F32 serve FLOAT
        // expert banks: resident float banks bind as effective f16 (`bind_weight` converts), paged
        // float banks stage raw GGUF bytes — hence both variants exist.
        ("native_gemv_id", "native_id_mxfp4", &["-DFMT_MXFP4"]),
        ("native_gemv_id", "native_id_nvfp4", &["-DFMT_NVFP4"]),
        ("native_gemv_id", "native_id_tq1_0", &["-DFMT_TQ1_0"]),
        ("native_gemv_id", "native_id_tq2_0", &["-DFMT_TQ2_0"]),
        ("native_gemv_id", "native_id_q2_0", &["-DFMT_Q2_0"]),
        (
            "native_gemv_id",
            "native_id_iq2xxs",
            &["-DFMT_IQ2XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id",
            "native_id_iq2xs",
            &["-DFMT_IQ2XS", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id",
            "native_id_iq2s",
            &["-DFMT_IQ2S", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id",
            "native_id_iq3xxs",
            &["-DFMT_IQ3XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id",
            "native_id_iq3s",
            &["-DFMT_IQ3S", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id",
            "native_id_iq1s",
            &["-DFMT_IQ1S", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id",
            "native_id_iq1m",
            &["-DFMT_IQ1M", "-DUSE_GRID"],
        ),
        ("native_gemv_id", "native_id_bf16", &["-DFMT_BF16"]),
        ("native_gemv_id", "native_id_f16", &["-DFMT_F16"]),
        ("native_gemv_id", "native_id_f32", &["-DFMT_F32"]),
        // Paged twins (`infr_vulkan::pager::GpuPager`): one extra LUT-buffer hop, `slot =
        // lut[expert_id]` — see native_gemv_id.comp's `-DPAGED` doc comment. Same format coverage
        // as the resident-bank kernels above (the pager only ever holds native-block formats).
        (
            "native_gemv_id",
            "native_id_q8_0_paged",
            &["-DFMT_Q8_0", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q4_0_paged",
            &["-DFMT_Q4_0", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q4_1_paged",
            &["-DFMT_Q4_1", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q5_0_paged",
            &["-DFMT_Q5_0", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q5_1_paged",
            &["-DFMT_Q5_1", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q2k_paged",
            &["-DFMT_Q2K", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q3k_paged",
            &["-DFMT_Q3K", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q4k_paged",
            &["-DFMT_Q4K", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q5k_paged",
            &["-DFMT_Q5K", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q6k_paged",
            &["-DFMT_Q6K", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq4nl_paged",
            &["-DFMT_IQ4NL", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq4xs_paged",
            &["-DFMT_IQ4XS", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_mxfp4_paged",
            &["-DFMT_MXFP4", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_nvfp4_paged",
            &["-DFMT_NVFP4", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_tq1_0_paged",
            &["-DFMT_TQ1_0", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_tq2_0_paged",
            &["-DFMT_TQ2_0", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_q2_0_paged",
            &["-DFMT_Q2_0", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq2xxs_paged",
            &["-DFMT_IQ2XXS", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq2xs_paged",
            &["-DFMT_IQ2XS", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq2s_paged",
            &["-DFMT_IQ2S", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq3xxs_paged",
            &["-DFMT_IQ3XXS", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq3s_paged",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq1s_paged",
            &["-DFMT_IQ1S", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_iq1m_paged",
            &["-DFMT_IQ1M", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_bf16_paged",
            &["-DFMT_BF16", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_f16_paged",
            &["-DFMT_F16", "-DPAGED"],
        ),
        (
            "native_gemv_id",
            "native_id_f32_paged",
            &["-DFMT_F32", "-DPAGED"],
        ),
        // Multi-slot id GEMV: all n_used experts in one dispatch (concurrent, no inter-expert barrier).
        ("native_gemv_id_multi", "native_idm_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemv_id_multi", "native_idm_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemv_id_multi", "native_idm_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemv_id_multi", "native_idm_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemv_id_multi", "native_idm_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemv_id_multi", "native_idm_q2k", &["-DFMT_Q2K"]),
        ("native_gemv_id_multi", "native_idm_q3k", &["-DFMT_Q3K"]),
        ("native_gemv_id_multi", "native_idm_q4k", &["-DFMT_Q4K"]),
        ("native_gemv_id_multi", "native_idm_q5k", &["-DFMT_Q5K"]),
        ("native_gemv_id_multi", "native_idm_q6k", &["-DFMT_Q6K"]),
        ("native_gemv_id_multi", "native_idm_iq4nl", &["-DFMT_IQ4NL"]),
        ("native_gemv_id_multi", "native_idm_iq4xs", &["-DFMT_IQ4XS"]),
        // Full dense-parity idm coverage — same rationale/defines as the native_gemv_id set above.
        ("native_gemv_id_multi", "native_idm_mxfp4", &["-DFMT_MXFP4"]),
        ("native_gemv_id_multi", "native_idm_nvfp4", &["-DFMT_NVFP4"]),
        ("native_gemv_id_multi", "native_idm_tq1_0", &["-DFMT_TQ1_0"]),
        ("native_gemv_id_multi", "native_idm_tq2_0", &["-DFMT_TQ2_0"]),
        ("native_gemv_id_multi", "native_idm_q2_0", &["-DFMT_Q2_0"]),
        (
            "native_gemv_id_multi",
            "native_idm_iq2xxs",
            &["-DFMT_IQ2XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq2xs",
            &["-DFMT_IQ2XS", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq2s",
            &["-DFMT_IQ2S", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq3xxs",
            &["-DFMT_IQ3XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq3s",
            &["-DFMT_IQ3S", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq1s",
            &["-DFMT_IQ1S", "-DUSE_GRID"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq1m",
            &["-DFMT_IQ1M", "-DUSE_GRID"],
        ),
        ("native_gemv_id_multi", "native_idm_bf16", &["-DFMT_BF16"]),
        ("native_gemv_id_multi", "native_idm_f16", &["-DFMT_F16"]),
        ("native_gemv_id_multi", "native_idm_f32", &["-DFMT_F32"]),
        // Paged twins — same LUT hop as above, for the decode/small-m multi-expert dispatch.
        (
            "native_gemv_id_multi",
            "native_idm_q8_0_paged",
            &["-DFMT_Q8_0", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q4_0_paged",
            &["-DFMT_Q4_0", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q4_1_paged",
            &["-DFMT_Q4_1", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q5_0_paged",
            &["-DFMT_Q5_0", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q5_1_paged",
            &["-DFMT_Q5_1", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q2k_paged",
            &["-DFMT_Q2K", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q3k_paged",
            &["-DFMT_Q3K", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q4k_paged",
            &["-DFMT_Q4K", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q5k_paged",
            &["-DFMT_Q5K", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q6k_paged",
            &["-DFMT_Q6K", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq4nl_paged",
            &["-DFMT_IQ4NL", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq4xs_paged",
            &["-DFMT_IQ4XS", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_mxfp4_paged",
            &["-DFMT_MXFP4", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_nvfp4_paged",
            &["-DFMT_NVFP4", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_tq1_0_paged",
            &["-DFMT_TQ1_0", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_tq2_0_paged",
            &["-DFMT_TQ2_0", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_q2_0_paged",
            &["-DFMT_Q2_0", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq2xxs_paged",
            &["-DFMT_IQ2XXS", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq2xs_paged",
            &["-DFMT_IQ2XS", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq2s_paged",
            &["-DFMT_IQ2S", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq3xxs_paged",
            &["-DFMT_IQ3XXS", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq3s_paged",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq1s_paged",
            &["-DFMT_IQ1S", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_iq1m_paged",
            &["-DFMT_IQ1M", "-DUSE_GRID", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_bf16_paged",
            &["-DFMT_BF16", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_f16_paged",
            &["-DFMT_F16", "-DPAGED"],
        ),
        (
            "native_gemv_id_multi",
            "native_idm_f32_paged",
            &["-DFMT_F32", "-DPAGED"],
        ),
        // Reassociation-tolerant subgroup+NR variant of the multi-slot id GEMV (wave32, subgroupAdd,
        // no shared reduce) for the latency-STARVED Q6_K MoE expert down-projection (out_f≈2048 — the
        // largest SG win of any decode shape). NOT bit-identical (reordered accumulation); gated to
        // the Q6_K projection band in the recorder. Q6_K ONLY (mirrors the dense native_gemv_sg
        // discipline — Q4_K idm already saturates). NR ∈ {2,4,8}.
        (
            "native_gemv_id_multi_sg",
            "native_idm_q6k_sg2",
            &["-DFMT_Q6K", "-DNR=2"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q6k_sg4",
            &["-DFMT_Q6K", "-DNR=4"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q6k_sg8",
            &["-DFMT_Q6K", "-DNR=8"],
        ),
        // Q5_K SG id variant: the Qwen3.6-A3B UD-quant stores most expert down-projections as Q5_K
        // (not Q6_K) at the same out_f≈2048 down shape — the heavy K-quant decode still nets out on
        // wave32+subgroupAdd (A/B-confirmed a win; Q4_K stays on the tree). NR ∈ {2,4,8}.
        (
            "native_gemv_id_multi_sg",
            "native_idm_q5k_sg2",
            &["-DFMT_Q5K", "-DNR=2"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q5k_sg4",
            &["-DFMT_Q5K", "-DNR=4"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_q5k_sg8",
            &["-DFMT_Q5K", "-DNR=8"],
        ),
        // IQ3_S SG id variant: the Qwen3.6-35B-A3B UD-IQ3_S quant stores the expert DOWN projection
        // as IQ3_S at the same out_f≈2048 shape the Q5_K/Q6_K band was cut for, and the grid
        // codebook is the heaviest unpack of the set — NR rows/workgroup ALSO amortizes
        // grid_init()'s per-workgroup LDS staging, which the one-row-per-workgroup tree kernel pays
        // for every output row (A/B-confirmed a win: native_idm_iq3s 42.0 → 34.8ms). IQ2_S is
        // deliberately absent — its mirrored gate/up shape (out_f=512) LOSES badly on this tier;
        // see native_id_sg_choice's doc for the measured numbers. NR ∈ {2,4,8}.
        (
            "native_gemv_id_multi_sg",
            "native_idm_iq3s_sg2",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DNR=2"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_iq3s_sg4",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DNR=4"],
        ),
        (
            "native_gemv_id_multi_sg",
            "native_idm_iq3s_sg8",
            &["-DFMT_IQ3S", "-DUSE_GRID", "-DNR=8"],
        ),
        ("moe_accumulate", "moe_accumulate", &[]),
        ("moe_accumulate_scaled", "moe_accumulate_scaled", &[]),
        ("native_mmv_id_q4k", "native_mmv_id_q4k", &[]),
        // Int8 dp4a decode GEMV (m=1, NUM_ROWS=2): one .spv per (format, residual).
        ("native_mmv", "native_mmv_q4k", &["-DFMT_Q4K"]),
        (
            "native_mmv",
            "native_mmv_q4k_res",
            &["-DFMT_Q4K", "-DUSE_RES"],
        ),
        ("native_mmv", "native_mmv_q6k", &["-DFMT_Q6K"]),
        (
            "native_mmv",
            "native_mmv_q6k_res",
            &["-DFMT_Q6K", "-DUSE_RES"],
        ),
        // Multi-warp int8 dp4a decode GEMV (llama mul_mat_vec_q block, warp-per-row subgroupAdd).
        // WARPS ∈ {4,8} rows/block × {plain,res}. Gated by adapter.rs `mmv_mw_choice`
        // (default-on Intel, INFR_MMV_MW=1 opt-in elsewhere).
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w4",
            &["-DFMT_Q4K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w4_res",
            &["-DFMT_Q4K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w8",
            &["-DFMT_Q4K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w8_res",
            &["-DFMT_Q4K", "-DWARPS=8", "-DUSE_RES"],
        ),
        // Dispatch-shape sweep (Q4_K only, root cause #2 in README footnote 3): WARPS ∈ {1,2,16}
        // extends the {4,8} set above. WARPS=1 is llama.cpp's rm_kq_int=1 shape (one output row
        // per workgroup, single-subgroup reduce, no cross-warp anything) — the AMD non-GCN mmvq
        // config footnote 3 names as the untried lever. Gated through the same INFR_MMV_MW_WARPS
        // env escape as {4,8}; not part of the default policy set.
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w1",
            &["-DFMT_Q4K", "-DWARPS=1"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w1_res",
            &["-DFMT_Q4K", "-DWARPS=1", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w2",
            &["-DFMT_Q4K", "-DWARPS=2"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w2_res",
            &["-DFMT_Q4K", "-DWARPS=2", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w16",
            &["-DFMT_Q4K", "-DWARPS=16"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q4k_w16_res",
            &["-DFMT_Q4K", "-DWARPS=16", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w4",
            &["-DFMT_Q6K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w4_res",
            &["-DFMT_Q6K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w8",
            &["-DFMT_Q6K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q6k_w8_res",
            &["-DFMT_Q6K", "-DWARPS=8", "-DUSE_RES"],
        ),
        // Q2_K / Q3_K mmv_mw (Intel R3, llama.cpp's measured Intel MMVQ table: Xe2 always-MMVQ for
        // Q2/Q3/Q6_K). Q2_K adds a per-16 min term (dp4a vs packed ones); Q3_K is symmetric after
        // the −4 rebias. Same {plain,res} × WARPS∈{4,8} set as Q4K/Q6K above.
        (
            "native_mmv_mw",
            "native_mmv_mw_q2k_w4",
            &["-DFMT_Q2K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q2k_w4_res",
            &["-DFMT_Q2K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q2k_w8",
            &["-DFMT_Q2K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q2k_w8_res",
            &["-DFMT_Q2K", "-DWARPS=8", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q3k_w4",
            &["-DFMT_Q3K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q3k_w4_res",
            &["-DFMT_Q3K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q3k_w8",
            &["-DFMT_Q3K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q3k_w8_res",
            &["-DFMT_Q3K", "-DWARPS=8", "-DUSE_RES"],
        ),
        // Q5_K mmv_mw: NEW int8 decode arm (previously no int8 tier in either stream — see
        // adapter.rs's `mmv_int8_decode_dtypes` doc). Same {plain,res} × WARPS∈{4,8} set as
        // Q2_K/Q3_K; no dispatch-shape sweep (AMD-only measurement). `native_mmv_mw` is in
        // SG16_SOURCES below, so a `_sg16` twin gets auto-compiled too, but `gemm.rs`
        // deliberately leaves the sg16 match arms unwired — Q5_K isn't in Intel's decode-dtype
        // set, so nothing would ever request it; add the arms alongside an Intel measurement.
        (
            "native_mmv_mw",
            "native_mmv_mw_q5k_w4",
            &["-DFMT_Q5K", "-DWARPS=4"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q5k_w4_res",
            &["-DFMT_Q5K", "-DWARPS=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q5k_w8",
            &["-DFMT_Q5K", "-DWARPS=8"],
        ),
        (
            "native_mmv_mw",
            "native_mmv_mw_q5k_w8_res",
            &["-DFMT_Q5K", "-DWARPS=8", "-DUSE_RES"],
        ),
        // IQ4_XS: codebook-gather-then-dp4a (the 4-bit code indexes KV_IQ4NL -> int8 before the dot).
        ("native_mmv", "native_mmv_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_mmv",
            "native_mmv_iq4xs_res",
            &["-DFMT_IQ4XS", "-DUSE_RES"],
        ),
        // Multi-row int8 dp4a GEMV (m=2..8): weight sub-block unpacked once, dp4a per row.
        ("native_mmv_mrow", "native_mmv_mrow_q4k", &["-DFMT_Q4K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_m4",
            &["-DFMT_Q4K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_m4",
            &["-DFMT_Q6K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4_m4",
            &["-DFMT_Q4K", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4_m4",
            &["-DFMT_Q6K", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4",
            &["-DFMT_Q4K", "-DOUTS4"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q6k", &["-DFMT_Q6K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4",
            &["-DFMT_Q6K", "-DOUTS4"],
        ),
        // rows=1 DECODE builds (-DUSE_RES, always -DMRV=4 since decode's rows is always 1): the fix
        // for the mmv_mw/mrow reassociation gap (README footnote 3) — on AMD, decode (m=1) now
        // dispatches THIS shader, not native_mmv_mw.comp, for the dtypes that need decode/verify
        // bit-identity (see adapter.rs; Intel keeps mmv_mw, unchanged, untested here).
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_m4_res",
            &["-DFMT_Q4K", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4_m4_res",
            &["-DFMT_Q4K", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_m4_res",
            &["-DFMT_Q6K", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4_m4_res",
            &["-DFMT_Q6K", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        // IQ4_XS multi-row (m=2..8): codebook-gather-then-dp4a, Q4_K-style single-dp accumulation.
        ("native_mmv_mrow", "native_mmv_mrow_iq4xs", &["-DFMT_IQ4XS"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_m4",
            &["-DFMT_IQ4XS", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_o4",
            &["-DFMT_IQ4XS", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_o4_m4",
            &["-DFMT_IQ4XS", "-DOUTS4", "-DMRV=4"],
        ),
        // Q2_K / Q3_K multi-row (m=2..8): symmetric-decode partners for `mmv_mw_choice`'s Q2_K/Q3_K
        // decode tier (adapter.rs) — without these the MTP verify batch would keep the f32-exact
        // dequant mrow for a dtype whose m=1 decode already went int8, exactly the asymmetry that
        // broke Q5_K token-identity. Math ported from native_mmv_mw.comp's FMT_Q2K/FMT_Q3K.
        ("native_mmv_mrow", "native_mmv_mrow_q2k", &["-DFMT_Q2K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q2k_m4",
            &["-DFMT_Q2K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q2k_o4",
            &["-DFMT_Q2K", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q2k_o4_m4",
            &["-DFMT_Q2K", "-DOUTS4", "-DMRV=4"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q3k", &["-DFMT_Q3K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q3k_m4",
            &["-DFMT_Q3K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q3k_o4",
            &["-DFMT_Q3K", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q3k_o4_m4",
            &["-DFMT_Q3K", "-DOUTS4", "-DMRV=4"],
        ),
        // rows=1 DECODE builds (-DUSE_RES) — see the Q4_K/Q6_K -DUSE_RES block above; same
        // reasoning, needed here too since Q2_K/Q3_K's decode tier is policy-gated symmetric with
        // their mrow tier (`mmv_int8_decode_dtypes`/`mrow_int8_dtype_ok`).
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q2k_m4_res",
            &["-DFMT_Q2K", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q2k_o4_m4_res",
            &["-DFMT_Q2K", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q3k_m4_res",
            &["-DFMT_Q3K", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q3k_o4_m4_res",
            &["-DFMT_Q3K", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        // Q5_K multi-row: NEW int8 arm — previously no int8 tier in either stream. Gated by
        // adapter.rs `mrow_int8_dtype_ok` the same way as Q2_K/Q3_K (tied to the decode policy set,
        // never unconditional) so this dtype can never ship verify-int8/decode-f32-exact by default
        // — that exact split is the historical Q5_K MTP token-divergence bug. Post-unification this
        // same shader serves BOTH the m>=2 verify/prefill tier and (on AMD) the rows=1 decode tier,
        // so the -DUSE_RES twins below are required for the decode Linear+Add fusion.
        ("native_mmv_mrow", "native_mmv_mrow_q5k", &["-DFMT_Q5K"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5k_m4",
            &["-DFMT_Q5K", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5k_o4",
            &["-DFMT_Q5K", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5k_o4_m4",
            &["-DFMT_Q5K", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5k_m4_res",
            &["-DFMT_Q5K", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5k_o4_m4_res",
            &["-DFMT_Q5K", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        // rows 9..=16 tier (-DMRV=16, 2-output layout only): the MTP spec-verify batch when the
        // rollback window has a few committed rows on top of the n_max drafts — these previously
        // fell off the mrow tier onto the split-K coopmat tile at 2-4x the per-row cost (measured
        // 9B mtp128: sk_ag m=11 down-proj 148us vs mmvr m=7 67.5us for the same weight stream).
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_m16",
            &["-DFMT_Q4K", "-DMRV=16"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_m16",
            &["-DFMT_Q6K", "-DMRV=16"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4xs_m16",
            &["-DFMT_IQ4XS", "-DMRV=16"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q2k_m16",
            &["-DFMT_Q2K", "-DMRV=16"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q3k_m16",
            &["-DFMT_Q3K", "-DMRV=16"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5k_m16",
            &["-DFMT_Q5K", "-DMRV=16"],
        ),
        // FUSE_QUANT variants — inline the activation quantization into the GEMV kernel,
        // saving the separate quant_q8 dispatch. Opt-in behind INFR_MMV_FUSE_QUANT=1.
        // Decode only (m4=true, rows<=4); Q4_K/Q6_K only (the two dtypes on the decode
        // int8 default set that use this kernel). m4+o4+res combos; no non-m4 or non-res
        // variants needed (decode always sets m4=true, and res is for the fused-add peephole).
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_m4_fuseq",
            &["-DFMT_Q4K", "-DMRV=4", "-DFUSE_QUANT"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4_m4_fuseq",
            &["-DFMT_Q4K", "-DOUTS4", "-DMRV=4", "-DFUSE_QUANT"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_m4_res_fuseq",
            &["-DFMT_Q4K", "-DMRV=4", "-DFUSE_QUANT", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4k_o4_m4_res_fuseq",
            &[
                "-DFMT_Q4K",
                "-DOUTS4",
                "-DMRV=4",
                "-DFUSE_QUANT",
                "-DUSE_RES",
            ],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_m4_fuseq",
            &["-DFMT_Q6K", "-DMRV=4", "-DFUSE_QUANT"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4_m4_fuseq",
            &["-DFMT_Q6K", "-DOUTS4", "-DMRV=4", "-DFUSE_QUANT"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_m4_res_fuseq",
            &["-DFMT_Q6K", "-DMRV=4", "-DFUSE_QUANT", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q6k_o4_m4_res_fuseq",
            &[
                "-DFMT_Q6K",
                "-DOUTS4",
                "-DMRV=4",
                "-DFUSE_QUANT",
                "-DUSE_RES",
            ],
        ),
        // LEGACY 32-BLOCK int8 mrow tier — Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/IQ4_NL. The dp4a *GEMM*
        // (native_gemm_mmq_*) has covered these for a while; the dp4a *GEMV* did not, so decode and
        // small-m prefill fell to the f32 dequant path for every non-k-quant integer file. Their
        // int8 packing is a port of the already-shipped mmq unpack (same nibble/qh/codebook/min
        // conventions), word-parallelized — see native_mmv_mrow.comp's legacy-format section. Full
        // {plain,o4} x {MR=8,MR=4} + the rows=1 -DUSE_RES decode twins + the rows 9..=16 MR=16 tier,
        // i.e. the same 7-build matrix Q4_K/Q5_K carry (IQ4_XS's missing _res builds are a historical
        // gap, not a pattern to copy: these dtypes need the res twin to be decode-A/B-able at all).
        ("native_mmv_mrow", "native_mmv_mrow_q8_0", &["-DFMT_Q8_0"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q8_0_m4",
            &["-DFMT_Q8_0", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q8_0_o4",
            &["-DFMT_Q8_0", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q8_0_o4_m4",
            &["-DFMT_Q8_0", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q8_0_m4_res",
            &["-DFMT_Q8_0", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q8_0_o4_m4_res",
            &["-DFMT_Q8_0", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q8_0_m16",
            &["-DFMT_Q8_0", "-DMRV=16"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q4_0", &["-DFMT_Q4_0"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_0_m4",
            &["-DFMT_Q4_0", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_0_o4",
            &["-DFMT_Q4_0", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_0_o4_m4",
            &["-DFMT_Q4_0", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_0_m4_res",
            &["-DFMT_Q4_0", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_0_o4_m4_res",
            &["-DFMT_Q4_0", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_0_m16",
            &["-DFMT_Q4_0", "-DMRV=16"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q5_0", &["-DFMT_Q5_0"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_0_m4",
            &["-DFMT_Q5_0", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_0_o4",
            &["-DFMT_Q5_0", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_0_o4_m4",
            &["-DFMT_Q5_0", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_0_m4_res",
            &["-DFMT_Q5_0", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_0_o4_m4_res",
            &["-DFMT_Q5_0", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_0_m16",
            &["-DFMT_Q5_0", "-DMRV=16"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q4_1", &["-DFMT_Q4_1"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_1_m4",
            &["-DFMT_Q4_1", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_1_o4",
            &["-DFMT_Q4_1", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_1_o4_m4",
            &["-DFMT_Q4_1", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_1_m4_res",
            &["-DFMT_Q4_1", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_1_o4_m4_res",
            &["-DFMT_Q4_1", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q4_1_m16",
            &["-DFMT_Q4_1", "-DMRV=16"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_q5_1", &["-DFMT_Q5_1"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_1_m4",
            &["-DFMT_Q5_1", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_1_o4",
            &["-DFMT_Q5_1", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_1_o4_m4",
            &["-DFMT_Q5_1", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_1_m4_res",
            &["-DFMT_Q5_1", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_1_o4_m4_res",
            &["-DFMT_Q5_1", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_q5_1_m16",
            &["-DFMT_Q5_1", "-DMRV=16"],
        ),
        ("native_mmv_mrow", "native_mmv_mrow_iq4nl", &["-DFMT_IQ4NL"]),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4nl_m4",
            &["-DFMT_IQ4NL", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4nl_o4",
            &["-DFMT_IQ4NL", "-DOUTS4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4nl_o4_m4",
            &["-DFMT_IQ4NL", "-DOUTS4", "-DMRV=4"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4nl_m4_res",
            &["-DFMT_IQ4NL", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4nl_o4_m4_res",
            &["-DFMT_IQ4NL", "-DOUTS4", "-DMRV=4", "-DUSE_RES"],
        ),
        (
            "native_mmv_mrow",
            "native_mmv_mrow_iq4nl_m16",
            &["-DFMT_IQ4NL", "-DMRV=16"],
        ),
        ("native_gemm_mmq_q4k", "native_gemm_mmq_q4k", &[]),
        ("native_gemm_mmq_q6k", "native_gemm_mmq_q6k", &[]),
        // DENSE (non-expert-grid) builds of the remaining mmq dtypes — the non-coopmat prefill
        // GEMM tier (adapter.rs `nc_mmq`): on a device without a usable 16x16x16 f16 coopmat
        // (Intel Arc/ANV) but WITH packed int8 dot, dense prefill GEMMs of every
        // `infr_core::tensor::MOE_MMQ_DTYPES` member route through these instead of the per-row
        // scalar GEMVs. Same shader sources as the `_xp` expert-grid builds below — only the
        // grid mapping differs (base compile, no defines), so the dp4a math is the already
        // parity-proven code. Q4_K/Q6_K dense builds pre-existed above (the coopmat tier's
        // Q4_K-mmq arm); these 12 complete the set.
        ("native_gemm_mmq_q8_0", "native_gemm_mmq_q8_0", &[]),
        ("native_gemm_mmq_q5_0", "native_gemm_mmq_q5_0", &[]),
        ("native_gemm_mmq_q5k", "native_gemm_mmq_q5k", &[]),
        ("native_gemm_mmq_q5_1", "native_gemm_mmq_q5_1", &[]),
        ("native_gemm_mmq_q2_k", "native_gemm_mmq_q2_k", &[]),
        ("native_gemm_mmq_q3_k", "native_gemm_mmq_q3_k", &[]),
        ("native_gemm_mmq_q4_0", "native_gemm_mmq_q4_0", &[]),
        ("native_gemm_mmq_q4_1", "native_gemm_mmq_q4_1", &[]),
        ("native_gemm_mmq_iq4_nl", "native_gemm_mmq_iq4_nl", &[]),
        ("native_gemm_mmq_iq4_xs", "native_gemm_mmq_iq4_xs", &[]),
        ("native_gemm_mmq_mxfp4", "native_gemm_mmq_mxfp4", &[]),
        ("native_gemm_mmq_nvfp4", "native_gemm_mmq_nvfp4", &[]),
        // IQ2_S / IQ3_S grid-codebook mmq (Qwen3.6-35B-A3B-UD-IQ3_S's expert pair: IQ2_S gate/up,
        // IQ3_S down). The grid LUT is staged into shared memory via native_grids.glsl's
        // grid_init() — the shared-staging fix that invalidated the original grid-mmq exclusion
        // (see the shaders' doc comments + MOE_MMQ_DTYPES's EXCLUSIONS doc).
        ("native_gemm_mmq_iq2_s", "native_gemm_mmq_iq2_s", &[]),
        ("native_gemm_mmq_iq3_s", "native_gemm_mmq_iq3_s", &[]),
        // Q2_0 (Bonsai ternary): symmetric small-int — codes-1 = {-1,0,+1,+2} feed dp4a directly
        // (IQ4_NL's treatment minus the codebook; see the shader's doc comment).
        ("native_gemm_mmq_q2_0", "native_gemm_mmq_q2_0", &[]),
        // Non-coopmat float-weight prefill GEMM ("fma-warp" tier, see native_gemm_fma.comp): the
        // shared-memory fma warptile for f16/bf16/f32 weights on devices without a usable f16
        // coopmat (adapter.rs `nc_fma`). No subgroup ops, no f16 extensions — dispatchable on
        // any device the backend accepts.
        ("native_gemm_fma", "native_gemm_fma_f16", &["-DFMT_F16"]),
        ("native_gemm_fma", "native_gemm_fma_bf16", &["-DFMT_BF16"]),
        ("native_gemm_fma", "native_gemm_fma_f32", &["-DFMT_F32"]),
        // Non-coopmat shared-memory fma flash-attention prefill (adapter.rs `nc_fa_ok`, see
        // attn_nc_fa.comp): the attention companion of the fma GEMM tier above. No subgroup ops.
        // One build per shared-Os ceiling: hd<=128 (37.6 KB), hd<=256 (54.0 KB), hd<=512 (BM=16,
        // 47.6 KB) — the recorder picks the smallest that fits the layer's head dim.
        ("attn_nc_fa", "attn_nc_fa_hd128", &[]),
        ("attn_nc_fa", "attn_nc_fa_hd256", &["-DHD_MAX=256"]),
        ("attn_nc_fa", "attn_nc_fa_hd512", &["-DHD_MAX=512"]),
        // int8 coopmat (WMMA) prefill GEMM, Q8_0 only — measurement kernel gated behind
        // INFR_I8_COOPMAT=1 (see adapter.rs / docs in the .comp file). Default-off; correctness
        // validated against native_gemm_mmq_q8_0/native_gemm_warp_q8_0.
        ("native_gemm_i8cm_q8_0", "native_gemm_i8cm_q8_0", &[]),
        // "Idea 2" measurement variant: whole-row (block-invariant) activation scale instead of
        // per-32-block, gated behind INFR_I8_ROW_SCALE=1 (see native_gemm_i8cm_q8_0.comp #ifdef
        // ROW_SCALE + quant_q8_row.comp).
        (
            "native_gemm_i8cm_q8_0",
            "native_gemm_i8cm_q8_0_rowscale",
            &["-DROW_SCALE"],
        ),
        // Row-wise activation quant for the int8-coopmat GEMM's "Idea 2" measurement (whole-K
        // scale instead of per-32-block; see quant_q8_row.comp). Gated by INFR_I8_ROW_SCALE=1.
        ("quant_q8_row", "quant_q8_row", &[]),
        // fp8 (E4M3) coopmat prefill GEMM, Q8_0 only — gated behind INFR_F8_COOPMAT=1 +
        // caps.f8_coopmat (RDNA4 native fp8 WMMA; see adapter.rs `f8cm_ok` / the .comp file's
        // design doc). Default-off; correctness UNVALIDATED on this box (no fp8 coopmat hardware
        // here — compile-checked only), pending an RDNA4 run. Same 256-thread/8-warp warptile as
        // native_gemm_warp (BM=64xBN=256 wide); -DNARROW_N mirrors native_gemm_warp's n%128
        // occupancy-fix variant (BN=128/BK=64).
        ("native_gemm_f8cm_q8_0", "native_gemm_f8cm_q8_0", &[]),
        (
            "native_gemm_f8cm_q8_0",
            "native_gemm_f8cm_q8_0_n128",
            &["-DNARROW_N"],
        ),
        // Row-wise activation quant for the fp8-coopmat GEMM (whole-K amax -> scale = amax/448,
        // range-scales activations into E4M3 before the cast; see quant_f8_row.comp). Gated by
        // the same INFR_F8_COOPMAT=1 + caps.f8_coopmat as native_gemm_f8cm_q8_0 above.
        ("quant_f8_row", "quant_f8_row", &[]),
        // -DPREPACK measurement variant: reads a pre-packed E4M3 weight buffer directly (no
        // in-shader Q8_0 dequant) — tests whether removing the dqblk bottleneck lets fp8 beat f16
        // (see native_gemm_f8cm_q8_0.comp header). Gated behind INFR_F8_COOPMAT=1 +
        // INFR_F8_PREPACK=1 (adapter.rs `f8cm_ok`'s INFR_F8_PREPACK arm). Default path (unset) is
        // completely unaffected — these are new, additive SPIR-V entries.
        (
            "native_gemm_f8cm_q8_0",
            "native_gemm_f8cm_q8_0_prepack",
            &["-DPREPACK"],
        ),
        (
            "native_gemm_f8cm_q8_0",
            "native_gemm_f8cm_q8_0_prepack_n128",
            &["-DPREPACK", "-DNARROW_N"],
        ),
        // Bakes each Q8_0 32-block's scale into an E4M3 output (decode-once via dqblk), producing
        // the pre-packed weight buffer the PREPACK GEMM variants above read directly. Gated by the
        // same INFR_F8_COOPMAT=1 + INFR_F8_PREPACK=1.
        ("repack_q8_to_f8", "repack_q8_to_f8", &[]),
        (
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            // BN=128 wide-N twin of the DEFAULT (BM=64) expert tile — the big-tile sibling of the
            // `_xp32w` small-tile variant below. At the mid rows/expert band (qwen3-30B-A3B:
            // 128 experts × top-8 ⇒ ~32 rows/expert at pp512) BM=64 is already the right row tile
            // (one row-tile per expert ⇒ each expert's weight bank is staged exactly ONCE, the
            // floor); the remaining staging cost is the ACTIVATION tile, re-read once per N-tile.
            // Doubling BN to 128 halves that As traffic and halves the workgroup count, without
            // adding a second row tile (which is what makes `_xp32*` lose here — it re-reads the
            // much larger weight bank). THREADS = (64/4)·(128/4) = 512. Needs n%128.
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xp128",
            &["-DEXPERT_GRID", "-DBN_TILE=128u"],
        ),
        (
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            // BN=128 wide-N twin of the default BM=64 expert tile — see q4k_xp128 above.
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xp128",
            &["-DEXPERT_GRID", "-DBN_TILE=128u"],
        ),
        (
            "native_gemm_mmq_q8_0",
            "native_gemm_mmq_q8_0_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q5_0",
            "native_gemm_mmq_q5_0_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q5k",
            "native_gemm_mmq_q5k_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q5_1",
            "native_gemm_mmq_q5_1_xp",
            &["-DEXPERT_GRID"],
        ),
        // Llama-4-Scout's shipped MoE expert dtypes (gate/up=Q2_K, down=Q3_K) — see the two
        // shaders' doc comments for the block layout / sact conventions.
        (
            "native_gemm_mmq_q2_k",
            "native_gemm_mmq_q2_k_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q3_k",
            "native_gemm_mmq_q3_k_xp",
            &["-DEXPERT_GRID"],
        ),
        // Q4_0 (symmetric trivial family member) / Q4_1 (min-carrying, Q5_1 minus the highbit) /
        // IQ4_NL (codebook, 32-elem block) / IQ4_XS (codebook, 256-elem superblock) — no shipped
        // MoE GGUF in the audited cache uses Q4_0/Q4_1 for expert banks, but unsloth's UD quants
        // mix IQ4_XS into most of Qwen3.6-35B-A3B's gate/up banks (see the shader doc comments).
        (
            "native_gemm_mmq_q4_0",
            "native_gemm_mmq_q4_0_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_q4_1",
            "native_gemm_mmq_q4_1_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_iq4_nl",
            "native_gemm_mmq_iq4_nl_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_iq4_xs",
            "native_gemm_mmq_iq4_xs_xp",
            &["-DEXPERT_GRID"],
        ),
        // Q2_0 (Bonsai ternary): symmetric trivial member like Q4_0 — no shipped MoE GGUF uses it
        // for expert banks (Bonsai models are dense), synthetic parity only.
        (
            "native_gemm_mmq_q2_0",
            "native_gemm_mmq_q2_0_xp",
            &["-DEXPERT_GRID"],
        ),
        // IQ2_S (gate/up) / IQ3_S (down) — the Qwen3.6-35B-A3B-UD-IQ3_S expert pair.
        (
            "native_gemm_mmq_iq2_s",
            "native_gemm_mmq_iq2_s_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_iq3_s",
            "native_gemm_mmq_iq3_s_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_iq2_s",
            "native_gemm_mmq_iq2_s_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq3_s",
            "native_gemm_mmq_iq3_s_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq2_s",
            "native_gemm_mmq_iq2_s_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_iq3_s",
            "native_gemm_mmq_iq3_s_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_iq2_s",
            "native_gemm_mmq_iq2_s_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq3_s",
            "native_gemm_mmq_iq3_s_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        // BM=32 row-tile variants of the expert-grid GEMM (see matmul_mmq_experts' `n_used` doc):
        // at small rows-per-expert (Qwen3.6-MoE's 256-expert pool averages ~16/expert at pp512)
        // the default BM=64 tile is ~75% masked waste — a BM=32 tile halves that. Selected
        // per-dispatch by the recorder based on the caller's actual avg rows/expert; qwen3-30B-A3B
        // (128 experts, ~32/expert) still gets BM=64 by default at pp512, so this is additive, not
        // a blanket swap.
        (
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            // BN=128 wide-N twin of the small tile (Q4_K/Q6_K only): halves the workgroup
            // count and the per-output As staging at the shallow-k (k=512-768) 256-expert
            // down shapes. Selected by matmul_mmq_experts when n%128==0 (Ornith-35B pp512
            // +6.5%, Qwen3.6-35B +4.5%, measured interleaved); n%128!=0 keeps the BN=64 tile.
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xp32w",
            &["-DEXPERT_GRID", "-DBM_TILE=32u", "-DBN_TILE=128u"],
        ),
        (
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            // BN=128 wide-N twin of the small tile (Q4_K/Q6_K only): halves the workgroup
            // count and the per-output As staging at the shallow-k (k=512-768) 256-expert
            // down shapes. Selected by matmul_mmq_experts when n%128==0 (Ornith-35B pp512
            // +6.5%, Qwen3.6-35B +4.5%, measured interleaved); n%128!=0 keeps the BN=64 tile.
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xp32w",
            &["-DEXPERT_GRID", "-DBM_TILE=32u", "-DBN_TILE=128u"],
        ),
        (
            "native_gemm_mmq_q8_0",
            "native_gemm_mmq_q8_0_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5_0",
            "native_gemm_mmq_q5_0_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5k",
            "native_gemm_mmq_q5k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5_1",
            "native_gemm_mmq_q5_1_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q2_k",
            "native_gemm_mmq_q2_k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q3_k",
            "native_gemm_mmq_q3_k_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q4_0",
            "native_gemm_mmq_q4_0_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q4_1",
            "native_gemm_mmq_q4_1_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq4_nl",
            "native_gemm_mmq_iq4_nl_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq4_xs",
            "native_gemm_mmq_iq4_xs_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q2_0",
            "native_gemm_mmq_q2_0_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        // PAGED expert-grid variants (Scout's batched prefill through the GpuPager arena+LUT —
        // see the shaders' PAGED doc): every dtype the batched-MoE gate (`mmq_ok`) covers, now
        // that `paged_mmq_ok` mirrors it in full (see adapter.rs's drift-guard doc).
        (
            "native_gemm_mmq_q2_k",
            "native_gemm_mmq_q2_k_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q3_k",
            "native_gemm_mmq_q3_k_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q2_k",
            "native_gemm_mmq_q2_k_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q3_k",
            "native_gemm_mmq_q3_k_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q4_0",
            "native_gemm_mmq_q4_0_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q4_0",
            "native_gemm_mmq_q4_0_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q2_0",
            "native_gemm_mmq_q2_0_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q2_0",
            "native_gemm_mmq_q2_0_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q4_1",
            "native_gemm_mmq_q4_1_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q4_1",
            "native_gemm_mmq_q4_1_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq4_nl",
            "native_gemm_mmq_iq4_nl_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_iq4_nl",
            "native_gemm_mmq_iq4_nl_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_iq4_xs",
            "native_gemm_mmq_iq4_xs_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_iq4_xs",
            "native_gemm_mmq_iq4_xs_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q8_0",
            "native_gemm_mmq_q8_0_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q8_0",
            "native_gemm_mmq_q8_0_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5_0",
            "native_gemm_mmq_q5_0_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q5_0",
            "native_gemm_mmq_q5_0_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5_1",
            "native_gemm_mmq_q5_1_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q5_1",
            "native_gemm_mmq_q5_1_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q4k",
            "native_gemm_mmq_q4k_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q5k",
            "native_gemm_mmq_q5k_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q5k",
            "native_gemm_mmq_q5k_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_q6k",
            "native_gemm_mmq_q6k_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        // MXFP4 / NVFP4 batched expert GEMMs (the MXFP4_MOE quant family): both share the signed
        // E2M1 codebook (kvalues_mxfp4), so they get the IQ4_NL treatment — codebook LUT → signed
        // dp4a int path, no sact. NVFP4's per-16 UE4M3 sub-block scales split each 32-block's
        // dp4a accumulation into two halves (see the shader doc). Same _xp/_xp32/_xpg/_xpg32
        // variant set as every other mmq dtype.
        (
            "native_gemm_mmq_mxfp4",
            "native_gemm_mmq_mxfp4_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_mxfp4",
            "native_gemm_mmq_mxfp4_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_mxfp4",
            "native_gemm_mmq_mxfp4_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_mxfp4",
            "native_gemm_mmq_mxfp4_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_nvfp4",
            "native_gemm_mmq_nvfp4_xp",
            &["-DEXPERT_GRID"],
        ),
        (
            "native_gemm_mmq_nvfp4",
            "native_gemm_mmq_nvfp4_xp32",
            &["-DEXPERT_GRID", "-DBM_TILE=32u"],
        ),
        (
            "native_gemm_mmq_nvfp4",
            "native_gemm_mmq_nvfp4_xpg",
            &["-DEXPERT_GRID", "-DPAGED"],
        ),
        (
            "native_gemm_mmq_nvfp4",
            "native_gemm_mmq_nvfp4_xpg32",
            &["-DEXPERT_GRID", "-DPAGED", "-DBM_TILE=32u"],
        ),
        ("quant_q8", "quant_q8_gather", &["-DGATHER"]),
        ("moe_scatter_reduce", "moe_scatter_reduce", &[]),
        ("moe_topk", "moe_topk", &[]),
        // Embedding-row gather+dequant (Op::EmbedGather): one .spv per table format.
        ("embed_gather", "embed_gather_q8_0", &["-DFMT_Q8_0"]),
        ("embed_gather", "embed_gather_bf16", &["-DFMT_BF16"]),
        ("embed_gather", "embed_gather_f16", &["-DFMT_F16"]),
        ("embed_gather", "embed_gather_q4_0", &["-DFMT_Q4_0"]),
        ("embed_gather", "embed_gather_q4_1", &["-DFMT_Q4_1"]),
        ("embed_gather", "embed_gather_q5_0", &["-DFMT_Q5_0"]),
        ("embed_gather", "embed_gather_q5_1", &["-DFMT_Q5_1"]),
        ("embed_gather", "embed_gather_q2k", &["-DFMT_Q2K"]),
        ("embed_gather", "embed_gather_q3k", &["-DFMT_Q3K"]),
        ("embed_gather", "embed_gather_q4k", &["-DFMT_Q4K"]),
        ("embed_gather", "embed_gather_q5k", &["-DFMT_Q5K"]),
        ("embed_gather", "embed_gather_q6k", &["-DFMT_Q6K"]),
        ("embed_gather", "embed_gather_iq4nl", &["-DFMT_IQ4NL"]),
        ("embed_gather", "embed_gather_iq4xs", &["-DFMT_IQ4XS"]),
        // Q2_0 (Bonsai ternary): the shipped Bonsai GGUFs quantize token_embd.weight itself as
        // Q2_0, so the gather kernel needs the format or prefill falls back to host embedding.
        ("embed_gather", "embed_gather_q2_0", &["-DFMT_Q2_0"]),
        // Chained-decode id ring log (ring[pos & 63] = sampled id) — see id_log.comp.
        ("id_log", "id_log", &[]),
        // Device-side decode-replay params advance ([pos, kv_len] += 1) — see params_advance.comp.
        ("params_advance", "params_advance", &[]),
        // Two-stage vocab-scale stochastic sampler (Op::Sample): per-slice top-k candidates,
        // then select+softmax+nucleus+CDF over the union.
        ("sample_topk", "sample_topk_part", &[]),
        ("sample_topk", "sample_topk", &["-DPASS2"]),
        // Chained-decode variant: stage 2 reads `u` from a 64-slot ring keyed by the self-advancing
        // `params[0]` instead of a 1-float buffer — see sample_topk.comp's CHAIN doc.
        ("sample_topk", "sample_topk_chain", &["-DPASS2", "-DCHAIN"]),
        // Two-stage greedy argmax (Op::Argmax): slice partials, then a one-workgroup reduce.
        ("argmax", "argmax_part", &[]),
        ("argmax", "argmax", &["-DPASS2"]),
        // Two-stage fused argmax+top1-prob (Op::ArgmaxProb, MTP draft-loop accept): same shape as
        // argmax, plus an online-softmax sum_exp carried alongside the (max, idx) reduction.
        ("argmax_prob", "argmax_prob_part", &[]),
        ("argmax_prob", "argmax_prob", &["-DPASS2"]),
        ("moe_sample", "moe_sample", &[]),
        // DiffusionGemma perf slice 3 (docs/diffusion-gemma.md): fused per-row entropy-bound
        // sampler reduction — argmax/entropy/CDF-sample over [rows, vocab] logits on-GPU.
        ("dg_eb_sample", "dg_eb_sample", &[]),
        ("moe_bucket_count", "moe_bucket_count", &[]),
        ("moe_bucket_scan", "moe_bucket_scan", &[]),
        ("moe_bucket_scatter", "moe_bucket_scatter", &[]),
        (
            "moe_bucket_scatter",
            "moe_bucket_scatter_scaled",
            &["-DDSCALE"],
        ),
        ("add_scaled_id", "add_scaled_id", &[]),
        // llama4 weight-before-FFN: in-place per-row scale of gate/up outputs by the routing
        // weight (see Op::MoeFfn's `weight_before`).
        ("moe_weight_scale", "moe_weight_scale", &[]),
        // Expert-parallel (multi-GPU EP) band remap: rewrite router-selected GLOBAL expert ids into
        // this rank's LOCAL expert-shard indices (out-of-band → id 0, weight 0). See Op::MoeFfn's
        // `ep_band`.
        ("moe_ep_band_remap", "moe_ep_band_remap", &[]),
        // Native-block prefill GEMMs: one .spv per quant format (coopmat tiled, no residual).
        ("native_gemm_warp", "native_gemm_warp_q4k", &["-DFMT_Q4K"]),
        ("native_gemm_warp", "native_gemm_warp_q6k", &["-DFMT_Q6K"]),
        ("native_gemm_warp", "native_gemm_warp_q8_0", &["-DFMT_Q8_0"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128",
            &["-DFMT_Q4K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128",
            &["-DFMT_Q6K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128",
            &["-DFMT_Q8_0", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_sk",
            &["-DFMT_Q4K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_sk",
            &["-DFMT_Q6K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_sk",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            // BK64W (double k-stage, half the barriers) measured +14% on the qwen35-4B wide
            // prefill GEMMs (pp512 3811→4330) — Q4K's cheap dqblk leaves the tile barrier-bound.
            // Q5K/Q6K/Q8_0 measured a NET LOSS with it (all-formats build 4330→4000: their
            // heavier decoders collide with the doubled stage), so Q4K only.
            "native_gemm_warp",
            "native_gemm_warp_q4k_ag",
            &["-DFMT_Q4K", "-DA_GLOBAL", "-DBK64W"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        // BK=32 NARROW_N measurement variant — halves per-stage shared memory
        // (10 KB vs 18 KB) and doubles barriers. Hypothesized higher occupancy.
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag_bk32",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL", "-DBK32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_sk_ag",
            &["-DFMT_Q4K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_ag",
            &["-DFMT_Q6K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_ag",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_sk_ag",
            &["-DFMT_Q6K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_ag",
            &["-DFMT_Q8_0", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_ag",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_sk_ag",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        // F16-weight SPLIT-K warptile (DG slice-7): the SC soft-embedding GEMM (m=canvas,
        // k=vocab=262144, n=ne=2816) sat on the legacy `gemm_proj` BN=64 tile at ~6.5 TFLOPS —
        // 4x slower than llama.cpp's same-shape f16 GEMM. Only the sk variant is instantiated:
        // the adapter routes F16 here ONLY when the narrow-grid split-K policy fires (deep-k,
        // underfilled grid); every other f16 GEMM keeps its existing `matmul_proj` route.
        (
            "native_gemm_warp",
            "native_gemm_warp_f16_sk",
            &["-DFMT_F16", "-DNARROW_N", "-DSPLIT_K"],
        ),
        ("native_gemm_warp", "native_gemm_warp_bf16", &["-DFMT_BF16"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_bf16_n128",
            &["-DFMT_BF16", "-DNARROW_N"],
        ),
        // NATIVE bf16 cooperative-matrix (WMMA) variant of the SAME production warptile above —
        // gated behind INFR_BF16_COOPMAT=1 + caps.bf16_coopmat (RDNA4 native bf16 WMMA; see
        // adapter.rs `bf16cm_ok` / native_gemm_warp.comp's BF16CM doc). -DBF16CM swaps ONLY the
        // coopmat A/B operand type (float16_t -> bfloat16_t, dropping the f16 clamp on FMT_BF16
        // weights) — every other structural choice (staging, tiling, dqblk, epilogue) is the
        // IDENTICAL tuned production kernel `native_gemm_warp_bf16` uses, so this should match its
        // speed while keeping bf16's full exponent range. Replaces the old standalone
        // native_gemm_bf16cm.comp measurement kernel (retired: this variant subsumes it).
        // Default-off; correctness UNVALIDATED on this box (no bf16 coopmat hardware here —
        // compile-checked only), pending an RDNA4 run. No A_GLOBAL variant: the existing
        // `native_gemm_warp_bf16` f16-clamp path doesn't use A_GLOBAL either (Bf16 isn't in
        // `native_gemm_warp_ag_build_spv`'s dtype match), so there's nothing to mirror there.
        (
            "native_gemm_warp",
            "native_gemm_warp_bf16cm",
            &["-DFMT_BF16", "-DBF16CM"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_bf16cm_n128",
            &["-DFMT_BF16", "-DBF16CM", "-DNARROW_N"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q3k", &["-DFMT_Q3K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_n128",
            &["-DFMT_Q3K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_sk",
            &["-DFMT_Q3K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            // A_GLOBAL family (occupancy 2→3 wgs/WGP) + split-K (grid-fill for narrow-n
            // down/o/kv projections). Q3_K was the only warp K-quant missing these, so its GEMMs
            // ran the plain n128 tile at ~9.6 TF (vs 30-52 TF for the ag siblings) — the dominant
            // cost in unsloth's Q2_K mixed quant (down/o/kv are Q3_K). NO BK64W: like Q6K, the
            // heavy Q3_K decoder collides with the doubled k-stage (BK64W is a Q4K-only win).
            "native_gemm_warp",
            "native_gemm_warp_q3k_ag",
            &["-DFMT_Q3K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_n128_ag",
            &["-DFMT_Q3K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q3k_sk_ag",
            &["-DFMT_Q3K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q5_0", &["-DFMT_Q5_0"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5_0_n128",
            &["-DFMT_Q5_0", "-DNARROW_N"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q5_1", &["-DFMT_Q5_1"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5_1_n128",
            &["-DFMT_Q5_1", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs",
            &["-DFMT_IQ4XS"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_n128",
            &["-DFMT_IQ4XS", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_sk",
            &["-DFMT_IQ4XS", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_ag",
            &["-DFMT_IQ4XS", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_n128_ag",
            &["-DFMT_IQ4XS", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4xs_sk_ag",
            &["-DFMT_IQ4XS", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        (
            // IQ4_NL was the only codebook-4bit format without a warp family, so unsloth mixed
            // quants that put ffn_up/gate + attn_q on IQ4_NL (gemma-3-1b "Q2_K": 1152×13824 at
            // m=512) fell to the 64×64 native_gemm tile at ~8 TF while the same file's Q3_K
            // tensors ran the sk_ag warptile at ~48 TF — pp512 0.37× vs llama.cpp from this arm.
            "native_gemm_warp",
            "native_gemm_warp_iq4nl",
            &["-DFMT_IQ4NL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4nl_n128",
            &["-DFMT_IQ4NL", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4nl_sk",
            &["-DFMT_IQ4NL", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4nl_ag",
            &["-DFMT_IQ4NL", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4nl_n128_ag",
            &["-DFMT_IQ4NL", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_iq4nl_sk_ag",
            &["-DFMT_IQ4NL", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q2k", &["-DFMT_Q2K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_n128",
            &["-DFMT_Q2K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_sk",
            &["-DFMT_Q2K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_ag",
            &["-DFMT_Q2K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_n128_ag",
            &["-DFMT_Q2K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q2k_sk_ag",
            &["-DFMT_Q2K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q4_0", &["-DFMT_Q4_0"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_n128",
            &["-DFMT_Q4_0", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_sk",
            &["-DFMT_Q4_0", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_ag",
            &["-DFMT_Q4_0", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_n128_ag",
            &["-DFMT_Q4_0", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q4_0_sk_ag",
            &["-DFMT_Q4_0", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        ("native_gemm_warp", "native_gemm_warp_q5k", &["-DFMT_Q5K"]),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128",
            &["-DFMT_Q5K", "-DNARROW_N"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_sk",
            &["-DFMT_Q5K", "-DNARROW_N", "-DSPLIT_K"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_ag",
            &["-DFMT_Q5K", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_ag",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_GLOBAL"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_sk_ag",
            &["-DFMT_Q5K", "-DNARROW_N", "-DSPLIT_K", "-DA_GLOBAL"],
        ),
        // BM=32 row-tile variants of the A_GLOBAL dense warp GEMM's n128_ag (non-split-K) family
        // — MTP verify's batched-prefill draft window (m≈6-24, growing under the no-rewind
        // fallback) on the default BM=64 tile is mostly masked waste (m=8 → 87.5%). Only the
        // formats the qwen35-4B-UD-Q4_K_XL verify GEMMs actually hit (Q4_K/Q5_K/Q6_K/Q8_0, per
        // INFR_PROF_OP_SHAPES profiling) get a variant; selected per-dispatch by the recorder from
        // `m` (see `DENSE_SMALL_TILE_MAX_M` in recorder.rs). Same K-accumulation order as BM=64 —
        // tile GRANULARITY only, bit-identical. NO sk_ag (split-K) variants: the split-K family's
        // own `splits` dimension already fills the device at these shapes, so a smaller row tile
        // there measured a net LOSS (`dense_small_m_row_tile_bench`) — BM=64 stays unconditional
        // on that path (see `matmul_native_splitk`'s doc).
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag_bm32",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_ag_bm32",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_ag_bm32",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_ag_bm32",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_GLOBAL", "-DBM32"],
        ),
        // BM=16 row-tile variants — one coopmat M-frag floor, mirroring the BM=32 family above but
        // for the m<=~12-16 sub-band where BM=32's own tile is still ~half masked waste (see
        // `DENSE_SMALL_TILE_MAX_M16` in recorder.rs for the measured crossover). Same formats as
        // BM32 (Q4_K/Q5_K/Q6_K/Q8_0), same NARROW_N + A_GLOBAL, no split-K variant (unmeasured;
        // BM32 already skips sk_ag for the same reason — see `matmul_native_splitk`'s doc).
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_ag_bm16",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_ag_bm16",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_ag_bm16",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_ag_bm16",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_GLOBAL", "-DBM16"],
        ),
        // A_DIRECT variants — eliminate the separate store_f16 dispatch by reading f32 activations
        // directly and staging them to As with an f32→f16 cast. The same tile shapes as the n128_ag
        // family (BM=64/32/16), but coopMatLoad from As instead of from global. Saves the dispatch
        // + barrier per GEMM at the cost of As staging (shared memory + stage loop). Opt-in behind
        // INFR_GEMM_DIRECT_A=1; the default A_GLOBAL path is unchanged.
        // BM=64 (standard):
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_direct",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_DIRECT"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_direct",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_DIRECT"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_direct",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_DIRECT"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_direct",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_DIRECT"],
        ),
        // BM=32 (small-tile):
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_direct_bm32",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_DIRECT", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_direct_bm32",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_DIRECT", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_direct_bm32",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_DIRECT", "-DBM32"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_direct_bm32",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_DIRECT", "-DBM32"],
        ),
        // BM=16 (min-tile):
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_n128_direct_bm16",
            &["-DFMT_Q4K", "-DNARROW_N", "-DA_DIRECT", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q5k_n128_direct_bm16",
            &["-DFMT_Q5K", "-DNARROW_N", "-DA_DIRECT", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_n128_direct_bm16",
            &["-DFMT_Q6K", "-DNARROW_N", "-DA_DIRECT", "-DBM16"],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_n128_direct_bm16",
            &["-DFMT_Q8_0", "-DNARROW_N", "-DA_DIRECT", "-DBM16"],
        ),
        // 8x8x16-fragment `_cm8` builds — Intel Arc/ANV XMX, which enumerates f16 coopmat ONLY at
        // M=8,N=8,K=16 (no 16x16x16 config; all default coopmat kernels stay dark there). Runtime
        // selection requires the device to enumerate the 8x8x16 f16 shape AND `INFR_CM_8X8=1`
        // (default OFF: Alchemist coopmat is a llama.cpp-documented regression; the nc_mmq/nc_fma
        // tiers stay the Arc default) — see lib.rs `select_coopmat_shape` and the adapter's
        // `cm8_ok`. Tile: NARROW_N+BM32 (BM=32, BN=128, BK=64) — the 8x8 fragments double
        // CMS_M/CMS_N (2x4 = 8 frags/warp), so the wide 64x256 tile's 4x8 = 32 acc frags would
        // blow the per-lane register budget; the small tile keeps 8 frags = the default build's
        // register footprint. -DWG_THREADS=128: pinned subgroup 16 (XMX/DPAS is SIMD16-native)
        // x the warp math's required 8 subgroups. Hot k-quants + Q8_0 only (the field testers'
        // model set); other formats keep the nc tier — extend after first A/B reports.
        (
            "native_gemm_warp",
            "native_gemm_warp_q4k_cm8",
            &[
                "-DFMT_Q4K",
                "-DNARROW_N",
                "-DBM32",
                "-DCM_M=8",
                "-DCM_N=8",
                "-DWG_THREADS=128",
            ],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q6k_cm8",
            &[
                "-DFMT_Q6K",
                "-DNARROW_N",
                "-DBM32",
                "-DCM_M=8",
                "-DCM_N=8",
                "-DWG_THREADS=128",
            ],
        ),
        (
            "native_gemm_warp",
            "native_gemm_warp_q8_0_cm8",
            &[
                "-DFMT_Q8_0",
                "-DNARROW_N",
                "-DBM32",
                "-DCM_M=8",
                "-DCM_N=8",
                "-DWG_THREADS=128",
            ],
        ),
        ("splitk_reduce", "splitk_reduce", &[]),
        ("native_gemm", "native_gemm_q8_0", &["-DFMT_Q8_0"]),
        ("native_gemm", "native_gemm_bf16", &["-DFMT_BF16"]),
        ("native_gemm", "native_gemm_q4_0", &["-DFMT_Q4_0"]),
        ("native_gemm", "native_gemm_q4_1", &["-DFMT_Q4_1"]),
        ("native_gemm", "native_gemm_q5_0", &["-DFMT_Q5_0"]),
        ("native_gemm", "native_gemm_q5_1", &["-DFMT_Q5_1"]),
        ("native_gemm", "native_gemm_q2k", &["-DFMT_Q2K"]),
        ("native_gemm", "native_gemm_q3k", &["-DFMT_Q3K"]),
        ("native_gemm", "native_gemm_q4k", &["-DFMT_Q4K"]),
        ("native_gemm", "native_gemm_q5k", &["-DFMT_Q5K"]),
        ("native_gemm", "native_gemm_q6k", &["-DFMT_Q6K"]),
        ("native_gemm", "native_gemm_iq4nl", &["-DFMT_IQ4NL"]),
        ("native_gemm", "native_gemm_iq4xs", &["-DFMT_IQ4XS"]),
        ("native_gemm", "native_gemm_mxfp4", &["-DFMT_MXFP4"]),
        ("native_gemm", "native_gemm_nvfp4", &["-DFMT_NVFP4"]),
        ("native_gemm", "native_gemm_tq1_0", &["-DFMT_TQ1_0"]),
        ("native_gemm", "native_gemm_tq2_0", &["-DFMT_TQ2_0"]),
        ("native_gemm", "native_gemm_q2_0", &["-DFMT_Q2_0"]),
        (
            "native_gemm",
            "native_gemm_iq2xxs",
            &["-DFMT_IQ2XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq2xs",
            &["-DFMT_IQ2XS", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq2s",
            &["-DFMT_IQ2S", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq3xxs",
            &["-DFMT_IQ3XXS", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq3s",
            &["-DFMT_IQ3S", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq1s",
            &["-DFMT_IQ1S", "-DUSE_GRID"],
        ),
        (
            "native_gemm",
            "native_gemm_iq1m",
            &["-DFMT_IQ1M", "-DUSE_GRID"],
        ),
    ];
    // Subgroup-16 twins for the bandwidth-critical decode GEMV/reduction family (Intel R2, see
    // caps.sg_pref): every build of these SOURCES also gets a `<stem>_sg16` variant compiled with
    // -DSG=16 (subgroup width parameterized in the shader — shared-array sizing and lane strides
    // derive from SG; subgroup ops are width-agnostic). The base (SG=32) builds are byte-identical
    // to the pre-SG SPIR-V — the define only folds to the constants they already contained.
    // Selected at kernel-cache time by `caps.sg_pref == 16` (Intel); never created elsewhere.
    const SG16_SOURCES: &[&str] = &[
        "native_gemv_sg",
        "native_gemv_id_multi_sg",
        "native_mmv_mw",
        "quant_q8_row",
    ];
    let builds: Vec<(String, String, Vec<String>)> = builds
        .iter()
        .flat_map(|(src_stem, dst_stem, defines)| {
            let base = (
                src_stem.to_string(),
                dst_stem.to_string(),
                defines.iter().map(|d| d.to_string()).collect::<Vec<_>>(),
            );
            if SG16_SOURCES.contains(src_stem) {
                let mut sg_defines = base.2.clone();
                sg_defines.push("-DSG=16".to_string());
                vec![
                    base.clone(),
                    (src_stem.to_string(), format!("{dst_stem}_sg16"), sg_defines),
                ]
            } else {
                vec![base]
            }
        })
        .collect();
    // STREAMED weight builds (resident-BDA): the `-DSTREAMED` build reads the weight from a
    // `bufferDeviceAddress` arena pool (native_weight_addr.glsl) instead of a bound SSBO, lifting the
    // ~4 GiB `maxStorageBufferRange` cap one SSBO binding imposes. `wants_streamed_twin` marks the
    // families that get a STREAMED build; `mechanism_b_resident` (below) marks the few whose
    // bound-SSBO build is STILL dispatched — every eager-runner / parity-recorder / production
    // consumer that used to justify a resident build for native_gemv(_sg/_rm/_rm_v2),
    // linear_f16/linear_bf16, or conv1d_silu(_par)/e2b_gate is gone (retired alongside the resident
    // paths themselves — production's `Recorder::conv1d_silu`/`conv1d_silu_batch`/`e2b_gate` now
    // resolve the weight's own BDA address and dispatch the STREAMED build directly, same as every
    // other ported family), so those families dropped out of the resident whitelist too. For every
    // other family STREAMED is now the SOLE weight build — the bound-SSBO "resident" build is no
    // longer emitted, and the only families with a genuinely-live resident build left are the
    // mechanism-(b) ones (below), which is why the whitelist is named for that. The list below
    // still enumerates which families qualify for a STREAMED twin at all:
    //   * native_gemv / native_gemv_mrow / native_gemv_sg / native_gemv_rm / native_gemv_rm_v2 —
    //     decode/small-m GEMV family (any-m rows loop, spec-decode multi-row, subgroup-reduce
    //     projection band, multi-output-row, and its experimental variants). Both plain and
    //     -DUSE_RES builds get a twin: production DOES dispatch fused-residual GEMVs on resident
    //     weights (the decode Linear+Add fusion — recorder.rs `linear_add_native`), and the
    //     resident-BDA endgame needs every bound-SSBO weight path covered, residual or not
    //     (`native_gemv_mrow` itself has no `-DUSE_RES` build today — the multi-row spec-decode path
    //     never fuses a residual — so the guard is a no-op there). The STREAMED arm only reroutes
    //     the weight-word chokepoint (NW), which is independent of the USE_RES residual-buffer
    //     binding shuffle — see native_gemv.comp's binding layout (WB dropped under STREAMED; RB2/YB
    //     shift is driven purely by USE_RES, unaffected by STREAMED). `native_gemv_sg` also rides
    //     the SG16 twin expansion above (this closure runs AFTER it), so both the SG=32 and SG=16
    //     builds get their own `_streamed` twin by construction — no separate SG16 handling needed
    //     here.
    //   * native_gemm / native_gemm_warp — the prefill coopmat-dequant GEMM family (base 64×64 and
    //     the 8-warp warptile: wide/NARROW_N/A_GLOBAL/SPLIT_K/BM* × every quant format, plus the
    //     opt-in -DBF16CM / `_cm8` (-DCM_M=8) coopmat variants and the -DFMT_F16 split-K build). A
    //     streamed dense prefill Linear routes through the SAME tile the resident path picks, just
    //     arena-addressed (adapter.rs `streamed_prefill_gemm`). Previously EXCLUDED BF16CM/CM_M=8/
    //     FMT_F16 as opt-in-only or float-upload-only — re-audited: none of the three change the
    //     weight-READ source. BF16CM/CM_M only retype/resize the coopmat A/B fragment (CMTYPE /
    //     CM_M/CM_N), and FMT_F16's dq/dqblk (native_decode.glsl) route through the exact same
    //     ru16->rb->NW() chokepoint every other format uses — so all three compile a working
    //     `_streamed` twin off the STREAMED arm this file already carries. Every native_gemm /
    //     native_gemm_warp variant now gets a twin.
    let wants_streamed_twin = |src_stem: &str, defines: &[String]| -> bool {
        match src_stem {
            "native_gemv" => true,
            "native_gemm" | "native_gemm_warp" => true,
            "native_gemv_mrow" | "native_gemv_sg" | "native_gemv_rm" | "native_gemv_rm_v2" => true,
            // Token-embedding gather (Op::EmbedGather): reads the GGUF table via the shared NW()
            // chokepoint, same as native_gemv. Every format variant gets a twin.
            "embed_gather" => true,
            // The int8 dp4a GEMV family. Same additive reasoning as the dequant GEMVs above
            // (including -DUSE_RES: production dispatches fused-residual int8 GEMVs on resident
            // weights too — recorder.rs `linear_add_mmv`/`linear_mmv_mrow`); these carry their own
            // decode (not native_decode.glsl's), so each declares the `NW(i)` chokepoint locally —
            // see native_mmv.comp.
            "native_mmv" | "native_mmv_mrow" | "native_mmv_mw" => true,
            // The tiled dp4a expert-GEMM family (base + every EXPERT_GRID tile variant). The
            // `-DPAGED` builds already read the arena by device address (LUT/slot form) and never
            // get a `_streamed` twin — STREAMED is the LUT-less resident form of the same read.
            stem if stem.starts_with("native_gemm_mmq_") => !defines.iter().any(|d| d == "-DPAGED"),
            // The id-indexed resident expert family. STREAMED repurposes `stride` as the
            // per-expert BYTE stride applied on the 64-bit pointer (ids picks the expert, no
            // LUT), killing the u32 element-space multiply's overflow reach; same PAGED
            // exclusion.
            "native_gemv_id"
            | "native_gemv_id_multi"
            | "native_gemv_id_multi_sg"
            | "native_mmv_id_q4k" => !defines.iter().any(|d| d == "-DPAGED"),
            // The float-weight resident family: the dedicated f16/bf16 linear GEMVs (typed
            // buffer_reference reads — see linear_f16.comp's STREAMED doc), the mrow/vec4-tiled
            // `linear_f32r` family, and the non-coopmat fma prefill GEMM. `linear_f32.comp` (the
            // eager thread-per-output f32 GEMV) is NOT here — see the base builds list above for why
            // it's gone entirely. linear_res IS the fused-residual f16 GEMV file, included since
            // the resident-BDA endgame deletes every bound-SSBO weight path.
            "linear_f16" | "linear_f16_noext" | "linear_bf16" | "linear_f32r" | "linear_res"
            | "native_gemm_fma" => true,
            // Model-specific float-weight kernels: qwen35's SSM depthwise conv1d+SiLU (both the
            // sequential and batch-pass-1 variants — conv1d_shift never reads the weight) and
            // gemma4 E2B's per-layer fused inp_gate GEMV. Same typed buffer_reference seam as the
            // linear family above (see conv1d_silu.comp's STREAMED doc). Every variant of these
            // sources gets a twin; NOT in `mechanism_b_resident` below — production's
            // `Recorder::conv1d_silu`/`conv1d_silu_batch`/`e2b_gate` now resolve the weight's own
            // BDA address and dispatch this STREAMED build directly (the old bound-SSBO resident
            // build is dead), same as linear_f16/linear_bf16 above. (e2b_proj — the per-layer PROJ
            // counterpart — is gone entirely: see the base builds list above.)
            "conv1d_silu" | "conv1d_silu_par" | "e2b_gate" => true,
            // The coopmat f16/repacked-quant projection GEMM (prefill C=A·Wᵀ): the quant arms ride
            // along on the shared WQ() seam — see gemm_proj.comp's STREAMED doc.
            "gemm_proj" | "gemm_proj_warp" => true,
            // int8-coopmat Q8_0 prefill GEMM (INFR_I8_COOPMAT=1 measurement kernel, base +
            // -DROW_SCALE): the raw Q8_0 weight byte read (a local `rb()`/`NW()` chokepoint, not
            // native_decode.glsl's) gets the same inline-buffer_reference seam as gemm_proj — see
            // native_gemm_i8cm_q8_0.comp's STREAMED doc. Both defines only affect the ACTIVATION
            // scale source, never the weight binding.
            "native_gemm_i8cm_q8_0" => true,
            // Load-time Q8_0->E4M3 repack (feeds the -DPREPACK fp8-coopmat GEMM's prepacked weight
            // buffer): reads the Q8_0 source via the same native_decode.glsl NW() chokepoint as
            // native_gemm_warp — see repack_q8_to_f8.comp's STREAMED doc. The E4M3 output buffer
            // (binding 1) is a derived scratch buffer the arena doesn't own, so it stays bound.
            "repack_q8_to_f8" => true,
            _ => false,
        }
    };
    // Resident-BDA endgame: for every twin family the recorder now dispatches STREAMED-only
    // (weights read by 64-bit device address), the resident (non-`-DSTREAMED`) build is dead — the
    // `_streamed` twin is the SOLE weight build, emitted under the base (unsuffixed) name (see the
    // `else` arm below). `mechanism_b_resident` is the whitelist of twin-family builds whose
    // RESIDENT SPV is still genuinely dispatched (bound-descriptor read), so it must keep compiling
    // alongside the SUFFIXED streamed twin instead. Every surviving entry is mechanism-(b): the
    // standalone mmq Q4_K/Q6_K, the int8/fp8-coopmat GEMMs, the Q8_0→E4M3 repack, and
    // native_gemm_warp's `_cm8`/`_bf16cm` coopmat tiles all bind a `BdaSub` sub-range via vkb and
    // read through the bound descriptor — their resident build IS live, which is why the whitelist
    // is named for that mechanism (nothing else qualifies any more). native_gemv/_sg/_rm/_rm_v2,
    // linear_f16/linear_bf16, and conv1d_silu/conv1d_silu_par/e2b_gate used to be here too (the
    // eager `VulkanBackend::linear_native`/`linear_{f16,bf16}`, the `Recorder::linear_native_{sg,
    // rm,rm_v2}` parity dispatchers, and production's own `Recorder::conv1d_silu`/
    // `conv1d_silu_batch`/`e2b_gate` all bound the weight as an SSBO), but every one of those
    // callers is gone — the STREAMED twin is now the sole route everywhere
    // (`linear_native_at` and its `_sg`/`_rm`/`_rm_v2` siblings,
    // `Recorder::linear_at`/`linear_bf16_at`, and `Recorder::conv1d_silu`/
    // `conv1d_silu_batch`/`e2b_gate` themselves now resolve `device_addr()` and delegate to their
    // own `_streamed` twin), so dropping them from this whitelist just stops compiling the now-dead
    // bound-SSBO half. Everything else in a twin family (the dense/expert coopmat GEMM tiles, the
    // mmv/mrow/id float and dequant GEMVs, embed_gather, gemm_proj, native_gemm_fma) was ALREADY
    // STREAMED-only.
    // NOTE: `-DSTREAMED` is the ONLY reason the source of a dual family compiles twice, so it names
    // a real second build for EXACTLY these mechanism-(b) entries — the 5 emitted `_streamed` twins
    // (native_gemm_i8cm_q8_0 base + `_rowscale`, native_gemm_mmq_q4k/_q6k standalone, repack_q8_to_f8;
    // native_gemm_warp's cm8/bf16cm are whitelisted but resident-ONLY via `dead_streamed_twin`, so
    // they carry no twin). Elsewhere the define rides unconditionally on the sole build. Once these
    // dual sources lose their resident builds, `-DSTREAMED` becomes unconditional everywhere and
    // folds away with them — which is why it is NOT renamed/removed in isolation.
    let mechanism_b_resident = |src: &str, defines: &[String]| -> bool {
        match src {
            "native_gemm_i8cm_q8_0" | "repack_q8_to_f8" => true,
            "native_gemm_warp" => defines.iter().any(|d| d == "-DCM_M=8" || d == "-DBF16CM"),
            "native_gemm_mmq_q4k" | "native_gemm_mmq_q6k" => defines.is_empty(),
            _ => false,
        }
    };
    // Dead `_streamed` twins (#75): a few `mechanism_b_resident` families keep their
    // bound-descriptor resident build but nothing ever dispatches the arena-addressed twin.
    // `native_gemm_warp`'s native-coopmat `_cm8`/`_bf16cm` tiles are read straight from the
    // resident SPV by `Recorder::matmul_native_cm8` (and its bf16cm sibling) — never through the
    // `native_gemm_spv` arena map — so their STREAMED twin was compiled but unreachable. Emit
    // resident-only for them.
    let dead_streamed_twin = |src: &str, defines: &[String]| -> bool {
        src == "native_gemm_warp" && defines.iter().any(|d| d == "-DCM_M=8" || d == "-DBF16CM")
    };
    let builds: Vec<(String, String, Vec<String>)> = builds
        .into_iter()
        .flat_map(|(src_stem, dst_stem, defines)| {
            if !wants_streamed_twin(&src_stem, &defines) {
                return vec![(src_stem, dst_stem, defines)];
            }
            let mut sd = defines.clone();
            sd.push("-DSTREAMED".to_string());
            if mechanism_b_resident(&src_stem, &defines) {
                if dead_streamed_twin(&src_stem, &defines) {
                    // Resident is dispatched directly; the STREAMED twin has no caller — drop it.
                    vec![(src_stem, dst_stem, defines)]
                } else {
                    // Dual build: the resident SPV is live and the SUFFIXED streamed twin is too
                    // (parity recorders / production mmq / int8-coopmat parity). Keep both names.
                    let streamed = (src_stem.clone(), format!("{dst_stem}_streamed"), sd);
                    vec![(src_stem, dst_stem, defines), streamed]
                }
            } else {
                // STREAMED is the SOLE weight build — emit it under the base name (no `_streamed`
                // suffix): there is no resident sibling to disambiguate against.
                vec![(src_stem, dst_stem, sd)]
            }
        })
        .collect();
    for (src_stem, dst_stem, defines) in &builds {
        let src = format!("shaders/{src_stem}.comp");
        let dst = format!("{out}/{dst_stem}.spv");
        println!("cargo:rerun-if-changed={src}");
        let mut args: Vec<String> = vec![
            "-fshader-stage=comp".into(),
            "--target-env=vulkan1.3".into(),
            "-O".into(),
            format!("-I{out}"),
        ];
        for d in defines {
            args.push(d.clone());
        }
        args.push(src.clone());
        args.push("-o".into());
        args.push(dst);
        let status = Command::new("glslc")
            .args(&args)
            .status()
            .expect("failed to run glslc — install shaderc (provides glslc)");
        assert!(status.success(), "glslc failed for {src}");
    }
    // Shader-set fingerprint for the on-disk vkPipelineCache (see src/pcache.rs): FNV-1a over
    // every compiled SPIR-V blob in the (stable) build-list order. Any shader edit, new variant,
    // or glslc/define change flips it, and the persisted cache file is discarded wholesale —
    // stale pipeline entries never accumulate across shader-set changes.
    let mut h: u64 = 0xcbf29ce484222325;
    for (_, dst_stem, _) in &builds {
        let bytes = std::fs::read(format!("{out}/{dst_stem}.spv")).expect("read spv for hash");
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    std::fs::write(
        format!("{out}/shader_fingerprint.rs"),
        format!("pub(crate) const SHADER_SET_FINGERPRINT: u64 = {h:#x};\n"),
    )
    .expect("write shader_fingerprint.rs");
    // Generated dense-GEMM kernel-name → SPIR-V lookup for the prefill GEMM family (native_gemm /
    // native_gemm_warp). `Recorder::matmul_native_{off,f16a,splitk}` compute the kernel name for the
    // tile they want, then load its SPIR-V here — a runtime `&str` → compile-time `include_bytes!`
    // bridge, single-sourced in the recorder (no hand-kept parallel match to drift). Since STREAMED
    // is the sole weight build for these tiles, the name and the emitted `.spv` stem are identical:
    // the map is the exhaustive name→SPV switch, not a resident→twin remap. Keyed by dst_stem; the
    // arena is read by 64-bit address, so nothing binds a resident SSBO. native_gemv keeps its own
    // selector. Excludes dst names ending in `_streamed` (the dual-build keepers' suffixed twins:
    // mmq / int8-coopmat, dispatched through their own getters).
    let mut gemm_names: Vec<String> = builds
        .iter()
        .filter(|(_, dst, defines)| {
            dst.starts_with("native_gemm")
                && !dst.ends_with("_streamed")
                && defines.iter().any(|d| d == "-DSTREAMED")
        })
        .map(|(_, dst, _)| dst.clone())
        .collect();
    gemm_names.sort();
    gemm_names.dedup();
    let mut map = String::from(
        "// Generated by build.rs — dense-GEMM kernel name to its (name, SPIR-V). The weight build is\n\
         // STREAMED-only, so the name IS the emitted .spv stem; 'static literals cache distinctly.\n\
         pub(crate) fn native_gemm_spv(name: &str) -> (&'static str, &'static [u32]) {\n\
         \x20   macro_rules! v {\n\
         \x20       ($n:literal) => {{\n\
         \x20           static S: OnceLock<Vec<u32>> = OnceLock::new();\n\
         \x20           (\n\
         \x20               $n,\n\
         \x20               S.get_or_init(|| spv_words(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/\", $n, \".spv\"))))\n\
         \x20                   .as_slice(),\n\
         \x20           )\n\
         \x20       }};\n\
         \x20   }\n\
         \x20   match name {\n",
    );
    for r in &gemm_names {
        map += &format!("        {r:?} => v!({r:?}),\n");
    }
    map += "        _ => panic!(\"no dense GEMM SPIR-V for kernel {name}\"),\n    }\n}\n";
    std::fs::write(format!("{out}/native_gemm_map.rs"), map).expect("write native_gemm_map.rs");
}

/// Generate `native_grids.glsl` (the i-quant lookup tables) from `infr_core::iquant_grids` —
/// single source of truth, baked into the per-format SPIR-V at build time. Each table is
/// `#if`-guarded by the formats that use it. WGSL/GLSL has no u64, so u64 grids are emitted as u32
/// pairs (lo, hi).
///
/// Each table is emitted TWICE: the raw data as a `const` array (`C_<name>`) plus a `shared`
/// mirror (`<name>`) that `grid_init()` fills cooperatively once per workgroup. The dequant code
/// only ever indexes the shared mirror: a dynamically-indexed large `const` array is lowered
/// (glslang emits one initialized Function-storage copy PER ACCESS SITE; RADV/ACO puts them in
/// per-invocation scratch) to a full table materialization in scratch memory by every invocation —
/// the IQ2_S id-GEMV carried 1 MB of scratch per wave and ran ~400x slower than its memory
/// traffic, stacking one real MoE decode step (40 layers x gate/up/down, 8 experts) past amdgpu's
/// ~10 s gfx-ring timeout (device-lost TDR on Qwen3.6-35B-A3B IQ3_S; `ring gfx_0.0.0 timeout` in
/// dmesg). The single sequential access site inside `grid_init()`'s copy loop is promoted to
/// SPIR-V constant data by Mesa (`nir_opt_large_constants`) instead — no scratch at all. Every
/// USE_GRID includer must run `GRID_INIT;` (see native_decode.glsl) at the top of main() — it
/// contains a barrier(), so it must sit in uniform control flow.
fn gen_grids(out: &str) {
    use infr_core::iquant_grids as g;
    // (name, guard, u32 word count) for every emitted table, driving grid_init()'s copy loops.
    let mut tables: Vec<(String, String, usize)> = Vec::new();
    let mut decl = |name: &str, guard: &str, words: &mut dyn Iterator<Item = u32>| -> String {
        let mut t = format!("#if {guard}\nconst uint C_{name}[] = uint[](");
        let mut n = 0usize;
        for v in words {
            t += &format!("{v}u,");
            n += 1;
        }
        t.pop();
        t += &format!(");\nshared uint {name}[{n}];\n#endif\n");
        tables.push((name.to_string(), guard.to_string(), n));
        t
    };
    let mut s =
        String::from("// Generated by build.rs from infr_core::iquant_grids. Do not edit.\n");
    let ks_guard = "defined(FMT_IQ2XXS) || defined(FMT_IQ2XS) || defined(FMT_IQ3XXS)";
    s += &decl(
        "KSIGNS",
        ks_guard,
        &mut g::KSIGNS_IQ2XS.iter().map(|&v| v as u32),
    );
    let lohi = |grid: &'static [u64]| grid.iter().flat_map(|&v| [v as u32, (v >> 32) as u32]);
    s += &decl(
        "G_IQ2XXS",
        "defined(FMT_IQ2XXS)",
        &mut lohi(&g::IQ2XXS_GRID),
    );
    s += &decl("G_IQ2XS", "defined(FMT_IQ2XS)", &mut lohi(&g::IQ2XS_GRID));
    s += &decl("G_IQ2S", "defined(FMT_IQ2S)", &mut lohi(&g::IQ2S_GRID));
    s += &decl(
        "G_IQ3XXS",
        "defined(FMT_IQ3XXS)",
        &mut g::IQ3XXS_GRID.iter().copied(),
    );
    s += &decl(
        "G_IQ3S",
        "defined(FMT_IQ3S)",
        &mut g::IQ3S_GRID.iter().copied(),
    );
    s += &decl(
        "G_IQ1S",
        "defined(FMT_IQ1S) || defined(FMT_IQ1M)",
        &mut lohi(&g::IQ1S_GRID),
    );
    s += "void grid_init() {\n";
    s += "    uint gstep = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;\n";
    for (name, guard, n) in &tables {
        s += &format!("#if {guard}\n");
        s += &format!(
            "    for (uint gi = gl_LocalInvocationIndex; gi < {n}u; gi += gstep) {{ {name}[gi] = C_{name}[gi]; }}\n"
        );
        s += "#endif\n";
    }
    s += "    barrier();\n}\n";
    std::fs::write(format!("{out}/native_grids.glsl"), s).expect("write native_grids.glsl");
}
