#![cfg(target_os = "macos")] // like the lib: empty on other targets (metal dep is cfg-gated)

//! Tripwire for the silent-missing-kernel class: every kernel name the executor can dispatch
//! must exist in the compiled MSL library.
//!
//! Why this exists: the executor's pipeline cap-checks treat a missing function as "capability
//! absent" and fall back gracefully — correct behavior for genuinely capped devices, but it
//! also means a kernel that silently VANISHES from the library (e.g. a rebase restoring a stale
//! copy of the shader source, as happened once: the native-read KV kernels disappeared and
//! q4_0 decode at depth ran 3x slower with zero errors anywhere) degrades performance invisibly
//! instead of failing. This test turns that class red: it scrapes every kernel-shaped string
//! literal out of exec.rs and asserts each one resolves in the library.

/// A string literal in exec.rs is "kernel-shaped" when it starts with one of the kernel-family
/// prefixes. Names are scraped from the SOURCE (include_str! at compile time), so a new dispatch
/// site is covered automatically — no list to maintain. If a literal matches a prefix but is
/// not meant to be a kernel, extend the exclusion below with a comment.
const KERNEL_PREFIXES: &[&str] = &[
    "add_",
    "argmax",
    "attention",
    "attnflash",
    "attnvec",
    "cast_",
    "cmm_",
    "conv1d_",
    "copy_",
    "deltanet_",
    "dequant_",
    "embed_gather_",
    "gatedact",
    "linear_",
    "moe_",
    "qknorm",
    "rmsnorm",
    "rope_",
    "sample_f32",
    "scale_f32",
    "softcap",
    "writekv_",
];

/// Strip ALL whitespace, so a shader tripwire matches on TOKENS rather than on exact source
/// formatting.
///
/// These `.metal` files are macro bodies with column-aligned `\` line continuations: adding one
/// long line re-pads the backslash column of its neighbours, and a formatter that rewrites
/// `16u * (sgid >> 1)` to `16u*(sgid>>1)` changes not one bit of generated code. A raw
/// `src.contains("...")` tripwire goes red on both — a false failure that teaches the next
/// person to "fix" the test, which is exactly how a tripwire stops being trusted. Comparing
/// de-spaced needles against de-spaced source survives reformatting while still going red if
/// the operative expression actually changes.
fn despace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Assert `needle` appears in `src`, ignoring whitespace. Panics with the needle for a readable
/// failure (the de-spaced haystack is a useless thing to print).
#[track_caller]
fn asserts_token_seq(src: &str, needle: &str) {
    assert!(
        despace(src).contains(&despace(needle)),
        "shader tripwire: expected to find `{needle}` (ignoring whitespace).\n\
         If this optimization was removed ON PURPOSE, delete the tripwire in the same commit \
         and say so — do not relax it.",
    );
}

// The four tripwires below guard OPTIMIZATIONS, not correctness. The parity tests pass whether
// or not these are present (a reverted optimization is still numerically right, just slower), so
// nothing else in the suite would notice if one silently vanished — the exact failure mode this
// file's header describes. That is why they assert on shader source at all.

#[test]
fn iq4nl_has_a_native_four_row_decode_body() {
    let src = include_str!("../shaders/linear.metal");
    asserts_token_seq(src, "inline void linear_iq4nl_body");
    asserts_token_seq(src, "kernel void linear_iq4nl_add");
}

#[test]
fn four_row_attention_has_a_wide_kv_chunk() {
    let src = include_str!("../shaders/attention.metal");
    asserts_token_seq(src, "template<uint hd, uint NSG, uint C>");
    asserts_token_seq(src, "constexpr uint NP = C / 64u");
    asserts_token_seq(src, "host_name(\"attnflash2_c128_f16kv_hd128\")");
}

#[test]
fn moe_cmm_masks_inactive_matrix_row_fragments() {
    // Partial expert tiles must skip dead 8-row fragments instead of running the full MMA and
    // discarding half of it (see moe.metal). `row_base` is derived from `sgid` alone, which is
    // what keeps the branch simdgroup-uniform and the `simdgroup_barrier` inside it legal.
    let src = include_str!("../shaders/moe.metal");
    asserts_token_seq(src, "uint row_base = 16u * (sgid >> 1);");
    asserts_token_seq(src, "if (row_base + 8u < nr1) {");
    asserts_token_seq(src, "else if (row_base < nr1) {");
}

#[test]
fn q5k_reconstructs_four_codes_per_word() {
    // SWAR: rebuild four 5-bit codes per word (4-bit code | 5th bit) instead of decoding 16
    // bytes one at a time.
    let src = include_str!("../shaders/linear.metal");
    asserts_token_seq(src, "uint packed = (q & 0x0F0F0F0Fu)");
    asserts_token_seq(src, "(h & 0x01010101u) << 4u");
    asserts_token_seq(src, "packed >> 24u");
}

#[test]
fn q4k_q5k_use_wide_headers_only_for_cooperative_prefill() {
    let linear = include_str!("../shaders/linear.metal");
    asserts_token_seq(linear, "inline uint4 load_k4_header");
    asserts_token_seq(linear, "inline uint2 decode_k4_scale_min");
    asserts_token_seq(linear, "template<bool WIDE> inline void decode16_q4k");
    asserts_token_seq(linear, "template<bool WIDE> inline void decode16_q5k");
    asserts_token_seq(linear, "#define DEC16_Q4K(wk) decode16_q4k<false>");
    asserts_token_seq(linear, "#define DEC16_Q5K(wk) decode16_q5k<false>");

    let moe = include_str!("../shaders/moe.metal");
    asserts_token_seq(moe, "CMM_KERNEL(linear_q4k_cmm, DEC16_Q4K_WIDE)");
    asserts_token_seq(moe, "CMM_KERNEL(linear_q5k_cmm, DEC16_Q5K_WIDE)");
    let compact = despace(moe);
    assert_eq!(compact.matches("DEC16_Q4K_WIDE").count(), 1);
    assert_eq!(compact.matches("DEC16_Q5K_WIDE").count(), 1);

    asserts_token_seq(moe, "GEMV_KERNEL(linear_q5k, DEC16_Q5K)");
    asserts_token_seq(moe, "RT_KERNEL(linear_q4k_rt, DEC16_Q4K)");
    asserts_token_seq(moe, "RT_KERNEL(linear_q5k_rt, DEC16_Q5K)");
    asserts_token_seq(moe, "CMMKS_KERNEL(linear_q4k_cmm_ks, DEC16_Q4K)");
    asserts_token_seq(moe, "CMMKS_KERNEL(linear_q5k_cmm_ks, DEC16_Q5K)");
    asserts_token_seq(moe, "HGEMM_KERNEL(linear_q4k_hmm, DEC16_Q4K)");
    asserts_token_seq(moe, "HGEMM_KERNEL(linear_q5k_hmm, DEC16_Q5K)");
    asserts_token_seq(moe, "MOE_CMM_KERNEL(linear_q4k_cmm_id, DEC16_Q4K");
    asserts_token_seq(moe, "MOE_CMM_KERNEL(linear_q4k_cmm_id_w, DEC16_Q4K");

    let gather = include_str!("../shaders/embed_gather.metal");
    asserts_token_seq(gather, "EMBED_GATHER_KERNEL(embed_gather_q4k, DEC16_Q4K)");
}

#[test]
fn q5k_four_row_can_prefer_the_existing_row_tile() {
    let exec = include_str!("../src/exec.rs");
    // The knob moved off the environment in the config campaign (S6): `INFR_METAL_NO_Q5K_RT` is
    // now `kernels.metal.q5k_rt`, read once off the backend's config. The tripwire follows the
    // value, not the spelling.
    asserts_token_seq(exec, "let q5k_rt = self.metal().q5k_rt;");
    asserts_token_seq(exec, "prefer_q5k_rt(qw.kern, m, q5k_rt)");
}

#[test]
fn regular_cmm_unrolls_its_fixed_tile_loops() {
    let src = include_str!("../shaders/moe.metal");
    asserts_token_seq(
        src,
        "#define CMM_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)",
    );
    let compact = despace(src);
    let cmm = compact
        .split_once(&despace(
            "#define CMM_KERNEL_TYPED(NAME, DEC, WTYPE, WMAT, XTYPE, XVEC, XMAT, SHWORDS, SBOFF)",
        ))
        .unwrap()
        .1
        .split_once(&despace("#define CMMKS_KERNEL(NAME, DEC)"))
        .unwrap()
        .0;
    assert!(cmm.matches("CMM_UNROLL").count() >= 8);
}

#[test]
fn f16_linear_reads_the_bound_weight_directly() {
    let shader = include_str!("../shaders/linear.metal");
    asserts_token_seq(shader, "kernel void linear_f16");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "DType::F16 if f16_native => (\"linear_f16\", 2u64)");
}

#[test]
fn f16_aligned_multirow_linear_uses_the_cooperative_tile() {
    let shader = include_str!("../shaders/moe.metal");
    asserts_token_seq(shader, "CMM_F16_KERNEL(linear_f16_cmm, DEC16_F16)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "f16_native && m >= 8");
    asserts_token_seq(exec, "self.pipelines.get(\"linear_f16_cmm\")");
}

#[test]
fn f16_small_multirow_linear_uses_the_exact_row_tile() {
    let shader = include_str!("../shaders/moe.metal");
    asserts_token_seq(shader, "RT_KERNEL(linear_f16_rt, DEC16_F16)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "let f16_rt = f16_native && (2..16).contains(&m)");
    asserts_token_seq(exec, "Some(\"linear_f16_rt\")");
}

#[test]
fn f32_linear_reads_the_bound_weight_directly() {
    let exec = include_str!("../src/exec.rs");
    // `INFR_METAL_NO_F32_NATIVE` became `kernels.metal.f32_native` in the config campaign (S6);
    // the optimization is unchanged, only where the gate's value comes from.
    asserts_token_seq(
        exec,
        "let f32_native = wdt == DType::F32 && mcfg.f32_native;",
    );
    asserts_token_seq(exec, "DType::F32 if f32_native => (\"linear_f32\", 4u64)");
}

#[test]
fn f32_aligned_multirow_linear_uses_the_cooperative_tile() {
    let shader = include_str!("../shaders/moe.metal");
    asserts_token_seq(shader, "CMM_F32_KERNEL(linear_f32_cmm, DEC16_F32)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "f32_native && m >= 8");
    asserts_token_seq(exec, "self.pipelines.get(\"linear_f32_cmm\")");
}

#[test]
fn f32_small_multirow_linear_uses_the_exact_row_tile() {
    let shader = include_str!("../shaders/moe.metal");
    asserts_token_seq(shader, "RT_KERNEL(linear_f32_rt, DEC16_F32)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "let f32_rt = f32_native && (2..16).contains(&m)");
    asserts_token_seq(exec, "Some(\"linear_f32_rt\")");
}

#[test]
fn bf16_linear_reads_the_bound_weight_directly() {
    let shader = include_str!("../shaders/linear.metal");
    asserts_token_seq(shader, "kernel void linear_bf16");
    asserts_token_seq(shader, "as_type<float>((uint)wo[i] << 16u)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(
        exec,
        "DType::Bf16 if bf16_native => (\"linear_bf16\", 2u64)",
    );
}

#[test]
fn bf16_multirow_linear_uses_the_exact_row_tile() {
    let shader = include_str!("../shaders/moe.metal");
    asserts_token_seq(shader, "RT_KERNEL(linear_bf16_rt, DEC16_BF16)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "let bf16_rt = bf16_native && m >= 2");
    asserts_token_seq(exec, "Some(\"linear_bf16_rt\")");
}

#[test]
fn bf16_aligned_multirow_linear_uses_the_cooperative_tile() {
    let shader = include_str!("../shaders/moe.metal");
    asserts_token_seq(shader, "CMM_BF16_KERNEL(linear_bf16_cmm, DEC16_BF16)");

    let exec = include_str!("../src/exec.rs");
    asserts_token_seq(exec, "bf16_native && m >= 6");
    asserts_token_seq(exec, "self.pipelines.get(\"linear_bf16_cmm\")");
}

// The two below test the TRIPWIRE ITSELF. A guard nobody has watched fail is not a guard: it can
// rot into a tautology (matching something that is always present) and nothing would say so.

#[test]
#[should_panic(expected = "shader tripwire")]
fn a_tripwire_goes_red_when_its_optimization_is_removed() {
    let gutted =
        include_str!("../shaders/moe.metal").replace("uint row_base = 16u * (sgid >> 1);", "");
    asserts_token_seq(&gutted, "uint row_base = 16u * (sgid >> 1);");
}

#[test]
fn a_tripwire_survives_reformatting_of_its_optimization() {
    let reformatted = include_str!("../shaders/moe.metal").replace(
        "uint row_base = 16u * (sgid >> 1);",
        "uint row_base=16u*(sgid>>1);", // same tokens, no whitespace
    );
    asserts_token_seq(&reformatted, "uint row_base = 16u * (sgid >> 1);");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn every_dispatchable_kernel_exists_in_the_library() {
    let src = include_str!("../src/exec.rs");
    // Scrape "..." literals. exec.rs kernel names are plain [a-z0-9_] identifiers.
    let mut names: Vec<String> = Vec::new();
    let mut rest = src;
    while let Some(q) = rest.find('"') {
        rest = &rest[q + 1..];
        let Some(end) = rest.find('"') else { break };
        let lit = &rest[..end];
        rest = &rest[end + 1..];
        // Kernel-SHAPED literals that are NOT dispatch targets. `linear_q2k`/`linear_q3k` used to
        // live here (no native kernel; they appeared only in a negative registry assertion), but
        // both are now NATIVE with real decode kernels in the library, so they ARE dispatched and
        // resolve like any other name. `linear_quik_bogus` is the deliberately-fake base in
        // exec.rs's `qui_linear_kerns` loud-miss assertion — never a real kernel, exclude it.
        const NOT_DISPATCHED: &[&str] = &["linear_quik_bogus"];
        if !lit.is_empty()
            && lit
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && KERNEL_PREFIXES.iter().any(|p| lit.starts_with(p))
            && !NOT_DISPATCHED.contains(&lit)
            && !names.iter().any(|n| n == lit)
        {
            names.push(lit.to_string());
        }
    }
    assert!(
        names.len() > 40,
        "kernel-name scrape looks broken: only {} names found",
        names.len()
    );

    let dev = metal::Device::system_default().expect("no Metal device");
    // Resolve each name against EXACTLY the source the runtime compiles (infr_metal::msl_source
    // is the backend's own assembly — no separately-maintained file list to drift).
    let lib = dev
        .new_library_with_source(&infr_metal::msl_source(), &metal::CompileOptions::new())
        .expect("MSL library compile");
    let have: std::collections::HashSet<String> = lib.function_names().into_iter().collect();

    let missing: Vec<&String> = names.iter().filter(|n| !have.contains(*n)).collect();
    assert!(
        missing.is_empty(),
        "exec.rs dispatches kernels that do NOT exist in the compiled MSL library \
         (the runtime would silently treat them as capability-absent and fall back): {missing:?}"
    );
}
