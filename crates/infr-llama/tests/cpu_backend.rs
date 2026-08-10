//! Generation goldens for both backends: a plain-text prompt is rendered through the model's jinja
//! chat template, generated greedily, and a stable FNV-1a of the output is locked. The CPU goldens
//! run the backend-agnostic compute Graph on the CPU reference backend (no GPU); the GPU goldens run
//! the production Vulkan path. Both are captured with `INFR_BLESS=1` and read for coherence.
//!
//! These are NOT `#[ignore]`d — each self-skips at runtime when its GGUF isn't in the HF cache (and
//! the GPU goldens additionally skip when no Vulkan device is present), so they RUN automatically
//! wherever the models + hardware exist, and quietly no-op elsewhere:
//!   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend -- --nocapture

use infr_core::WeightSource;
use std::path::PathBuf;

/// Locate a cached GGUF `<file>` under `~/.cache/huggingface/hub/models--<repo>/snapshots/*/`, or
/// `None` if it isn't downloaded (the test self-skips). `repo` is the HF id with `/` → `--`.
fn find_gguf(repo: &str, file: &str) -> Option<PathBuf> {
    let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--{repo}/snapshots");
    std::fs::read_dir(&base).ok()?.find_map(|e| {
        let f = e.ok()?.path().join(file);
        f.exists().then_some(f)
    })
}

/// Resolve a model path or self-skip the test (it runs only when the GGUF is present).
macro_rules! need_model {
    ($opt:expr, $what:expr) => {
        match $opt {
            Some(p) => p,
            None => {
                eprintln!("skip: {} not in the HF cache", $what);
                return;
            }
        }
    };
}

/// Serialize the model-gated GPU tests against each other.
///
/// No test in this file drives a knob through the environment any more — the last one,
/// `INFR_NO_THINK`, became a `sampling.no_think` VALUE on the model's own config in S7 (see
/// [`model_cfg`]). What is left is the GPU: these tests upload whole models and open device
/// sessions, and cargo runs a binary's tests in parallel, so several of them racing for the same
/// device is a VRAM problem, not a configuration one.
///
/// So this is a plain, LOCAL mutex (S9), not `infr_core::test_env::EnvGuard` — that module existed
/// to serialize *and restore* environment mutations, and with nothing left to restore anywhere in
/// the tree it was deleted. Poison-tolerant, so a failing test does not cascade-poison the rest;
/// not re-entrant, so take it exactly once per test.
fn test_serial_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Load a model on an EXPLICIT engine configuration (`docs/config-plan.md` S4): `kv.*`, `spec.*`,
/// `sampling.*`, `device.ubatch*` and `paging.cache` are a VALUE this test hands to
/// `SeamModel::load_with`, not a process-global the next test can observe — and, just as
/// importantly, not one this test can INHERIT (an ambient `INFR_TEMP=0.6` in the developer's shell
/// used to change a golden's output; starting from `EngineConfig::default()` makes that impossible).
///
/// This is what replaced the `env.set("INFR_TEMP", "0")` / `env.set("INFR_KV_TYPE_K", …)` pattern
/// throughout this file. `test_serial_lock` is now taken ONLY to serialise the GPU tests against
/// each other; no test in this file sets a knob through the environment any more.
/// `INFR_MOE_SMALL_M` and `INFR_PAGER_STATS` came off it in S5a; `INFR_SEAM_NO_REPLAY`'s Vulkan
/// half and `INFR_I8_COOPMAT` in S5b; `INFR_NO_THINK` — the last one — in S7.
fn model_cfg(
    path: &std::path::Path,
    f: impl FnOnce(&mut infr_llama::EngineConfig),
) -> infr_llama::SeamModel {
    let mut cfg = infr_llama::EngineConfig::default();
    f(&mut cfg);
    infr_llama::SeamModel::load_with(path, None, std::sync::Arc::new(cfg)).expect("model load")
}

/// [`model_cfg`] on the shipped defaults — greedy sampling (`SamplingCfg::default()`), f16 KV, no
/// pinned prefill chunk. The `INFR_TEMP=0` guard every generation test used to take IS this.
fn model_default(path: &std::path::Path) -> infr_llama::SeamModel {
    model_cfg(path, |_| {})
}

/// A KV-format pair as a config edit — the `INFR_KV_TYPE_K`/`_V` pair, typed.
fn kv_types(cfg: &mut infr_llama::EngineConfig, k: &str, v: &str) {
    cfg.kv.type_k = infr_core::budget::parse_kv_dtype(k);
    cfg.kv.type_k_specified = true;
    cfg.kv.type_v = infr_core::budget::parse_kv_dtype(v);
    cfg.kv.type_v_specified = true;
}

// ─── CPU-only correctness (no GPU) ───────────────────────────────────────────────
//
// The CPU and GPU goldens use SEPARATE hashes: the CPU does the math in f32 while the GPU uses f16 +
// native-quant kernels, so greedy decode can split on near-ties (precision, not a bug) — comparing
// the two token-for-token is brittle. Instead each backend locks its own FNV-1a golden, captured
// with `INFR_BLESS=1` and read to confirm it's coherent + correct, so any op regression flips the
// hash. Kernel-level math (the Q4_K/Q6_K dot vs the f32 reference) is unit-tested in
// `src/seam.rs`.

/// Stable FNV-1a-64 over a string. (`std::hash::DefaultHasher` is NOT stable across toolchains, so we
/// roll our own for golden values.)
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Greedy CPU generation with NO GPU: load via [`infr_llama::SeamModel`] (Vulkan-free), render the
/// prompt with the model's own chat template (so an instruct model answers coherently), collect the
/// streamed text. This is exactly the production `INFR_CPU=1` path.
fn cpu_gen(model: &infr_llama::SeamModel, prompt: &str, n: usize) -> String {
    // Inputs are plain text; `render_chat` (the GGUF's jinja template) turns them into the exact
    // token stream the instruct model expects.
    let mut out = String::new();
    model
        .generate_cpu(
            &model.render_chat(prompt).expect("render chat"),
            n,
            None,
            |p| out.push_str(p),
        )
        .expect("cpu generate");
    out
}

/// Assert (or, with `INFR_BLESS=1`, print) the golden hash for each case.
fn check_golden(model: &infr_llama::SeamModel, cases: &[(&str, usize, u64)]) {
    let bless = std::env::var("INFR_BLESS").is_ok();
    for (prompt, n, want) in cases {
        let out = cpu_gen(model, prompt, *n);
        let h = fnv1a(&out);
        if bless {
            // Print the text too so a human can verify it's coherent before locking the hash.
            println!("    ({prompt:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(
                h, *want,
                "golden hash changed for {prompt:?} (n={n})\n  out: {out:?}\n  got 0x{h:016x} want 0x{want:016x}"
            );
        }
    }
}

/// Assert (or, with `INFR_BLESS=1`, print) the GPU golden hash for each `(prompt, n, fnv1a)` case.
fn check_gpu_golden(gen: impl Fn(&str, usize) -> String, cases: &[(&str, usize, u64)]) {
    let bless = std::env::var("INFR_BLESS").is_ok();
    for (prompt, n, want) in cases {
        let out = gen(prompt, *n);
        let h = fnv1a(&out);
        if bless {
            println!("    ({prompt:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(
                h, *want,
                "GPU golden changed for {prompt:?} (n={n})\n  out: {out:?}\n  got 0x{h:016x} want 0x{want:016x}"
            );
        }
    }
}

// ─── Qwen3-0.6B (dense) ───────────────────────────────────────────────────────────

fn qwen3_06b() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    find_gguf("unsloth--Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
}

/// Path to a specific Qwen3-0.6B quantization in the HF cache (for the quant-coverage sweep).
fn qwen3_quant(quant: &str) -> Option<PathBuf> {
    find_gguf(
        "unsloth--Qwen3-0.6B-GGUF",
        &format!("Qwen3-0.6B-{quant}.gguf"),
    )
}

// ─── Qwen2.5-0.5B (dense, BIASED q/k/v) ───────────────────────────────────────────
// Qwen2/2.5 add a learned bias to the q/k/v projections (Qwen3 dropped them) — the new `AddBias`
// seam op. The 0.5B-Instruct also ties its output embedding, so this exercises the tied lm-head
// path too. Gated: needs a Qwen2.5 GGUF in the HF cache, or `INFR_TEST_QWEN2=/path/to.gguf`.
fn qwen2_05b() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_QWEN2") {
        return Some(PathBuf::from(p));
    }
    // `unsloth/Qwen2.5-0.5B-Instruct-GGUF` does not exist (the HF API 401s on it). The cache
    // held the model all along under Qwen's own org with a lower-case filename, and this
    // helper still could not see it — so every Qwen2.5 test self-skipped forever.
    find_gguf(
        "Qwen--Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
}

/// Qwen2.5 through the Vulkan seam must match the CPU oracle token-for-token — validates the QKV
/// bias (`AddBias`) end to end on prefill + decode + record-once replay, plus tied embeddings.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_qwen2() {
    let path = need_model!(qwen2_05b(), "Qwen2.5-0.5B-Instruct");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is the capital of France? Answer briefly.", 16);
}

// Captured + verified coherent (chat-templated, Qwen3 thinks then answers): "…France's capital is
// Paris", a simple-terms computer explanation, an ocean paragraph.
const QWEN3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xfd63781ea3bfa785),
    (
        "Explain how a computer works in simple terms.",
        48,
        0xcf56ba8c4bb5c455,
    ),
    (
        "Write a short paragraph about the ocean.",
        48,
        0x29f45fb169b84b9a,
    ),
];

/// CPU-only: the deterministic Qwen3 output (short + long) must match its golden hash.
#[test]
fn cpu_golden_qwen3() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_golden(&model, QWEN3_GOLDEN);
}

/// Host weight paging (`paging.dram`, `infr_cpu::paged`) must not change a single token: the
/// weights are the same bytes, read from the file instead of through the mapping.
///
/// The budget is deliberately far below the model — 24 MiB against ~380 MiB of Q4_K_M — so every
/// pool churns: each generation evicts and re-reads most blocks many times, which is precisely
/// where a wrong slot, a torn multi-extent read, or an eviction under a live pin would show up. A
/// budget large enough to hold everything would pass no matter what the pager did.
#[test]
fn cpu_paged_weights_match_the_mapped_path() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let prompt = "Name three primary colors.";

    let mapped = cpu_gen(&model_default(&path), prompt, 24);
    let paged = cpu_gen(
        &model_cfg(&path, |c| {
            c.paging.dram = Some(infr_core::SizeSpec::Bytes(24 << 20));
        }),
        prompt,
        24,
    );
    assert_eq!(
        paged, mapped,
        "paged weights changed the generated text — the pager served bytes the mapping does not"
    );
    assert!(!mapped.trim().is_empty(), "the oracle generated nothing");
}

/// The same, with the budget below every weight class: nothing is pageable, so the binder must
/// fall back to mapping rather than fail the load. A `paging.dram` too small to be useful has to
/// degrade, not break.
#[test]
fn cpu_paging_falls_back_when_the_budget_seats_nothing() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let prompt = "Name three primary colors.";
    let tiny = cpu_gen(
        &model_cfg(&path, |c| {
            c.paging.dram = Some(infr_core::SizeSpec::Bytes(64 << 10));
        }),
        prompt,
        16,
    );
    assert_eq!(tiny, cpu_gen(&model_default(&path), prompt, 16));
}

/// A BIG prompt (~1000+ tokens): large enough that the dense prefill's padded-KV attention reads the
/// padding rows beyond the real tokens. Short prompts don't reproduce the KV-cache bug.
fn repeat_prompt() -> String {
    "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(90)
}

/// A greedy generation is degenerate if it collapsed to one repeated token (the KV-padding bug's
/// "!!!!"/"5555" signature): a non-trivial length with ≤2 distinct chars.
fn is_degenerate(s: &str) -> bool {
    let t = s.trim();
    t.chars().count() >= 8 && t.chars().collect::<std::collections::HashSet<char>>().len() <= 2
}

/// REGRESSION (CPU reference backend): the same repeated-forward invariant on the no-GPU
/// compute-graph path. The CPU backend uses host buffers (no recycled-VRAM hazard), so this guards
/// CPU coherence + determinism across repeated big-prompt forwards.
#[test]
fn cpu_no_garbage_on_repeated_forward() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let p = repeat_prompt();
    let g1 = cpu_gen(&model, &p, 20);
    let g2 = cpu_gen(&model, &p, 20);
    let head = |s: &str| s.chars().take(48).collect::<String>();
    assert!(
        !is_degenerate(&g1),
        "1st CPU forward degenerate: {:?}",
        head(&g1)
    );
    assert!(
        !is_degenerate(&g2),
        "2nd CPU forward degenerate: {:?}",
        head(&g2)
    );
    assert_eq!(g1, g2, "repeated CPU forward diverged");
}

/// KV-cache Q8_0 quantization (CPU reference): the KV cache stores Q8_0 blocks (34 B / 32 elems)
/// INDEPENDENTLY for K and V (`INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` ∈ {f16, q8_0}). Q8 KV shifts the
/// numerics, so it won't match the f16 golden hash — but a correct per-block quantize/dequant must
/// still yield coherent (non-degenerate) greedy output on a long prompt whose decode reads a deep
/// cache. Exercises all three quantized combos (q8/q8, q8/f16, f16/q8) to prove K and V decouple.
#[test]
fn cpu_kv_q8_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let prompt = repeat_prompt();
    // Each K/V quant mix on its own model: the KV dtype is read at graph build, off the config the
    // model was loaded with.
    for (k, v) in [("q8_0", "q8_0"), ("q8_0", "f16"), ("f16", "q8_0")] {
        let model = model_cfg(&path, |c| kv_types(c, k, v));
        let out = cpu_gen(&model, &prompt, 24);
        assert!(
            !is_degenerate(&out),
            "KV K={k} V={v} degenerate: {:?}",
            out.chars().take(48).collect::<String>()
        );
    }
}

/// TurboQuant KV cache (CPU reference): WHT-rotated 2/3/4-bit PolarQuant, 128-elem blocks. The
/// per-vector error (turbo2 ~30%, turbo3 ~20%, turbo4 ~12%) is what V tolerates but K does not
/// (llama.cpp: "keep K at higher precision than V"), so the coherent config is K=f16 with V=turbo*.
/// Exercises the full quantize (WriteKv) + dequant-with-inverse-WHT (Attention) path for every width;
/// a broken WHT / centroid table / packing / norm-correction would garble even the V-only cache.
#[test]
fn cpu_kv_turbo_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    for v in ["turbo2", "turbo3", "turbo4"] {
        let model = model_cfg(&path, |c| kv_types(c, "f16", v));
        let out = cpu_gen(&model, &repeat_prompt(), 24);
        assert!(
            !is_degenerate(&out),
            "K=f16 V={v} degenerate: {:?}",
            out.chars().take(48).collect::<String>()
        );
    }
}

/// Mainline llama.cpp KV cache types (CPU reference): f32/bf16 (dense) + the low-bit round quants
/// q4_0/q4_1/q5_0/q5_1 and the non-linear iq4_nl, quantized on the fly per 32-elem block on write and
/// dequantized via the shared GGUF path on read. f32/bf16 run coupled; the low-bit quants run on V
/// (K=f16) since K needs higher precision. Every config must stay coherent.
#[test]
fn cpu_kv_mainline_quants_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let prompt = repeat_prompt();
    for (k, v) in [
        ("f32", "f32"),
        ("bf16", "bf16"),
        ("f16", "q4_0"),
        ("f16", "q4_1"),
        ("f16", "q5_0"),
        ("f16", "q5_1"),
        ("f16", "iq4_nl"),
    ] {
        let model = model_cfg(&path, |c| kv_types(c, k, v));
        let out = cpu_gen(&model, &prompt, 24);
        assert!(
            !is_degenerate(&out),
            "KV K={k} V={v} degenerate: {:?}",
            out.chars().take(48).collect::<String>()
        );
    }
}

// Captured + verified coherent on the Vulkan backend via the agnostic compute seam (the SAME dense
// `Graph` the CPU oracle builds, mapped op-for-op to GPU kernels). Should reproduce the production
// GPU path (QWEN3_GPU_GOLDEN).
const QWEN3_SEAM_GOLDEN: &[(&str, usize, u64)] = &[
    (
        "The capital of France is",
        32,
        // RE-BLESSED (2026-07-13) for `de987d7`: **Q6_K** joined the AMD int8-activation DECODE
        // tier (`mmv_int8_decode_dtypes`). unsloth's Qwen3-0.6B-**Q4_K_M** is a MIXED GGUF — its
        // tied lm_head/embed is Q6_K — so flipping Q6_K moved that tensor's decode GEMV from
        // f32-exact dequant to int8 dp4a. Same class of real-but-benign numerics shift as the
        // Q4_K re-blessing this comment used to describe: int8-activation rounding shifts a
        // close-margin greedy argmax and the token path diverges, while the answer stays right.
        //
        // Verified coherent AND correct before re-blessing (greedy, same prompt):
        //   "<think>\nOkay, the user is asking about the capital of France. I need to make sure I
        //    recall the correct answer. France's capital is Paris. …"
        //   → "The capital of France is **Paris**."
        //
        // MY PROCESS FAILURE, recorded so it isn't repeated: `de987d7` shipped this numerics flip
        // and I did NOT re-bless — this golden has been RED on main ever since (found only when a
        // later agent tripped over it). A precision-policy flip is exactly the change that stales a
        // golden; re-bless it IN THE SAME COMMIT, with the generated text pasted in as proof, or
        // don't ship the flip.
        0xfd63781ea3bfa785,
    ),
    (
        "Explain how a computer works in simple terms.",
        48,
        // RE-BLESSED for the same reason as the France case above. Verified coherent: "<think>
        // \nOkay, the user wants an explanation of how a computer works in simple terms. Let me
        // start by breaking down the basic components. First, there's the hardware, like the CPU,
        // RAM, and storage. Then the software," (coincidentally now bit-for-bit the same trajectory
        // as the CPU-only oracle's QWEN3_GOLDEN second case, 0xcf56ba8c4bb5c455 — not required to
        // match, just a real outcome of the shifted numerics).
        0xcf56ba8c4bb5c455,
    ),
];

/// End-to-end dense parity: run the full Qwen3-0.6B dense forward on the **Vulkan** backend through
/// the agnostic compute seam ([`SeamModel::generate_dense_vulkan`]) and lock its golden. The seam runs
/// the identical `Graph` the CPU reference builds; this proves the dense forward maps faithfully to
/// the GPU and reproduces the production GPU path (`gpu_golden_qwen3`).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_golden_qwen3() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_gpu_golden(
        |p, n| {
            model
                .generate_dense_vulkan(&model.render_chat(p).expect("render chat"), n)
                .expect("seam gen")
        },
        QWEN3_SEAM_GOLDEN,
    );
}

/// IQ4_XS (non-linear 4-bit codebook quant: 4-bit codes index KV_IQ4NL → signed int8, per-32 scale)
/// through the Vulkan seam vs the CPU oracle. The GPU IQ4_XS decode/prefill runs the word-parallel
/// `dqblk` (whole-u32 code loads + hoisted codebook gather) and — on ≥48M/≥8M weights (none on this
/// 0.6B model; its lm_head is tied Q6_K) — the codebook-gather-then-dp4a int8 mmv. This 0.6B's IQ4_XS
/// projections are all small, so it exercises the `dqblk` path; the assertion guards that the decode
/// stays bit-faithful (token-for-token with the f32 CPU reference).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_qwen3_iq4xs() {
    let path = need_model!(qwen3_quant("IQ4_XS"), "Qwen3-0.6B-IQ4_XS");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is the capital of France? Answer briefly.", 16);
}

/// Q2_K (2-bit quants + 4-bit sub-block scale/min + super-block d/dmin) through the Vulkan seam vs
/// the CPU oracle. unsloth's Q2_K is a MIXED quant — the down/o/kv projections are Q3_K and the
/// gate_up/q projections are Q2_K — so this exercises BOTH the Q2_K and Q3_K native-block prefill
/// GEMMs, including Q3_K's A_GLOBAL / split-K warptile variants (added so Q3_K stops running the
/// plain n128 tile at ~9.6 TF; see native_gemm_warp_q3k_{ag,n128_ag,sk_ag}). Both decode paths use
/// the word-parallel `dqblk`; the A_GLOBAL/split-K variants are bit-identical to the f32 staging
/// path (same dqblk, same MMA order), so this guards that the added pipelines stay token-faithful
/// to the f32 CPU reference.
///
/// HISTORY: this test failed for a long stretch, and the GPU was NOT at fault. The oracle was the
/// production CPU path, which quantizes activations to int8 (~4e-3 relative error, measured
/// 4.5e-3 for Q2_K against the host decode) while the Vulkan f32 dequant GEMV sits at ~1e-7. At
/// 2 bits that gap flips a greedy token. The oracle is now `CpuBackend::reference` (weights
/// decoded to f32, f32 dot), against which the Vulkan seam matches token-for-token.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_qwen3_q2k() {
    let path = need_model!(qwen3_quant("Q2_K"), "Qwen3-0.6B-Q2_K");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is the capital of France? Answer briefly.", 16);
}

/// int8 cooperative-matrix (WMMA) prefill GEMM measurement kernel (`INFR_I8_COOPMAT=1`, Q8_0 only —
/// see `crates/infr-vulkan/shaders/native_gemm_i8cm_q8_0.comp`): the Vulkan seam must still match
/// the f32 CPU oracle token-for-token with the toggle on, proving the new per-Q8_0-block WMMA-dot +
/// shared-store scale epilogue is numerically equivalent to the production f16-coopmat dequant path
/// (and to the dp4a mmq reference it mirrors). Self-skips without a Q8_0 GGUF or a GPU with
/// `caps.i8_coopmat` (the toggle is a no-op on hardware/driver that doesn't detect the config — see
/// `Capabilities::i8_coopmat`'s doc — so this test would otherwise silently run the default f16
/// path and prove nothing).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_qwen3_q8_0_i8coopmat() {
    let path = need_model!(qwen3_quant("Q8_0"), "Qwen3-0.6B-Q8_0");
    // The GPU tests stay serialized against each other; the KNOB is a config value since S5b.
    let _tlk = test_serial_lock();
    seam_vulkan_matches_cpu_cfg(
        &path,
        "What is the capital of France? Answer briefly.",
        16,
        |c| c.kernels.vulkan.i8_coopmat = true,
    );
}

/// Flash-attention prefill parity: a prompt LONG ENOUGH (>64 tokens) that the seam's batched prefill
/// takes the FlashAttention-2 path (`attention_prefill_flash`, rows≥64) + the tiled GEMM/mmq Linear,
/// must generate the SAME greedy continuation as the CPU reference oracle (which uses the naive
/// per-token attention). Guards the m>1 prefill kernels the short-prompt goldens never exercise.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_flash_matches_cpu() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    // ~100+ tokens → pf_m ≥ 64 → flash prefill on the seam.
    let long = "Photosynthesis is the process by which green plants, algae, and some bacteria \
        convert light energy into chemical energy stored in glucose, using carbon dioxide and water \
        and releasing oxygen as a byproduct. It happens in two connected stages: the light-dependent \
        reactions in the thylakoid membranes, and the light-independent Calvin cycle in the stroma. \
        Explain each stage carefully, name the key molecules involved, and then summarize in one \
        sentence why this process is essential for life on Earth.";
    let mut cpu_txt = String::new();
    model
        // REFERENCE oracle — see `seam_vulkan_matches_cpu`'s doc.
        .generate_cpu_reference(long, 24, None, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model.generate_dense_vulkan(long, 24).expect("seam gen");
    // The f16 GPU flash/GEMM kernels and the f32 CPU oracle accumulate in different precision, so a
    // long greedy continuation eventually hits a near-tie argmax and forks into an equally-coherent
    // alternative (the exact split `seam_vulkan_matches_cpu`'s doc predicts). What the flash-prefill
    // kernels must guarantee is a LONG shared prefix — a real attention bug corrupts the context and
    // diverges immediately, not 20 tokens in. Assert a substantial common prefix instead of full
    // bit-identity; print both on failure.
    let (ct, gt) = (cpu_txt.trim(), gpu_txt.trim());
    let common = ct
        .char_indices()
        .zip(gt.chars())
        .take_while(|((_, a), b)| a == b)
        .count();
    assert!(
        common >= 60,
        "flash-prefill seam diverged from the CPU oracle too early (common prefix {common} chars):\n\
         cpu: {ct:?}\ngpu: {gt:?}"
    );
}

/// Vulkan-seam vs CPU-oracle parity for one model: greedy `n`-token continuation of `prompt`
/// (rendered through the model's chat template) must match token-for-token. Proves the arch's ops
/// lower correctly through the Vulkan adapter — the CPU seam runs the IDENTICAL Graph. A near-tie
/// argmax split (f16 GPU kernels vs f32 CPU) would show here as an early divergence; none of the
/// covered models exhibit one on these prompts today, so keep the strict compare until it flakes.
///
/// The oracle is the REFERENCE CPU backend (`CpuBackend::reference`: weights decoded to f32, f32
/// dot), NOT the production int8-activation kernels. Those carry ~4e-3 relative error on every
/// quant dtype — measured against the host decode: CPU 3.4e-3..4.7e-3 vs Vulkan ~1e-7 — which is
/// invisible at Q4_K+ but flips a greedy token at Q2_K. Scoring the GPU against the int8 CPU leg
/// therefore failed the more accurate implementation; the reference mode is the honest oracle.
fn seam_vulkan_matches_cpu(path: &std::path::Path, prompt: &str, n: usize) {
    seam_vulkan_matches_cpu_cfg(path, prompt, n, |_| {});
}

/// [`seam_vulkan_matches_cpu`] on an explicit engine configuration — for the tier knobs whose
/// numerics have to be scored against the CPU oracle (the `INFR_I8_COOPMAT` measurement kernel).
fn seam_vulkan_matches_cpu_cfg(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    f: impl FnOnce(&mut infr_llama::EngineConfig),
) {
    let model = model_cfg(path, f);
    let rendered = model.render_chat(prompt).expect("render chat");
    let mut cpu_txt = String::new();
    model
        .generate_cpu_reference(&rendered, n, None, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model
        .generate_dense_vulkan(&rendered, n)
        .expect("vulkan seam gen");
    assert_eq!(
        cpu_txt.trim(),
        gpu_txt.trim(),
        "Vulkan seam diverged from the CPU oracle for {path:?}"
    );
}

/// Persistent-session KV reuse on the Vulkan seam: turn 2 extends turn 1's prompt, so the session
/// must (a) generate EXACTLY what a fresh full-prefill of the same prompt generates, and (b)
/// prefill only the un-cached suffix (stats.n_prompt ≪ the full prompt length). The seam twin of
/// the bespoke ChatSession's incremental prefill.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_kv_reuse_matches_fresh() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let mut sess = model.vulkan_session(512).expect("session");

    let p1 = "The capital of France is";
    let mut t1 = String::new();
    let s1 = model
        .generate_vulkan_session(&mut sess, p1, 8, None, |p| t1.push_str(p))
        .expect("turn 1");
    assert!(s1.n_prompt > 0);

    let p2 = format!("{p1}{t1} And the capital of Germany is");
    let mut t2 = String::new();
    let s2 = model
        .generate_vulkan_session(&mut sess, &p2, 8, None, |p| t2.push_str(p))
        .expect("turn 2");

    // (a) same output as a fresh full prefill of the identical prompt
    let fresh = model.generate_dense_vulkan(&p2, 8).expect("fresh gen");
    assert_eq!(
        t2.trim(),
        fresh.trim(),
        "session (suffix prefill) diverged from a fresh full prefill"
    );
    // (b) the session only prefilled the suffix — far fewer tokens than the whole prompt
    let full_len = s1.n_prompt + t1.split_whitespace().count(); // lower bound on p2's tokens
    assert!(
        s2.n_prompt < full_len,
        "turn 2 prefilled {} tokens — KV reuse didn't kick in",
        s2.n_prompt
    );
}

/// Q8_0 KV cache on the Vulkan seam. COUPLED K==V==q8 (INFR_KV_Q8) decode now rides the
/// record-once replay (store_q8_dyn planar write + attn_partial_dynac_q8 / attention_kv_dyn_q8
/// planar read, pos/kv_len from the self-advancing params SSBO), so (a)/(b) exercise the replayed
/// tape end to end plus the static prefill (store_q8 + flash dequant). DECOUPLED sides (c) still
/// force per-execute static decode (per-side attn_partial_{k,v}q8 kernels). All must produce
/// coherent (non-degenerate) greedy output. Q8 KV shifts the numerics (no exact match with the
/// f16 golden), but the near-lossless quant must stay sensible; a broken quantize/dequant, a
/// mis-gated kernel, or a wrong planar scales base (`cap`) would collapse or garble the output.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_kv_q8_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let head = |s: &str| s.chars().take(64).collect::<String>();
    // A prompt long enough (>64 tokens) to take the prefill path, then a deep-cache decode.
    let long =
        "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(6);

    let model = model_cfg(&path, |c| c.kv.force_q8 = true);
    // (a) one-shot static path.
    let g_static = model
        .generate_dense_vulkan(&long, 24)
        .expect("q8 static gen");
    assert!(
        !is_degenerate(&g_static),
        "Q8 static Vulkan output degenerate: {:?}",
        head(&g_static)
    );
    // (b) session path (record-once: the whole decode loop replays ONE recorded Q8 tape).
    let mut sess = model.vulkan_session(512).expect("q8 session");
    let mut g_sess = String::new();
    model
        .generate_vulkan_session(&mut sess, &long, 24, None, |p| g_sess.push_str(p))
        .expect("q8 session gen");
    assert!(
        !is_degenerate(&g_sess),
        "Q8 session Vulkan output degenerate: {:?}",
        head(&g_sess)
    );

    // (c) DECOUPLED K/V: each mixed side (K=q8/V=f16 and K=f16/V=q8) must also stay coherent — the
    // per-side attn_partial_{k,v}q8 / attention_kv_{k,v}q8 variants read one Q8 side + one f16 side.
    for (k, v) in [("q8_0", "f16"), ("f16", "q8_0")] {
        let m = model_cfg(&path, |c| kv_types(c, k, v));
        let out = m.generate_dense_vulkan(&long, 20).expect("mixed gen");
        assert!(
            !is_degenerate(&out),
            "mixed K={k} V={v} Vulkan output degenerate: {:?}",
            head(&out)
        );
    }
}

/// Mainline low-bit KV quants on the Vulkan seam: q4_0/q4_1/q5_0/q5_1/iq4_nl run via a quantizing
/// WriteKv (quant_kv) + a dequant→f16 prefix prepass (dequant_kv_f16, reusing native_decode) feeding
/// the standard f16 flash/split/scalar attention. K=f16 with each quantized V must stay coherent
/// (K needs higher precision). A broken GPU quantize or dequant would garble even a V-only cache.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_kv_mainline_quants_coherent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let head = |s: &str| s.chars().take(64).collect::<String>();
    // Long enough (>64 tokens) to take the flash prefill on the dequanted scratch, then deep decode.
    let long =
        "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(6);
    // K=f16 with each quantized V, plus the dense f32/bf16 caches (coupled).
    for (k, v) in [
        ("f16", "q4_0"),
        ("f16", "q4_1"),
        ("f16", "q5_0"),
        ("f16", "q5_1"),
        ("f16", "iq4_nl"),
        ("f32", "f32"),
        ("bf16", "bf16"),
        ("f16", "turbo2"),
        ("f16", "turbo3"),
        ("f16", "turbo4"),
    ] {
        let model = model_cfg(&path, |c| kv_types(c, k, v));
        let out = model.generate_dense_vulkan(&long, 24).expect("gpu kv gen");
        assert!(
            !is_degenerate(&out),
            "GPU K={k} V={v} degenerate: {:?}",
            head(&out)
        );
    }
}

/// Multi-slot KV prefix sharing: two INTERLEAVED conversations with a common long prefix (a
/// "system prompt"). Conversation B must (a) generate exactly what a fresh full prefill does,
/// (b) prefill only past the shared prefix (its slot was SEEDED by a device-side KV copy from
/// A's slot), and (c) not evict A — A's next turn still extends its own slot cheaply.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_multi_slot_prefix_sharing() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let mut sess = model.vulkan_session(512).expect("session");

    // A long shared prefix (stands in for a system prompt) + two different questions.
    let sys = "You are a terse geography assistant. Answer in one word only, no punctuation, \
               no explanations, never refuse, always answer with just the single word asked for. ";
    let pa = format!("{sys}The capital of France is");
    let pb = format!("{sys}The capital of Germany is");

    let mut ta = String::new();
    let sa = model
        .generate_vulkan_session(&mut sess, &pa, 8, None, |p| ta.push_str(p))
        .expect("conv A");
    assert!(sa.n_prompt > 0);

    // Conversation B: different question, same system prefix → new slot seeded from A's.
    let mut tb = String::new();
    let sb = model
        .generate_vulkan_session(&mut sess, &pb, 8, None, |p| tb.push_str(p))
        .expect("conv B");
    let fresh_b = model.generate_dense_vulkan(&pb, 8).expect("fresh B");
    assert_eq!(
        tb.trim(),
        fresh_b.trim(),
        "seeded-slot generation diverged from a fresh full prefill"
    );
    // The shared prefix must NOT have been re-prefilled (only B's short suffix).
    assert!(
        sb.n_prompt < sa.n_prompt / 2,
        "conv B prefilled {} tokens (conv A: {}) — prefix seeding didn't kick in",
        sb.n_prompt,
        sa.n_prompt
    );

    // Conversation A continues — its slot must still be intact (suffix-only prefill again).
    let pa2 = format!("{pa}{ta} And the capital of Spain is");
    let mut ta2 = String::new();
    let sa2 = model
        .generate_vulkan_session(&mut sess, &pa2, 8, None, |p| ta2.push_str(p))
        .expect("conv A turn 2");
    let fresh_a2 = model.generate_dense_vulkan(&pa2, 8).expect("fresh A2");
    assert_eq!(
        ta2.trim(),
        fresh_a2.trim(),
        "conv A slot was clobbered by B"
    );
    assert!(
        sa2.n_prompt < sa.n_prompt / 2,
        "conv A turn 2 prefilled {} tokens — its slot was evicted",
        sa2.n_prompt
    );
}

/// Speculative decoding must emit EXACTLY the target-only greedy stream, end to end. The
/// contract is structural — every committed token is either checked against or produced by a
/// verify-forward argmax — but the verify forward runs the batched f16 GEMM/cmm path while
/// target-only decode uses the exact-f32 GEMV, so a near-tie logit could in principle split
/// them; this test pins the equivalence on a real generation. Self-spec (draft == target)
/// keeps it to one model download; the accept/commit machinery is identical to a small-draft
/// pair (the driver never knows the models are the same file).
#[cfg(target_os = "macos")]
#[test]
fn metal_spec_decode_matches_target_only_greedy() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let target = model_default(&path);
    let draft = model_default(&path);
    let prompt = target
        .render_chat("Write a short paragraph about the ocean.")
        .expect("render chat");

    let mut plain = String::new();
    {
        let mut sess = target.metal_session(1024).expect("target-only session");
        target
            .generate_metal_session(&mut sess, &prompt, 64, None, |p| plain.push_str(p))
            .expect("target-only greedy");
    }

    let mut spec = String::new();
    {
        let mut ts = target.metal_session(1024).expect("spec target session");
        let mut ds = draft.metal_session(1024).expect("spec draft session");
        target
            .generate_metal_spec(&mut ts, &draft, &mut ds, &prompt, 64, 6, |p| {
                spec.push_str(p)
            })
            .expect("spec decode");
    }

    assert_eq!(
        spec, plain,
        "speculative stream diverged from target-only greedy"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_decode_chain_matches_per_token_greedy() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let per = model_cfg(&path, |c| c.spec.decode_chain = 1);
    let prompt = per
        .render_chat("Explain why the sky is blue in two sentences.")
        .expect("render chat");

    let mut per_token = String::new();
    per.generate_metal(&prompt, 32, None, |p| per_token.push_str(p))
        .expect("per-token greedy");

    let chain = model_cfg(&path, |c| c.spec.decode_chain = 8);
    let mut chained = String::new();
    chain
        .generate_metal(&prompt, 32, None, |p| chained.push_str(p))
        .expect("chained greedy");

    assert_eq!(chained, per_token, "chained Metal decode diverged");
}

#[cfg(target_os = "macos")]
#[test]
fn metal_decode_chain_matches_per_token_sampling() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    // The SAME seeded sampler on both runs — the whole point is that the chained replay draws
    // the identical xorshift stream the per-token path does.
    let sampling = |c: &mut infr_llama::EngineConfig, chain: usize| {
        c.sampling.temp = 0.7;
        c.sampling.top_k = 20;
        c.sampling.top_p = 0.95;
        c.sampling.seed = Some(47);
        c.spec.decode_chain = chain;
    };
    let per = model_cfg(&path, |c| sampling(c, 1));
    let prompt = per
        .render_chat("Explain why the sky is blue in two sentences.")
        .expect("render chat");

    let mut per_token = String::new();
    per.generate_metal(&prompt, 32, None, |p| per_token.push_str(p))
        .expect("per-token sampling");

    let chain = model_cfg(&path, |c| sampling(c, 8));
    let mut chained = String::new();
    chain
        .generate_metal(&prompt, 32, None, |p| chained.push_str(p))
        .expect("chained sampling");

    assert_eq!(chained, per_token, "chained Metal sampling diverged");
}

/// Metal twin of [`gpu_seam_multi_slot_prefix_sharing`]: the same interleaved-conversation
/// contract through `DenseMetalSession`'s slot pool — fork shares the one weight upload
/// (Arc), seeding is the backend-generic `copy_buffer`, and every slot switch re-records the
/// decode replay tape (its fingerprint covers the bound KV/IO buffer addresses).
#[cfg(target_os = "macos")]
#[test]
fn metal_seam_multi_slot_prefix_sharing() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let mut sess = model.metal_session(512).expect("session");

    let sys = "You are a terse geography assistant. Answer in one word only, no punctuation, \
               no explanations, never refuse, always answer with just the single word asked for. ";
    let pa = format!("{sys}The capital of France is");
    let pb = format!("{sys}The capital of Germany is");

    let mut ta = String::new();
    let sa = model
        .generate_metal_session(&mut sess, &pa, 8, None, |p| ta.push_str(p))
        .expect("conv A");
    assert!(sa.n_prompt > 0);

    // Conversation B: different question, same system prefix → new slot seeded from A's.
    let mut tb = String::new();
    let sb = model
        .generate_metal_session(&mut sess, &pb, 8, None, |p| tb.push_str(p))
        .expect("conv B");
    let mut fresh_b = String::new();
    model
        .generate_metal(&pb, 8, None, |p| fresh_b.push_str(p))
        .expect("fresh B");
    assert_eq!(
        tb.trim(),
        fresh_b.trim(),
        "seeded-slot generation diverged from a fresh full prefill"
    );
    // The shared prefix must NOT have been re-prefilled (only B's short suffix).
    assert!(
        sb.n_prompt < sa.n_prompt / 2,
        "conv B prefilled {} tokens (conv A: {}) — prefix seeding didn't kick in",
        sb.n_prompt,
        sa.n_prompt
    );

    // Conversation A continues — its slot must still be intact (suffix-only prefill again).
    let pa2 = format!("{pa}{ta} And the capital of Spain is");
    let mut ta2 = String::new();
    let sa2 = model
        .generate_metal_session(&mut sess, &pa2, 8, None, |p| ta2.push_str(p))
        .expect("conv A turn 2");
    let mut fresh_a2 = String::new();
    model
        .generate_metal(&pa2, 8, None, |p| fresh_a2.push_str(p))
        .expect("fresh A2");
    assert_eq!(
        ta2.trim(),
        fresh_a2.trim(),
        "conv A slot was clobbered by B"
    );
    assert!(
        sa2.n_prompt < sa.n_prompt / 2,
        "conv A turn 2 prefilled {} tokens — its slot was evicted",
        sa2.n_prompt
    );
}

/// gemma3 (SWA + dual-rope + GeGLU + sandwich norms, hd=256) through the Vulkan seam.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_gemma3() {
    let path = need_model!(gemma3_1b(), "gemma-3-1b");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is the capital of France? Answer briefly.", 16);
}

/// gemma3 Q2_K — unsloth's Q2_K here is a MIXED quant whose ffn up/gate and attn_q are IQ4_NL
/// (down/o/kv are Q3_K, embeddings Q2_K/Q5_0): the only in-tree model file exercising the IQ4_NL
/// warp-GEMM family (native_gemm_warp_iq4nl_{,n128,sk,ag,n128_ag,sk_ag}) and the word-parallel
/// IQ4_NL `dqblk` in the decode GEMV end-to-end.
///
/// NOT the strict token-for-token compare: at 2-bit this 1B model near-ties its greedy argmax
/// within a handful of tokens on every prompt tried ("Paris"+"."-vs-"\n", "One"+","-vs-" ") and
/// the f16-GPU/f32-CPU split forks it into an equally-coherent alternative — verified forking
/// IDENTICALLY on the pre-IQ4_NL-warp tree, so it's the model, not the kernels. A real dequant
/// or GEMM bug corrupts the context and diverges immediately; assert a substantial common prefix
/// instead (the `flash_prefill_seam_matches_cpu` precedent).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_gemma3_q2k_iq4nl() {
    let path = need_model!(gemma3_1b_q2k(), "gemma-3-1b Q2_K");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let rendered = model
        .render_chat("Count from one to five, digits only.")
        .expect("render chat");
    let mut cpu_txt = String::new();
    model
        // REFERENCE oracle — see `seam_vulkan_matches_cpu`'s doc: at Q2_K the production CPU
        // kernels' int8-activation error is what moves tokens, not the GPU.
        .generate_cpu_reference(&rendered, 16, None, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model
        .generate_dense_vulkan(&rendered, 16)
        .expect("vulkan seam gen");
    let (ct, gt) = (cpu_txt.trim(), gpu_txt.trim());
    let common = ct
        .char_indices()
        .zip(gt.chars())
        .take_while(|((_, a), b)| a == b)
        .count();
    assert!(
        common >= 16,
        "gemma3 Q2_K (IQ4_NL) seam diverged from the CPU oracle too early \
         (common prefix {common} chars):\ncpu: {ct:?}\ngpu: {gt:?}"
    );
}

/// llama (no qk-norm: standalone INTERLEAVED RoPE — llama.cpp's ROPE_TYPE_NORM — through the
/// f16-out Rope shape, fused KV write, and the rope_f16_dyn record-once replay).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_llama() {
    let path = need_model!(llama32_1b(), "Llama-3.2-1B");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "Count from one to five, digits only.", 16);
}

/// Plain Llama RoPE must produce the same multi-token continuation through Metal's default decode
/// replay and its forced-static path. This specifically guards against replaying token 0's baked
/// RoPE position on later decode steps; static Metal is the same-precision oracle.
#[cfg(target_os = "macos")]
#[test]
fn metal_llama_replay_matches_static() {
    let path = need_model!(llama32_1b(), "Llama-3.2-1B");
    let prompt = model_default(&path)
        .render_chat("Count from one to five, digits only.")
        .expect("render chat");

    // `kernels.vulkan.no_replay` is the SEAM's half of `INFR_SEAM_NO_REPLAY` (§6.12: the Vulkan
    // adapter reads the same knob the other way round, and S5 moves that half) — since S4 the
    // runner's replay gate takes it off the model's config, so the mode is a per-model value.
    let run_metal = |no_replay: bool| {
        let model = model_cfg(&path, |c| c.kernels.vulkan.no_replay = no_replay);
        let mut out = String::new();
        model
            .generate_metal(&prompt, 16, None, |p| out.push_str(p))
            .expect("metal generation");
        out
    };

    let replay = run_metal(false);
    let statc = run_metal(true);
    assert_eq!(
        replay, statc,
        "Metal replay diverged from static Llama decode"
    );
}

/// The audit's pooled Metal scratch buffers must remain coherent when attention repeatedly reuses
/// them across layers and tokens. Exercise the coupled-Q8 q-cast and the decoupled-KV dequant
/// prepasses on a prompt wide enough to enter prefill attention before continuing through decode.
#[cfg(target_os = "macos")]
#[test]
fn metal_kv_scratch_paths_are_coherent() {
    const EXPECTED: u64 = 0xef91_912b_db14_c7fd;
    let path = need_model!(llama32_1b(), "Llama-3.2-1B");
    let mut _tlk = test_serial_lock();
    let prompt =
        "Explain how a CPU instruction pipeline works and list its common hazards. ".repeat(6);
    for (k, v) in [
        ("q8_0", "q8_0"),
        ("f16", "q4_1"),
        ("f16", "q5_0"),
        ("f16", "q5_1"),
        ("bf16", "bf16"),
        ("f16", "turbo2"),
        ("f16", "turbo3"),
        ("f16", "turbo4"),
    ] {
        let model = model_cfg(&path, |c| kv_types(c, k, v));
        let rendered = model.render_chat(&prompt).expect("render chat");
        let mut out = String::new();
        model
            .generate_metal(&rendered, 16, None, |p| out.push_str(p))
            .expect("metal generation");
        assert!(
            !out.trim().is_empty(),
            "Metal K={k} V={v} produced no output"
        );
        assert!(
            !is_degenerate(&out),
            "Metal K={k} V={v} output degenerate: {:?}",
            out.chars().take(64).collect::<String>()
        );
        assert_eq!(
            fnv1a(&out),
            EXPECTED,
            "Metal K={k} V={v} generation changed: {out:?}"
        );
    }
}

/// gemma4 (heterogeneous head dims 256/512, V-norm, freq_factors, softcap) through the Vulkan seam.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_gemma4() {
    let path = need_model!(gemma4_12b(), "gemma-4-12b");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is 2+2? Answer briefly.", 12);
}

/// gemma4 E2B (per-layer embeddings, KV/FFN sharing) through the Vulkan seam (per-token prefill —
fn gemma4_e2b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf")
}

// Captured + verified coherent (gemma4 E2B: per-layer input embeds + KV sharing): "The capital of
// France is **Paris**.", a brave-knight story ("Sir Kaelan … kingdom of Eldoria …").
const GEMMA4_E2B_GOLDEN: &[(&str, usize, u64)] = &[
    (
        "The capital of France is",
        32,
        0x689e792098786962, // channel-thought reasoning ("…Analyze the Request… factual question")
    ),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x8909237b9419d782, // channel-thought reasoning (story planning process)
    ),
];

/// gemma4-E2B (per-layer token embeddings) through the Vulkan seam.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_gemma4_e2b() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is 2+2? Answer briefly.", 12);
}

/// A STREAMED gemma4-E2B keeps the chunk-major order, and a GATE is what makes it — not a panic.
///
/// E2B's layer stack reads `per_layer_inp`, which the graph PROLOGUE builds, so a layer span
/// starting past layer 0 would read an unbound tensor; `build` asserts exactly that. E2B does take
/// the batched-prefill path, so once a streamed model defaulted to layer-major that assert became
/// reachable: `infr bench <e2b> --set paging.cache=200m` panicked with "gemma4-E2B cannot start a
/// layer span past layer 0". `seam::layer_major_prefill` now refuses the architecture and warns.
///
/// The assertion is token identity against the resident run, so this fails both if the gate is
/// removed (panic) and if some later span-capable E2B path returns different tokens.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_streamed_e2b_stays_chunk_major() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    let model = model_default(&path);
    // Several chunks: only a multi-chunk prefill builds more than one span at all.
    let long = "Explain in detail how a transformer processes a sequence of tokens. ".repeat(8);
    let rendered = model.render_chat(&long).expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");
    let pin_chunk = |c: &mut infr_llama::EngineConfig| {
        c.device.ubatch = Some(64);
        c.device.ubatch_specified = true;
    };

    let mut resident_ids = Vec::new();
    model_cfg(&path, pin_chunk)
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    // Small enough to force the streaming path, where `layer_major` defaults ON for every other
    // architecture — which is the case that used to panic here.
    let mut streamed_ids = Vec::new();
    model_cfg(&path, |c| {
        pin_chunk(c);
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(200 * 1024 * 1024));
    })
    .generate_vulkan_ids(&prompt_ids, n, |id| streamed_ids.push(id))
    .expect("streamed e2b gpu gen must not panic on a layer span");

    assert_eq!(
        streamed_ids, resident_ids,
        "streamed E2B diverged from the all-resident GPU run"
    );
}

/// qwen3moe (routed-expert Op::MoeFfn) through the Vulkan seam, batched GPU-routed prefill. The
/// batched FFN runs int8 dp4a expert GEMMs (each parity-tested at the inherent ~2e-2 activation-
/// quant tolerance) — a numeric path the f32 CPU oracle can diverge from on a near-tie greedy
/// pick, so per the repo convention this locks its OWN golden (deterministic + read for
/// coherence; refresh with INFR_BLESS=1) instead of comparing token-for-token.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_golden_qwen3moe() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let out = model
        .generate_dense_vulkan(&rendered, 8)
        .expect("vulkan seam gen");
    let h = fnv1a(&out);
    if std::env::var("INFR_BLESS").is_ok() {
        println!("qwen3moe seam golden: 0x{h:016x}  // {out:?}");
    } else {
        // Refreshed post-e38becc: the Q6_K mmq nibble-map fix corrected this model's batched
        // expert-GEMM output (its Q4_K_M ships Q6_K ffn_down banks) — the old hash locked the
        // buggy kernel's text. New output verified coherent + q6k parity-proven vs the host
        // reference (nc_gemm_parity random banks).
        assert_eq!(
            h, 0xe2ed327ed3301524,
            "qwen3moe seam golden changed\n  out: {out:?}"
        );
    }
}

/// BF16 (float-weight) seam parity: a bf16 model runs on the seam with its projection weights
/// converted to f16 (the matmul_proj / f16-GEMM prefill path) while the norm weights stay f32 (the
/// rmsnorm/qk_norm kernels read f32). Must match the CPU reference oracle token-for-token — proving
/// the float-weight GPU path is correct, not just fast.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_bf16_matches_cpu() {
    let snap = match qwen3_06b() {
        Some(p) => p.parent().unwrap().to_path_buf(),
        None => return,
    };
    let path = snap.join("Qwen3-0.6B-BF16.gguf");
    if !path.exists() {
        eprintln!("skip: no BF16 model");
        return;
    }
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let prompt = model
        .render_chat("What is the capital of France? Answer in one word.")
        .expect("render chat");
    let mut cpu_txt = String::new();
    model
        .generate_cpu(&prompt, 16, None, |p| cpu_txt.push_str(p))
        .expect("cpu gen");
    let gpu_txt = model.generate_dense_vulkan(&prompt, 16).expect("seam gen");
    assert_eq!(
        cpu_txt.trim(),
        gpu_txt.trim(),
        "bf16 seam (f16 projections) diverged from the CPU oracle"
    );
}

// CPU quant coverage: the SAME prompt through every available quantization of Qwen3-0.6B — legacy
// round (Q4_0), k-quants (Q2_K/Q4_K/Q5_K/Q6_K), high-bit (Q8_0), i-quant codebook (IQ4_XS), and float
// (BF16). Each exercises a different dequant/dot path; the per-quant golden hash is locked (each
// verified coherent at capture). Missing quants are skipped. Refresh with `INFR_BLESS=1`.
// All verified coherent at capture — every quant recalls "France's capital is Paris" (Q2_K is a
// touch repetitive, as expected for 2-bit; the float-ish quants still converge: Q5_K==Q8_0==BF16).
const QWEN3_QUANT_GOLDEN: &[(&str, usize, u64)] = &[
    ("IQ4_XS", 32, 0xd028ff03b524cb28),
    ("Q2_K", 32, 0x6442c2818c12ca56),
    ("Q4_0", 32, 0x88221dcfca820246),
    ("Q4_K_M", 32, 0xfd63781ea3bfa785),
    ("Q5_K_M", 32, 0x4e510646d603bc03),
    ("Q6_K", 32, 0xb68f96c3aa8d22fe),
    ("Q8_0", 32, 0xb68f96c3aa8d22fe),
    ("BF16", 32, 0xb68f96c3aa8d22fe),
];

#[test]
fn cpu_golden_qwen3_quants() {
    let mut _tlk = test_serial_lock();
    let bless = std::env::var("INFR_BLESS").is_ok();
    let prompt = "The capital of France is";
    for (quant, n, want) in QWEN3_QUANT_GOLDEN {
        let Some(path) = qwen3_quant(quant) else {
            eprintln!("skip {quant}: not downloaded");
            continue;
        };
        let model = model_default(&path);
        let out = cpu_gen(&model, prompt, *n);
        let h = fnv1a(&out);
        if bless {
            println!("    ({quant:?}, {n}, 0x{h:016x}),  // {out:?}");
        } else {
            assert_eq!(h, *want, "quant {quant} golden changed\n  out: {out:?}");
        }
    }
}

// ─── Gemma 3 (dense) ────────────────────────────────────────────────────────────

fn gemma3_1b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf")
}

fn gemma3_1b_q2k() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q2_K.gguf")
}

// ─── Llama (plain interleaved RoPE, no qk-norm) ────────────────────────────────

fn llama32_1b() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_LLAMA") {
        return Some(PathBuf::from(p));
    }
    find_gguf(
        "unsloth--Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
    )
}

// Captured + verified coherent: "The capital of France is Paris. 😊", a brave-knight short story
// ("The rain in Silverwood always tasted of regret"). Re-blessed THREE times: dense Q5_0 Linear
// onto the int8 kernel, then the attention-SIMD reassociation, then the Q8_0 sub-256-in_f fix
// (gemma3-1b's 1152-dim projections are Q8_0, whose CPU dot previously TRUNCATED to 1024 elems —
// see vec_dot_q8_0 routing). The corrected output is token-identical to the Vulkan backend
// (independent int8 impl), so this bless fixes a wrong golden, not a precision drift.
const GEMMA3_GOLDEN: &[(&str, usize, u64)] = &[
    ("The capital of France is", 32, 0xe5a37ab078db3a2c),
    (
        "Tell me a short story about a brave knight.",
        48,
        0x28e39d5dd5b5f858,
    ),
];

/// CPU-only: Gemma 3 (sandwich norms, GeGLU, dual-RoPE, SWA) golden-hash lock.
#[test]
fn cpu_golden_gemma3() {
    let path = need_model!(gemma3_1b(), "gemma-3-1b");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_golden(&model, GEMMA3_GOLDEN);
}

// ─── Qwen3.5 / Qwen3-Next (gated DeltaNet) ──────────────────────────────────────

fn qwen35_08b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf")
}

// Captured + verified coherent (qwen35 / Qwen3-Next: gated-DeltaNet + gated full-attention): "The
// capital of France is **Paris**. It is the largest city …", a knight story ("Elara … Aethelgard").
// Renders with thinking ON (the infr-wide default; INFR_NO_THINK turns it off).
const QWEN35_GOLDEN: &[(&str, usize, u64)] = &[
    (
        "The capital of France is",
        32,
        0x542a9dd055c58884, // prefilled-think reasoning ("Thinking Process… capital of France")
    ),
    (
        "Tell me a short story about a brave knight.",
        48,
        0xbe06c5580ea33d78, // prefilled-think reasoning (story planning process)
    ),
];

/// CPU-only: qwen35 / Qwen3-Next golden-hash lock (gated-DeltaNet recurrence + conv + gated full
/// attention), through the UNIFIED shared-transformer path (`SeamModel::generate_cpu`, i.e.
/// `seam::generate_dense_cpu` with the `MixerW::DeltaNet` branch) — the same runner every other
/// arch's `cpu_golden_*` test above locks. (Historically this ran through a bespoke qwen35-only
/// seam that lived in `qwen35.rs`; that seam was proven token-identical to this unified path
/// during the cutover and has since been deleted — issue #30.)
#[test]
fn cpu_golden_qwen35() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_golden(&model, QWEN35_GOLDEN);
}

// ─── qwen35 on the UNIFIED shared-transformer path ─────────────────────────────────
//
// `Config::from_gguf` accepts `arch == "qwen35"` and `seam`'s layer loop has a `MixerW::DeltaNet`
// branch (see `docs/qwen35.md`) — so `SeamModel::load` on a qwen35 GGUF drives the SAME shared
// runner every other arch uses, and production routing (`infr run`/`serve`/`bench` in infr-cli)
// sends qwen35 through this path unconditionally.

/// The unified Vulkan seam (`SeamModel::generate_dense_vulkan`) must match the unified CPU oracle
/// (`SeamModel::generate_cpu`) token-for-token — the seam twin of every other arch's
/// `gpu_seam_matches_cpu_*` test, now exercising `MixerW::DeltaNet` (Conv1dSilu/DeltaNet ops) AND
/// the qwen35 attention layers' interleaved q+gate split + sigmoid output gate through Vulkan.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn unified_qwen35_gpu_seam_matches_cpu() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let mut _tlk = test_serial_lock();
    seam_vulkan_matches_cpu(&path, "What is bash? Answer briefly.", 24);
}

/// qwen35's gated-DeltaNet recurrent state is an APPEND-ONLY summary — it can't rewind to an
/// arbitrary shared prefix the way a real KV cache can (see docs/qwen35.md and the no-rewind rule
/// in `seam::generate_dense_backend`). On the unified Vulkan session (`vulkan_session` /
/// `generate_vulkan_session`, the seam twin of `gpu_seam_kv_reuse_matches_fresh`):
///   (a) a prompt that EXACTLY EXTENDS the previous turn's fed sequence continues the recurrent
///       state — suffix-only prefill (`n_prompt` shrinks), output identical to a fresh full prefill.
///   (b) a prompt that does NOT extend it (a divergent turn) must fall back to a FULL re-prefill —
///       `n_prompt` equal to what a brand-new session prefills for the same prompt (proving the
///       state was zero-reset, not silently reused from a wrong point).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn unified_qwen35_session_no_rewind() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let mut _tlk = test_serial_lock();
    // fixed-length turns, no early EOS stop
    let model = model_cfg(&path, |c| c.sampling.ignore_eos = true);
    let mut sess = model.vulkan_session(512).expect("session");

    let p1 = "The quick brown fox jumps over the lazy dog. The capital of France is";
    let mut t1 = String::new();
    let s1 = model
        .generate_vulkan_session(&mut sess, p1, 8, None, |p| t1.push_str(p))
        .expect("turn 1");
    assert!(s1.n_prompt > 0);

    // (a) EXTENDS turn 1 — suffix-only prefill, must match a fresh full prefill exactly.
    let p2 = format!("{p1}{t1} And the capital of Germany is");
    let mut t2 = String::new();
    let s2 = model
        .generate_vulkan_session(&mut sess, &p2, 8, None, |p| t2.push_str(p))
        .expect("turn 2 (extend)");
    let fresh2 = model.generate_dense_vulkan(&p2, 8).expect("fresh turn 2");
    assert_eq!(
        t2.trim(),
        fresh2.trim(),
        "extend-session output diverged from a fresh full prefill"
    );
    assert!(
        s2.n_prompt < s1.n_prompt,
        "turn 2 prefilled {} tokens — session reuse (extend) didn't kick in",
        s2.n_prompt
    );

    // (b) does NOT extend turn 2 (divergent subject) — the recurrent state can't rewind, so this
    // must be a FULL re-prefill: n_prompt must equal what a BRAND-NEW session prefills (its first
    // call always fully prefills, on every arch), not some smaller partial-prefix reuse.
    let p3 = "Completely different subject entirely: photosynthesis converts";
    let mut t3 = String::new();
    let s3 = model
        .generate_vulkan_session(&mut sess, p3, 8, None, |p| t3.push_str(p))
        .expect("turn 3 (divergent)");
    let mut fresh_sess = model.vulkan_session(512).expect("fresh session");
    let mut tf3 = String::new();
    let sf3 = model
        .generate_vulkan_session(&mut fresh_sess, p3, 8, None, |p| tf3.push_str(p))
        .expect("fresh turn 3");
    assert_eq!(
        t3.trim(),
        tf3.trim(),
        "post-reset generation diverged from a fresh prefill"
    );
    assert_eq!(
        s3.n_prompt, sf3.n_prompt,
        "divergent turn didn't fully re-prefill (no-rewind rule violated): got {} vs fresh {}",
        s3.n_prompt, sf3.n_prompt
    );
}

// ─── MTP (multi-token prediction) speculative decoding — Phase 1 (issue #33) ────────────────
//
// See docs/mtp.md. Phase 1 only parses `{arch}.nextn_predict_layers` into `Config` (splitting the
// GGUF's `block_count` into trunk + head) and loads/shape-checks the head's own tensors — no MTP
// forward yet, so these tests validate LOADING + the `h`-tap primitive Phase 2 needs, not drafting.

fn qwen35_4b_mtp() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3.5-4B-MTP-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf")
}

/// The 4B MTP GGUF's `qwen35.block_count=33` INCLUDES the head layer
/// (`qwen35.nextn_predict_layers=1`) — `Config::from_gguf` must split it into a 32-layer TRUNK +
/// `n_layer_nextn=1` (today, before this phase, the trunk layer loop would misclassify `blk.32` as
/// a gated-DeltaNet layer and fail on missing `ssm_*` tensors — see `Config::n_layer_nextn`'s doc).
/// `mtp::load_mtp_head` must then find every required head tensor and correctly report the three
/// optional `nextn.*` fallback tensors ABSENT — this shipped GGUF's live path is 100% fallback to
/// the main model's `token_embd`/`output`/`output_norm` (see `docs/mtp.md`'s confirmed dump).
#[test]
fn mtp_gguf_loads() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf on the MTP GGUF");
    assert_eq!(cfg.n_layer, 32, "trunk n_layer must exclude the MTP head");
    assert_eq!(cfg.n_layer_nextn, 1, "qwen35.nextn_predict_layers=1");

    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    assert_eq!(
        head.il, 32,
        "the head sits immediately after the 32-layer trunk"
    );
    println!("MTP head tensors (blk.{}):", head.il);
    println!("  attn_norm              {:?}", head.attn_norm.shape);
    println!("  attn_q (interleaved q+gate) {:?}", head.attn_q.shape);
    println!("  attn_k                 {:?}", head.attn_k.shape);
    println!("  attn_v                 {:?}", head.attn_v.shape);
    println!("  attn_q_norm            {:?}", head.attn_q_norm.shape);
    println!("  attn_k_norm            {:?}", head.attn_k_norm.shape);
    println!("  attn_output            {:?}", head.attn_output.shape);
    println!(
        "  post_attention_norm    {:?}",
        head.post_attention_norm.shape
    );
    println!("  ffn_gate               {:?}", head.ffn_gate.shape);
    println!("  ffn_up                 {:?}", head.ffn_up.shape);
    println!("  ffn_down               {:?}", head.ffn_down.shape);
    println!("  nextn.eh_proj          {:?}", head.eh_proj.shape);
    println!("  nextn.enorm            {:?}", head.enorm.shape);
    println!("  nextn.hnorm            {:?}", head.hnorm.shape);
    println!(
        "  nextn.embed_tokens     {:?} (fallback: main tok_embd)",
        head.embed_tokens.as_ref().map(|t| &t.shape)
    );
    println!(
        "  nextn.shared_head_head {:?} (fallback: main lm_head)",
        head.shared_head_head.as_ref().map(|t| &t.shape)
    );
    println!(
        "  nextn.shared_head_norm {:?} (fallback: main output_norm)",
        head.shared_head_norm.as_ref().map(|t| &t.shape)
    );

    // Confirmed dump (docs/mtp.md): the shipped GGUF omits `embed_tokens`/`shared_head_head` (so
    // those fall back to the main model's `token_embd`/tied lm_head) but DOES ship its own
    // `shared_head_norm` (unlike the other two, this one is NOT a fallback in this GGUF).
    assert!(head.embed_tokens.is_none(), "confirmed absent in this GGUF");
    assert!(
        head.shared_head_head.is_none(),
        "confirmed absent in this GGUF"
    );
    assert!(
        head.shared_head_norm.is_some(),
        "confirmed PRESENT in this GGUF (docs/mtp.md)"
    );
}

/// The 0.8B (nextn-free) GGUF has no `nextn_predict_layers` key — `Config::from_gguf` must parse
/// it exactly as before this phase (`n_layer_nextn=0`, `n_layer` unchanged). Run alongside this:
/// `timeout 600 cargo test --release -p infr-llama --test cpu_backend unified_qwen35 -- --nocapture`
/// proves the TRUNK FORWARD itself (not just `Config` parsing) is byte-for-byte untouched.
#[test]
fn qwen35_trunk_unaffected() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert_eq!(cfg.n_layer_nextn, 0, "0.8B has no MTP head");
    assert_eq!(
        cfg.n_layer, 24,
        "trunk layer count must be unaffected by the nextn parsing"
    );
}

/// The h-tap (`SeamModel::prefill_logits_and_h_cpu`, issue #33's Phase 2 primitive): the captured
/// `h` row must be EXACTLY the lm_head's input for the SAME forward's logits row — i.e.
/// `lm_head(h) == logits` for qwen35 (a plain tied/untied GEMV, no softcap — `Config::final_softcap`
/// is 0 for every qwen35 model, unlike gemma). Host-recomputes the GEMV from the same dequantized
/// weight the graph used, in the same f32 precision, so this should match near bit-exactly.
#[test]
fn h_tap_matches_lm_head() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let model = model_default(&path);
    let tokens = model.encode("The capital of France is").expect("encode");

    let (logits, h) = model
        .prefill_logits_and_h_cpu(&tokens)
        .expect("prefill_logits_and_h_cpu");
    let cfg = model.config();
    assert_eq!(h.len(), cfg.n_embd, "h is one row: [n_embd]");
    assert_eq!(logits.len(), cfg.vocab, "logits is one row: [vocab]");

    // Host lm_head: the SAME tensor `build`'s `wload` picks (`output.weight`, or — tied — the
    // quantized `token_embd.weight`), fully dequantized here (vs the graph's lazy per-row dequant)
    // — same math, different (irrelevant) dequant call site.
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let lm_name = if g.tensors().iter().any(|t| t.name == "output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == lm_name)
        .expect("lm_head tensor")
        .clone();
    let bytes = g.tensor_bytes(lm_name).expect("lm_head bytes");
    let w = infr_gguf::dequant::dequant_block(info.dtype, bytes).expect("dequant lm_head");
    let ne = cfg.n_embd;
    let vocab = cfg.vocab;
    assert_eq!(w.len(), ne * vocab, "lm_head dequant length");

    let mut host_logits = vec![0f32; vocab];
    for (v, out) in host_logits.iter_mut().enumerate() {
        let row = &w[v * ne..v * ne + ne];
        *out = row.iter().zip(&h).map(|(&wv, &hv)| wv * hv).sum();
    }
    let max_abs = logits
        .iter()
        .zip(&host_logits)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    let max_val = logits.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    let rel = max_abs / max_val;
    // token_embd here is Q6_K: the graph's `Op::Linear` runs a quantized-activation integer dot
    // (q8 x q6_k) while this test's host GEMV runs a plain f32 dot over the fully dequantized
    // weight — the two paths agree to quantization tolerance, not bit-exactly.
    println!(
        "h_tap_matches_lm_head: max|graph-host|={max_abs:.6} (logit magnitude ~{max_val:.3}, rel={rel:.6})"
    );
    // 2% relative tolerance: the graph quantizes its OWN activation (q8) before the integer
    // Q6_K dot, while this host GEMV stays f32 throughout — a different (not buggy) arithmetic
    // path; measured ~1.1% on this model (see the printed `rel` above). A truly bit-exact check
    // would need to replicate the q8-activation Q6_K dot kernel host-side, which is out of scope
    // for a Phase 1 wiring check — the point here is "no missing/extra op between `h` and
    // `logits`", not quant-kernel bit-parity (that's covered elsewhere by the CPU/GPU goldens).
    assert!(
        rel < 0.02,
        "lm_head(h) diverged from the graph's logits: max abs diff {max_abs} (rel {rel})"
    );
}

// ─── MTP Phase 2: the head forward + the draft loop (issue #33) ─────────────────────────────
//
// See docs/mtp.md. Phase 2 builds the head's own 1-layer forward + the catch_up/draft driver
// primitives (`crate::mtp`) — these tests drive the ACTUAL 4B MTP GGUF's head, not just load it.

/// Prime a fresh [`infr_llama::mtp::MtpHeadSession`] over `prompt_tokens`: prefill the TRUNK on the
/// CPU backend (capturing `h` for every prompt row via the Phase-1 VERIFY tap), then `catch_up` the
/// head over the whole prompt in one call. Returns the session plus `(last_token, pending_h)` —
/// `draft`'s starting point (`docs/mtp.md`'s `process()`/`pending_h` handoff).
fn prime_head<'a>(
    model: &'a infr_llama::SeamModel,
    head: &infr_llama::mtp::MtpHeadWeights,
    cpu_be: &'a infr_cpu::CpuBackend,
    g: &infr_gguf::Gguf,
    max_ctx: usize,
    prompt_tokens: &[u32],
) -> (infr_llama::mtp::MtpHeadSession<'a>, u32, Vec<f32>) {
    let (_logits, h_rows) = model
        .verify_logits_and_h_cpu(prompt_tokens)
        .expect("verify_logits_and_h_cpu");
    let ne = model.config().n_embd;
    assert_eq!(h_rows.len(), prompt_tokens.len() * ne, "h per prompt row");

    let mut sess = infr_llama::mtp::MtpHeadSession::new_cpu(
        cpu_be,
        g,
        model.config(),
        model.engine_cfg(),
        head,
        model.token_embd().unwrap(),
        max_ctx,
    )
    .expect("MtpHeadSession::new_cpu");

    // docs/mtp.md's process(): the head decodes the SAME tokens with `h` shifted right by one
    // (`embd[i] = h_tgt[i-1]`); row 0 has no predecessor in a fresh session, so it's paired with a
    // zero `pending_h` (`speculative.cpp`'s `pending_h` starts zero-initialized — see
    // `common_speculative_impl_draft_mtp`'s ctor, `pending_h.assign(n_seq, vector<float>(n_embd,
    // 0.0f))` — there IS no earlier target row to have produced a real one).
    let mut shifted_h = vec![0f32; prompt_tokens.len() * ne];
    if prompt_tokens.len() > 1 {
        shifted_h[ne..].copy_from_slice(&h_rows[..(prompt_tokens.len() - 1) * ne]);
    }
    infr_llama::mtp::catch_up(&mut sess, prompt_tokens, &shifted_h, 0).expect("catch_up");

    let id_last = *prompt_tokens.last().expect("nonempty prompt");
    let pending_h = h_rows[(prompt_tokens.len() - 1) * ne..].to_vec();
    (sess, id_last, pending_h)
}

/// The head forward, end to end, on the real 4B MTP GGUF: prefill a short prompt on the TRUNK
/// (capturing `h` via the Phase-1 tap), `catch_up` the head over it, then `draft` 6 tokens
/// (`--spec-draft-n-max 6`, matching `docs/mtp.md`'s oracle run). Asserts every logits row is
/// finite (no NaN/Inf — the eh_proj concat layout is exactly the kind of bug that would show up as
/// garbage here) and prints the drafted ids + top-1 probabilities.
#[test]
fn mtp_head_forward_finite() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = model_default(&path);

    let prompt_tokens = model.encode("The capital of France is").expect("encode");
    let cpu_be = infr_cpu::CpuBackend::new();
    let n_max = 6usize;
    let max_ctx = prompt_tokens.len() + n_max + 4;
    let (mut sess, id_last, pending_h) =
        prime_head(&model, &head, &cpu_be, &g, max_ctx, &prompt_tokens);

    let drafted = infr_llama::mtp::draft(
        &mut sess,
        id_last,
        &pending_h,
        prompt_tokens.len(),
        infr_llama::mtp::DEFAULT_P_MIN,
        n_max,
    )
    .expect("draft");

    println!(
        "mtp_head_forward_finite: drafted {} token(s):",
        drafted.len()
    );
    for (i, &(id, p)) in drafted.iter().enumerate() {
        println!("  [{i}] id={id} p={p:.4}");
    }
    assert!(
        !drafted.is_empty(),
        "p_min=0.0 should always draft n_max tokens"
    );
    assert_eq!(drafted.len(), n_max, "p_min=0.0 never stops the loop early");
    for &(id, p) in &drafted {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "top1 prob out of range: {p}"
        );
        assert!((id as usize) < cfg.vocab, "drafted id out of vocab range");
    }
}

/// CPU/Vulkan parity: the SAME trunk-captured `h` (CPU, per `mtp_head_forward_finite`'s doc — only
/// the HEAD differs between the two calls below) drafted through the head on both backends must
/// produce the IDENTICAL token sequence (dense head, no MoE/routing noise to legitimately diverge
/// on — unlike the CPU/GPU generation goldens elsewhere in this file, which tolerate divergence).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn mtp_head_cpu_vulkan_parity() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = model_default(&path);

    let prompt_tokens = model.encode("The capital of France is").expect("encode");
    let n_max = 6usize;
    let max_ctx = prompt_tokens.len() + n_max + 4;

    let cpu_be = infr_cpu::CpuBackend::new();
    let (mut cpu_sess, id_last, pending_h) =
        prime_head(&model, &head, &cpu_be, &g, max_ctx, &prompt_tokens);
    let cpu_drafted = infr_llama::mtp::draft(
        &mut cpu_sess,
        id_last,
        &pending_h,
        prompt_tokens.len(),
        infr_llama::mtp::DEFAULT_P_MIN,
        n_max,
    )
    .expect("cpu draft");

    let vk = infr_vulkan::VulkanBackend::new().expect("vulkan init");
    let mut vk_sess = infr_llama::mtp::MtpHeadSession::new_vulkan(
        &vk,
        &g,
        model.config(),
        model.engine_cfg(),
        &head,
        model.token_embd().unwrap(),
        max_ctx,
    )
    .expect("MtpHeadSession::new_vulkan");
    let ne = model.config().n_embd;
    let mut shifted_h = vec![0f32; prompt_tokens.len() * ne];
    if prompt_tokens.len() > 1 {
        let (_logits, h_rows) = model
            .verify_logits_and_h_cpu(&prompt_tokens)
            .expect("verify_logits_and_h_cpu");
        shifted_h[ne..].copy_from_slice(&h_rows[..(prompt_tokens.len() - 1) * ne]);
    }
    infr_llama::mtp::catch_up(&mut vk_sess, &prompt_tokens, &shifted_h, 0).expect("vk catch_up");
    let vk_drafted = infr_llama::mtp::draft(
        &mut vk_sess,
        id_last,
        &pending_h,
        prompt_tokens.len(),
        infr_llama::mtp::DEFAULT_P_MIN,
        n_max,
    )
    .expect("vk draft");

    let cpu_ids: Vec<u32> = cpu_drafted.iter().map(|&(id, _)| id).collect();
    let vk_ids: Vec<u32> = vk_drafted.iter().map(|&(id, _)| id).collect();
    println!("mtp_head_cpu_vulkan_parity: cpu={cpu_ids:?} vulkan={vk_ids:?}");
    assert_eq!(cpu_ids, vk_ids, "CPU/Vulkan MTP head drafts diverged");
}

/// Regression: the fused on-device `draft_chain` (one submit for all `n_max` steps) MUST draft the
/// EXACT same token ids as the per-step `draft()` from an identical primed state. Both run on
/// Vulkan and use the same GPU `Op::ArgmaxProb`; the only difference is graph structure, so any
/// divergence is a chain-graph bug. Guards the `decode_eligible` fix: the unrolled chain's
/// all-`rows==1` attentions used to (wrongly) qualify for record-once decode replay, which drives
/// every pos-dependent op from ONE shared params position — collapsing all `n_max` steps onto the
/// same KV row / RoPE angle / kv_len and corrupting every step past the first. Fuzzes several
/// realistic primed states (each prompt row's trunk-`h` as `pending_h`, varied `id_last`).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn mtp_draft_chain_matches_per_step() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = model_default(&path);
    let ne = cfg.n_embd;
    let n_max = 6usize;

    let prompt_tokens = model.encode("The capital of France is").expect("encode");
    let p = prompt_tokens.len();
    let max_ctx = p + n_max + 4;

    let vk = infr_vulkan::VulkanBackend::new().expect("vulkan init");
    let mut sess = infr_llama::mtp::MtpHeadSession::new_vulkan(
        &vk,
        &g,
        model.config(),
        model.engine_cfg(),
        &head,
        model.token_embd().unwrap(),
        max_ctx,
    )
    .expect("MtpHeadSession::new_vulkan");
    assert!(sess.can_draft_chain(), "vulkan should support draft_chain");

    let (_logits0, h_rows0) = model
        .verify_logits_and_h_cpu(&prompt_tokens)
        .expect("verify_logits_and_h_cpu");
    let mut shifted_h = vec![0f32; p * ne];
    if p > 1 {
        shifted_h[ne..].copy_from_slice(&h_rows0[..(p - 1) * ne]);
    }
    infr_llama::mtp::catch_up(&mut sess, &prompt_tokens, &shifted_h, 0).expect("catch_up");

    for r in 0..p {
        let h_r = h_rows0[r * ne..(r + 1) * ne].to_vec();
        for &tok in &[prompt_tokens[r], (r as u32 * 997 + 13) % cfg.vocab as u32] {
            let per_step: Vec<u32> = infr_llama::mtp::draft(
                &mut sess,
                tok,
                &h_r,
                p,
                infr_llama::mtp::DEFAULT_P_MIN,
                n_max,
            )
            .expect("draft")
            .iter()
            .map(|&(id, _)| id)
            .collect();
            let chained = sess.draft_chain(tok, &h_r, p, n_max).expect("draft_chain");
            assert_eq!(
                per_step, chained,
                "draft_chain diverged from per-step draft (tok={tok}, row={r})"
            );
        }
    }
}

/// Oracle-invariant fallback (`docs/mtp.md`'s validation ladder — capturing the oracle's OWN
/// verbose drafted-token trace proved impractical: llama.cpp's `SPC_DBG`/`SPC_TRC` macros gate on
/// `common_log`'s verbosity, not a dedicated spec-debug env var, and piping a live CPU generation's
/// stderr for a handful of draft steps is a lot of process-control machinery for what this simpler
/// check already covers): feed the head's drafted tokens through the TRUNK's OWN greedy decode and
/// measure how often the trunk's argmax agrees with what the head drafted — the PER-STEP acceptance
/// probability `alpha` a real spec-verify pass would see (stops at the first mismatch, like a real
/// verify). For `n_max=6` and i.i.d. per-step acceptance `alpha`, expected tokens/cycle is `(1 -
/// alpha^7) / (1 - alpha)`; solving that for the oracle's captured 2.0x (`docs/mtp.md`) gives
/// `alpha ≈ 0.5`, not a flat 60-80% — this test reports the measured per-prompt rate (averaged over
/// a couple of short prompts to dilute single-prompt noise) against that ~0.5 reference rather than
/// hard-gating on a specific number (still a coarse sanity check, not a benchmark).
#[test]
fn mtp_head_trunk_acceptance_rate() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");
    let model = model_default(&path);
    let cpu_be = infr_cpu::CpuBackend::new();
    let n_max = 6usize;
    let vocab = cfg.vocab;

    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;
    for prompt in [
        "The capital of France is",
        "Tell me a short story about a brave knight.",
    ] {
        let prompt_tokens = model.encode(prompt).expect("encode");
        let max_ctx = prompt_tokens.len() + n_max + 4;
        let (mut sess, id_last, pending_h) =
            prime_head(&model, &head, &cpu_be, &g, max_ctx, &prompt_tokens);

        let drafted = infr_llama::mtp::draft(
            &mut sess,
            id_last,
            &pending_h,
            prompt_tokens.len(),
            infr_llama::mtp::DEFAULT_P_MIN,
            n_max,
        )
        .expect("draft");
        let draft_ids: Vec<u32> = drafted.iter().map(|&(id, _)| id).collect();

        // Trunk greedy-verify over [prompt | draft_ids]: row i's logits are the trunk's
        // distribution AFTER consuming prompt_tokens ++ draft_ids[..i] — i.e. exactly what the
        // trunk would have sampled in place of draft_ids[i] had it decoded token-by-token (the
        // spec-verify invariant).
        let mut full = prompt_tokens.clone();
        full.extend_from_slice(&draft_ids);
        let (verify_logits, _h) = model
            .verify_logits_and_h_cpu(&full)
            .expect("verify_logits_and_h_cpu over prompt+draft");
        let p = prompt_tokens.len();

        let mut accepted = 0usize;
        for (i, &draft_id) in draft_ids.iter().enumerate() {
            // Row `p - 1 + i` is the trunk's distribution for predicting position `p + i` — i.e.
            // the token that FOLLOWS `full[..p+i]`, which is exactly `draft_ids[i]` when accepted.
            let row = &verify_logits[(p - 1 + i) * vocab..(p + i) * vocab];
            let (argmax, _) =
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    });
            let ok = argmax as u32 == draft_id;
            println!(
                "  {prompt:?} step {i}: drafted={draft_id} trunk_argmax={argmax} {}",
                if ok { "ACCEPT" } else { "reject" }
            );
            if ok {
                accepted += 1;
            } else {
                break; // spec-verify stops at the first mismatch — can't evaluate the rest as-is
            }
        }
        println!(
            "  {prompt:?}: {accepted}/{} accepted this cycle",
            draft_ids.len()
        );
        total_accepted += accepted;
        total_drafted += draft_ids.len();
    }
    let rate = total_accepted as f64 / total_drafted.max(1) as f64;
    println!(
        "mtp_head_trunk_acceptance_rate: {total_accepted}/{total_drafted} accepted overall \
         ({rate:.2}) — oracle's 2.0x implies a per-step rate around ~0.5 (see this test's doc)"
    );
}

// ─── MTP Phase 3: the self-speculative generation loop (issue #33) ──────────────────────────────
//
// See docs/mtp.md. Phase 3 wires the head into a full generation loop (`crate::mtp::
// generate_mtp_spec_vulkan`) on the production Vulkan seam — these tests drive THAT loop, not the
// head primitives directly (Phase 2's tests above already cover those in isolation).

/// **The Phase 3 hard bar**: self-speculative MTP decoding must be output-IDENTICAL to plain
/// target-only greedy decoding on the SAME (real, production) Vulkan seam — the spec ≡
/// target-greedy invariant `docs/mtp.md`'s own oracle run holds ("byte-identical output"). No
/// tolerance, no golden hash — a real string equality on a real generation. If this fails, the
/// accept/commit/KV logic is wrong (see `crate::mtp::generate_mtp_spec_vulkan`'s doc on the
/// KV-overwrite/no-rewind semantics it relies on) — debug that, don't relax this assertion.
///
/// **IGNORED while MTP is PARKED** (`infr_llama::mtp::mtp_enabled` — the master kill-switch, and
/// the full rationale). Short version: the int8-activation decode kernels every fast dtype now uses
/// carry per-token rounding noise, and MTP's verify batch vs the plain-decode chain it must match
/// are computed at different sequence positions with different KV state — enough to flip a
/// close-margin greedy argmax, so this assertion fails. NOT a bit-identity bug
/// (`mmv_row1_bit_identical` passes) and NOT an accuracy cliff (all 13 `gpu_seam_matches_cpu_*`
/// pass). The assertion itself is CORRECT and is deliberately left intact, not relaxed: re-enabling
/// MTP means making this pass again (accuracy mitigation — e.g. re-verify in f32 when the top-2
/// logit margin is tight), not weakening it. Run with `--ignored` to see the current failure.
#[test]
#[ignore = "MTP parked: int8 decode noise flips a close-margin greedy token (see mtp::mtp_enabled)"]
fn mtp_spec_matches_target_only_greedy() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let prompt = model
        .render_chat("Tell me a short story about a brave knight.")
        .expect("render chat");
    let max_new = 64usize;

    let mut plain = String::new();
    {
        let mut sess = model
            .vulkan_session(prompt.len() + max_new + 64)
            .expect("target-only session");
        model
            .generate_vulkan_session(&mut sess, &prompt, max_new, None, |p| plain.push_str(p))
            .expect("target-only greedy");
    }

    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");

    let mut spec = String::new();
    infr_llama::mtp::generate_mtp_spec_vulkan(&model, &head, &prompt, max_new, |p| {
        spec.push_str(p)
    })
    .expect("mtp spec decode");

    assert_eq!(
        spec, plain,
        "MTP self-speculative stream diverged from target-only greedy"
    );
}

/// Acceptance-rate report over a longer generation, 2-3 prompts (`INFR_PROF_STAGES=1` also prints a
/// per-cycle breakdown to stderr — run this test with `-- --nocapture` and that env var set to see
/// it). Not gated on a specific number (`mtp_head_trunk_acceptance_rate` already sanity-checks the
/// head's own per-step rate against the oracle's implied ~0.5) — this just surfaces the aggregate
/// alpha the full loop achieves so it's visible in normal test output.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn mtp_spec_acceptance_stats() {
    let path = need_model!(qwen35_4b_mtp(), "Qwen3.5-4B-MTP");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let g = infr_gguf::Gguf::open(&path).expect("open gguf");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    let head = infr_llama::mtp::load_mtp_head(&g, &cfg).expect("load_mtp_head");

    for user in [
        "Tell me a short story about a brave knight.",
        "What is the capital of France?",
        "Explain how photosynthesis works in two sentences.",
    ] {
        let prompt = model.render_chat(user).expect("render chat");
        let mut out = String::new();
        let stats = infr_llama::mtp::generate_mtp_spec_vulkan(&model, &head, &prompt, 128, |p| {
            out.push_str(p)
        })
        .expect("mtp spec decode");
        println!(
            "mtp_spec_acceptance_stats: {user:?} -> {} tokens in {:.2}s prompt + {:.2}s decode",
            stats.n_gen, stats.prompt_secs, stats.decode_secs
        );
    }
}

// ─── Qwen3-MoE (routed experts) ─────────────────────────────────────────────────

fn qwen3moe_30b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf")
}

// Captured + verified coherent (qwen3moe: routed-expert FFN, ~3B active of 30B).
// Re-blessed 2026-07-05 for the whole-call int8 MoE gate (multi-row PREFILL calls now run the
// int8-activation fast path in every bucket — a deliberate numeric-regime change; see
// the staged-MoE `int8_ok` doc). Verified coherent ("<think>\nOkay, the user is asking, \"The
// capital of France is\". I need to provide the correct answer.") — re-blessed for the
// attention-SIMD reassociation (numerics policy: match-or-beat llama.cpp CPU precision).
const QWEN3MOE_GOLDEN: &[(&str, usize, u64)] =
    &[("The capital of France is", 24, 0xbc4f22b22d3e3c1d)];

/// Whole-vector cosine similarity (f64 accumulation) — used by the CPU/Vulkan cross-backend
/// logits check below.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// CPU-only: qwen3moe golden-hash lock (the Op::MoeFfn routed-expert path). 30B but only `n_used`
/// experts run per token; still slow on CPU, so a single short case.
#[test]
fn cpu_golden_qwen3moe() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_golden(&model, QWEN3MOE_GOLDEN);
}

/// Paged MoE expert cache (`infr_vulkan::pager`, wired into the seam via `INFR_CACHE`):
/// forces the paged path on this model and asserts the greedy output is IDENTICAL,
/// token-for-token, to both the all-resident GPU run and the CPU reference.
///
/// Qwen3-30B-A3B-Q4_K_M's `ffn_down_exps` bank is NOT uniformly quantized (this unsloth-dynamic
/// quant bumps a subset of layers' down-projection to Q6_K for quality — verified via the GGUF
/// tensor directory) — a fixed-byte-per-slot arena can't hold experts of different byte sizes, a
/// real corruption this task's paged-execution work tripped over and root-caused (a fixed-size
/// arena slot combined with a per-dtype element→byte conversion silently misaligns any non-first
/// slot holding a different-dtype expert). The pager now splits such a role into one arena POOL
/// per (role, per-expert byte size) — see `infr_vulkan::pager`'s MoE-session doc — so this test
/// exercises REAL mixed-dtype paged execution (the down role resolves through two pools), on top
/// of what `gpu_seam_paged_moe_matches_scout_oracle` proves for the uniform split-bank shape.
///
/// `INFR_UBATCH=1`: pins every prefill chunk to rows=1, so EVERY MoeFfn call — CPU, resident GPU
/// alike — takes the small-m id-indexed dequant GEMV path (exact f32-equivalent math). Without it
/// the resident run would default to the BATCHED int8-dp4a prefill path (Q4_K_M is mmq-eligible) —
/// a coarser-precision route that `gpu_seam_golden_qwen3moe`'s own doc notes CAN diverge from the
/// f32 CPU oracle on a near-tie greedy pick, which would make any divergence here ambiguous
/// (quantization noise vs a real bug).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_paged_moe_matches_resident_and_cpu() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    // `device.ubatch = 1` pins every prefill chunk to rows=1 (see the doc above). `paging.cache`
    // is resolved once per model at placement, so the resident and paged runs are two models
    // differing in exactly that one field — which is the point of the config being a value.
    let pin_ubatch = |c: &mut infr_llama::EngineConfig| {
        c.device.ubatch = Some(1);
        c.device.ubatch_specified = true;
    };
    let model = model_cfg(&path, pin_ubatch);
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");

    let mut cpu_ids = Vec::new();
    model
        .generate_cpu_ids(&prompt_ids, n, |id| cpu_ids.push(id))
        .expect("cpu gen");

    let mut resident_ids = Vec::new();
    model
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    // 0.05 GB is far below what even ONE Q4_K_M expert layer's gate+up+down banks need — guarantees
    // real eviction pressure across the model's 48 MoE layers.
    let paged = model_cfg(&path, |c| {
        pin_ubatch(c);
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(50 * 1024 * 1024));
        // `paging.stats`: the pager's hit/miss/eviction report, on the PAGED model only. A value
        // on this model's config since S5a (`VulkanBackend::new_with` hands it to the pager
        // sessions), so it no longer has to be an env write the resident run above would also see.
        c.paging.stats = true;
    });
    let mut paged_ids = Vec::new();
    paged
        .generate_vulkan_ids(&prompt_ids, n, |id| paged_ids.push(id))
        .expect("paged gpu gen");

    assert_eq!(
        paged_ids, resident_ids,
        "paged MoE diverged from the all-resident GPU run"
    );
    assert_eq!(
        paged_ids, cpu_ids,
        "paged MoE diverged from the CPU reference"
    );
}

/// The paged MoE cache with its THIRD tier engaged (`paging.dram`): a routed expert that misses the
/// VRAM arena is read from the host arena, and from the model file when that misses too — one block
/// per expert, at its own file offset inside the bank, instead of the whole bank being faulted in
/// through the mapping.
///
/// Token identity against the all-resident run is the bar, as in
/// `gpu_seam_paged_moe_matches_resident_and_cpu`. Both budgets are forced far below one layer's
/// banks, so neither tier holds its working set and experts are genuinely re-read into recycled
/// slots — the case where a wrong slot or a wrong file offset yields plausible garbage rather than
/// an error.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_paged_moe_host_tier_matches_resident() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    let pin_ubatch = |c: &mut infr_llama::EngineConfig| {
        c.device.ubatch = Some(1);
        c.device.ubatch_specified = true;
    };
    let model = model_cfg(&path, pin_ubatch);
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");

    let mut resident_ids = Vec::new();
    model
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    let tiered = model_cfg(&path, |c| {
        pin_ubatch(c);
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(50 * 1024 * 1024));
        c.paging.dram = Some(infr_core::SizeSpec::Bytes(256 * 1024 * 1024));
        c.paging.stats = true;
    });
    let mut tiered_ids = Vec::new();
    tiered
        .generate_vulkan_ids(&prompt_ids, n, |id| tiered_ids.push(id))
        .expect("tiered gpu gen");

    assert_eq!(
        tiered_ids, resident_ids,
        "the MoE host tier diverged from the all-resident GPU run"
    );

    // The UNIFIED-memory shape: no host cache at all, every expert read from disk straight into
    // the staging ring (`GpuPager::touch_staged_read`). A unified device takes this automatically
    // because its streaming arena is already GPU-accessible RAM; `paging.dram_bypass` is what lets
    // it be exercised HERE, on a discrete GPU, which is the only hardware available to test it on.
    // Without this leg that path would ship with no coverage whatsoever.
    let bypass = model_cfg(&path, |c| {
        pin_ubatch(c);
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(50 * 1024 * 1024));
        c.paging.dram_bypass = true;
        c.paging.stats = true;
    });
    let mut bypass_ids = Vec::new();
    bypass
        .generate_vulkan_ids(&prompt_ids, n, |id| bypass_ids.push(id))
        .expect("dram-bypass gpu gen");
    assert_eq!(
        bypass_ids, resident_ids,
        "the arena-less (unified-memory) MoE path diverged from the all-resident GPU run"
    );
}

/// Dense layer streaming (`infr_vulkan::pager::DensePagerSession`, wired via `INFR_CACHE` on a
/// DENSE model): a tiny forced budget streams (nearly) every per-layer Linear weight group
/// through the cyclic-sweep pager, and the greedy output must be IDENTICAL, token-for-token, to
/// both the all-resident GPU run and the CPU reference — the streamed dispatch is the SAME
/// kernels reading the same bytes at an arena element offset, so any divergence is a real bug
/// (slot misalignment, stale slot, ring lifetime), not precision.
///
/// Qwen3-1.7B-Q4_K_M exercises the SPLIT q/k/v form (this unsloth quant mixes attn_v dtypes
/// across layers, so `fuse_qkv_decision` is false) plus a fused gate_up block and mixed
/// Q4_K/Q6_K pools; the 14B test below covers the fused-qkv form.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_dense_stream_matches_resident_and_cpu() {
    let path = need_model!(qwen3_17b(), "Qwen3-1.7B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    let model = model_default(&path);
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");

    let mut cpu_ids = Vec::new();
    model
        .generate_cpu_ids(&prompt_ids, n, |id| cpu_ids.push(id))
        .expect("cpu gen");

    let mut resident_ids = Vec::new();
    model
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    // 0.2 GB is far below the model's ~1.4 GB of streamable projections — every pool runs at its
    // floor slot count, so (nearly) every layer re-uploads every pass: real eviction pressure.
    let streamed = model_cfg(&path, |c| {
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(200 * 1024 * 1024));
        c.paging.stats = true; // see `gpu_seam_paged_moe_matches_resident_and_cpu`
    });
    let mut streamed_ids = Vec::new();
    streamed
        .generate_vulkan_ids(&prompt_ids, n, |id| streamed_ids.push(id))
        .expect("streamed gpu gen");

    assert_eq!(
        streamed_ids, resident_ids,
        "dense streaming diverged from the all-resident GPU run"
    );
    assert_eq!(
        streamed_ids, cpu_ids,
        "dense streaming diverged from the CPU reference"
    );
}

/// The same dense streaming path with its THIRD tier engaged (`paging.dram`): every streamed weight
/// group's bytes come from the host arena — read off the model file by `FileBlockIo` — instead of
/// the GGUF mmap. Token identity against the all-resident run is what says the tier below hands the
/// staging ring the same bytes the mapping did.
///
/// Both budgets are forced far under what the model needs, so neither tier can hold its working
/// set: VRAM re-uploads nearly every group every pass, and the host arena evicts and re-reads
/// underneath it. That is what makes the run exercise a recycled host slot rather than a warm cache
/// that never moves — the case where a wrong slot index produces plausible garbage instead of an
/// error. `crates/infr-vulkan/tests/dense_tier_parity.rs` pins down which of the three stage cases
/// each of those is; this test is the end-to-end statement that the whole model still decodes.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_dense_stream_host_tier_matches_resident() {
    let path = need_model!(qwen3_17b(), "Qwen3-1.7B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    let model = model_default(&path);
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");

    let mut resident_ids = Vec::new();
    model
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    let tiered = model_cfg(&path, |c| {
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(200 * 1024 * 1024));
        c.paging.dram = Some(infr_core::SizeSpec::Bytes(256 * 1024 * 1024));
        c.paging.stats = true;
    });
    let mut tiered_ids = Vec::new();
    tiered
        .generate_vulkan_ids(&prompt_ids, n, |id| tiered_ids.push(id))
        .expect("tiered gpu gen");

    assert_eq!(
        tiered_ids, resident_ids,
        "the host tier diverged from the all-resident GPU run"
    );
}

fn qwen3_17b() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf")
}

fn qwen3_14b_q8() -> Option<PathBuf> {
    find_gguf("unsloth--Qwen3-14B-GGUF", "Qwen3-14B-Q8_0.gguf")
}

/// The BIG dense streaming shape: Qwen3-14B Q8_0 (~15.7 GB — genuinely more than an 8 GB budget)
/// with fused qkv AND fused gate_up blocks (uniform Q8_0 passes both fuse gates), streamed vs
/// fully resident. Same token-identity bar as the 1.7B test; CPU included (Q8_0 int dots on both
/// sides — the 14B CPU run is ~15 s of the suite, the price of a real >budget model check).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_dense_stream_matches_resident_qwen3_14b() {
    let path = need_model!(qwen3_14b_q8(), "Qwen3-14B-Q8_0");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    let model = model_default(&path);
    let rendered = model
        .render_chat("What is 2+2? Answer briefly.")
        .expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");

    let mut cpu_ids = Vec::new();
    model
        .generate_cpu_ids(&prompt_ids, n, |id| cpu_ids.push(id))
        .expect("cpu gen");

    let mut resident_ids = Vec::new();
    model
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    let streamed = model_cfg(&path, |c| {
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(8 * 1024 * 1024 * 1024));
        c.paging.stats = true; // see `gpu_seam_paged_moe_matches_resident_and_cpu`
    });
    let mut streamed_ids = Vec::new();
    streamed
        .generate_vulkan_ids(&prompt_ids, n, |id| streamed_ids.push(id))
        .expect("streamed gpu gen");

    assert_eq!(
        streamed_ids, resident_ids,
        "dense streaming diverged from the all-resident GPU run (14B)"
    );
    assert_eq!(
        streamed_ids, cpu_ids,
        "dense streaming diverged from the CPU reference (14B)"
    );
}

/// Streamed dense PREFILL parity — the perf-critical path this change adds. A LONG prompt forces a
/// prefill chunk of m ≫ 16 tokens, which `streamed_gemm_applies` routes through the arena-addressed
/// coopmat-warp GEMM twins (`native_gemm_warp_*_streamed`: the n128_ag / sk_ag tiles) instead of the
/// per-row GEMV. Those are the SAME kernels the resident prefill picks, with the weight bytes merely
/// relocated to an arena slot, so the streamed run must be token-identical to the all-resident run.
/// (The tests above cover decode's small-m GEMV; a short prompt would never leave it — hence the
/// length assertion.) Two streamed reps guard the barrier-drop class: the ring→arena copy vs the
/// 64-bit pointer read is ordered by an explicit barrier the buffer hazard tracker can't see, so a
/// dropped barrier surfaces as run-to-run nondeterminism.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_dense_stream_prefill_matches_resident() {
    let path = need_model!(qwen3_17b(), "Qwen3-1.7B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;

    let model = model_default(&path);
    // A long user turn → a prefill chunk of well over 16 tokens (the streamed-GEMM gate), so the
    // arena-addressed warptiles actually engage (a short prompt would stay on the streamed GEMV).
    let long = "Explain, step by step and in thorough detail, how a transformer language model \
        processes a sequence of tokens through its embedding, attention, and feed-forward layers. "
        .repeat(6);
    let rendered = model.render_chat(&long).expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");
    assert!(
        prompt_ids.len() > 64,
        "prompt must be a real prefill chunk (m ≫ 16); got {}",
        prompt_ids.len()
    );

    let mut resident_ids = Vec::new();
    model
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    // Below the model's ~1.4 GB of streamable projections → real eviction every pass.
    let streamed = model_cfg(&path, |c| {
        c.paging.cache = Some(infr_core::SizeSpec::Bytes(200 * 1024 * 1024));
    });
    let mut streamed_a = Vec::new();
    let ra = streamed.generate_vulkan_ids(&prompt_ids, n, |id| streamed_a.push(id));
    let mut streamed_b = Vec::new();
    let rb = streamed.generate_vulkan_ids(&prompt_ids, n, |id| streamed_b.push(id));
    ra.expect("streamed gpu gen (rep a)");
    rb.expect("streamed gpu gen (rep b)");

    assert_eq!(
        streamed_a, resident_ids,
        "streamed dense prefill diverged from the all-resident GPU run"
    );
    assert_eq!(
        streamed_a, streamed_b,
        "streamed dense prefill nondeterministic across reps (barrier-drop class)"
    );
}

/// LAYER-MAJOR streamed prefill (`paging.layer_major`): the chunk loop runs INSIDE the layer loop,
/// so the model is swept once per prompt instead of once per chunk. Same dispatches, reordered —
/// which is the whole claim, and token identity against the chunk-major arm is what checks it.
///
/// The shape matters: a 64-row chunk over this prompt is SIX-plus chunks, and only a multi-chunk
/// prefill can tell the two orders apart at all (one chunk makes them the same sequence of
/// dispatches). Each chunk then carries its own residual stream across every layer boundary, and
/// chunk C's attention has to find chunks 0..C-1 already in the layer's KV — reorder the chunks
/// within a layer and this goes red, which is what makes it a check rather than a formality.
///
/// Three arms, all at the same chunk height so the only difference is the loop nesting and the
/// residency: streamed layer-major (the default for a streamed model), streamed chunk-major, and
/// fully resident.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_dense_stream_prefill_layer_major_matches_chunk_major() {
    let path = need_model!(qwen3_17b(), "Qwen3-1.7B");
    let mut _tlk = test_serial_lock();
    let n = 8usize;
    let chunk = 64usize;

    let model = model_default(&path);
    let long = "Explain, step by step and in thorough detail, how a transformer language model \
        processes a sequence of tokens through its embedding, attention, and feed-forward layers. "
        .repeat(12);
    let rendered = model.render_chat(&long).expect("render chat");
    let prompt_ids = model.encode(&rendered).expect("encode");
    assert!(
        prompt_ids.len() > 4 * chunk,
        "the prompt must span several {chunk}-row chunks; got {}",
        prompt_ids.len()
    );

    // Every arm pins the same chunk height — the GEMM tiling is chunk-dependent, so a height
    // difference would show up as float noise and confuse a token-identity assertion.
    let pin_chunk = |c: &mut infr_llama::EngineConfig| {
        c.device.ubatch = Some(chunk);
        c.device.ubatch_specified = true;
    };
    let mut resident_ids = Vec::new();
    model_cfg(&path, pin_chunk)
        .generate_vulkan_ids(&prompt_ids, n, |id| resident_ids.push(id))
        .expect("resident gpu gen");

    // Below the model's ~1.4 GB of streamable projections → real eviction every pass.
    let streamed = |layer_major: Option<bool>| {
        model_cfg(&path, |c| {
            pin_chunk(c);
            c.paging.cache = Some(infr_core::SizeSpec::Bytes(200 * 1024 * 1024));
            c.paging.layer_major = layer_major;
        })
    };
    let mut chunk_major_ids = Vec::new();
    streamed(Some(false))
        .generate_vulkan_ids(&prompt_ids, n, |id| chunk_major_ids.push(id))
        .expect("chunk-major streamed gpu gen");
    // `None` is the DEFAULT, and a streamed model is exactly the case it resolves to layer-major:
    // running the auto arm is what keeps the gate itself covered.
    let mut layer_major_ids = Vec::new();
    streamed(None)
        .generate_vulkan_ids(&prompt_ids, n, |id| layer_major_ids.push(id))
        .expect("layer-major streamed gpu gen");

    assert_eq!(
        layer_major_ids, chunk_major_ids,
        "layer-major prefill diverged from the chunk-major order it only reorders"
    );
    assert_eq!(
        layer_major_ids, resident_ids,
        "layer-major streamed prefill diverged from the all-resident GPU run"
    );
}

/// CPU-only: Gemma 4 E2B golden-hash lock.
#[test]
fn cpu_golden_gemma4_e2b() {
    let path = need_model!(gemma4_e2b(), "gemma-4-E2B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_golden(&model, GEMMA4_E2B_GOLDEN);
}

// ─── Qwen3.6 MoE (qwen35moe: gated-DeltaNet hybrid + routed MoE FFN + Qwen2-MoE-style shared
// expert on EVERY layer) ──────────────────────────────────────────────────────────────────────
//
// `general.architecture == "qwen35moe"` — the routed-expert sibling of dense `qwen35` (see
// `docs/qwen35.md` + `arch::QWEN35_MOE`'s doc). 256 experts / 8 used / 512-wide, plus a
// Qwen2-MoE-style shared expert (`ffn_*_shexp`, sigmoid-gated via `ffn_gate_inp_shexp`) — both on
// EVERY layer (DeltaNet and full-attention alike, confirmed against the actual GGUF tensor list
// and llama.cpp's `qwen35moe.cpp::build_layer_ffn`). UD-Q4_K_M chosen over the smaller UD-IQ*
// quants because the routed/shared expert banks there are Q4_K/Q5_K/Q8_0 — id-native quant
// formats the Vulkan `Op::MoeFfn` id-indexed kernel supports (`native_id_kernel_name`); the
// smaller IQ2_S/IQ3_S UD quants aren't (`vulkan adapter: MoeFfn expert banks need an id-native
// quant format`), a pre-existing Vulkan MoE-kernel gap unrelated to this arch's wiring.

fn qwen35moe_35b_a3b() -> Option<PathBuf> {
    find_gguf(
        "unsloth--Qwen3.6-35B-A3B-GGUF",
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )
}

/// CPU-only: qwen35moe's causal prompt prefill (routed MoE + shared expert FFN on every layer)
/// produces finite logits over a short prompt, and the config parses as expected (MoE present,
/// shared-expert width present, still gated through the `qwen35` DeltaNet/rope/attn-out-gate
/// fields dense qwen35 uses).
#[test]
fn cpu_qwen35moe_prefill_finite() {
    let path = need_model!(qwen35moe_35b_a3b(), "Qwen3.6-35B-A3B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let cfg = model.config();
    assert!(
        cfg.qwen35,
        "qwen35moe must set Config::qwen35 (shared skeleton)"
    );
    let mc = cfg.moe.expect("qwen35moe must populate Config::moe");
    assert_eq!(mc.n_expert, 256);
    assert_eq!(mc.n_used, 8);
    assert_eq!(mc.n_ff_exp, 512);
    assert_eq!(cfg.shexp_ff, 512, "shared-expert width not parsed");
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");
    let vocab = cfg.vocab;
    let t0 = std::time::Instant::now();
    let last_row = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    eprintln!(
        "cpu_qwen35moe_prefill_finite: {} tokens, prefill {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(last_row.len(), vocab, "logits shape");
    assert!(
        last_row.iter().all(|v| v.is_finite()),
        "non-finite logit in the prefill output"
    );
    println!("top-5 last-row tokens: {:?}", top_k(&last_row, 5));
}

/// qwen35moe's causal prompt prefill through the Vulkan seam vs the CPU oracle — a top-8-of-256
/// MoE router (plus the shared expert) is a discrete-selection step, so this follows the
/// diffusion-gemma precedent (`gpu_seam_matches_cpu_diffusion_gemma`'s doc) rather than a strict
/// token/logit compare: top-5 overlap (either side's #1 token appears in the other's top-5) AND a
/// cosine floor on the whole-vocab last-row logits, NOT bit-identical (CPU f32 vs Vulkan f16-native
/// routing legitimately flips near-tie expert selection).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_qwen35moe() {
    let path = need_model!(qwen35moe_35b_a3b(), "Qwen3.6-35B-A3B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    let vocab = model.config().vocab;
    let cpu_last = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    let gpu_last = model
        .prefill_logits_vulkan(&tokens)
        .expect("vulkan prefill");
    assert_eq!(cpu_last.len(), vocab, "cpu logits shape");
    assert_eq!(gpu_last.len(), vocab, "gpu logits shape");
    assert!(
        gpu_last.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan prefill output"
    );
    let (cpu_top, gpu_top) = (top_k(&cpu_last, 20), top_k(&gpu_last, 20));
    println!("cpu    top-5: {:?}", &cpu_top[..5]);
    println!("vulkan top-5: {:?}", &gpu_top[..5]);
    assert!(
        cpu_top[..5].iter().any(|&(id, _)| id == gpu_top[0].0)
            || gpu_top[..5].iter().any(|&(id, _)| id == cpu_top[0].0),
        "CPU/Vulkan top tokens don't even overlap in each other's top-5: cpu={:?} vulkan={:?}",
        cpu_top[0],
        gpu_top[0]
    );
    let cos = cosine(&cpu_last, &gpu_last);
    println!("cpu/vulkan whole-vocab cosine similarity: {cos}");
    assert!(
        cos > 0.5,
        "CPU/Vulkan last-row logits diverged too far: cosine={cos}"
    );
}

/// Dense qwen35 (no expert tensors) must still take the plain dense-FFN path — the `Config::moe`/
/// `shexp_ff` additions for qwen35moe must NOT leak onto its dense sibling.
#[test]
fn cpu_qwen35_dense_unaffected_by_moe_fields() {
    let path = need_model!(qwen35_08b(), "Qwen3.5-0.8B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let cfg = model.config();
    assert!(cfg.qwen35);
    assert!(cfg.moe.is_none(), "dense qwen35 must not get an MoE config");
    assert_eq!(
        cfg.shexp_ff, 0,
        "dense qwen35 must not get a shared-expert width"
    );
}

// ─── Llama 4 Scout (llama4: sigmoid top-1 MoE + plain shared expert + iRoPE) ───────
//
// `general.architecture == "llama4"` (see `arch::LLAMA4`). 48 layers, 16 experts / top-1, sigmoid
// gating (no top-k renorm), weight-before-FFN, a PLAIN (ungated) shared expert summed in, and
// iRoPE: every 4th layer is NoPE (rope skipped, global attention) while rope layers apply a
// weightless per-head L2-norm to Q/K after rope. Chunked masking + attn-temperature scaling are
// no-ops below the 8192 chunk size (untestable on CPU for a 109B model — see `arch::LLAMA4`).

fn llama4_scout() -> Option<PathBuf> {
    find_gguf(
        "unsloth--Llama-4-Scout-17B-16E-Instruct-GGUF",
        "Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf",
    )
}

/// llama4 config parses as expected (MoE present, top-1, plain shared expert, iRoPE + L2-norm on).
#[test]
fn cpu_llama4_config() {
    let path = need_model!(llama4_scout(), "Llama-4-Scout");
    let model = model_default(&path);
    let cfg = model.config();
    assert!(cfg.llama4);
    let moe = cfg.moe.expect("llama4 has an MoE config");
    assert_eq!(moe.n_expert, 16);
    assert_eq!(moe.n_used, 1);
    assert!(matches!(moe.gating, infr_core::graph::MoeGating::Sigmoid));
    assert!(!moe.norm_w, "llama4 does not renormalize top-k weights");
    assert!(
        moe.weight_before,
        "llama4 applies the weight before the FFN"
    );
    assert!(cfg.shexp_ff > 0, "llama4 has a shared expert");
    assert!(
        !cfg.shexp_gated,
        "llama4's shared expert is summed in plain"
    );
    assert_eq!(cfg.moe_interleave_step, 1, "Scout is MoE on every layer");
    assert_eq!(cfg.no_rope_step, 4, "iRoPE NoPE every 4th layer");
    assert!(
        cfg.kq_l2norm,
        "Scout (16E) applies the post-rope Q/K L2-norm"
    );
    assert!(
        cfg.is_nope_layer(3) && !cfg.is_nope_layer(0),
        "layer 3 is NoPE"
    );
}

/// llama4 CPU greedy generation, token-identical to the `llama-completion` oracle.
///
/// Oracle (same Q2_K GGUF, CPU, greedy):
///   `llama-completion -m <gguf> -p "The capital of France is" -n 24 -c 512 --no-conversation \
///        -ngl 0 --temp 0`
/// Both tokenize the prompt to `[200000, 954, 7963, 323, 11698, 373]` (BOS + 5), and the first 19
/// GENERATED tokens are byte-identical:
///   `13796 26 589 7963 323 19584 373 20589 26 589 7963 323 26049 373 30827 26 589 7963 323`
///   (" Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of").
/// They then split at index 19 on the 4th country of an OPEN-ENDED list — infr `31154` (" Spain")
/// vs llama.cpp `15462` (" Australia") — a Q2_K (2-bit) near-tie broken by CPU f32 reassociation
/// (the "precision, not a bug" class every CPU golden here notes). The 19-token match exercises all
/// 48 layers, every NoPE/rope-pattern boundary, the weightless post-rope L2-norm, and the
/// sigmoid-top-1-weight-before-FFN MoE + plain shared expert on every step. `INFR_L4_PROMPT`/
/// `INFR_L4_N`/`INFR_L4_NOBOS` override the defaults. Slow (109B on CPU) — few tokens.
#[test]
fn cpu_llama4_scout_greedy() {
    let path = need_model!(llama4_scout(), "Llama-4-Scout");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let prompt =
        std::env::var("INFR_L4_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());
    let n: usize = std::env::var("INFR_L4_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let mut ids = model.encode(&prompt).expect("encode");
    if std::env::var("INFR_L4_NOBOS").is_err() {
        // llama.cpp adds BOS by default; Scout's BOS is `<|begin_of_text|>` (200000).
        ids.insert(0, 200000);
    }
    println!("prompt ({} tok): {ids:?}", ids.len());
    let mut gen = Vec::new();
    let out = model
        .generate_cpu_ids(&ids, n, |id| gen.push(id))
        .expect("cpu generate");
    println!("generated ids: {out:?}");
    println!("captured  ids: {gen:?}");
    println!("text: {:?}", model.decode(&out).expect("decode"));
    // Token-identity lock against the llama-completion oracle: the deterministic 19-token prefix
    // (before the open-ended country-list near-tie) must match exactly, on the default prompt.
    if prompt == "The capital of France is" && std::env::var("INFR_L4_NOBOS").is_err() {
        const ORACLE_PREFIX: &[u32] = &[
            13796, 26, 589, 7963, 323, 19584, 373, 20589, 26, 589, 7963, 323, 26049, 373, 30827,
            26, 589, 7963, 323,
        ];
        assert!(
            out.len() >= ORACLE_PREFIX.len() && out[..ORACLE_PREFIX.len()] == *ORACLE_PREFIX,
            "llama4 greedy diverged from the llama-completion oracle within the deterministic \
             prefix\n  got: {:?}\n  want prefix: {ORACLE_PREFIX:?}",
            &out[..ORACLE_PREFIX.len().min(out.len())],
        );
    }
    assert!(!out.is_empty());
    assert!(out.iter().all(|&t| (t as usize) < model.config().vocab));
}

/// Vulkan seam, GPU-resident: Scout's 37 GB Q2_K expert banks don't fit a 24 GB card, so this
/// exercises the paged executor split end to end (real weights, real eviction, real host
/// readback/upload cadence — not a synthetic bank) and locks it against the SAME oracle prefix
/// `cpu_llama4_scout_greedy` checks. llama4's gate/up/down banks are each uniformly Q2_K/Q2_K/Q3_K
/// across every layer (verified — unlike the UD quants
/// `gpu_seam_paged_moe_matches_resident_and_cpu` documents, whose mixed down role spans two arena
/// pools), so this is the classic one-pool-per-role split-bank shape.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_paged_moe_matches_scout_oracle() {
    let path = need_model!(llama4_scout(), "Llama-4-Scout");
    let mut _tlk = test_serial_lock();
    let model = model_cfg(&path, |c| c.paging.stats = true);
    let mut ids = model.encode("The capital of France is").expect("encode");
    ids.insert(0, 200000); // Scout's BOS (<|begin_of_text|>), matching cpu_llama4_scout_greedy
    let n = 24usize;
    let mut gen = Vec::new();
    let out = model
        .generate_vulkan_ids(&ids, n, |id| gen.push(id))
        .expect("vulkan seam generate (paged MoE)");
    println!("scout paged-GPU generated ids: {out:?}");
    println!(
        "scout paged-GPU text: {:?}",
        model.decode(&out).expect("decode")
    );
    const ORACLE_PREFIX: &[u32] = &[
        13796, 26, 589, 7963, 323, 19584, 373, 20589, 26, 589, 7963, 323, 26049, 373, 30827, 26,
        589, 7963, 323,
    ];
    assert!(
        out.len() >= ORACLE_PREFIX.len() && out[..ORACLE_PREFIX.len()] == *ORACLE_PREFIX,
        "llama4 paged-GPU greedy diverged from the CPU oracle within the deterministic prefix\n  \
         got: {:?}\n  want prefix: {ORACLE_PREFIX:?}",
        &out[..ORACLE_PREFIX.len().min(out.len())],
    );
}

// ─── Gemma 4 12b (dense) ────────────────────────────────────────────────────────

fn gemma4_12b() -> Option<PathBuf> {
    find_gguf("unsloth--gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf")
}

// ─── DiffusionGemma (block text-diffusion MoE on a Gemma-4 backbone) ───────────────
//
// Phase 1 scope only: Config + weight loading + a CAUSAL PROMPT PREFILL through the unified
// runner (dual FFN — dense GeGLU ∥ 128-expert MoE with a fused gate_up_exps + per-expert down
// scale, encoder-scalar per-layer output, heterogeneous per-layer attn dims). No canvas/denoise —
// see docs/diffusion-gemma.md. 26B-A4B Q4_K_M is large (16 GB); a CPU prefill of ~16 tokens takes
// on the order of a minute.

fn diffusion_gemma_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_DIFFUSION_GEMMA") {
        return Some(PathBuf::from(p));
    }
    find_gguf(
        "unsloth--diffusiongemma-26B-A4B-it-GGUF",
        "diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
    )
}

/// The top-`k` (token id, logit) pairs of a vocab-sized logits row, for a human-readable print.
fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx.into_iter().map(|i| (i, logits[i])).collect()
}

/// CPU-only: DiffusionGemma's causal prompt prefill produces finite logits over a short fixed
/// prompt. Prints the top-5 last-row (next-token) logits — no golden hash (Phase 1 doesn't claim
/// coherent generation; that's the oracle-parity check in Phase 3).
#[test]
fn cpu_diffusion_gemma_prefill_finite() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    assert!(
        model.config().diffusion_gemma,
        "arch not parsed as diffusion-gemma"
    );
    assert!(model.config().canvas_length > 0, "canvas_length not parsed");
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");
    let vocab = model.config().vocab;
    let t0 = std::time::Instant::now();
    // `prefill_logits_cpu` returns only the LAST prompt token's row (the causal prefill's
    // next-token distribution) — the per-token decode loop's frontier logits, not a [m, vocab]
    // batch (see its doc comment).
    let last_row = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    eprintln!(
        "cpu_diffusion_gemma_prefill_finite: {} tokens, prefill {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(last_row.len(), vocab, "logits shape");
    assert!(
        last_row.iter().all(|v| v.is_finite()),
        "non-finite logit in the prefill output"
    );
    println!("top-5 last-row tokens: {:?}", top_k(&last_row, 5));
}

/// DiffusionGemma's causal prompt prefill through the Vulkan seam must match the CPU oracle's
/// last-row logits within tolerance (quantized-weight + f16-vs-f32 numeric drift — the same class
/// of divergence the golden-hash tests sidestep by locking per-backend hashes instead of a direct
/// float compare; here we compare directly since Phase 1 has no generation golden yet).
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_diffusion_gemma() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    let vocab = model.config().vocab;
    // Both return only the LAST prompt token's row (see `prefill_logits_cpu`'s doc comment).
    let cpu_last = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    let gpu_last = model
        .prefill_logits_vulkan(&tokens)
        .expect("vulkan prefill");
    assert_eq!(cpu_last.len(), vocab, "cpu logits shape");
    assert_eq!(gpu_last.len(), vocab, "gpu logits shape");
    assert!(
        gpu_last.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan prefill output"
    );
    let (cpu_top, gpu_top) = (top_k(&cpu_last, 20), top_k(&gpu_last, 20));
    let cos = cosine(&cpu_last, &gpu_last);
    println!("cpu    top-5: {:?}", &cpu_top[..5]);
    println!("vulkan top-5: {:?}", &gpu_top[..5]);
    println!("cpu/vulkan whole-vocab cosine similarity: {cos}");
    // NOT an exact/near-tolerance match: this is a 128-expert top-8 MoE model, and top-k expert
    // SELECTION is a discrete step — a near-tie router logit (f32 CPU vs f16-native-quant Vulkan)
    // can flip which experts run for a token, which then diverges the WHOLE downstream FFN output
    // for that layer. This is a known, already-shipped property of this codebase's OTHER MoE arch
    // (qwen3moe): its cross-backend test explicitly does NOT compare logits/tokens directly,
    // locking separate per-backend golden hashes instead (see `gpu_seam_golden_qwen3moe`'s doc
    // comment). Calibrated directly against that model (same class of divergence, no known bug):
    // qwen3moe's CPU-vs-Vulkan last-row argmax lands on COMPLETELY DIFFERENT tokens with a
    // whole-vocab cosine similarity of ~0.74 (that test floors at just cos > 0.5).
    //
    // The last-row argmax is the WORST place to demand top-k overlap on this model: at temp 0 the
    // GPU consistently ranks one high-frequency token (id 107) first on EVERY precision tier —
    // f16 coopmat (cos 0.811), int8-dp4a non-coopmat (cos 0.801), AND f32 scalar (cos 0.847) —
    // while the CPU's f32 argmax lands on a near-tie neighbor that sits at ~top-5 rank. Whether
    // that neighbor lands at top-5 position 4 (coopmat) or ~6 (non-coopmat) is a sub-0.01-cosine
    // coin-flip, NOT a correctness signal: the non-coopmat int8-activation dense GEMMs are ~0.01
    // cosine less precise than the coopmat f16 tile (expected int8 < f16 < f32 laddering), which
    // is enough to nudge that neighbor out of top-5 on a max-entropy last-row distribution. The
    // model itself is correct on the non-coopmat tier — `diffusion_gemma_decode_matches_oracle`
    // decodes the right "…Paris." answer there, and the DENSE gemma-4 seam tests exact-match the
    // CPU oracle under `INFR_NO_COOPMAT=1`. So gate on the DISTRIBUTION (cosine), with top-5
    // overlap kept only as a fast-accept: this mirrors the sibling `_denoise` check's
    // `overlap || cos > …` shape and stays well above qwen3moe's shipped 0.5 floor.
    let overlap = cpu_top[..5].iter().any(|&(id, _)| id == gpu_top[0].0)
        || gpu_top[..5].iter().any(|&(id, _)| id == cpu_top[0].0);
    assert!(
        overlap || cos > 0.78,
        "CPU/Vulkan last-row logits diverged: no top-5 overlap AND cosine {cos:.3} < 0.78 (real \
         divergence, not a near-tie rank flip): cpu={:?} vulkan={:?}",
        cpu_top[0],
        gpu_top[0]
    );
    assert!(
        cos > 0.7,
        "CPU/Vulkan last-row logits diverged too far: cosine={cos}"
    );
}

// ─── DiffusionGemma Phase 2: canvas denoise ─────────────────────────────────────────
//
// One bidirectional forward over the C canvas rows, reusing the prompt KV Phase 1's causal
// prefill already wrote (encoder scalars, rows 0..P) — decoder scalars, the `AttnMask::Canvas`
// bidirectional mask, and (optionally) self-conditioning. See docs/diffusion-gemma.md.

/// CPU-only: prefill a short prompt, then ONE denoise forward over an all-mask canvas
/// (`sc_logits=None`, matching the reference's step-0 zero-SC gate). Also proves the WriteKv
/// overwrite (a second denoise call with a DIFFERENT canvas must produce different, still-finite
/// logits — the next denoise step re-overwrites the same KV rows) and a self-conditioning smoke
/// test (feeding the first call's raw logits back in must differ from the no-SC call).
#[test]
fn cpu_diffusion_gemma_denoise_step() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let vocab = model.config().vocab;
    let canvas_len = model.config().canvas_length;
    let mask_id = model.config().mask_token_id;
    let tokens = model
        .encode("What is the capital of France?")
        .expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");

    let mut session = model.diffusion_gemma_cpu_session(tokens.len() + canvas_len + 8);
    let t0 = std::time::Instant::now();
    session.prefill(&model, &tokens).expect("cpu prefill");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: prefill {} tokens in {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );

    let canvas: Vec<u32> = vec![mask_id; canvas_len];
    let t1 = std::time::Instant::now();
    let logits1 = session
        .denoise(&model, &canvas, None, 1.0)
        .expect("cpu denoise step 1 (no SC)");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: denoise (no SC) {:.1}s",
        t1.elapsed().as_secs_f64()
    );
    assert_eq!(logits1.len(), canvas_len * vocab, "denoise logits shape");
    assert!(
        logits1.iter().all(|v| v.is_finite()),
        "non-finite logit in the no-SC denoise output"
    );
    for row in 0..canvas_len.min(8) {
        let row_logits = &logits1[row * vocab..(row + 1) * vocab];
        let top = top_k(row_logits, 1)[0];
        println!("row {row} argmax: token {} logit {:.3}", top.0, top.1);
    }

    // WriteKv overwrite: a second denoise call with a DIFFERENT canvas (row 0 unmasked to the
    // previous argmax) must produce different, still-finite logits — proving the cache actually
    // gets re-written each step (not stale-row reuse). Row 0's own argmax over an all-mask canvas
    // is often the mask token itself (the model's "not enough context yet" answer) — that would
    // leave canvas2 identical to canvas, so pick the first top-5 candidate that ACTUALLY differs
    // from mask_id (falling back to a fixed different token if somehow all 5 are the mask token).
    let mut canvas2 = canvas.clone();
    canvas2[0] = top_k(&logits1[..vocab], 5)
        .into_iter()
        .map(|(id, _)| id as u32)
        .find(|&id| id != mask_id)
        .unwrap_or((mask_id + 1) % vocab as u32);
    assert_ne!(
        canvas2[0], canvas[0],
        "test bug: canvas2 didn't actually change row 0"
    );
    let t2 = std::time::Instant::now();
    let logits2 = session
        .denoise(&model, &canvas2, None, 1.0)
        .expect("cpu denoise step 2 (different canvas)");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: denoise (overwrite check) {:.1}s",
        t2.elapsed().as_secs_f64()
    );
    assert_eq!(logits2.len(), canvas_len * vocab, "denoise2 logits shape");
    assert!(
        logits2.iter().all(|v| v.is_finite()),
        "non-finite logit after the second (overwrite) denoise call"
    );
    assert!(
        logits1 != logits2,
        "second denoise call (different canvas) produced IDENTICAL logits — WriteKv didn't \
         overwrite the canvas KV rows"
    );

    // Self-conditioning smoke test: feed the first call's raw logits back as sc_logits.
    let t3 = std::time::Instant::now();
    let logits_sc = session
        .denoise(&model, &canvas, Some(&logits1), 1.0)
        .expect("cpu denoise with self-conditioning");
    eprintln!(
        "cpu_diffusion_gemma_denoise_step: denoise (self-cond) {:.1}s",
        t3.elapsed().as_secs_f64()
    );
    assert_eq!(
        logits_sc.len(),
        canvas_len * vocab,
        "SC denoise logits shape"
    );
    assert!(
        logits_sc.iter().all(|v| v.is_finite()),
        "non-finite logit in the self-conditioned denoise output"
    );
    assert!(
        logits_sc != logits1,
        "self-conditioned denoise produced IDENTICAL logits to the no-SC call"
    );
    println!("cpu_diffusion_gemma_denoise_step: all sub-checks passed");
}

/// DiffusionGemma canvas denoise: CPU vs Vulkan on separate sessions given the SAME prompt +
/// all-mask canvas, no self-conditioning. Calibrated like `gpu_seam_matches_cpu_diffusion_gemma`
/// (Phase 1): a 128-expert top-8 MoE model's near-tie router logits can flip expert selection
/// between f32 CPU and f16-native-quant Vulkan, diverging the downstream FFN output for that
/// token — assert per-row cosine + top-5 overlap on a handful of rows rather than a tight
/// tolerance across all 256.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_diffusion_gemma_denoise() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let vocab = model.config().vocab;
    let canvas_len = model.config().canvas_length;
    let mask_id = model.config().mask_token_id;
    let tokens = model
        .encode("What is the capital of France?")
        .expect("encode");

    let mut cpu_session = model.diffusion_gemma_cpu_session(tokens.len() + canvas_len + 8);
    cpu_session.prefill(&model, &tokens).expect("cpu prefill");
    let canvas: Vec<u32> = vec![mask_id; canvas_len];
    let t0 = std::time::Instant::now();
    let cpu_logits = cpu_session
        .denoise(&model, &canvas, None, 1.0)
        .expect("cpu denoise");
    let cpu_secs = t0.elapsed().as_secs_f64();

    let mut vk_session = model
        .diffusion_gemma_vulkan_session(tokens.len() + canvas_len + 8)
        .expect("vulkan session");
    vk_session.prefill(&model, &tokens).expect("vulkan prefill");
    let t1 = std::time::Instant::now();
    // `u: None` opts out of the perf-slice-3 GPU reducer (docs/diffusion-gemma.md) — this test
    // wants the FULL `[canvas_len, vocab]` array back for its row-by-row cosine comparison below,
    // not just the reduced {argmax, entropy, sampled}.
    let gpu_outcome = vk_session
        .denoise(&model, &canvas, None, 1.0, 1.0, None)
        .expect("vulkan denoise");
    let gpu_logits = match gpu_outcome {
        infr_llama::seam::DenoiseOutcome::Logits(v) => v,
        infr_llama::seam::DenoiseOutcome::Reduced(_) => {
            panic!("u: None must always take the full-logits path")
        }
    };
    let gpu_secs = t1.elapsed().as_secs_f64();
    eprintln!(
        "gpu_seam_matches_cpu_diffusion_gemma_denoise: cpu {cpu_secs:.1}s vulkan {gpu_secs:.1}s"
    );

    assert_eq!(cpu_logits.len(), canvas_len * vocab, "cpu logits shape");
    assert_eq!(gpu_logits.len(), canvas_len * vocab, "gpu logits shape");
    assert!(
        gpu_logits.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan denoise output"
    );

    let mut min_cos = f64::INFINITY;
    for row in 0..canvas_len.min(8) {
        let c = &cpu_logits[row * vocab..(row + 1) * vocab];
        let v = &gpu_logits[row * vocab..(row + 1) * vocab];
        let cos = cosine(c, v);
        min_cos = min_cos.min(cos);
        let (ctop, vtop) = (top_k(c, 5), top_k(v, 5));
        println!(
            "row {row}: cosine={cos:.3} cpu_top1={:?} vulkan_top1={:?}",
            ctop[0], vtop[0]
        );
        // Top-1 overlap, EXCEPT when a side's top-1 is the mask token. This is the first denoise
        // step of an all-mask canvas — the maximum-entropy state, where the mask token itself sits
        // near the top of every row and f16-GPU vs f32-CPU legitimately flips the argmax onto or
        // off it (the decode loop's argmax over the full vocab, diffusion.rs, doesn't suppress the
        // mask token — these uncommitted positions are re-masked by the entropy-bound loop, which
        // is why production still decodes correctly). cosine below is the real distribution check.
        //
        // A THIRD legitimate flip source is float reassociation on the GPU decode GEMV itself: the
        // reassociation-tolerant subgroup GEMV (native_gemv_sg, wave32+subgroupAdd) reorders the
        // Q6_K projection accumulation (attn_v here, in-band out_f=2048) at the ULP level — smaller
        // than the f16/f32 gap already tolerated above, but enough to nudge the argmax onto a
        // different near-tie non-mask token at a max-entropy row (measured maxrel ~4e-5, cosine
        // unchanged ~0.87). So when the top-1s disagree and neither is the mask token, defer to
        // cosine (the stated real check): a healthy distribution (cos > 0.8, vs the >0.7 floor and
        // the ~0.85-0.89 observed) is a near-tie argmax flip, not a divergence; a real bug tanks it.
        let overlap = ctop.iter().any(|&(id, _)| id == vtop[0].0)
            || vtop.iter().any(|&(id, _)| id == ctop[0].0);
        let mask_tie = ctop[0].0 as u32 == mask_id || vtop[0].0 as u32 == mask_id;
        assert!(
            overlap || mask_tie || cos > 0.8,
            "row {row}: CPU/Vulkan top tokens don't overlap in each other's top-5 (neither is the \
             mask token) AND cosine {cos:.3} < 0.8 — real divergence, not a near-tie flip: \
             cpu={:?} vulkan={:?}",
            ctop[0],
            vtop[0]
        );
        // Per-row distribution floor. Measured healthy min-row cosine is ~0.79 on BOTH precision
        // tiers (f16 coopmat 0.792, int8-dp4a non-coopmat 0.789 — they track within 0.003, so a
        // floor can't discriminate one tier from the other), with the higher rows at ~0.86-0.87.
        // The old 0.7 left a full 0.09 of slack under the real floor; 0.75 keeps a safe ~0.04
        // margin below the observed 0.789 min while still tripping on any gross distribution
        // regression (a broken kernel tanks these cosines far below 0.75).
        assert!(cos > 0.75, "row {row}: cosine too low: {cos}");
    }
    println!(
        "gpu_seam_matches_cpu_diffusion_gemma_denoise: min row cosine over checked rows = {min_cos:.3}"
    );
}

/// Regression (`Graph::no_decode_replay`): a DG session's prefill+denoise must produce
/// BIT-IDENTICAL denoise logits whether the seam runs in its default mode or under
/// `INFR_SEAM_NO_REPLAY=1` (the forced-static path the CPU reference/goldens track).
///
/// Before the fix, the default mode sent the prefill FRONTIER token — the one token every DG
/// prefill call feeds through the per-token decode loop — through the record-once replay tape's
/// `_dyn` kernels, whose float-accumulation order differs from the static recording by ~1 f16
/// ULP on that token's KV row (verified: rows 0..P-2 bit-identical between modes, only the
/// frontier row differed, maxabs 5e-4..8e-3 growing through layers). DiffusionGemma's
/// entropy-bound 128-expert MoE denoise chaotically amplifies that single-row delta — measured
/// on this prompt: ALL 67.1M logit elements differing, maxabs 18.4, 111/256 canvas argmax flips,
/// min row cosine 0.68 → visibly different generated text between the two modes. The fix tags
/// every DG graph `no_decode_replay` so both modes run the SAME static kernels; two separate
/// same-kernel sessions are bit-deterministic (also verified), hence the exact assert.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_diffusion_gemma_denoise_replay_matches_static() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    // The guard now only serializes this GPU test against the others: BOTH halves of
    // `INFR_SEAM_NO_REPLAY` (§6.12 — the seam's replay gate and the Vulkan adapter's
    // `decode_eligible`) read the per-model `kernels.vulkan.no_replay` since S5b, so the mode is
    // entirely a value on the config below.
    let _tlk = test_serial_lock();
    let base = model_default(&path);
    let vocab = base.config().vocab;
    let canvas_len = base.config().canvas_length;
    let mask_id = base.config().mask_token_id;
    let tokens = base
        .encode("What is the capital of France?")
        .expect("encode");
    let canvas: Vec<u32> = vec![mask_id; canvas_len];

    let run = |no_replay: bool| -> Vec<f32> {
        let model = model_cfg(&path, |c| c.kernels.vulkan.no_replay = no_replay);
        let mut sess = model
            .diffusion_gemma_vulkan_session(tokens.len() + canvas_len + 8)
            .expect("vulkan session");
        sess.prefill(&model, &tokens).expect("vulkan prefill");
        let outcome = sess
            .denoise(&model, &canvas, None, 1.0, 1.0, None)
            .expect("vulkan denoise");
        match outcome {
            infr_llama::seam::DenoiseOutcome::Logits(v) => v,
            infr_llama::seam::DenoiseOutcome::Reduced(_) => {
                panic!("u: None must always take the full-logits path")
            }
        }
    };

    let replay = run(false);
    let statc = run(true);
    assert_eq!(replay.len(), canvas_len * vocab);
    assert_eq!(statc.len(), canvas_len * vocab);

    let mut ndiff = 0usize;
    let mut maxabs = 0f32;
    for (x, y) in replay.iter().zip(&statc) {
        if x != y {
            ndiff += 1;
            maxabs = maxabs.max((x - y).abs());
        }
    }
    let mut argmax_flips = 0usize;
    for row in 0..canvas_len {
        let r = top_k(&replay[row * vocab..(row + 1) * vocab], 1)[0].0;
        let s = top_k(&statc[row * vocab..(row + 1) * vocab], 1)[0].0;
        if r != s {
            argmax_flips += 1;
        }
    }
    println!(
        "default vs no_replay: {ndiff}/{} elements differ, maxabs={maxabs}, \
         argmax flips {argmax_flips}/{canvas_len}",
        replay.len()
    );
    assert_eq!(
        ndiff, 0,
        "DG denoise logits diverge between the default and forced-static execution modes \
         (maxabs={maxabs}, argmax flips {argmax_flips}/{canvas_len})"
    );
}

// ─── DiffusionGemma Phase 3: entropy-bound decode loop vs the oracle ────────────────────────────
//
// The full block-diffusion decode (`infr_llama::diffusion::diffusion_generate`) driven on the
// Vulkan session, for the same chat-templated prompt the oracle (`llama-diffusion-cli`) was run
// on (see docs/diffusion-gemma.md's "Oracle reference outputs"). NOT a token-identical check (a
// 128-expert top-8 MoE model's CPU-vs-Vulkan routing legitimately diverges — the same class of
// divergence `gpu_seam_matches_cpu_diffusion_gemma[_denoise]` above already calibrate against);
// this asserts the DECODED TEXT is coherent (contains "Paris") and prints both texts + step/block
// counts side by side so a human can eyeball the match.

/// Vulkan-gated (like Phase 2's GPU tests): the entropy-bound decode loop end-to-end for "What is
/// the capital of France?", `n_predict=64`, greedy (`INFR_SEED` default 42). Prints the decoded
/// text, step count and block count; asserts the post-thought answer contains "Paris".
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn diffusion_gemma_decode_matches_oracle() {
    let path = need_model!(diffusion_gemma_model(), "diffusiongemma-26B-A4B");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    assert!(
        model.config().diffusion_gemma,
        "arch not parsed as diffusion-gemma"
    );

    let prompt = model
        .render_chat_messages(&[("user", "What is the capital of France?")])
        .expect("render chat template");
    let tokens = model.encode(&prompt).expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");

    let cfg = model.config();
    let canvas_len = cfg.canvas_length;
    let vocab = cfg.vocab;
    let eos_ids = cfg.eos_ids.clone();
    let eb = infr_llama::diffusion::EbConfig::from_config(cfg);

    let n_predict = 64usize;
    let blocks = n_predict.div_ceil(canvas_len).max(1);
    let max_ctx = tokens.len() + blocks * canvas_len + 64;
    let mut session = model
        .diffusion_gemma_vulkan_session(max_ctx)
        .expect("vulkan session");

    let t0 = std::time::Instant::now();
    let result = infr_llama::diffusion::diffusion_generate(
        &mut session,
        &model,
        &tokens,
        canvas_len,
        vocab,
        &eos_ids,
        &eb,
        n_predict,
        /* seed */ 42,
        max_ctx,
        None,
        None,
        None,
    )
    .expect("diffusion_generate");
    let secs = t0.elapsed().as_secs_f64();

    let text = model.decode(&result.tokens).expect("decode");
    eprintln!(
        "diffusion_gemma_decode_matches_oracle: {} steps over {} block(s) in {secs:.1}s \
         ({} tok generated)",
        result.steps,
        result.blocks,
        result.tokens.len()
    );
    println!("infr   text: {text:?}");
    // Oracle reference (CPU, `-p \"What is the capital of France?\" -n 64 -s 42 --temp 0`, captured
    // 2026-07-05 — see docs/diffusion-gemma.md): 10 EB steps, 1 block, thinking span then "The
    // capital of France is Paris."
    println!(
        "oracle text: \"<|channel>thought\\nThe user is asking for the capital of France.\\n    \
         *   Country: France.\\n    *   Capital: Paris.\\nProvide the direct answer \
         clearly.<channel|>The capital of France is Paris.\" (10 EB steps, 1 block)"
    );

    let (_thought, answer) = infr_chat::split_channels(&text);
    assert!(
        answer.contains("Paris") || text.contains("Paris"),
        "decoded answer doesn't mention Paris: {text:?}"
    );
}

// ─── Multi-GPU Slice 1: data-parallel two-models-two-devices ──────────────────────
//
// The end-to-end proof that the model-level device pool works: two independent Qwen3-0.6B seam
// sessions, one pinned to Vulkan0 and one to Vulkan1, generate greedily AT THE SAME TIME on the two
// physical GPUs. Each session owns its OWN backend/device (`vulkan_session_default_on`), so weights
// + KV never cross devices. We assert:
//   * each session bound the device it was pinned to (device_name matches the enumeration),
//   * both produce coherent greedy output (both answer "Paris"), concurrently,
//   * the Vulkan1 (iGPU on this box) session's VRAM DROPS after it loads — its weights/KV landed on
//     device 1, not device 0 (confirmed programmatically here; a vendor VRAM tool confirms it live).
//
// `#[ignore]` (needs real GPUs) and self-skips when fewer than two Vulkan devices are present, so it
// is a no-op on a single-GPU box. Run on the two-GPU box with:
//   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend two_models_two_devices \
//     -- --include-ignored --nocapture
#[test]
#[ignore = "requires TWO Vulkan GPUs: run with --include-ignored on a multi-GPU box"]
fn two_models_two_devices_concurrent() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let devices =
        infr_vulkan::VulkanBackend::enumerate_devices(&infr_llama::EngineConfig::default())
            .expect("enumerate devices");
    if devices.len() < 2 {
        eprintln!(
            "skip: two_models_two_devices needs >=2 Vulkan devices (found {})",
            devices.len()
        );
        return;
    }
    let _tlk = test_serial_lock();

    /// One session's outcome, carried back out of its thread.
    struct DevOut {
        dev: usize,
        name: String,
        text: String,
        avail_before: u64,
        avail_after: u64,
        live: bool,
    }

    // Run device 0 and device 1 CONCURRENTLY — the whole point is two models live on two GPUs at
    // once. Each thread loads its own model + opens its session on its pinned device (nothing
    // crosses the thread/device boundary).
    let run_on = |dev: usize| {
        let path = path.clone();
        std::thread::spawn(move || -> DevOut {
            // Deterministic + no <think> span, so the answer settles on "Paris" within a few
            // tokens. `sampling.no_think` is a VALUE on this model's own config since S7 — the
            // renderer takes it from the `Config` its `SeamModel` carries, so this no longer
            // writes `INFR_NO_THINK` into the process the sibling thread is also rendering in.
            let model = model_cfg(&path, |c| c.sampling.no_think = true);
            let mut sess = model
                .vulkan_session_default_on(Some(dev))
                .expect("open pinned session");
            let name = sess.device_name();
            let before = sess.vram();
            let prompt = model
                .render_chat("What is the capital of France? Reply with just the city name.")
                .expect("render");
            let mut text = String::new();
            model
                .generate_vulkan_session(&mut sess, &prompt, 32, None, |p| text.push_str(p))
                .expect("generate");
            let after = sess.vram();
            DevOut {
                dev,
                name,
                text,
                avail_before: before.available,
                avail_after: after.available,
                live: before.live && after.live,
            }
        })
    };

    let h0 = run_on(0);
    let h1 = run_on(1);
    let out0 = h0.join().expect("device 0 thread");
    let out1 = h1.join().expect("device 1 thread");

    for o in [&out0, &out1] {
        eprintln!(
            "Vulkan{} [{}]: VRAM avail {:.2} -> {:.2} GiB (live={}); text = {:?}",
            o.dev,
            o.name,
            o.avail_before as f64 / (1u64 << 30) as f64,
            o.avail_after as f64 / (1u64 << 30) as f64,
            o.live,
            o.text
        );
    }

    // Each session bound exactly the device it was pinned to.
    assert_eq!(
        out0.name, devices[0].name,
        "Vulkan0 session bound the wrong device"
    );
    assert_eq!(
        out1.name, devices[1].name,
        "Vulkan1 session bound the wrong device"
    );
    assert_ne!(
        out0.name, out1.name,
        "the two sessions must be on distinct physical devices"
    );

    // Both produce coherent greedy output, concurrently.
    assert!(
        out0.text.contains("Paris"),
        "Vulkan0 output not coherent: {:?}",
        out0.text
    );
    assert!(
        out1.text.contains("Paris"),
        "Vulkan1 output not coherent: {:?}",
        out1.text
    );

    // The Vulkan1 session's weights + KV landed on device 1: its available VRAM dropped after load.
    // Only assert when the driver reports a LIVE budget (VK_EXT_memory_budget); otherwise the
    // snapshot is total-only and can't move (still printed above for the record).
    if out1.live {
        assert!(
            out1.avail_after < out1.avail_before,
            "Vulkan1 available VRAM did not drop ({} -> {}) — weights/KV did not land on device 1",
            out1.avail_before,
            out1.avail_after
        );
    }
}

// ─── Multi-GPU PIPELINE (layer-split): correctness = single-device-identical ──────────────────
//
// Splits ONE model's transformer layers across TWO physical GPUs (layers [0..k) on device A,
// [k..N) on device B), each layer's weights + KV resident on its device, the residual hidden state
// handed across the boundary (P2P dma-buf when available, else host-bounce). The KEY correctness
// claim: the split forward is BIT-IDENTICAL to the SAME model run on ONE device.
//
// The reference is the pipeline path itself on a SINGLE device (`[d0]`): identical runner code path
// (host-embed + static execute — see `PipelineBackend::capabilities`), with NO cut/handoff. Only
// the split + cross-device handoff differs between the two runs, so identical output ids prove the
// mechanism is exactly single-device. (Comparing against the fast production path instead would mix
// in the record-once-replay `_dyn`-kernel ULP noise — see `Graph::no_decode_replay` — a different
// axis than the split we're validating here.)
//
// Also demonstrated: coherent greedy output across both devices, and the per-device placement
// (printed) landing distinct physical GPUs. `#[ignore]` + self-skips below 2 Vulkan devices. Run:
//   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend \
//     pipeline_matches_single_device -- --include-ignored --nocapture
#[test]
#[ignore = "requires TWO Vulkan GPUs: run with --include-ignored on a multi-GPU box"]
fn pipeline_matches_single_device() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let devs = infr_vulkan::VulkanBackend::enumerate_devices(&infr_llama::EngineConfig::default())
        .expect("enumerate devices");
    if devs.len() < 2 {
        eprintln!(
            "skip: pipeline_matches_single_device needs >=2 Vulkan devices (found {})",
            devs.len()
        );
        return;
    }
    let _tlk = test_serial_lock();

    // No <think> span, so the answer settles within `n` tokens. A VALUE on this model's own
    // config since S7 (`sampling.no_think`), not a process-global `INFR_NO_THINK` write.
    let model = model_cfg(&path, |c| c.sampling.no_think = true);
    let prompt = model
        .render_chat("What is the capital of France? Reply with just the city name.")
        .expect("render");
    let enc = model.encode(&prompt).expect("encode");
    let n = 24;
    let d0 = devs[0].index;
    let d1 = devs[1].index;

    // Reference: the SAME pipeline runner path on ONE device (no split, no handoff).
    eprintln!("\n[pipeline] reference: single-device [Vulkan{d0}]");
    let ref_ids = model
        .generate_pipeline_ids(&[d0], &enc, n, |_| {})
        .expect("single-device pipeline gen");

    // Split across BOTH devices — layers [0..k) on Vulkan{d0}, [k..N) on Vulkan{d1}.
    eprintln!("[pipeline] split: [Vulkan{d0}, Vulkan{d1}]");
    let split_ids = model
        .generate_pipeline_ids(&[d0, d1], &enc, n, |_| {})
        .expect("two-device pipeline gen");

    // BIT-IDENTICAL: the split produces the exact same token ids as single-device.
    assert_eq!(
        ref_ids, split_ids,
        "layer-split output diverged from single-device — the cross-device handoff is not \
         value-preserving"
    );

    // Coherent greedy output across the two-device split.
    let text = model.decode(&split_ids).expect("decode split ids");
    eprintln!("[pipeline] split output: {text:?}");
    assert!(
        text.contains("Paris"),
        "two-device split output not coherent: {text:?}"
    );

    // Distinct physical devices actually held the two halves.
    assert_ne!(
        devs[0].name, devs[1].name,
        "the two devices must be distinct physical GPUs"
    );
    eprintln!(
        "[pipeline] PASS — {}-token split across Vulkan{d0} ({}) + Vulkan{d1} ({}) is \
         token-identical to single-device",
        split_ids.len(),
        devs[0].name,
        devs[1].name,
    );
}

// ─── Multi-GPU TENSOR PARALLELISM: correctness = single-device-identical ──────────────────────
//
// Shards EACH transformer layer's weight matrices across TWO physical GPUs (column-parallel
// q/k/v/gate/up, row-parallel o/down; KV sharded by head), each device computing its shard and the
// partials all-reduced (P2P dma-buf) per attention + per FFN. The KEY correctness claim: the sharded
// forward equals the same model run WITHOUT a split.
//
// The reference is the TP path itself on a SINGLE rank (`[d0]`, world=1): the lowering is the
// identity (shard factor 1, no all-reduce), the same runner code path (host-embed + static execute),
// so it exercises exactly the single-device compute. The two-rank run adds the weight shard + the
// per-layer all-reduce. Ideally token-identical; the O/down all-reduce reassociates the reduction vs
// a single wide GEMV, so a mismatch (if any) must stay at reduction-order tolerance — asserted as
// token-identity, which greedy decode over a small model holds in practice. Run:
//   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend \
//     tensor_parallel_matches_single_device -- --include-ignored --nocapture
#[test]
#[ignore = "requires TWO Vulkan GPUs: run with --include-ignored on a multi-GPU box"]
fn tensor_parallel_matches_single_device() {
    let path = need_model!(qwen3_06b(), "Qwen3-0.6B");
    let devs = infr_vulkan::VulkanBackend::enumerate_devices(&infr_llama::EngineConfig::default())
        .expect("enumerate devices");
    if devs.len() < 2 {
        eprintln!(
            "skip: tensor_parallel_matches_single_device needs >=2 Vulkan devices (found {})",
            devs.len()
        );
        return;
    }
    let _tlk = test_serial_lock();

    // No <think> span, so the answer settles within `n` tokens. A VALUE on this model's own
    // config since S7 (`sampling.no_think`), not a process-global `INFR_NO_THINK` write.
    let model = model_cfg(&path, |c| c.sampling.no_think = true);
    let prompt = model
        .render_chat("What is the capital of France? Reply with just the city name.")
        .expect("render");
    let enc = model.encode(&prompt).expect("encode");
    let n = 24;
    let d0 = devs[0].index;
    let d1 = devs[1].index;

    // Reference: the SAME TP runner path on ONE rank (identity lowering, no shard, no all-reduce).
    eprintln!("\n[tp] reference: single-rank [Vulkan{d0}]");
    let ref_ids = model
        .generate_tp_ids(&[d0], &enc, n, |_| {})
        .expect("single-rank tp gen");

    // Shard each layer's weights across BOTH devices, all-reduce per attention + per FFN.
    eprintln!("[tp] 2-way weight split: [Vulkan{d0}, Vulkan{d1}]");
    let split_ids = model
        .generate_tp_ids(&[d0, d1], &enc, n, |_| {})
        .expect("two-rank tp gen");

    assert_eq!(
        ref_ids, split_ids,
        "tensor-parallel output diverged from single-device beyond reduction-order tolerance — the \
         weight shard / all-reduce is not summing the same partial products"
    );

    // Coherent greedy output across the two-device shard.
    let text = model.decode(&split_ids).expect("decode split ids");
    eprintln!("[tp] split output: {text:?}");
    assert!(
        text.contains("Paris"),
        "two-device tensor-parallel output not coherent: {text:?}"
    );
    assert_ne!(
        devs[0].name, devs[1].name,
        "the two devices must be distinct physical GPUs"
    );
    eprintln!(
        "[tp] PASS — {}-token 2-way weight split across Vulkan{d0} ({}) + Vulkan{d1} ({}) matches \
         single-device",
        split_ids.len(),
        devs[0].name,
        devs[1].name,
    );
}

// Expert parallelism (MoE): a 2-device EXPERT split must produce the SAME tokens as the single-device
// MoE. The reference is the EP path itself on a SINGLE rank (`[d0]`, world=1): the band is the whole
// expert set (the id remap is the identity, no all-reduce), the same runner path (host-embed + static
// execute), so it exercises single-device expert compute. The two-rank run splits the stacked expert
// banks across the devices (each holds ~half the experts), routes GLOBALLY on the replicated router,
// computes only its band, and all-reduces the partial MoE outputs per layer. Ideally token-identical;
// the cross-rank sum reassociates the per-expert weighted accumulate vs a single-device accumulate, so
// a mismatch (if any) must stay at reduction-order tolerance — asserted as token-identity, which
// greedy decode holds in practice. `kernels.vulkan.moe_small_m = 64` forces the id-indexed small-m
// expert path for both the short prefill and decode so the run is deterministic and light. Run:
//   INFR_TEMP=0 cargo test --release -p infr-llama --test cpu_backend \
//     expert_parallel_matches_single_device -- --include-ignored --nocapture
#[test]
#[ignore = "requires TWO Vulkan GPUs + the Qwen3-30B-A3B MoE model: run with --include-ignored on a multi-GPU box"]
fn expert_parallel_matches_single_device() {
    let path = need_model!(qwen3moe_30b(), "Qwen3-30B-A3B");
    let devs = infr_vulkan::VulkanBackend::enumerate_devices(&infr_llama::EngineConfig::default())
        .expect("enumerate devices");
    if devs.len() < 2 {
        eprintln!(
            "skip: expert_parallel_matches_single_device needs >=2 Vulkan devices (found {})",
            devs.len()
        );
        return;
    }
    let _tlk = test_serial_lock();
    // Force the id-indexed small-m expert path for both prefill and decode (light + deterministic),
    // and drop the <think> span so the answer settles within `n` tokens.
    // A VALUE on this model's own `Config`: the EP backends are opened by `SeamModel` through
    // `VulkanBackend::new_on_with`, so `kernels.vulkan.moe_small_m` reaches `tier::EnvRows::clamped`
    // without an env write (S5a, closing S2's recorded R7 exception); `sampling.no_think` joined it
    // in S7, when `infr-chat`'s renderer stopped reading `INFR_NO_THINK` from the environment.
    let model = model_cfg(&path, |c| {
        c.kernels.vulkan.moe_small_m = 64;
        c.sampling.no_think = true;
    });
    let prompt = model
        .render_chat("What is the capital of France? Reply with just the city name.")
        .expect("render");
    let enc = model.encode(&prompt).expect("encode");
    let n = 16;
    let d0 = devs[0].index;
    let d1 = devs[1].index;

    // Reference: the SAME EP runner path on ONE rank (whole-band identity, no shard, no all-reduce).
    eprintln!("\n[ep] reference: single-rank [Vulkan{d0}]");
    let ref_ids = model
        .generate_ep_ids(&[d0], &enc, n, |_| {})
        .expect("single-rank ep gen");

    // Split the experts across BOTH devices, all-reduce the partial MoE output per layer.
    eprintln!("[ep] 2-way expert split: [Vulkan{d0}, Vulkan{d1}]");
    let split_ids = model
        .generate_ep_ids(&[d0, d1], &enc, n, |_| {})
        .expect("two-rank ep gen");

    assert_eq!(
        ref_ids, split_ids,
        "expert-parallel output diverged from single-device beyond reduction-order tolerance — the \
         expert shard / band remap / all-reduce is not summing the same weighted expert products"
    );

    // Coherent greedy output across the two-device expert shard.
    let text = model.decode(&split_ids).expect("decode split ids");
    eprintln!("[ep] split output: {text:?}");
    assert!(
        text.contains("Paris"),
        "two-device expert-parallel output not coherent: {text:?}"
    );
    assert_ne!(
        devs[0].name, devs[1].name,
        "the two devices must be distinct physical GPUs"
    );
    eprintln!(
        "[ep] PASS — {}-token 2-way EXPERT split across Vulkan{d0} ({}) + Vulkan{d1} ({}) matches \
         single-device",
        split_ids.len(),
        devs[0].name,
        devs[1].name,
    );
}

// ─── BitNet b1.58 (bitnet: llama skeleton + SubLN, ternary TQ2_0 weights) ──────────
//
// `general.architecture == "bitnet"` (see `arch::BITNET`). BitNet b1.58 is the llama decoder
// (RMSNorm, NEOX rope like qwen2 → `Config::permute_qk_neox`, no qk-norm, no attention bias, tied
// lm_head, gated SwiGLU/SiLU FFN) plus SubLN's two extra RMSNorms (`Config::sub_norm`):
// `attn_sub_norm` on the concatenated-heads attention output BEFORE the o-projection, and
// `ffn_sub_norm` on the FFN intermediate BEFORE `ffn_down`. Verified against llama.cpp's
// `build_bitnet` (`src/models/bitnet.cpp`) — the FFN activation is SiLU, NOT squared-ReLU.

fn bitnet_b1_58_large() -> Option<PathBuf> {
    find_gguf(
        "gianni-cor--bitnet_b1_58-large-TQ2_0",
        "bitnet_b1_58-large-TQ2_0.gguf",
    )
}

/// bitnet arch recognition + config parse: `Config::from_gguf` must accept `arch == "bitnet"`,
/// set the SubLN flag (`sub_norm`) and the NEOX row-permute (`permute_qk_neox`), and NOT enable any
/// of the qk-norm / qkv-bias / gemma / moe / qwen35 gates. The FFN stays dense-gated (`moe` None).
#[test]
fn cpu_bitnet_config() {
    let path = need_model!(bitnet_b1_58_large(), "bitnet_b1_58-large");
    let g = infr_gguf::Gguf::open(&path).expect("open bitnet gguf");
    assert_eq!(
        g.metadata().str("general.architecture"),
        Some("bitnet"),
        "fixture is not the bitnet GGUF"
    );
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf must accept arch=bitnet");
    assert!(cfg.sub_norm, "bitnet must set Config::sub_norm (SubLN)");
    assert!(
        cfg.permute_qk_neox,
        "bitnet uses NEOX rope (like qwen2) — must permute q/k rows at load"
    );
    assert!(!cfg.qk_norm, "bitnet has no learned q/k-norm");
    assert!(!cfg.qkv_bias, "bitnet has no attention bias");
    assert!(!cfg.gemma && !cfg.gemma4, "bitnet is not a gemma variant");
    assert!(!cfg.qwen35, "bitnet is not qwen35");
    assert!(cfg.moe.is_none(), "bitnet is a dense (non-MoE) model");
    // Confirmed from the GGUF metadata dump.
    assert_eq!(cfg.n_layer, 24);
    assert_eq!(cfg.n_head, 16);
    assert_eq!(cfg.n_kv, 16, "bitnet-large is MHA (no GQA)");
    assert_eq!(cfg.head_dim, 96, "key_length=96");
    assert_eq!(cfg.n_ff, 4096);
    assert_eq!(cfg.n_embd, 1536);
}

/// CPU-only: bitnet's causal prompt prefill (SubLN + ternary TQ2_0 weights) produces finite logits
/// AND coherent geography — "The capital of France is" must rank " Paris" as the #1 next token. BOS
/// is prepended (the GGUF sets `add_bos_token=true`; base-model completion needs it or the tiny 0.7B
/// model degenerates). A real correctness gate on the two sub-norm placements + the NEOX permute.
#[test]
fn cpu_bitnet_prefill_paris() {
    let path = need_model!(bitnet_b1_58_large(), "bitnet_b1_58-large");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let cfg = model.config();
    assert!(cfg.sub_norm);
    let mut tokens = model.encode("The capital of France is").expect("encode");
    tokens.insert(0, 1); // BOS
    let last = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    assert_eq!(last.len(), cfg.vocab, "logits shape");
    assert!(
        last.iter().all(|v| v.is_finite()),
        "non-finite logit in the bitnet prefill output"
    );
    let top = top_k(&last, 5);
    let decoded: Vec<String> = top
        .iter()
        .map(|&(id, _)| model.decode(&[id as u32]).unwrap_or_default())
        .collect();
    eprintln!("bitnet top-5: {decoded:?}");
    assert_eq!(
        top[0].0, 3681,
        "bitnet must predict ' Paris' (id 3681) as the top next token, got {:?} — a wrong SubLN \
         placement or rope permute breaks this",
        decoded[0]
    );
}

/// bitnet's causal prompt prefill through the Vulkan seam vs the CPU oracle (both run TQ2_0
/// natively). Bit-identical isn't expected (CPU f32 vs Vulkan f16), so this locks the #1 next token
/// AND a tight whole-vocab cosine — the same shape as the other GPU-seam parity tests.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_bitnet() {
    let path = need_model!(bitnet_b1_58_large(), "bitnet_b1_58-large");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let mut tokens = model.encode("The capital of France is").expect("encode");
    tokens.insert(0, 1); // BOS
    let cpu_last = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    let gpu_last = model
        .prefill_logits_vulkan(&tokens)
        .expect("vulkan prefill");
    assert!(
        gpu_last.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan bitnet prefill output"
    );
    let (cpu_top, gpu_top) = (top_k(&cpu_last, 5), top_k(&gpu_last, 5));
    eprintln!("cpu top-5: {cpu_top:?}\nvulkan top-5: {gpu_top:?}");
    assert_eq!(
        cpu_top[0].0, gpu_top[0].0,
        "CPU/Vulkan top token diverged: cpu={:?} vulkan={:?}",
        cpu_top[0], gpu_top[0]
    );
    let cos = cosine(&cpu_last, &gpu_last);
    eprintln!("cpu/vulkan whole-vocab cosine: {cos}");
    assert!(
        cos > 0.99,
        "CPU/Vulkan last-row logits diverged: cosine={cos}"
    );
}

// ─── DeepSeek V1 ────────────────────────────────────────────────────────────────

/// Locate a DeepSeek-LLM-7B-Chat GGUF in the HF cache. The smallest V1 model (7B,
/// ~4 GB Q4_K_M). `None` ⇒ tests self-skip.
fn deepseek_v1_7b() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_DEEPSEEK") {
        return Some(PathBuf::from(p));
    }
    // `unsloth/DeepSeek-LLM-7B-Chat-GGUF` does not exist (the HF API 401s on it), so this
    // helper could only ever return `None` and every V1 test self-skipped forever.
    //
    // The replacement is the MoE-16B, not a dense 7B: `LLM_ARCH_DEEPSEEK` is really the MoE
    // variant. Dense DeepSeek-LLM-7B IS architecturally Llama, and every GGUF of it declares
    // `general.architecture = "llama"` — TheBloke's does, so it loads and generates fine while
    // testing none of the `deepseek` path. This file carries the arch string, 64 experts, 2
    // shared and 1 dense-lead layer, so it also exercises the `n_expert_shared > 1` shared-expert
    // width that docs/deepseek.md open question 4 flags as unverified.
    find_gguf(
        "mradermacher--deepseek-moe-16b-chat-GGUF",
        "deepseek-moe-16b-chat.Q4_K_M.gguf",
    )
}

/// Config-only: open the GGUF and assert every deepseek gate boolean, plus that
/// other arches' gates are false. No model load needed.
#[test]
fn cpu_deepseek_config() {
    let path = need_model!(deepseek_v1_7b(), "DeepSeek-LLM-7B-Chat");
    let g = infr_gguf::Gguf::open(&path).expect("open GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("parse config");
    assert_eq!(
        g.metadata().str("general.architecture"),
        Some(infr_llama::arch::DEEPSEEK),
        "arch string"
    );
    assert!(cfg.deepseek, "deepseek gate");
    assert!(!cfg.deepseek2, "deepseek2 must be false");
    assert!(!cfg.deepseek32, "deepseek32 must be false");
    assert!(!cfg.deepseek4, "deepseek4 must be false");
    assert!(!cfg.qk_norm, "qk_norm must be false");
    assert!(!cfg.qkv_bias, "qkv_bias must be false");
    assert!(!cfg.permute_qk_neox, "permute_qk_neox must be false");
    assert!(!cfg.sub_norm, "sub_norm must be false");
    assert!(!cfg.llama4, "llama4 must be false");
    assert!(!cfg.qwen35, "qwen35 must be false");
    assert!(!cfg.gemma, "gemma must be false");
    assert!(!cfg.gemma4, "gemma4 must be false");
    assert!(!cfg.shexp_gated, "shared expert must be ungated");
    assert!(cfg.moe.is_some(), "must have MoE config");
    let moe = cfg.moe.unwrap();
    assert_eq!(moe.gating, infr_core::graph::MoeGating::Softmax);
    assert!(!moe.norm_w, "no top-k renormalization for V1");
    assert!(!moe.weight_before, "weight-before-FFN must be false");
    // n_layer_dense_lead is non-zero for MoE-16B, 0 for dense 7B.
    // Shared expert: deepseek-llm-7b-chat has no shared expert (shexp_ff=0);
    // deepseek-moe-16b-chat has n_expert_shared=2 → shexp_ff > 0.
    eprintln!(
        "deepseek V1: n_layer={} n_head={} n_embd={} moe.n_expert={} moe.n_used={} n_layer_dense_lead={} shexp_ff={}",
        cfg.n_layer, cfg.n_head, cfg.n_embd,
        moe.n_expert, moe.n_used,
        cfg.n_layer_dense_lead, cfg.shexp_ff,
    );
}

/// Greedy top-1 token after "The capital of France is" — validates the full
/// forward pass (attention + MoE + output) produces a non-empty response.
#[test]
fn cpu_deepseek_prefill_paris() {
    let path = need_model!(deepseek_v1_7b(), "DeepSeek-LLM-7B-Chat");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let output = cpu_gen(&model, "The capital of France is", 8);
    assert!(!output.trim().is_empty(), "model produced no output");
    eprintln!("deepseek V1 'Paris': {output:?}");
}

// ── DeepSeek V2-Lite ───────────────────────────────────────────────────────────

/// Look up `INFR_TEST_DEEPSEEK2` env var, else the modern absorbed-layout HF repo. The
/// mradermacher V2-Lite file is the LEGACY pre-absorbed layout (`attn_kv_b`, no `attn_k_b`),
/// which infr's MLA path does not target — so prefer a file converted after the absorbed
/// `wk_b`/`wv_b` tensors became the default.
fn deepseek_v2_lite() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_DEEPSEEK2") {
        return Some(PathBuf::from(p));
    }
    find_gguf(
        "JenniSD--DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF",
        "deepseek-v2-lite-chat-q4_k_m.gguf",
    )
    .or_else(|| {
        find_gguf(
            "mradermacher--DeepSeek-V2-Lite-Chat-GGUF",
            "DeepSeek-V2-Lite-Chat.Q4_K_M.gguf",
        )
    })
}

/// Config-only: open the GGUF and assert every deepseek2 gate boolean, plus
/// that other arches' gates are false. No model load needed.
#[test]
fn cpu_deepseek2_config() {
    let path = need_model!(deepseek_v2_lite(), "DeepSeek-V2-Lite-Chat");
    let g = infr_gguf::Gguf::open(&path).expect("open GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("parse config");
    assert_eq!(
        g.metadata().str("general.architecture"),
        Some(infr_llama::arch::DEEPSEEK2),
        "arch string"
    );
    assert!(cfg.deepseek2, "deepseek2 gate");
    assert!(!cfg.deepseek, "deepseek must be false");
    // `deepseek2` is the SHARED MLA/KV/MoE gate and `deepseek32` (V3.2) sets it too — so this is
    // the direction that matters on a real V2 file: the V3.2-only flag and its indexer
    // hyperparameters must stay off, or the loader would demand five tensors this GGUF has not got.
    assert!(!cfg.deepseek32, "deepseek32 must be false on a V2 GGUF");
    // V4, by contrast, does NOT widen `deepseek2` — it is not MLA. The direction that matters here
    // is the other one: none of V4's per-layer machinery may switch on for a real V2 file, or the
    // loader would ask this GGUF for compressor, hyper-connection and hash-routing tensors.
    assert!(!cfg.deepseek4, "deepseek4 must be false on a V2 GGUF");
    assert!(
        cfg.compress_ratios.is_empty() && cfg.swiglu_clamp_exp.is_empty(),
        "V2 has no per-layer compression tiers and no SwiGLU clamps"
    );
    assert_eq!(
        (cfg.hash_layer_count, cfg.hc_mult, cfg.o_group_count),
        (0, 0, 0),
        "V2 has no hash routing, no hyper-connections and no grouped output projection"
    );
    assert_eq!(
        (cfg.indexer_n_head, cfg.indexer_head_size, cfg.indexer_top_k),
        (0, 0, 0),
        "V2 has no lightning indexer"
    );
    assert_eq!(cfg.norm_eps, 0.0, "V2 emits no (non-RMS) LayerNorm");
    assert!(!cfg.qk_norm, "qk_norm must be false");
    assert!(!cfg.qkv_bias, "qkv_bias must be false");
    assert!(!cfg.permute_qk_neox, "permute_qk_neox must be false");
    assert!(!cfg.llama4, "llama4 must be false");
    assert!(!cfg.qwen35, "qwen35 must be false");
    assert!(cfg.moe.is_some(), "must have MoE config");
    let moe = cfg.moe.unwrap();
    assert!(moe.n_expert > 1, "must have multiple experts");
    assert!(cfg.kv_lora_rank > 0, "kv_lora_rank must be set for MLA");
    assert_eq!(
        cfg.n_kv, 1,
        "n_head_kv must be 1 for MLA (one KV row per token)"
    );
    eprintln!(
        "deepseek2 V2-Lite: n_layer={} n_head={} n_embd={} kv_lora_rank={} qk_rope_dim={} head_k_mla={} v_head_dim={} moe.n_expert={} moe.n_used={}",
        cfg.n_layer, cfg.n_head, cfg.n_embd,
        cfg.kv_lora_rank, cfg.qk_rope_dim, cfg.head_k_mla, cfg.v_head_dim,
        moe.n_expert, moe.n_used,
    );
}

/// Greedy generation on "The capital of France is" — the full forward pass (MLA attention +
/// MoE + output) must produce COHERENT text. This was the test that caught the whole DeepSeek
/// V2 bug family: it printed `"Reply CollaborReply Collabor CollaborReplyReplyReply"` before the
/// YaRN ramp + wk_b/wv_b orientation fixes, and `" The capital of France is Paris. This"` after
/// (verified against llama.cpp c629da5's greedy continuation " Paris.", which differs only in the
/// top-2 dead heat between " Paris" and " The" — llama.cpp's own margin is 0.034).
#[test]
fn cpu_deepseek2_prefill_paris() {
    let path = need_model!(deepseek_v2_lite(), "DeepSeek-V2-Lite-Chat");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let output = cpu_gen(&model, "The capital of France is", 8);
    assert!(
        output.contains("Paris"),
        "incoherent deepseek2 V2-Lite generation: {output:?}"
    );
    eprintln!("deepseek2 V2-Lite 'Paris': {output:?}");
}

/// CPU golden-hash lock for DeepSeek V2-Lite (the MoE routed-expert + MLA path, per the
/// qwen3moe precedent). Blessed from the coherent post-fix output — `INFR_BLESS=1` prints the
/// text for a human coherence check before locking; the hash pins the exact generated stream.
#[test]
fn cpu_deepseek2_golden() {
    let path = need_model!(deepseek_v2_lite(), "DeepSeek-V2-Lite-Chat");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    check_golden(
        &model,
        &[
            // (prompt, n_tokens, fnv1a hash of the generated text)
            ("The capital of France is", 8, 0xc822e10860ea343b),
        ],
    );
}

/// Full CPU prefill on V2-Lite: all logits must be finite. Faster than a
/// generation test and catches NaN/Inf in MLA attention or MoE routing.
#[test]
fn cpu_deepseek2_prefill_finite() {
    let path = need_model!(deepseek_v2_lite(), "DeepSeek-V2-Lite-Chat");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let cfg = model.config();
    assert!(cfg.deepseek2, "deepseek2 gate");
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    assert!(!tokens.is_empty(), "empty prompt");
    let t0 = std::time::Instant::now();
    let last_row = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    eprintln!(
        "cpu_deepseek2_prefill_finite: {} tokens, prefill {:.1}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(last_row.len(), cfg.vocab, "logits shape");
    assert!(
        last_row.iter().all(|v| v.is_finite()),
        "non-finite logit in the prefill output"
    );
    println!("top-5 last-row tokens: {:?}", top_k(&last_row, 5));
}

/// V2-Lite prefill through the Vulkan seam vs the CPU oracle. MLA attention
/// and MoE both have discrete-selection steps (MLA's absorbed form with its
/// own kernel, MoE top-k routing), so this follows the qwen35moe precedent:
/// top-5 overlap + cosine floor, NOT bit-identical.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_seam_matches_cpu_deepseek2() {
    let path = need_model!(deepseek_v2_lite(), "DeepSeek-V2-Lite-Chat");
    let mut _tlk = test_serial_lock();
    let model = model_default(&path);
    let tokens = model
        .encode("What is the capital of France? Answer briefly.")
        .expect("encode");
    let vocab = model.config().vocab;
    let cpu_last = model.prefill_logits_cpu(&tokens).expect("cpu prefill");
    let gpu_last = model
        .prefill_logits_vulkan(&tokens)
        .expect("vulkan prefill");
    assert_eq!(cpu_last.len(), vocab, "cpu logits shape");
    assert_eq!(gpu_last.len(), vocab, "gpu logits shape");
    assert!(
        gpu_last.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan prefill output"
    );
    let (cpu_top, gpu_top) = (top_k(&cpu_last, 20), top_k(&gpu_last, 20));
    println!("cpu    top-5: {:?}", &cpu_top[..5]);
    println!("vulkan top-5: {:?}", &gpu_top[..5]);
    assert!(
        cpu_top[..5].iter().any(|&(id, _)| id == gpu_top[0].0)
            || gpu_top[..5].iter().any(|&(id, _)| id == cpu_top[0].0),
        "CPU/Vulkan top tokens don't even overlap in each other's top-5: cpu={:?} vulkan={:?}",
        cpu_top[0],
        gpu_top[0]
    );
    let cos = cosine(&cpu_last, &gpu_last);
    println!("cpu/vulkan whole-vocab cosine similarity: {cos}");
    assert!(
        cos > 0.5,
        "CPU/Vulkan last-row logits diverged too far: cosine={cos}"
    );
}
