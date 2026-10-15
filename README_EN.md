# MoE4All

**Run models far larger than VRAM on gaming GPUs. AMD, NVIDIA, and Intel are all working.**

The portable Windows and Linux packages are about 17 MiB and require no installation;
an experimental Metal package is also available for Apple Silicon.
Choose a recommended model or bring your own GGUF, select an automatic profile, and
start terminal chat or an OpenAI-compatible API. Weights move across VRAM, system
RAM, and SSD under one memory plan.

**1.0.0 performance reference:** RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB
DDR4, automatic aggressive profile, Q8 K/V, with MTP and CPU miss offload disabled.
All speeds are tok/s.

| Model | 2K total Decode mean x1 / x2 / x3 | 30K single-stream Prefill | 30K total Decode mean x1 / x2 / x3 | 150K single-stream Prefill | 150K total Decode mean x1 / x2 / x3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | — | **1,414.75** | **50.44 / 66.19 / 71.20** | **1,445.50** | **44.16 / 49.50 / 45.47** |
| Qwen3.6 35B Balanced | **130.26 / 168.09 / 206.63** | **2,749.06** | **80.49 / 118.34 / 110.78** | **1,169.40** | **41.91 / 50.87 / 49.27** |

<sub>Decode is aggregate mean throughput with 1 / 2 / 3 simultaneously active streams, not per-stream speed. Prefill is the full-interval single-stream mean. See the [1.0.0 measurements](#100-30k--150k-mean-performance-2026-10-14) for conditions.</sub>

**Same-host Strata v0.1.44.1 reference:** automatic configuration, without MTP;
`x2` is aggregate throughput for two streams.

| Model | 30K single-stream Prefill | 30K Decode x1 / x2 | 150K single-stream Prefill | 150K Decode x1 |
| --- | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 576.15 | 30.17 / 48.14 | 650.01 | 36.52 |
| Qwen3.6 35B Balanced | 1,325.32 | 19.37 / unsupported | 436.51 | 11.00 |

<sub>This is a directional same-host reference, not a strict same-weights, same-memory, same-harness engine A/B. See the [Strata comparison notes](#same-host-strata-v01441-comparison-2026-10-15) for compatibility conversions and methodological differences.</sub>

[Released builds](https://github.com/Headmaster218/MoE4All/releases/latest) |
[Quick start](#quick-start) |
[Measured results](#measured-results) |
[Community results](#community-results) |
[简体中文](README.md) |
[Technical documentation](documentation/README.md) |
[Low-weight expert omission experiment (Chinese)](documentation/evidence/campaigns/qwen38/2026-10-11-low-weight-expert-omission.md) |
[Expert cache and multi-GPU split (Chinese)](documentation/evidence/benchmarks/2026-10-11-flashnext-expert-cache-capacity.md)

## Quick start

### 1. Download the program

Open [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases) and
download `MoE4All-Windows-x86_64-v*.zip` on Windows, the matching `.tar.gz`
package on Linux x86_64, or the experimental `MoE4All-macOS-arm64-v*.tar.gz`
package on Apple Silicon. This page covers **1.0.0**; the results below describe
the test conditions and provenance of the performance tables.

### 2. Extract it

Fully extract the package into a directory such as `D:\MoE4All` on Windows.

### 3. Run

1. On Windows, double-click **`Start-INFR-Wizard.cmd`**. On Linux, run
   `bash Start-INFR-Wizard-Linux.sh`. On Apple Silicon, run
   `./Start-INFR-Wizard-macOS.sh`.
2. Choose terminal chat or the OpenAI-compatible API. The model list shows local
   models first and a `D` download entry below them. Open it to select one or more
   main models yourself; Windows and Linux also expose Vision and Embedding components. The wizard returns
   to local model selection after downloading and never chooses one model for you.
   On Windows, you can also drag an existing GGUF into the window and press Enter.
3. Select a device and automatic profile. **Aggressive performance** is used for
   the published results; **conservative** leaves more headroom for a first run or
   other active workloads.
4. Windows/Linux API mode can optionally enable Vision and Embedding. The experimental
   Metal package currently provides serialized, single-slot text serving only. Set context
   and API options and confirm launch.

All three release launchers show download progress, retry failures, resume from `.part`
files, and store content in the package-local `models` directory. The Linux wizard
does not require Python or jq. The macOS wizard uses the system `zsh`, `curl`, and
standard Apple utilities, with no Homebrew, Python, jq, or Rust runtime dependency.
Automatic profiles use Q8 K/V by default and let the engine plan available resources.
Terminal chat is text-only; Vision API serving is currently limited to Windows/Linux.

The experimental macOS binary is ad-hoc signed but not Apple-notarized. If Gatekeeper
blocks the first launch, run `xattr -dr com.apple.quarantine /path/to/MoE4All-macOS-arm64-v*`
on the extracted package, then start the wizard again.

See the [capability matrix](documentation/reference/model-capabilities.md) for combination limits. The default API base URL is `http://127.0.0.1:8080/v1`. See the [API guide](documentation/guide/serving/api-quickstart.md) and the
[configuration reference](documentation/reference/configuration.md) for all available settings.

## Recommended models and downloads

The Windows and Linux launch wizards expose one download menu for these models and
components; the experimental Metal launcher currently lists the two main models only.
You can also use the links below and
store them on a local SSD. Automatic downloads are placed in the `models` directory
beside the launcher.

| Model / component | Download | File and purpose |
| --- | --- | --- |
| **Qwen3.6 35B model** | [Download APEX-I-Balanced](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true) | `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`; this single file is sufficient for the 35B model |
| **Flash-Next main model** | [Download all AD-4.27bpw-Q4_K_M-M64 shards](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64) | Download all **33 GGUF shards** into one folder and select the first shard at launch |
| **Flash-Next vision** | [Download the F16 vision projector](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true) | `mmproj-Qwen3.8-Flash-Next-F16.gguf`; loaded for API image understanding |
| **Qwen3 Embedding 0.6B** | [Download the Q8_0 GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true) | `Qwen3-Embedding-0.6B-Q8_0.gguf`; serves `/v1/embeddings` |

The first Flash-Next shard is
`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`. The automatic
Vision download is placed in the same model directory for discovery by the wizard.

## Build from source (Linux)

Experimental Linux x86_64 packages ship alongside the Windows release; feedback is welcome. Source builds remain recommended
until the first CI validation. Once a `.tar.gz` asset is available, extract it and run
`Start-INFR-Wizard-Linux.sh` inside the package; Rust and glslc are not needed.
The source installation command below targets Ubuntu 26.04; other distributions need
`glslc` from shaderc 2025 or newer first:

```sh
sudo apt-get update && sudo apt-get install -y glslc
git clone https://github.com/Headmaster218/MoE4All.git && cd MoE4All
cargo build --release --locked -p infr-cli
./Start-INFR-Wizard-Linux.sh      # interactive launcher; --dry-run prints only
```

Requires Rust 1.97.1 (installed by `rustup` from `rust-toolchain.toml`) and
`glslc` from shaderc 2025 or newer. The full guide is
[Build and run on Linux](documentation/guide/linux-build-and-run.md); toolchain
and platform details are in
[Building from source](documentation/development/building-from-source.md).

## Measured results

### 1.0.0: 30K / 150K mean performance (2026-10-14)

Hardware: RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB DDR4, and Windows 11.
Configuration: automatic aggressive, Q8 K/V, `ctx=163840`, with MTP and CPU miss
offload disabled. Decode uses 1 / 2 / 3 simultaneously active slots and reports
aggregate full-interval mean throughput. Prefill reports the single-stream mean.

| Model | Input length | Single-stream Prefill | Decode x1 total mean | Decode x2 total mean | Decode x3 total mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 30K | **1,414.75** | **50.44** | **66.19** | **71.20** |
| Flash-Next 4.27bpw | 150K | **1,445.50** | **44.16** | **49.50** | **45.47** |
| Qwen3.6 35B Balanced | 2K | — | **130.26** | **168.09** | **206.63** |
| Qwen3.6 35B Balanced | 30K | **2,749.06** | **80.49** | **118.34** | **110.78** |
| Qwen3.6 35B Balanced | 150K | **1,169.40** | **41.91** | **50.87** | **49.27** |

### Same-host Strata v0.1.44.1 comparison (2026-10-15)

Strata's official v0.1.44.1 Windows HIP build was run on the same RX 7900 XTX,
Ryzen 5 5600X, 64 GiB Windows host. It used `ctx=163840`, INT8 K/V, automatic
expert-cache and Prefill settings, prompt cache off, and no MTP. Each case was
warmed before recording full Prefill and a sustained Decode interval.

| Model | Input length | Single-stream Prefill | Decode x1 mean | Decode x2 total mean | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Flash-Next 4.27bpw | 30K | 576.15 | 30.17 | 48.14 | x2 is aggregate throughput during the synchronized overlap |
| Flash-Next 4.27bpw | 150K | 650.01 | 36.52 | Not tested | Decode front / middle / late: 21.28 / 40.88 / 45.71 |
| Qwen3.6 35B Balanced | 30K | 1,325.32 | 19.37 | Unsupported by this build | Prefill is the median of three post-warmup runs; Decode range 18.51-21.61 |
| Qwen3.6 35B Balanced | 150K | 436.51 | 11.00 | Unsupported by this build | One complete Prefill followed by a 512-token Decode |

Compatibility changes were required for this Strata build. The Flash-Next copy
converted 363 fixed projection tensors to BF16. The 35B copy added compatibility
metadata and converted 60 small GDN alpha/beta state tensors from Q6_K to F32.
Routed-expert quantization was unchanged and the original model files were left
untouched, but the fixed weights are not bit-identical, so this is directional
rather than a strict engine A/B.

Strata automatic mode allocated about 47.17 GiB for its full host expert arena;
the single-stream Flash-Next run also used about 13 GiB of GPU expert cache, with
misses computed on CPU or transferred over PCIe. Windows commit approached 95 GiB,
and Strata did not apply MoE4All's 48 GiB total-process RAM budget. Its hybrid
CPU/GPU path may behave differently on a faster CPU. Prompts, fixed-weight format,
memory policy, and measurement harness were not identical, so the observed gap
cannot be attributed to one kernel or one design choice.

Concurrency gains are not linear: at 150K, both models have slightly lower
three-stream total throughput than two-stream throughput. Choose the slot count
according to aggregate-throughput and per-request-latency needs. The detailed
1.0.0-development and 0.10.0 measurements below remain as historical baselines.

### Current source: 35B true parallel Decode (2026-10-12)

Commit `25e6d1944` connects Qwen3.5 / Qwen3.6 MoE to layer-synchronous batched
Decode while keeping each session's KV, Gated DeltaNet state, and sampler
independent. Long Prefill remains serialized by slot; once all active prompts
are ready, their token rows run in one Decode graph.

Hardware was an RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB DDR4, and Windows 11,
using `Qwen3.6-35B-A3B-APEX-I-Balanced`. Configuration was automatic aggressive,
Q8 K/V, `ctx=163840`, three slots, `temperature=0`, with thinking, MTP, and CPU
miss offload disabled. One- and two-stream cases ran twice with 256 generated
tokens per stream; the three-stream case ran once with 512 tokens per stream.

| Input per stream | Active streams | Total Decode mean | Approx. per stream | Notes |
| --- | ---: | ---: | ---: | --- |
| 30K | 1 | **58.7–66.6** | 58.7–66.6 | Two-run range |
| 30K | 2 | **90.1–97.4** | 45.0–48.7 | Two-run range |
| 30K | 3 | **112.0** | 37.35 | Warm median 115.1; range 110.3–117.2 |
| 150K | 1 | **33.3–36.8** | 33.3–36.8 | Two-run range |
| 150K | 2 | **45.1–46.2** | 22.5–23.1 | Two-run range |
| 150K | 3 | **49.0** | 16.33 | Warm median 49.45; range 46.4–51.8 |

For the three-stream run, the three 512-token Decode completion times differed
by about 2.4 ms at 30K and 9.7 ms at 150K; every request completed successfully.
Warm statistics exclude the first two partial one-second intervals after Decode
started. Because long Prefill is serialized by slot, the per-request Prefill time
reported by the API includes time spent waiting for the other slots and is not
single-stream Prefill throughput.

### 0.10.0: 30K / 150K measurements

Measured on **2026-10-09** using the generic x86-64 0.10.0 release build,
source commit `53afa86f3`. Hardware: RX 7900 XTX 24 GiB, Ryzen 5 5600X,
64 GiB DDR4, Windows 11. Models: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`
and `Qwen3.6-35B-A3B-APEX-I-Balanced`.
The service used the automatic aggressive profile, Q8 K/V, `ctx=163840`, and
three configured slots with 1 / 2 / 3 simultaneously active. **MTP and CPU miss
offload were disabled.** Sampling and thinking used model defaults. Vision and
Embedding APIs were enabled; the table uses text-only requests, each with a
different long technical background and a normal question.

Each cell starts a fresh server: one cold Prefill round followed by two KV-reuse
rounds, generating 768 tokens per stream per round. All speeds are tok/s.
Prefill measures group token progress from the first Prefill sample to the last
prompt completion, excluding startup before that sample; it is not a sum of
stream rates. Decode averages three rounds over the interval when all streams are
decoding, excluding pauses for another request's Prefill. Approximate per-stream
Decode divides the total mean by the number of streams. Peak is the maximum
1.5-second rolling window. Prefill peaks and thirds interpolate chunk-completion
progress, not precise instantaneous GPU speeds.

| Model | Input per stream | Active streams | Cold Prefill mean / peak | Decode total mean / peak | Approx. Decode per stream |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 30K | 1 | 1,280.4 / 1,439.4 | **42.8** / 47.7 | 42.8 |
| Flash-Next 4.27bpw | 30K | 2 | 1,288.2 / 1,445.2 | **60.7** / 71.3 | 30.3 |
| Flash-Next 4.27bpw | 30K | 3 | 1,283.9 / 1,437.0 | **64.0** / 79.4 | 21.3 |
| Flash-Next 4.27bpw | 150K | 1 | 1,307.5 / 1,444.6 | **39.4** / 44.7 | 39.4 |
| Flash-Next 4.27bpw | 150K | 2 | 1,301.9 / 1,446.3 | **52.8** / 63.6 | 26.4 |
| Flash-Next 4.27bpw | 150K | 3 | 1,228.3 / 1,434.2 | **45.4** / 62.5 | 15.1 |
| 35B Balanced | 30K | 1 | 2,579.9 / 3,824.8 | **63.2** / 65.4 | 63.2 |
| 35B Balanced | 150K | 1 | 1,164.9 / 3,981.7 | **35.9** / 36.7 | 35.9 |

Front / middle / late split each measured interval into equal wall-time thirds;
Decode thirds also average three rounds.

| Model | Input per stream | Active streams | Prefill front / middle / late | Decode total front / middle / late |
| --- | --- | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 30K | 1 | 1,225.1 / 1,414.3 / 1,201.9 | 40.8 / 42.2 / 45.3 |
| Flash-Next 4.27bpw | 30K | 2 | 1,315.6 / 1,214.0 / 1,335.0 | 58.5 / 63.3 / 60.2 |
| Flash-Next 4.27bpw | 30K | 3 | 1,302.4 / 1,285.8 / 1,263.3 | 65.5 / 62.6 / 63.8 |
| Flash-Next 4.27bpw | 150K | 1 | 1,358.2 / 1,345.7 / 1,218.8 | 36.3 / 40.4 / 41.5 |
| Flash-Next 4.27bpw | 150K | 2 | 1,353.4 / 1,281.4 / 1,270.8 | 50.3 / 55.7 / 52.4 |
| Flash-Next 4.27bpw | 150K | 3 | 1,303.6 / 1,263.0 / 1,118.3 | 44.0 / 46.7 / 45.4 |
| 35B Balanced | 30K | 1 | 3,254.9 / 2,797.3 / 1,687.6 | 63.6 / 62.6 / 63.4 |
| 35B Balanced | 150K | 1 | 1,829.5 / 968.9 / 696.4 | 35.8 / 36.0 / 35.8 |

At 150K, Flash-Next's three-stream total is lower than its two-stream total;
concurrency gains are not linear. The full matrix completed 36 rounds and 72
stream responses without request failures; both warm rounds hit the expected
KV cache. Flash-Next ran with tight Windows commit headroom, which
can also affect reproducibility.

## Community results

**AMD, NVIDIA RTX, and Intel Arc all have user-tested configurations and
generation-speed reports.**

| GPU / CPU and memory | Model and conditions | Prefill | Decode | Version and source |
| --- | --- | ---: | ---: | --- |
| **AMD R9700 + Ryzen 9 9950X3D + 48GB RAM** | Flash-Next; quantization and context not reported | **800-1,100 tok/s** | **20-50 tok/s** | v0.10.0, user report |
| **Intel Arc A770 16GB + 64GB DDR4-3200** | Flash-Next, three runs; quantization and context not reported | Not reported | **17.44 / 17.30 / 17.45 tok/s** | v0.10.0, user report |
| **AMD RX 7700 XT 12GB + 64GB RAM** | Ornith 1.5 35B-A3B, `serve`, 131K context capacity | Not reported | **41.5–42.7 tok/s** | v0.5.2 community baseline, [PR #21](https://github.com/Headmaster218/MoE4All/pull/21) |
| **Intel Arc A770 16GB + 64GB DDR4-3200** | Ornith 1.5 35B Q4_K_M, F16 KV, 96K context capacity, three REAL-workload runs | Not reported | **30.15–30.28 tok/s** | v0.6.0-beta.1, [Issue #41](https://github.com/Headmaster218/MoE4All/issues/41) |
| **NVIDIA RTX 3090 Ti + 64GB RAM** | Successful community run; model, quantization, and context were not included in the original comment | Not reported | **about 29 tok/s** | Version not reported, [Bilibili user report](https://www.bilibili.com/video/BV1ALha63Eyd/) (rpid `318146261056`) |

Community speeds retain the users' original definitions and were not independently retested. Average versus peak was not specified, so they are not directly comparable to the fastest-third speed table above.

Share successful configurations in
[Discussions](https://github.com/Headmaster218/MoE4All/discussions), or report
problems through [Issues](https://github.com/Headmaster218/MoE4All/issues).
Include the GPU/VRAM, RAM, OS and driver, MoE4All version, model quantization,
context, automatic profile or launch command, and Prefill/Decode speeds.

## What it does

- **Runs models beyond VRAM:** coordinates MoE expert caching and loading across
  VRAM, system RAM, and SSD.
- **Vulkan inference:** executes directly through the Windows GPU driver; AMD is
  the primary validation platform, with community results from NVIDIA and Intel.
- **Interactive chat:** keeps context across turns and supports model-default,
  enabled, or disabled thinking modes.
- **OpenAI-compatible serving:** provides chat and Embedding APIs for existing
  clients.
- **Parallel serving:** independent K/V slots handle requests arriving at
  different times and using different context lengths; Qwen3.5 / Qwen3.6 MoE
  batches simultaneously active text requests in layer-synchronous Decode.
- **Persistent sessions:** an optional SSD cache stores idle text K/V and restores
  it after a server restart.
- **Long context:** supports quantized KV Cache, KV overflow, and long-context
  performance tests.
- **Qwen3.8 MTP:** optional single-stream and two-slot opportunistic speculative decoding;
  gains depend on draft acceptance. It remains available through manual CLI/configuration,
  but is not exposed by the Windows release launch wizard.
- **Experimental CPU miss offload:** disabled by default, requiring AVX2 + FMA3.
  On a Ryzen 5 5600X, four cores handling one miss delivered roughly the same
  end-to-end speed as the GPU path. A stronger CPU may help; benchmark locally.
- **Measurement and diagnostics:** built-in prefill/decode benchmarks, synthetic
  depth, and paging statistics.

## Current model support

| Model family | GGUF architecture | Status |
| --- | --- | --- |
| Llama and Llama 4 | `llama`, `llama4` | Dense and MoE Vulkan inference |
| Qwen2 / Qwen2.5 / Qwen3 | `qwen2`, `qwen3`, `qwen3moe` | Dense and Qwen3 MoE |
| Qwen3.5 / Qwen3.6 | `qwen35`, `qwen35moe` | Gated DeltaNet, attention, paged MoE, and layer-synchronous parallel generation |
| Qwen3.8 Flash Next | `qwen4exp` | Vulkan text and vision inference, parallel generation, hyper-connections, DeltaNet, PLE, QSA, and paged MoE |
| Gemma 3 / Gemma 4 | `gemma3`, `gemma4` | Dense, MoE, and E2B variants |
| Ling 3.0 Flash | `bailingmoe3` | KDA, gated MLA, 512 experts, and RAM/SSD paging |
| DeepSeek V4 Flash | `deepseek4` | FP8 KV, MXFP4 indexer cache, and paged MoE |
| DiffusionGemma | `diffusion-gemma` | Text-diffusion inference |
| Embedding GGUFs | Supported embedding architectures | Native CPU/Vulkan OpenAI Embedding API |

Fine-tunes using an existing architecture can often reuse the same
implementation. Compatibility depends on complete GGUF metadata, quantization
format, tokenizer, and chat template.

## How models can exceed VRAM

Large MoE models typically activate only a small fraction of their experts for
each token. MoE4All maintains three storage tiers:

```text
Complete GGUF on SSD
        ↓
Full host store or bounded RAM cache
        ↓
Elastic GPU expert cache
        ↓
Vulkan GPU execution
```

Frequently used experts remain in VRAM when possible, RAM provides a larger hot
tier, and SSD supplies the rest. Fixed model weights, KV Cache, runtime scratch,
and the expert cache share a coordinated VRAM budget. Elastic space can be
reassigned when execution switches between prefill and decode.

Implementation details are in the [documentation index](documentation/README.md);
historical optimization decisions are in the [change records](documentation/evidence/changes/README.md).

How much VRAM should the expert cache retain, and how should layers be split across GPUs with different PCIe bandwidths? See [FlashNext expert cache and multi-GPU split](documentation/evidence/benchmarks/2026-10-11-flashnext-expert-cache-capacity.md)
for a 0.5-48 GiB offline replay of real accesses, a continuous piecewise exponential approximation, a curve-based layer-allocation formula, and its limitations (Chinese documentation).

## Project and attribution

MoE4All is maintained by John / [Headmaster218](https://github.com/Headmaster218).
It is based on kryptic.sh's Pure-Rust, Vulkan-first inference engine
[infr](https://github.com/kryptic-sh/infr). The upstream project description is
in the [infr README](https://github.com/kryptic-sh/infr#readme).

The maintainer directs architecture, performance investigations, priorities,
and acceptance. AI coding agents assist extensively with Rust, Vulkan, testing,
and documentation work.

MoE4All modifications and the collective distribution use the
[Apache License 2.0](LICENSE). Code inherited from infr retains its original
[MIT License](LICENSE-MIT) and copyright notice; see [NOTICE](NOTICE) for
attribution. The two license files record the provenance of the MoE4All and
upstream portions respectively.
