# MoE4All

**Run models far larger than VRAM on gaming GPUs. AMD, NVIDIA, and Intel are all working.**

The portable Windows package is 13 MiB. Download a GGUF, choose automatic
configuration, and start local chat or an OpenAI-compatible API. MoE expert
weights are coordinated across VRAM, system RAM, and SSD.

**0.8.0 release benchmark:** RX 7900 XTX 24 GiB + 64 GiB DDR4, using the
automatic aggressive-performance profile. The table shows measured generation
speed in tok/s; see [Measured results](#measured-results) for full conditions and
per-segment results.

| Recommended model and quantization | MTP | 20K input | 150K input |
| --- | --- | ---: | ---: |
| **Qwen3.6-35B-A3B · APEX-I-Balanced** | Off | **65.6** | **39.9** |
| **Qwen3.8-Flash-Next · AD-4.27bpw-Q4_K_M-M64** | On | **60.2** | **48.1** |

[Released Windows builds](https://github.com/Headmaster218/MoE4All/releases/latest) |
[Quick start](#quick-start) |
[Measured results](#measured-results) |
[Community results](#community-results) |
[简体中文](README.md) |
[Technical documentation](docs/README.md)

## Quick start

### 1. Download the program

Open [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases) and
download the matching `MoE4All-Windows-x86_64-v*.zip`. The features and
measurements on this page correspond to the upcoming **0.8.0** release.

### 2. Extract it

Fully extract the ZIP into a directory such as `D:\MoE4All`.

### 3. Download a GGUF model

The current release is optimized and measured with the following two
quantizations. Store the model files on a local SSD.

| Model / component | Download | File and purpose |
| --- | --- | --- |
| **Qwen3.6 35B model** | [Download APEX-I-Balanced](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true) | `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`; this single file is sufficient for the 35B model |
| **Flash-Next main model** | [Download all AD-4.27bpw-Q4_K_M-M64 shards](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64) | Download all **33 GGUF shards** from this directory into one folder |
| **Flash-Next vision** | [Download the F16 vision projector](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true) | `mmproj-Qwen3.8-Flash-Next-F16.gguf`; load it for image understanding |
| **Flash-Next MTP** | [Download the shared Q4_K_M MTP head](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true) | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`; the text-acceleration component used in these measurements |

For Flash-Next, select the first shard when launching:
`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
The vision and MTP files can be placed beside the main-model shards and selected
for their respective modes in the wizard.

### 4. Run

1. Double-click **`Start-INFR-Wizard.cmd`** in the extracted directory.
2. Choose terminal chat or the OpenAI-compatible API, drag the main-model GGUF
   into the prompt, and press Enter.
3. Choose an automatic profile. **Aggressive performance** is the profile used
   for the results on this page; **conservative** provides more headroom for a
   first run or a system with other active workloads.
4. Leave context blank for automatic sizing, then confirm launch. To reproduce
   these measurements, use `32768` for 20K input and `163840` for 150K input.

Automatic profiles use Q8 K/V by default and plan VRAM, system RAM, expert
cache, and Ubatch. The 35B model is ready for chat with its main GGUF alone.
Flash-Next offers two optional paths:

- **MTP text acceleration:** enable “Qwen3.8 MTP single-stream acceleration,”
  select the MTP head above, and use verification width `4`. This path uses
  greedy decoding (`temperature=0`) and one session.
- **Image understanding:** choose API mode, leave MTP disabled, enable vision,
  and select the F16 vision projector above.

The default API base URL is `http://127.0.0.1:8080/v1`. See the
[configuration reference](docs/config.md) for all available settings.

## Build from source (Linux)

The release archives are Windows-only; on Linux, build from source. The result is
`target/release/infr`.

**Prerequisites**

- **Rust** — `rust-toolchain.toml` pins **1.97.1** (with `rustfmt` and `clippy`);
  `rustup` installs it automatically on the first build.
- **`glslc` (shaderc)** — the compute shaders are compiled at build time. The
  dp4a shaders need `GL_EXT_integer_dot_product`, so the compiler must be
  **shaderc 2025 or newer**; Ubuntu 24.04 ships shaderc 2023.8, which is too old
  (Ubuntu 26.04 ships a usable one).
- **A Vulkan driver is needed to run, not to build** — `ash` loads `libvulkan` at
  runtime via `dlopen`. On AMD that means RADV (Mesa).

```sh
sudo apt-get update && sudo apt-get install -y glslc
git clone https://github.com/Headmaster218/MoE4All.git
cd MoE4All
cargo build --release --locked -p infr-cli
```

**Run**

```sh
./target/release/infr devices                    # list visible Vulkan devices + VRAM
./target/release/infr run   <shard-00001.gguf>   # terminal chat (auto-pulls)
./target/release/infr serve <shard-00001.gguf>   # OpenAI-compatible API
```

**Notes**

- `.cargo/config.toml` sets `-C target-cpu=native` (the CPU backend leans on it
  to autovectorize), so the binary is ISA-specific to the build machine; override
  it when distributing across machines.
- The GPU integration tests are `#[ignore]`d (they need a real Vulkan device):
  `cargo test --workspace --locked -- --include-ignored`
- Large MoE models map a host tier into the GPU aperture; on AMD you may need a
  larger GTT — set `options amdgpu gttsize=<MiB>` under `/etc/modprobe.d/`
  (applied when the amdgpu module loads; needs a reboot).

## Measured results

### 0.8.0: Qwen3.8-Flash-Next, 20K versus 150K input

Both input sizes use the **automatic aggressive-performance profile**. Context,
Q8 K/V, and sampling are specified; the engine plans the remaining resources
and automatically selects `ubatch=4096`.

- **Hardware and OS:** RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB DDR4, Windows 11.
- **Model:** `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`; MTP uses
  `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- **Shared settings:** Q8 K/V, greedy, thinking disabled, single-stream
  generation, automatic aggressive profile.
- **Context capacity:** 32K for the 20K input and `ctx=163840` for the 150K input.

Prefill is input-processing speed and Decode is end-to-end generation speed.
Both are measured in tok/s.

| Workload | Mode | 20K Prefill | 20K Decode | 150K Prefill | 150K Decode |
| --- | --- | ---: | ---: | ---: | ---: |
| Complex prompt | MTP | 937 | **39.9** | 766 | **33.4** |
| Complex prompt | No MTP | 1,034 | **35.0** | 889 | **30.8** |
| Simple prompt (high acceptance) | MTP | 936 | **60.2** | 769 | **48.1** |
| Simple prompt (high acceptance) | No MTP | 1,035 | **37.7** | 888 | **32.4** |

- **Long-input generation remains fast:** from 20K to 150K, end-to-end Decode
  retains approximately **80%–88%** across the four workloads.
- **MTP gains vary with content:** complex prompts improve by approximately
  **14.0% / 8.4%** at 20K / 150K; simple prompts improve by approximately
  **59.7% / 48.5%**.
- **Input and output are timed separately:** MTP adds Prefill work in this test.
  Total request time combines input processing and output generation.

<details>
<summary>Full per-segment results, MTP acceptance, and long-context changes</summary>

Speeds are in tok/s. Middle covers 25%–62.5% of generated tokens and Late covers
62.5%–100%. Alpha is the MTP draft acceptance rate.

| Input size | Mode and workload | Prefill | Full Decode | Middle | Late | Alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20K | MTP simple prompt | 936 | 60.2 | 61.4 | 61.3 | 0.994 |
| 20K | MTP complex prompt | 937 | 39.9 | 42.0 | 39.7 | 0.594 |
| 20K | No-MTP simple prompt | 1,035 | 37.7 | 38.6 | 37.8 | — |
| 20K | No-MTP complex prompt | 1,034 | 35.0 | 35.1 | 36.8 | — |
| 150K | MTP simple prompt | 769 | 48.1 | 49.2 | 46.8 | 0.979 |
| 150K | MTP complex prompt | 766 | 33.4 | 35.5 | 34.3 | 0.642 |
| 150K | No-MTP simple prompt | 888 | 32.4 | 32.0 | 32.9 | — |
| 150K | No-MTP complex prompt | 889 | 30.8 | 31.4 | 31.2 | — |

Actual input lengths for the 20K simple / complex prompts are **19,988 / 20,019**
tokens; the 150K inputs contain **149,849 / 149,857** tokens.

| Mode and workload | Prefill change, 150K vs 20K | Full Decode change |
| --- | ---: | ---: |
| MTP simple prompt | -17.8% | -20.1% |
| MTP complex prompt | -18.2% | -16.3% |
| No-MTP simple prompt | -14.2% | -14.1% |
| No-MTP complex prompt | -14.0% | -12.0% |

[20K test setup, results, and analysis](docs/perf/qwen38-mtp-20k-comparison-20260924.md).

</details>

### 0.8.0: Qwen3.6 35B without MTP

**The complex prompt generates at 59.1 tok/s with 20K input and 39.0 tok/s
with 150K input.**

- **Model:** `Qwen3.6-35B-A3B-APEX-I-Balanced`, without MTP.
- **Hardware and OS:** the same RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB DDR4,
  and Windows 11 system.
- **Test settings:** automatic aggressive-performance profile, Q8 K/V, greedy,
  thinking disabled, and single-stream generation. Context capacity is 32K for
  the 20K input and `ctx=163840` for the 150K input.

Prompt is the actual input token count. Prefill and Decode are measured in
tok/s. Middle and Late cover 25%–62.5% and 62.5%–100% of generated tokens.

| Input / workload | Prompt | Prefill | Full Decode | Middle | Late |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20K simple prompt | 20,041 | 2,484 | **65.6** | 66.7 | 66.8 |
| 20K complex prompt | 20,049 | 2,421 | **59.1** | 59.5 | 62.6 |
| 150K simple prompt | 149,849 | 1,135 | **39.9** | 40.3 | 39.9 |
| 150K complex prompt | 149,857 | 1,140 | **39.0** | 39.6 | 39.8 |

## Community results

**AMD, NVIDIA RTX, and Intel Arc all have user-tested configurations and
generation-speed reports.**

| GPU and memory | Model and conditions | Generation speed | Version and source |
| --- | --- | ---: | --- |
| **AMD RX 7700 XT 12GB + 64GB RAM** | Ornith 1.5 35B-A3B, `serve`, 131K context capacity | **41.5–42.7 tok/s** | v0.5.2 community baseline, [PR #21](https://github.com/Headmaster218/MoE4All/pull/21) |
| **Intel Arc A770 16GB + 64GB DDR4-3200** | Ornith 1.5 35B Q4_K_M, F16 KV, 96K context capacity, three REAL-workload runs | **30.15–30.28 tok/s** | v0.6.0-beta.1, [Issue #41](https://github.com/Headmaster218/MoE4All/issues/41) |
| **NVIDIA RTX 3090 Ti + 64GB RAM** | Successful community run; model, quantization, and context were not included in the original comment | **about 29 tok/s** | Version not reported, [Bilibili user report](https://www.bilibili.com/video/BV1ALha63Eyd/) (rpid `318146261056`) |

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
  different times and using different context lengths.
- **Persistent sessions:** an optional SSD cache stores idle text K/V and restores
  it after a server restart.
- **Long context:** supports quantized KV Cache, KV overflow, and long-context
  performance tests.
- **Qwen3.8 MTP (0.8.0 preview):** optional single-stream speculative decoding;
  gains depend on draft acceptance. Setup is covered in [Quick start](#quick-start).
- **Measurement and diagnostics:** built-in prefill/decode benchmarks, synthetic
  depth, and paging statistics.

## Current model support

| Model family | GGUF architecture | Status |
| --- | --- | --- |
| Llama and Llama 4 | `llama`, `llama4` | Dense and MoE Vulkan inference |
| Qwen2 / Qwen2.5 / Qwen3 | `qwen2`, `qwen3`, `qwen3moe` | Dense and Qwen3 MoE |
| Qwen3.5 / Qwen3.6 | `qwen35`, `qwen35moe` | Gated DeltaNet, attention, and paged MoE |
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

Implementation details are available in the
[technical documentation index](https://github.com/Headmaster218/MoE4All/blob/main/docs/README.md)
and the
[MoE4All Wiki](https://github.com/Headmaster218/MoE4All/blob/main/infr-fork-wiki/README.md).

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
