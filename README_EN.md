# MoE4All

**Run models far larger than VRAM on gaming GPUs. AMD, NVIDIA, and Intel are all working.**

The portable Windows download is about 14 MiB. Download a GGUF, choose automatic
configuration, and start local chat or an OpenAI-compatible API. MoE expert
weights are coordinated across VRAM, system RAM, and SSD.

**0.10.0 peak-speed reference:** RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB
DDR4, automatic aggressive profile. Flash-Next 4.27bpw peaks at about
**1,446 tok/s Prefill** and **79.4 tok/s total Decode** with three active streams;
35B Balanced peaks at **65.4 tok/s Decode** with one active stream at 30K.
Three configured slots, MTP and CPU miss offload disabled.
The table shows 2026-10-09 measured peaks in tok/s; see
[Measured results](#measured-results) for measurement definitions.

| Model | Active streams | 30K Prefill peak | 30K Decode total peak | 150K Prefill peak | 150K Decode total peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 1 | 1,439.4 | **47.7** | 1,444.6 | **44.7** |
| Flash-Next 4.27bpw | 2 | 1,445.2 | **71.3** | 1,446.3 | **63.6** |
| Flash-Next 4.27bpw | 3 | 1,437.0 | **79.4** | 1,434.2 | **62.5** |
| 35B Balanced | 1 | 3,824.8 | **65.4** | 3,981.7 | **36.7** |

<sub>Peak means the highest speed within a 1.5-second rolling window. Prefill peaks interpolate completed-chunk progress.</sub>

[Released Windows builds](https://github.com/Headmaster218/MoE4All/releases/latest) |
[Quick start](#quick-start) |
[Measured results](#measured-results) |
[Community results](#community-results) |
[简体中文](README.md) |
[Technical documentation](documentation/README.md)

## Quick start

### 1. Download the program

Open [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases) and
download the matching `MoE4All-Windows-x86_64-v*.zip`. This page covers
**0.10.0**; the results below describe the test conditions and provenance of
the performance tables.

### 2. Extract it

Fully extract the ZIP into a directory such as `D:\MoE4All`.

### 3. Download a GGUF model

The following models and components are recommended. Store the model files on
a local SSD.

| Model / component | Download | File and purpose |
| --- | --- | --- |
| **Qwen3.6 35B model** | [Download APEX-I-Balanced](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true) | `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`; this single file is sufficient for the 35B model |
| **Flash-Next main model** | [Download all AD-4.27bpw-Q4_K_M-M64 shards](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64) | Download all **33 GGUF shards** from this directory into one folder |
| **Flash-Next vision** | [Download the F16 vision projector](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true) | `mmproj-Qwen3.8-Flash-Next-F16.gguf`; load it for image understanding |
| **Flash-Next MTP** | [Download the shared Q4_K_M MTP head](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true) | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`; an optional text-acceleration component |

For Flash-Next, select the first shard when launching:
`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
The vision and MTP files can be placed beside the main-model shards and selected
for their respective modes in the wizard.

### 4. Run

1. Double-click **`Start-INFR-Wizard.cmd`** in the extracted directory.
2. Choose terminal chat or the OpenAI-compatible API, drag the main-model GGUF
   into the prompt, and press Enter.
3. Select optional MTP, vision, and Embedding components, then the device and
   profile. **Aggressive performance** is used for these results; **conservative**
   provides more headroom for a first run or other active workloads.
4. Set concurrency, context, and API options, then confirm launch. To reproduce
   these measurements, configure three slots and `ctx=163840`, with MTP and
   experimental CPU offload disabled.

Automatic profiles use Q8 K/V by default and plan VRAM, system RAM, expert
cache, and Ubatch. The 35B model is ready for chat with its main GGUF alone.
Flash-Next has these optional components:

- **MTP text acceleration:** enable "Qwen3.8 MTP acceleration," select the head
  above, and use greedy decoding (`temperature=0`). Two-slot opportunistic MTP
  uses MTP with one active stream and ordinary batched Decode when both are active.
- **Image understanding:** choose API mode, enable vision, and select the F16
  projector above. Vision and MTP can be used together.

See the [capability matrix](documentation/reference/model-capabilities.md) for combination limits. The default API base URL is `http://127.0.0.1:8080/v1`. See the [API guide](documentation/guide/serving/api-quickstart.md) and the
[configuration reference](documentation/reference/configuration.md) for all available settings.

## Measured results

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

Community speeds retain the users' original definitions and were not independently retested. Average versus peak was not specified, so they are not directly comparable to the fixed-window peak table above.

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
- **Qwen3.8 MTP:** optional single-stream and two-slot opportunistic speculative decoding;
  gains depend on draft acceptance. Setup is covered in [Quick start](#quick-start).
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

Implementation details are in the [documentation index](documentation/README.md);
historical optimization decisions are in the [change records](documentation/evidence/changes/README.md).

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
