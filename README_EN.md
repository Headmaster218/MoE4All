# MoE4All

**Run models far larger than VRAM on gaming GPUs. AMD, NVIDIA, and Intel are all working.**

The portable Windows package is about 14 MiB. Download a GGUF, choose automatic
configuration, and start local chat or an OpenAI-compatible API. MoE expert
weights are coordinated across VRAM, system RAM, and SSD.

**0.10.0 performance reference:** RX 7900 XTX 24 GiB, Ryzen 5 5600X, 64 GiB
DDR4, automatic aggressive profile. Qwen3.8-Flash-Next, AD-4.27bpw-Q4_K_M-M64,
three configured slots, MTP and CPU miss offload disabled. The table uses the
2026-10-09 test records in tok/s; see [Measured results](#measured-results) for
their provenance and measurement definitions.

| Active streams | 30K Prefill | 30K Decode total | 150K Prefill | 150K Decode total |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,093 | **42.8** | 752 | **39.5** |
| 2 | 960 | **59.7** | 958 | **48.1** |
| 3 | 1,033 | **70.2** | 1,049 | **40.4** |

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
a local SSD. The performance table above uses the Flash-Next main model without MTP.

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

If saved settings exist, you can launch directly with them. The wizard checks
for updates first, but updating is opt-in.

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

### 0.10.0: Flash-Next, 30K / 150K and 1 / 2 / 3 active streams

These measurements used the local **0.9.0 service before the 0.10.0 upgrade**;
they are not a new 0.10.0 benchmark. Hardware: RX 7900 XTX 24 GiB, Ryzen 5 5600X,
64 GiB DDR4, Windows 11. Model: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`.
The service used the automatic aggressive profile, Q8 K/V, `ctx=163840`, and
three configured slots with 1 / 2 / 3 simultaneously active. **MTP and CPU miss
offload were disabled.** Sampling was greedy with thinking off, and each lane
received a different long technical-background prompt.

All speeds are tok/s. Cold Prefill is the first-round group throughput with no
KV hits on any request: total input tokens divided by the group phase through
the last lane's Prefill completion. It includes scheduling and waiting; it is
not a sum of lane rates. Decode averages three subsequent KV-reuse rounds,
each generating 512 tokens per lane. Total Decode sums the final API
`predicted_per_second` rates, **not terminal instantaneous rates**. Large
Prefills run sequentially, followed by concurrent Decode.

| Input per lane | Active streams | Cold Prefill total | Decode total | Approx. Decode per lane |
| --- | ---: | ---: | ---: | ---: |
| 30K | 1 | 1,093 | **42.8** | 42.8 |
| 30K | 2 | 960 | **59.7** | 29.9 |
| 30K | 3 | 1,033 | **70.2** | 23.4 |
| 150K | 1 | 752 | **39.5** | 39.5 |
| 150K | 2 | 958 | **48.1** | 24.0 |
| 150K | 3 | 1,049 | **40.4** | 13.5 |

The service was not restarted between cells, so resident idle-slot KV and SSD
spill also affect these results. The first 150K single-stream Prefill had only
one expert Prefill lane; after other idle slots were released, separate full
Prefills reached **1,313-1,316 tok/s**. The 30K two-stream measurements include
one approximately one-second pause; 150K three-stream Decode ranged from
**37.6 to 43.1 tok/s** across its three rounds. This is a finite live-service
test, not an isolated kernel benchmark or a long-term stability guarantee.

<details>
<summary>Identical versus different prompts in concurrent Decode</summary>

Two separate control suites also used greedy sampling with thinking off and
averaged three KV-reuse rounds. Different prompts cover three technical topics;
identical prompts use the same complete input, seed, and generated output.
The following rates are still **total Decode** in tok/s. Identical-output peak
throughput should not be treated as general Agent-workload performance.

| Input per lane | Active streams | Different prompts | Identical complete prompts |
| --- | ---: | ---: | ---: |
| 30K | 2 | 61.5 | **69.9** |
| 30K | 3 | 69.5 | **84.9** |
| 150K | 2 | 48.6 | **59.8** |
| 150K | 3 | 44.5 | **64.4** |

These are independent prompt-control suites, not pooled with the all-cold
Prefill suite above. Identical prompts do not guarantee that concurrent first
requests require only one Prefill.

</details>

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
