# MoE4All

**Making huge MoE LLMs accessible to AMD users.**

[Latest Windows release](https://github.com/Headmaster218/MoE4All/releases/latest) |
[Getting started](GETTING_STARTED.md#english-quick-start) |
[简体中文](README.md) |
[Technical documentation](https://github.com/Headmaster218/MoE4All/blob/main/docs/README.md)

MoE4All is a local LLM inference project focused on AMD GPUs and native
Windows 11. It moves MoE expert weights between VRAM, system RAM, and SSD on
demand, allowing models much larger than GPU memory to run on consumer Radeon
hardware.

You do not need to understand paging, KV caches, or Vulkan before using it.
Download the portable package, prepare a supported GGUF model, launch the
bilingual wizard, and choose automatic configuration.

> Native Windows 11, AMD Radeon RX 7900 XTX, and Vulkan are the primary
> development and validation platform. Other Vulkan GPUs may work, but they do
> not currently receive the same compatibility and performance coverage.

## Latest milestone: full Qwen3.8-Flash-Next support

Qwen3.8-Flash-Next now generates stable, coherent output through MoE4All on a
consumer **AMD Radeon RX 7900 XTX**, including correct answers to simple math
and reasoning questions. Both Q2_K_XL and IQ4_XS quantizations have been tested
on Windows 11 with SSD paging under a **40 GiB bounded-RAM** budget.

The `qwen4exp` text path now covers the released model's four-stream
hyper-connections, gated DeltaNet/full-attention layer mix, SSD-backed PLE,
paged MoE, and **QSA sparse attention**. QSA maintains a separate F16 index-key
cache, selects complete history blocks, and preserves the incomplete causal
tail for long-context inference. The main K/V cache supports Q8_0, while the
QSA index-key cache remains F16 and is budgeted and allocated independently.

The table below was measured on an RX 7900 XTX with Vulkan0, Q8 K/V, and a
40 GiB bounded-RAM budget. Decode uses `tg128`, prefill uses `pp1024`, and the
ubatch is 1024. Depths 128K and 250K use synthetic depth to construct the real
KV length. Every entry is a three-run average and reports `kv_q8=true` and
`kv_layout=q8_0`.

| Context depth | Q2_K_XL decode | Q2_K_XL prefill | IQ4_XS decode | IQ4_XS prefill |
|---:|---:|---:|---:|---:|
| 0 | **29.45 tok/s** | **155.16 tok/s** | **16.85 tok/s** | **244.68 tok/s** |
| 128K | **26.23 tok/s** | **170.44 tok/s** | **14.15 tok/s** | **250.55 tok/s** |
| 250K | **22.82 tok/s** | **152.27 tok/s** | **15.26 tok/s** | **239.00 tok/s** |

Q2 and IQ4_XS both passed three-round API conversations while preserving a
verification code, completing cross-turn arithmetic, and summarizing prior
content. QSA uses radix top-k with exact score/index ordering, and batched
QSA/PLE prefill is enabled. Decode remains sensitive to expert RAM/SSD coverage
and has further optimization headroom.

## Start in three steps

### 1. Download

Open the [latest Release](https://github.com/Headmaster218/MoE4All/releases/latest),
download `MoE4All-Windows-x86_64-v*.zip`, and fully extract the archive.

The package includes `infr.exe` and the bilingual launch wizard. Running the
packaged build does not require Rust, Visual Studio, or the Vulkan SDK. It does
require a working 64-bit AMD GPU driver with Vulkan support.

### 2. Prepare a model

Models are not bundled. Download a GGUF using an architecture and quantization
supported by MoE4All. For a split GGUF, keep every shard in the same directory;
the loader can discover the complete model from any shard.

Start with a small model to verify the driver and generation path before
trying an MoE model that is tens or hundreds of gigabytes. See the
[getting-started guide](GETTING_STARTED.md#english-quick-start) for model and
shard guidance.

### 3. Launch

Double-click:

```text
Start-INFR-Wizard.cmd
```

Choose interactive terminal chat, the OpenAI-compatible API, or benchmark
mode, then paste or drag the GGUF path into the open model prompt. Automatic
configuration is recommended: MoE4All detects the GPU, available VRAM, system
memory, model structure, KV requirements, runtime space, and expert cache.

At startup, the wizard makes a short request to check the latest GitHub
Release. It only displays a download link when an update exists and never
modifies the installation automatically. Offline startup continues normally.

## What it does

- **Runs models beyond VRAM:** uses VRAM, RAM, and SSD on demand instead of
  requiring every MoE expert to stay on the GPU.
- **Native AMD Vulkan:** runs directly on Windows without CUDA or WSL.
- **Interactive chat:** keeps conversation state across turns and supports the
  model default, forced reasoning, and no-reasoning modes.
- **OpenAI-compatible serving:** exposes chat and embedding APIs for existing
  clients.
- **Long-context execution:** supports quantized KV caches, KV overflow, and
  long-context performance tests.
- **Built-in measurement:** includes prefill/decode benchmarks, synthetic
  context depth, pager statistics, and profiling controls.

## Requirements

| Component | Notes |
|---|---|
| Operating system | Primarily tested on 64-bit Windows 11 |
| GPU | AMD Vulkan is the current focus; more VRAM keeps more weights and KV resident |
| Driver | Install a current stable AMD driver; `infr.exe devices` should list the GPU |
| System RAM | Large MoE models use RAM as an expert tier; models beyond RAM can continue paging from SSD |
| Storage | A fast local SSD is recommended, with enough room for every model shard |
| Model | Bring a supported local GGUF, either a single file or a complete shard set |

The largest usable model depends on quantization, fixed weights, context size,
VRAM, system RAM, and storage performance. MoE4All aims to use the available
hardware effectively; it does not promise identical performance for every
model on every Radeon GPU.

## Measured results

The following measurements come from one Windows 11 machine with an RX 7900
XTX 24 GiB, Ryzen 5 5600X, and 64 GiB DDR4. They demonstrate current project
capabilities; rows use different workloads and are not directly comparable.

| Model and workload | Key conditions | Result |
|---|---|---:|
| Qwen3.6-35B-A3B decode after 250K synthetic depth | Q8 K/V, 1,000 generated tokens | **41.2 tok/s** |
| Qwen3.6-35B-A3B prefill 4,096 after 250K synthetic depth | Q8 K/V | **477.9 tok/s** |
| Qwen3.6-35B-A3B prefill 4,096 at depth 0 | Q8 K/V | **2,855.6 tok/s** |
| Qwen3.5-122B-A10B decode at depth 0 | F16 K/V, 45 GiB bounded RAM, 3 repetitions | **23.2 tok/s** |
| Qwen3.8-Flash-Next Q2_K_XL decode after 250K synthetic depth | Q8 K/V, 40 GiB bounded RAM, tg128, 3-run average | **22.82 tok/s** |
| Qwen3.8-Flash-Next Q2_K_XL prefill 1,024 after 250K | Q8 K/V, 40 GiB bounded RAM, 3-run average | **152.27 tok/s** |
| Qwen3.8-Flash-Next IQ4_XS decode after 250K synthetic depth | Q8 K/V, 40 GiB bounded RAM, tg128, 3-run average | **15.26 tok/s** |
| Qwen3.8-Flash-Next IQ4_XS prefill 1,024 after 250K | Q8 K/V, 40 GiB bounded RAM, 3-run average | **239.00 tok/s** |

Full conditions and engineering history:

- [Qwen3.6 RX 7900 XTX optimization history](https://github.com/Headmaster218/MoE4All/blob/main/docs/perf/qwen36-rx7900xtx-optimization-history-20260819.md)
- [Unified elastic VRAM acceptance](https://github.com/Headmaster218/MoE4All/blob/main/docs/unified-vram-elastic-acceptance-20260824.md)
- [DeepSeek V4 Flash closeout](https://github.com/Headmaster218/MoE4All/blob/main/docs/perf/deepseek-v4-flash-rx7900xtx-closeout-20260824.md)

## Current model support

| Model family | GGUF architecture | Status |
|---|---|---|
| Llama and Llama 4 | `llama`, `llama4` | Dense and MoE Vulkan inference |
| Qwen2 / Qwen2.5 / Qwen3 | `qwen2`, `qwen3`, `qwen3moe` | Dense and Qwen3 MoE |
| Qwen3.5 / Qwen3.6 | `qwen35`, `qwen35moe` | Gated DeltaNet, attention, and paged MoE |
| Qwen3.8 Flash Next | `qwen4exp` | Vulkan text inference with hyper-connections, gated DeltaNet, PLE, QSA, and paged MoE |
| Gemma 3 / Gemma 4 | `gemma3`, `gemma4` | Dense, MoE, and E2B variants |
| Ling 3.0 Flash | `bailingmoe3` | KDA, gated MLA, 512 experts, and RAM/SSD paging |
| DeepSeek V4 Flash | `deepseek4` | FP8 KV, MXFP4 indexer cache, and paged MoE |
| DiffusionGemma | `diffusion-gemma` | Text-diffusion inference |
| Embedding GGUFs | Supported embedding architectures | Native CPU/Vulkan OpenAI embedding API |

Fine-tunes using an existing architecture often work without a new runner,
but the GGUF metadata, quantization format, tokenizer, and chat template must
still be complete. Compatibility is never inferred from a model name alone.

## Everyday usage

### Terminal chat

Choose interactive chat in the wizard, or run:

```powershell
.\infr.exe run 'D:\Models\model.gguf'
```

### OpenAI-compatible API

```powershell
.\infr.exe serve --addr 127.0.0.1:8080 'D:\Models\model.gguf'
```

The API base URL is `http://127.0.0.1:8080/v1`. Configure an API key before
binding to a LAN address, and never expose an unauthenticated server directly
to the internet.

### Benchmark

```powershell
# Prefill
.\infr.exe bench -p 1024 -n 0 -r 1 'D:\Models\model.gguf'

# Decode
.\infr.exe bench -p 0 -n 128 -r 1 'D:\Models\model.gguf'
```

The [getting-started guide](GETTING_STARTED.md#9-benchmark-示例) explains the
common options and synthetic depth for measuring inference after a long
existing context.

## How models can exceed VRAM

An MoE model normally activates only a small fraction of its experts for each
token. MoE4All therefore maintains three storage tiers instead of requiring
every expert to remain on the GPU:

```text
Complete GGUF on SSD
        ↓
Full host store or bounded RAM cache
        ↓
Elastic GPU expert cache
        ↓
AMD Vulkan execution
```

Frequently used experts remain in VRAM when possible, RAM provides a larger
hot tier, and SSD supplies the rest. Fixed model weights, KV caches, runtime
scratch, and expert residency share a coordinated VRAM budget. Elastic space
can be reassigned when execution changes between prefill and decode.

For implementation details, see the
[documentation index](https://github.com/Headmaster218/MoE4All/blob/main/docs/README.md)
and the
[MoE4All wiki](https://github.com/Headmaster218/MoE4All/blob/main/infr-fork-wiki/README.md).

## Current limitations

- The first turns of a large model can be slower while the RAM cache is being
  populated from SSD.
- Large-model prefill still has room for a more complete asynchronous SSD-to-RAM
  lookahead pipeline.
- Host DMA is limited by the amount of system memory the Windows driver accepts
  for external import; other ranges fall back to CPU writes into ReBAR.
- Automatic budgeting prioritizes a reliable launch and is not guaranteed to
  be the highest-throughput configuration for every machine.
- The current portable Windows package focuses on the command-line wizard; the
  browser GUI remains a source-development entry point.

## Project and attribution

MoE4All is maintained by John / [Headmaster218](https://github.com/Headmaster218).
It is based on kryptic.sh's Pure-Rust, Vulkan-first inference engine
[infr](https://github.com/kryptic-sh/infr). Read the original upstream project
description in the [infr README](https://github.com/kryptic-sh/infr#readme);
it is not embedded in this repository's README.

The maintainer directs architecture, performance investigations, priorities,
and acceptance. AI coding agents are used extensively to assist with Rust,
Vulkan, testing, and documentation work.

MoE4All modifications and the collective distribution use the
[Apache License 2.0](LICENSE). Code inherited from infr retains its original
[MIT License](LICENSE-MIT) and copyright notice. See [NOTICE](NOTICE) for
attribution. The two license files describe different parts of the project's
provenance; they are not alternative buttons the user must choose between.
