# MoE4All on Linux

MoE4All's inference engine is the Vulkan backend (`infr-vulkan`), which is
cross-platform. On Linux it targets any Vulkan-capable GPU — AMD (RADV/Mesa),
NVIDIA (proprietary driver), or Intel (ANV). The code already carries the
correct `#[cfg(target_os = "linux")]` / `#[cfg(unix)]` branches for memory
probes, file I/O, DMA-buf P2P, and timeline-semaphore sharing, so **no source
changes are needed for the Rust side**.

The single thing that is *not* portable is the **build-time shader compiler**
`glslc`. MoE4All compiles ~100 GLSL compute shaders to SPIR-V during
`cargo build`; the GEMM/attention shaders use the `GL_KHR_cooperative_matrix`
extension, so `glslc` must be new enough to support it.

## The one gotcha: `glslc` version

Many distros bundle an old `shaderc`/`glslc` (Ubuntu 24.04 ships `glslc 2023.8`,
and the `shaderc` GitHub `main` is frozen at 2022). Those fail with:

```
shaders/gemm_coopmat_tiled.comp:7: error: '#extension' : extension not supported: GL_KHR_cooperative_matrix
shaders/gemm_coopmat_tiled.comp:34: error: 'coopmat' : undeclared identifier
```

The reference toolchain the Windows build uses is the **LunarG Vulkan SDK**'s
`glslc` (currently 1.4.357.0), which compiles all MoE4All shaders. The quickest
fix is the bundled installer:

```bash
cd MoE4All
./scripts/install-linux.sh        # installs a known-good glslc under ~/.local (no sudo)
```

The installer:
1. Probes whether a `glslc` on `PATH` already compiles cooperative-matrix shaders.
   If yes, it does nothing.
2. Otherwise it downloads the LunarG Vulkan SDK tarball, extracts **only** the
   `glslc` binary + its runtime libraries (not the whole SDK), and writes a
   self-contained wrapper to `~/.local/bin/glslc`.
3. Appends `~/.local/bin` to `~/.bashrc` if it isn't already on `PATH`.

After that, open a new shell (or `export PATH="$HOME/.local/bin:$PATH"`) and build:

```bash
cargo build --release --locked -p infr-cli
./target/release/infr devices
```

### Installing `glslc` manually

If you prefer not to use the script, any of these works — just make sure the
resulting `glslc` passes the cooperative-matrix probe:

- **Ubuntu/Debian** (newer releases only; verify the version):
  `sudo apt install glslc` then check `glslc` compiles a `GL_KHR_cooperative_matrix`
  shader. If it doesn't, fall back to the Vulkan SDK below.
- **LunarG Vulkan SDK** (always works): download
  `https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkan-sdk-1.4.357.0.0.tar.xz`,
  extract, and use `<sdk>/x86_64/bin/glslc`.
- **Build shaderc from a recent tag** (not the 2022 `main`):
  `glslc` must be from a shaderc release whose `glslang` dependency supports
  `GL_KHR_cooperative_matrix`.

### Pointing the build at a specific `glslc`

If your good `glslc` is not on `PATH`, set `INFR_GLSLC` to its full path:

```bash
INFR_GLSLC=/opt/vulkan/1.4.357.0/x86_64/bin/glslc cargo build --release --locked -p infr-cli
```

`build.rs` honours `INFR_GLSLC` and will print a clear error (with this pointer)
if `glslc` is missing or too old.

## Full install from source (Ubuntu/Debian example)

```bash
# 1. Rust toolchain (if not present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 2. Build + runtime dependencies
#    Build needs a C/C++ toolchain and cmake for a couple of native deps.
sudo apt install -y build-essential cmake curl git xz-utils
#    Runtime (to actually run, not just build) needs a Vulkan loader + your GPU driver:
sudo apt install -y libvulkan1 vulkan-tools          # NVIDIA: the nvidia driver instead

# 3. Install a suitable glslc (see above)
./scripts/install-linux.sh

# 4. Build and verify
cargo build --release --locked -p infr-cli
./target/release/infr devices      # should list your Vulkan GPU
./target/release/infr run 'MODEL.gguf'
```

`cargo build --locked -p infr-cli` builds the `infr` binary (chat, serve, bench,
devices). The workspace also contains `infr-gui` (the browser-based dev GUI) —
build it the same way (`cargo build --release -p infr-gui`) if you want it; the
Windows release package omits it.

## Running on Linux

The Vulkan backend is the default device. Select a GPU with `--dev VulkanN` /
`INFR_DEV=VulkanN`. Typical invocations are identical to Windows:

```bash
./target/release/infr run 'D/Models/model.gguf'                      # terminal chat
./target/release/infr serve --addr 127.0.0.1:8080 'model.gguf'        # OpenAI-compatible API
./target/release/infr bench -p 1024 -n 0 -r 1 'model.gguf'            # prefill
```

The OpenAI API base URL is `http://127.0.0.1:8080/v1`.

## Notes and limitations

- **Verified**: full workspace release build on Ubuntu 24.04 (WSL2, Rust 1.98.1,
  LunarG SDK `glslc` v2026.3) compiles cleanly with no source changes beyond the
  `build.rs` diagnostic/`INFR_GLSLC` improvement and an unused-import fix in
  `infr-embedding`. `./target/release/infr --version` and `infr devices` run.
- MoE4All's headline performance numbers are measured on a Windows 11 / RX 7900
  XTX / Vulkan host. Linux works through the same Vulkan path, but per-GPU
  performance and the expert RAM/SSD paging behavior are not separately
  benchmarked here — treat the Windows numbers as the reference, not a Linux
  guarantee.
- The Embedding API spawns an external `llama-server`; on Linux it is located on
  `PATH` or via `--embedding-runner` / `INFR_EMBEDDING_RUNNER` (the LM Studio
  auto-detect is Windows-only by design).
- Cooperative-matrix kernels require a driver that exposes
  `VK_KHR_cooperative_matrix`; modern RADV, ANV, and NVIDIA drivers do.
