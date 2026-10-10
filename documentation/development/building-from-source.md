---
kind: development-guide
status: current
scope: build-from-source
last_verified: 2026-10-10
verification: local-launcher-and-package-tests; Linux-release-CI-pending
---

# 从源码构建

Linux x86_64 的发包 CI 已加入；首次完整 CI 与 GPU 验收尚待执行。源码构建与运行见
[在 Linux 上构建并运行](../guide/linux-build-and-run.md)。

## 前置条件

- **Rust**：`rust-toolchain.toml` 固定 **1.97.1**（含 `rustfmt`、`clippy`），
  `rustup` 会在首次构建时自动安装，不需要手动 `rustup toolchain install`。
- **Bash 4.3+**：Linux 启动和更新仅使用单个 SH 与系统基础工具，不依赖 Python 或 jq。
- **Python 3.8+（仅开发侧）**：CI 打包、包夹具测试及从源码构建 shaderc 使用标准库，不需 pip 包；发行包不包含 Python 文件。
- **`glslc`（shaderc）**：compute shader 在**构建期**编译。dp4a 系列的 shader 需要
  `GL_EXT_integer_dot_product`，因此编译器必须是 **shaderc 2025 或更新**。
  - Ubuntu 24.04 自带的 shaderc 2023.8 **过旧**，构建会失败；
  - Ubuntu 26.04 自带的版本可用（本仓库实测 shaderc 2026.1-1 / glslang 16.1.0）；
  - 其他发行版请确认 `glslc --version` 报告的 shaderc 版本。
- **Vulkan 驱动只在运行阶段需要**，构建不需要：`ash` 通过 `dlopen` 在运行时加载
  `libvulkan`，所以构建机可以没有驱动。运行则需要可用的 Vulkan 实现——AMD 上是
  RADV（Mesa）或 AMDVLK。

## 构建

以下安装命令适用于 Ubuntu 26.04；其他发行版先按上述要求安装较新的 `glslc`。

```sh
sudo apt-get update && sudo apt-get install -y glslc build-essential
git clone https://github.com/Headmaster218/MoE4All.git
cd MoE4All
cargo build --release --locked -p infr-cli
```

产物是 `target/release/infr`。

Ubuntu 22.04/24.04 可以保留旧 glibc，并单独编译新版 shaderc：

```sh
sudo apt-get install -y cmake ninja-build build-essential python3 git
bash scripts/install-shaderc-linux.sh "$HOME/.local/moe4all-shaderc"
export PATH="$HOME/.local/moe4all-shaderc/bin:$PATH"
cargo build --release --locked -p infr-cli
```

### 为什么 CI 钉 `ubuntu-26.04`

`.github/workflows/ci.yml` 的 fmt / clippy / test / build 作业都跑 `ubuntu-26.04`，
原因就是上面那条 shaderc 版本要求：24.04 的 2023.8 编不过 dp4a shader，26.04 的
版本满足下限。这不是为了追新，而是被 shader 编译器卡住的结果。

## 二进制可移植性

`.cargo/config.toml` 设置了 `-C target-cpu=native`。CPU 后端的标量 dequant/matvec
循环依赖它自动向量化（AVX2 / AVX-512），所以**构建产物与本机 ISA 绑定**，拷到指令集
不同的机器上可能直接非法指令退出。需要跨机分发时，请覆盖该设置，例如：

```sh
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release --locked -p infr-cli
```

`.github/workflows/release-linux.yml` 在 Ubuntu 22.04 构建，目标为 Linux x86_64、
glibc 2.35+，独立构建 shaderc 2026.3 并覆盖 `target-cpu=native`。不适用于 Alpine/musl、
ARM64 或旧于该 glibc 的系统。AVX2 + FMA3 的 CPU-miss 内核仍由运行时检测后启用。
首次 CI 实际执行前，以上是构建目标而非已通过的跨发行版验收。

构建后打包（版本必须与 `infr --version` 相同）：

```sh
python3 scripts/build-image-codecs.py
python3 scripts/linux_release.py package --version 0.10.0 --codecs target/image-codecs
```

产物为 `dist/MoE4All-Linux-x86_64-v0.10.0.tar.gz` 及 `.sha256`；只打包引擎、
单个 SH 向导（含更新）、HEIC/AVIF 解码动态库、公开文档与许可证，不包含 Python、`infr.toml`、保存设置、模型和 KV 缓存。
解码库构建还需要 CMake、Ninja、Git；AVIF 需要 Meson、NASM。这些仅为开发/CI 依赖，
最终用户不需要安装 Python 或系统 HEIC/AVIF 库。Windows 在 x64 MSVC 开发环境运行相同
`scripts/build-image-codecs.py` 后使用 `scripts/package-windows.ps1` 打包；可用 `--heic-only`
生成不含 AVIF 的解码库。对应源码包 `image-codec-sources.tar.gz` 作为独立发布附件提供。
推送 `release-<版本>` 标签时 CI 构建并上传；手动执行工作流只生成 artifact，不发布。

## 测试

```sh
cargo test --workspace --locked                          # 默认：纯 CPU 套件
cargo test --workspace --locked -- --include-ignored     # 额外跑真实 GPU 测试
```

不编译引擎、不占显卡的 Linux 向导/打包/更新测试：

```sh
bash scripts/smoke-linux-wizard.sh
```

Vulkan 集成测试都标了 `#[ignore]`，因为它们需要一块真实的 Vulkan 设备并会占用大量
显存。CI 只跑前者（`cargo nextest run --workspace --locked`，不带 `--include-ignored`），
所以这些测试需要在本机手动执行。跑之前确认显存空闲：同一块 GPU 上同时跑模型服务会
让测试因分配失败而报错。

## 平台注意：大 MoE 的宿主层与 GTT

分页 MoE 模型可以把宿主层映射进 GPU 孔径。AMD 上该孔径受 **GTT** 限制，其大小在
`amdgpu` 模块初始化时一次确定：

```
# /etc/modprobe.d/amdgpu-gtt.conf
options amdgpu gttsize=<MiB>
```

它在模块加载（重启）时生效，运行期改不动。调大时要保证宿主层、页缓存与模型本身
加起来仍在物理内存之内。机制细节见
[运行时资源生命周期](../architecture/memory/runtime-resource-lifecycle.md)。
