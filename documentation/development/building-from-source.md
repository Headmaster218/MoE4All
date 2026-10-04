---
kind: development-guide
status: current
scope: build-from-source
last_verified: 2026-10-04
verified_commit: 41345e6f2caace03869cc0d2e2e89a577650ee46
---

# 从源码构建

发布包目前只有 Windows 版；Linux（以及其他平台）从源码构建。跑起来的部分见
[在 Linux 上构建并运行](../guide/linux-build-and-run.md)。

## 前置条件

- **Rust**：`rust-toolchain.toml` 固定 **1.97.1**（含 `rustfmt`、`clippy`），
  `rustup` 会在首次构建时自动安装，不需要手动 `rustup toolchain install`。
- **`glslc`（shaderc）**：compute shader 在**构建期**编译。dp4a 系列的 shader 需要
  `GL_EXT_integer_dot_product`，因此编译器必须是 **shaderc 2025 或更新**。
  - Ubuntu 24.04 自带的 shaderc 2023.8 **过旧**，构建会失败；
  - Ubuntu 26.04 自带的版本可用（本仓库实测 shaderc 2026.1-1 / glslang 16.1.0）；
  - 其他发行版请确认 `glslc --version` 报告的 shaderc 版本。
- **Vulkan 驱动只在运行阶段需要**，构建不需要：`ash` 通过 `dlopen` 在运行时加载
  `libvulkan`，所以构建机可以没有驱动。运行则需要可用的 Vulkan 实现——AMD 上是
  RADV（Mesa）或 AMDVLK。

## 构建

```sh
sudo apt-get update && sudo apt-get install -y glslc
git clone https://github.com/Headmaster218/MoE4All.git
cd MoE4All
cargo build --release --locked -p infr-cli
```

产物是 `target/release/infr`。

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

## 测试

```sh
cargo test --workspace --locked                          # 默认：纯 CPU 套件
cargo test --workspace --locked -- --include-ignored     # 额外跑真实 GPU 测试
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
