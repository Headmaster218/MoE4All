# INFR 从零安装与启动

本文面向刚刚 clone 本仓库、尚未安装构建环境的用户。主流程是本项目当前实际开发和验证的环境：**原生 Windows 11 + Vulkan**。Linux 和 macOS 的最短流程在文末。

截至 2026-08-26，Windows 流程已在 AMD Radeon RX 7900 XTX 上用 Rust stable、Vulkan SDK 1.4.357.0（shaderc 2026.3）验证。版本不要求完全相同，但 Rust 和 Vulkan SDK 建议使用当前稳定版。

## 1. 需要安装什么

| 项目 | 是否必需 | 用途 |
|---|---:|---|
| 64 位 Windows 11 和最新稳定 GPU 驱动 | 是 | 驱动提供运行时 Vulkan 实现 |
| [Git for Windows](https://git-scm.com/download/win) | 是 | clone 和更新仓库 |
| [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) | 是 | Rust MSVC 目标所需的链接器和 Windows 库 |
| [Rust stable（rustup）](https://www.rust-lang.org/tools/install) | 是 | 编译 Rust workspace |
| [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) | 是 | 构建时用 `glslc` 把大量 GLSL compute shader 编译为 SPIR-V |
| Node.js、Python、CMake、Ninja | 否 | CLI 和内置 GUI 都不依赖这些工具 |
| llama.cpp / llama-server | 否 | 仅 `compare` 或显式选择 Embedding 兼容 runner 时需要 |

模型必须是本仓库支持架构的 **GGUF**。大模型还需要足够的 SSD 空间；分片 GGUF 的全部分片必须位于同一目录。建议使用本地 NVMe SSD，并让 Windows 页面文件保持启用。

## 2. Windows 11 安装

### 2.1 安装驱动和工具链

1. 从 AMD、NVIDIA 或 Intel 官网安装适合 GPU 的最新稳定驱动，完成后重启。
2. 安装 Git for Windows。
3. 安装 Visual Studio Build Tools（2022 或更新稳定版），选择 **Desktop development with C++ / 使用 C++ 的桌面开发**，并确认包含：
   - MSVC x64/x86 C++ build tools；
   - Windows 10 或 Windows 11 SDK。
4. 通过 rustup 安装 64 位 MSVC Rust，然后选择 stable 工具链。
5. 安装当前稳定版 LunarG Vulkan SDK。SDK 安装器应设置 `VULKAN_SDK`，并把其 `Bin` 目录加入 `PATH`。
6. 关闭并重新打开 PowerShell，使新的环境变量生效。

本项目的 shader 使用 Vulkan 1.3 和较新的 GLSL 扩展。旧版 `glslc`（例如 shaderc 2023.8）不够；请使用当前 Vulkan SDK，建议 shaderc 2025 或更新版本。

### 2.2 检查环境

在一个新的 **64 位 PowerShell** 中执行：

```powershell
git --version
rustup default stable-x86_64-pc-windows-msvc
rustup update stable
rustc -vV
cargo --version
glslc --version
$env:VULKAN_SDK
```

`rustc -vV` 的 `host` 应为 `x86_64-pc-windows-msvc`，`glslc --version` 必须成功。普通 PowerShell 中看不到 `cl.exe` 不一定有问题，Rust 可以通过 Visual Studio 的安装信息找到 MSVC；最终以 Cargo 能否链接为准。

若 SDK 已安装但 `glslc` 仍找不到，可在当前 PowerShell 临时补上路径：

```powershell
$env:Path = (Join-Path $env:VULKAN_SDK 'Bin') + ';' + $env:Path
glslc --version
```

### 2.3 Clone 和构建

```powershell
Set-Location 'D:\Projects'
git clone https://github.com/Headmaster218/infr.git
Set-Location '.\infr'
cargo build --release --locked -p infr-cli
```

第一次构建会下载 crates，并编译数量很多的 Vulkan shader，耗时和 `target` 目录都可能明显大于普通 Rust 项目。后续构建会增量复用结果。

仓库的 [`.cargo/config.toml`](.cargo/config.toml) 使用 `target-cpu=native`，因此 release 二进制针对**执行构建的这台 CPU**生成。请在目标机器上构建；不要把它随意复制到指令集更旧的电脑。

构建成功后，程序位于：

```text
target\release\infr.exe
```

## 3. 先验证 Vulkan，再加载模型

```powershell
$infr = (Resolve-Path '.\target\release\infr.exe').Path
& $infr --version
& $infr devices
```

正常情况下会列出至少一个 `VulkanN`，例如 `Vulkan0: AMD Radeon RX 7900 XTX`。后续的 `--dev Vulkan0` 使用这里显示的编号。`external_memory_host` 等扩展会影响 Host DMA 等性能路径，但不是基本推理能否运行的前提；不支持时会回退到已有搬运路径。

如果 `devices` 看不到 GPU，先修复驱动/Vulkan 运行时，不要通过调模型缓存参数绕过。Vulkan SDK 提供的 `vulkaninfo --summary` 也可用于区分“SDK 已安装”和“GPU 驱动工作正常”。

## 4. 用小模型做首次冒烟测试

第一次不要直接从 80–160 GiB 的 MoE 开始。先用 Qwen3 0.6B 验证下载、GGUF、Vulkan、tokenizer 和生成全链路：

```powershell
$model = 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M'
& $infr run --dev Vulkan0 --ctx 8192 --max-new 64 $model '请用一句话介绍自己。'
```

`run` 支持自动下载缺失模型。也可以先显式下载：

```powershell
& $infr pull $model
```

网络环境不适合自动下载时，建议直接下载这个单文件模型，然后按下文的本地 GGUF 方式启动：

- [Hugging Face 官方直链](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true)
- [ModelScope 国内镜像直链](https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_K_M.gguf)

访问受限的 HuggingFace 仓库时，在当前会话设置 token：

```powershell
$env:HF_TOKEN = 'hf_...'
```

使用本地 GGUF 时，直接传绝对路径。PowerShell 变量和调用运算符 `&` 可以可靠处理空格：

```powershell
$model = 'D:\Models\Qwen3-0.6B-Q4_K_M.gguf'
& $infr run --dev Vulkan0 --ctx 8192 --max-new 64 $model 'Hello'
```

分片模型应传第一片，例如 `model-00001-of-00004.gguf`；其余分片由加载器从同一目录发现。

做一个短 benchmark 冒烟：

```powershell
& $infr bench --dev Vulkan0 --ctx 8192 -u 256 -p 256 -n 0 -r 1 $model
& $infr bench --dev Vulkan0 --ctx 8192 -u 256 -p 0 -n 32 --synthetic-depth 4096 -r 1 $model
```

第二条命令会分配并初始化真实 KV 状态，但使用无语义的 synthetic context；它用于性能测试，不用于检查回答质量。

## 5. 三种日常启动方式

### 5.1 交互式终端向导

源码目录中先完成一次 CLI 构建，然后双击根目录的 `Start-INFR-Wizard.cmd`，或在 PowerShell 执行：

```powershell
.\Start-INFR-Wizard.cmd
```

向导支持中英双语提示，默认进入实时终端对话，也可启动 OpenAI 兼容 API 服务器或运行 benchmark。自动配置模式会保留引擎的 GPU、上下文、VRAM/RAM 和 KV 自动探测；高级模式可设置设备、ubatch、分页、submit splitter 和 profiler。服务器模式会引导配置监听地址、并发会话数和可选 API key；API key 不写入设置文件。

发布包可将 `infr.exe`、`Start-INFR-Wizard.cmd` 和 `scripts\infr-wizard.ps1` 按原目录关系放在一起，向导会优先使用 CMD 同目录的 `infr.exe`。源码目录中没有该文件时，则自动使用 `target\release\infr.exe`。上次非敏感设置保存在 `gui-data\wizard-state.json`。

如果本机执行策略阻止 `.ps1`，可只为这次启动绕过：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File '.\scripts\infr-wizard.ps1'
```

### 5.2 浏览器 GUI

首次仅在本机使用时，建议显式绑定回环地址：

```powershell
.\Start-INFR-GUI.cmd -ListenAddress 127.0.0.1:8180
```

GUI 启动器会自动执行 release 增量构建，同时构建 `infr.exe` 和 `infr-gui.exe`；不需要 Node.js。浏览器打开 `http://127.0.0.1:8180`，输入控制台显示的 management key。GUI 可管理模型目录、配置、内存估算、下载和 `infr serve` worker。

直接双击脚本时默认监听 `0.0.0.0:8180`，会对本机网络接口开放。除非已经配置好防火墙和可信内网，不要把 8180 暴露到公网。详细说明见 [`crates/infr-gui/README.md`](crates/infr-gui/README.md)。

### 5.3 OpenAI 兼容 API

```powershell
$env:INFR_API_KEY = 'change-me'
& $infr serve --dev Vulkan0 --ctx 8192 --addr 127.0.0.1:8080 $model
```

在另一个 PowerShell 中测试：

```powershell
$headers = @{ Authorization = 'Bearer change-me' }
$request = @{
    model = 'local'
    messages = @(@{ role = 'user'; content = '你好，请回复一句话。' })
    stream = $false
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/chat/completions' `
    -Method Post -Headers $headers -ContentType 'application/json' -Body $request
```

不设置 `INFR_API_KEY` 时默认无鉴权，只适合回环地址或受信网络。

## 6. 配置和环境变量

基本启动**不需要**设置任何 `INFR_*`：设备、上下文、VRAM 和可用主机内存都有自动探测。推荐的优先级是：先用默认值跑通，再通过 GUI 估算或日志决定是否固定预算。

配置的覆盖顺序是：

```text
内置默认值 < infr.toml < INFR_* 环境变量 < CLI 参数/--set
```

要创建项目本地配置，可复制带完整注释的示例：

```powershell
Copy-Item '.\infr.example.toml' '.\infr.toml'
```

常用的当前会话变量示例：

```powershell
$env:INFR_DEV = 'Vulkan0'
$env:INFR_CTX = '32k'
$env:INFR_UBATCH = '512'
$env:INFR_KV_TYPE_K = 'q8_0'
$env:INFR_KV_TYPE_V = 'q8_0'
```

这些设置会影响从该 PowerShell 启动的后续进程。删除一个临时值：

```powershell
Remove-Item Env:INFR_CTX
```

长期配置优先写入 `infr.toml`，避免遗忘的全局环境变量悄悄改变 benchmark。完整字段、环境变量映射和 `--set` 语义见 [`docs/config.md`](docs/config.md)。

## 7. 大 MoE 模型的首次启动

1. 先确认小模型已经能生成，并保留完整启动日志。
2. 让所有 GGUF 分片位于同一块高速 SSD 的同一目录，并传入第一片。
3. 首次使用自动 VRAM/RAM 预算，或先在 GUI 中加入模型并执行“重新估算”。不要直接照抄另一台机器的 12g/45g 等实验值。
4. 需要固定 Qwen 长上下文 KV 时，可显式使用 `--set kv.type_k=q8_0 --set kv.type_v=q8_0`；其他架构先确认其 KV 实现支持该格式。
5. 显存预算是 INFR 的总设备内存上限，不是只给 Expert cache 的大小。固定权重、KV、运行时 scratch、staging 和 Expert arena 都要留在预算内。
6. RAM 预算小于 Expert 总量时会启用 bounded RAM/SSD tier。SSD miss 会影响首轮 prefill/decode，因此“大模型能加载”与“已经预热后的稳定吞吐”要分开判断。
7. Host DMA 默认开启；驱动支持并成功导入的 RAM 前缀使用 Vulkan DMA，其余范围自动回退，不需要手工设置环境变量。

一个不固定内存预算的 Qwen 启动模板：

```powershell
$model = 'D:\Models\Qwen-MoE-00001-of-00004.gguf'
& $infr run --dev Vulkan0 --ctx 32k -u 512 `
    --set kv.type_k=q8_0 --set kv.type_v=q8_0 $model
```

如果出现显存预算 guard、设备丢失或 Windows 桌面同时占用大量显存，依次尝试减小 `--ctx`、`-u`，再设置保守的 `device.vram_budget` / `device.vram_reserve`；不要先关闭 guard。

## 8. 常见问题

### `failed to run glslc`

Vulkan SDK 未安装或其 `Bin` 不在 `PATH`。重新打开终端，检查 `$env:VULKAN_SDK` 和 `glslc --version`。

### shader 编译报扩展或语法错误

通常是 shaderc 太旧。更新 LunarG Vulkan SDK；本项目不支持 Ubuntu 24.04 自带的 shaderc 2023.8。

### Rust 链接报 `link.exe`、Windows SDK 或系统库错误

通过 Visual Studio Installer 修改 Build Tools，安装“使用 C++ 的桌面开发”、MSVC x64/x86 tools 和 Windows SDK，然后重新打开 PowerShell。

当前 Windows release 链接可能输出 `LNK4098: 默认库 LIBCMT 与其他库的使用冲突`。它目前是 warning；若 Cargo 最终显示 `Finished release profile`、退出码为 0，且 `infr.exe devices` 正常，就不属于构建失败。不要因为这条 warning 删除或替换 `Cargo.lock`。

### `infr devices` 没有设备或加载 Vulkan 失败

这是 GPU 驱动/Vulkan runtime 问题，不是 `glslc` 问题。更新厂商驱动，重启，并用 `vulkaninfo --summary` 复核。

### 模型只加载到第一片后报缺文件

确认文件名仍为标准 `00001-of-000NN` 形式，全部分片完整且在同一目录；启动参数传第一片。

### 大模型 OOM 或预算 guard 拒绝启动

先减小 context 和 ubatch；使用 GUI 的当前模型估算；为 Windows 桌面、驱动和运行时保留显存。只有看懂日志里的 fixed/KV/runtime/Expert 分项后，再固定 `paging.cache` 或 `paging.dram`。

### 第一次构建很慢

这是预期行为：Vulkan crate 会编译大量 shader 变体。不要删除 `target`；后续增量构建会快很多。`--locked` 应保留，以确保使用仓库提交的 `Cargo.lock`。

## 9. Linux 和 macOS

Linux CI 使用 Ubuntu 26.04，因为它提供足够新的 `glslc`。典型环境：

```bash
sudo apt update
sudo apt install -y git build-essential glslc libvulkan1 vulkan-tools
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup update stable
git clone https://github.com/Headmaster218/infr.git
cd infr
cargo build --release --locked -p infr-cli
./target/release/infr devices
```

另行安装 GPU 厂商的 Vulkan 驱动。Ubuntu 24.04 用户需要从其他可信来源安装新版 shaderc，不能使用其旧版 `glslc` 包。

Apple Silicon/macOS 使用 Metal 运行，但当前 workspace 构建仍会编译 Vulkan crate 的 shader，因此也需要 `glslc`：

```bash
xcode-select --install
brew install shaderc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
git clone https://github.com/Headmaster218/infr.git
cd infr
cargo build --release --locked -p infr-cli
./target/release/infr run --dev metal MODEL.gguf
```

## 10. 可交付检查单

一个新 clone 至少应完成以下检查，才算环境真正可用：

- `cargo build --release --locked -p infr-cli` 成功；
- `target\release\infr.exe devices` 能列出目标 Vulkan GPU；
- 一个小 GGUF 能完成加载并生成可读文本；
- 短 prefill 和 synthetic-depth decode benchmark 都能输出 tok/s；
- 计划使用 GUI 时，`http://127.0.0.1:8180` 可打开并能启动/停止 worker；
- 计划提供 API 时，`/v1/chat/completions` 能完成一次带鉴权请求。
