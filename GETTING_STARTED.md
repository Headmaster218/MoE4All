# MoE4All 快速开始 / Getting Started

这份指南优先面向直接下载 Windows 发布包的用户。普通使用不需要安装 Rust、
Visual Studio 或 Vulkan SDK。源码构建和开发环境放在后半部分。

[返回项目首页](README.md) |
[下载最新版](https://github.com/Headmaster218/MoE4All/releases/latest) |
[English quick start](#english-quick-start)

## 1. 使用前需要准备什么

| 项目 | 是否必须 | 说明 |
|---|---:|---|
| 64 位 Windows 11 | 是 | 当前主要开发和验证环境 |
| AMD 显卡驱动 | 是 | 驱动需要提供可用的 Vulkan 运行时 |
| MoE4All Windows ZIP | 是 | 从 GitHub Release 下载并完整解压 |
| GGUF 模型 | 是 | 模型自备；单文件或完整分片组 |
| 足够的 SSD 空间 | 大模型需要 | 模型可大于显存和内存，但完整文件必须位于本地存储 |
| Rust、VS、Vulkan SDK | 运行发布版不需要 | 只有从源码构建时才需要 |

MoE4All 当前重点优化和实测 AMD Radeon。其他带 Vulkan 驱动的 GPU 可能能够
运行上游支持的路径，但本项目不会对它们作同等程度的兼容和性能保证。

大型 MoE 对内存和 SSD 的需求差异很大。24 GiB 显存不代表只能运行 24 GiB
模型，但模型超出显存越多，就越依赖 RAM 容量、SSD 速度和专家命中率。

## 2. 下载和解压

1. 打开 [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases/latest)。
2. 下载 `MoE4All-Windows-x86_64-v版本号.zip`。
3. 将 ZIP 完整解压到一个普通目录，例如 `D:\MoE4All`。
4. 不要直接在压缩包预览窗口中运行 CMD，也不要只单独取出 `infr.exe`。

解压后至少应看到：

```text
infr.exe
Start-INFR-Wizard.cmd
GETTING_STARTED.md
scripts\infr-wizard.ps1
LICENSE
LICENSE-MIT
NOTICE
```

发布包使用通用 x86-64 CPU 目标并静态链接 Visual C++ runtime。Vulkan 由显卡
驱动在运行时提供，因此用户不需要安装用于编译 shader 的 Vulkan SDK。

### Windows 安全提示

当前发布版尚未进行商业 Authenticode 代码签名。Windows SmartScreen 可能显示
“无法识别的应用”。请只从本项目 GitHub Release 下载，并可使用同一 Release
提供的 `.sha256` 文件校验 ZIP。

PowerShell 执行策略较严格的机器也可能拦截向导脚本。遇到这种情况先确认文件
确实来自官方 Release，再参考[常见问题](#10-常见问题--troubleshooting)。

## 3. 准备 GGUF 模型

### 什么是 GGUF

GGUF 是模型权重和运行 metadata 的文件格式。MoE4All 读取其中的架构、张量、
tokenizer 和 chat template。仅有模型名称或普通 Transformers 权重目录不够，
需要下载对应的 `.gguf` 文件。

### 单文件和分片模型

小模型通常只有一个文件：

```text
Qwen3-0.6B-Q4_K_M.gguf
```

大模型常被拆成多片：

```text
Model-Q5_K_M-00001-of-00004.gguf
Model-Q5_K_M-00002-of-00004.gguf
Model-Q5_K_M-00003-of-00004.gguf
Model-Q5_K_M-00004-of-00004.gguf
```

所有分片必须属于同一量化、位于同一目录且下载完整。加载器支持从其中任意一片
识别整个分片组；为了和其他工具的习惯一致，选择第一片最直观。

### 量化名称怎么选

模型名中的 `Q4`、`Q5`、`Q6`、`Q8`、`IQ4`、`MXFP4` 等通常表示权重量化。
数字和格式会影响文件大小、精度、速度及 kernel 支持，不能只按“数字越大越好”
判断。

第一次使用建议：

- 先选择项目已明确支持的 GGUF 架构和常见量化。
- 小模型可从 `Q4_K_M` 或 `Q5_K_M` 开始。
- 大型 MoE 优先参考项目实测模型所用的量化，不要只看总参数量。
- 确认 SSD 还有足够空间保存全部分片。

### 一个用于确认环境的小模型

建议第一次先用 Qwen3 0.6B 验证驱动、GGUF、tokenizer 和生成链路：

- [Hugging Face 直链](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true)
- [ModelScope 国内直链](https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_K_M.gguf)

推荐自行下载本地 GGUF。CLI 也支持 `org/repo:quant` 形式的自动下载，但网络、
代理、鉴权和大文件断点环境差异很大，本地模型路径通常最容易排错。

## 4. 第一次启动

双击解压目录中的：

```text
Start-INFR-Wizard.cmd
```

向导会记住上一次的非敏感设置。看到带方括号的默认值时，直接按 Enter 就会
复用它。

### 选择模型

向导会列出最近使用的模型。也可以选择输入新路径，然后：

- 粘贴完整 GGUF 路径；或
- 把 GGUF 文件拖进已经打开的终端窗口，再按 Enter。
- 直接把 GGUF 拖到 `Start-INFR-Wizard.cmd` 上，以该模型打开向导。

路径可以包含空格和中文。分片模型只需拖入其中一片，建议选择第一片。

向导启动时会检查 GitHub 上的最新 Release。请求超时很短，断网时会直接继续；
它只提示下载地址，不会自动下载或替换程序。如需完全禁用，可在启动前设置
`MOE4ALL_NO_UPDATE_CHECK=1`。

### 配置模式

普通用户请选择：

```text
[1] 自动配置：保守（推荐）/ Automatic: conservative (recommended)
```

向导提供三种模式：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| 自动配置：保守 | 自动探测资源，保留较多 RAM/VRAM 余量，使用稳妥的 Ubatch 与 Submit 校准 | 首次运行、后台程序较多、未知 GPU |
| 自动配置：激进性能 | 仍自动探测并保留硬性 OOM/TDR 保护，但减少部分余量，使用更大的 Ubatch，并让 Submit 校准探索更大的 cap | 已确认机器稳定，希望提高 prefill/decode 吞吐 |
| 全手动配置 | 显式设置设备、上下文、KV、RAM/VRAM、Ubatch、Submit 和诊断选项 | 可复现实验与高级调优 |

两种自动配置都会让引擎探测 Vulkan GPU、可用显存、系统 RAM 和模型结构，并规划：

- 固定模型权重；
- KV Cache 和 recurrent state；
- prefill/decode 运行时空间；
- GPU expert cache；
- full-RAM 或 bounded RAM/SSD 专家层。

保守档以“在当前机器上可靠启动”为优先目标，不保证一定是最高性能设置。激进档
仍不会绕过统一显存 guard 或 MoE 可运行性下限；如果 Windows 开始换页、模型加载
失败或其它程序也在占用 GPU，请退回保守档。第一次不要照抄其他电脑的 `12g`、
`45g` 或 submit cap。

激进档的较大 Ubatch 通常提高长 prompt 的 prefill 吞吐，但也需要更大的临时工作区，
因此某些显存紧张、专家 miss 较多的 decode 负载未必更快；档位表示资源取向，不是对
所有模型都保证提速。

直接使用 CLI 时可写 `--set device.auto_profile=aggressive`，或设置
`INFR_AUTO_PROFILE=aggressive`。显式的 `device.ram_budget`、
`device.vram_budget`、`device.ubatch` 和 `device.submit_dispatches` 始终优先于自动档位。

### 启动前确认

向导会打印最终命令并询问是否启动。模型加载期间可能出现大量显存、RAM 和分页
规划日志。大型模型首次填充 RAM cache 时会读取 SSD，开头几轮可能明显慢于预热
后的稳定速度。

## 5. 三种运行模式

### 5.1 实时终端对话

这是默认和最适合首次使用的模式。模型加载完成后，在 `>` 提示符输入消息。

```text
exit
quit
:q
```

以上任意一个命令可以退出。`Ctrl+C` 会请求引擎排空当前 GPU 工作并退出。

对话会保留当前会话的 KV 和 recurrent state，不应在每轮都重新计算完整历史。
上下文超过窗口或切换进程后则需要重新建立状态。

### 5.2 OpenAI 兼容 API

服务器模式适合接入聊天前端、脚本或支持自定义 OpenAI Base URL 的应用。

默认地址：

```text
http://127.0.0.1:8080/v1
```

`127.0.0.1` 仅供本机访问。使用 `0.0.0.0` 对局域网开放时应启用 API key，
并配置 Windows 防火墙；不要将无鉴权服务暴露到公网。

并发会话数不是免费的。每个并发 slot 都需要独立 KV Cache，增加 `parallel`
可能缩小每个请求可用的自动上下文窗口。

### 5.3 性能测试

Benchmark 用于测量：

- **Prefill / pp**：读取和处理输入 token 的速度；
- **Decode / tg**：逐 token 生成的速度；
- **Synthetic depth**：不真实计算前面几十万 token，但建立对应长度的 KV 与
  allocator 状态，用于测量长上下文后的推理开销。

Benchmark 不检查回答质量。Synthetic KV 是确定性测试数据，不包含有意义的
语言历史。

## 6. 常用参数是什么意思

普通用户可以一直使用自动配置。下面这些解释主要用于看懂日志和高级设置。

| 参数 | 通俗解释 | 建议 |
|---|---|---|
| Context / `--ctx` | 一次会话最多保留多少 token；越大通常需要越多 KV 内存 | 首次留空自动 |
| Auto profile (`device.auto_profile` / `INFR_AUTO_PROFILE`) | 未手动指定资源与执行参数时使用保守或激进策略 | 首次使用 `conservative`；确认稳定后再试 `aggressive` |
| Max new tokens | 每轮最多生成多少 token，不是上下文总长度 | 避免设置得远高于实际需要 |
| Thinking | 是否向支持的模型请求思考模式 | 不确定时使用模型默认 |
| KV Cache 类型 | 保存历史注意力状态的格式；Q8 通常比 F16 更省空间 | 留空让架构选择；确认支持后再固定 Q8 |
| Ubatch | prefill 每次送入 GPU 的 token 块大小 | 大值可能更快，也需要更多运行时显存 |
| VRAM budget | 引擎可使用的总显存，不只是专家缓存 | 不要把显卡标称容量全部填满 |
| VRAM reserve | 给桌面、驱动波动和额外资源留下的显存 | Windows 主显示卡需要合理余量 |
| GPU expert cache | 显存中可常驻多少专家权重 | 只是总显存预算的一部分 |
| 总 RAM budget (`device.ram_budget` / `INFR_RAM_BUDGET`) | infr 进程的总常驻内存目标；扣除现有工作集后，余量用于专家 cache | 留空自动；手动值可挤出系统冷页；百分比按物理 RAM 总量计算 |
| Host DMA | 让兼容驱动用 Vulkan DMA 从导入 RAM 搬到 VRAM | 默认开启；失败会回退 |
| Submit splitter | 把长 GPU 工作切成多次提交，涉及性能和 Windows TDR | 普通用户保持自动 |
| Parallel slots | API 同时生成的会话数，每个 slot 有独立 KV | 从 1 开始 |

### 关于 Q8 KV

Q8 KV 可以显著降低长上下文缓存空间，但不是所有架构都使用同一套 KV 路径。
向导的“引擎自动”最稳妥。对已验证的 Qwen3.5/Qwen3.6 路径，可以在高级模式
选择 Q8 K + Q8 V；若加载日志报告格式不可用，应退回自动或 F16，而不是只修改
显示字符串。

### 关于显存数字

日志中的 Expert cache 不是整个显存占用。以下内容也会使用 VRAM：

- 固定 Dense、Attention、Embedding 等权重；
- KV Cache 和 recurrent state；
- prefill/decode activation scratch；
- staging、量化临时空间和驱动余量。

因此“显卡还有 18 GiB”不等于可以手工设置 18 GiB expert cache。

## 7. 直接使用命令行

在发布包目录打开 PowerShell：

```powershell
$infr = (Resolve-Path '.\infr.exe').Path
& $infr --version
& $infr devices
```

`devices` 应列出 AMD GPU、设备编号和显存。如果这里没有 Vulkan GPU，应先修复
驱动，不要通过减小模型缓存参数绕过。

### 本地 GGUF 对话

```powershell
$model = 'D:\Models\Qwen3-0.6B-Q4_K_M.gguf'
& $infr run --max-new 256 $model
```

也可以附带第一条消息：

```powershell
& $infr run --max-new 256 $model '请用三句话介绍这个项目。'
```

### 思考模式

```powershell
& $infr run --think $model
& $infr run --no-think $model
```

不传这两个参数时使用模型默认行为。

### 可选的 Hugging Face model ref

```powershell
& $infr pull 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M'
& $infr run  'unsloth/Qwen3-0.6B-GGUF:Q4_K_M'
```

受限仓库可在当前 PowerShell 会话设置 `HF_TOKEN`。对于国内网络或数十 GiB 的
模型，仍建议通过熟悉的下载工具准备本地 GGUF。

## 8. API 示例

启动服务器：

```powershell
$env:INFR_API_KEY = 'change-me'
& $infr serve --addr 127.0.0.1:8080 $model
```

测试健康状态：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8080/health'
```

发送聊天请求：

```powershell
$headers = @{ Authorization = 'Bearer change-me' }
$body = @{
    model = 'local-model'
    messages = @(@{ role = 'user'; content = '你好，请简短介绍自己。' })
    stream = $false
} | ConvertTo-Json -Depth 6

Invoke-RestMethod `
    -Uri 'http://127.0.0.1:8080/v1/chat/completions' `
    -Method Post `
    -Headers $headers `
    -ContentType 'application/json' `
    -Body $body
```

不设置 API key 时默认无鉴权，只适合回环地址或受信网络。

## 9. Benchmark 示例

```powershell
# Prefill 1024 token
& $infr bench -p 1024 -n 0 -r 1 $model

# Decode 128 token
& $infr bench -p 0 -n 128 -r 1 $model

# 模拟已有 100K 上下文后的 decode，不真实 prefill 前面的 100K
& $infr bench --synthetic-depth 100000 -p 0 -n 128 --ctx 131072 -r 1 $model

# 已有 100K context 后再 prefill 4096
& $infr bench --synthetic-depth 100000 -p 4096 -n 0 --ctx 131072 -r 1 $model
```

真实 `-d N` 会实际运行 N token warmup；`--synthetic-depth N` 则直接初始化等效
KV/context 状态。两者不可同时使用。

性能比较时必须记录模型量化、上下文、KV 类型、ubatch、cache 设置、重复次数和
是否已预热。只比较一个 `tok/s` 数字很容易得到错误结论。

## 10. 常见问题 / Troubleshooting

### 双击 CMD 后提示 PowerShell 脚本被禁止

先确认 ZIP 来自项目 Release。然后在解压目录打开 PowerShell，只为本次进程运行：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File '.\scripts\infr-wizard.ps1'
```

后续版本会改善双击入口对下载脚本执行策略的处理。

### `infr.exe devices` 看不到显卡

安装或更新 AMD 官方驱动并重启。发布版不需要 Vulkan SDK，但必须有驱动提供的
Vulkan runtime。先让 `devices` 正常，再加载模型。

### 找不到模型或缺少分片

确认路径指向 `.gguf`，所有 `-NNNNN-of-MMMMM` 文件位于同一目录，并且总分片
数、量化名称和文件前缀一致。不要混用不同下载来源或不同量化的分片。

### 模型加载时显存预算拒绝启动

先关闭占用显存的程序，使用自动配置，减小 context，必要时在已确认支持的架构
上使用 Q8 KV。不要先设置 `INFR_NO_VRAM_GUARD=1`，因为驱动过量提交可能表现为
权重经总线回读、设备丢失或 Windows TDR，而不是干净的 OOM。

### 大模型开头很慢

当专家总量大于 RAM budget 时，MoE4All 使用 bounded RAM/SSD tier。首次加载会
按层预热一部分专家，后续 SSD miss 仍可能逐步填充 RAM cache。首轮速度和稳定
预热速度应分开观察。

### 系统内存占用很高

大型 MoE 会主动利用 RAM 缓存专家。留空时，引擎按当前可用内存自动留出系统余量；
手工 `device.ram_budget`（或 `INFR_RAM_BUDGET`）表示 infr 进程总常驻 RAM 预算，
并允许 Windows 换出其他冷页。该值属于高级覆盖项，应按机器负载和页面文件容量
谨慎设置。旧 `paging.dram` / `INFR_DRAM_CACHE` 只为复现历史基准保留，表示原始
host cache 大小，不应写入新配置。

### 下载的 EXE 被 SmartScreen 提示

当前二进制尚未购买商业代码签名证书。请核对下载域名、Release tag 和 SHA256。
不要从聊天附件或不明网盘获取重打包版本。

## 11. 从源码构建 / Build from source

开发者和需要修改代码的用户才需要本节。

### Windows 工具链

安装：

1. [Git for Windows](https://git-scm.com/download/win)
2. [Rust stable](https://www.rust-lang.org/tools/install)
3. Visual Studio Build Tools，选择“使用 C++ 的桌面开发”、MSVC x64/x86 和
   Windows 10/11 SDK
4. [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)，用于构建时
   的 `glslc`

检查环境：

```powershell
git --version
rustc -vV
cargo --version
glslc --version
```

构建：

```powershell
git clone https://github.com/Headmaster218/MoE4All.git
Set-Location '.\MoE4All'
cargo build --release --locked -p infr-cli
.\target\release\infr.exe devices
```

仓库的 `.cargo/config.toml` 面向本机开发性能，可能使用 `target-cpu=native`。
本地 release 二进制不应随意复制到指令集更旧的 CPU。GitHub Release 使用通用
x86-64 和静态 CRT 配置，更适合作为公开分发版本。

第一次构建会下载 Rust crates，并编译大量 Vulkan compute shader，耗时和
`target` 目录体积都可能明显大于普通 Rust 项目。后续构建会复用增量产物。

浏览器 GUI 当前是源码开发入口，可运行：

```powershell
.\Start-INFR-GUI.cmd -ListenAddress 127.0.0.1:8180
```

GUI 启动器会构建 `infr.exe` 和 `infr-gui.exe`，因此不包含在当前轻量 Windows
用户发布包中。

### Linux

Linux 保留上游 Vulkan 路径，但 MoE4All 当前主要在 Windows AMD 主机验证：

```bash
sudo apt install -y git build-essential glslc libvulkan1 vulkan-tools
git clone https://github.com/Headmaster218/MoE4All.git
cd MoE4All
cargo build --release --locked -p infr-cli
./target/release/infr devices
```

发行版自带的 `glslc` 必须足够新，能够编译项目使用的 Vulkan 扩展。

### macOS

Apple Silicon 使用 Metal backend，但 workspace 构建仍会编译 Vulkan shader：

```bash
xcode-select --install
brew install shaderc
git clone https://github.com/Headmaster218/MoE4All.git
cd MoE4All
cargo build --release --locked -p infr-cli
./target/release/infr run --dev metal MODEL.gguf
```

## 12. 配置来源

基本启动不需要设置任何 `INFR_*` 环境变量。配置优先级为：

```text
内置默认值 < infr.toml < INFR_* 环境变量 < CLI 参数 / --set
```

长期配置建议写入 `infr.toml`，临时实验使用 CLI。环境变量会被子进程继承，容易
在几天后忘记并影响 benchmark。完整字段见
[配置参考](https://github.com/Headmaster218/MoE4All/blob/main/docs/config.md) 和
[`infr.example.toml`](infr.example.toml)。

## English quick start

MoE4All's portable Windows package is the recommended path for end users:

1. Install a current AMD GPU driver with Vulkan support.
2. Download and fully extract the latest `MoE4All-Windows-x86_64-v*.zip` from
   [GitHub Releases](https://github.com/Headmaster218/MoE4All/releases/latest).
3. Download a supported GGUF model. Keep all shards of a split model together.
4. Double-click `Start-INFR-Wizard.cmd`.
5. Choose interactive chat and automatic configuration, then paste a GGUF path,
   drag it into the open prompt, or drop the file directly onto the CMD launcher.

The release package does not require Rust, Visual Studio, or the Vulkan SDK.
Models are not bundled. Start with a small GGUF to validate the driver and
generation path before loading a very large MoE model.

The wizard can also start an OpenAI-compatible server or run prefill/decode
benchmarks. Advanced memory, KV, paging, and submit controls are intended for
users who understand the corresponding startup log. Automatic mode is the
recommended baseline.

For upstream engine documentation, see the original
[kryptic-sh/infr README](https://github.com/kryptic-sh/infr#readme). MoE4All
project documentation starts at [README.md](README.md) and
[docs/README.md](https://github.com/Headmaster218/MoE4All/blob/main/docs/README.md).
