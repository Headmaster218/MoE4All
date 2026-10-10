---
kind: guide
status: current
scope: linux-build
last_verified: 2026-10-10
verification: launcher-and-package-fixtures; Linux-GPU-and-release-CI-pending
---

# 在 Linux 上构建并运行

Linux x86_64 发行包暂作为**实验性版本**，与同版本 Windows 包一起由 `release-*` 标签构建。
首次完整构建与 GPU 验收仍待执行；社区试用请反馈系统、驱动及具体启动参数。
构建工具链见 [从源码构建](../development/building-from-source.md)，本页说明启动和更新。

## 发行包

发布页出现 Linux 资产后，下载同版本 `.tar.gz` 与 `.tar.gz.sha256`，放在同一目录：

```sh
sha256sum -c MoE4All-Linux-x86_64-v0.10.0.tar.gz.sha256
tar -xzf MoE4All-Linux-x86_64-v0.10.0.tar.gz
cd MoE4All-Linux-x86_64-v0.10.0
bash ./Start-INFR-Wizard-Linux.sh
```

版本号按实际下载包替换。发行包目标是 x86_64、glibc 2.35+（Ubuntu 22.04 或更新的
兼容系统），不包含 GPU 驱动，也不适用于 Alpine/musl 或 ARM64。
需要 Bash 4.3+、系统自带的 awk/coreutils/findutils 和可用的 Vulkan loader/GPU 驱动；
更新使用 curl、GNU tar/gzip 与 flock（util-linux），没有 Python、jq 或 pip 依赖。
发行包用户不需要 Rust、C++ 编译器或 glslc。Ubuntu 可安装 `libvulkan1`，
AMD/Intel 还需对应 Mesa Vulkan 驱动，NVIDIA 使用其 Vulkan 驱动。

## 源码目录与相对路径

源码构建完成后，保留如下位置：

```text
MoE4All/
  Start-INFR-Wizard-Linux.sh
  target/release/infr
```

发行包结构相同，但 `infr` 在包根目录。启动、交互和更新都在这一个 SH 中，不需要相邻的 `scripts/`。
向导优先找自身目录下的 `infr`，然后找 `target/release/infr`，最后查找 PATH。
可以从任意工作目录用绝对路径调用向导；模型、配置与缓存的相对路径均以**向导所在目录**
为基准，启动引擎时工作目录也切到该目录。带空格的路径作为独立参数传递，不经过 `eval`。

## 用启动向导

```sh
./Start-INFR-Wizard-Linux.sh
```

流程与当前 Windows 向导对齐，不适用的步骤会跳过：

1. 检查更新；有保存设置则询问是否直接按上次启动。
2. 用途：终端聊天 / API / benchmark。
3. LLM、MTP、视觉、Embedding；各模型输入处可选 `R` 打印官方推荐链接。
4. 列出实际 Vulkan GPU，手动选择，不列 CPU 后端。
5. 保守 / 激进 / 手动；手动才问 KV、ubatch、分页、splitter 和 profiler。
6. 默认思考及支持的强度、最大生成量（默认 65536）、可选采样配置。
7. 实验性 CPU-miss：开关、1-3 miss 上限、计算核数。
8. 并发槽位、上下文窗口、可选 SSD KV 缓存。
9. API 监听地址和鉴权，或 benchmark 参数。
10. 打印最终命令并确认启动。

MTP 开启时默认 greedy（`--temp 0`）；双槽机会式 MTP 固定 4-token 验证，
两路同时 decode 时使用普通批量 decode。单槽纯文本 MTP 的串行服务不启用 SSD 会话缓存。
CPU-miss 核数默认 `max(1, 可用物理核心数 - 2)`；Linux 当前仅设置线程数，由 OS 调度，
**没有 Windows 的大小核/物理核绑核能力**，不保证实际预留了两颗大核。
配置项、主要默认值和生成的引擎参数与 Windows 对齐；系统路径、保存目录和更新实现
按平台区分，Linux 不依赖 PowerShell。模型也可输入目录；分片模型排除非首片，
存在多个候选时必须指定具体文件，不会默选其中一个。

向导会在启动前打印最终命令并要求确认。选择记录在
`${XDG_CONFIG_HOME:-~/.config}/infr/wizard-state.json`，首次兼容导入旧 `wizard.conf`，
旧文件不修改。缺少的新字段采用默认值，未知字段忽略；配置只作为数据读取，绝不执行。
JSON 解析器直接内嵌在 SH 中。命令行参数优先于记忆值；
`--no-mtp`、`--no-mmproj`、`--no-embedding` 用于清除已记住的可选项。
已保存的视觉与 Embedding 路径在询问时作为默认值，直接回车不会清空。
显式指定的上下文和资源参数在自动档也生效；交互式重新选择自动档会清除旧手动覆盖。
提示框直接回车保留默认，输入 `-` 清空可选值。SSD KV 默认在包根目录 `kv-sessions/`。
无输入的非交互启动需要 `--yes`；
非本机监听且未启用鉴权时默认拒绝启动，交互模式需要明确确认。
`--no-api-key` 会清除传给子进程的 `INFR_API_KEY`，并显式禁用配置文件中的 API key；
密钥不打印、不写入保存文件。非交互鉴权通过 `INFR_API_KEY` 提供。

`--dry-run` 根据保存设置和命令行参数打印命令，不询问、不检查网络、不保存也不启动：

```sh
./Start-INFR-Wizard-Linux.sh --dry-run --mode serve --model m.gguf \
    --profile aggressive --addr 127.0.0.1:8080 --parallel 1
```

> API 鉴权，以及把服务开放到局域网时的注意事项，见
> [使用本地 API](serving/api-quickstart.md)。

## 更新

正常启动只检查新版并打印链接，不下载或替换文件；源码目录不会被更新器替换。
正式包存在兼容 Linux 资产及其校验文件时，需显式运行 `--update`，
再交互确认更新，**默认不更新**；`--yes` 不会自动同意更新。
只检查更新并退出可运行：

```sh
bash ./Start-INFR-Wizard-Linux.sh --check-update
```

已解压的正式包显式更新：

```sh
bash ./Start-INFR-Wizard-Linux.sh --update
```

`--skip-update-check` 或 `MOE4ALL_NO_UPDATE_CHECK=1` 跳过常规启动检查。
网络检查失败继续离线启动。更新只接受稳定 `release-*` 标签，不混入 Agent 或预发布版本。
更新会校验 SHA-256、包内文件清单和平台，拒绝路径穿越、符号链接与特殊文件；
只替换发行包管理的文件，保留模型、`infr.toml`、保存设置和 `kv-sessions/`。
检测到引擎仍运行时拒绝替换。替换/新引擎版本检查失败会回滚；成功后需重新运行向导。
回滚覆盖更新进程内的失败，不承诺断电或进程被强杀后的事务恢复。
首次 CI 发包、真实 Linux 更新和 GPU 推理尚未验收，当前本地验证为脚本与包夹具测试。

## 直接用 CLI

```sh
./target/release/infr devices               # 列出可见 Vulkan 设备及其显存
./target/release/infr run   <模型>          # 终端聊天
./target/release/infr serve <模型>          # OpenAI 兼容 API
```

`<模型>` 可以是本地 `.gguf` 路径，也可以是 Hugging Face 引用（`org/repo[:quant]`）；
**只有后者在缺失时自动下载，本地路径不会**。

## 相关

- [从源码构建](../development/building-from-source.md)：Rust 工具链、glslc 版本要求、
  测试命令与二进制可移植性注意事项。
- [配置参考](../reference/configuration.md)：完整 TOML、环境变量与 `--set` 规则。
- [使用本地 API](serving/api-quickstart.md)：Base URL、鉴权与接口示例。
- [运行时资源生命周期](../architecture/memory/runtime-resource-lifecycle.md)：显存预算、
  宿主层与 Host DMA 的机制说明。
