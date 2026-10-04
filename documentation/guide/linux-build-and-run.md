---
kind: guide
status: current
scope: linux-build
last_verified: 2026-10-04
verified_commit: 41345e6f2caace03869cc0d2e2e89a577650ee46
---

# 在 Linux 上构建并运行

发布包目前只有 Windows 版；Linux 从源码构建。构建前的工具链与依赖见
[从源码构建](../development/building-from-source.md)，本页讲怎么把它跑起来。

## 用启动向导

```sh
./Start-INFR-Wizard-Linux.sh
```

向导依次询问：

- 启动模式：终端聊天 / OpenAI 兼容 API / 基准测试；
- 模型：本地 `.gguf` 路径，或 `org/repo[:quant]` 形式的 Hugging Face 引用；
- 资源档位：`aggressive` / `conservative` / `manual`；
- `manual` 档位下再问上下文、Ubatch、KV 格式、RAM 与 VRAM 预算；
- MTP 头（可选，留空即关闭）；
- API 模式下问监听地址、并行槽位，以及可选的视觉投影（mmproj）与嵌入模型。

向导会在启动前打印最终命令并要求确认。选择记录在
`${XDG_CONFIG_HOME:-~/.config}/infr/wizard.conf`，下次直接复用。命令行参数优先于
记忆值；`--no-mtp`、`--no-mmproj`、`--no-embedding` 用于清除已记住的可选项。

`--dry-run` 只打印命令、不启动，且不依赖终端，便于脚本化：

```sh
./Start-INFR-Wizard-Linux.sh --dry-run --mode serve --model m.gguf \
    --profile aggressive --addr 127.0.0.1:8080 --parallel 1
```

> API 鉴权，以及把服务开放到局域网时的注意事项，见
> [使用本地 API](serving/api-quickstart.md)。

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
