---
kind: change-history
status: historical
scope: browser-control-plane
preserved_in_tag: release-0.9.0
---

# INFR 浏览器控制面

> 历史 GUI 操作与设计记录；其中绝对路径、默认地址和资源估算行为来自当时环境。0.9.0 发行版的实际启动参数以安装包和配置为准。

`infr-gui` 是一个由服务器托管的小型 Web 控制面。它在启动、停止或切换独立的 `infr serve` 工作进程时保持在线。
因此，模型切换会在加载下一个模型前释放旧 Vulkan 进程及其全部分配。

## 在 Windows 上启动

双击仓库根目录中的 `Start-INFR-GUI.cmd`，或运行：

```powershell
& 'D:\AIinfr\infr\crates\infr-gui\Start-INFR-GUI.ps1'
```

启动器将：

1. 以发布模式增量构建 `infr.exe` 和 `infr-gui.exe`；
2. 首次创建 `gui-data\admin.key`，后续启动复用它；
3. 监听 `0.0.0.0:8180`，以便通过 ZeroTier 访问页面；
4. 打印本地/ZeroTier URL 和管理密钥；
5. 在前台运行。`Ctrl+C` 会在工作进程优雅排空后停止 GUI。

不会创建计划任务或自启动项。要使用其他监听地址：

```powershell
& 'D:\AIinfr\infr\crates\infr-gui\Start-INFR-GUI.ps1' -ListenAddress '192.168.195.1:8180'
```

如果 Windows 防火墙阻止所选端口，请在 ZeroTier 网络配置文件中允许该 TCP 端口。
启动器不会修改防火墙规则。

## 常规工作流

- 在 **模型库** 中添加一个或多个服务器目录。不会自动扫描磁盘。
- 选择一个 GGUF。分片 GGUF 按其第一个分片分组；会检测 `mmproj` 文件，但不会将其作为语言模型处理。
- 保存一个或多个配置文件。收藏项优先排列，其后是最近使用的模型。
- 加载前选择 **重新估算**。KV、权重打包余量、架构特定驱动保留、加载后保留、专家目标和弹性池数值使用与
  Vulkan 加载器相同的核算方式。主机缓存自动大小取样自当前可用系统内存。
- 选择 **启动 / 切换**。切换会请求优雅关闭，最多等待 660 秒让活跃 GPU 工作排空，然后启动替代工作进程。
- 仅当优雅排空卡住时使用 **强制停止**。它会直接终止工作进程。

GUI 显示工作进程阶段、PID、地址、最近日志、最新上报的预填充/解码速率，以及运行中工作进程上报的内存方案：
KV 布局/上下文、专家目标、弹性/统一 arena、主机缓存模式/覆盖范围和主机 DMA 导入覆盖范围。一次只管理一个工作进程和一个后台下载。

## 下载

使用类似 `org/repo:Q6_K` 的 INFR/HuggingFace 模型引用。选择器支持：

- `https://huggingface.co`
- `https://hf-mirror.com`
- 其他兼容 HuggingFace 的 HTTP(S) 源

下载使用现有的 `infr pull` 缓存、续传、校验和、分片和伴随文件逻辑。
生成的 GGUF 会自动加入目录。对于需要凭据的仓库，可在启动器环境中设置 `HF_TOKEN`。

## 密钥与暴露范围

GUI 管理密钥与工作进程的兼容 OpenAI API 密钥相互独立：

- `gui-data\admin.key` 通过 `--key-file` 传入，并保护每个管理 API。浏览器会将输入值存入本地存储。
- **OpenAI API 密钥** 通过 `INFR_API_KEY` 传给工作进程，而不是写入进程命令行。

通信使用明文 HTTP，设计用于加密的 ZeroTier 网络。请勿将端口 8180 暴露给公共互联网。配置文件（包括已配置的
工作进程 API 密钥）存储在 `gui-data\state.json` 中；若不信任其他本地用户，请将该目录限制为服务器帐户访问。

## 当前能力

- 聊天/补全配置文件运行 `infr serve`。它们可将一个原生嵌入 GGUF 附加到同一兼容 OpenAI 的服务。在 Vulkan 上，LLM 先初始化，嵌入请求从统一弹性显存 arena 借用冷区间；每个请求后释放权重并恢复专家槽位。显式设置嵌入运行器可保留 `llama.cpp` 兼容路径。
- 独立嵌入配置文件默认使用 INFR 原生 CPU/Vulkan 引擎。显式运行器选择兼容模式。
- 分页 MoE 模型使用当前的显存/RAM/SSD 层级。GUI 写入 `device.ram_budget`；显式值是工作进程的总常驻内存目标，先扣除其活动工作集，再将剩余字节分配给完整主机存储或有界包容式 RAM/SSD 缓存。自动模式保留主机余量。也会公开 `paging.dram_bypass`。
- 默认启用主机 DMA。受支持的 Vulkan 驱动通过 `VK_EXT_external_memory_host` 导入对齐的 RAM arena；不受支持的范围会回退到 CPU/ReBAR 路径。GUI 从工作进程启动日志中报告实际导入字节数。
- 每个配置文件都可使用分页器统计和 CSV 跟踪。通过由 INFR 配置清单生成的高级编辑器，其他配置路径仍然可用。
- 会发现并显示 Vision/mmproj 文件，但尚不会将其传给推理工作进程。

持久状态仅位于被 Git 忽略的 `gui-data` 下。移除该目录会重置 GUI 目录、配置文件、收藏、最近使用项和生成的管理密钥；不会
删除模型文件。
