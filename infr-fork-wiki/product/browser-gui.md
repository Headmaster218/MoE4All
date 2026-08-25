# 浏览器 GUI 与服务 Supervision

[首页](../README.md) / [Product](README.md) / Browser GUI

## 目标

GUI 运行在服务器，通过 ZeroTier 地址和端口用浏览器访问。它不是把 inference engine 重写
进 Web 进程，而是一个长期在线的 control plane，管理独立 `infr serve` worker：

```text
Browser ─HTTP→ infr-gui :8180
                    ├─ catalog/profile/state
                    ├─ download task
                    └─ supervised infr serve worker :user-port
```

模型切换时旧 worker 完整退出，Vulkan device/VRAM/Host allocations 由进程生命周期释放，
再启动新模型。这样比试图在同一 Rust object graph 内热卸载所有 kernel/resource 稳妥。

## 已实现能力

- 手动添加服务器模型目录，不主动扫描全盘；
- GGUF shard group、mmproj 识别；
- 收藏、最近使用、多个 profile；
- Hugging Face、`hf-mirror.com` 和兼容镜像下载；
- 下载 resume/checksum/shard/companion file 复用现有 pull 逻辑；
- VRAM/RAM budget、context、KV、port、API key 与 advanced config；
- 启动、优雅停止、强制停止、切换；
- worker phase、PID、address、最近日志、Prefill/Decode 实时速度；
- Chat/Completion 和 Embedding task/profile；
- Vision/mmproj、Rerank、memory tier 字段预留。

## 启动器生命周期

Windows 启动脚本：

1. incremental release build `infr.exe` / `infr-gui.exe`；
2. 第一次生成 `gui-data/admin.key`，后续复用；
3. 默认监听 `0.0.0.0:8180`；
4. foreground 运行并输出本机/ZeroTier URL；
5. `Ctrl+C` 触发 graceful shutdown，并先 drain worker。

早期“关掉 GUI cmd 后 worker 留在后台、重开 GUI 无法控制”来自进程生命周期没有挂到 GUI
shutdown。当前正常 Ctrl+C/console shutdown 会 `stop_and_wait`。但如果 GUI 被任务管理器强杀
或机器异常终止，当前没有通用的 orphan PID adoption；这应视为剩余边界，不把强杀等同于
优雅关闭。

## Worker 停止语义

- Graceful：写独立 stop file，worker 停止接收新请求并等待 GPU work drain；最长等待约
  660 s，适合长 Prefill/加载。
- Force：直接 terminate child，仅在 drain 卡死时使用。
- Switch：graceful stop 当前 worker，确认退出后再 spawn 新 worker。

stop file 带 stamp，避免旧 profile/旧 worker 的停止信号误伤新进程。

## 日志乱码修复

Windows child stdout/stderr 可能同时包含：

- UTF-8 中文；
- 非法 byte sequence；
- ANSI color；
- OSC title control；
- `\r` progress update。

旧 GUI 直接按不匹配编码显示，出现类似 UUID/agent 字段和乱码。修复后的 normalization：

- 优先保留合法 UTF-8；
- invalid bytes 用 loss-tolerant replacement；
- 去 ANSI/OSC 控制；
- 统一换行/进度行；
- 保留可解析的 colored metrics 文本。

有单测覆盖合法 UTF-8、非法 bytes 和 OSC。

## 实时速度

页面右上角曾一直为 `—`，原因是 worker metric line 在 normalization 后未进入统一解析路径。
现在日志 reader 从 Prefill/Decode 行提取最新 rate，写入 runtime status，Browser polling 同一
status API。速度是最近完成阶段的结果，不是 `nvidia-smi`/GPU utilization。

## 安全边界

- GUI admin key 与 worker OpenAI API key 分离；
- worker key 通过 `INFR_API_KEY` 环境传递，不出现在命令行；
- `gui-data/state.json` 保存 profile，可能包含配置 key，应限制服务器本地 ACL；
- GUI 是 plain HTTP，只面向加密 ZeroTier，不建议暴露公网；
- launcher 不自动修改 Windows Firewall。

## 当前限制与下一步

- GUI 同时只 supervision 一个 worker 和一个 download；
- CLI `infr serve --embedding-model` 已能在一个 native endpoint 挂 Chat + Embedding，但 GUI
  profile 仍以单 task worker 为主要模型，组合配置可继续暴露；
- Vision/mmproj 只 catalog，不进执行 worker；
- Rerank 是 reserved task；
- 重启后不自动接管未知 orphan worker；
- 更完整 dashboard 可加入 cache hit、RAM/SSD demand、arena accounting，而不仅是 tok/s。

---

[Product](README.md) · [Embedding API](embedding-api.md) ·
[Memory budget](../architecture/memory-budget.md)
