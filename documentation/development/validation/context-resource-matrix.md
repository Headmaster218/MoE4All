---
kind: validation-method
status: current
scope: windows-context-resource-matrix
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 长上下文资源矩阵

这个仅适用于 Windows 的测试工具在较大的主机上复现受限机器，并驱动真实的多轮推理跨越三个动态 KV 边界。它是正确性和容量测试，不是吞吐量基准测试。

## 矩阵

API 矩阵包含八种情形：

| 模拟机器 | 配置 | 模型 |
|---|---|---|
| 16 GiB VRAM，已占用 2 GiB；32 GiB RAM，已占用 10 GiB | 自动和手动（14 GiB VRAM / 22 GiB 进程 RAM） | Qwen 35B 和 Qwen3.8 Q4 |
| 24 GiB VRAM，已占用 2 GiB；64 GiB RAM，已占用 10 GiB | 自动和手动（22 GiB VRAM / 54 GiB 进程 RAM） | Qwen 35B 和 Qwen3.8 Q4 |

额外的一次 Qwen 35B CLI 运行会在 16/32 自动配置下完成三轮普通短对话。API 情形使用一个服务器槽位、Q8 K/V、确定性采样和三个请求：

1. 32K 的提示词，随后解码 32 个 token。解码必须将 KV 扩展至 64K。
2. 64K + 256 的提示词。增量预填充必须将 KV 扩展至 96K。
3. 96K 的提示词，随后解码 32 个 token。解码必须将 KV 扩展至 128K。

因此最终上下文超过 96K。请求携带完整的先前对话，服务器必须在第二、三轮报告较大的 `cached_tokens` 前缀。

## 资源覆盖

`--test-resource-profile PATH` 是隐藏的仅测试 CLI 选项。JSON 文件提供 `vram_total`、`vram_used`、`ram_total` 和 `ram_used`。

- 在自动 RAM 策略运行前，主机内存探测结果会被限制。
- Vulkan 规划会看到限制后的总 VRAM 和可用 VRAM。
- Vulkan 分配保护会从模拟余量中扣除后端分配，即使驱动不支持 `VK_EXT_memory_budget` 也如此。
- 配置只能减少真实容量，不能凭空增加 RAM 或 VRAM。
- 启动时始终输出 `TEST RESOURCE OVERRIDE ACTIVE`。

外部监视器采样进程总工作集和私有工作集、私有字节、Windows 每进程的专用和共享 GPU 内存、系统可用 RAM 以及缺页数。RAM 限制按私有工作集执行：由文件支持的 GGUF 页面仍会出现在总工作集诊断中，但它们在受限机器上可以回收，因此不构成进程 RAM 预算违规。仅当私有常驻 RAM 或专用 GPU 内存超过配置的可用容量时，监视器才会终止已验证的测试进程。

## 设置和使用

将 `tests/context-resource/matrix.example.json` 复制到 `tests/context-resource/matrix.local.json`，并设置两个本地 GGUF 路径。本地清单和所有结果产物均被 gitignore 忽略。

```powershell
# Expand the nine cases without loading a model.
powershell.exe -NoLogo -NoProfile -File scripts/context-resource-matrix.ps1 -List

# Run all unfinished cases.
powershell.exe -NoLogo -NoProfile -File scripts/context-resource-matrix.ps1

# Retry one case, overwriting its previous files.
powershell.exe -NoLogo -NoProfile -File scripts/context-resource-matrix.ps1 `
  -CaseId 16vram-32ram-auto-qwen35 -Force
```

后续调用会跳过成功的情形。每种情形都会在本机忽略的 `benchmark-data/artifacts/context-resource-matrix/` 下保留请求、响应、服务器日志、精确的 KV 增长事件、每 500 ms 的资源采样和结果 JSON。汇总表位于该目录的 `report.md`；对外结论应提炼到 `documentation/evidence/`。

提示词规划器是内部辅助工具：

```text
infr __test-plan-prompt MODEL --messages template.json --target 32768 --output planned.json
```

它仅打开 GGUF 元数据和模型分词器，从不构造 GPU 后端。该辅助工具渲染与 `serve` 相同的聊天模板，恰好替换一个 `{{INFR_FILLER}}` 标记，并寻找不超过目标值的最接近提示词深度。
