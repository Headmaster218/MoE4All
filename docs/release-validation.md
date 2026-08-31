# 发版前长上下文验证手册

本文是 `infr` Windows 命令行版本的固定发版门禁。以后可以直接要求 Codex：

> 按 `docs/release-validation.md` 执行发版验证。小问题最小修复后重跑；涉及架构、
> 推理正确性或资源策略的大问题立即停止并等待讨论。

测试矩阵的实现与参数说明见 [context-resource-matrix.md](context-resource-matrix.md)。

## 验证目标

在真实模型、真实 API 多轮对话和真实动态 KV 增长下，确认以下内容：

- 自动资源探测和手动资源预算都能正常启动。
- Qwen 35B 和 Qwen3.8 Q4 均能运行。
- 模拟 16 GiB VRAM + 32 GiB RAM，以及 24 GiB VRAM + 64 GiB RAM。
- 模拟系统占用分别为 2 GiB VRAM 和 10 GiB RAM。
- API 对话依次跨过 32K、64K、96K 三个动态 KV 增长边界，最终上下文超过 96K。
- 每次 decode 32 token，K/V cache 使用 Q8_0。
- 第二、三轮实际复用已有上下文，而不是完整重新 prefill。
- 进程实际 RAM 和独占 VRAM 始终不超过模拟机器可用上限。
- 额外进行一次 Qwen 35B CLI 三轮对话，验证非 API 入口。

总计 9 个用例：8 个 API 用例和 1 个 CLI 用例。

## 首次准备

1. 从 `tests/context-resource/matrix.example.json` 复制出
   `tests/context-resource/matrix.local.json`。
2. 在 local manifest 中填写 Qwen 35B 和 Qwen3.8 Q4 的本地 GGUF 路径。
3. 确认 Windows GPU 性能计数器可用，并尽量关闭其他大量占用 GPU/RAM 的程序。
4. 不需要手动清理 `INFR_*`。矩阵会为每个子进程移除继承的 `INFR_*`，再传入本次
   用例的明确配置。

local manifest 和测试产物均已加入 `.gitignore`。

## 发版前命令

先完成静态检查、核心测试和 release 编译：

```powershell
cargo fmt --all -- --check
cargo test --locked -p infr-core
cargo build --release --locked -p infr-cli
git diff --check
```

只展开矩阵并检查模型、profile 和可执行文件路径，不加载模型：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File scripts/context-resource-matrix.ps1 -List
```

运行所有尚未通过的用例：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File scripts/context-resource-matrix.ps1
```

已经通过的用例会自动跳过。单独强制重跑某一项：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File scripts/context-resource-matrix.ps1 `
  -CaseId 16vram-32ram-auto-qwen35 -Force
```

## Codex 执行方式

- 启动矩阵后，让测试进程独立运行，使用长时间阻塞等待或低频轮询。
- 不持续 tail 完整日志，不按 token 或秒级反复读取状态，也不因模型加载阶段暂时没有
  输出就重启测试。
- 仅在一个用例完成、发现失败、超过脚本超时或全部结束时读取和分析新输出。
- 等待本身不需要持续推理。模型运行期间不做无关分析或代码修改。
- 不在所需 `exec`/测试进程仍运行时结束任务；发生上下文压缩后继续等待同一个进程，
  不从头启动第二份矩阵。
- 用户中途要求停止时，先安全结束当前测试及其服务进程，再汇报已完成的用例和产物。

## 失败处理规则

每个失败都先保留对应目录中的日志、请求、响应、资源采样和 `result.json`，确认原因后
再修改。不要先删除现场。

以下通常属于可以自主处理的小问题：

- PowerShell 兼容性、参数转义、路径或 JSON 序列化错误。
- 测试脚本的启动、等待、清理、报告或日志解析错误。
- 测试专用 resource profile 接线中的明确局部错误。
- API 返回字段发生兼容性变化，但实际推理、缓存复用和资源行为仍然正确。
- 不影响生产架构和推理语义的编译错误、类型错误或边界检查遗漏。

处理小问题时：

1. 找到根因并做最小修改，不顺手优化生产代码。
2. 重新运行格式检查、相关测试、release 编译和 `git diff --check`。
3. 使用 `-CaseId ... -Force` 重跑失败用例。
4. 如果修改触及共享生产逻辑，使用 `-Force` 重跑完整矩阵；如果只修改测试脚本，确认
   失败用例通过后再运行一次无参数矩阵，让报告汇总全部 9 项。
5. 重复以上流程，直到矩阵全部通过。

遇到以下大问题时立即停止，不自行改变架构，整理证据后等待用户讨论：

- 模型生成内容明显错误、跨轮上下文错误或 Q8 KV 数值正确性异常。
- pager panic、silent corruption 风险、device lost、无法解释的 OOM 或资源越界。
- 动态 KV 的布局、增长、回收或 prefix-cache 复用不符合设计。
- 修复需要改变显存统一管理、pager、缓存策略、KV allocator、paging policy、kernel、
  模型执行图或 API 会话语义。
- 出现明显性能回退，或必须改变性能关键路径才能通过。
- 现有测试暴露了设计层面的预算矛盾，无法通过局部正确性修复解决。

停止时应报告：失败用例 ID、复现阶段、关键错误、峰值 RAM/VRAM、最后一个成功用例、
相关产物路径、初步根因和可选方案。不要在讨论前继续大改。

## 不允许的通过方式

不得用以下方式让门禁变绿：

- 提高模拟 RAM/VRAM 上限或手动预算。
- 缩短上下文、减少动态 KV 边界或减少 32-token decode。
- 将 Q8 KV 改回 F16。
- 关闭 VRAM guard、资源监控或错误检查。
- 忽略 resource violation、panic、device lost、错误输出或缓存未复用。
- 单纯放宽匹配条件、阈值或超时来掩盖真实故障。

确有硬件速度差异时可以合理增加超时，但必须先确认进程仍有进展且没有换页、死锁或
资源越界。

## 产物与通过标准

所有产物位于 `artifacts/context-resource-matrix/`：

- `report.md`：9 个用例的汇总表。
- `<case>/result.json`：单个用例的最终判定和 KV 增长事件。
- `<case>/resource-summary.json`：峰值 RAM/VRAM 与监控状态。
- `<case>/resource-samples.jsonl`：500 ms 资源采样。
- `<case>/server.*.log` 或 `cli.*.log`：完整进程日志。
- `<case>/turn-*-*.json`：API 请求规划与响应。

只有在以下条件同时满足时，才可报告发版验证通过：

- 四条静态/构建命令全部成功。
- `report.md` 中 9 个用例全部为 `pass`。
- 没有资源违规、panic、device lost、OOM 或非预期退出码。API 服务由 shutdown file
  触发与 SIGTERM 相同的安全排空流程，完成 GPU 释放后的 `143` 是预期退出码。
- 三个动态 KV 增长点和 API prefix reuse 均通过检查。
- 没有为了通过测试而修改预算、工作负载或生产语义。

最终汇报应简短列出提交/版本、9 项通过情况、峰值资源摘要、总耗时，以及过程中是否
修复过小问题。若没有全部通过，不得称为发版通过。
