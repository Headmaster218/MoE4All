---
kind: reference
status: current
scope: configuration
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 配置 infr

infr 只有一个配置值：启动时从四个层级解析一次，然后显式传递给后端和会话。每个调节项都是 `infr_core::config::Config` 上的类型化字段；没有任何部分会在背后读取环境变量。

带注释的起始配置位于 [`infr.example.toml`](../../infr.example.toml)。

## 优先级

共有四层。**后面的层级优先**，且每层只覆盖它实际指定的字段；在配置文件中省略 `temp` 不会将它重置为默认值，只是将该值交由更低层级决定。

| 层级                  | 来源                                 | 说明                                                |
| ---------------------- | -------------------------------------- | ---------------------------------------------------- |
| 1. 默认值（最低）   | `impl Default for Config`              | 随发行版提供的行为。                               |
| 2. 配置文件         | TOML（见下文）                       | 文件缺失 = 无操作。文件格式错误 = 报错。         |
| 3. 环境变量         | `INFR_*`                               | 名称保持不变，未重命名。          |
| 4. CLI 标志（最高） | `--dev`, `--ctx`, `--temp`, …, `--set` | 专用标志优先于同一字段的 `--set`。 |

有两个值得了解的后果：

- **每个已文档化的 `INFR_*` 变量仍然可用。** 引入 `Config` 的优化专项改变了 _值的去向_，而不是输入内容。现有的 `INFR_PROF_OPS=1 infr bench …` 脚本不受影响。
- **标志优先于继承的环境变量**，因为 CLI 层位于环境变量层之上。`INFR_CTX=32k infr run … --ctx 8192` 会以 8192 运行。

## 例外：`INFR_UNCENSOR_*` 不走 `Config`

去审查投影（Qwen3.8 拒绝方向投影）是这套规则目前唯一的例外：`INFR_UNCENSOR_VECTOR`、`INFR_UNCENSOR_FIRST_LAYER`、`INFR_UNCENSOR_LAST_LAYER` 由 `crates/infr-uncensor` 在自己的模块里解析，**没有** `[uncensor]` 配置段、没有 `--uncensor` 标志、也没有 `--set uncensor.…` 路径。

理由不是省事：这三项决定的是"这次会话要不要存在这个特性"，而不是"运行到一半调节它"。方向文件在模型加载时被读一次、投影缓冲在会话建立时分配或根本不分配，所以一个可以被 `--set`、被热重载、被会话缓存复用的配置字段会承诺一种实际不存在的能力。它登记在 `crates/infr-core/src/config/manifest.rs` 的 `NOT_MIGRATED` 清单里（防漂移测试因此仍然覆盖这三个键）。

真正运行期可改的那一半不在配置层：每次 API 请求的 `uncensor` 字段（Strata 拼写 `experimental_speed_projection` 亦可）。见[去审查集成记录](../evidence/changes/models/qwen38-uncensor.md)。

## 配置文件

### 查找

**第一个存在的**文件生效。不会跨文件合并。

1. `--config <PATH>`：若该路径不存在即为**错误**。
2. `./infr.toml`
3. `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`

完全找不到文件是无操作，绝不会报错。

### 格式

采用 TOML。段路径就是结构体路径，因此键的完整名称正是 `--set` 接受的内容：`[kernels.vulkan] flash_splits = 2` 等同于 `--set kernels.vulkan.flash_splits=2`。

文件使用**肯定式**字段名。环境变量中出现 `INFR_NO_*` 禁用开关的位置，配置中使用被启用的事物，默认值为 `true`：

```toml
[device]
dev = "Vulkan1"      # same grammar as --dev / INFR_DEV
ctx = "32k"          # the shared size grammar: 8192 / 256k / 50%

[kv]
type_k = "q8_0"
type_v = "q8_0"

[paging]
cache = "8g"         # force the paged expert cache with an 8 GiB budget

[kernels.vulkan]
flash_splits = 2
gemm_warp = false    # NOT `no_gemm_warp` — INFR_NO_GEMM_WARP's field, inverted

[multi]
pipeline = [0, 1]    # or ["Vulkan0", "Vulkan1"]

[serve]
max_tokens_cap = 8192
stats_interval_secs = 10  # throughput line every 10 s; 0 turns it off
```

取值语法：布尔值接受 `true`/`false`（`--set` 还接受 `1`/`0`、`yes`/`no`、`on`/`off`）；大小采用共用的 `8192` / `256k` / `50%` 语法；设备列表接受索引数组或 `VulkanN` 字符串；`Option` 字段用 `""` 或 `"none"` 清除。

### 未知键会警告；错误类型会失败

无法识别的键，或整个无法识别的节，都会在 **stderr 上警告并忽略**，同时给出“你是否想要”：

```
[infr] config: unknown key `bogus` (ignored)
[infr] config: unknown key `kernels.vulkan.flash_split` (ignored) — did you mean `kernels.vulkan.flash_splits`?
```

这是有意设计的：为较新 infr 编写的配置文件不得在较旧二进制上硬失败，移除调节项也不得破坏所有在文件中设置它的用户。拼写错误保护来自该警告行。

向**已知**键提供无法解析为其类型的值是硬错误：

```
Error: config `device.ctx`: expected a size/count like 8192, 256k or 50% (got "banana")
```

请注意与环境变量的不对称性，环境变量冻结为当前行为：文件中的 `ctx = "banana"` 会加载失败，而 `INFR_CTX=banana` 会被静默忽略并回退。五个环境变量键例外，会在每个子命令启动时明确拒绝错误值：`INFR_SG`、`INFR_SUBMIT_DISPATCHES`、`INFR_PIPELINE`、`INFR_TENSOR_PARALLEL`、`INFR_EXPERT_PARALLEL`。

### 诊断会自行声明

若**文件**启用 `prof.*` 或 `debug.*` 调节项，infr 会在启动时输出一行，指明文件和字段：

```
[infr] config: /home/you/.config/infr/config.toml enabled diagnostics: prof.ops
```

因此，“为什么我的服务器在输出计时信息”可以直接从命令行回答，即使原因是你忘记了全局配置文件的存在。

## `--set`

大多数配置调节项没有专用标志。`--set <config.path>=<value>` 可访问全部调节项，并使用与 TOML 文件相同的路径语法：

```bash
infr bench "$M" -p 512 -n 0 --set kernels.vulkan.flash_splits=2
infr run "$M" "hi" --set kv.type_k=q8_0 --set kv.type_v=q8_0
```

- 路径是**配置路径**，绝不是 `INFR_*` 名称。它们并非一一对应：`INFR_NO_GEMM_WARP` 对应 `kernels.vulkan.gemm_warp=false`，`INFR_NO_GEMV_REG` 和 `INFR_GEMV_VARIANT` 都对应 `kernels.vulkan.gemv.variant`，而 `INFR_MMV_MW` 是三态。
- **未知**路径会产生带“你是否想要”提示的硬错误（不同于文件层：它是你为本次运行输入的，静默忽略会给出错误答案且没有第二次发现机会）：
  ```
  Error: unknown config path `kernels.vulkan.flash_split` — did you mean `kernels.vulkan.flash_splits`?
  ```
- **同一路径出现两次**是错误，不是静默的后者优先：
  ```
  Error: `--set kv.slots=` given more than once
  ```
- `--set` 与专用标志是**叠加**关系，但优先级低于专用标志。同时传递时会输出指明字段的警告：
  ```
  $ infr run "$M" "hi" --ctx 4096 --set device.ctx=8192
  [infr] config: `--set device.ctx=8192` ignored — the dedicated flag for `device.ctx` wins
  ```

### 专用标志

`--config` 和 `--set` 是全局选项。其余选项位于 `run` / `serve`（设备标志也位于 `bench`）；以 `infr <cmd> --help` 为准。

| 标志                     | 配置路径         | 环境变量             |
| ------------------------ | ------------------- | --------------- |
| `--dev`                  | `device.dev`        | `INFR_DEV`      |
| `--ctx`                  | `device.ctx`        | `INFR_CTX`      |
| `-u` / `--ubatch`        | `device.ubatch`     | `INFR_UBATCH`   |
| `-t` / `--threads`       | `device.threads`    | —               |
| `--temp`                 | `sampling.temp`     | `INFR_TEMP`     |
| `--top-k`                | `sampling.top_k`    | `INFR_TOP_K`    |
| `--top-p`                | `sampling.top_p`    | `INFR_TOP_P`    |
| `--seed`                 | `sampling.seed`     | `INFR_SEED`     |
| `--max-new`              | `sampling.max_new`  | `INFR_MAX_NEW`  |
| `--no-think` / `--think` | `sampling.no_think` | `INFR_NO_THINK` |
| `--reasoning-effort`      | `sampling.reasoning_effort` | — |

`device.threads` 没有 `INFR_*` 对应项；它以 `RAYON_NUM_THREADS` 发布，因为 rayon 的全局线程池没有其他输入。

## 可调节项

权威来源是 [`crates/infr-core/src/config/manifest.rs`](../../crates/infr-core/src/config/manifest.rs)：它在一张表中列出每个 `INFR_*` 键、其对应的配置路径和取值语法，测试会将其与树中的实现进行核对。`Config::all_paths()` 是完整路径列表。下文给出结构以及用户实际会使用的调节项。

**`[device]`** — 使用哪块 GPU（`dev`）、上下文大小（`ctx`）、预填充微批（`ubatch`、`ubatch_parallel`）、CPU `threads`，以及两个低层设备调节项（用于 iGPU 提交拆分器的 `submit_dispatches`，以及强制子组为 16 或 32 的 `subgroup_pref`）。

`device.auto_profile` 可设为 `conservative`（默认）或 `aggressive`。未显式指定 RAM 预算或旧分页覆盖时，启动阶段将保守档的“当前可用 RAM - 3 GiB”、激进档的“总物理 RAM - 14 GiB”写入 `device.ram_budget`，之后按显式总进程预算路径处理。离散 GPU 两档的 Prefill ubatch 都从 4096 行开始，并可因显存放置不足而下调；并发 ubatch 未单独指定时继承最终值，iGPU 使用独立的小批默认值。

`device.vram_budget` / `INFR_VRAM_BUDGET` 限制后端的总设备内存占用：常驻权重、KV、运行时分配以及专家/稠密分页器都计入其中。`device.vram_reserve` / `INFR_VRAM_RESERVE` 还会在 Vulkan 内置的 256 MiB 安全保护之上额外保留相应的物理 VRAM。例如，`INFR_VRAM_BUDGET=23g` 和 `INFR_VRAM_RESERVE=512m` 表示严格的 23 GiB 进程上限加 512 MiB 额外物理余量。两者都未显式指定时，保守自动档在分配器 guard 外再留 768 MiB（合计约 1 GiB），激进档直接使用扣除 guard 后的设备当前可用量。任一显式值出现时走显式约束路径；统一 arena 仍以固定资源落地后的真实余量定容，物理分配失败时会缩小 arena 或下调 Ubatch 后重新探测。

**`[sampling]`** — `temp`（0 = 贪心）、`top_k`、`top_p`、`seed`、`max_new`、`ignore_eos`、`no_think`、`reasoning_effort`、`preserve_thinking`。`reasoning_effort` 和 `preserve_thinking` 没有 `INFR_*` 环境变量别名；省略它们即可保留内嵌聊天模板的默认值。

**来源很重要**：`infr run` / `infr serve` 仅为**没有任何层指定**的调节项，从模型自身的推荐采样值（架构家族表及模型旁的任意 `generation_config.json`）填充 `temp` / `top_k` / `top_p` / `max_new`。在配置文件中填写 `temp` 会将它固定，并禁止该回退。库调用方得到的 `Config` 默认值是贪心：`temp = 0.0`、`top_k = 20`、`top_p = 0.95`、`max_new = 2048`。在 `serve` 中，这些是服务器默认值；单个请求的 OpenAI `temperature`/`top_p` 仍会覆盖它们。

**`[kv]`** — 缓存元素格式（`type_k` / `type_v`，以及旧别名 `force_q8`）、前缀缓存 `slots`、滑动窗口 `ring`，以及在 VRAM 耗尽时将 KV 溢出至系统 RAM 的三项配置（`overflow`、`overflow_vram_mb`、`overflow_reserve_mb`）。两个 `_mb` 路径名及其 `INFR_*_MB` 环境变量别名是冻结的兼容拼写；其数值始终表示 MiB。

设置 `kv.session_cache_dir` 可选择启用 Qwen3.5/3.6/3.8 动态 Q8 Vulkan KV 的磁盘支持空闲会话。空闲槽位会在 `kv.session_idle_secs`（默认 120）后流式写入一个带校验和的文件，并释放其动态 VRAM 段；后续请求的 token 前缀匹配时会恢复该会话，而非重新预填充已保存的前缀。`kv.session_cache_max` 限制根目录下全部 `.infrkv` 文件（默认 `5GiB`，仅接受绝对大小），`kv.session_cache_ttl_hours` 会删除旧文件（默认 24；0 禁用按时间过期）。环境变量别名为 `INFR_KV_SESSION_CACHE_DIR`、`INFR_KV_SESSION_IDLE_SECS`、`INFR_KV_SESSION_CACHE_MAX` 和 `INFR_KV_SESSION_CACHE_TTL_HOURS`。目录未设置或大小上限为零时，此功能关闭。缓存文件包含 token ID 和模型状态，因此目录应为私有目录。`--parallel` 仍表示同时活跃的常驻槽位；冷会话延长的是保留历史，而不是计算并行度。

**`[paging]`** — MoE 专家缓存和稠密层流式加载：`cache` 设置分页 VRAM 预算（即使权重本可容纳也会强制分页），`ring` 覆盖上传暂存环，`stats` 输出每个池的命中/未命中/驱逐次数。`paging.cache` / `INFR_CACHE` 为兼容性保留为专家/稠密内存区覆盖值，但超过统一总内存预算时会被限制。

**`[kernels]`** — 两个与后端无关的图形状门控（`qkv_fuse`、`gated_rmsnorm`）以及每个后端的一个子节。其下的所有项都是内核**层级**覆盖：引擎会选择设备支持的最佳层级，而这些设置用于在二分定位正确性或性能问题时强制禁用某一层级。

- **`[kernels.vulkan]`**（最大节，62 个键）：能力掩码（`coopmat`、`f16`、`i8_dot`、`coopmat_8x8`、`i8_coopmat`）、GEMM/GEMV 层级（`gemm_warp`、`mmq`、`mmv`、`mrow`、`moe_small_m`、`[kernels.vulkan.gemv]` 子表）、注意力（`flash_warp`、`flash_splits`、`flash_min_rows`、`pv_splits`）、DeltaNet（`dn_chunk_scan`、`dn_chunk`、`dn_split`）以及管线开关（`push_desc`、`pipeline_cache_disk`、`no_replay`、`no_vram_guard`、BDA 块上限）。注意：无论 `coopmat` 为何值，`f16 = false` 都会禁用 coopmat，与当前 `INFR_NO_F16` 的效果一致。
- **`[kernels.metal]`** — Apple 后端按 dtype 区分的原生/CMM/RT 内核家族，以及 `deltanet`、`moe` 和 `pipeline_cache`（将编译后的 `MTLComputePipelineState` 作为 `MTLBinaryArchive` 持久化到 `~/.cache/infr`，这样每次启动无需为每个内核重新运行驱动的 AIR → GPU-ISA 后端；MSL → AIR 前端不可缓存，仍会在每次启动时运行）。与 `kernels.cpu.reference` 一样，它没有 `INFR_*` 对应项。
- **`[kernels.cpu]`** — `spin`（自旋池空闲上限）、`spinpool`、`repack_mb` 和 `reference`（位参考内核路径，没有也从未有过 `INFR_*` 对应项）。`repack_mb` 同样是兼容名称；其数值表示 MiB。

**`[spec]`** — MTP / 推测式解码：`mtp`、`k`、`decode_chain`、`draft`（草稿模型路径）以及 GPU 侧采样步骤（`gpu_argmax`、`gpu_sample`、`gpu_embed` 等）。MTP 当前已搁置；参见 README。

**`[multi]`** — 多 GPU 切分：`pipeline`、`tensor_parallel`（两者均需要 ≥ 2 个设备）和 `expert_parallel`（≥ 1）；设备过少是硬错误。三个 `*_p2p` 标志选择 GPU 到 GPU 的传输（默认 `true`），而非经由主机 RAM 暂存。

**`[prof]`** — 每个键均为 `INFR_PROF_*`，且都可通过三种方式设置（环境变量、`--set prof.<name>=…` 或文件中的 `[prof]` 节）：

| 调节项                | 作用                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `ops`               | 所有后端的逐算子设备性能剖析：vulkan、metal、cpu                                                           |
| `op_shapes`         | 按形状逐项列出算子标签，而不是将一个种类折叠在一行中（vulkan）                                         |
| `stages`            | 主机端阶段计时：吞吐量、解码设置与执行、预填充构建/编译/执行、MTP 验证、扩散步骤 |
| `vram`              | 权重加载后记录正在使用的 VRAM                                                                                  |
| `out`               | 同时将退出报告作为 JSON 写入此路径                                                                         |
| `diffusion_trace`   | 扩散模型的逐步调度/熵跟踪                                                                    |
| `metal_device_time` | `off` / `flush` / `counters`：Metal 获取逐算子设备时间的方式                                                     |
| `metal_debug`       | 额外的 Metal 性能剖析器输出                                                                                             |

如果你有旧脚本，有两项整合值得了解。逐算子性能剖析过去在各后端使用不同调节项（vulkan 使用 `INFR_PROF2`、cpu 使用 `INFR_PROF_OPS`、metal 使用 `INFR_METAL_PROFILE`），现在统一为 `ops`。主机端阶段计时过去也有五个调节项，每条管线一个（`INFR_PROF`、`INFR_PROF_DEC`、`INFR_PROF_PF`、`INFR_MTP_TIME`、`INFR_DIFFUSION_TIME`），现在统一为 `stages`。旧拼写已被彻底移除，不再读取。

**`[serve]`** — `api_key`（Bearer 令牌；**空**值表示不鉴权，它限制 `/v1/chat/completions`、`/v1/responses`、`/v1/embeddings` 和 `/v1/models`，但从不限制 `/health`）、`max_tokens_cap`、`request_timeout_secs`（单个请求以秒计的实际时钟期限；默认值 `0` 表示无期限。期限会截断本来合法但较慢的回复，因此为选择启用）以及 `stats_interval_secs`（`INFR_SERVE_STATS_SECS`，控制服务器记录汇总吞吐量和每请求进度的频率；默认 `5`，`0` 关闭周期性行）。`shutdown_file`（`INFR_SHUTDOWN_FILE`）是可选的监督器 IPC 路径：创建该文件会请求与 SIGTERM 相同的优雅排空，包括模型加载期间。`embedding_runner`（`INFR_EMBEDDING_RUNNER`）可选择受管理的 llama-server 可执行文件；否则 INFR 会从自身目录、`PATH` 或 LM Studio 中发现兼容运行器。每请求采样不在此处，仍保留在请求中。

汇总 `serve stats` 行仅在**有活动时**输出：无事件的区间不输出任何内容，因此空闲服务器会留下干净日志，且不会出现可能被误认为负载的心跳。它的 `prefill_tps`/`decode_tps` 是整个服务器的模型 token 数除以该区间的实际时钟时间，而非累计值。每个活跃请求还会在完成预填充块时及解码期间输出受限频率的 `request progress` 行。它报告精确的模型 token `context_tokens/context_limit`、未缓存预填充进度、缓存前缀、已生成 token（包含推理）和请求本地速率。即使禁用周期性行，阶段变化和 `request done` 仍会记录。请求日志**仅包含计数，绝不包含提示词文本**。

成功的非流式响应和流式响应的终止完成帧，均包含权威的 `usage` 及 llama.cpp 兼容的 `timings` 对象。`timings.context_n` 是完整提示词加补全深度，`context_limit` 是 KV 槽位容量，`cached_n` 将复用的提示词 token 与实际评估的 `prompt_n` token 区分开来。流式指标无条件输出，因此未发送 `stream_options.include_usage` 的客户端不会静默记录零 token。

`POST /v1/responses` 提供无状态 Responses API，复用相同的模型调度与请求上限；支持文本消息、重放的 function call/output、自定义 function tools，以及 `stream=true` 的 typed SSE。启用视觉模型及 projector 时，`input_image` 可用 base64 `data:` URI；不抓取远程 HTTP(S) URL 或 file ID。多轮请求由客户端在 `input` 中重放此前项目，服务端不保存对话状态。`previous_response_id`、`conversation`、`store=true`、文件输入、托管工具、结构化输出模式和自动截断均返回 400。

Responses API 只接受 plain text 与默认 `text.verbosity="medium"`；`low`/`high` 不支持。`reasoning.effort="minimal"` 在模板支持时近似映射为本地 `low`。本地 backend 没有独立的 hidden reasoning token 计量，因此 `reasoning_tokens` 为 0，外显思考内容计入 `output_tokens`。流式响应使用 `Cache-Control: no-store` 和 `X-Accel-Buffering: no`；生成在响应创建后失败时发出 typed `response.failed`。这些 HTTP 缓存头不会关闭模型 KV prefix cache。

**`[hub]`** — 模型获取（`infr pull`，以及模型缺失时 `infr run` / `infr serve` 执行的自动拉取）。`endpoint`（`INFR_HF_ENDPOINT`）选择 HuggingFace 兼容源：默认 `https://huggingface.co`，也可以是 `https://hf-mirror.com`。`pull_jobs`（`INFR_PULL_JOBS`，默认 `8`）表示单个模型下载可同时使用的连接数。慢的是连接而非链路：针对相同 CDN 对象测量，一个连接持续为 8.39 MiB/s，五个连接持续为 75.1 MiB/s，即每个连接 15.0 MiB/s，因此每连接速率不降反升。

连接用于何处取决于模型，设置本身无需说明：分片模型是 `-NNNNN-of-MMMMM` 分片，每条连接获取一个文件；单文件发布的模型则被切分为字节范围后重组（`unsloth/DeepSeek-V3.2-GGUF` 的单个 150 GiB `UD-TQ1_0` 在同一 60 秒窗口中，`1` 时为 10.9 MiB/s，默认 `8` 时为 76.8 MiB/s；整次 150 GiB 拉取端到端为 76.3 MiB/s）。一个数字覆盖两种情况，因为按轴分别限制会相乘：在文件数量由发布者决定的仓库中，8 个文件 × 8 个范围就是 64 个套接字（`DeepSeek-V3.2-REAP` 发布了 236 个文件）。

范围下载需要服务器支持；不支持的服务器（没有 `Accept-Ranges`，或请求 `206` 却返回 `200`）会回退为单一流，64 MiB 或更小的文件绝不切分。`0` 和 `1` 都严格表示一条连接，适用于计费链路，或反对来自同一客户端多连接的代理。

**`[debug]`** — 填毒/屏障/转储开关：`coopmat`（输出枚举到和选定的 coopmat 形状，对 Intel Arc 很有用）、`bda_chunk`、`wide_dispatch`、`chat`、`moe_counts`、`moe_counts_dump`、`poison_uninit`、`no_barrier`、`full_barrier`。

## 有意不属于配置的 `INFR_*` 键

有四个键继续直接读取环境变量，原因是运行时 `Config` 无法解决：

| 键                        | 原因                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `INFR_PROFILE`             | **构建时**输入：由 core/cpu/gguf/llama/vulkan 中的 `build.rs` 读取以设置 `cfg(infr_profile)`。读取时不存在运行时值。 |
| `INFR_TEST_GGUF`           | 测试固件：将 `infr-gguf` 的测试指向磁盘上的 `.gguf`。                                                                              |
| `INFR_TEST_MODEL`          | 测试固件：覆盖由模型支持的测试的 HF 缓存查找。                                                                     |
| `INFR_LLAMA_DIFFUSION_CLI` | 开发固件：将 `infr compare` 指向磁盘上的 `llama-diffusion-cli` 二进制文件。                                                               |

有两点相关说明。`INFR_DIFFUSION_VISUAL` 也不是 `Config` 字段；由于它只控制 CLI 呈现，已变为普通标志 `infr run --diffusion-visual`，其 clap `env =` 回退保持旧拼写可用。`INFR_CPU` / `INFR_METAL` 已**废弃**：它们已作为后端选择器移除，且没有别名。请使用 `--dev cpu` / `--dev metal`，或 `INFR_DEV=cpu` / `INFR_DEV=metal`。

## 库调用方

`Config` 是一个值，而非全局对象。请构建一个实例并传递它：

```rust
use infr_core::config::{Config, ConfigOverrides};
use std::sync::Arc;

// The full four-layer resolve (file + env + the CLI overrides you pass).
let cfg = Arc::new(Config::load(&ConfigOverrides::default())?);

// Or construct exactly what you want — no environment, no file, no ordering hazard.
let cfg = Arc::new(Config { kv: infr_core::config::KvCfg { slots: 8, ..Default::default() },
                            ..Default::default() });
```

后端在构造时接收它（`VulkanBackend::new_with(cfg)`、`Backend::new_with(device_id, cfg)`、`CpuBackend::new_with(cfg)`），`Config::load_from_env()` 则是为希望遵循环境变量但不使用配置文件的调用方提供的“默认值加环境变量”归并。

已退役的迁移优化专项包含层级机制、每个调节项的极性表和逐片记录。它已在提交 `3010e452` 中移除；需要该实现记录时请查阅 Git 历史。
