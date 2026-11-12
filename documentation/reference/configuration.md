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

## 去审查投影：`[uncensor]`

`[uncensor]` 是一张普通配置表，三个键都走同一套层序（TOML → 环境变量 → `--set`）：`vector` 指向 llama.cpp 兼容的 `controlvector` GGUF（每层一条 `[n_embd]` 的 f32 方向，命名 `direction.1` …），`first_layer` / `last_layer` 是**闭区间**层范围，默认 4 与 44，上限按模型实际层数裁剪。环境变量别名：`INFR_UNCENSOR_VECTOR`、`INFR_UNCENSOR_FIRST_LAYER`、`INFR_UNCENSOR_LAST_LAYER`。

**默认关闭**：`vector` 为空或未设置时，这个特性对本会话根本不存在——不分配缓冲、计算图里没有投影算子、生成结果与未引入该特性时逐字节一致。而一旦指名了文件就是要求投影：文件读不到、层范围与模型不符、或模型不是 **Qwen3.8-Flash-Next（`general.architecture = qwen4exp`）**，启动直接**报错退出**，不会悄悄按没投影跑起来。目前只有 qwen4exp 的多流宽残差有这项操作的语义，单流模型不适用。

风险与同意：这里去掉的是模型训练出来的拒绝行为，服务器之后回答什么由操作者承担——安全、合规与法律风险**自行承担**，模型许可证的使用条款依旧适用。启动向导会先显示这段说明再询问，提示符默认值与 30 秒超时**都是"否"**，记住的上次选择不会替这次作答；**开启即视为同意**向导在同意后按**钉死的 sha256** 取方向文件（先 `huggingface.co`，失败回退 `hf-mirror.com`，校验不过即删除），存到 `models/control-vectors/`（`models/` 不入库）：唯一认可的来源是 `cjxzdzh/Qwen3.8-Flash-Next-refusal-direction-qwen4exp-cvec` 的 `cvec-qwen4exp.gguf`；hub 上另有同名的 `Huihui-…-refusal-direction-r.gguf` 是**数值不同的另一版方向**，所以只认哈希不认文件名。

运行期可改的那一半在 API：每次请求的 `uncensor` 字段（Strata 拼写 `experimental_speed_projection` 同样接受，两者取值冲突返回 400）。字段缺省的含义是"沿用本会话当前设置"；开关一旦翻转，该槽位已物化的 KV 前缀会被作废并重读提示词——缓存是在另一种设置下算出来的，留着等于用别的模型续写。见[去审查集成记录](../evidence/changes/models/qwen38-uncensor.md)。

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

[vision]
image_url_allow_private = true  # let an image_url fetch a NAS / LAN host; off by default
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

### Pager 传输诊断 / Pager Transfer Diagnostics

两个开关默认均为 `true`，相互独立，不改变 RAM 专家缓存预算或 GPU 的 LRU 所有权。
Both switches default to `true` and are independent; neither changes the RAM cache budget or GPU LRU ownership.

| 配置 / Setting | 控制内容 / Controls |
| --- | --- |
| `paging.host_dma` | 导入 RAM 专家缓存供 Vulkan 访问；关闭后无法通过该导入路径由 GPU 自取专家，改走 CPU 协助的直接或 staging 上传。Import RAM expert-cache buffers for Vulkan access; disabling it removes this GPU self-copy source and retains CPU-assisted direct/staged uploads. |
| `paging.dedicated_transfer` | 可用时使用独立传输队列；关闭后，需提交的复制走主队列或既有直接上传路径，不关闭 GPU 的 CU 自复制。Use a dedicated transfer queue when available; disabling it retains main-queue/direct fallbacks and does not disable shader-based GPU self-copy. |

保持其他参数、模型和输入相同，分别追加以下配置进行对比：
Keep the model, input and other parameters identical and compare these configurations:

```text
--set paging.host_dma=true  --set paging.dedicated_transfer=true
--set paging.host_dma=true  --set paging.dedicated_transfer=false
--set paging.host_dma=false --set paging.dedicated_transfer=true
--set paging.host_dma=false --set paging.dedicated_transfer=false
```

环境变量 `INFR_NO_HOST_DMA=1` 和 `INFR_NO_DEDICATED_TRANSFER=1` 分别关闭对应功能；CLI `--set` 优先。
Environment variables `INFR_NO_HOST_DMA=1` and `INFR_NO_DEDICATED_TRANSFER=1` disable the respective features; CLI `--set` takes precedence.
启动日志 `pager transfer policy` 显示请求值和实际是否创建独立队列；请求开启不代表设备一定支持。RAM 导入实际覆盖量仍查看 `host DMA import total` 日志。
The startup `pager transfer policy` log reports requested settings and whether a dedicated queue was created; requested does not mean supported. Check `host DMA import total` for actual imported RAM coverage.

### 通用用法 / General Usage

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

`device.auto_profile` 可设为 `conservative`（默认）或 `aggressive`。未显式指定 RAM 预算或旧分页覆盖时，启动阶段按物理内存冻结一条 `device.ram_budget`：保守档 `当前可用 RAM - 3 GiB`、激进档 `总物理 RAM - 13 GiB`，之后按显式总进程预算路径处理。**commit 轴自 2026-11-03 起不再参与定容**（见 `device.mem_guard_headroom`：系统自动管理的页面文件会让上限在加载中途移动，同一份二进制实测 104.6 GB → 116.8 GB，拿它定容既少装专家又抓不住洪水），启动日志仍把当时的 commit 读数打出来（`available_commit_bytes`），store 超过它时打 WARN 说明这是赌页面文件会长大。这条信封是**主机承诺的总额**，专家 store 吃满它、**不预扣帧池**：帧池的真实成本要等冷链下沉才知道，按声明额（比如 16 GiB）先扣等于把能常驻的专家白白饿掉——实测 Qwen3.8-Flash-Next 4.27bpw 的 payload 47.17 GiB 曾因 4 GiB 预扣被压成 bounded 43.17 GiB，运行时 SSD 层为这 4 GiB 一直开着。分工改为：会话建立分页树时，帧池按**当时的空闲物理内存**自己夹紧（`HOST_FRAME_RAM_FLOOR` = 1 GiB 不许花），夹不住的冷前缀逐出而不下沉；故意不按 commit 夹，否则这个tiers会在最坏的时刻（store 刚吃完）被一个瞬时读数清零。显式写了 `device.ram_budget` 或 `paging.dram*` 时逐字兑现，引擎不替你缩。离散 GPU 两档的 Prefill ubatch 都从 4096 行开始，并可因显存放置不足而下调；并发 ubatch 未单独指定时继承最终值，iGPU 使用独立的小批默认值。

`device.vram_budget` / `INFR_VRAM_BUDGET` 限制后端的总设备内存占用：常驻权重、KV、运行时分配以及专家/稠密分页器都计入其中。`device.vram_reserve` / `INFR_VRAM_RESERVE` 还会在 Vulkan 内置的 256 MiB 安全保护之上额外保留相应的物理 VRAM。例如，`INFR_VRAM_BUDGET=23g` 和 `INFR_VRAM_RESERVE=512m` 表示严格的 23 GiB 进程上限加 512 MiB 额外物理余量。两者都未显式指定时，保守自动档在分配器 guard 外再留 768 MiB（合计约 1 GiB），激进档直接使用扣除 guard 后的设备当前可用量。任一显式值出现时走显式约束路径；统一 arena 仍以固定资源落地后的真实余量定容，物理分配失败时会缩小 arena 或下调 Ubatch 后重新探测。

`device.mem_guard_headroom` / `device.mem_guard_ram_headroom`（`INFR_MEM_GUARD_*`）是 commit 熔断的两条水位线：命中时进程以退出码 87 主动结束，而不是滑进换页风暴或显示驱动 AV。commit 轴的判据是**超出已授权主机预算的增量**：`device.ram_budget`（未显式给出时即启动阶段冻结的自动预算）之内的消耗属于计划，不算异常，水位线最低收到 512 MiB 硬底。这条标定来自实测——96 GiB 内存 + 6 GiB 页面文件（commit 上限 102 GiB）的机器上，47.17 GiB 的 MoE 主机 store 加 31.35 GiB host-DMA 导入与 WDDM 设备账之后，剩余 commit 只有约 4.06 GiB，旧的 8 GiB 绝对线会把两次合法启动直接打死。物理-RAM 轴（默认 4 GiB）仍是抓失控泄漏的主探针；做 miss-storm 测试时应把它拉高（例如 `50%`），让熔断在 RAM 耗尽之前落地。任一轴设为 `0` 即关闭该轴。**commit 轴现在默认不装**（`device.mem_guard_headroom` 缺省为空）：它和系统自动管理的页面文件不兼容——上限是 OS 在进程加载过程中会动的数，同一份二进制同一串参数，13:00 那次上限 104.6 GB（store 撑到 71.35 GiB 时熔断在 529 MiB），13:35 那次上限已经变到 116.8 GB（同尺寸跑了整整一轮服务，剩 0.1 GB 也没死）。拿一个会动的天花板去定容又去掐表，既白白少装 35% 的常驻专家，也照样抓不住真正的洪水。要这条线的场景（页面文件固定在上限、或机器与他人共用）显式写一个值，主 store 定容会跟着同一条线收（`seam::moe_host_backing_for_device`）；物理-RAM 轴不受页面文件策略影响，默认仍装着（4 GiB）。**授权额度本身现在是 commit 感知的**：自动标定的 `device.ram_budget` 在 Windows 上取物理轴与 `commit 余量 - 20/10 GiB` 的较小者（见 `device.auto_profile`），因此这条水位线不会再被一个兑现不了的数字放到地板；固定 Vulkan 分配之后引擎还会用实时 commit 快照再夹一次自动 store，帧池不在 store 里预扣，而是在建树时按空闲物理内存夹（`MoE host store sized from physical RAM; Windows commit headroom reported for the record`）。只有显式预算能越过这个上限——它按操作者意志逐字兑现，并打出 `an explicit budget is honored verbatim, so the memory circuit breaker may end this process`。

**`[sampling]`** — `temp`（0 = 贪心）、`top_k`、`top_p`、`seed`、`max_new`、`ignore_eos`、`no_think`、`reasoning_effort`、`preserve_thinking`。`reasoning_effort` 和 `preserve_thinking` 没有 `INFR_*` 环境变量别名；省略它们即可保留内嵌聊天模板的默认值。

**来源很重要**：`infr run` / `infr serve` 仅为**没有任何层指定**的调节项，从模型自身的推荐采样值（架构家族表及模型旁的任意 `generation_config.json`）填充 `temp` / `top_k` / `top_p` / `max_new`。在配置文件中填写 `temp` 会将它固定，并禁止该回退。库调用方得到的 `Config` 默认值是贪心：`temp = 0.0`、`top_k = 20`、`top_p = 0.95`、`max_new = 2048`。在 `serve` 中，这些是服务器默认值；单个请求的 OpenAI `temperature`/`top_p` 仍会覆盖它们。

**`[kv]`** — 缓存元素格式（`type_k` / `type_v`，以及旧别名 `force_q8`）、前缀缓存 `slots`、滑动窗口 `ring`，以及在 VRAM 耗尽时将 KV 溢出至系统 RAM 的三项配置（`overflow`、`overflow_vram_mb`、`overflow_reserve_mb`）。两个 `_mb` 路径名及其 `INFR_*_MB` 环境变量别名是冻结的兼容拼写；其数值始终表示 MiB。

设置 `kv.session_cache_dir` 可选择启用 Qwen3.5/3.6/3.8 动态 Q8 Vulkan KV 的磁盘支持空闲会话。空闲槽位会在 `kv.session_idle_secs`（默认 120）后流式写入一个带校验和的文件，并释放其动态 VRAM 段；后续请求的 token 前缀匹配时会恢复该会话，而非重新预填充已保存的前缀。`kv.session_cache_max` 限制根目录下全部 `.infrkv` 文件（默认 `5GiB`，仅接受绝对大小），`kv.session_cache_ttl_hours` 会删除旧文件（默认 24；0 禁用按时间过期）。环境变量别名为 `INFR_KV_SESSION_CACHE_DIR`、`INFR_KV_SESSION_IDLE_SECS`、`INFR_KV_SESSION_CACHE_MAX` 和 `INFR_KV_SESSION_CACHE_TTL_HOURS`。目录未设置或大小上限为零时，此功能关闭。缓存文件包含 token ID 和模型状态，因此目录应为私有目录。`--parallel` 仍表示同时活跃的常驻槽位；冷会话延长的是保留历史，而不是计算并行度。

**分页 KV 前缀树**（`kv.paged*`）把每个会话的 KV 切成固定大小的**帧**（一帧 = 一页 token × 该模型的全部 plane），命中时只改写设备页表、不搬任何字节。`kv.paged_page_tokens`（默认 32768，即历史的 32K 增长步长，因此默认几何与旧路径逐字节一致）需要设成更小的 2 的幂（如 64）帧才谈得上共享与 DMA。`kv.paged_cache_max` 打开这棵树并额外给它一个 host-RAM 帧池预算：显存走廊放不下的冷前缀按**帧**下沉到 RAM、命中再换回，host 内存按实际下沉量惰性增长、不预占。下沉的单位是**冷尾后缀**而不是整条链：一次命中真正复用到的那段前缀留在显存，只把它后面的尾巴搬走，所以链永远是「显存→RAM→磁盘」的单调梯度（换回时遇到第一个 host 帧就截断复用，前面插一个 host 帧等于把后面所有页白白丢掉）。RAM 装不下整条链时也不再整条跳过——能装多少沉多少，剩下的留在显存继续可寻址。**惰性不等于免费，但也不预扣**：帧的 commit 成本在下沉那一刻才发生，所以自动预算不会替它从专家 store 里扣额度（扣过的版本把 4.27bpw 的 Full store 压成 bounded，SSD 层白开）；真实约束落在建树时——帧池按当时的空闲物理内存夹紧帧数、留 1 GiB 不花，超出声明额的冷前缀逐出而不下沉，`paged KV RAM tier clamped` 会把夹了多少写进日志。`kv.paged`（`INFR_KV_PAGED=1`）单独打开树而**不**分配 RAM 帧池——帧全部住在显存走廊内，冷前缀直接逐出而不下沉；混合递归模型（DeltaNet/KDA/PLE）续接所需的状态快照仍需要一个小的主机内存上限，此时取固定的 512 MiB（快照按实际发布逐个分配，一个 Qwen3.8 边界快照约 113 MiB）。`kv.disk_cache_max` / `kv.disk_cache_path` 在 RAM 帧池之后再挂一层稀疏文件帧层，仅在 RAM 帧池存在时生效。环境变量别名：`INFR_KV_PAGED`、`INFR_KV_PAGED_ELASTIC`、`INFR_KV_PAGED_YIELD_UNDER_PRESSURE`、`INFR_KV_PAGED_PAGE_TOKENS`、`INFR_KV_PAGED_CACHE_MAX`、`INFR_KV_DISK_CACHE_MAX`、`INFR_KV_DISK_CACHE_PATH`。它与 `kv.ram_cache_max`（RAM 基数树）互斥，并要求动态分段 KV（Qwen3.5/3.6/3.8 + Q8 KV + `kv.dynamic`）。分段 KV 的物理段取自**统一显存 arena**，而 arena 只在 MoE 专家被分页时才建立：全部专家常驻显存的会话（常见于显存/统一内存充裕的设备，例如核显）没有 arena，此时启动会拒绝并逐条列出未满足的条件——需要这棵树就用 `paging.cache`（`INFR_CACHE`）预留专家缓存以建立 arena，否则关闭 `kv.paged*`。**带图会话同样进这棵树**，但树收到的不是原始 token 而是**内容键**：每个图片 span 的 pad 段被替换成由该图指纹与网格派生、带 `1<<31` 保留位的 id 序列，长度逐 token 对应。于是两张不同的图（同一张图的不同分辨率也算）永远无法认领彼此的 KV，而 span 之前的文本前缀照常与纯文本会话互相共享；反过来，一张图的 KV 不再随槽位被其它请求占用而消失——实测 Qwen3.8-Flash-Next、`kv.paged=true`、6.4K token 带图轮次：背靠背 2.1s，中间插入一次无关文本请求后旧行为是 10.0s（整张图重算），现在 1.6s（`cached_tokens=4096`，贪心解码的答案与冷跑逐字一致）。

**弹性显存走廊**（`kv.paged_elastic`，`INFR_KV_PAGED_ELASTIC=1`，默认关）让分页树的显存帧走廊按需增长、并把整段空闲的走廊归还专家 pager，而不是开机就租下「会话数 × ceil(ctx / 每帧 token)」并终身不退。开启后建树只提交一个 chunk（预算的 1/8，至少 128 帧），自由帧耗尽时再从统一 arena 认领下一个 chunk，**上限仍是同一个最坏情况**——arena 的 KV 保留区本来就按它 sizing，所以没有任何其它显存预算会移动。增长发生在 prefill 中，会先退掉 Prefill 对走廊未提交单元的临时借用（与分段 KV 增长同一条串行边界）；归还在请求边界发生，且**先退役帧池槽位、再解除 backend 租约**，因此归还失败只会泄漏 arena 字节（可恢复），绝不会让活帧指向已经交给专家的显存。只有整段全空才归还，所以一条被树 entry 钉住的链不会让它自己的走廊消失；配合 RAM 帧池时冷链先下沉、腾空的段再归还，命中时换回并重新增长。实测同一 release 二进制、同窗口、Qwen3.6-35B-A3B、ctx 131072、`-u 2048`、Q8_0 KV、贪心：4345 token 的请求在固定走廊下租 1.33GiB 却只用 69/2048 帧（0.04GiB），解码 140.4 t/s；弹性走廊只租 0.17GiB，解码回到 145.9 t/s（+3.9%，无树基线 165.1）。30685 token 的请求增长一次到 512 帧（0.33GiB），125445 token 的请求连续增长 7 次到 2048 帧上限、与固定走廊落到完全相同的 1962 帧，这两档解码持平（128.6 / 82.5 t/s），冷启 TTFT 只多 0.3%。三档的 `cached_tokens`（4288 / 30656 / 125440）与生成文本 sha 都与固定走廊逐字节一致，0 条验证错误。ctx 32768 的压力序列另测了完整往返：30K 链下沉 RAM（`sunk=481`）后走廊 512→128 帧、`corridor_shrinks=3`，回访把它换回（`promoted=479`）并重新增长，TTFT 287ms 对冷启 13466ms。日志的 `paged KV prefix cache status` 行带 `corridor_grows` / `corridor_shrinks` / `corridor_ceiling` 三个计数器。注意它只收回**没人用的租约**：paged 路径剩下的逐 token 开销与走廊大小无关（4K 档专家驻留同为 218.6MiB、KV 足迹更小，`decode_ms` 仍是 775→879），那部分要靠帧级冷热分层而不是弹性走廊。

**帧级冷热分层**（随 `kv.paged_cache_max` 一起生效，没有单独开关）把下沉从「整条链」改成「冷尾后缀」。最要紧的一档是链比 RAM 帧池还长：旧路径的整链容量检查直接跳过它，接着被逐出，回访等于全量重算。实测同一 release 二进制、同窗口、Qwen3.6-35B-A3B、ctx 131072、`-u 2048`、page 64、Q8_0 KV、贪心、`kv.paged_cache_max=1GiB`（1542 帧）：125450 token 的请求占 1960 帧，插入一个 13150 token 的无关请求制造显存压力后回访——旧路径 `sunk=0 evictions=1`、`cached_tokens=0`、TTFT 94369ms；新路径把 1542 帧尾巴沉进 RAM、`evictions=0`、`cached_tokens=125440`、TTFT 453ms（208×），生成文本 sha 逐字节一致，0 条验证错误。代价落在制造压力那个请求上（6420→6875ms，多付一次下沉 DMA），冷启两档持平（95395/95910ms）。日志的 `paged KV prefix cache status` 行新增 `tail_sinks`：只统计**真的留住了热前缀**的那些下沉。热前缀地板按「上一次真正 attach 的深度」算，不是 token 匹配深度——递归模型只能在快照边界续接，匹配得再深、没有快照就是 0，按匹配深度留前缀等于钉住一段没人读得到的显存。也因此在单请求/多轮分支这两种实测形态里 `tail_sinks` 都是 0：分支处没有可续接的快照，压根没 attach，地板还没轮到上场。

**专家压力驱动的冷尾让渡**（`kv.paged_yield_under_pressure`，`INFR_KV_PAGED_YIELD_UNDER_PRESSURE=1`，默认关）是上面两条的跨 owner 版本：专家缓存和分页 KV 子分配**同一个**统一 arena，但两边各自只看自己的上限——树没到自己的帧上限就把冷页全留在显存里，专家那边每次准入都在驱逐、RAM 帧池却空着。开启后在请求边界（`paged_publish_detach` 里、`release_idle_corridors` 之前）采一次专家未命中率，**超过 3%** 才让树把冷会话尾巴下沉到 RAM 帧池，腾空的整段走廊在同一个边界归还专家。窗口是「距上次采样以来」而不是生命周期速率，否则一小时前热过身的模型会让树永远让渡。上次真正 attach 的复用深度仍是硬底线，所以每轮整段复用前缀的会话一帧也不会让；下沉量受 RAM 自由帧夹紧，且**只沉该会话落在目标段里的那些帧**——`sink_entry_tail` 沉的是整段 VRAM run，host 层宽裕时不设这条地板就会把永不归还的 base chunk 也搬出去，白付一次 DMA 出去再回来。这条地板必须换算成会话自己的 chain 下标：段首帧是**池里的全局槽号**，而 `keep` 索引的是**该会话自己的链**，两者只在「独占走廊、从槽 0 开始」的单会话里恰好相等；直接拿全局槽号当本地下标比，会让所有比目标段起点短的会话 `keep >= end` 全部跳过，多会话场景下整条让渡静默空转。需要 `kv.paged_elastic` + RAM 帧池（`kv.paged_cache_max`），缺一不生效；向导只在两者都开时才问这一项。实测同一 release 二进制（sha `e6249000…`）、同窗口背靠背两臂、Qwen3.6-35B-A3B、ctx 65536、`-u 2048`、page 64、Q8_0 KV、贪心、`--parallel 1`、弹性走廊 + 16GiB RAM 帧池，**只改这一个开关**：出货默认的 `expert_cache_target=13.60GiB` 下窗口未命中率是 7.35%→2.64%→0%，专家热完就再也不到 3% 门槛，这一项**正确地不动作**；所以用 `paging.cache=4GiB` 造压力（专家未命中率稳定 ~16%）。负载是 5 条 14545 token 的冷会话 + 2 条 14578 token 的新会话各生成 1024 token + 1 条逐字回访。开启臂让渡触发 7 次、`corridor_shrinks=13`，树最终只占 227/256 帧（0.15GiB），关闭臂占 940/1024 帧（0.61GiB）——**0.46GiB 走廊真的回到了专家手里**：专家驻留槽 5359/3825 对 4945/3550（+8.4%/+7.7%），驱逐 166722/36533 对 189197/41071（−11.9%/−11.0%）。两条长解码 110.2/109.0 t/s 对 106.0/104.9（**+4.0%/+3.9%**），`decode_ms` 9291/9398 对 9661/9763，这两条请求的 `total_ms` 17416/17628 对 17781/17915（−2.1%/−1.6%），整个序列 76602 对 77147ms（−0.7%）。两臂 `cached_tokens` 都是 14528（99.9%）、生成文本 sha 逐字节一致、`evictions=0 aux_refusals=0`、0 条验证错误。**收益只在解码腿上**：同一套压力换成 24 token 的短生成（prefill 占 97%），序列总和反而 +0.55%，因为回访那次 DMA 换回成了唯一可见的差量——所以这一项适合「专家缓存被压小 + 生成长」的场景，不是给短问答提速的。

**这个信号本身有个坑**：`kernels.vulkan.decode_gpu_resident` 默认开，MTP 关闭时 GPU 拥有 Decode 驻留，host LRU 计数器只记录「交回 host 的那部分」——而 owner 的策略是整层全命中就在设备上继续、遇到第一个 miss 才暂停交回，所以 host-only 的未命中率**系统性偏高**，拿它当压力会在专家其实很舒服的时候白沉冷尾。因此这里走已有的 `owned_statistics` 排空边界读回（和 `paging.stats` 退出时同一条传输），在 `free_kv_frames` 同一把统一排他闸内取；读回失败就报「无压力」并保留驻留权威，绝不猜。代价是**每个请求边界一次设备 LRU 读回**，只有开启这一项时才付（默认关的配置一次也不采）。

**`[paging]`** — MoE 专家缓存和稠密层流式加载：`cache` 设置分页 VRAM 预算（即使权重本可容纳也会强制分页），`ring` 覆盖上传暂存环，`stats` 输出每个池的命中/未命中/驱逐次数。`paging.cache` / `INFR_CACHE` 为兼容性保留为专家/稠密内存区覆盖值，但超过统一总内存预算时会被限制。

**`[kernels]`** — 两个与后端无关的图形状门控（`qkv_fuse`、`gated_rmsnorm`）以及每个后端的一个子节。其下的所有项都是内核**层级**覆盖：引擎会选择设备支持的最佳层级，而这些设置用于在二分定位正确性或性能问题时强制禁用某一层级。

- **`[kernels.vulkan]`**（最大节，67 个键）：能力掩码（`coopmat`、`f16`、`i8_dot`、`coopmat_8x8`、`i8_coopmat`）、GEMM/GEMV 层级（`gemm_warp`、`mmq`、`mmv`、`mrow`、`moe_small_m`、`[kernels.vulkan.gemv]` 子表）、注意力（`flash_warp`、`flash_splits`、`flash_min_rows`、`pv_splits`、`amdvlk_hd256_flash`）、DeltaNet（`dn_chunk_scan`、`dn_chunk`、`dn_split`）以及管线开关（`push_desc`、`pipeline_cache_disk`、`no_replay`、`no_vram_guard`、BDA 块上限）。注意：无论 `coopmat` 为何值，`f16 = false` 都会禁用 coopmat，与当前 `INFR_NO_F16` 的效果一致。`amdvlk_hd256_flash` 默认开启：AMD 专有驱动（amdvlk）上，hd256 非 register-O 几何走 bm16 融合 flash。早先的 LLPC 崩溃来自全展开的 IR，因此该内核同时构建展开版与 `-DNO_UNROLL` 版（`attn_flash_warp_hd256_bm16_nounroll`）并按驱动选用，26.10.1 上已验收。设为 `false` 时 hd256 prefill 退回 coopmat `nonfa` 层级（供仍会故障的驱动使用）。
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

**`[serve]`** — `api_key`（Bearer 令牌；**空**值表示不鉴权，它限制 `/v1/chat/completions`、`/v1/responses`、`/v1/embeddings` 和 `/v1/models`，但从不限制 `/health`）、`max_tokens_cap`、`request_timeout_secs`（单个请求以秒计的实际时钟期限；默认值 `0` 表示无期限。期限会截断本来合法但较慢的回复，因此为选择启用）以及 `stats_interval_secs`（`INFR_SERVE_STATS_SECS`，控制服务器记录汇总吞吐量和每请求进度的频率；默认 `5`，`0` 关闭周期性行）。`shutdown_file`（`INFR_SHUTDOWN_FILE`）是可选的监督器 IPC 路径：创建该文件会请求与 SIGTERM 相同的优雅排空，包括模型加载期间。`embedding_runner`（`INFR_EMBEDDING_RUNNER`）可选择受管理的 llama-server 可执行文件；否则 INFR 会从自身目录、`PATH` 或 LM Studio 中发现兼容运行器。`response_store_entries`（`INFR_RESPONSE_STORE_ENTRIES`）是 `/v1/responses` 服务端状态的 LRU 上限；默认 `0` 表示完全关闭。`response_store_path`（`INFR_RESPONSE_STORE_PATH`）把这份状态镜像成一个 JSON 文件，使 `previous_response_id` 链在重启后仍然有效；不设（默认）则只留在进程内存里，见下。每请求采样不在此处，仍保留在请求中。

`serve` 还会在启动时报告锁状态：与 `run`/`serve-embedding`/`multi`/`bench` 一样，它先取得 GPU 启动锁（`device.gpu_lock`，见 `[device]`）再触碰设备，因此日志里的 `INFR-E1001` 意味着另一个 `infr` 进程仍在这块卡上运行，而不是显存不足的崩溃。

汇总 `serve stats` 行仅在**有活动时**输出：无事件的区间不输出任何内容，因此空闲服务器会留下干净日志，且不会出现可能被误认为负载的心跳。它的 `prefill_tps`/`decode_tps` 是整个服务器的模型 token 数除以该区间的实际时钟时间，而非累计值。每个活跃请求还会在完成预填充块时及解码期间输出受限频率的 `request progress` 行。它报告精确的模型 token `context_tokens/context_limit`、未缓存预填充进度、缓存前缀、已生成 token（包含推理）和请求本地速率。即使禁用周期性行，阶段变化和 `request done` 仍会记录。请求日志**仅包含计数，绝不包含提示词文本**。

每个 `POST /v1/responses` 额外输出一条 `responses:` 状态行：HTTP 结果、整次交换的耗时，以及决定走哪条校验路径的**请求形状**（`model`、`stream`、tools 数量、是否带 `tool_choice`、客户端送的 `store`、是否带链式 id、`max_output_tokens`）；非 2xx 时再附上客户端收到的 `error.message`，这样“请求直接失败”在服务器终端就能读出是哪一条契约拒绝的。请求体无法解析时形状显示为 `model=<unparseable body>`——缺 `Content-Type: application/json` 与非法 UTF-8 都属于这一类。该行同样只含元信息，不含提示词文本。

成功的非流式响应和流式响应的终止完成帧，均包含权威的 `usage` 及 llama.cpp 兼容的 `timings` 对象。`timings.context_n` 是完整提示词加补全深度，`context_limit` 是 KV 槽位容量，`cached_n` 将复用的提示词 token 与实际评估的 `prompt_n` token 区分开来。流式指标无条件输出，因此未发送 `stream_options.include_usage` 的客户端不会静默记录零 token。

`POST /v1/responses` 复用相同的模型调度与请求上限；支持文本消息、重放的 function call/output、自定义 function tools，以及 `stream=true` 的 typed SSE。启用视觉模型及 projector 时，`input_image` 的 `image_url` 可以是 base64 `data:` URI、裸 base64，或 http(s) URL；URL 由服务端抓取，默认只允许公网可路由地址（见 `[vision]`）。与存储无关、始终返回 400 的语义有：文件输入（`input_file`）、`file_id`、hosted tools、结构化输出模式、`truncation="auto"`、`background`、除 `reasoning.encrypted_content` 之外的 `include`，以及任何未被识别的请求字段。`parallel_tool_calls=false` 已被接受：解码器没有并行开关，模型仍可能给出多个调用，因此多余的调用会被丢掉，并在响应的 `warnings` 里说明丢掉了几个——静默少给调用会让 agent 循环永远等一个没人问过的结果。错误体的 `error.param` 指向客户端实际发送的那个字段；无法归到单一字段时记为 `request`（未识别字段），`GET`/`DELETE /v1/responses/{id}` 这类路径参数则记为 `response_id`。

链式请求上重发 `instructions` 是允许的，只要它与链上已经存储的系统前缀逐字相同（SDK 每轮都重发同一份提示词的形状）；改动过的、或者链上根本没有系统前缀却带来了新 `instructions` 的请求，会以 `param="instructions"` 报 400，而不是悄悄多插一个 system 轮次、把 KV 前缀打断。

`serve.response_store_entries`（默认 `0`）决定服务端是否保存对话状态。为 `0` 时契约是无状态的：多轮请求由客户端在 `input` 中重放此前项目，而 `previous_response_id`、`conversation`、`store=true` 和 `item_reference` 输入条目一律返回 400 并提示开启该配置。设为正数后，`store=true`（此时 `store` 的默认值也变为 `true`，与 OpenAI 一致）的完成与截断响应会连同其原始信封与本次提示条目一起进入 LRU，于是 `previous_response_id` 链、`conversation` 游标、`{"type":"item_reference","id":...}` 输入条目，以及 `GET /v1/responses`（列表）、`GET|DELETE /v1/responses/{response_id}`、`GET /v1/responses/{response_id}/input_items`、`POST /v1/responses/{response_id}/cancel` 都可用；列表与 `input_items` 接受 `limit`（默认 20，上限 100）和 `order`（`asc`/`desc`）。

链式重放之所以便宜，是因为它把存储的历史原样拼回提示词，重建出与上一轮完全相同的 token 前缀，分页 KV 的 radix tree 自己就会命中它；这个存储只保存 DTO 与 JSON，从不固定或引用 KV 页。状态是有界的：被淘汰的 id 之后再被引用会得到明确的 400（`no stored response ...`），绝不静默降级成空上下文；`conversation` 配 `store=false` 同样直接 400，因为未存储的轮次无法推进下一轮要读的游标。`POST .../cancel` 对本服务器永远是 400（`param="background"`）：生成是同步的，一个响应在被赋予 id 时已经到达终态，要停止生成就断开这条流。

默认状态下这份状态随进程消失，重启后旧 id 一样是明确的 400。需要跨重启的链就设 `serve.response_store_path`：启动时加载，此后每次改动 LRU 都整体重写（先写 `<path>.tmp` 再 rename，读者看不到半个文件），加载时按 `last_used` 顺序套用上限、丢掉超出窗口的旧轮次。文件是**明文**的——每个保存的提示词和回复都在里面——所以默认不写盘；路径为空串等于不设。文件缺失视为空库开始，版本号不是本构建读取的 `1`、或内容无法解析，则**拒绝启动**而不是空库启动：后者会让所有已存 id 看起来像丢了，而数据其实还在盘上。`response_store_entries=0` 时该路径被忽略。

`include:["reasoning.encrypted_content"]` 也已兑现，但它不是加密：推理条目上的 `encrypted_content` 是一个指向本存储的不透明句柄（`rehs_<response_id>`），客户端把它原样带回 `input` 时，服务端据此重放那一轮的推理文本。因此它要求 `store=true`（否则 400，提示先开启服务端状态），句柄的生命周期就是那一轮在 LRU 里的生命周期；本服务器从未签发过的句柄，或指向已被淘汰/重启前那一轮的句柄，都会以明确的 400 拒绝。仍带可读 `summary` 的推理条目照旧可以无状态重放。

Responses API 只接受 plain text 与默认 `text.verbosity="medium"`；`low`/`high` 不支持。`reasoning.effort="minimal"` 在模板支持时近似映射为本地 `low`。本地 backend 没有独立的 hidden reasoning token 计量，因此 `reasoning_tokens` 为 0，外显思考内容计入 `output_tokens`。流式响应使用 `Cache-Control: no-store` 和 `X-Accel-Buffering: no`；生成在响应创建后失败时发出 typed `response.failed`。这些 HTTP 缓存头不会关闭模型 KV prefix cache。

**`[vision]`** — `image_url_allow_private`（`INFR_IMAGE_URL_ALLOW_PRIVATE`，默认 `false`）决定服务端替客户端取 `image_url` 时可以连哪些地址。抓取本身已经锁了三条：单图最多 32 MiB、每一跳最多 10 秒（连接 5 秒）、以及只允许公网可路由地址——回环、RFC1918、链路本地（云元数据端点就住在 `169.254.169.254`）、CGNAT、组播和保留段全部拒绝；重定向由本项目自己逐跳跟，每一跳重过一遍地址规则，`file://` 这类 scheme 直接报错，不会当成非法 base64。设成 `true` 只抬掉地址这一条，给确实要抓本机图片的部署：NAS、局域网摄像头、跟服务器同在一只容器里的测试进程；字节和时间上界照旧。要明白抬掉代价是什么——"连上了但 404"和"连不上"是两种不同的回答，图片又能被模型描述回请求方，不设这条就等于让请求者拿这台机器探内网，所以默认关。启动向导在"启用视觉图片理解"之后接着问这一条，答"否"也会把 `false` 写进命令行，盖掉配置文件或环境里已经抬开的值。

**`[hub]`** — 模型获取（`infr pull`，以及模型缺失时 `infr run` / `infr serve` 执行的自动拉取）。`endpoint`（`INFR_HF_ENDPOINT`）选择 HuggingFace 兼容源：默认 `https://huggingface.co`，也可以是 `https://hf-mirror.com`。`pull_jobs`（`INFR_PULL_JOBS`，默认 `8`）表示单个模型下载可同时使用的连接数。慢的是连接而非链路：针对相同 CDN 对象测量，一个连接持续为 8.39 MiB/s，五个连接持续为 75.1 MiB/s，即每个连接 15.0 MiB/s，因此每连接速率不降反升。

连接用于何处取决于模型，设置本身无需说明：分片模型是 `-NNNNN-of-MMMMM` 分片，每条连接获取一个文件；单文件发布的模型则被切分为字节范围后重组（`unsloth/DeepSeek-V3.2-GGUF` 的单个 150 GiB `UD-TQ1_0` 在同一 60 秒窗口中，`1` 时为 10.9 MiB/s，默认 `8` 时为 76.8 MiB/s；整次 150 GiB 拉取端到端为 76.3 MiB/s）。一个数字覆盖两种情况，因为按轴分别限制会相乘：在文件数量由发布者决定的仓库中，8 个文件 × 8 个范围就是 64 个套接字（`DeepSeek-V3.2-REAP` 发布了 236 个文件）。

范围下载需要服务器支持；不支持的服务器（没有 `Accept-Ranges`，或请求 `206` 却返回 `200`）会回退为单一流，64 MiB 或更小的文件绝不切分。`0` 和 `1` 都严格表示一条连接，适用于计费链路，或反对来自同一客户端多连接的代理。

**`[debug]`** — 填毒/屏障/转储开关：`coopmat`（输出枚举到和选定的 coopmat 形状，对 Intel Arc 很有用）、`bda_chunk`、`wide_dispatch`、`chat`、`moe_counts`、`moe_counts_dump`、`poison_uninit`、`no_barrier`、`full_barrier`。

## GPU 启动锁 / GPU startup lock

`device.gpu_lock`（默认 `true`，环境变量 `INFR_NO_GPU_LOCK`）让 `infr` 在触碰设备之前检查这块卡上是否已经有另一个 `infr` 进程在运行。原因很实际：两个进程共享一张卡会互相压缩统一显存 arena（实测 30K 解码每 1 GiB arena 约 1 ms/token），所以第二个进程既跑得更差，它测出的数字也不可信——在此之前，唯一的污染信号就是吞吐量本身。

锁是 loopback 上一个固定监听端口（`127.0.0.1:28765`，低于两平台的临时端口下限）的 `bind()`：内核拥有它，进程以任何方式退出（包括 `kill -9` 与断电）都会立刻释放，因此不存在需要清理的陈旧锁文件。持锁进程在同一端口应答一行身份信息（pid、设备、调用形式），所以中断启动只发生在一个活着的 `infr` **自己声称**持有该设备时：

| 结果 | 代码 | 行为 |
| --- | --- | --- |
| 同一设备的另一个 `infr` 持锁 | `INFR-E1001` | **启动失败**，错误信息给出持锁者的 pid、设备与命令 |
| `device.gpu_lock = false` / `INFR_NO_GPU_LOCK=1` | `INFR-W1001` | 警告并启动，不再检测并发实例 |
| 持锁者是另一块设备上的 `infr` | `INFR-W1002` | 警告并启动（多卡各跑一个进程是合法用法） |
| 端口被占用，但应答不像 `infr` | `INFR-W1003` | 警告并启动 |
| 端口根本无法绑定（权限、保留区间） | `INFR-W1004` | 警告并启动 |

设备键按集合比较，并区分编号与否：`Vulkan0,Vulkan1` 与 `Vulkan1` 重叠（共享一张卡），`Vulkan0` 与 `Vulkan1` 不重叠；而**未编号**的默认值 `vulkan`（“由引擎挑选”）与任何 `VulkanN` 都算重叠，因为无法证明它们不是同一块卡——只有可证明不相交的索引才会以 `INFR-W1002` 放行。单卡机器上这正好是想要的严格性；多卡机器上若默认挑选与显式索引确实不同，请用 `--dev` 显式指定两者，或用下面的开关放行。

后四种是刻意的失败开放（fail-open）：这是一把诊断用的锁，不是一份资源预留，任何无法证明「另一个 `infr` 正在用这块卡」的情况都不会拦住启动。`--dev cpu` 完全不参与该锁；`infr devices`、`resources`、`pull`、`report` 这类不加载模型的命令也不。`infr compare` 自己不持锁，它逐个启动的 `infr bench` 子进程各自取锁，因此外部并发仍会被检出。

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
