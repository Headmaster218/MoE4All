---
kind: change-history
status: historical
scope: qwen38-uncensor
last_verified: 2026-10-07
---

# Qwen3.8 拒绝方向投影（去审查）：独立模块与双层控制

> 本页记录把一个可选的"注入式去审查"特性接入本 fork 的过程与取舍。它是该特性的**第二次实现**：第一版把加载器放在 `infr-llama/src/uncensor.rs`、把三个键登记进配置层、只有"开/关"一档。本版的目标写在下面三条里：**少改引擎**、**单独成模块**、**控制逻辑补全到"启动可选 + 每请求可关"**。当前可配置行为以[配置手册](../../../reference/configuration.md)为准，本页不重复字段表。

## 特性的来源与语义

Strata 的"注入式去审查"不是提示词技巧，而是 **refusal-direction control-vector projection**：把"模型即将说出『不』之前所趋向的那个方向"从残差流里减去。

```
h ← h − (h·v)v          v 为每层一个的单位方向，h 为残差
```

三种做法的取舍是明确的：**微调**改变权重、污染知识且不可逆；**提示注入**（"请不要拒绝"）不稳定、占用上下文，也解释不了模型为什么拒绝；**投影**只减去一个方向，保留权重、知识和语法，代价可测（Qwen3.8 实测约占 token 时间的 0.2–0.4%）。llama.cpp 的等价参数是 `--cvec-mode project --cvec-dir per-layer --control-vector-layer-range 4 44`。

方向文件是 `general.architecture = controlvector` 的 GGUF，每层一行 `direction.N`（`[n_embd]` 的 f32）。**文件里的层号从 1 开始，本引擎的层号从 0 开始**，这个差一是移植时最容易出错的地方，因此加载期归一化与层号映射都有专门的单元测试。

对 `qwen4exp` 而言 `h` 不是单条残差而是**宽残差里的每一条 hyper-connection 流**：投影必须对 `hc_mult` 条流各做一次，位置在 `QwenHcInject` **之后**、对 `qwen_wide` **原地**执行——"第 N 层之后"的含义就是下一层的 `QwenHcMix` 读到已投影的残差，而该注入正是本层对宽残差的最后一次写。

## 关键决策

| 决策 | 选定 | 理由与代价 |
| --- | --- | --- |
| 特性代码放哪 | 新建 `crates/infr-uncensor` crate | 加载、校验、层范围、请求语义全部留在一处，引擎只剩三个动作：加载期读一次文件、每层发一个算子、每请求 honour 一个布尔。代价是多一个 crate 与两处 Cargo 依赖声明 |
| 是否走 `Config` | **不走**，由 `infr_uncensor::Config::from_env()` 自读三个变量 | 走配置层要同时改 `cfg_struct!`、`manifest.rs`、`env.rs`、`infr.example.toml`、参考文档与 `POST_MIGRATION` 清单。投影在模型加载时被分配或根本不分配，不是运行期调节项，登记为配置键会承诺一个谁都不能在运行期改的旋钮。三个键仍出现在 `manifest.rs` 的 `NOT_MIGRATED` 里并附理由，否则"manifest 与树一致"的防漂移测试会失败 |
| 运行期开关机制 | **设备侧标量**（`scale_buf`，4 字节），图两种情况完全一样 | 沿用 `Op::Softmax { scale, scale_buf }` 的先例：`scale = 0` 时算的是 `h − 0·v`，逐位等价于原模型。于是已编译的 plan、pipeline 缓存、kernel 名单全部继续有效，切换只是一次 4 字节上传。不用 push constant：它会改 `push_size` → pipeline layout → 撞 kernel 名单键缓存。明确接受的代价：加载了方向文件之后，即使关着也付那 0.2–0.4% |
| 方向与标量归谁 | **per-slot 两块 buffer**：`dirs`（`BufferUsage::Weights`，永不改写）+ `scale`（`Staging`，host-visible） | 第一版把方向块塞进权重上传循环，`wspecs` 与 `wpush` 必须严格同序——那是整个集成里最脆的一处耦合。本版完全不动权重上传顺序，两块 buffer 只在 `bind_layer_io` 一处绑定（15 个调用点全部经过它，与 `qwen_wide`/`ple_embd` 同一范式） |
| 切换时丢什么 | 丢**该 slot 自己的** materialized tokens（`SeamKv::reset()`），逐 slot 判定 | 用另一种投影算出的热前缀不是当前模型的前缀，必须重读 prompt——这正是 llama.cpp 改 `--cvec-mode`、Strata 调 `cvec_set_enabled()` 时的强制重读。逐 slot 而非全局：一个从未跑过的 peer（新建兄弟槽、或从磁盘恢复因而没有记录状态的槽）不应抹掉主槽从未被失效的热前缀 |
| 请求字段 | `uncensor: true\|false`，并接受 Strata 的 `experimental_speed_projection` 作别名；**两者冲突 → 400**；缺省 = 跟随该 slot 现态 | 别名让 Strata 客户端零改动迁移；冲突不静默选边（`api_merge` 返回 `Conflict`）。缺省"跟随会话"而不是"默认开"，因为无请求上下文的 pass（MTP 草稿/验证、denoise）也走同一个 runner，它们必须采样自己那一份模型 |
| 两条 API | 字段加在 `ChatRequest` 上，`/v1/responses` 通过 `into_chat()` 复用同一入口 | 但 Responses 侧必须**声明**该字段：那里任何未知的非 null 键都会被拒为 unsupported request field |
| 未配置时 | 不建算子、不建缓冲 | 没有方向文件就没有 `UncensorState`，图与改动前一致 |
| 架构不匹配 | 硬错误 | 非 `qwen4exp` 指定方向文件、行宽不等于本模型 `n_embd`、文件覆盖层数不足，都是错误而非静默不生效；**只夹紧层范围上限**，下限越界即空区间并报错（第一版曾在三层模型上把"从第 4 层起"变成"从第 2 层起"） |
| Metal | 显式 `Unsupported` | 与本 fork 其余 Qwen HC 算子同处（Mac 目前没有 `qwen4exp` 路径）；写明白比落到通用报错强 |

## 改动清单

以 `git diff --stat` 为准：**29 个文件、新增 1756 行、删除 8 行**，其中 6 个新文件（4 个属于新 crate、1 个 shader、1 个本页）。删除的行里有 3 行与本特性无关：CI 的 `cargo clippy -D warnings` 是 PR 门槛，而 base 上已有三处告警（`hcw % 32 == 0` 两处、`build_draft_graph` 参数过多），顺手改成 `hcw.is_multiple_of(32)` 与 `#[allow(clippy::too_many_arguments)]`（后者与本文件其余 6 个同名函数的写法一致）。

| 模块 | 文件 | 改动 |
| --- | --- | --- |
| 新模块 | `crates/infr-uncensor/{Cargo.toml,src/lib.rs,src/config.rs,src/vectors.rs}`（新） | `lib.rs`：`Projection`、纯函数 `span/resolve`、请求语义 `api_merge/effective/scale/flipped`；`config.rs`：三个 `INFR_UNCENSOR_*` 的解析（未设与设空串都算关闭，非数字层号是硬错误）；`vectors.rs`：`controlvector` GGUF 解析、逐层 dtype/shape 校验、**加载期一次性单位归一化**、非有限/零向量报错 |
| 图 | `infr-core/src/graph.rs` | 新增 `Op::UncensorProject`（`x`/`dir`/`dst` + `rows`/`hc`/`n_embd`/`dir_off`/`scale`/`scale_buf`）及 `kind()`、`io()` 两处穷尽匹配。刻意不叫 `QwenHc*`：投影不专属某个架构。元素偏移字段叫 `dir_off` 而非 `voff`，因为本 fork 的向量里没有 `voff` 这个概念 |
| 接线 | `infr-llama/src/seam/runner.rs` | 冷启动 `resolve` + `UncensorState::new`；`uncensor_apply()` 每请求决定并逐 slot 失效；建图时按绝对层号在覆盖段内发射；`bind_layer_io`/`bind_parallel_layer_io` 各加两个参数（并行路径共享主槽的 `scale`，因为宽残差对所有 lane 一次投影） |
| 状态 | `infr-llama/src/seam/weights.rs` | `UncensorState`（`dirs_buf`/`scale_buf`/`uploaded`/`ran`）挂进 `SeamKv`，`fork` 时继承父槽的 `ran`：分叉出的 KV 行出自同一个模型，不该在第一次请求时被判定为"切换" |
| CPU | `infr-cpu/src/lib.rs` | 参考实现；先 clone 输入行再整体写回，因此天然支持 `x == dst`，并同样 honour `scale_buf` |
| Vulkan | `shaders/uncensor_project.comp`（新）、`build.rs`、`gemm.rs`、`recorder.rs`、`adapter.rs` | 256 线程、`subgroupAdd` + `shared` 两级归约的整行内核；**一个 workgroup 独占一条流**，这是原地写合法性的来源；`USE_SCALE_BUF` 双变体，动态变体的 push 只剩 8 字节（`n_embd`、`dir_off`），被写的残差仍是最后一个 binding（`dispatch` 以"最后 n_out 个绑定是输出"做 hazard 跟踪） |
| Metal | `infr-metal/src/exec.rs` | 显式 `Unsupported` |
| 配置层 | `infr-core/src/config/manifest.rs` | 三个键进 `NOT_MIGRATED` 并写明"启动参数而非调节项"的理由——这是本版对配置子系统唯一的改动 |
| API | `infr-server/src/{lib.rs,responses.rs}`、`infr-llama/src/sampling.rs`、`infr-cli/src/main.rs` | `ChatRequest` 两个字段 + `GenParams::from_request` 里一次 `api_merge`（冲突 → 400）+ `RequestSampling.uncensor` + CLI 一行透传 |
| 启动器 | `Start-INFR-Wizard.cmd` | 询问（默认取上次、30 秒超时）、**只有交互选择才写** `gui-data/uncensor.txt`、`-uncensor on\|off\|<PATH>` / `-no-uncensor` 单次覆盖不改默认、导出三个环境变量 |
| 文档 | `documentation/reference/configuration.md`、`documentation/evidence/changes/README.md`（索引）、`CHANGELOG.md`、本页 | 配置层的"例外"说明、索引可达、`## [Unreleased]` 条目 |

## 双层控制的确切语义

**启动决定投影是否存在**（`INFR_UNCENSOR_VECTOR` / `INFR_UNCENSOR_FIRST_LAYER` / `INFR_UNCENSOR_LAST_LAYER`，默认层范围 4–44，只夹紧上限）。关闭的正确含义是"变量不存在"；本版也把"设成空串"视为关闭，但脚本仍按前者的保守写法。

**每个请求决定是否生效**，三态明确：

```jsonc
{"model": "...", "messages": [...], "uncensor": false}   // 这一次用未经投影的模型
{"model": "...", "messages": [...]}                       // 保持该 slot 当前状态
{"experimental_speed_projection": false}                  // Strata 拼写，等价
{"uncensor": true, "experimental_speed_projection": false} // 400，不猜
```

请求级字段在**没有加载方向文件**时被接受并忽略（没有可切换的东西），不会报错——否则所有客户端都得为一种部署改请求、为另一种改不了。

投影随层段自然生效：草稿图与验证图只有在覆盖到投影层段时才发算子，而承载开关的标量属于 slot 而非请求，所以无请求上下文的 pass 一定沿用该 slot 的现态，不会拿另一个模型去验证刚采出的 token。

## 移植步骤（在别的副本或分支上复刻）

顺序有意义：先让**类型**存在，再让**内核**存在，最后接线。

1. 新 crate 建好、进 `workspace.members` 与 `workspace.dependencies`，此时它还不被任何人依赖，可以先把自己的单测编好。
2. `Op` 枚举 + `kind()` + `io()` 三处同时加。`io()` 里必须把 `scale_buf` 一并登记为 READ，否则绑定校验看不见它。
3. **让编译器枚举后端面**：`cargo check --workspace --all-targets` 只会对非穷尽匹配的后端报错。本 fork 里 CPU 会报，Metal 由于 `#![cfg(target_os = "macos")]` 在 Windows 上不参与编译，**必须手动补**它的 `Unsupported` 分支，否则这条断裂要到 Mac 上才暴露。
4. shader 文件名必须在 `build.rs` 注册（`OUT_DIR` 下生成 `.spv`）、在 `gemm.rs` 暴露 `*_spv()`、在 `recorder.rs` 写 `dispatch`、在 `adapter.rs` 下命令。**四处名字必须一致**，注册缺失表现为运行期找不到内核而不是编译失败。
5. 引擎接线：`SeamKv` 加状态 → 冷启动创建 → `bind_layer_io` 一处绑定 → 建图闭包里按层段声明输入并发射。声明输入的门控与发射的门控必须是**同一个**条件（本 fork 里是 `(l_first..l_end).any(|l| proj.covers(l))` 对 `proj.covers(l)`），否则会出现"发射了但没绑定"的运行期未绑定输入。
6. per-request 字段自下而上：`RequestSampling` → `GenParams`/`ChatRequest`/`ResponsesRequest` → CLI 透传 → runner 里一次 `uncensor_apply`。
7. 启动器与文档。

## 验证

- 基线：改动前 `cargo check --workspace --all-targets` 通过，之后的报错才都归因于本次改动。
- `cargo check --workspace --all-targets` 通过，且**没有新增警告**——`DecodeHandles` 里两个 uncensor 句柄与 15 个绑定调用点都被真实读取，这排除了"声明了但忘了绑"。
- `cargo build --release -p infr-cli` 通过，产出 `target/release/infr.exe`；两个 SPIR-V 变体（静态 scale 与 `USE_SCALE_BUF`）均在构建期编译过。
- 单元测试 18 项（`infr-uncensor`）：`config.rs` 6 项（默认关闭、空串=关闭、未设的边界保留默认、非数字层号拒绝启动、特性关闭时边界仍被校验）、`lib.rs` 8 项（层范围与文件覆盖、无文件即关、架构检查先于读文件、本模型没有那么多层则报错、两种拼写的合并与冲突、缺省跟随会话而非默认、`scale` 的两个取值、翻转以"上次真跑的是哪边"为准）、`vectors.rs` 4 项（加载并逐行单位归一化、行宽不符、不是 `controlvector`、全零方向）——本地已运行：**18 通过 / 0 失败**。`infr-core` 的 `config` 46 项同样通过，含 `manifest_matches_the_tree` 与 `no_infr_env_reads_outside_the_config_layer` 两道防漂移测试（说明三个 `INFR_UNCENSOR_*` 的 `NOT_MIGRATED` 登记既必要又够用）。CI 中本地可复现的门槛也都干净：`cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets --locked -- -D warnings`，以及 `cargo check -p infr-metal --all-targets --locked --target aarch64-apple-darwin`——最后这条尤其值得跑，因为 Windows 上 `infr-metal` 整个 crate 被 `#![cfg(target_os = "macos")]` 关成空模块，新增的那条 `Unsupported` 分支只有交叉检查能证明它编得过。

过程中值得留下的两个坑（第一版也踩过）：

1. 手写 GGUF 时**值类型标签必须是 u32**，写成 1 字节会让整段 KV 解析错位，症状是难解的 `unknown value type 3336`。
2. `span()` 只夹紧**上限**。把下限也夹紧会让"投影第 4 层起"在三层模型上变成"投影第 2 层"——一个谁都没要求的投影。

**验证边界**：加载器、层范围算术、请求语义、CPU 投影数学与编译期接线都有测试或编译证据；**Vulkan 内核未在真实 GPU + 真实 `qwen4exp` 模型 + 真实方向文件上运行过**，per-request 翻转与并行路径（多 lane 共享一次投影）也未实机验证。首次实机运行建议先 `--dev cpu` 确认行为，再切 Vulkan；并特意跑一次"同一会话里交替 `uncensor: true/false`"，用来观察热前缀被丢弃、prompt 重读是否符合预期（代价是重读，收益是不串模型）。
