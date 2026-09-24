# 从 infr 引擎到 Agent 发行版：架构调研与建议

日期：2026-09-23。代码基线：`9085a78`。

状态：设计提案，尚未实施。此次只阅读代码、已安装 DSH 的包和脱敏配置结构，不启动模型，不改运行配置，不迁移用户数据。未读取凭据文件、会话内容或记忆库。下面明确区分现状与建议，新增目录、命令和协议名称均为暂定。

## 1. 对目标的理解

要做的不是给 `/v1/chat/completions` 加一个聊天页面，而是一个完整的本地优先 Agent 产品：

- 安装后包含推理引擎、harness、前端、经过选择的插件和可用的默认配置。
- Agent 能执行工具、持续推进任务、管理工作区、恢复会话、使用记忆，并逐步支持后台任务与子 Agent。
- 模型启动、资源配置、插件安装和任务执行在同一产品中管理，不要求用户手工拼服务。
- 本地 infr 是核心差异化能力，但 harness 不被绑死在某一个模型或 GPU 后端上。
- 保留独立引擎用途：现有 CLI、OpenAI 兼容 API、bench 和外部客户端仍然能单独使用。

建议形成同一代码仓库下的两种发行物：轻量 Engine 包，以及包含 Engine 的 Agent 包。Web 和 Desktop 是同一个 Agent 产品的两种入口，不做两套会话系统。

用户已补充：当前安装的 DSH 只是日常使用和功能参考；MoE4All 发行版将采用源码，并做较多修改。因此这里不以“给现成 DSH 配一个 provider”作为最终产品方案。

**推荐路线：基于 DSH 源码维护自有 harness/Web UI + 自有产品功能和插件 + 独立 infr 引擎进程。保留可用的基础机制，按产品需求接管实现；固定上游源码基线，不依赖用户安装的 DSH。**

## 2. 当前 infr 的真实边界

当前有 17 个 workspace crates。以下规模是 `src` 中 Rust 源文件的大致物理行数，包含测试、注释和数据表，只用于定位职责集中处，不能据此直接判断代码质量。

| 模块 | 现状与证据 | 建议 |
|---|---|---|
| `infr-engine` | 仅约 19 行聊天类型重导出，不负责 Engine 生命周期；manifest 仍列出多项未在该文件使用的依赖 | 优先处理这个名不副实的边界；承接真正的对外引擎契约，而不是再放 Agent 代码 |
| `infr-llama` | 约 4.2 万行；实际上负责多架构模型、session、采样、MTP、并发以及后端适配 | 明确为模型执行层；内部先分模块，之后再考虑更名 `infr-model` |
| `infr-cli` | `main.rs` 约 5700 行；除了参数入口，还包含模型装配、生成器适配、embedding/vision 集成、benchmark | 先按 command/runtime/presentation 拆文件，再把可复用装配移到引擎层 |
| `infr-server` | 约 5900 行；HTTP DTO、生成器 trait、参数校验、SSE、统计、终端表格都在这里 | 留协议适配与请求管理；生成契约下沉，终端展示上移 |
| `infr-gui` | 既是页面服务器，又负责模型目录、估算、下载、worker 生命周期；直接依赖 llama/Vulkan | 作为已有产品能力来源；新产品中逐步收敛为模型管理页面和管理服务，不能再复制一套永久维护 |
| `infr-core` | 约 2.5 万行；图、后端契约、配置、资源、分页、平台内存等共存 | 先整理内部领域边界，不马上拆成更多小 crates，更不加入 Agent/插件/会话数据库 |
| Vulkan/CPU/Metal | 各自有真实硬件、构建和算子边界 | 保留独立；大文件可拆模块，不因接入 harness 重写 kernel/调度 |
| GGUF/Hub/Chat | 分别负责格式、模型获取、模板与消息解析 | 保留；`infr-chat` 不是 Agent 会话数据库 |
| Embedding/Vision | 有独立模型职责，但运行时可能与 LLM 共用 GPU 资源 | 保留模块边界，不误拆成每个插件一套独立 GPU 服务 |
| Prof/Prof-rt/Testkit | 小，但分别有 proc-macro、运行时和 dev-only 依赖隔离原因 | 不按行数合并；`testkit` 的独立性还避免 core/gguf 依赖环 |

关键代码入口：

- [`infr-engine`](../crates/infr-engine/src/lib.rs)：现有 facade。
- [`infr-server`](../crates/infr-server/src/lib.rs)：`ChatGenerator` 在 218 行附近；路由在 2134 行附近。
- [`infr-cli`](../crates/infr-cli/src/main.rs)：`SeamGenerator`/`ParallelGenerator` 在 2092 行附近；`cmd_serve` 在 4509 行附近。
- [`GUI worker`](../crates/infr-gui/src/worker.rs)：101 行附近启动进程，610 行附近从人类可读日志提取内存字段。
- [`GUI catalog`](../crates/infr-gui/src/catalog.rs)：调用 llama 的内存估算和 Vulkan 探测，说明它不是单纯 UI。

最核心的问题不是 crate 数量，而是：**执行模型的入口没有独立归属，应用装配堆在 CLI，控制面混在 GUI，协议和展示混在 server。** 在这些地方继续塞 Agent 功能会扩大混乱。

此外确实存在文档漂移：`infr-llama/src/lib.rs` 的开头还写着无 KV cache 的 bring-up 状态；`docs/README.md` 对 DeepSeek 仍有“Nothing implemented yet”的旧描述。这比目录名称不整齐更容易误导后续开发。

## 3. 本机 DSH 调研

### 3.1 已确认的安装形态

- 安装目录为 `D:\DSH Desktop`，Desktop 包声明版本 `0.6.3`。
- 随附 `@deepseek-ai/dsh` 声明版本为 `0.1.1-rc.2`，有独立 Node 启动入口。
- Electron 负责桌面外壳，harness 运行在独立 Node 进程。现有结构已经支持复用 Web UI 和插件运行时。
- 随附 `@deepseek-ai` 命名空间目录下有 197 个包目录，包括 Cordis 和各类服务/实现/UI 包。**安装数量不等于启用数量，也不代表本产品应该照搬同等数量。**
- 用户 profile 位于 `%APPDATA%\dsh-desktop\harness\profiles\web`，与程序安装目录分开。

profile manifest 中列出的第三方组合包：

| 包 | 本机解析版本 | 可吸收的能力 |
|---|---|---|
| `dshmarket` | 1.45.1 | 插件发现和管理 |
| `@modusensus/dsh-mneme` | 0.7.29 | 跨会话记忆、检索、整理 |
| `dsh-easyrewrite` | 2.4.1 | 消息撤回和重新编辑 |
| `dsh-plugin-cron-scheduler` | 0.2.7 | 定时任务和运行历史 |
| `@deads-inc/dsh-web-search-brave` | 1.0.0 | 搜索服务提供方 |
| `@linxin666/dsh-remote-web-ui` | 0.3.17 | 远程界面与配对 |
| `@dsh-external/dsh-plugin-tts` | 0.3.1 | 朗读与语音设置 |

这里记录的是安装包和 profile 声明，不是逐项运行验收结果。未对整个安装目录做上游逐文件比对，因此不能声称已找全个人修改。

### 3.2 已找到的改造目录和具体修改

已定位主要本地工作目录 `D:\AISuperAssistant\DSH`，其中有 `PLUGINS.md`、`MEMORY-PLUGINS.md`、远程审批状态与回退说明、原始 JS 备份、插件源码审阅目录及工具脚本。该目录混合工程改造和个人任务数据，不是可以直接复制进公开仓库的源码包。

| 已查到的内容 | 实际证据 | 源码发行版中的归属 |
|---|---|---|
| 固定访问端口 | 安装目录 `out/main/index.js:7517` 的 LOCAL PATCH 及原文件备份，把动态端口改为固定 1633 | 正式 listen/public endpoint 配置；launcher、健康检查、UI 和远程入口读取同一配置 |
| 手机远程审批与事件 | remote-web-ui 的 `lib/index.js` 和 `lib/client.js` 增补旧版 `events.mux/events.host` WebSocket 路由 | 统一远程事件与审批传输；不能只做到远程聊天 |
| transport 兼容 | remote-web-ui 的 head 注入脚本补 `createApiClient` fallback | 在选定源码版本统一客户端 transport 契约，避免保留页面注入兼容技巧 |
| 文件上传插件加固 | `review/dsh-web-file-uploader` 为 Git 工作区；`src/core/host-core.js`、构建产物已改，另有两份补丁测试 | 附件服务正式处理文件名、目录/会话归属和删除授权；迁移测试后重新验证，不将本地补丁自动视为完整安全保证 |
| 记忆使用原生 embedding | profile 的 dependency override 和说明文件 | 统一 embedding 能力及可选后端，见下文 |
| 独立业务工具和 skills | 存在 `bili-comment-tool`、媒体/ASR 脚本、局部 skill 目录 | 提炼为可选工具/skill 集；不附带账号、私有内容或用户历史 |
| 标题请求代理、抓流/诊断脚本 | `title-proxy.mjs` 明确写为抓包后可撤的调试代理 | 开发诊断/fixtures，不作为正式产品必经代理 |

本地远程审批说明还记录了一个很有价值的集成失败：profile 的端口覆盖与桌面 watchdog 所等待的端口不同，导致启动超时。新产品应消除这种双重配置所有权，而不是仅把本机端口数字写死。

上传插件没有出现在当前 web profile 的第三方 bundles 清单中，应视为本地审阅/修改过的候选能力，不能声称现在正在启用。只检查了源码 diff，未运行这些外部脚本或测试。历史调研文档的下载量、安全评价和版本结论也不作为此次重新验证的事实。

profile 有明确的补丁说明：把 `@huggingface/transformers` override 到本地 stub，原因是记忆的 embedding 已走自建 GPU/OpenAI 兼容服务，不需要附带 ONNX 推理依赖。源码也可见独立的 external embedding 路径，以及 local/reranker 路径对 transformers 的动态导入。

这说明发行版可以有意去掉重复模型栈，但**不宜把静默 stub 当作长期产品接口**。建议把本地 embedding 改为明确的可选依赖/独立插件；缺少该能力时在设置阶段报告，而不是切换后悄悄退化。短期固定此插件版本并保留可重现补丁与回归测试。

profile patch 还涉及搜索提供方、记忆整理、标题生成和远程 UI。这些应分别归为发行默认值、用户设置和代码补丁，不能统统复制为新产品的硬编码默认值。

另一个兼容性信号：本机 remote-web-ui 的 manifest 声明 `dsh >=0.1.2-rc.1`，而随附 dsh 声明为 `0.1.1-rc.2`；上述事件路由和 transport 补丁正是在弥合这类差异。本地说明记录了远程审批成功，但本次没有重新启动服务做 E2E 验证。新源码版应选择统一基线并迁移功能，不应长期维持两套版本协议。

### 3.3 能复用到什么程度

本机包中已经有 agent-loop、工具注册与执行、会话持久化、压缩、子 Agent、MCP、skills、审批、沙箱适配、前端扩展点。`dsh-llm-pi-ai` 的配置支持自建 OpenAI 兼容端点，代码通过 `ctx.llm.registerAdapter()` 注册模型提供方。因此最小接入不需要修改 Agent 循环。

上游的 [架构文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md) 同样采用 profile/bundle 组合和可替换服务。注意这是当前上游文档，不能把其中较新的 SDK/Desktop/事件格式直接当成本机旧版本的接口。上游明确提示仍处于快速变化的 developer preview，应锁定验证过的版本。[项目说明](https://github.com/deepseek-ai/deepseek-harness)

## 4. 路线选择

| 路线 | 优点 | 主要代价 | 判断 |
|---|---|---|---|
| 基于 DSH 源码维护产品 fork | 复用现有插件、会话、工具、UI，允许较深产品改造 | 需要上游差异管理、明确代码所有权和兼容性测试 | 符合用户目标，首选 |
| 只在发布的 DSH 包上加 profile/provider | 最小验证路径短，维护差异较少 | 产品修改受外部版本和扩展点限制 | 可做接入验证，不作为最终形态 |
| 只用 Cordis，自写 TypeScript harness | 可完全控制事件与产品模型 | 要重新实现循环、恢复、取消、工具历史、压缩与 UI 适配；现有插件未必兼容 | 有明确不可满足的需求时再评估 |
| Rust 全栈自写 harness | 单一服务语言，容易嵌入原生能力 | 重做 Node 插件生态，收益暂不足以抵消工作量 | 当前不选 |
| 直接复制已安装 DSH 并改名打包 | 首次看似快 | 来源、补丁、版本和用户数据无法可靠区分，更新不可控 | 不作为发行方案 |

建议把选定 DSH commit 的源码作为 `harness/` 子树纳入产品仓库，保留上游目录结构和来源记录，之后维护自己的提交。用可审查的 subtree/vendor 同步引入上游变更，不在每次构建时拉最新主线，也不把已安装的编译产物当源码源头。

代码所有权分三层：上游机制层（Cordis、loop、事件/持久化基础），MoE4All 产品层（模型/资源管理、工作区、任务、设置和主界面），功能插件层（记忆、搜索、远程、语音等）。机制层可以修改，但要为每项差异保留动机和回归测试；产品层按自身需求设计，不要求永久跟随原版页面布局。

初期保留上游 package/service ID，避免全局改名破坏已有插件注入；产品品牌、启动入口和发行版本独立。只承诺验证过的插件版本集，不承诺所有 DSH 插件永久无修改兼容。若采用 DSH Desktop 源码，它是另一个独立上游，需要独立记录版本与差异。首版不同时替换桌面框架、重写整个前端和重写 Agent 循环。

## 5. 建议的运行架构

```text
Desktop Shell (可选，首版复用 Electron 路线)
  └─ Product Host / Node
       ├─ MoE4All harness（基于 DSH 源码）：agent、会话、工具、审批、Web API/UI
       ├─ 自有 engine-manager：模型配置、进程生命周期、准入策略
       ├─ 自有本地模型 provider / embedding 集成
       ├─ 受选插件；MCP 或有隔离需求的工具子进程
       └─ infr worker / Rust
            ├─ OpenAI 兼容推理 API
            ├─ 结构化管理与状态接口
            ├─ 模型执行、KV/session cache、MTP、并发调度
            └─ LLM + 受支持的 embedding/vision + 统一 GPU 资源

Browser ── 同一 Product Host 的 Web UI/API
Headless ── 同一 harness 的无界面入口
外部推理客户端 ── infr API（保留独立引擎运行方式）
```

### 5.1 唯一所有者

- Product Host 中的 engine-manager 是产品模式下 infr worker 的唯一生命周期所有者。
- Electron 只管理 Product Host，不再直接管理同一个 infr worker。
- 普通插件只能请求模型能力，不能各自启动 infr/ONNX 模型抢 GPU。
- 引擎持有 GPU 与推理资源；harness 持有 Agent 运行状态、任务和持久会话。
- Agent 会话可以在引擎重启后恢复；KV cache 丢失最多增加 prefill，不能丢失会话事实。
- 分进程不等于每个插件一个进程：受信任 Cordis 服务仍可同进程，隔离需求另行处理。

现有 `infr-gui` 的 worker 管理是迁移素材，不是第二个永久调度器。迁移时先保留旧 GUI 的独立运行模式；新产品不同时启动它来竞争同一 worker。新管理模块按现有 start/stop/drain 测试复现行为，功能覆盖后再淘汰旧入口。内存估算留在 Rust，不在 TypeScript 翻译一份公式。

### 5.2 需要补齐的三个接口

**推理数据面：** 继续使用 `/v1/chat/completions`、`/v1/embeddings`、`/v1/models`。先用已有 DSH 兼容适配器接通；只有能力发现、取消或协议差异确实需要时，再实现自有 `infr-provider`。

**引擎控制面：** 版本/构建标识、inspect/estimate、加载与 readiness、实际 capabilities、资源快照、队列/slot 状态、drain/shutdown。可采用版本化 stdio JSONL 控制协议：专用 worker 模式的 stdout 只写协议，stderr 写日志，HTTP 承载推理；加载前也能上报进度和错误。现有 `infr serve` 的输出和使用方式不变，不直接拿现有混合输出当协议。metadata-only inspect 可以先做独立 JSON 子命令。

**Agent 产品面：** 会话、任务、工具记录、审批、插件、工作区和事件订阅。沿用选定 DSH 版本的 Web API，不把它们塞进 infr 的 OpenAI API。第一版不再发明一整套平行的 Agent RPC。

控制协议至少带 `protocol_version`、`request_id` 和 worker 实例标识；启动、退出和迟到事件必须关联到同一实例，防止旧进程覆盖新状态。生命周期为 stopped/loading/ready/draining/failed，切模型先关闭新准入、排空旧请求，再退出并重新加载。端口由子进程实际 bind 后报告，不依赖“父进程探测空闲再释放”的竞态窗口。

### 5.3 capabilities 必须反映有效能力

不能仅凭“模型支持视觉/MTP”就让 UI 开启。应报告当前已加载组合的有效能力：模型 ID/指纹、实际上下文上限、实际并发数、tool calling、reasoning 参数、vision、embedding、MTP 及不可组合项。

例如当前 `cmd_serve` 中，Qwen3.8 MTP 路径序列化为单请求，并明确拒绝与 vision、同进程 embedding 组合。产品应如实显示这一限制，而不是把普通并发数或 embedding 可用状态原样沿用。

`ModelCapabilities` 与 `ResourceSnapshot` 是跨语言 DTO，不暴露 Vulkan buffer、KV 指针或 Rust 内部类型。Rust 内部配置与公开协议分开版本化，选择一种 schema 作为契约源并生成/校验 TypeScript 类型。

## 6. Harness 接入必须守住的约束

### 6.1 不改变已有 GPU 不变量

- 固定权重和固定资源（含 MTP）先真实分配，再查询设备真实剩余空间创建统一池。
- 长 prefill 期间不插入 decode，继续遵守当前环形缓冲区和专家重建约束。
- 受支持的原生 embedding/vision 复用既有 GPU 管理路径；不由记忆插件偷偷再加载一份模型。
- 不在每 token 热路径加入 Node RPC、数据库写入或插件回调；跨进程流做有界缓冲和背压，慢 UI 不能无限占用内存。

### 6.2 Agent 并发不等于 GPU 并发

多个 Agent 可以同时等网络或跑工具，但模型请求仍经过一个本地准入入口。第一版采用有界队列和简单优先级：用户交互优先；记忆整理、标题生成、定时任务低优先级，并通过老化避免永久饥饿。

准入策略作用于请求边界，不抢占进行中的长 prefill，不让一个子 Agent 独占一个固定 KV slot。引擎继续决定批处理和 slot 分配。已有服务端保护保留，但只有一个产品调度决策点，避免双重无限排队。

进程内存预算也要扩展：系统、Electron/Node、浏览器/MCP、工具子进程都需要空间。用户此前确定的普通/激进策略不能因为接入 harness 被悄悄改义；可以新增显式的“Agent 工作负载预留”，显示最终生效预算，并重新实测默认值。

### 6.3 Prefix reuse 是 Agent 体验的关键

工具 schema、系统前缀和静态说明采用稳定顺序；不要在最前方每轮写当前时间或随机内容。只向当前任务暴露需要的工具，避免把所有已安装插件的 schema 一次性塞入模型上下文。

优先复用既有 prefix cache。需要会话亲和提示时再加可选 request/session 标识，不把 Agent 会话与引擎 slot 绑定。压缩和消息编辑会使相应位置之后的前缀失效，这是正确行为，不能为了命中率保留已被用户修改的旧内容。

首轮必须做真实工具多轮测试：tool_call ID/参数、reasoning 历史、图片、取消、超时、上下文压缩和模型切换。OpenAI 兼容并不自动意味着这些边界都能正确往返。

### 6.4 恢复、审批与插件信任

直接复用 harness 的会话持久化与恢复，不在前端或 infr 再维护一套权威消息记录。任务运行 ID、工具调用 ID、状态和审批属于 Agent 层；KV cache、草稿验证状态属于引擎层。

对有副作用的工具，崩溃时“已执行但结果尚未持久化”可能无法判定，不能承诺 exactly-once 或自动无条件重放。需要幂等键、可检查的结果或人工确认；插件升级也不能静默重放这些调用。

Cordis 解决组合和生命周期，不是恶意插件的隔离边界。同进程 Node 插件有宿主权限；权限声明本身不能阻止直接读文件。只把审核过的插件纳入默认集，其他插件明确提示信任级别；需要隔离的执行经受约束的子进程/MCP/OS sandbox，不把 Worker Thread 当安全沙箱。上游也明确不应把其沙箱作为不可信工作负载的唯一防线。[安全说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md)

本地控制面默认 loopback 并鉴权；远程 UI/公网隧道作为用户显式启用的能力。远程会话必须同时支持审批、提问、取消和断线恢复；远程使用权限不等于插件安装、主机管理权限。稳定访问地址由正式配置或配对网关提供，不把本机网卡地址、portproxy 或固定端口补丁写成所有用户的默认值。发行包不携带个人密钥、历史、路径和数据库。

## 7. Rust 结构怎样调整

先建立职责边界，再移动目录。建议目标依赖方向：

```text
infr-cli ---------------------> infr-server (HTTP adapter)
    |                                |
    +----> infr-engine <-------------+  (generation contracts)
               |
               +-- runtime 装配 --> infr-llama/model + embedding/vision
                                            |
                                      backends + core + gguf

infr-chat：共享模型消息、模板、流与工具解析，不依赖 Agent 产品
Product Host：通过进程协议/API 使用 Engine，不链接 Rust 内部模块
```

具体顺序：

1. 将 CLI 的 commands、runtime adapters、terminal presentation 拆成内部模块；server 按 routes/DTO/streaming/stats/tests 拆文件。行为保持不变。
2. 将 `ChatGenerator`、`EmbeddingGenerator`、内部生成请求/结果/能力描述归入真正的引擎契约。HTTP DTO、HTTP 参数错误和 SSE 编码仍属于 server。
3. 复用现有 `infr-engine` 名称承接这些契约。把具体模型装配放在其可选 runtime 模块/feature 中，server 只使用无后端契约；独立 server 构建不应强制 Vulkan。若 feature 组合变得难维护，再按实际需求拆 runtime crate，不预先新增一堆空壳。
4. CLI 是装配入口，engine/runtime 不反向依赖 server；消除当前 adapter trait 由 server 拥有带来的不自然归属。
5. `infr-llama` 先保持路径，分清 model/load、session/cache、scheduler、MTP、backend adapter。更名是后续单独提交，不能混入执行行为修改。
6. core、Vulkan 等深层模块整理放在产品最小闭环之后。优先明确私有模块与 public API，而不是为了架构图拆包。

`infr-engine` 的旧聊天重导出可以暂时兼容；新代码直接依赖应有的模块，后续再删除兼容层和多余依赖。Cargo feature 并不提供运行时隔离，workspace feature union 也需在独立包构建中验证。

## 8. 仓库与数据目录

近期保留 Cargo workspace 根和 `crates/` 路径，避免一次移动所有源文件、shader include、CI 和测试引用。增加产品层，而不是先把引擎藏到新目录里：

```text
Cargo.toml / Cargo.lock
crates/                         # 现有 Rust engine/backend crates
harness/                        # 基于 DSH 源码的 TypeScript workspace
  apps/                         # 自有 Web/headless/可选 Desktop 入口
  packages/                     # 保留上游领域布局，逐步修改有明确归属的模块
  extensions/moe4all/            # engine-manager、provider、模型 UI、自有插件
  profiles/                     # 发行默认组合和可选功能集
  pnpm-workspace.yaml
  pnpm-lock.yaml                 # TS 依赖统一锁定；Rust 仍由 Cargo.lock 管理
patches/                        # 不直接维护源码的第三方依赖补丁
packaging/                      # 发行清单、锁定版本、第三方 notices
scripts/                        # 构建/测试/发布；按用途逐步分组
tests/                          # engine、protocol、agent-e2e 边界测试
docs/                           # 当前权威说明、设计决策、历史实验
dist/ target/ artifacts/ tmp/    # 构建或实验产物，不是产品源码
```

这不是要求立即创建所有空目录。先导入明确基线并加实际使用的 app、manager、profile；保留上游分包不等于为自有代码照搬同样细的粒度。自己的小功能先做 package 内模块，有独立依赖、测试或发布边界再拆包。也不再在仓库根另起第二套 TS apps/packages 和锁文件。

用户数据另放 `%APPDATA%`/`%LOCALAPPDATA%` 下明确的产品 home，或用户主动选择的 portable data 目录。程序资源、用户配置、会话/记忆数据库、缓存、插件覆盖层、模型文件路径分开。首次迁移从 DSH 导入时使用副本和 schema/version 检查，保留原安装和数据；绝不让两个版本并发写同一数据库。

### 8.1 已证实的清理对象与不能直接删除的内容

- 可优先消除：engine 的空 facade 职责、多余依赖、明显过期模块说明、CLI/server 内部混合职责。
- 可逐步归档：根目录专题报告以及历史性能实验，移动时同步所有链接和证据引用。
- `docs/` 37 个 tracked 文件、`infr-fork-wiki/` 41 个 tracked 文件，经文件哈希检查没有字节完全相同的副本。Wiki 明确记录 fork 历史，和现行手册主题有交叉但用途不同。建议统一导航、标记 current/history，再决定归档或由一处源生成发布页面。
- 对 README、GETTING_STARTED、docs 索引检查到的本地 Markdown 链接未发现缺失目标；主要是内容时效问题，不应描述为整个文档目录已坏。
- `target/dist/artifacts/tmp/gui-data` 不能统称废物：有构建产物、实验数据和用户状态。先做 tracked/ignored/引用清单，产物统一忽略，证据保留索引，数据迁移单独处理。
- 小型 profiling/test crates、跨平台后端、失败实验记录不能因为当前 Windows 路径没直接用到就删除。

## 9. 插件与发行策略

第一版默认集控制在完成工作所需范围：文件读写/搜索、PowerShell 执行、审批、任务记录、会话恢复、模型管理。记忆、Web 搜索、定时任务逐项通过测试后预装；TTS、远程访问等作为可选集。已安装不等于把所有工具默认暴露给每个 Agent。

插件至少区分三种契约：工具能力、harness 扩展、UI 扩展。Rust compute backend 不纳入热加载插件市场。MCP 可作为外部工具协议，但不能替代会话、任务和 UI 插件协议。

发行清单应记录产品版本、引擎 commit/构建、harness 精确版本、插件精确版本与校验、补丁、Node 版本、协议版本和数据 schema 版本。内置基线只读，用户覆盖层可回滚；默认插件更新随产品验证，不在启动时无条件升级所有 npm 依赖。

DSH/DSH Desktop 根项目声明 MIT，所查第三方包同时存在 MIT 和 Apache-2.0 声明。正式发包按实际依赖生成 notices/SBOM 并核查分发要求，不能把整个产品一概视为单一许可证。[DSH Desktop 项目](https://github.com/dataelement/dsh-desktop)

现有 [`Windows workflow`](../.github/workflows/release-windows.yml) 构建的是 `infr-cli`；[`打包脚本`](../scripts/package-windows.ps1) 也是引擎和向导包，并未形成 Agent/GUI/Node/插件的一体化发行物。应保留该轻量包，另加 Agent 打包入口，而不是让纯引擎用户必须安装整个桌面栈。

## 10. 分阶段落地与验收

### A. 固化已知可用基线

以 `D:\AISuperAssistant\DSH` 的补丁、说明与源码审阅目录为迁移证据，分出配置、正式功能、版本兼容和调试材料；选择 DSH 源码 commit 并建立自有 fork 构建。当前安装版本是行为参照，不要求继续锁死在旧版；较新基线要通过工具/会话/插件验证后再确定。

验收：干净数据目录能启动同样的受选能力；不依赖现有安装目录中的隐含修改。

### B. 先跑通最小 Agent 产品

在已导入源码中增加自有 profile、engine-manager 和模型管理入口；保留 loop 与可用 UI 组件，先用通用兼容 provider 连接 infr，打通本地工具多轮。提供显式启动/停止、模型选择、失败提示和运行状态。随后按产品需要改造主界面，而不是发布两个并排的管理页面。

先做这一纵向闭环，不以“全部 Rust 目录重构完成”为前提。prototype 可以暂沿用现有 worker 的 health/停止机制，但不能把日志抓取固化成正式控制协议。

验收：启动产品后自动管理一个引擎；在工作区执行“读文件、修改、运行测试、汇报”；中途取消能释放请求；重启后恢复会话；纯引擎入口仍可用。

### C. 收敛引擎契约与管理接口

按第 7 节整理 Rust 入口，补结构化加载/资源事件、inspect、capabilities 和管理协议；把旧 GUI 模型能力迁入新产品，不重写内存公式。锁定新旧模式行为一致性，之后才移除旧入口。

验收：UI 不依赖日志措辞；实际并发/上下文/MTP/视觉/embedding 状态一致；切换模型完整 drain；worker 崩溃或 host 退出后无失控子进程；独立 server/backend feature 构建通过。

### D. 加入经过验证的产品能力

接入记忆的原生 embedding、后台/定时任务、子 Agent 与插件设置，建立任务优先级、预算、工作区作用域和升级/恢复流程。增加离线可启动的完整 Agent 包；需要网络的服务单独标识，不冒充离线能力。

验收：前台任务与记忆后台作业竞争时可控；插件失败能降级/安全模式启动；升级可回滚且不损坏会话；不复制用户密钥或数据库进发行物。

### 贯穿所有阶段的测试

- 引擎：保留当前正确性和单路/并发/MTP 固定基准，不能因产品化引入明显吞吐回退。
- 协议：工具流参数跨分片、reasoning 回放、finish/usage、取消、超时、模型切换、视觉能力组合。
- Agent：真实任务成功率、端到端完成时间、每轮 prefill/cached tokens、工具重试、上下文膨胀，而不仅是 decode tok/s。
- 生命周期：工具执行中断线/崩溃、切模型、加载失败、端口冲突、插件升级、后台任务恢复。
- 资源：引擎、Host、Electron、工具/MCP 的总 RAM/VRAM，尤其是低余量机器和长上下文。

## 11. 建议先确定的决策

当前建议以“基于 DSH 源码的 MoE4All Agent 发行版”为主线，以完整 Web Agent 为第一交付、Desktop 为同一产品的封装；优先个人本机/单用户，远程能力保留明确边界与验收路径。

近期真正需要做的是：**整理已经找到的个人改造，建立可构建的 harness 源码基线，打通产品级引擎管理，然后逐步整理 Rust 入口。** 源码 fork 可以做较深修改，但不意味着先推倒整个引擎或重写所有 Agent 基础机制。

仍需产品决策的事项：是否要求现有 DSH 插件原样兼容；首发必须具备哪些个人定制；正式公开前是否必须带桌面安装器。这些会影响包装和迁移工作量，但不改变“harness 与引擎分层、单一资源所有者、可独立使用”的主架构。
