---
kind: change-history
status: historical
scope: tiered-weight-paging
preserved_in_tag: release-0.9.0
---

# 分层权重分页：显存 → DRAM → 磁盘

> 历史实现计划与阶段记录。文中的“当前”“尚未实现”按写作时点理解；0.9.0 资源与分页现状以[运行时资源生命周期](../../../architecture/memory/runtime-resource-lifecycle.md)和源码为准。性能数字见各自的[历史证据](../../README.md)。

本计划面向既放不进显存也放不进 DRAM 的模型：将现有块分页器扩展为**分层**分页器，其底层直接是模型文件本身，使用显式定位 I/O 读取，而不是交由 OS 页缓存处理。

状态：**第 0–3 阶段已落地，且该层目前在两个后端上均优于 mmap**（CPU 在 1.5 GB 上限时为 2.06 倍；Vulkan 在 8 GB 上限、7 GB 内存区下的解码为 2.17 倍，见 `documentation/evidence/benchmarks/2026-08-03-validated-models.md`），并且在模型放不下时会自行启用（§4.1），而不是等待无人能够猜准的预算。第 4 阶段尚未构建；第 5 阶段已引入三个杠杆：并发读取器、准入把关器和自动定容。§5 说明每个阶段交付与推迟的内容；§7 保留仍未决的问题，包括是否应在无法运行它的机器上尝试第 4 阶段。关于代码树的论断会列出其来源文件与符号。数值来自实际运行的命令，而不是估算；§2 唯一的演算示例标为说明性算术，而非预测。

## 1. 当前已有内容

驻留机制已经与块类型无关，并且已有两种策略：

-- `infr_core::pager::Pager`：纯主机侧簿记（`n_slots` 个槽位、`BlockId → slot` 映射、LRU 顺序、批处理 epoch）。不持有字节，也不涉及设备类型。三个入口为：`touch`（新近度/LRU、MoE 解码）、`touch_cold`（抗扫描、预填充扫描）和 `schedule`（精确循环扫描、Belady 对等性：稠密层流式传输）。`ring_bytes` 为暂存 ring 计价。
- `infr_vulkan::pager::GpuPager`：`Pager` 加显存槽位内存区（BDA）、设备 LUT，以及经由复用固定暂存 ring 的上传。`MoePagerSession` 用 `(layer, role, expert)` 块驱动它；`DensePagerSession` 用逐层权重组块（`DenseSource`、`DensePoolSpec`）驱动它。
- 两者的**字节来源**原本均为零拷贝 GGUF mmap 视图：`DenseSource::segments` 是 `Vec<Arc<dyn AsRef<[u8]> + Send + Sync>>`，由 Vulkan 绑定器（`infr-llama/src/seam/mod.rs`）对每个组件张量调用一次 `Gguf::tensor_bytes_arc` 构建。第 3 阶段将此字段改为 `DenseBytes`，并将 `ExpertSource::bytes` 改为 `ExpertBytes`，其第二个分支改从主机内存层读取（§3.7）。
- 放置策略在接口层每次加载时确定一次：先是 MoE 分层阶梯，再是稠密的“尝试常驻 → 更小 ubatch → 自动 q8 → 流式”阶梯，并以 `Backend::device_alloc_room` 计价。
- CPU 后端从不复制权重：`CpuBuffer::Mapped(TensorBytes)` 直接从映射读取（`CpuBackend::map_weight`）。
- Metal 后端**完全没有分页器**，也不使用 mmap：`alloc`/`upload` 将每个权重放入 `StorageModeShared` `MTLBuffer`，即固定的匿名内存。当前 Apple silicon 上超过 RAM 的模型完全无法加载。

因此缺失的恰好是一层：**DRAM 以下**。其上的策略层已存在且已有单元测试。

### 1.1 当前“DRAM 层”的含义及其不足之处

当前 DRAM 层是通过 GGUF mmap 访问的 OS 页缓存（`Gguf::open` 映射文件并 `madvise` `WillNeed` + `HugePage`）。它已经可以运行大于 RAM 的模型：内核按需分页。因此此功能**不是**“使之可运行”，而是“使其不发生抖动”：

1. **错误的驱逐策略。**页缓存基于新近度。稠密前向过程会循环扫描整个权重集，这是 LRU 的病态情形：每一页恰在下一次使用之前被驱逐。`Pager::touch_cold`/`schedule` 正是为修复同一病态而存在（相关文档记录了代码树内测量：普通 LRU 下 Scout pp512 每次重复上传 768/768 个块）。我们可在每层应用 Belady 对等策略；内核无法这样做，因为它不知道扫描顺序。
2. **没有我们可控的预取。**内核预读是基于文件偏移量的顺序启发式。对于稠密模型，我们知道整个过程的访问顺序；对于 CPU 上的 MoE，知道提前一层的顺序（路由器运行在主机上），所以可在停顿前而不是缺页后发起读取。
3. **没有核算。**没有任何机制能回答“我的权重工作集有多大比例处于常驻”，因此放置阶梯在显存以下是盲目的，用户也得不到运行成本的诚实报告。

第四个原因与类别无关，而是现有停顿的程度：`DensePagerSession::stage` 会在持有 session
mutex 时将 mmap 字节 memcpy 到 pinned ring（见 `infr-vulkan/src/adapter.rs` 中的
`stage_dense_linear`）；`schedule_staged` 的复制也会在同一个 `std::sync::Mutex` 下进入
rayon 并行区段。发生 major fault 时，线程**本来就会**持锁等待磁盘。显式读取不会凭空制造
这类停顿，只会延长它；但与缺页不同，显式读取可以通过预取移出关键路径。之后的 fence wait
位于锁保护区之外，因此提交期间不会一直持有该锁。

**阶段 0：验证前提，已完成。**`scripts/paging-baseline.py` 会先清除模型的 page cache，再在
cgroup-v2 `MemoryMax` 限制下运行 CPU 后端，确保权重确实放不进进程可用内存。完整数据见
`documentation/evidence/benchmarks/2026-08-03-validated-models.md`。以 Llama-3.2-1B F16
（2.48 GB）为例：

- **Decode 慢了 23～33 倍**：内存上限生效后，速度从不限内存时的 22.5 t/s 降到 2 GB 时的
  0.96 t/s、1.5 GB 时的 0.67 t/s；每次生成 32 个 token 会触发 42～46 万次 major fault。
- **Prefill 基本不变**：速度从 46.9 降至 46.6 t/s（−0.6%），整轮只读取 3.7 GB；一次权重
  扫描由 512 个 token 分摊。
- 1.5 GB 限制下，Decode 生成 32 个 token 共读取 **153 GB，相当于每个 token 读取 1.9 倍
  模型大小**；完全不缓存本来只需读取 1.0 倍。对循环扫描采用 recency eviction，再叠加
  4 KiB 粒度和 readahead，实际代价几乎是干脆放弃缓存的两倍。

最后一项数据给出了目标：如果某一层每轮传输量为 `model − VRAM_home − DRAM_home`，并以
完整 block 为单位发起 I/O，那么在计算任何 hit-rate 收益之前，传输字节数就已不到上述方案的
一半。基线的缺口也随数据一并记录：这台主机没有 GPU 侧数据，也没有真正大于 RAM 的模型文件。

### 1.2 已确立的约束先例

Backlog **B30** 记录了一项被否决的实验：将 GGUF 复制到 anonymous mapping。在 16 GiB
Qwen3.6-27B、warm cache 条件下实测，加载时间从 1.87 秒升至 10.5 秒（5.6 倍）；原本可回收
的 14 GiB page cache 变成了 20.2 GiB anonymous RSS。

- **Anonymous、不可驱逐的主机内存代价高。**我们的 DRAM arena 正是这类内存，因此必须明确
  设定上限；除非放置计划判定模型无法装入，否则默认关闭。
- **必须完整保留 mmap 快路径。**模型能够装入时，继续使用 zero-copy mmap 视图，不额外付出
  arena、复制或 worker thread 的成本。

## 2. 物理限制：能与不能实现的目标

每次前向过程中，每个 block 都分配到唯一的常驻层级（§3.5）：

```
disk→host bytes = model − VRAM_home − DRAM_home
host→VRAM bytes = model − VRAM_home                    [discrete GPU only]
t_pass         ≈ max(disk_bytes/disk_bw, host_bytes/pcie_bw, t_compute)
```

两个层级边界需要传输的字节数不同，因此单轮耗时由较慢的那次传输决定，不能用一个比例概括。

稠密模型没有可利用的局部性：每个 token 都恰好读取一次全部权重，因此 Decode 吞吐上限就是
上述传输速度除以每个 token 的数据量。以下使用整数量级作说明（**不是性能预测**；实际
`disk_bw` 由阶段 0 测得，`VRAM_home` 则要扣除 KV、`dense_act_reserve_at` 计算的 activation
reserve 和 staging ring，不能直接使用显卡标称容量）：一个 40 GB 模型若有 20 GB 权重常驻
VRAM、8 GB 常驻 DRAM，则每个 token 仍需从磁盘流入 12 GB；磁盘速度只有几 GB/s 时，生成速度
远低于 1 token/s。**这才是诚实的上限，也是设计要解决的问题**：让原本“无法运行”的模型
变成“可以运行”。同一批数据可以由整个 Prefill chunk 分摊，因此 Prefill 仍可能实用，但交互式
Decode 不会快。§7 据此作出设计决策。

阶段 0 的基线从另一侧验证了同一结论：在严格内存上限下，Prefill 只慢 0.6%，Decode 却慢了
23～33 倍，因为只有 Decode 需要为每个 token 承担一次完整权重扫描（§1.1）。

MoE is the opposite case and the real prize: routing is skewed, so a small hot
set carries most tokens and the in-tree MoE pager already runs at a high
steady-state hit rate on Scout prefill (recorded in the pager campaign notes;
phase 0 re-measures it rather than carrying the figure forward). A DRAM tier
below it only has to cover the cold tail. The dense parts of a MoE model
(attention, norms, shared expert, router, embeddings, lm_head) stay fully
resident — small, and hit every token.

## 3. 架构

### 3.1 模块布局

Two new modules in `infr-core`, and **no reorganisation of what exists**:

- `infr-core/src/blockio.rs` — `BlockExtent`, `BlockDesc`, the `BlockIo` trait,
  `FileBlockIo`, and the reader pool.
- `infr-core/src/hostpager.rs` — `HostPager`: the DRAM tier (arena, pins,
  prefetch queue), built on today's `infr_core::pager::Pager`.

`pager.rs` stays where it is. Folding all three into a `paging/` directory is
churn until there is a third reason to; do it then, with re-exports.

Both modules stay device-free: no `ash`, no `metal`, no GGUF types. That is what
lets three backends share them and what keeps the policy unit-testable without
hardware, as `pager.rs`'s own test module already is.

### 3.2 块模型

A **block** is one unit of paging: exactly what the seam's `wload` closure
(`infr-llama/src/seam/runner.rs`) uploads as a group — a fused qkv triple, a
fused gate+up pair, a single projection, one MoE expert's one role. That
definition already exists on both sides and keeping it is what stops the plan
and the loader drifting.

```rust
/// One contiguous byte range of the model file.
pub struct BlockExtent { pub offset: u64, pub len: usize }

/// A block's identity and where its bytes are, in upload order. A fused group
/// lists one extent per component tensor.
pub struct BlockDesc { pub id: BlockId, pub extents: Vec<BlockExtent>, pub nbytes: usize }
```

`TensorInfo` already carries `offset`/`nbytes` and `Gguf` knows
`data_region_start`, so the extents are derivable — but `Gguf` keeps only the
`Mmap`, not the path or the `File` (`Gguf::open` drops the descriptor). So
`infr-gguf` must **gain** a retained path (or `File`) plus an accessor returning
`(absolute_offset, len)` for a named tensor. Single file only: `Gguf::open` maps
one file, and sharded-GGUF support is deferred (§7).

**Blocks that cannot be re-read from the file.** The seam rewrites some tensors
at load — the qwen2 NEOX q/k row permute, the BitNet `I2S` → f16 dequant — and
the Vulkan dense streaming path already excludes them for that reason. It also
它还排除内核不接受权重偏移的 dtype（`native_dense_supported`；F16/F32 不在其中，因此当前 f16 checkpoint 在 Vulkan 上**没有**可流式传输的稠密权重）。两项排除都是分层规划器的条件，而非事后补丁：资格谓词从 Vulkan 专用的 `seam/mod.rs` `dense_plan` 块中移出，放到 `fuse_gu_decision` / `fuse_qkv_decision` 旁边；其他共享枚举规则也在此处，因此 CPU 与 Metal 规划读取同一谓词。

**融合组拼接。**当前 `wload` 将每个多名称组具体化为 `WBytes::Owned(Vec<u8>)`，而 Vulkan 流式绑定器随后忽略这些字节，重新获取逐组件 mmap 视图：拼接被构建后立即丢弃。分页块完全不应构建它，因此 `wload` 需要“该组已分页”分支，在不拼接的情况下计算组的字节总数（供现有漂移防护使用，该防护将计划段总量与 `tb.len()` 比较）。

### 3.3 固定：核心 `Pager` 所需的变更

两个新增消费者都会在分页器不可见的工作期间持有块字节：CPU 内核在一个操作期间读取完整权重；GPU 暂存拷贝在记录拷贝前读取主机槽位。借用期间槽位必须不可驱逐，批处理 epoch 无法表达这一点（epoch 按批处理，而借用按操作）。

这比“为 `Pager` 增加两个方法”是更大的改动：

- 为 `pin(id)` / `unpin(id)` 增加引用计数，并让 `take_slot` 跳过已固定条目。
- `take_slot` 的“所有槽位均不可用”路径当前会**panic**（批内断言）。固定耗尽在运行时可达，因此必须返回错误；而 `touch`/`touch_cold`/`schedule` 返回 `Resolution` 而不是 `Result`。使耗尽可恢复会改变三个公共签名及 `infr-vulkan` 中的所有调用方。应为此预留工作量，或者保留耗尽 panic，并将**定容下限**做成运行时无法违背的加载期检查。

**定容下限不是每次过程，而是每次过程 × 并发度。**`infr-server` 通过一个后端准入 `n_parallel` 个并发生成（`slots: Arc<Semaphore>`），每个生成都位于各自的层。因此下限是 `n_parallel × (一个操作固定的最大块数)`：稠密模型为一层权重组，MoE 为每层 top-k 专家。`TierPlan` 对此计价，无法覆盖的预算会在加载时以配置项名称明确拒绝。跨 N 个请求增量阻塞地获取无序固定集合会造成死锁，因此预算紧张时回退为分页前向过程仅一个许可证（串行化），并在横幅中说明，绝不等待。

**内存区需要别名安全论证，而不是生命周期技巧。**`HostPager::pin(&self) -> Pin<'_>` 向外提供位于内存区中的 `&[u8]`，而其他线程可通过同一个 `&self` 修改该内存区。这是内部可变性，具有真实的健全性义务：槽位存储放在 `UnsafeCell` 中；不变量是“已固定槽位永不写入，且只有拥有槽位的读取器会在固定存在前写入它”；还要有实际**强制**该不变量的机制（在分发槽位的同一把锁内检查固定引用计数），而不是只写注释。第 1 阶段对该模块运行 Miri：工作区已有针对 `SpinPool` 的每周 miri 任务，这正是它存在的代码类型。

```rust
pub struct Pin<'a> { /* Deref<Target = [u8]>; Drop → unpin */ }
impl HostPager {
    pub fn pin(&self, id: BlockId) -> Result<Pin<'_>>;      // blocking: may read disk
    pub fn try_pin(&self, id: BlockId) -> Option<Pin<'_>>;  // hit-only, never reads
    pub fn prefetch(&self, ids: &[BlockId]);                // queue; never blocks
}
```

### 3.4 按地址键控的缓存：静默损坏陷阱

代码树内有三个缓存以**权重切片地址**为键；该地址仅因权重被 mmap 或上传一次才稳定。分页槽位会为等长但不同的块复用一个地址，因此这些键发生碰撞并返回另一块的数据：看似合理的垃圾，而不是错误：

- `infr-cpu`'s `repack_cache` / `repack6_cache`: `q4k_pack_for` / `q6k_pack_for`
  key on `(w.as_ptr() as usize, w.len())`. **Must** be re-keyed on `BlockId`
  before any CPU paging lands.
- `infr-cpu`'s `weight_cache` is safe as written — it keys on the `CpuBuffer`
  object address, not the slice — but it caches a dequantized f32 copy, so it is
  a budget consumer, not a correctness one.
- `infr-metal`'s `qui_cache` keys on `MTLBuffer::id()`, so a placeholder per
  block keeps it correct; but its factored arm **copies the transformed weight
  out** and retains it unboundedly. On a paged Metal model that is a second full
  copy of every touched block in host RAM, which defeats the whole budget. The
  Metal tier is therefore gated on the native-kernel arm (Q4_K/Q6_K read the
  bound buffer directly), with the factored path either bypassed or budgeted.

每一个都是键中编码了分页会破坏的假设的缓存。审查“还有什么以地址为键”属于第 1 阶段，而不是后续清理。

### 3.5 I/O 引擎

```rust
pub trait BlockIo: Send + Sync {
    /// Fill `dst[..desc.nbytes]` from the block's extents, in order.
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()>;
}
```

`FileBlockIo` 使用定位读取：`std::os::unix::fs::FileExt::read_at` / `std::os::windows::fs::FileExt::seek_read`，因此没有共享游标，也没有工作进程之间的 seek/read 竞态。无需新增依赖。固定工作进程池服务预取队列；线程数与预取深度在测量要求配置项前均**硬编码**（§4）。

`infr-testkit` 中带故障注入（短读、错误、延迟）的简单内存内 `BlockIo`，使每项分层断言都可验证失败。

**有意不纳入 v1**，记录在此以免再次提议：io_uring（仅 Linux、新依赖，且其收益由队列深度与线程已提供）；`O_DIRECT`/`F_NOCACHE`（对齐约束，且只要页缓存有益就会落败）；映射文件后从中 memcpy（仍是页缓存，且锁内缺页问题仍存在）。

**双重缓存确实存在。**缓冲 `read_at` 会在页缓存中留下我们也保存在内存区内的数据副本；对于远大于 RAM 的模型，这会使有效 DRAM 减半。`posix_fadvise(DONTNEED)` / `F_NOCACHE` 是第 5 阶段杠杆，受第 0 阶段计数器约束。

**文件可能在我们运行期间变更。**`Gguf::open` 将此记为未强制的不变量，而 `infr_gguf::watch::WeightWatch` 仅在 CLI/请求检查点检测它（积压项 B30）。运行时重新读取使问题比映射更严重：截断导致短读，重写则在生成中静默产生不同字节。最低处理方式是：`FileBlockIo` 在打开时从自身 fd 记录 `(len, mtime, ino)`，并**每次前向过程一次**（而非每次读取）重新 stat；发生变化时明确使生成失败。这每个过程只需一个 syscall，却将静默错误输出转为错误。

### 3.6 分层策略：按模型类别选择的两种形态

**稠密：分区，不缓存。**对于循环扫描，在两层缓存同一块是浪费：常驻显存的块永远不需要 DRAM 副本。因此 `TierPlan` 在加载时为每个块分配一个归属层：显存归属层获取预算扣除流式窗口后容纳的块，DRAM 归属层获取剩余中预算可容纳的块，其余每次过程都按现有 `schedule`（冷插入、Belady 对等性）策略从磁盘流入这些窗口。每次过程的磁盘流量为 `model − VRAM_home − DRAM_home`，是任何策略在扫描中能达到的最小值。

**该分区的 DRAM 部分已落地，且不依赖 `TierPlan`。**`HostPager::fill` 通过首次触及而不是计划达到同一效果：仅在存在空闲槽位时接纳块，内存区满后直接读入调用方缓冲区，绝不驱逐。在循环扫描中，这**就是**分区：首轮填满内存区，以后的每轮都发现同一集合常驻；扫描顺序已提供计划，故无需计划。真正 `TierPlan` 的新增价值是选择**哪些**块获得 DRAM 归属，而不是接纳先到者；这只在块大小或访问频率不同时有意义。测得在 Vulkan 路径上相较“驱逐并缓存”形态有 1.6 倍价值（§5，第 3 阶段）。

**MoE: cache, and fill from the read path only.** Routing is skewed and
unpredictable, so both tiers cache with the existing policies (`touch` for
decode, `touch_cold` for prefill sweeps). The DRAM tier is filled **only** from
disk reads — never by copying a block back from VRAM, which spends a device→host
transfer on bytes that can be re-read from disk instead. **LANDED** on the
Vulkan MoE session: both entry points take `HostPager::pin` (the evicting
shape), and each hands the tier below the same insertion policy it is using
itself, so the two tiers cannot disagree about which experts are hot.

**Prefetch:**

- Dense, any backend: the order is known for the whole pass; issue block `l+k`'s
  read when layer `l` starts.
- MoE on CPU: the router runs on the host, so layer `l`'s ids are known before
  its FFN executes — prefetch at the router, execute after.
- MoE on GPU: ids arrive by device→host readback per layer (the paged `MoeFfn`
  arm in `adapter.rs`), so the next layer's ids do not exist yet and there is no
  exact prefetch to be had. v1 does nothing clever here; frequency-warmed
  promotion is a phase-5 lever, not part of the architecture.

### 3.7 后端集成

`WBytes` (`seam/mod.rs`) gains a `Paged(BlockDesc)` variant — and **loses its
`Deref<Target = [u8]>` impl in the same change**. The `Deref` is infallible, so
leaving it forces a panic arm and silently keeps compiling at the sites that use
it (`pipeline_binder`'s `pad_to_u32_align(&tb)`, `tensor_parallel_binder`'s
`tp_slice_column(&tb, …)` / `tb.to_vec()`). Replacing it with
`fn bytes(&self) -> Result<&[u8]>` makes the compiler enumerate every site that
must now handle a paged block. A binder that receives `Paged` registers the
block with its backend's pager instead of allocating and uploading — what the
Vulkan dense binder already does via `DensePagerSession::register`.

**CPU (`infr-cpu`).** New `CpuBuffer::Paged` beside `Mapped`/`Owned`, and a
`CpuRead::Pinned(Pin<'_>)` variant so the existing `Deref<Target = [u8]>`
uniformity carries every kernel unchanged. `CpuBuffer::read` returns no
`Result`, so the disk error must surface **before** it: the interpreter pins
every weight an op names in a fallible pre-step, then executes with `read()`
infallible over already-pinned slots. That pre-step is also where prefetch for
the next op is issued. This is where tiering matters most — today the CPU
backend's only answer to "bigger than RAM" is the page cache.

**Vulkan (`infr-vulkan`) — LANDED.** `DenseSource`'s segments became
`DenseBytes`: `Mmap(segments)` (fast path, unchanged) or `Host`, which reads the
block from the pool's own `HostPager` (`DensePoolSpec::host`). One host pager
per VRAM pool, because a pool is already exactly a block-size class — the
uniform-slot shape the host tier needs — so both tiers name the same blocks by
the same `block_id`, with no mapping table between them. `stage` has the three
cases: VRAM hit (nothing copied); VRAM miss with DRAM hit (memcpy arena → pinned
ring); VRAM miss with DRAM miss (`HostPager::pin` reads the model file, then the
same memcpy). The pin is released before `stage` returns, so one host slot per
pool is enough for a sweep to make progress. The `-DSTREAMED` shader twins are
**not** a new cost: `build.rs` already makes the streamed form the sole weight
build.

**What that gives up, and why.** The plan originally had the double-miss read go
**straight into the pinned ring**, to avoid a second host touch. It is not
built, because the saving only exists for a block that will never be re-read:
reading into the arena and memcpying costs one read plus one memcpy, and reading
into the ring and admitting afterwards costs exactly the same — the copy
disappears only if you **skip** admission, and which blocks may be skipped is
the `TierPlan` disk/DRAM partition (§3.6), which is deferred. Against that, the
pinned ring is write-combined device-local memory on a ReBAR host, so a `pread`
into it has an unmeasured cost a memcpy out of cached DRAM does not. Phase-5
lever, gated on a measurement, once there is a partition that makes it
meaningful.

**Vulkan MoE (`MoePagerSession`) — LANDED.** The same shape, one tier lower down
the same file: `ExpertSource::bytes` became `ExpertBytes::{Mmap, Host}`, and a
`Host` bank registers **one block per expert** in the pool's `HostPager`, at
that expert's own file offset inside the bank, under the same global
`layer_base + local_id` the arena already keys on. That per-expert granularity
is the whole point — a routed miss reads ONE expert off the file instead of
faulting in a whole bank through the mapping, which is what §2 means by MoE
being the tier's best case. Both entry points carry it: the demand path
(`touch_role`, decode's routed readback) and the recorded path (`stage_role`),
each passing the tier below the SAME insertion policy it uses itself — cold for
a full-set prefill sweep, MRU for a routed decode touch — because §3.6 says MoE
caches rather than partitions, so the two tiers should agree on what "hot"
means. `register` checks every one of a layer's experts is present below, so a
bank that was mis-registered fails at load rather than on the first routed miss
that names the missing id.

**Metal (`infr-metal`).** Unified memory collapses the VRAM and DRAM tiers into
**one**: an arena slot is host memory the GPU reads directly, so a disk read
lands in the final destination with no staging ring and no second copy. Slot
offsets need no new kernels — `set_buffer` takes a per-binding byte offset and
`exec.rs` already binds weights at non-zero offsets; weights bind to `device`
address space, so the only alignment obligation is the 4-byte one the encoder
path already checks, and `slot_bytes % 4 == 0` (which the Vulkan arena already
requires). The real Metal work is the arena + `Pager` + `read_block` into
`contents()` at the slot offset, plus the `qui_cache` gate from §3.4. The same
single-tier collapse applies to **UMA Vulkan** (APU / iGPU / Strix Halo).

**Out of scope, stated so it is not assumed:** the multi-GPU wrappers
(`tensor_parallel_binder`, `expert_parallel_binder`, `pipeline_binder`) bypass
the pager entirely today — EP keeps experts resident per rank — and a host tier
under N devices needs its own budget story. MTP declares its own `BindWeight`
and loads a second weight set, which is a second unbudgeted claim on the tier.
Both keep the mmap path until a later slice.

### 3.8 放置

`TierPlan` is pure arithmetic in `infr-core`, fed by the seam: the block list
with sizes, the VRAM ceiling (`Backend::device_alloc_room` — the measurement
that already outranks estimates), the host DRAM budget, the KV and activation
reserves the ladder already prices, and `n_parallel`. It returns each block's
home tier, the per-tier window sizes, and the numbers for the banner.

The existing ladder gains one rung at the bottom, and the banner states per-pass
disk bytes and the throughput ceiling they imply — so a user sees the number
before waiting for it.

**Host DRAM budget.** There is no host-memory probe in the tree. Explicit
`paging.dram` always wins; otherwise probe via the existing `libc` dependency
(`sysconf(_SC_PHYS_PAGES)`/`_SC_AVPHYS_PAGES`, `sysctl hw.memsize` + `vm_stat`).
Where no probe exists — Windows, which CI does not build (B30) — an explicit
budget is required and the absence is reported, never guessed. See §7.

## 4. 配置

Two new keys in the `paging` section of `infr-core/src/config/manifest.rs`,
following the spellings already there (`paging.cache`, `paging.ring`,
`paging.stats`):

| env                | key                  | type | meaning                                                          |
| ------------------ | -------------------- | ---- | ---------------------------------------------------------------- |
| `INFR_DRAM_CACHE`  | `paging.dram`        | Size | DRAM tier budget — unset = automatic, `0` = off, `N` = pinned    |
| `INFR_DRAM_BYPASS` | `paging.dram_bypass` | Flag | No host cache: blocks read disk → GPU memory (the unified shape) |

**LANDED with three states, not two.** Unset means SIZE IT AUTOMATICALLY (see
below); a value pins the arena and wins over every automatic rung; and `0` turns
the tier off entirely. The third state is not decoration — once unset stopped
meaning "off", there was no way left to A/B the tier against the mmap path it
replaces, and `scripts/paging-baseline.py`'s baseline arm had silently become a
second paged run until it was fixed to pass `paging.dram=0`.

**`INFR_DISK_STREAM` / `paging.disk` was never built and is withdrawn.** It
would have been a flag asking the user to say something the placement ladder
already knows: both Vulkan tiers are only reached once residency was REJECTED,
and the CPU backend can compare the pageable bytes against available RAM itself.
Turning the tier on is therefore a consequence of the model not fitting, not a
switch.

**`INFR_DRAM_BYPASS` is how the unified shape is tested on hardware that is not
unified.** A unified device takes that path automatically (below); this flag
forces it anywhere, which is the only reason it has any coverage at all — the
machine this was built on has a discrete GPU. It is also a legitimate setting in
its own right: a host whose RAM is better spent elsewhere still wants
block-granular reads rather than the page cache.

**`INFR_CACHE` (`paging.cache`) keeps its meaning** — the VRAM paging budget —
and is the other half of the testing story: capping it is what forces residency
to be rejected on hardware that would otherwise hold the model, and
`INFR_DRAM_CACHE` then pins what sits underneath. Every figure in
`documentation/evidence/benchmarks/2026-08-03-validated-models.md` was taken that way on a card with room to spare.

Reader threads, prefetch depth and page-cache dropping are **hardcoded** until a
measurement says otherwise — a knob with no known good value is a question
shipped to the user. `INFR_PAGER_STATS` grows per-tier lines.

### 4.0 统一内存有两层，而非三层

On an iGPU or APU the streaming arena is already GPU-accessible host RAM. A host
cache beneath it would hold a second copy of the same bytes in the same RAM,
readable only by the CPU — every hit still copied through the staging ring,
while the identical bytes spent on the arena above are read in place. So the
ladder there is `DISK → GPU-accessible RAM`, and what sits at the bottom is a
READER, not a cache: `HostPager::stream_only`, whose every `fill` is a
positioned read straight into the ring. That is what the arena above needs in
order to hold a model far larger than the machine, and it is strictly better
than the mmap path it replaces for the same reason the discrete tier is — the
page cache evicts by recency, which is the pathological policy for a cyclic
sweep.

**Unverified on unified hardware**: none was available. What IS verified, on a
discrete GPU via `INFR_DRAM_BYPASS`, is the mechanism — `dense_tier_parity`'s
third leg content-checks the arena-less dense path, and
`gpu_seam_paged_moe_host_tier_matches_resident`'s bypass leg is token-identical
on the MoE path (both shown to fail when the tier serves a neighbouring block).
What is NOT verified is the SELECTION: that `DeviceCaps::unified_memory` is set
where it should be, and that the arena above is sized sensibly out of shared RAM
on such a part. Metal has no pager at all (phase 4) and is unaffected until it
does.

### 4.1 如何确定自动预算

`infr_core::hostmem`. The probe and the arithmetic are separate so the policy is
testable without a machine that happens to have the right amount of RAM free.

- **`available_bytes`** — Linux reads `MemAvailable` (the kernel's own estimate
  of what a new allocation can have without swapping, so reclaimable page cache
  is already accounted for) and then takes the MINIMUM with the tightest cgroup
  ancestor's `memory.max - memory.current` (v2) or
  `limit_in_bytes - usage_in_bytes` (v1). The cgroup half is not optional:
  `/proc/meminfo` is host-wide, and measured on this box an 8 GiB `systemd-run`
  scope still reports 54.6 GB available — sizing an anonymous arena from that is
  an OOM kill in every container. Verified at 53.57 GB unconstrained, 8.59 GB
  under an 8 GiB cap, 2.15 GB under 2 GiB. Other platforms answer `None`, which
  callers treat as "do not auto-size" and never as "assume plenty".
- **`auto_arena_bytes`** — leaves the larger of 1 GiB and an eighth of what is
  available, never budgets past the pageable bytes (more would buy nothing), and
  returns 0 below a floor worth having.
- **`cpu_arena_plan` / `streaming_arena_plan`** — the two decisions, differing
  in one rung: the Vulkan tiers are already past the fit decision, while the CPU
  backend has to make it. Both answer with a reason when they decline, because
  "we cannot tell" and "it fits" lead to opposite advice and must never be
  collapsed into one silent `None`.

## 5. 分阶段实施

**Phase 0 — measure the baseline (no code). DONE.** `scripts/paging-baseline.py`
and its table in `documentation/evidence/benchmarks/2026-08-03-validated-models.md`; headline in §1.1. The bar every later
phase is judged against, re-run with the same harness.

**Phase 1 — core, no backend wired. DONE.** `blockio.rs` (`BlockDesc`,
`FileBlockIo`, the file-replaced stamp), `hostpager.rs` (arena, `Pin`, the
`Loading`/`Ready` handshake), pins in `Pager`, `Gguf::tensor_file_range` /
`Gguf::path`, and the address-keyed cache fix from §3.4 (`CpuBuffer::uid`).
Verified by unit tests over a fake `BlockIo` — content correct after churn, a
pinned block never evicted, exhaustion surfacing rather than corrupting, failed
reads leaving nothing resident, extent order deciding the layout, the
file-change check firing — each shown to fail by breaking what it guards. The
arena's unsafe is clean under Miri (tree-borrows), now a weekly cron step.

Deferred out of this phase, deliberately: **`TierPlan`** (its only consumer is
the phase-2 placement decision; a plan type with no caller is machinery shaped
by a guess) and the **prefetch worker pool** (same reason — the pool's depth and
thread count are meaningless until something measures them, and `HostPager::pin`
reads synchronously in the meantime).

**Phase 2 — CPU backend on the DRAM tier. DONE (prefetch deferred).**
`CpuBuffer::Paged` + `CpuRead::Pinned`, the per-op pin pre-step driven by
`Op::io()`, `infr_cpu::paged` (one pool per weight-size class, planned up front
from the GGUF's tensor directory), the `paging.dram` key, and the
`INFR_PAGER_STATS` per-pool report. Measured against phase 0 in
`documentation/evidence/benchmarks/2026-08-03-validated-models.md`: decode 1.28x at a 2 GB cap and **2.06x at 1.5 GB**,
major faults 210-335x lower, and read volume flat as the cap tightens where
mmap's grows. Prefill costs 3-7.5%.

Prefetch is still not built: the synchronous read is what the numbers above
already beat, and the pool/depth knobs have no measured values yet.

The ORIGINAL phase-2 verification, for the record: greedy token identity against
the CPU reference path with the budget forced small enough to churn; hit rates
matching the policy's predicted `(n_slots − 1) / n_blocks` per sweep; a model
larger than the budget completing at all; and beating phase 0 on throughput
**and** major-fault count.

**Phase 3 — Vulkan third tier. DONE (prefetch and the direct-to-ring read
deferred).** `DenseBytes`, `DensePoolSpec::host`, the three-case
`schedule_staged`, `DensePagerSession::pool_stats`, and the seam's
`dense_host_tier` (one host pager per dense pool, budget split by
`hostpager::plan_slots` — the same function `infr_cpu::paged::plan_pools` now
calls, so the two tiers cannot drift apart on the rule).

Verified by `infr-vulkan/tests/dense_tier_parity.rs`, which forces all three
cases in one sweep (VRAM 3 slots, DRAM 5 slots, 8 blocks, 3 passes) and asserts
each was taken from the counters rather than assuming: 4 VRAM hits, 20 VRAM
misses, 4 DRAM hits, 16 file reads, across 9 ring-half rotations — the ring
cursor persists across blocks, so `stage` really does refuse a full half and the
caller really does swap. Correctness is content-checked through the streamed
GEMV against the same weight in a plain arena, so a wrong slot decodes to
visibly different finite floats. It also pins the accounting — the tier below is
consulted exactly once per VRAM miss, one DRAM miss is one file read, and a read
moves a whole block — because a probe that fired on hits too would report the
sweep as warmer than it is. End-to-end, `gpu_seam_dense_stream_host_tier_`
`matches_resident` (Qwen3-1.7B, 200 MB VRAM and 256 MB DRAM budgets, both far
under the working set) is token-identical to the all-resident run. Clean under
the Khronos validation layer, with the loader confirmed to have loaded it.

Each new assertion was shown to go red by breaking what it guards: serving a
neighbouring block from the host tier (both the unit and the end-to-end test
fail — which is also what proves the end-to-end test engages the tier at all
rather than passing vacuously), consulting the tier twice per miss, and dropping
the registration check that a `Host` block exists below.

The MoE half landed with it: `ExpertBytes`, `MoePoolSpec::host`, per-expert
blocks in both `touch_role` and `stage_role`, and
`gpu_seam_paged_moe_host_tier_matches_resident` (Qwen3-30B-A3B, 50 MB VRAM and
256 MB DRAM budgets) token-identical to the all-resident run — with the same
break-probe, which diverged the output token-for-token and so proves the tier is
on the path.

**It now beats the mmap it replaces, and that is measured.**
`documentation/evidence/benchmarks/2026-08-03-validated-models.md` carries the table: Qwen3-14B Q8_0 streamed under a forced
2 GB VRAM budget, the tier runs at **2.17x of mmap on decode** under an 8 GB cap
with a 7 GB arena (0.39 vs 0.18 t/s), 1.41x with a 3 GB one, and at parity
unlimited — while doing what it targets: 38x fewer major faults under the cap,
232 to 110 GB read.

Three fixes got it there, and ALL THREE were found by measuring rather than by
reasoning about the design. The third is not code at all: the arena BUDGET is
the single biggest lever in the feature (3 GB → 7 GB is worth 1.6x on its own),
and nothing auto-sizes it — see §5's phase-5 list.

The first was worth 1.6x. The tier originally pinned each block in its arena and
memcpy'd it into the ring, and measured 0.48x of mmap: on CPU the arena REPLACES
the mapping so it adds no copy, but on Vulkan the bytes reach the ring either
way, making `disk -> arena -> ring` one copy more than `page-cache -> ring`.
`HostPager::fill` now admits only while a slot is free and, once the arena is
full, reads straight into the ring — which is also the correct residency call
and is exactly §3.6's "dense: partition, do not cache", arrived at from the
other direction. Decode 0.83 -> 1.36 t/s, prefill 54.7 -> 85.8.

The third fix was the ADMISSION RULE. This tier sits under one that only calls
down on its own misses, and on the first pass nothing is resident above — so
admitting on the first miss filled the arena with exactly the prefix the VRAM
pager then keeps forever, blocks that never call down again. `INFR_PAGER_STATS`
showed 9 slots holding 9 blocks of which 5 could ever be hit. Admission now
requires a SECOND miss, which no block the tier above keeps ever reaches: useful
hits per pass 5 → 9, bytes read −10.5%, decode 0.22 → 0.24.

That left the tier at 0.79x, and the second fix was the READER, not the policy
this plan spent its time on. `FileBlockIo::read_block` issued one `pread` per
extent on one thread. A drive delivers bandwidth on queue depth: measured on
this NVMe over 16-128 MB blocks, a single positioned read sustains 1.2-1.5 GB/s
against a 2.2 GB/s device ceiling reached at depth 2-4. So the tier was losing
to the mapping for a structural reason — the kernel issues readahead faults in
parallel for free — and a block is now split across `IO_FANOUT` concurrent
positioned reads. Read volume, fault counts and residency are unchanged across
that fix; only bandwidth moved. Decode 0.15 -> 0.22 under the cap.

**The lesson worth keeping: this plan's §5 named prefetch as the lever the
phase-3 result "points at hardest", and that was wrong.** The regime is
I/O-bound by orders of magnitude, so hiding a read behind compute had nearly
nothing to hide it behind; the read was not too LATE, it was too SLOW. What is
left is reading fewer bytes — the double-caching item below — not overlapping
the reads.

`paging.dram` is **no longer off by default**: the performance case is made, so
a run that has to stream now builds the tier and sizes it itself (§4.1). What is
still missing is coverage — one GPU, one drive, Linux only — which is why the
probe declines rather than guesses on platforms it cannot measure, and why `0`
exists to turn the whole thing off.

**Phase 4 — Metal / UMA collapse.** Arena, pager, offset binding, the
`qui_cache` gate. Verification: Metal decode parity against CPU reference under
a forced-tiny budget, and the first over-RAM model to load on Apple silicon at
all.

**Phase 5 — levers, each gated on a measurement.** The concurrent reader (see
phase 3) already landed the one that mattered; what follows is ordered by what
the measurement now says, which is NOT what this section said before it.

- **Auto-size `paging.dram` — LANDED, and it was the top lever.** The budget
  alone moves decode 0.24 → 0.29 → 0.39 t/s at 3, 6 and 7 GB under an 8 GB cap
  (mmap: 0.18), so the value a user used to have to guess was worth **1.6x** —
  and an unset budget used to disable the tier entirely, on exactly the runs
  that needed it. §4.1 has the sizing rule and §7's question 3 is answered
  (Linux probes `MemAvailable` and the cgroup limit; other platforms decline
  rather than guess). The arena must stay ANONYMOUS memory for a large budget to
  be safe — the kernel reclaims page cache in its favour, which is why 7 GB
  under an 8 GB cap does not thrash (major faults flat at ~1 700).
- **Double-caching: CLOSED, the premise was wrong both ways.** This plan
  asserted that a buffered `pread` halves the tier's effective budget and that
  `posix_fadvise(DONTNEED)` **cannot** reclaim the duplicate because
  `Gguf::open` maps the whole file. A `mincore` probe refutes both. `DONTNEED`
  DOES reclaim mapped-but-untouched pages (65 536 → 0); a page is exempt only
  once it is actually faulted into a page table, and this tier never touches
  paged ranges through the mapping. And the reclaim is not needed anyway: an
  anonymous arena already wins page-cache reclaim under a cap, which is exactly
  what the 7 GB row demonstrates. No `O_DIRECT`/`F_NOCACHE` rewrite, no
  alignment constraints, no work here. Do not reopen without new evidence.
- **Prefetch — deprioritized, and the reason is worth recording.** The
  synchronous read does sit on the critical path under the session mutex, and
  this plan called it "the one the phase-3 result points at hardest". That was
  wrong. Roughly 12.5 GB is read per token against tens of milliseconds of GPU
  compute, so overlapping the two hides a read behind almost nothing. Prefetch
  becomes interesting only once the tier is no longer I/O-bound — e.g. after the
  arena grows enough that most of a pass hits it.

Still speculative, still gated: frequency-warmed DRAM for MoE-on-GPU; io_uring
if the reader proves queue-depth bound BEYOND what `IO_FANOUT` concurrent
`pread`s already reach (measured: they hit the device ceiling on this drive, so
there may be nothing left here); exclusive VRAM/DRAM placement for MoE;
multi-GPU and MTP coverage.

Coverage the reader change does NOT have: the concurrent-read speedup is
measured on Linux/NVMe only. On Windows `seek_read` issues `ReadFile` with an
`OVERLAPPED` offset and a handle not opened `FILE_FLAG_OVERLAPPED` has its
concurrent operations serialized, so the fanout may buy nothing there; reads
stay correct either way. A rotational disk is also untested and is the one case
where concurrency could plausibly HURT.

## 6. 此功能专用的验证规则

- **Every tier transition is observable.** `INFR_PAGER_STATS` reports per tier:
  hits, misses, evictions, bytes read, and for disk how many reads were served
  from a completed prefetch versus blocked on the critical path. A prefetch that
  silently never fires is indistinguishable from one that works in every metric
  except throughput.
- **Correctness under churn is what breaks silently.** A wrong slot, a torn
  multi-extent read, an eviction under a live pin, or a stale address-keyed
  cache entry all produce plausible garbage rather than an error — hence
  content-checked tests at every tier, not residency-count tests.
- **Report the ceiling.** The banner states per-pass disk bytes and the implied
  throughput. A user waiting on sub-1 t/s should have been told.

## 7. 决策

Made here, changeable by the user:

1. **The dense disk tier is opt-in, not an automatic ladder rung.** At the §2
   ceiling, silently choosing it for an interactive decode turns a clean "does
   not fit" into a session that looks hung. `paging.disk` (or an explicit
   `paging.dram`) engages it; the auto ladder stops where it does today unless
   asked. MoE is different — skewed routing makes it genuinely fast — so MoE may
   take the tier automatically.
2. **Sharded GGUF is out of scope for v1.** `Gguf::open` maps one file; the
   `-NNNNN-of-MMMMM` set is understood only by `infr-hub`'s downloader. A paged
   sharded model is rejected at load with that reason, and `BlockExtent` carries
   no file id until a second file exists to point at.

Needing the user's call:

3. **Windows host-memory probe.** `libc` covers Linux and macOS; Windows needs
   `GlobalMemoryStatusEx`, i.e. a new dependency (`sysinfo` or `windows-sys`).
   Recommendation: neither — Windows requires an explicit `paging.dram` and says
   so. CI builds only ubuntu and macos today.

   Note this is not yet load-bearing: nothing probes host memory on ANY
   platform. `paging.dram` is explicit-only everywhere today (both
   `cpu_paged_store` and `dense_host_tier` return "no tier" when it is unset),
   which is the safe default B30 argues for — an anonymous arena sized from a
   guess is the expensive kind of wrong. The question only becomes live if the
   tier should ever engage automatically.

4. **Phase 4 (Metal) cannot be verified on this machine.** There is no Apple
   hardware here and `infr-metal` does not compile on this box, so the only
   evidence a Metal tier could carry is that CI type-checks it. That is exactly
   the shape the repo rules call a stub documented as working. Options:
   - **Skip it** and leave §3.7's Metal paragraph as the design (recommended if
     no Mac is coming): the tier stays a CPU + Vulkan feature, and Apple silicon
     keeps today's behaviour, which is that an over-RAM model cannot load at
     all.
   - **Write it unverified**, clearly marked in code and docs as never having
     run, for a later Mac session to finish. Cheap to write, and the risk is a
     reader mistaking "compiles" for "works".
   - **Defer until a Mac is available**, which is the honest version of option 1
     if one is.

   Nothing else in the plan is blocked on this — phase 5's levers are all
   CPU/Vulkan and each is gated on its own measurement.

## 8. 非目标

KV cache on disk (`INFR_KV_OVERFLOW` already spills KV to host, and KV has
different access physics); compressed or re-quantized on-disk formats; network
or object-store block sources; training. Each changes the block model and
belongs to its own plan.
