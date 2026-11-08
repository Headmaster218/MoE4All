

---

## Round 86（续）：最后一个前置条件也清了

### engram 表怎么打开

`engram.rs:465` 的 `EngramTable::open(path: &Path) -> Result<Self>` 就是入口：

```rust
pub fn open(path: &Path) -> Result<Self> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    ...
}
```

**表路径来自 `INFR_ENGRAM_TABLE` 环境变量**（`engram.rs:1606`、`1655`）。

### 而且 mmap 不占 RAM

`engram.rs:436` 写得很清楚：

> *"The mapping costs ADDRESS SPACE, not RAM: only the pages a gather touches ever become
> resident, and `advise(Random)` stops the kernel readahead-ing the neighbours of the 1 KiB
> row that was actually asked for."*

**189 GB 文件映射只占虚拟地址空间**（x86-64 用户空间 128 TB，够用）。
每次 random 访问只把真正读的那一页带入内存。

### 于是整条链路都通了

| 环节 | 状态 |
|---|---|
| 表路径 | ✅ `INFR_ENGRAM_TABLE` |
| 打开 + mmap | ✅ `EngramTable::open`，已实现 |
| 行号计算 | ✅ `EngramMap::row_indices`，已实现 |
| 行反量化 | ✅ `dequant_mxfp8_row`，已实现 |
| 传给 graph | ✅ `ple_embd` 的 handle/alloc/bind 三件套 |
| 投影 + gate | ✅ `Linear` + `RmsNorm` + `QwenPleGate` |
| 接残差 | ✅ `hcr[0]`，6300 行之后 |

**每一环都有现成实现，缺的只是把它们连起来。**

### 这也解释了为什么这是一个"连线"问题而不是"实现"问题

`engram.rs` 994 行、有完整单测、接口清晰；`QwenPleGate` 两个后端都有；
`ple_embd` 提供了 input binding 范例。**所有零件都在，只是没有装配。**

### 下一轮：动手

按 Round 86 开头列的五点改动写代码。目标是**先编译通过**，然后：

1. `cargo test -p infr-llama` 保持绿
2. `scripts/check-v41-answers.sh` 看 `2+2=` 是否修复

### 预期

如果 engram 缺失确实是根因，`2+2=` 应该能答对。
如果不能，说明还有别的问题——但至少这一个确定的缺陷会被补上。

**风险**：prefill 每步多传 6 MiB，速度会降。**先要正确，再谈速度。**

