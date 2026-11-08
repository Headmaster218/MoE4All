

---

## Round 52：真机复测 —— prefill 12 → 25 tok/s（2.1 倍），确认

### 两个修复都进了二进制

之前那次 2.1 倍是在**坏 shader** 下量的，所以复测前先确认二进制时间戳晚于修复提交。

### 模型说明（容易搞错）

`dsv41-uncensored-iq2xxs` 这个目录名是误导的。解析 GGUF 头：

```
exps 张量类型分布: {'IQ1_M': 30, 'IQ2_XXS': 8}
blk.0.ffn_gate_exps.weight   IQ1_M  [5120, 2304, 384]
blk.0.ffn_down_exps.weight   IQ1_M  [2304, 5120, 384]
```

**30/38 的专家张量是 IQ1_M**，不是 IQ2_XXS。修复针对的正是这条路径。

### 测量

```
INFR_DSV41_BATCHED_PREFILL=1 ./target/release/infr run <model> "$(cat /tmp/p512.txt)" \
  --max-new 1 --temp 0 --ctx 4096 --ubatch 256 --dev Vulkan1
```

| | prefill |
|---|---|
| 修复前（坏 shader） | 731 tok @ **12 tok/s**（62,630 ms） |
| 修复后（两个修复都进） | 1431 tok @ **25 tok/s**（57,136 ms） |

**约 2.1 倍**，与旧 shader 下量的 2.16 倍一致。

两次 token 数不同（731 vs 1431）是因为提示词文件重新生成过；**可比的是速率**。

### 一个当初值得怀疑的点

**坏 shader 下时长明明改善了，输出却是错的。**
这恰好说明：**加速是真实的，只有结果是坏的**——两者独立。

### decode 仍是 4.6 tok/s

还没碰。对标 poorseek 的 ~27.7 需要更多工作，而且**不是同口径**：
poorseek 用 `--route-bias 1.0`（近似路由）+ RTX 4090 + 90 GB DDR4。

### 下一步

decode 路径。当前 4.6 tok/s，差距在专家换入（`single-decode-layer` 显示
`paging_sync` 占每层 wall 的 90%+）。

