

---

## Round 134：engram 真对照 —— 确定无罪；嫌疑只剩压缩注意力 / 超连接 / hash 路由

### 之前的「关掉 engram」为什么不算数

```
infr: INFR_ENGRAM_TABLE is unset; V4.1 engram will contribute garbage
```

不设表 = 让 engram **注入垃圾**，不是关掉它。Round 129 据此排除 engram 是错的（Round 133 已撤）。

### 这次是真对照：`INFR_ENGRAM_ZERO`

在 `engram_apply` 闭包开头直接 `return`，跳过整个注入。
**往残差流加零，与用一张全零表完全等价**，而后者的文件是 202 GB，不现实。

（本机另有 `/data/models/dsv41-engram-stub/engram-q6k-stub.gguf`，数据区实测 0 非零字节，
但它是 ggml type 14 / Q6_K，本 reader 只吃 32/34，用不了。）

### 实测（`A B C D`，temp 0，ctx 512）

| 配置 | 输出 |
|---|---|
| 有 engram（MXFP8 表） | `**  **  **  **  ** ` |
| **零注入（真·无 engram）** | `- 0.5` |

**两种都错，但输出不同。**

### 两个结论

1. **engram 无罪。** 完全不注入，模型照样答错。这是一次有效对照，与前几轮不同。
2. **engram 确实在参与计算。** 有无之间输出不同（星号 vs `- 0.5`），
   说明它注入的向量真的流进了残差流 —— 与更早的 A/B 结论一致。
   它只是不是**根因**。

### 嫌疑范围收敛到 3 个

对照组（qwen2.5，Round 133）已证明引擎通用路径健康，所以只剩 V4.1 专有机制：

1. **压缩注意力** —— `kv_source_layer_ids=[2,8,14,20]` 四个源层 + 34 个复用层，
   源平面 gather 拓扑
2. **超连接 `hyper_connection`** —— `hc` 份残差副本 + `post_layer_mix` / `comb_res_mix` 混合
3. **MoE hash 路由** —— `ffn_gate_tid2eid` 按 token id 直查专家

### 下一步判据

超连接是最可疑的：它把残差流复制成 `hc` 份再混合，
**任何一处 mix 系数或副本顺序错位，都会系统性地搅乱整个表示**
——正好对应「语法完整、语义错位、logits 仍高置信」的症状画像。

