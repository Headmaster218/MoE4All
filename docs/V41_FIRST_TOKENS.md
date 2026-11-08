

---

## Round 77：找到一个可能的外部 oracle，但被 engram 打包方式挡住

### 本机有一个 llama.cpp 分支

```
/home/dyy/deepseek-work/llm-for-glm-flash/llama-cpp-dsv41/
  src/models/deepseek41.cpp        948 行
  build-cpu/bin/llama-cli          已构建
```

`config.rs` 的注释一直在引用 `deepseek41.cpp`，**源文件就在本机**。
这是本会话第一次有可能拿到独立参照。

### 尝试直接用

```
llama-cli -m ...-IQ2_XXS-00001-of-00004.gguf -p "2+2=" -n 8 --temp 0
```

结果：

```
E llama_model_load: error loading model: blk.1.engram_embd.weight is missing
```

### 为什么

这个 llama.cpp 分支**期待 engram 张量打包在同一个 gguf 里**。

而本项目的布局是：engram 表放**独立文件**（`engram-mxfp8.gguf`，189 GB；
`engram.rs:38` 有完整文档），通过 `INFR_ENGRAM_TABLE` 指定，运行时 gather 后作为
graph Input 喂进去，**不当作上传权重**。

两种打包方式不兼容。本机没有把 engram 内联进主 gguf 的版本
（`dsv41-engram-stub` 只有 2 个张量、架构名 `deepseek41-engram`，不能被当作主模型加载）。

### 为什么不继续

要让 llama.cpp 接受这个模型，得改它的加载器去读外部 engram 表——
**那是给 llm.cpp 打补丁，超出了"给 V4.1 做验收"的范围**，而且改完的 oracle
本身就不可信了（是我改过的）。

### 但这仍然是有价值的发现

| 项目 | 状态 |
|---|---|
| llama.cpp 分支源码 | ✅ 在本机，948 行 |
| llama-cli 可执行 | ✅ 已构建 |
| 能直接跑本 repo 的 V4.1 | ❌ engram 打包方式不同 |
| 改造可行性 | 需改 llama.cpp 加载器 |

**`deepseek41.cpp` 仍然可以当"读代码的参照"用**——它实现了同一套 CSA2 source/reuse、
engram、dspark，可以逐段对照我们的 `runner.rs`。只是不能当"跑出答案的 oracle"。

### 下一步

如果要拿答案级 oracle，最省事的是**下载一个把 engram 内联的 V4.1 gguf**
（如果发布方提供），而不是改 llama.cpp。

