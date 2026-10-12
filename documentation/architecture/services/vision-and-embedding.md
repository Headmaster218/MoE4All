---
kind: architecture
status: current
scope: auxiliary-engines
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 视觉与嵌入

## 共同资源边界

Vision 和 Embedding 有独立模型逻辑，但在 Vulkan 模式下可从运行中的 LLM backend 派生 client，并使用同一个 unified
VRAM pool。它们不能绕过资源管理器再建立一份不可见的 GPU 模型栈。

## 视觉

当前原生路径面向 Qwen3-VL-style CLIP projector 的受限子集：

- 解析 mmproj GGUF；
- 解码 data URI 或 base64 image；
- resize、patch 与位置预处理；
- 请求级载入 projector weights；
- 生成 embedding 并注入文本 runtime；
- image batch 完成后释放临时 VRAM。

不支持的 projector/deepstack 或远程 HTTP image URL 应明确失败，避免静默降级和 SSRF 面。

输入按文件内容识别 PNG、JPEG、WebP、GIF、BMP、TIFF、ICO；HEIC/AVIF 使用发行包随带的 `image-codecs/` 动态解码库。动态图像路径不是视频推理，只处理解码出的静态帧/主图像。HEIC/AVIF 原生解码限制为压缩输入不超过 256 MiB、每边不超过 8192、总像素不超过 64 × 1024²；HTTP 输入仍先受 64 MiB body 限制。解码后按 `MAX_PATCHES=4096` 进行 smart resize；不代表先无界解码再缩小。

解码库从引擎可执行文件旁定位；源码测试可用绝对路径 `INFR_IMAGE_CODEC_DIR`。缺库、无 decoder 或非法 payload 明确报错，日志包含猜测格式和 magic 摘要；不抓取远程 URL。格式支持不意味着支持任意 mmproj，详见[图片与 Embedding 使用](../../guide/serving/images-and-embeddings.md)。

## 嵌入

原生 Embedding engine 与 llama.cpp compatibility worker 共享一个上层 contract。原生 Vulkan 模式可以：

- 第一次请求从 GGUF/SSD 加载权重；
- 在 idle timeout 内复用；
- 超时后释放权重并恢复 Expert filler；
- 通过 `/v1/embeddings` 返回 normalized vectors 和 token usage。

## 并发原则

- Active LLM、Vision 和 Embedding request 都通过统一 owner/lease 保护资源。
- 辅助模型不能回收正在执行的 LLM/MTP runtime range 或 frozen Expert slot。
- auxiliary workspace 在其执行期间保留 owner lease；权重 idle eviction 与临时 scratch 回收分别处理。
- state transfers 和 staging-ring recording 与主 Recorder 共用执行门；避免只锁 command-buffer alloc/free、却让 recording lifetime 重叠。

Agent 插件通过 Provider/API 接入，见[产品架构](../product/agent-distribution.md)。本页的共享显存边界对应 0.10.0 引擎；并发视觉的历史崩溃与修复见[事故记录](../../evidence/incidents/INC-20261009-vulkan-aux-command-pool.md)。
