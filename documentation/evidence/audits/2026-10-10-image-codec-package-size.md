---
kind: audit
status: measured
scope: image-decoder-package-size
last_verified: 2026-10-10
verification: Windows-x86_64-DLL-build-and-RGB-decode; engine-integration-pending
---

# HEIC 与 AVIF 解码库体积

仅独立构建与解码，不编译或启动推理引擎、不使用 GPU。数字不是 Linux SO 体积，
也不是已集成后整包的精确增量；Rust glue 与发布配置仍需接入后验证。

## 构建范围

- libheif **1.23.6**，commit `f81f28ac014b1c28dee483ac45b72c6b05bc421a`。
- libde265 **1.1.3**，commit `ba62bf4cfb3242f3bf0a45617ff09e35236e4d82`。
- AVIF 组额外使用 dav1d **1.5.4**，commit `54706fc6bc0cdecab7e9593974a4039cc038fca7`。
- Windows x86_64，MSVC 19.51.36256，Release、LTO、静态 CRT；库本身保持 DLL。
- libheif 只启用 libde265，第二组额外启用 dav1d；插件加载、所有编码器、
  AOM、FFmpeg、JPEG/J2K、uncompressed codec、header compression、libsharpyuv、
  测试、示例和文档均关闭。保留正常 RGB 转换、变换与多线程能力。
- libde265 保留 SIMD，关闭 AVX-512；本次 MSVC 构建实际包含 SSE 路径。
  dav1d 保留 8/10/12-bit 与上游运行时 ISA 分派的 ASM，实现不是纯 C 最小慢速版。
- CMake `BUILD_SHARED_LIBS=ON`、`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`、
  `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`；Meson `default_library=shared`、
  `buildtype=release`、`b_vscrt=mt`、`b_lto=true`，禁用 tools/tests/examples/docs。

## 实测结果

压缩采用 ZIP DEFLATE level 9；ZIP 包含 DLL 与对应许可证文本，不包含 PDB、
导入库、头文件、编译器、测试图片或 glue。单位为 MiB（1048576 bytes）。

| 方案 | 运行库数量 | DLL 总字节 | DLL 总大小 | ZIP 字节 | ZIP 大小 |
|---|---:|---:|---:|---:|---:|
| HEIC | 2 | 2271744 | 2.17 MiB | 1030526 | 0.98 MiB |
| HEIC + AVIF | 3 | 4281344 | 4.08 MiB | 1842846 | 1.76 MiB |
| 增加 AVIF 的差额 | +1 | 2009600 | +1.92 MiB | 812320 | +0.77 MiB |

| 文件 | HEIC 组字节 | HEIC + AVIF 组字节 |
|---|---:|---:|
| `heif.dll` | 1708544 | 1715712 |
| `libde265.dll` | 563200 | 563200 |
| `dav1d.dll` | 不携带 | 2002432 |

`dumpbin /dependents` 确认运行时仅依赖上述 DLL 与 Windows `KERNEL32.dll`，
没有遗漏 MinGW、MSVC CRT、FFmpeg 或其他解码库；CRT 已计入各 DLL。
ZIP 里的许可证文本不等于完成最终分发合规检查，接入发包还需提供对应源码与构建说明。

## 解码验证

使用 libheif C API，从内存读入图片，解码为 RGB8 并按真实 stride 读取每行。

- 两组均成功解码上游 `examples/example.heic`（1280 x 854）、
  `colors-with-alpha.heic` 与 `colors-no-alpha.heic`（64 x 64）。
- 三张 HEIC 的两组 RGB SHA-256 完全一致。
- HEIC-only 拒绝上游 `example.avif`，错误明确为缺少 AV1 decoder。
- HEIC + AVIF 成功解码同一 AVIF（800 x 533）。
- 能力查询分别为 HEVC=true / AV1=false 和 HEVC=true / AV1=true。

这是基本功能验证，不代表已覆盖所有 iPhone HEIC：真实手机 10-bit、网格分块、
旋转/镜像、HDR、损坏文件与尺寸限制须在实际接入时增加回归用例。此次单次调用耗时
受冷启动/缓存影响，不用于解码速度比较。

## 结论与后续

推荐精简 libheif + libde265，避免依赖系统商店扩展；若同时加入 AVIF，当前测得
额外下载体积约 0.77 MiB，代价不大。是否本次一起加入由产品范围决定。
两者可以复用同一 C API glue，转成 RGB 后继续走现有 resize/patchify；
无需改变 GPU、KV、MTP 或 scheduler。

本轮没有修改 `infr-vision`、Cargo 依赖或正式发行包，两个独立包只是体积测量工件。
本机原始结果、构建脚本、探针与 ZIP：
`benchmark-data/artifacts/image-codec-size-20261010/`（Git 忽略）。

上游：[libheif](https://github.com/strukturag/libheif/releases/tag/v1.23.6)、
[libde265](https://github.com/strukturag/libde265/releases/tag/v1.1.3)、
[dav1d](https://code.videolan.org/videolan/dav1d/-/tags/1.5.4)。
