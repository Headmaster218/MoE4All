# 启动解压依赖

`fflate.cjs` 来自仓库已锁定的 `fflate@0.8.3` 的 `lib/browser.cjs`，源码地址为 <https://github.com/101arrowz/fflate>。此文件未修改，MIT 许可保存在 `fflate.LICENSE`。

只使用同步流式 ZIP 解压接口，不使用浏览器或 worker API。直接随包提供此依赖，使首次解压不依赖尚未恢复的 DSH node_modules。
