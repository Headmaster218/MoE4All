#!/usr/bin/env bash
set -euo pipefail
# Build the shader compiler on the release builder's glibc, independent of distro packages.
prefix="${1:?usage: install-shaderc-linux.sh PREFIX}"
if [ ! -x "$prefix/bin/glslc" ]; then
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    git clone --depth 1 --branch v2026.3 https://github.com/google/shaderc.git "$work/shaderc"
    (cd "$work/shaderc" && python3 utils/git-sync-deps)
    cmake -S "$work/shaderc" -B "$work/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DSHADERC_SKIP_TESTS=ON \
        -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON
    cmake --build "$work/build" --target glslc --parallel 2
    mkdir -p "$prefix/bin"
    install -m 755 "$work/build/glslc/glslc" "$prefix/bin/glslc"
fi
"$prefix/bin/glslc" --version
