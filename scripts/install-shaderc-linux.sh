#!/usr/bin/env bash
set -euo pipefail
# Build the shader compiler on the release builder's glibc, independent of distro packages.
prefix="${1:?usage: install-shaderc-linux.sh PREFIX}"
if [ ! -x "$prefix/bin/glslc" ]; then
    work="$prefix/build"
    mkdir -p "$work"
    if [ ! -d "$work/shaderc/.git" ]; then
        git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 clone --depth 1 --branch v2026.3 https://github.com/google/shaderc.git "$work/shaderc"
    fi
    (cd "$work/shaderc" && python3 utils/git-sync-deps)
    cmake -S "$work/shaderc" -B "$work/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DSHADERC_SKIP_TESTS=ON \
        -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON
    cmake --build "$work/build" --target glslc_exe --parallel "${MOE4ALL_BUILD_JOBS:-2}"
    mkdir -p "$prefix/bin"
    install -m 755 "$work/build/glslc/glslc" "$prefix/bin/glslc"
fi
"$prefix/bin/glslc" --version
