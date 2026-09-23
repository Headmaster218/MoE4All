#!/usr/bin/env bash
# MoE4All — Linux build-toolchain installer (no sudo required).
#
# What this does
# --------------
# MoE4All compiles its Vulkan compute shaders with `glslc` at `cargo build` time.
# The shaders use `GL_KHR_cooperative_matrix`, so the bundled `glslc` must be
# recent enough. Many distros ship an old `shaderc`/`glslc` that fails on those
# shaders with:
#     error: '#extension' : extension not supported: GL_KHR_cooperative_matrix
#
# This script installs a known-good `glslc` (from the LunarG Vulkan SDK — the same
# reference toolchain the Windows build uses) into your home directory and puts a
# wrapper on your PATH. It is idempotent: if a suitable glslc is already available
# it leaves it alone.
#
# Usage
# -----
#   ./scripts/install-linux.sh                 # auto: reuse existing good glslc, else install SDK one
#   ./scripts/install-linux.sh --sdk-version 1.4.357.0
#   ./scripts/install-linux.sh --prefix ~/.local
#   ./scripts/install-linux.sh --force         # reinstall even if a good glslc is on PATH
#
# After it succeeds, build MoE4All with:
#   cargo build --release --locked -p infr-cli
#   ./target/release/infr devices
#
set -euo pipefail

SDK_VERSION="${SDK_VERSION:-1.4.357.0}"
PREFIX="${PREFIX:-$HOME/.local}"
FORCE=0
# The Vulkan SDK tarball layout is <ver>/x86_64/{bin,lib}. URL is the standard LunarG host.
SDK_TARBALL="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/linux/vulkan-sdk-${SDK_VERSION}.0.tar.xz"

# --- args ---
while [ $# -gt 0 ]; do
  case "$1" in
    --sdk-version) SDK_VERSION="$2"; shift 2;;
    --sdk-version=*) SDK_VERSION="${1#*=}"; shift;;
    --prefix) PREFIX="$2"; shift 2;;
    --prefix=*) PREFIX="${1#*=}"; shift;;
    --force) FORCE=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# A shader that exercises GL_KHR_cooperative_matrix — the exact feature the
# MoE4All GEMM/attention shaders require. `#extension ... : require` fails on a
# glslc that predates KHR cooperative matrix, and succeeds (unused or not) on one
# that has it — so this is a clean version gate with no other extensions to drag in.
COOPMAT_PROBE='#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 32) in;
void main() { }'

glslc_ok() {
  # $1 = candidate glslc executable
  local exe="$1"
  [ -x "$exe" ] || return 1
  local tmp
  tmp="$(mktemp -d)" || return 1
  printf '%s' "$COOPMAT_PROBE" > "$tmp/probe.comp"
  "$exe" -fshader-stage=comp --target-env=vulkan1.3 -O "$tmp/probe.comp" -o "$tmp/probe.spv" >/dev/null 2>&1
  local rc=$?
  rm -rf "$tmp"
  [ "$rc" -eq 0 ]
}

echo "==> MoE4All Linux glslc installer"
echo "    prefix        : $PREFIX"
echo "    sdk version   : $SDK_VERSION"

# 1) Is there already a suitable glslc?
if [ "$FORCE" -ne 1 ]; then
  existing="$(command -v glslc || true)"
  if [ -n "$existing" ]; then
    if glslc_ok "$existing"; then
      echo "==> Found a suitable glslc on PATH: $existing"
      "$existing" --version 2>/dev/null | head -1 || true
      echo "    Nothing to do. Build with: cargo build --release --locked -p infr-cli"
      exit 0
    else
      echo "==> Existing glslc ($existing) is too old (no GL_KHR_cooperative_matrix)."
      echo "    Installing a known-good one under $PREFIX."
    fi
  else
    echo "==> No glslc on PATH. Installing a known-good one under $PREFIX."
  fi
fi

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }
command -v tar  >/dev/null 2>&1 || { echo "error: tar is required" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
tarball="$work/vulkan-sdk.tar.xz"

echo "==> Downloading LunarG Vulkan SDK ${SDK_VERSION} (glslc + libs only)..."
curl -fSL --retry 3 -o "$tarball" "$SDK_TARBALL"

echo "==> Extracting glslc and its runtime libraries..."
mkdir -p "$work/extract"
# Only pull the binary and the shared libs it links against; ignore the rest of the SDK.
tar -xJf "$tarball" -C "$work/extract" \
  "*/x86_64/bin/glslc" \
  "*/x86_64/lib/libshaderc_shared.so.1" \
  "*/x86_64/lib/libshaderc_shared.so" \
  "*/x86_64/lib/libSPIRV-Tools-shared.so" \
  "*/x86_64/lib/libSPIRV.so.16" \
  "*/x86_64/lib/libSPIRV.so.16.4.0" \
  "*/x86_64/lib/libglslang.so.16" \
  "*/x86_64/lib/libglslang.so.16.4.0" \
  "*/x86_64/lib/libglslang-default-resource-limits.so.16.4.0" \
  2>/dev/null || true

glslc_bin="$(find "$work/extract" -type f -name glslc -path '*/bin/*' | head -1)"
[ -n "$glslc_bin" ] || { echo "error: glslc not found in SDK tarball" >&2; exit 1; }
libdir="$(find "$work/extract" -type d -name lib -path '*/x86_64/lib' | head -1)"
[ -n "$libdir" ] || libdir="$(dirname "$glslc_bin")/../lib"

# 2) Lay it down under $PREFIX with a self-contained wrapper (no LD_LIBRARY_PATH
#    needed at call time because the wrapper sets it).
install_lib="$PREFIX/lib/moe4all-glslc"
install_bin="$PREFIX/bin"
mkdir -p "$install_lib" "$install_bin"
cp "$glslc_bin" "$install_lib/glslc"
for so in "$libdir"/*.so*; do
  [ -f "$so" ] && cp -L "$so" "$install_lib/"
done
chmod +x "$install_lib/glslc"

wrapper="$install_bin/glslc"
cat > "$wrapper" <<EOF
#!/bin/sh
# Installed by MoE4All scripts/install-linux.sh — LunarG Vulkan SDK glslc ${SDK_VERSION}.
exec env LD_LIBRARY_PATH="$install_lib:\$LD_LIBRARY_PATH" "$install_lib/glslc" "\$@"
EOF
chmod +x "$wrapper"

echo "==> Verifying the installed glslc can compile GL_KHR_cooperative_matrix..."
if ! glslc_ok "$wrapper"; then
  echo "error: installed glslc failed the cooperative-matrix probe" >&2
  exit 1
fi
echo "    OK: $wrapper"
"$wrapper" --version 2>/dev/null | head -1 || true

# 3) Make sure $PREFIX/bin is on PATH for future shells.
case ":$PATH:" in
  *":$install_bin:"*) echo "==> $install_bin is already on PATH.";;
  *)
    echo "==> Add this to your shell rc (~/.bashrc or ~/.zshrc) to put glslc on PATH:"
    echo "    export PATH=\"$install_bin:\$PATH\""
    if [ -f "$HOME/.bashrc" ]; then
      if ! grep -qF "$install_bin" "$HOME/.bashrc" 2>/dev/null; then
        printf '\n# MoE4All glslc (installed by scripts/install-linux.sh)\nexport PATH="%s:$PATH"\n' "$install_bin" >> "$HOME/.bashrc"
        echo "    (appended to ~/.bashrc)"
      fi
    fi
  ;;
esac

echo
echo "==> Done. Now build MoE4All:"
echo "    cd <MoE4All checkout>"
echo "    export PATH=\"$install_bin:\$PATH\"   # or open a new shell"
echo "    cargo build --release --locked -p infr-cli"
echo "    ./target/release/infr devices"
echo
echo "    (Runtime deps for the Vulkan path: a Vulkan driver + libvulkan, e.g."
echo "     Ubuntu: sudo apt install libvulkan1 vulkan-tools  — only needed to RUN, not to build.)"
