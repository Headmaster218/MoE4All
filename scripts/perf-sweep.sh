#!/usr/bin/env bash
# Perf snapshot: run the model x quant sweep vs llama.cpp and archive the matrix
# under target/perf/<utc>-<sha>.txt, so every commit's ratios are a diff away.
#
#   INFR_METAL=1 scripts/perf-sweep.sh model1.gguf model2.gguf ...
#   scripts/perf-sweep.sh model1.gguf ...          # Vulkan (default backend)
#
# Pass the SAME model list every time — the archive is only comparable when the
# rows line up. r=3, idle machine, nothing else on the GPU (see docs/perf.md).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: [INFR_METAL=1] $0 <model.gguf>..." >&2; exit 1; }

cargo build --release -p infr-cli

dev="Vulkan0"
if [ "${INFR_METAL:-}" = "1" ] && [ "$(uname)" = "Darwin" ]; then dev="MTL0"; fi

sha=$(git rev-parse --short HEAD)
dirty=$(git diff --quiet && echo "" || echo "-dirty")
out="target/perf/$(date -u +%Y%m%dT%H%M%SZ)-${sha}${dirty}.txt"
mkdir -p target/perf

{
  echo "commit: ${sha}${dirty}  date: $(date -u +%FT%TZ)  dev: ${dev}"
  echo "models: $*"
  echo
  ./target/release/infr compare --sweep --dev "$dev" "$@" 2>/dev/null
} | tee "$out"

echo
echo "archived: $out"
ls target/perf | tail -5
