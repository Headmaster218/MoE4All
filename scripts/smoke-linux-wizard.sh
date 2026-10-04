#!/usr/bin/env bash
#
# smoke-linux-wizard.sh — non-interactive smoke test for
# Start-INFR-Wizard-Linux.sh.
#
# It drives the launcher with `--dry-run` (so nothing is ever launched), in a
# private XDG_CONFIG_HOME (so the caller's saved selections are never touched),
# and asserts the argument list that was printed.
#
#   ./scripts/smoke-linux-wizard.sh
#
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
wizard="$repo_root/Start-INFR-Wizard-Linux.sh"
model="/tmp/smoke-model-00001-of-00002.gguf"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export XDG_CONFIG_HOME="$tmp/config"

# A stub `infr` so the test does not depend on a build; `--dry-run` never runs it.
mkdir -p "$tmp/bin"
printf '#!/bin/sh\nexit 0\n' > "$tmp/bin/infr"
chmod +x "$tmp/bin/infr"
export PATH="$tmp/bin:$PATH"

failures=0

argv_of() { # the wizard prints the command on the line after "Command:"
    printf '%s\n' "$1" | sed -n '/^Command:/{n;s/^ *//;p;}'
}

expect() { # expect <description> <expected substring> <args...>
    local desc="$1" want="$2"; shift 2
    local out argv
    if ! out="$("$wizard" --dry-run "$@" </dev/null 2>&1)"; then
        printf 'FAIL  %s (wizard exited non-zero)\n%s\n' "$desc" "$out" >&2
        failures=$((failures + 1)); return
    fi
    argv="$(argv_of "$out")"
    case "$argv" in
        *"$want"*) printf 'ok    %s\n' "$desc" ;;
        *) printf 'FAIL  %s\n        want substring: %s\n        got: %s\n' \
                  "$desc" "$want" "$argv" >&2
           failures=$((failures + 1)) ;;
    esac
}

expect_absent() { # expect_absent <description> <substring that must NOT appear> <args...>
    local desc="$1" unwanted="$2"; shift 2
    local out argv
    if ! out="$("$wizard" --dry-run "$@" </dev/null 2>&1)"; then
        printf 'FAIL  %s (wizard exited non-zero)\n%s\n' "$desc" "$out" >&2
        failures=$((failures + 1)); return
    fi
    argv="$(argv_of "$out")"
    case "$argv" in
        *"$unwanted"*) printf 'FAIL  %s (unexpectedly found: %s)\n        got: %s\n' \
                                  "$desc" "$unwanted" "$argv" >&2
                       failures=$((failures + 1)) ;;
        *) printf 'ok    %s\n' "$desc" ;;
    esac
}

expect_fail() { # expect_fail <description> <args...>
    local desc="$1"; shift
    if "$wizard" --dry-run "$@" </dev/null >/dev/null 2>&1; then
        printf 'FAIL  %s (expected a non-zero exit)\n' "$desc" >&2
        failures=$((failures + 1))
    else
        printf 'ok    %s\n' "$desc"
    fi
}

printf '\n== modes ==\n'
expect 'run'   "run $model"   --mode run   --model "$model" --profile aggressive
expect 'serve' "serve $model" --mode serve --model "$model" --profile conservative \
                              --addr 127.0.0.1:8080 --parallel 1
expect 'bench' "bench $model" --mode bench --model "$model" --profile manual --ctx 8192

printf '\n== profiles ==\n'
expect 'aggressive'   'device.auto_profile=aggressive'   --mode run --model "$model" --profile aggressive
expect 'conservative' 'device.auto_profile=conservative' --mode run --model "$model" --profile conservative
expect 'manual'       '--ctx 8192'                       --mode run --model "$model" --profile manual --ctx 8192
expect_fail 'invalid profile is rejected'                --mode run --model "$model" --profile whimsical

printf '\n== saved state ==\n'
mkdir -p "$XDG_CONFIG_HOME/infr"
cat > "$XDG_CONFIG_HOME/infr/wizard.conf" <<EOF
MODE=serve
MODEL=$model
PROFILE=conservative
CTX=
UBATCH=
KV_K=q8_0
KV_V=q8_0
RAM=
VRAM=
MTP=/tmp/saved-mtp.gguf
MTP_K=4
ADDR=0.0.0.0:8080
PARALLEL=2
MMPROJ=/tmp/saved-mmproj.gguf
EMBEDDING=/tmp/saved-embed.gguf
EOF

expect        'reused MTP head'        'spec.draft=/tmp/saved-mtp.gguf'          --mode serve --model "$model" --profile conservative
expect        'reused mmproj'          '--mmproj /tmp/saved-mmproj.gguf'         --mode serve --model "$model" --profile conservative
expect        'reused embedding'       '--embedding-model /tmp/saved-embed.gguf' --mode serve --model "$model" --profile conservative
expect_absent 'overridden address'     '0.0.0.0:8080'                            --mode serve --model "$model" --profile conservative --addr 127.0.0.1:18080

printf '\n== explicit flags clear saved state ==\n'
expect_absent '--no-mtp clears the saved MTP head'           'spec.mtp=1'                                --mode serve --model "$model" --profile conservative --no-mtp
expect_absent '--no-mmproj clears the saved projector'       '--mmproj /tmp/saved-mmproj.gguf'           --mode serve --model "$model" --profile conservative --no-mmproj
expect_absent '--no-embedding clears the saved embedding'    '--embedding-model /tmp/saved-embed.gguf'   --mode serve --model "$model" --profile conservative --no-embedding

printf '\n== corrupt state ==\n'
printf 'MODE=serve\nMODEL=%s\nPROFILE=whimsical\n' "$model" > "$XDG_CONFIG_HOME/infr/wizard.conf"
expect_fail 'invalid saved profile is rejected' --mode serve --model "$model"

printf '\n'
if [ "$failures" -eq 0 ]; then
    printf 'smoke-linux-wizard: all checks passed\n'
else
    printf 'smoke-linux-wizard: %d check(s) failed\n' "$failures" >&2
    exit 1
fi
