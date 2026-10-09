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
model="/tmp/smoke-model-00001-of-00002.gguf"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export XDG_CONFIG_HOME="$tmp/config"
unset INFR_API_KEY

# Isolate executable discovery too: launch checks must never find a real model engine.
mkdir -p "$tmp/work"
cp "$repo_root/Start-INFR-Wizard-Linux.sh" "$tmp/work/"
wizard="$tmp/work/Start-INFR-Wizard-Linux.sh"
cat > "$tmp/work/infr" <<'EOF'
#!/bin/sh
printf 'STUB_LAUNCHED auth=%s\n' "${INFR_API_KEY:+set}"
EOF
chmod +x "$wizard" "$tmp/work/infr"

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
expect 'automatic context override' '--ctx 150k' --mode run --model "$model" --profile aggressive --ctx 150k
expect 'automatic Q8 default' 'kv.type_k=q8_0' --mode run --model "$model" --profile aggressive
expect 'explicit automatic ubatch' '-u 3072' --mode run --model "$model" --profile aggressive --ubatch 3072
expect 'explicit automatic RAM' 'device.ram_budget=16GiB' --mode run --model "$model" --profile aggressive --ram 16GiB
expect_fail 'invalid mode is rejected' --mode bad --model "$model"
expect_fail 'missing model is rejected' --mode run --profile conservative
expect_fail 'missing option value is rejected' --mode run --model
expect_fail 'an option cannot consume another option' --mode run --model --profile aggressive
expect 'model with spaces' 'model\ with\ spaces.gguf' --mode run --model 'model with spaces.gguf'
expect 'serve default stays loopback' '--addr 127.0.0.1:8080' --mode serve --model "$model"

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
expect 'explicit MTP width beats saved value' 'spec.k=2' --mtp-k 2

printf '\n== corrupt state ==\n'
printf 'MODE=serve\nMODEL=%s\nPROFILE=whimsical\n' "$model" > "$XDG_CONFIG_HOME/infr/wizard.conf"
expect_fail 'invalid saved profile is rejected' --mode serve --model "$model"
printf 'MODE=invalid\nMODEL=%s\nPROFILE=conservative\n' "$model" > "$XDG_CONFIG_HOME/infr/wizard.conf"
expect_fail 'invalid saved mode is rejected' --model "$model"
expect 'explicit mode beats invalid saved value' "run $model" --mode run
rm "$XDG_CONFIG_HOME/infr/wizard.conf"

printf '\n== launch and authentication ==\n'
launch_check() {
    local desc="$1" status="$2" want="$3"; shift 3
    local out rc=0
    out="$("$wizard" "$@" </dev/null 2>&1)" || rc=$?
    if [ "$status" = ok ] && ! printf '%s\n' "$out" | grep -Fxq "$want"; then
        printf 'FAIL  %s (unexpected child state)\n%s\n' "$desc" "$out" >&2
        failures=$((failures + 1))
        return
    fi
    if { [ "$status" = ok ] && [ "$rc" -eq 0 ]; } ||
       { [ "$status" = fail ] && [ "$rc" -ne 0 ]; }; then
        if [[ "$out" == *"$want"* ]] &&
           { [ "$status" = ok ] || [[ "$out" != *STUB_LAUNCHED* ]]; }; then
            printf 'ok    %s\n' "$desc"
            return
        fi
    fi
    printf 'FAIL  %s (status=%s)\n%s\n' "$desc" "$rc" "$out" >&2
    failures=$((failures + 1))
}
base=(--mode serve --model "$model" --profile conservative)
launch_check 'network EOF cannot launch' fail 'requires explicit --yes' "${base[@]}" --addr 0.0.0.0:8080
launch_check 'loopback EOF cannot confirm launch' fail 'requires input or --yes' "${base[@]}" --addr 127.0.0.1:8080
launch_check 'explicit unauthenticated network launch' ok 'STUB_LAUNCHED auth=' "${base[@]}" --addr 0.0.0.0:8080 --no-api-key --yes
export INFR_API_KEY='wizard-smoke-fixture'
launch_check 'inherited key reaches child' ok 'STUB_LAUNCHED auth=set' "${base[@]}" --addr 127.0.0.1:8080 --yes
launch_check 'no-api-key clears inherited child environment' ok 'STUB_LAUNCHED auth=' "${base[@]}" --addr 127.0.0.1:8080 --no-api-key --yes
out="$("$wizard" --dry-run "${base[@]}" --addr 127.0.0.1:8080)"
if [[ "$out" == *"$INFR_API_KEY"* ]] || grep -q "$INFR_API_KEY" "$XDG_CONFIG_HOME/infr/wizard.conf"; then
    printf 'FAIL  key must not appear in output or saved state\n' >&2
    failures=$((failures + 1))
else
    printf 'ok    key is not printed or saved\n'
fi
unset INFR_API_KEY
launch_check 'localhost lookalike is not loopback' fail 'requires explicit --yes' "${base[@]}" --addr localhost.example:8080
out="$("$wizard" --dry-run "${base[@]}" --addr '[::1]:8080' 2>&1)"
if [[ "$out" == *warning:* ]]; then
    printf 'FAIL  IPv6 loopback has no exposure warning\n' >&2
    failures=$((failures + 1))
else
    printf 'ok    IPv6 loopback has no exposure warning\n'
fi

printf '\n== interactive defaults ==\n'
# Exercise the real prompt helper without requiring a terminal on CI.
if bash -c '
    source <(sed -n "/^ask() {/,/^}/p" "$1")
    ask MMPROJ projector /tmp/saved-mmproj.gguf <<< ""
    ask EMBEDDING embedding /tmp/saved-embed.gguf <<< ""
    [ "$MMPROJ" = /tmp/saved-mmproj.gguf ] && [ "$EMBEDDING" = /tmp/saved-embed.gguf ]
' _ "$wizard"; then
    printf 'ok    Enter retains saved optional paths\n'
else
    printf 'FAIL  Enter must retain saved optional paths\n' >&2
    failures=$((failures + 1))
fi

printf '\n'
if [ "$failures" -eq 0 ]; then
    printf 'smoke-linux-wizard: all checks passed\n'
else
    printf 'smoke-linux-wizard: %d check(s) failed\n' "$failures" >&2
    exit 1
fi
