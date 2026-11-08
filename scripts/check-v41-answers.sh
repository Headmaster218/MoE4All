#!/usr/bin/env bash
# Does this build actually answer questions?
#
# The point of this script is the control model. Reading V4.1's output and deciding whether
# it "looks wrong" is not a test - it depends on the reader knowing the answer, and it gives
# no way to tell a broken model path from a broken engine. Running a small model whose answer
# is known pins the engine, so a V4.1 failure can only be V4.1.
#
# Usage:
#   scripts/check-v41-answers.sh                      # control model only
#   scripts/check-v41-answers.sh <control.gguf> <v41.gguf>
#
# Environment:
#   INFR_BIN         infr executable (default: target/release/infr)
#   INFR_DEV         device name (default: Vulkan1)
#   INFR_TIMEOUT     per-run timeout in seconds (default: 900)
#   INFR_PROMPT      prompt to ask (default: "2+2=")
#   INFR_EXPECT      substring the control must produce (default: "4")
#
# Exit codes: 0 everything checked answered correctly; 1 something did not; 2 setup problem.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${INFR_BIN:-$root/target/release/infr}"
dev="${INFR_DEV:-Vulkan1}"
timeout_s="${INFR_TIMEOUT:-900}"
prompt="${INFR_PROMPT:-2+2=}"
expect="${INFR_EXPECT:-4}"

if [[ ! -x $bin ]]; then
    printf 'error: no infr binary at %s (build it, or set INFR_BIN)\n' "$bin" >&2
    exit 2
fi

# A compromised GPU wedges the xe driver and needs a reboot to clear, so never let two of
# these overlap. flock on a fixed path is enough: the device is a singleton resource.
lock="${TMPDIR:-/tmp}/moe4all-v41-answer-check.lock"
exec 9>"$lock"
if ! flock -n 9; then
    printf 'error: another answer check holds %s - the GPU must not be shared\n' "$lock" >&2
    exit 2
fi

# Answer text only: progress lines and log records go to stderr, the reply to stdout. Strip
# anything that still looks like a log record so a stray INFO line cannot satisfy the check.
ask() {
    local model="$1"
    timeout "$timeout_s" "$bin" run "$model" "$prompt" \
        --max-new 8 --temp 0 --ctx 512 --ubatch 256 --dev "$dev" 2>/dev/null |
        grep -vE '^\[|^[0-9]{4}-[0-9]{2}-[0-9]{2}T' | tr -d '\r' | tail -n 3
}

failed=0

report() {
    local label="$1" model="$2" ok="$3" answer="$4"
    if [[ $ok == 1 ]]; then
        printf 'PASS  %-10s %s\n      output: %s\n' "$label" "$(basename -- "$model")" "$answer"
    else
        printf 'FAIL  %-10s %s\n      output: %s\n' "$label" "$(basename -- "$model")" "$answer"
        failed=1
    fi
}

control="${1:-}"
v41="${2:-}"

if [[ -n $control ]]; then
    answer="$(ask "$control")"
    ok=0
    [[ $answer == *"$expect"* ]] && ok=1
    report control "$control" "$ok" "$answer"
    if [[ $ok == 0 ]]; then
        printf '\nThe CONTROL model did not answer %q either. That means the engine, the\n' "$prompt" >&2
        printf 'sampling path, or this checkpoint is broken - fix that before reading anything\n' >&2
        printf 'into the V4.1 result below.\n' >&2
    fi
else
    printf 'note: no control model given; pass one as $1 to make the V4.1 result meaningful.\n'
fi

if [[ -n $v41 ]]; then
    for mode in default batched; do
        if [[ $mode == batched ]]; then
            answer="$(INFR_DSV41_BATCHED_PREFILL=1 ask "$v41")"
        else
            answer="$(ask "$v41")"
        fi
        ok=0
        [[ $answer == *"$expect"* ]] && ok=1
        report "$mode" "$v41" "$ok" "$answer"
    done
    printf '\nNote: the batched path is an opt-in bring-up switch. runner.rs documents that no\n'
    printf 'batch>1 V4 graph has executed correctly, so a batched PASS would be surprising and a\n'
    printf 'batched FAIL alone says nothing about the default path.\n'
fi

if [[ $failed != 0 ]]; then
    printf '\nAt least one check failed. Any throughput number measured on a build that fails\n' >&2
    printf 'here is measuring a computation that does not answer questions.\n' >&2
fi
exit "$failed"
