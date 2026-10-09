#!/usr/bin/env bash
#
# Start-INFR-Wizard-Linux.sh — interactive launcher for `infr` on Linux.
#
# The Windows package ships a PowerShell wizard; this covers the same common
# workflow on Linux, and nothing more (no self-update, no rarely-used advanced
# options). The command is assembled in a Bash array and executed directly —
# there is no `eval` anywhere in this script.
#
#   ./Start-INFR-Wizard-Linux.sh                 # interactive
#   ./Start-INFR-Wizard-Linux.sh --dry-run       # print the command, do not launch
#
# Every prompt can also be answered with an option, in which case it is not
# asked for — and an option always wins over a value reloaded from the previous
# session. With `--dry-run` the script never needs a TTY, so it is testable:
#
#   ./Start-INFR-Wizard-Linux.sh --dry-run --mode serve --model m.gguf \
#       --profile aggressive --addr 127.0.0.1:8080 --parallel 1
#
set -euo pipefail

PROG="${0##*/}"
STATE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/infr"
STATE_FILE="$STATE_DIR/wizard.conf"

# --------------------------------------------------------------- selection ---
MODE=""          # run | serve | bench
MODEL=""
PROFILE=""       # aggressive | conservative | manual
CTX=""
UBATCH=""
KV_K=""
KV_V=""
RAM=""
VRAM=""
MTP=""           # empty = off; otherwise the path to the MTP head
MTP_K="4"
ADDR=""
PARALLEL=""
MMPROJ=""
EMBEDDING=""
API_KEY=""       # empty = no auth; otherwise exported as INFR_API_KEY
BENCH_P="512"
BENCH_N="128"
BENCH_D="0"

DRY_RUN=0
ASSUME_YES=0
SKIP_API_KEY=0

# Options that were named on the command line. Anything in here beats the saved
# state, which is what lets `--no-mtp` (etc.) clear a remembered value.
declare -A GIVEN=()

usage() {
    cat <<'EOF'
Start-INFR-Wizard-Linux.sh — interactive launcher for infr on Linux

Usage:
  ./Start-INFR-Wizard-Linux.sh [options]

Any option that is not given is asked for interactively; an option always wins
over the value reloaded from the previous session. `--dry-run` prints the final
command and exits without launching, and works without a TTY.

Options:
  --dry-run                print the final command, do not launch
  --yes                    do not ask for confirmation before launching
  --mode <run|serve|bench> what to start
  --model <path>           main model (.gguf path, or org/repo[:quant])
  --profile <name>         aggressive | conservative | manual
  --ctx <tokens>           context window (e.g. 32768, 256k)
  --ubatch <n>             prefill micro-batch (alias: -u)
  --kv-k <dtype>           KV format for K (e.g. q8_0, q4_0, f16)
  --kv-v <dtype>           KV format for V
  --ram <size>             device.ram_budget (e.g. 16GiB)
  --vram <size>            device.vram_budget (e.g. 23GiB)
  --mtp <path>             enable MTP with this MTP head
  --no-mtp                 disable MTP (clears a saved MTP head)
  --mtp-k <n>              MTP verification width (default: 4)
  --addr <host:port>       serve: listen address
  --parallel <n>           serve: parallel slots
  --mmproj <path>          serve: vision projector
  --no-mmproj              serve: disable the vision projector
  --embedding <path>       serve: embedding model
  --no-embedding           serve: disable the embedding model
  --no-api-key             serve: do not authenticate (ignores INFR_API_KEY)
  --bench-p <n>            bench: prompt tokens (default: 512)
  --bench-n <n>            bench: generated tokens (default: 128)
  --bench-d <n>            bench: context depth (default: 0)
  -h, --help               this help

The API key is never passed on the command line: it is read with hidden input
and exported as INFR_API_KEY for the child process, so it stays out of the
process list and the shell history.
EOF
    exit 0
}

die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 1; }

# ----------------------------------------------------------- CLI arguments ---
while [ $# -gt 0 ]; do
    case "$1" in
        --mode|--model|--profile|--ctx|--ubatch|-u|--kv-k|--kv-v|--ram|--vram|--mtp|--mtp-k|--addr|--parallel|--mmproj|--embedding|--bench-p|--bench-n|--bench-d)
            [ $# -ge 2 ] && [ -n "$2" ] && [[ "$2" != --* ]] || die "missing value for $1"
            ;;
    esac
    case "$1" in
        --dry-run)          DRY_RUN=1 ;;
        --yes|-y)           ASSUME_YES=1 ;;
        --mode)             MODE="${2-}"; GIVEN[MODE]=1; shift ;;
        --model)            MODEL="${2-}"; GIVEN[MODEL]=1; shift ;;
        --profile)
            PROFILE="${2-}"; GIVEN[PROFILE]=1; shift
            case "$PROFILE" in
                aggressive|conservative|manual) : ;;
                *) die "invalid --profile '$PROFILE' (expected aggressive, conservative or manual)" ;;
            esac
            ;;
        --ctx)              CTX="${2-}"; GIVEN[CTX]=1; shift ;;
        --ubatch|-u)        UBATCH="${2-}"; GIVEN[UBATCH]=1; shift ;;
        --kv-k)             KV_K="${2-}"; GIVEN[KV_K]=1; shift ;;
        --kv-v)             KV_V="${2-}"; GIVEN[KV_V]=1; shift ;;
        --ram)              RAM="${2-}"; GIVEN[RAM]=1; shift ;;
        --vram)             VRAM="${2-}"; GIVEN[VRAM]=1; shift ;;
        --mtp)              MTP="${2-}"; GIVEN[MTP]=1; shift ;;
        --no-mtp)           MTP=""; GIVEN[MTP]=1 ;;
        --mtp-k)            MTP_K="${2-}"; GIVEN[MTP_K]=1; shift ;;
        --addr)             ADDR="${2-}"; GIVEN[ADDR]=1; shift ;;
        --parallel)         PARALLEL="${2-}"; GIVEN[PARALLEL]=1; shift ;;
        --mmproj)           MMPROJ="${2-}"; GIVEN[MMPROJ]=1; shift ;;
        --no-mmproj)        MMPROJ=""; GIVEN[MMPROJ]=1 ;;
        --embedding)        EMBEDDING="${2-}"; GIVEN[EMBEDDING]=1; shift ;;
        --no-embedding)     EMBEDDING=""; GIVEN[EMBEDDING]=1 ;;
        --no-api-key)       SKIP_API_KEY=1; GIVEN[API_KEY]=1 ;;
        --bench-p)          BENCH_P="${2-}"; shift ;;
        --bench-n)          BENCH_N="${2-}"; shift ;;
        --bench-d)          BENCH_D="${2-}"; shift ;;
        -h|--help)          usage ;;
        *)                  die "unknown option: $1 (try --help)" ;;
    esac
    shift
done

# ------------------------------------------------------------ saved values ---
# Reuse the previous selections, but never over an option given on the command
# line — that is how the explicit flags above beat the state file.
if [ -f "$STATE_FILE" ]; then
    while IFS= read -r line; do
        case "$line" in
            *=*) : ;;
            *)   continue ;;
        esac
        key=${line%%=*}
        value=${line#*=}
        case "$key" in
            MODE)      [ -z "${GIVEN[MODE]:-}" ]      && MODE=$value ;;
            MODEL)     [ -z "${GIVEN[MODEL]:-}" ]     && MODEL=$value ;;
            PROFILE)   [ -z "${GIVEN[PROFILE]:-}" ]   && PROFILE=$value ;;
            CTX)       [ -z "${GIVEN[CTX]:-}" ]       && CTX=$value ;;
            UBATCH)    [ -z "${GIVEN[UBATCH]:-}" ]    && UBATCH=$value ;;
            KV_K)      [ -z "${GIVEN[KV_K]:-}" ]      && KV_K=$value ;;
            KV_V)      [ -z "${GIVEN[KV_V]:-}" ]      && KV_V=$value ;;
            RAM)       [ -z "${GIVEN[RAM]:-}" ]       && RAM=$value ;;
            VRAM)      [ -z "${GIVEN[VRAM]:-}" ]      && VRAM=$value ;;
            MTP)       [ -z "${GIVEN[MTP]:-}" ]       && MTP=$value ;;
            MTP_K)     [ -z "${GIVEN[MTP_K]:-}" ]     && MTP_K=$value ;;
            ADDR)      [ -z "${GIVEN[ADDR]:-}" ]      && ADDR=$value ;;
            PARALLEL)  [ -z "${GIVEN[PARALLEL]:-}" ]  && PARALLEL=$value ;;
            MMPROJ)    [ -z "${GIVEN[MMPROJ]:-}" ]    && MMPROJ=$value ;;
            EMBEDDING) [ -z "${GIVEN[EMBEDDING]:-}" ] && EMBEDDING=$value ;;
        esac
    done < "$STATE_FILE"
fi

# A corrupt state file must not silently drop us into "no profile".
case "$PROFILE" in
    ''|aggressive|conservative|manual) : ;;
    *) die "invalid profile '$PROFILE' in $STATE_FILE (expected aggressive, conservative or manual)" ;;
esac

# ------------------------------------------------------------- locate infr ---
# Resolve the launcher's own directory, not the caller's, so the wizard works
# when invoked by path from anywhere — mirroring how the Windows wizard finds
# infr.exe next to itself.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

find_infr() {
    local candidate
    for candidate in \
        "$SCRIPT_DIR/target/release/infr" \
        "$SCRIPT_DIR/infr" \
        ./target/release/infr \
        ./infr \
        "$(command -v infr || true)"; do
        [ -n "$candidate" ] && [ -x "$candidate" ] && { printf '%s\n' "$candidate"; return 0; }
    done
    return 1
}
INFR_BIN="$(find_infr || true)"
[ -n "$INFR_BIN" ] || die "infr not found. Build it first:
  cargo build --release --locked -p infr-cli"

# --------------------------------------------------------------- prompting ---
ask() { # ask <name> <prompt> <default>
    local name="$1" prompt="$2" default="${3-}" answer=""
    if [ -n "$default" ]; then
        read -r -p "$prompt [$default]: " answer || true
        [ -n "$answer" ] || answer="$default"
    else
        read -r -p "$prompt: " answer || true
    fi
    printf -v "$name" '%s' "$answer"
}

choose() { # choose <name> <prompt> <default-index> <option>...
    local name="$1" prompt="$2" default="$3"; shift 3
    local -a options=("$@")
    local i choice
    printf '%s\n' "$prompt"
    for i in "${!options[@]}"; do
        printf '  %d) %s\n' "$((i + 1))" "${options[$i]}"
    done
    read -r -p "Choice [$default]: " choice || true
    case "$choice" in
        ''|*[!0-9]*) choice="$default" ;;
    esac
    [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ] || choice="$default"
    printf -v "$name" '%s' "${options[$((choice - 1))]}"
}

scan_models() { # list nearby *.gguf, first shard first
    find . -maxdepth 2 -type f -name '*.gguf' 2>/dev/null | sort | head -20
}

# The same warning the Windows wizard prints when the server is opened up.
loopback_only() {
    [[ "$1" =~ ^127\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$ ||
       "$1" =~ ^localhost:[0-9]+$ || "$1" =~ ^\[::1\]:[0-9]+$ ]]
}

# Prompt only when there is a terminal to answer on. With --dry-run and no TTY
# the flags (and the saved selections) fully determine the command, so it stays
# scriptable.
INTERACTIVE=0
if [ -t 0 ]; then INTERACTIVE=1; fi

if [ "$INTERACTIVE" -eq 1 ]; then
    printf '\n=== infr Linux wizard (%s) ===\n\n' "$INFR_BIN"

    if [ -z "$MODE" ]; then
        choose MODE 'Start what?' 2 'run (terminal chat)' 'serve (OpenAI-compatible API)' 'bench (benchmark)'
        MODE="${MODE%% *}"
    fi

    if [ -z "$MODEL" ]; then
        printf 'Models found nearby:\n'
        scan_models | sed 's/^/  /' || true
        ask MODEL 'Model path (.gguf, or org/repo[:quant])'
    fi
    [ -n "$MODEL" ] || die "a model is required"

    if [ -z "$PROFILE" ]; then
        choose PROFILE 'Resource profile?' 2 'aggressive' 'conservative' 'manual'
    fi

    case "$PROFILE" in
        manual)
            [ -n "$CTX" ]    || ask CTX    'Context window (e.g. 32768, 256k; blank = engine default)'
            [ -n "$UBATCH" ] || ask UBATCH 'Prefill micro-batch (blank = engine default)'
            [ -n "$KV_K" ]   || ask KV_K   'KV format for K (e.g. q8_0, q4_0, f16; blank = default)'
            [ -n "$KV_V" ]   || ask KV_V   'KV format for V (blank = default)'
            [ -n "$RAM" ]    || ask RAM    'device.ram_budget (e.g. 16GiB; blank = engine default)'
            [ -n "$VRAM" ]   || ask VRAM   'device.vram_budget (e.g. 23GiB; blank = default)'
            ;;
        *)
            [ -n "$KV_K" ] || KV_K="q8_0"
            [ -n "$KV_V" ] || KV_V="q8_0"
            ;;
    esac

    if [ -z "$MTP" ] && [ -z "${GIVEN[MTP]:-}" ]; then
        answer_mtp=""
        ask answer_mtp 'MTP head path (blank = disabled)'
        MTP="$answer_mtp"
    fi

    if [ "$MODE" = serve ]; then
        [ -n "$ADDR" ]     || ask ADDR     'Listen address' '127.0.0.1:8080'
        [ -n "$PARALLEL" ] || ask PARALLEL 'Parallel slots' '1'
        if [ -z "${GIVEN[MMPROJ]:-}" ]; then
            ask MMPROJ 'Vision projector (mmproj, blank = none)' "$MMPROJ"
        fi
        if [ -z "${GIVEN[EMBEDDING]:-}" ]; then
            ask EMBEDDING 'Embedding model (blank = none)' "$EMBEDDING"
        fi
    fi
fi

case "$MODE" in
    run|serve|bench) : ;;
    *) die "mode must be run, serve or bench (got '${MODE:-<empty>}')" ;;
esac
[ -n "$MODEL" ] || die "a model is required"
PROFILE="${PROFILE:-conservative}"
if [ "$MODE" = serve ]; then
    ADDR="${ADDR:-127.0.0.1:8080}"
    PARALLEL="${PARALLEL:-1}"
fi
case "$PROFILE" in
    aggressive|conservative)
        KV_K="${KV_K:-q8_0}"
        KV_V="${KV_V:-q8_0}"
        ;;
esac

# --------------------------------------------------------- API key handling ---
# Only `serve` authenticates. The key travels in the environment, never in the
# command line, so it cannot leak through `ps` or the shell history.
if [ "$MODE" = serve ] && [ "$SKIP_API_KEY" -eq 0 ]; then
    if [ -z "$API_KEY" ] && [ -n "${INFR_API_KEY:-}" ]; then
        API_KEY="$INFR_API_KEY"
    fi
    if [ "$INTERACTIVE" -eq 1 ]; then
        enable_key="n"
        [ -z "$API_KEY" ] || enable_key="y"
        answer_key=""
        read -r -p "Enable Bearer API-key authentication? [$enable_key]: " answer_key || die 'authentication selection aborted'
        enable_key="${answer_key:-$enable_key}"
        case "$enable_key" in
            y|Y|yes|YES)
                key_in=""
                read -r -s -p 'API key (hidden, not saved; Enter reuses inherited key): ' key_in || die 'API key input aborted'
                printf '\n'
                [ -n "$key_in" ] && API_KEY="$key_in"
                [ -n "$API_KEY" ] || die 'API-key authentication requires a nonempty key'
                ;;
            *) API_KEY="" ;;
        esac
    fi
fi

if [ "$MODE" = serve ] && ! loopback_only "$ADDR"; then
    printf '\n%s\n' '本机使用 127.0.0.1；局域网访问可用 0.0.0.0，但应启用 API key。'
    printf '%s\n' 'Use 127.0.0.1 locally. For LAN access use 0.0.0.0 and enable an API key.'
    if [ -z "$API_KEY" ]; then
        printf '%s\n' "warning: $ADDR is reachable from the network and no API key is set." >&2
        if [ "$DRY_RUN" -eq 0 ] && [ "$ASSUME_YES" -eq 0 ]; then
            [ "$INTERACTIVE" -eq 1 ] || die 'unauthenticated network serving requires explicit --yes'
            answer_network=""
            read -r -p 'Continue without authentication? [y/N]: ' answer_network || die 'network launch aborted'
            case "$answer_network" in
                y|Y|yes|YES) : ;;
                *) die 'network launch aborted' ;;
            esac
        fi
    fi
fi

# ----------------------------------------------------------- build command ---
# Everything below only appends to an array. Nothing is ever passed through a
# shell parser, which is why `eval` is not needed.
cmd=("$INFR_BIN" "$MODE")
[ -n "$MODEL" ] && cmd+=("$MODEL")

case "$PROFILE" in
    aggressive|conservative)
        cmd+=(--set "device.auto_profile=$PROFILE")
        ;;
esac
[ -n "$CTX" ]    && cmd+=(--ctx "$CTX")
[ -n "$UBATCH" ] && cmd+=(-u "$UBATCH")
[ -n "$RAM" ]    && cmd+=(--set "device.ram_budget=$RAM")
[ -n "$VRAM" ]   && cmd+=(--set "device.vram_budget=$VRAM")
[ -n "$KV_K" ]   && cmd+=(--set "kv.type_k=$KV_K")
[ -n "$KV_V" ]   && cmd+=(--set "kv.type_v=$KV_V")

if [ -n "$MTP" ]; then
    cmd+=(--set spec.mtp=1 --set "spec.draft=$MTP" --set "spec.k=$MTP_K" --temp 0)
fi

if [ "$MODE" = serve ]; then
    [ -n "$ADDR" ]      && cmd+=(--addr "$ADDR")
    [ -n "$PARALLEL" ]  && cmd+=(-n "$PARALLEL")
    [ -n "$MMPROJ" ]    && cmd+=(--mmproj "$MMPROJ")
    [ -n "$EMBEDDING" ] && cmd+=(--embedding-model "$EMBEDDING")
fi

if [ "$MODE" = bench ]; then
    cmd+=(-p "$BENCH_P" -n "$BENCH_N" -d "$BENCH_D")
fi

print_cmd() {
    printf '\nCommand:\n  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [ -n "$API_KEY" ]; then
        printf '  (API key set; passed through the environment, not shown here)\n'
    fi
    printf '\n'
}

print_cmd

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

if [ "$ASSUME_YES" -eq 0 ]; then
    answer=""
    read -r -p 'Launch? [Y/n]: ' answer || die 'launch confirmation requires input or --yes'
    case "$answer" in
        ''|Y|y|yes|YES) : ;;
        *) printf 'aborted.\n'; exit 0 ;;
    esac
fi

# ------------------------------------------------------------ remember it ----
mkdir -p "$STATE_DIR"
( umask 077; cat > "$STATE_FILE" <<EOF
MODE=$MODE
MODEL=$MODEL
PROFILE=$PROFILE
CTX=$CTX
UBATCH=$UBATCH
KV_K=$KV_K
KV_V=$KV_V
RAM=$RAM
VRAM=$VRAM
MTP=$MTP
MTP_K=$MTP_K
ADDR=$ADDR
PARALLEL=$PARALLEL
MMPROJ=$MMPROJ
EMBEDDING=$EMBEDDING
EOF
) || printf '%s: could not save selections to %s\n' "$PROG" "$STATE_FILE" >&2

if [ -n "$API_KEY" ]; then
    printf '%s\n' 'API key 将通过当前子进程环境传入，未显示在命令中，也不会保存。'
    printf '%s\n' 'The API key is passed through the child-process environment; it is hidden above and not saved.'
    export INFR_API_KEY="$API_KEY"
else
    unset INFR_API_KEY
fi

exec "${cmd[@]}"
