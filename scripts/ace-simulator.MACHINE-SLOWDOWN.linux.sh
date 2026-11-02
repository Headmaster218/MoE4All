#!/bin/bash
#
# ace-simulator.MACHINE-SLOWDOWN.linux.sh - DANGER, opt-in, READ-ONLY, ***EXPERIMENTAL***
#
# Linux port of scripts/ace-simulator.MACHINE-SLOWDOWN.ps1. Same joke, same five fences, same exit
# codes - and the same rule that nothing here is wired into Start-INFR-Wizard.cmd, infr-wizard.ps1,
# the engine, the release scripts or CI. If you want it in a startup path, do not file a bug about
# it; that is the one version of this that would hurt other people.
#
# EXPERIMENTAL, in the precise sense: written on Windows, so this file was exercised through Git Bash
# (a Linux-ish userland: /proc/mounts, sha256sum, stat -c%s, GNU find/xargs all present) rather than
# on a Linux box. The volume enumeration therefore ran against MSYS's own mount table here, and the
# real thing on a distro is a one-line question of whether /proc/mounts is what it says it is. Read
# the PR's test record before trusting this one; the macOS sibling is even less verified.
#
# What it does: enumerates the filesystems the kernel has mounted (asked at run time, nothing
# hardcoded), walks them for loadable model files, SHA-256s every one across every logical CPU at
# once, asks HuggingFace exactly one HEAD - are you there? - falling back out loud to hf-mirror.com
# if the hub is silent, pauses like a real verification, and reports CHECK FAILED.
#
# Fences:
#   * nothing calls this file;
#   * --i-accept is required to reach the prompts, and is NOT an answer to them;
#   * a typed `y` before the scan. Enter is No, any other key is No, and a stdin that is not a TTY
#     is No, because a pipe is not a human;
#   * a second typed `y` before the hashing, once the bill can be quoted in real numbers;
#   * --dry-run prints the plan and stops; --root / --ext / --limit-files shrink the blast radius.
# Neither answer and neither timeout is stored anywhere: there is no config file and no state.
#
# Read-only by construction: files are opened for reading, nothing is written, moved or deleted
# outside its own temp directory, and no hash ever leaves the machine.
#
# Exit codes: 0 clean, 1 neither endpoint answered and no --tolerate-offline / INFR_HF_OFFLINE,
# 2 stopped at a fence, 3 somebody answered and verification still FAILED (a HEAD cannot check a
# hash).
#
# Usage:
#   bash scripts/ace-simulator.MACHINE-SLOWDOWN.linux.sh --dry-run
#   bash scripts/ace-simulator.MACHINE-SLOWDOWN.linux.sh --i-accept --root /home --limit-files 8
#

set -u

ROOTS=()
EXTS=(gguf mmproj)
DEGREE=0
HUB_ENDPOINT="${INFR_HF_ENDPOINT:-https://huggingface.co}"
MIRROR_ENDPOINT="${INFR_HF_MIRROR:-https://hf-mirror.com}"
HANG_SEC=30
CONNECT_TIMEOUT_SEC=10
PROMPT_TIMEOUT_SEC=300
LIMIT_FILES=0
DRY_RUN=0
I_ACCEPT=0
TOLERATE_OFFLINE=0

WORK=''
FILE_LIST=''
FILE_COUNT=0
TOTAL_BYTES=0
ROOTS_WERE_GIVEN=0
TOTAL_GIB='0.00'
HASH_OUT=''
HASHED_COUNT=0
HASH_SECONDS=0

usage() {
    sed -n '3,45p' "$0"
}

die() { printf '%s\n' "$1" >&2; exit "${2:-2}"; }

cleanup() { [ -n "$WORK" ] && rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --root)                ROOTS+=("$2"); shift 2 ;;
            --ext)                 EXTS+=("$2"); shift 2 ;;
            --degree)              DEGREE="$2"; shift 2 ;;
            --hub-endpoint)        HUB_ENDPOINT="$2"; shift 2 ;;
            --mirror-endpoint)     MIRROR_ENDPOINT="$2"; shift 2 ;;
            --hang-sec)            HANG_SEC="$2"; shift 2 ;;
            --connect-timeout-sec) CONNECT_TIMEOUT_SEC="$2"; shift 2 ;;
            --prompt-timeout-sec)  PROMPT_TIMEOUT_SEC="$2"; shift 2 ;;
            --limit-files)         LIMIT_FILES="$2"; shift 2 ;;
            --dry-run)             DRY_RUN=1; shift ;;
            --i-accept)            I_ACCEPT=1; shift ;;
            --tolerate-offline)    TOLERATE_OFFLINE=1; shift ;;
            -h|--help)             usage; exit 0 ;;
            *) die "  unknown option: $1 (see --help)" 2 ;;
        esac
    done
}

# ── the four things a Linux box does differently ─────────────────────────────
logical_cpus() { nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4; }
file_size()    { stat -c%s "$1" 2>/dev/null || echo 0; }

# Real storage, not the kernel's bookkeeping: tmpfs, proc, sysfs, cgroup, overlay, squashfs (snaps
# and flatpak loop images) and the rest are filtered by FSTYPE instead of by path, because guessing
# "which mount points look like disks" is how a script ends up hashing 40 GiB of snap squashfs.
# findmnt knows this better than an awk does, so use it when it is installed; /proc/mounts is the
# fallback that exists on every kernel.
detect_roots() {
    local real='^(ext[234]|xfs|btrfs|f2fs|jfs|reiserfs|vfat|exfat|ntfs|ntfs3|fuseblk|cifs|nfs|nfs4|smb3|smb2)$'
    if command -v findmnt >/dev/null 2>&1; then
        findmnt -rn -o TARGET -t "$(echo "$real" | sed 's/[()^$]//g; s/|/,/g')" 2>/dev/null
    fi
    if [ -r /proc/mounts ]; then
        awk -v re="$real" '$3 ~ re && $2 ~ /^\// {print $2}' /proc/mounts | sort -u
    fi
}

banner() {
    local auto='asked the kernel at run time, nothing hardcoded'
    [ "$ROOTS_WERE_GIVEN" = 1 ] && auto='named by --root'
    printf '\n'
    printf '  ace-simulator.MACHINE-SLOWDOWN.linux.sh  -  DANGER, opt-in, read-only\n'
    printf '  EXPERIMENTAL Linux port - exercised under Git Bash, not on a Linux box. Expect surprises.\n'
    printf '  volumes: %s root(s), %s\n' "${#ROOTS[@]}" "$auto"
    local r
    for r in "${ROOTS[@]}"; do
        printf '           %s\n' "$(df -Pk "$r" 2>/dev/null | awk 'NR==2 && $6 ~ /^\// {printf "%-22s %8.1f GiB total, %8.1f GiB free", $6, ($2*1024)/1073741824, ($4*1024)/1073741824}')"
    done
    printf '  types:   %s\n' "${EXTS[*]}"
    printf '  hash fs: %s workers (one per logical CPU)\n' "$DEGREE"
    printf '  hub:     %s (one HEAD, then %s only if the hub stays silent)\n' "$HUB_ENDPOINT" "$MIRROR_ENDPOINT"
}

# Ask on the console, with a deadline, and mean No on every ambiguity. A stdin that is not a TTY is
# not a human typing `y`, so a pipe cannot start this - same rule as the PowerShell version, same
# reason: the point of the prompt is that a person is present.
ask_or_refuse() {
    local question="$1" what="$2" ans='' timed=0 reason=''
    printf '%s' "$question"
    if [ ! -t 0 ]; then
        printf '\n'
        reason='no terminal attached to stdin'
    else
        if [ "$PROMPT_TIMEOUT_SEC" -gt 0 ]; then
            read -r -s -n 1 -t "$PROMPT_TIMEOUT_SEC" ans 2>/dev/null || timed=1
        else
            read -r -s -n 1 ans 2>/dev/null || timed=1
        fi
        printf '\n'
        case "$ans" in y|Y) return 0 ;; esac
        [ "$timed" = 1 ] && reason='timeout'
        [ -z "$reason" ] && reason="answered '${ans:-Enter}'"
    fi
    printf '\n' >&2
    case "$reason" in
        timeout)
            if [ "$PROMPT_TIMEOUT_SEC" -gt 0 ]; then
                printf '  No answer in %ss - the default is No, so %s did not happen.\n' "$PROMPT_TIMEOUT_SEC" "$what" >&2
            else
                printf '  No answer - the default is No, so %s did not happen.\n' "$what" >&2
            fi ;;
        'no terminal attached to stdin')
            printf '  No terminal on stdin - a pipe is not a human, so %s did not happen.\n' "$what" >&2 ;;
        *)
            printf '  %s - only a typed y is a yes, so %s did not happen.\n' "$reason" "$what" >&2 ;;
    esac
    printf '  (Nothing about this answer is saved: the next run asks again, defaulting to No.)\n'
    exit 2
}

phase1_scan() {
    printf '\n[1/3] scanning the volumes...\n'
    : > "$FILE_LIST"
    local name_args=() e r before first=1
    # find joins tests with an implicit AND, so "-name a -name b" matches nothing at all: the
    # extensions have to be OR'ed explicitly, which is the bug a first harness run caught.
    for e in "${EXTS[@]}"; do
        [ "$first" = 1 ] || name_args+=(-o)
        name_args+=(-name "*.$e")
        first=0
    done
    for r in "${ROOTS[@]}"; do
        before=$FILE_COUNT
        # -print0 / read -d '' survives spaces and newlines in paths; -xdev keeps one filesystem on
        # its own filesystem; -type f never matches a symlink, so a link cannot drag this off to
        # another disk (the same reparse-point guard the PowerShell version has, in its own idiom).
        while IFS= read -r -d '' f; do
            if [ "$LIMIT_FILES" -gt 0 ] && [ "$FILE_COUNT" -ge "$LIMIT_FILES" ]; then break; fi
            printf '%s\0' "$f" >> "$FILE_LIST"
            FILE_COUNT=$((FILE_COUNT + 1))
            TOTAL_BYTES=$((TOTAL_BYTES + $(file_size "$f")))
        done < <(find "$r" -xdev -type f \( "${name_args[@]}" \) -print0 2>/dev/null)
        if [ "$FILE_COUNT" = "$before" ]; then
            printf '      nothing under %s (empty, unreadable or gone)\n' "$r"
        fi
    done
    TOTAL_GIB=$(awk -v b="$TOTAL_BYTES" 'BEGIN {printf "%.2f", b/1073741824}')
    printf '      %s files, %s GiB\n' "$FILE_COUNT" "$TOTAL_GIB"
}

phase2_hash() {
    printf '\n[2/3] hashing in parallel (this is the part that stutters)...\n'
    local start=$SECONDS rate
    # GNU xargs takes -0 -n1 -P just like the BSD one, which is the whole reason this phase is a one
    # liner instead of a job pool: the parallelism is the OS's job, the stutter is the joke.
    xargs -0 -n 1 -P "$DEGREE" sha256sum < "$FILE_LIST" > "$HASH_OUT" 2> "$WORK/errors"
    HASH_SECONDS=$((SECONDS - start))
    HASHED_COUNT=$(grep -c . "$HASH_OUT" 2>/dev/null)
    [ -n "$HASHED_COUNT" ] || HASHED_COUNT=0
    rate=$(awk -v g="$TOTAL_GIB" -v s="$HASH_SECONDS" 'BEGIN {printf "%.2f", g/(s<0.001?0.001:s)}')
    printf '      %s/%s files, %s GiB in %ss = %s GiB/s\n' "$HASHED_COUNT" "$FILE_COUNT" "$TOTAL_GIB" "$HASH_SECONDS" "$rate"
    if [ -s "$WORK/errors" ]; then
        while IFS= read -r line; do printf '      unreadable: %s\n' "$line" >&2; done < "$WORK/errors"
    fi
}

# Any HTTP status at all is an answer - 200, a 405 on a verb it dislikes, a 429 on a burst. Only a
# transport failure (DNS, refused, timeout) means "not there". One HEAD: no body, no path, no query,
# no retry, and never a per-model metadata fetch, which is what an earlier revision of this joke got
# wrong. Echoes the detail when something answered; says the reason on stderr and returns 1 if not.
probe() {
    local url="$1" code
    if command -v curl >/dev/null 2>&1; then
        code=$(curl --head --silent --max-time "$CONNECT_TIMEOUT_SEC" -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
        if [ -n "$code" ] && [ "$code" != "000" ]; then printf 'HTTP %s' "$code"; return 0; fi
        printf 'curl got no HTTP response for %s\n' "$url" >&2
        return 1
    fi
    if command -v wget >/dev/null 2>&1; then
        # wget --spider is a HEAD as well, and a non-2xx makes it exit non-zero - which is a status it
        # disliked, not an outage - so look for the response line instead of the exit code.
        if wget --spider --server-response --quiet --timeout="$CONNECT_TIMEOUT_SEC" "$url" 2>&1 | grep -q 'HTTP/'; then
            printf 'HTTP (wget)'
            return 0
        fi
        printf 'wget got no HTTP response for %s\n' "$url" >&2
        return 1
    fi
    printf 'no HTTP client installed (curl or wget)\n' >&2
    return 1
}

phase3_verdict() {
    printf '\n[3/3] asking %s whether it is there (one HEAD - that is the whole ask)...\n' "$HUB_ENDPOINT" >&2
    local offline_reason='' answered_via='' detail mirror_detail='' verdict
    if [ "${INFR_HF_OFFLINE:-0}" != "0" ] && [ -n "${INFR_HF_OFFLINE:-}" ]; then
        # Declaring yourself offline also means "do not die on me" - the same rule the PowerShell
        # version has, so the env var never becomes the thing that kills the run.
        TOLERATE_OFFLINE=1
        offline_reason='INFR_HF_OFFLINE is set, so the question is not asked at all'
    else
        if detail=$(probe "$HUB_ENDPOINT"); then
            answered_via="$HUB_ENDPOINT"
            printf '      %s answered (%s) - and gets no second request.\n' "$HUB_ENDPOINT" "$detail"
        else
            printf '      %s stayed silent.\n' "$HUB_ENDPOINT" >&2
            printf '      so, like a real client would: asking %s instead.\n' "$MIRROR_ENDPOINT" >&2
            if mirror_detail=$(probe "$MIRROR_ENDPOINT"); then
                answered_via="$MIRROR_ENDPOINT"
                printf '      %s answered (%s). Mirror verdicts are worth exactly as much.\n' "$MIRROR_ENDPOINT" "$mirror_detail"
            else
                offline_reason=$(printf 'neither %s nor %s answered' "$HUB_ENDPOINT" "$MIRROR_ENDPOINT")
            fi
        fi
    fi

    if [ -n "$offline_reason" ]; then
        printf '      unreachable: %s\n' "$offline_reason" >&2
        if [ "$TOLERATE_OFFLINE" != 1 ]; then
            printf '\n' >&2
            printf '  Not one hash can be confirmed, so this is where it dies. That is the feature.\n' >&2
            printf '  (Escape hatch: --tolerate-offline, or INFR_HF_OFFLINE=1 to stay and enjoy the pause.)\n'
            exit 1
        fi
        printf '      staying on the offline branch, as asked - the pause still applies.\n'
    else
        printf '      something is out there (%s). Nothing was compared, because one HEAD cannot check a hash.\n' "$answered_via"
    fi

    if [ "$HANG_SEC" -gt 0 ]; then
        printf '      contacting the hub for the authoritative verdict... (%ss)\n' "$HANG_SEC"
        sleep "$HANG_SEC"
    fi

    verdict='CHECK FAILED'
    [ -n "$offline_reason" ] && verdict='NOT CHECKED (offline)'
    # The sum tools print "<hash>  <path>", but a coreutils that keeps a text/binary split (MSYS,
    # Cygwin, and `sha256sum` on some distros in binary mode) writes "<hash> *<path>" in that gap.
    # Accept both shapes, and keep the path whole.
    local path
    while IFS= read -r line; do
        path=${line#* }
        printf '      %s %s\n' "$verdict" "${path#\*}"
    done < "$HASH_OUT"

    printf '\ndone: verdict=%s checked=0 files=%s gib=%s hash-seconds=%s\n' "$verdict" "$HASHED_COUNT" "$TOTAL_GIB" "$HASH_SECONDS"
    printf '  At most two HEADs left this machine, no hash ever did, and the feeling is intact.\n'
    if [ -z "$offline_reason" ]; then
        printf '  Answered and still failed: the hub is innocent, the script is the anti-cheat.\n'
        exit 3
    fi
    exit 0
}

main() {
    parse_args "$@"
    [ "$DEGREE" -gt 0 ] || DEGREE=$(logical_cpus)
    if [ ${#ROOTS[@]} -eq 0 ]; then
        local v
        while IFS= read -r v; do
            [ -d "$v" ] && ROOTS+=("$v")
        done < <(detect_roots)
    else
        ROOTS_WERE_GIVEN=1
    fi
    if [ ${#ROOTS[@]} -eq 0 ]; then
        die '  No mounted filesystem could be enumerated, and --root named none. Nothing to do.' 0
    fi

    banner
    if [ "$DRY_RUN" = 1 ]; then
        printf '  --dry-run: nothing hashed, nothing asked, nobody prompted.\n'
        exit 0
    fi
    if [ "$I_ACCEPT" != 1 ]; then
        printf '\n' >&2
        printf '  Refusing to start: this pins every core and reads every disk until it is done.\n' >&2
        printf '  Re-run with --i-accept when you really mean it (or --dry-run to see the plan).\n' >&2
        exit 2
    fi

    printf '\n' >&2
    printf '  WARNING - this is the machine-slowdown one.\n' >&2
    printf '  It will recursively read every file under: %s\n' "${ROOTS[*]}" >&2
    printf '  ...and hash them on all %s logical CPUs at once.\n' "$DEGREE" >&2
    printf '  Disk and CPU go to 100%% for as long as it takes. Nothing is written, but nothing is\n' >&2
    printf '  fast either: your other work - a running engine, a compile, a desktop - stalls.\n' >&2
    printf '  --root / --limit-files shrink that; Ctrl+C works; the answer here is not remembered.\n'
    ask_or_refuse '  Continue? [y/N] ' 'the scan'

    WORK=$(mktemp -d "${TMPDIR:-/tmp}/ace-linux.XXXXXX") || die '  cannot make a temp directory' 2
    FILE_LIST="$WORK/list"
    HASH_OUT="$WORK/hashes"
    phase1_scan
    if [ "$FILE_COUNT" -eq 0 ]; then
        printf '      nothing to hash; exiting clean.\n'
        exit 0
    fi

    printf '\n' >&2
    printf '      the scan is done: %s model file(s), %s GiB, about to be read to the end by %s workers.\n' \
        "$FILE_COUNT" "$TOTAL_GIB" "$DEGREE" >&2
    printf '      This is the part that makes the mouse stutter. Nothing has been hashed yet.\n' >&2
    ask_or_refuse '      Hash them all? [y/N] ' 'any hashing'

    phase2_hash
    phase3_verdict
}

# Sourced (a harness) or executed (a person): only a person runs the thing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
