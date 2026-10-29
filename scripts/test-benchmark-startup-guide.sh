#!/usr/bin/env bash
# Run with bash for Linux, or zsh for macOS. Argument 2 supplies the fixture root.
set -eu
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
platform=${1:-linux}
task_root=${2:-${TMPDIR-}}
: "${task_root:?Pass a task root as argument 2 or set TMPDIR}"
export TMPDIR=$task_root TEMP=$task_root TMP=$task_root
if [[ -n ${ZSH_VERSION-} && -n ${MOE4ALL_ZSH_MODULE_PATH-} ]]; then
    module_path=("$MOE4ALL_ZSH_MODULE_PATH" "${module_path[@]}")
fi
case $platform in
    linux) source "$repo/Start-INFR-Wizard-Linux.sh" ;;
    macos) MOE4ALL_WIZARD_LIBRARY=1 source "$repo/Start-INFR-Wizard-macOS.sh" ;;
    *) exit 2 ;;
esac
work=$(mktemp -d "$TMPDIR/guide-$platform.XXXXXX")
unset CI
benchmark_guide_now() { printf '%s\n' "$test_now"; }
check() { if ! eval "$1"; then printf 'FAIL: %s\n' "$2" >&2; exit 1; fi; }
reset_case() {
    ROOT=$work/$1; CONFIG_DIR=$ROOT/gui-data; STATE=$CONFIG_DIR/wizard-state.json
    defaults
    INTERACTIVE=true; HEADLESS=false; DRY_RUN=false; BENCHMARK_GUIDE_ACTIVE=false
    test_now=1800000000
    guide_path=$CONFIG_DIR/private/benchmark-first-submission.json
}
receipt_fixture() {
    mkdir -p "$CONFIG_DIR/private/submission-receipts"
    printf '{"api_version":1,"kind":"%s","receipt_id":"rc_fixture","report_id":"fixture","file_sha256":"%s","payload_sha256":"%s","status":"%s","received_at":"%s","receipt_token":"never-log-this-fixture-secret"}\n' \
        "$1" "$hash_a" "$2" "$3" "$4" > "$CONFIG_DIR/private/submission-receipts/fixture.private-receipt.json"
}
hash_a=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
hash_b=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

reset_case transitions
benchmark_startup_guide <<< '' > "$work/first"
benchmark_guide_read
check '[[ $BENCHMARK_GUIDE_ACTIVE == false && $GUIDE_NEXT_REMIND_AT == 1800172800 ]]' 'Later must defer 48 hours'
grep -Fq '试试！ / Try it!' "$work/first"
grep -Fq '* [2] 后面再说~ / Maybe later~' "$work/first"
first=$(< "$guide_path")
test_now=1800172799
benchmark_startup_guide </dev/null > "$work/cooldown"
check '[[ ! -s $work/cooldown && $(< "$guide_path") == "$first" ]]' 'No prompt/write during cooldown'
test_now=1800172800
benchmark_startup_guide <<< 1 > "$work/try"
benchmark_guide_read
check '[[ $BENCHMARK_GUIDE_ACTIVE == true && $GUIDE_NEXT_REMIND_AT == 1800345600 ]]' 'Try must enter benchmark and retain failure cooldown'
check '[[ ${S[launch_mode]} == bench && ${S[bench_kind]} == quick && ${S[bench_parallel]} == 1 ]]' 'Try must select quick x1'
test_now=1800345600
if benchmark_startup_guide </dev/null > "$work/cancel" 2>&1; then exit 1; fi
benchmark_guide_read
check '[[ $GUIDE_NEXT_REMIND_AT == 1800518400 ]]' 'EOF cancellation must defer 48 hours'
benchmark_submission_succeeded error
benchmark_guide_read; check '[[ $GUIDE_SUBMITTED == false ]]' 'Error upload must not count'
benchmark_submission_succeeded benchmark
benchmark_guide_write false 1800999999
test_now=1900000000
benchmark_startup_guide </dev/null > "$work/disabled"
check '[[ ! -s $work/disabled ]]' 'Success must disable future invitations'
benchmark_guide_read; check '[[ $GUIDE_SUBMITTED == true ]]' 'Reminder rewrite must not erase success'
! grep -Eq 'token|receipt_id' "$guide_path"

for mode in headless dry json noninteractive ci; do
    reset_case "$mode"
    case $mode in headless) HEADLESS=true ;; dry) DRY_RUN=true ;; json) S[json_output]=true ;; noninteractive) INTERACTIVE=false ;; ci) CI=true ;; esac
    benchmark_startup_guide </dev/null > "$work/guard-$mode"
    check '[[ ! -e $CONFIG_DIR && ! -s $work/guard-$mode ]]' "$mode must not prompt or write"
    unset CI
done
for receipt_status in forwarding pending_review published rejected delivery_failed; do
    reset_case "backfill-$receipt_status"
    receipt_fixture benchmark "$hash_b" "$receipt_status" '2026-10-25T00:00:00Z'
    benchmark_startup_guide </dev/null > "$work/backfill"
    benchmark_guide_read
    check '[[ $GUIDE_SUBMITTED == true && ! -s $work/backfill ]]' 'Body-received history must suppress even if forwarding/review later fails'
done
for invalid in error authorization hash received corrupt; do
    reset_case "invalid-$invalid"
    case $invalid in
        error) receipt_fixture error "$hash_b" forwarding '2026-10-25T00:00:00Z' ;;
        authorization) receipt_fixture benchmark "$hash_b" ready '2026-10-25T00:00:00Z' ;;
        hash) receipt_fixture benchmark '' forwarding '2026-10-25T00:00:00Z' ;;
        received) receipt_fixture benchmark "$hash_b" forwarding '' ;;
        corrupt) receipt_fixture benchmark "$hash_b" forwarding ''; printf 'broken' > "$CONFIG_DIR/private/submission-receipts/fixture.private-receipt.json" ;;
    esac
    benchmark_startup_guide <<< '' > "$work/invalid"
    benchmark_guide_read; check '[[ $GUIDE_SUBMITTED == false && -s $work/invalid ]]' "$invalid must not backfill"
done
reset_case browser-pref
S[submission_browser_action]=sent
benchmark_startup_guide <<< '' > "$work/browser-pref.out"
benchmark_guide_read; check '[[ $GUIDE_SUBMITTED == false ]]' 'Saved sent preference is not proof'
submission_next_action browser <<< '' > "$work/browser-default"
check '[[ $REPLY == later ]]' 'Enter must not confirm a different report'

# Mock only external interactions; exercise the real configuration, command and save functions.
reset_case guided-flow
mkdir -p "$CONFIG_DIR"
S[launch_mode]=serve; S[bench_kind]=full; S[bench_parallel]=4; S[server_parallel]=3
S[model]=$ROOT/model.gguf; BINARY=/bin/true
save_state
# Unknown nested data must survive a preference-only merge as well.
printf '{"launch_mode":"serve","bench_kind":"full","bench_parallel":"4","server_parallel":"3","unknown":{"list":[1,"preserved"]}}\n' > "$STATE"
benchmark_startup_guide <<< 1 > "$work/guided"
model_path() { :; }
choose_device() { S[device]=Vulkan0; }
ask_yes() { S[$1]=false; }
guided_profile=conservative
choice() {
    case $1 in
        launch_mode|bench_kind|manual_bench_kind) fail 'Guided benchmark must skip plan selection' ;;
        setup_mode) S[setup_mode]=$guided_profile ;;
    esac
}
configure > "$work/configure"
if [[ $platform == macos ]]; then S[device]=metal; fi
build_command
check '[[ " ${COMMAND[*]} " == *" device.auto_profile=conservative "* && " ${COMMAND[*]} " != *" device.auto_profile=aggressive "* ]]' 'Guided quick benchmark must preserve a conservative selection'
if [[ $platform == linux ]]; then
    check '[[ " ${COMMAND[*]} " == *" --suite quick --parallel 1 "* ]]' 'Linux must build quick suite'
else
    check '[[ ${S[bench_kind]} == manual && ${S[depth_tokens]} == 6144 && ${S[gen_tokens]} == 1024 && ${S[reps]} == 1 ]]' 'Metal must use short manual quick plan'
fi
guided_profile=aggressive
configure > "$work/configure-aggressive"
if [[ $platform == macos ]]; then S[device]=metal; fi
build_command
check '[[ " ${COMMAND[*]} " == *" device.auto_profile=aggressive "* && " ${COMMAND[*]} " != *" device.auto_profile=conservative "* ]]' 'Guided quick benchmark must preserve an aggressive selection'
save_state
json_load "$STATE"
json_get '["launch_mode"]'; check '[[ $REPLY == serve ]]' 'Guided run must preserve daily launch mode'
json_get '["bench_kind"]'; check '[[ $REPLY == full ]]' 'Guided run must preserve daily benchmark kind'
json_get '["bench_parallel"]'; check '[[ $REPLY == 4 ]]' 'Guided run must preserve daily benchmark concurrency'
json_get '["server_parallel"]'; check '[[ $REPLY == 3 ]]' 'Guided run must preserve daily server concurrency'
json_get '["unknown","list",1]'; check '[[ $REPLY == preserved ]]' 'Preference merge must preserve unknown nested settings'
S[benchmark_upload]=true; S[submission_browser_action]=sent
save_state
json_load "$STATE"
json_get '["benchmark_upload"]'; check '[[ $REPLY == true ]]' 'Guided upload consent default must be remembered'
json_get '["submission_browser_action"]'; check '[[ $REPLY == sent ]]' 'Guided browser action default must be remembered'
json_get '["launch_mode"]'; check '[[ $REPLY == serve ]]' 'Upload preference merge must preserve daily mode'
benchmark_guide_read; check '[[ $GUIDE_SUBMITTED == false ]]' 'Benchmark execution alone must not count'

# The actual Cloudflare client must mark only after a valid POST /report response.
ensure_submission_curl() { :; }
open_submission_url() { return 0; }
sleep() { :; }
submission_api_request() {
    local method=$1 url=$2 output=$3
    case $url in
        */upload-sessions)
            printf '{"api_version":1,"state":"awaiting_verification","session_id":"us_fixture","session_token":"private-fixture","verification_url":"https://moe4all.com/community/upload.html#fixture","max_upload_bytes":131072}' > "$output" ;;
        */report)
            [[ $body_ok == true ]] || return 1
            printf '{"api_version":1,"receipt_id":"rc_fixture","receipt_token":"private-fixture","kind":"%s","report_id":"fixture","file_sha256":"%s","payload_sha256":"%s","status":"forwarding","received_at":"2026-10-25T00:00:00Z"}' "$report_kind" "$report_hash" "$hash_b" > "$output" ;;
        *)
            printf '{"api_version":1,"state":"ready","kind":"%s","report_id":"fixture","file_sha256":"%s","size_bytes":%s}' "$report_kind" "$report_hash" "$report_size" > "$output" ;;
    esac
}
for report_kind in benchmark error; do
    reset_case "cloudflare-$report_kind"
    mkdir -p "$ROOT"; BINARY=/bin/true
    extension=m4b; [[ $report_kind != error ]] || extension=m4e
    report=$ROOT/fixture.$extension
    printf 'MOE4ALL-SUBMISSION/1\nkind: %s\nid: fixture\n' "$report_kind" > "$report"
    file_sha256 "$report"; report_hash=$REPLY; report_size=$(wc -c < "$report"); trim "$report_size"; report_size=$REPLY
    body_ok=false
    if website_submission_upload "$report" "$report_kind" > "$work/authorization-only" 2>&1; then exit 1; fi
    check '[[ ! -e $guide_path ]]' 'Authorization alone must not count'
    body_ok=true
    website_submission_upload "$report" "$report_kind" > "$work/body-received"
    benchmark_guide_read
    if [[ $report_kind == benchmark ]]; then
        check '[[ $GUIDE_SUBMITTED == true ]]' 'Valid body receipt must count immediately'
    else
        check '[[ $GUIDE_SUBMITTED == false && ! -e $guide_path ]]' 'Error body receipt must not count'
    fi
done

# Exercise PR validation without allowing the real gh executable to run.
mkdir -p "$work/bin"
printf '#!/bin/sh\nexit 99\n' > "$work/bin/gh"; chmod +x "$work/bin/gh"
PATH=$work/bin:$PATH
submission_gh() {
    case "$*" in
        'api user --jq .login') printf 'fixture-user\n' ;;
        *'.object.sha'*) printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' ;;
        'pr create '*) printf '%s\n' "$pr_response" ;;
        *) return 0 ;;
    esac
}
for report_kind in benchmark error; do
    for valid in false true; do
        reset_case "github-$report_kind-$valid"
        mkdir -p "$CONFIG_DIR"; BINARY=/bin/true
        report=$ROOT/fixture.m4b
        printf 'MOE4ALL-SUBMISSION/1\nkind: %s\nid: fixture\ncreated-at: 2026-10-25T00:00:00Z\n' "$report_kind" > "$report"
        pr_response=not-a-pr
        [[ $valid != true ]] || pr_response=https://github.com/Headmaster218/MoE4All/pull/123
        if gh_submission_upload "$report" "$report_kind" <<< y > "$work/github-response"; then
            check '[[ $valid == true ]]' 'Invalid PR response was accepted'
        else
            check '[[ $valid == false ]]' 'Valid PR response was rejected'
        fi
        benchmark_guide_read
        if [[ $valid == true && $report_kind == benchmark ]]; then
            check '[[ $GUIDE_SUBMITTED == true ]]' 'Valid benchmark PR must count'
        else
            check '[[ $GUIDE_SUBMITTED == false ]]' 'Invalid PR or error upload must not count'
        fi
    done
done

# Explicit browser self-confirmation in the real dispatcher counts; declining consent does not.
submission_gh() { return 1; }
queue_pending_submission() { :; }
open_github_submission() { :; }
submission_next_action() { REPLY=sent; }
for report_kind in benchmark error; do
    reset_case "browser-sent-$report_kind"
    BINARY=/bin/true
    if offer_submission_upload "$ROOT/fixture.m4b" "$report_kind" <<< n > "$work/declined"; then exit 1; fi
    check '[[ ! -e $guide_path ]]' 'Declined upload consent must not mark success'
    offer_submission_upload "$ROOT/fixture.m4b" "$report_kind" <<< $'y\ny\n' > "$work/browser-sent"
    benchmark_guide_read
    if [[ $report_kind == benchmark ]]; then
        check '[[ $GUIDE_SUBMITTED == true ]]' 'Explicit browser sent must count for benchmark'
    else
        check '[[ $GUIDE_SUBMITTED == false ]]' 'Explicit error browser sent must not count'
    fi
done
printf 'PASS %s benchmark startup guide: clocks, guards, backfill, explicit browser confirmation, preserved defaults and upload preferences, GitHub PR validation and Cloudflare body receipt\n' "$platform"
