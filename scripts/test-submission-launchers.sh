#!/usr/bin/env bash
# Run with bash on Linux and zsh on macOS. All network operations are fixtures.
set -eu
repo=$1
if [[ -n ${ZSH_VERSION-} ]]; then
    export MOE4ALL_WIZARD_LIBRARY=1
    source "$repo/Start-INFR-Wizard-macOS.sh"
else
    source "$repo/Start-INFR-Wizard-Linux.sh"
fi
if [[ -n ${ZSH_VERSION-} && -n ${MOE4ALL_ZSH_MODULES-} ]]; then module_path=("$MOE4ALL_ZSH_MODULES"); fi
work=$(mktemp -d "${TMPDIR:?}/submission-launchers.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
ROOT=$work/package; CONFIG_DIR=$work/config; BINARY="$work/engine with spaces"
defaults
mkdir -p "$ROOT" "$CONFIG_DIR"
cat > "$BINARY" <<'ENGINE'
#!/bin/sh
if [ "$1" = --version ]; then printf 'infr 1.0.3\n'; exit 0; fi
if [ "$1" = submission ] && [ "$2" = pack-benchmark ]; then
    shift 2
    while [ "$#" -gt 0 ]; do
        case "$1" in --result-dir) result_dir=$2; shift 2 ;; *) shift ;; esac
    done
    mkdir -p "$result_dir/v1/2026-10"
    printf fixture > "$result_dir/v1/2026-10/manual-fixture.m4b"
    printf 'MOE4ALL_SUBMISSION_PATH=%s/v1/2026-10/manual-fixture.m4b\n' "$result_dir"
fi
exit 0
ENGINE
chmod +x "$BINARY"
checks=0
check() { if ! "$@"; then printf 'Failed: %s\n' "$*" >&2; exit 1; fi; checks=$((checks + 1)); }
native_save_definition=$(typeset -f save_state)
eval "${native_save_definition/save_state/native_save_state}"
sleep() { printf '%s\n' "$1" >> "$work/sleeps"; }
save_state() { :; }
yes() { REPLY=true; }
open_submission_url() { printf '%s' "$1" > "$work/opened-url"; }
# BSD wc pads its count; protocol comparisons must not depend on that padding.
wc() { local count; count=$(command wc "$@") || return 1; printf '        %s\n' "$count"; }
original_request=$(typeset -f submission_api_request)
eval "${original_request/submission_api_request/original_submission_api_request}"

gh() {
    printf '%s\n' "$*" >> "$work/gh-calls"
    case "$*" in
        'auth status '*) return 0 ;;
        'api user '*) printf fixture ;;
        'api repos/Headmaster218/MoE4All/git/ref/heads/main '*) printf '%040d' 0 ;;
        'api --method POST '*) [[ ${repeat_git-false} != true ]] ;;
        'api --method PUT '*) [[ ${repeat_git-false} != true ]] ;;
        'api repos/fixture/MoE4All/contents/'*) base64 < "$current_file" ;;
        'api repos/fixture/MoE4All'*) printf '{}' ;;
        'pr create '*) [[ ${repeat_git-false} != true ]] || return 1; printf 'https://github.com/Headmaster218/MoE4All/pull/7' ;;
        'pr list '*) printf 'https://github.com/Headmaster218/MoE4All/pull/7' ;;
        *) printf 'Unexpected fixture gh command\n' >&2; return 1 ;;
    esac
}

submission_api_request() {
    case "$2" in
        */upload-sessions)
            grep -q '"summary":\["fixture summary"\]' "$5" || return 1
            printf '{"api_version":1,"state":"awaiting_verification","session_id":"us_fixture","session_token":"session_fixture","verification_url":"http://127.0.0.1:18765/community/upload.html#session=fixture","max_upload_bytes":131072}' > "$3"
            ;;
        */upload-sessions/us_fixture)
            printf '{"api_version":1,"state":"ready","kind":"%s","report_id":"fixture-%s","file_sha256":"%s","size_bytes":%s}' "$current_kind" "$current_kind" "$current_hash" "$current_size" > "$3"
            ;;
        */report)
            printf '{"api_version":1,"receipt_id":"rc_%s","receipt_token":"private_fixture_token","kind":"%s","report_id":"fixture-%s","file_sha256":"%s","submitted_file_sha256":"%s","payload_sha256":"%s","status":"forwarding","received_at":"2026-10-19T00:00:00Z","public_url":null}' "$current_kind" "$current_kind" "$current_kind" "$original_hash_fixture" "$current_hash" "$current_hash" > "$3"
            ;;
        *) return 1 ;;
    esac
}

export MOE4ALL_SUBMISSION_API_BASE=http://127.0.0.1:18765/v1
for current_kind in benchmark error; do
    extension=m4b; [[ $current_kind != error ]] || extension=m4e
    current_file=$work/fixture-$current_kind.$extension
    printf 'MOE4ALL-SUBMISSION/1\nkind: %s\nid: fixture-%s\ncreated-at: 2026-10-19T00:00:00Z\nsummary-001: fixture summary\n' "$current_kind" "$current_kind" > "$current_file"
    file_sha256 "$current_file"; current_hash=$REPLY
    current_size=$(wc -c < "$current_file"); trim "$current_size"; current_size=$REPLY
    original_hash_fixture=$(printf '%064d' 0)
    repeat_git=false; gh_submission_upload "$current_file" "$current_kind" > "$work/git-output"
    check grep -q "$current_kind-database/v1/2026-10/fixture-$current_kind.$extension" "$work/gh-calls"
    check grep -q "community/$current_kind/$current_hash" "$work/gh-calls"
    repeat_git=true; gh_submission_upload "$current_file" "$current_kind" > "$work/git-repeat-output"
    check grep -q 'already exists' "$work/git-repeat-output"
    website_submission_upload "$current_file" "$current_kind" > "$work/upload-output"
    check test -f "$CONFIG_DIR/private/submission-receipts/rc_$current_kind.private-receipt.json"
    check grep -q 'submitted_file_sha256' "$CONFIG_DIR/private/submission-receipts/rc_$current_kind.private-receipt.json"
    check grep -q "$current_hash" "$CONFIG_DIR/private/submission-receipts/rc_$current_kind.private-receipt.json"
    check grep -q '/community/upload.html#session=' "$work/opened-url"
    check test "$(grep -c '#receipt=' "$work/opened-url" || true)" = 0
    check grep -q 'status: forwarding' "$work/upload-output"
done

yes() { REPLY=false; }
if gh_submission_upload "$current_file" "$current_kind" > "$work/git-cancel-output"; then exit 1; else cancel_status=$?; fi
check test "$cancel_status" = 2

real_gh_submission_upload=$(typeset -f gh_submission_upload)
real_open_github_submission=$(typeset -f open_github_submission)
real_save_state=$(typeset -f save_state)
reset_routes() {
    git_attempts=0; web_attempts=0; browser_attempts=0
    git_failures=0; web_failures=0; signed_in=true; fixture_github=true; cancel_git=false; invalid_git=false
    S[submission_browser_action]=later; S[submission_failure_action]=later
}
gh() { [[ $signed_in == true ]]; }
gh_submission_upload() {
    git_attempts=$((git_attempts + 1))
    [[ $cancel_git != true ]] || return 2
    [[ $invalid_git != true ]] || return 3
    ((git_attempts > git_failures))
}
website_submission_upload() { web_attempts=$((web_attempts + 1)); ((web_attempts > web_failures)); }
open_github_submission() { browser_attempts=$((browser_attempts + 1)); }
yes() { REPLY=$fixture_github; }

reset_routes
offer_submission_upload "$current_file" error true
check test "$git_attempts" = 1
check test "$browser_attempts" = 0
check test "$web_attempts" = 0
reset_routes; git_failures=1
offer_submission_upload "$current_file" error true
check test "$git_attempts" = 2
check test "$browser_attempts" = 0
reset_routes; git_failures=2
if offer_submission_upload "$current_file" error true <<< '4'; then exit 1; fi
check test "$git_attempts" = 2
check test "$browser_attempts" = 1
check test "$web_attempts" = 0
pending_file="$CONFIG_DIR/private/pending-submissions/${current_file##*/}"
check test -f "$pending_file"
reset_routes; git_failures=2
offer_submission_upload "$current_file" error true <<< '3'
check test "$web_attempts" = 1
check test ! -f "$pending_file"
reset_routes; signed_in=false
offer_submission_upload "$current_file" error true <<< $'2\n1'
check test "$git_attempts" = 0
check test "$browser_attempts" = 2
check test "$web_attempts" = 0
reset_routes; fixture_github=false
offer_submission_upload "$current_file" error true
check test "$web_attempts" = 1
check test "$git_attempts" = 0
check test "$browser_attempts" = 0
reset_routes; fixture_github=false; web_failures=1
offer_submission_upload "$current_file" error true <<< '1'
check test "$web_attempts" = 2
reset_routes; fixture_github=false; signed_in=false; web_failures=1
fixture_account_reads=0
yes() { fixture_account_reads=$((fixture_account_reads + 1)); REPLY=false; ((fixture_account_reads < 2)) || REPLY=true; }
offer_submission_upload "$current_file" error true <<< $'2\n1'
check test "$web_attempts" = 1
check test "$browser_attempts" = 1
yes() { REPLY=$fixture_github; }
reset_routes; fixture_github=false; web_failures=1
if offer_submission_upload "$current_file" error true <<< '3'; then exit 1; fi
check test -f "$pending_file"
reset_routes; cancel_git=true
if offer_submission_upload "$current_file" error true; then exit 1; fi
check test "$git_attempts" = 1
check test "$web_attempts" = 0
check test "$browser_attempts" = 0
check test -f "$pending_file"
reset_routes; invalid_git=true
if offer_submission_upload "$current_file" error true; then exit 1; fi
check test "$git_attempts" = 1
check test "$browser_attempts" = 0
check test "$web_attempts" = 0
reset_routes
PENDING_SUBMISSION_PROMPTED=false
yes() { REPLY=true; }
retry_pending_submissions
check test ! -f "$pending_file"
check test -f "$current_file"

# Saving settings can modify REPLY; the selected action must survive it.
save_state() { REPLY=clobbered; }
reset_routes
yes() { REPLY=true; }
offer_submission_upload "$current_file" error
check test "${S[error_upload]}" = true
check test "$git_attempts" = 1
queue_pending_submission "$current_file" error true
PENDING_SUBMISSION_PROMPTED=false
retry_pending_submissions
check test ! -f "$pending_file"
submission_next_action browser <<< '3'
check test "$REPLY" = cloudflare
submission_next_action <<< '2'
check test "$REPLY" = channel
STATE="$CONFIG_DIR/wizard-state.json"
native_save_state
json_load "$STATE"
json_get '["submission_browser_action"]'; check test "$REPLY" = cloudflare
json_get '["submission_failure_action"]'; check test "$REPLY" = channel
eval "$real_save_state"

# Real web form construction is local and never places the report body in the URL.
eval "${real_gh_submission_upload/gh_submission_upload/original_gh_submission_upload}"
eval "$real_open_github_submission"
copy_submission_text() { return 1; }
open_github_submission "$current_file" error > "$work/github-form-output"
check grep -q '/new/main/error-database/v1/2026-10?filename=' "$work/opened-url"
check grep -q '&message=' "$work/opened-url"
check grep -q '&value=' "$work/opened-url"
check grep -q '%43%74%72%6c%2b%41' "$work/opened-url"
check test "$(grep -c 'summary-001' "$work/opened-url" || true)" = 0
check test -f "$current_file"

printf '[{"avg_ts":123.4}]' > "$work/manual.json"
S[model]="$work/model with spaces.gguf"
pack_manual_benchmark "$work/manual.json" > "$work/manual-output"
check test -f "$REPLY"
check grep -q '123.4' "$work/manual-output"

curl() {
    output='' headers=''
    while (($#)); do
        case $1 in --output) output=$2; shift 2 ;; --dump-header) headers=$2; shift 2 ;; --request|--write-out|--header|--data-binary|--connect-timeout|--max-time) shift 2 ;; *) shift ;; esac
    done
    if [[ $retry_mode == short && -f $work/retried ]]; then printf '{}' > "$output"; printf 200
    else printf 'Retry-After: %s\r\n' "$retry_seconds" > "$headers"; touch "$work/retried"; printf '{}' > "$output"; printf 429; fi
}
rm -f "$work/sleeps"; retry_mode=short; retry_seconds=3
original_submission_api_request GET 'http://127.0.0.1:18765/v1/health' "$work/response"
check grep -qx 3 "$work/sleeps"
rm -f "$work/retried" "$work/sleeps"; retry_mode=long; retry_seconds=180
if original_submission_api_request GET 'http://127.0.0.1:18765/v1/health' "$work/response" > "$work/retry-output" 2>&1; then exit 1; fi
check test ! -f "$work/sleeps"
check test -f "$current_file"
printf 'submission-launchers: %s checks passed; GitHub and HTTP are fixtures.\n' "$checks"
