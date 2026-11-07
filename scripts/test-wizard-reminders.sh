#!/usr/bin/env bash
# bash ... WORK linux, or zsh ... WORK macos. Local clocks/HTTP/updater fixtures only.
set -eu
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=${1:?Pass a task directory below MOE4ALL_WORK_ROOT/tmp/moe4all}
platform=${2:-linux}
case $work in "${MOE4ALL_WORK_ROOT:?}/tmp/moe4all/"*) ;; *) exit 1 ;; esac
case /$work/ in */../*|*/./*) exit 1 ;; esac
case $platform in
    linux) source "$repo/Start-INFR-Wizard-Linux.sh" ;;
    macos) MOE4ALL_WIZARD_LIBRARY=1 source "$repo/Start-INFR-Wizard-macOS.sh" ;;
    *) exit 2 ;;
esac
mkdir -p -- "$work"
work=$(mktemp -d "$work/reminders-$platform.XXXXXX")
export TMPDIR=$work TEMP=$work TMP=$work
unset CI
checks=0
check() { if ! eval "$1"; then printf 'FAIL: %s\n' "$2" >&2; exit 1; fi; checks=$((checks + 1)); }
benchmark_guide_now() { printf '%s\n' "$test_now"; }
update_reminder_now() { printf '%s\n' "$test_now"; }
reset_case() {
    ROOT=$work/$1; CONFIG_DIR=$ROOT/gui-data; STATE=$CONFIG_DIR/wizard-state.json; BINARY=$ROOT/infr
    LEGACY_CONFIG_DIR=''; LEGACY_STATE=''
    defaults
    INTERACTIVE=true; HEADLESS=false; DRY_RUN=false
    test_now=1800000000; test_release=1.4.0
    export REMINDER_TEST_VERSION=1.3.0
    mkdir -p -- "$ROOT"
    cat > "$BINARY" <<'ENGINE'
#!/bin/sh
printf 'infr %s\n' "$REMINDER_TEST_VERSION"
ENGINE
    chmod +x "$BINARY"
    touch "$ROOT/install-manifest.json"
    guide=$CONFIG_DIR/private/benchmark-first-submission.json
    update_state=$CONFIG_DIR/private/update-reminder.json
    output=$ROOT/output
}
write_legacy() {
    mkdir -p -- "$CONFIG_DIR/private"
    printf '{"schema_version":1,"submitted":%s,"next_remind_at":%s}\n' "$1" "$2" > "$guide"
}
check_prompt() {
    benchmark_startup_guide <<< 2 > "$output"
    benchmark_guide_read
    check '[[ -s $output && $GUIDE_LAST_PROMPT_VERSION == "$REMINDER_TEST_VERSION" ]]' 'Expected version invitation'
}

reset_case fresh
benchmark_installed_version
check '[[ $REPLY == 1.3.0 ]]' 'Installed version must come from the actual executable'
check_prompt
check '[[ $GUIDE_REMINDER_REASON == first_use && $GUIDE_NEXT_REMIND_AT == 1800172800 ]]' 'First invitation retains 48 hours'
json_load "$guide"; json_get '["schema_version"]'; check '[[ $REPLY == 2 ]]' 'New state uses schema2'
before=$(< "$guide"); test_now=1800172799
benchmark_startup_guide </dev/null > "$output"
check '[[ ! -s $output && $(< "$guide") == "$before" ]]' 'First-use cooldown is read only before its boundary'
test_now=1800172800; check_prompt
check '[[ $GUIDE_NEXT_REMIND_AT == 1800345600 ]]' '48-hour boundary is inclusive'

# Upgrade interrupts a first-use cooldown; the saved reason keeps all later delays at seven days.
export REMINDER_TEST_VERSION=1.4.0
check_prompt
check '[[ $GUIDE_REMINDER_REASON == new_version && $GUIDE_NEXT_REMIND_AT == 1800777600 ]]' 'New version overrides old cooldown with seven days'
grep -Fq '已更新到v1.4.0' "$output"
test_now=1800777599; before=$(< "$guide")
benchmark_startup_guide </dev/null > "$output"
check '[[ ! -s $output && $(< "$guide") == "$before" ]]' 'Seven-day cooldown has no prompt or write'
test_now=1800777600; check_prompt
check '[[ $GUIDE_NEXT_REMIND_AT == 1801382400 && $GUIDE_REMINDER_REASON == new_version ]]' 'Same version keeps the seven-day reason after its boundary'
test_now=1801382400
if benchmark_startup_guide </dev/null > "$output" 2>&1; then exit 1; fi
benchmark_guide_read
check '[[ $GUIDE_NEXT_REMIND_AT == 1801987200 ]]' 'EOF also retains the seven-day reason'
benchmark_submission_succeeded benchmark 1.3.0
check_prompt_after=$GUIDE_NEXT_REMIND_AT
benchmark_guide_read
check '[[ $GUIDE_SUBMITTED == true && $GUIDE_LAST_SUBMITTED_VERSION == 1.3.0 && $GUIDE_NEXT_REMIND_AT == "$check_prompt_after" ]]' 'Old pending upload preserves current cooldown'
benchmark_submission_succeeded benchmark ''
benchmark_guide_read
check '[[ $GUIDE_LAST_SUBMITTED_VERSION != 1.4.0 ]]' 'Explicit unknown upload cannot mark the current version'
benchmark_submission_succeeded benchmark
benchmark_guide_read
check '[[ $GUIDE_LAST_SUBMITTED_VERSION == 1.4.0 ]]' 'Omitted legacy argument may use the current known version'
test_now=1900000000; benchmark_startup_guide </dev/null > "$output"
check '[[ ! -s $output ]]' 'Current version success stops this version only'
benchmark_submission_succeeded benchmark 1.2.0
benchmark_guide_read
check '[[ $GUIDE_LAST_SUBMITTED_VERSION == 1.4.0 ]]' 'Older pending success cannot erase current success'
export REMINDER_TEST_VERSION=1.5.0; check_prompt
check '[[ $GUIDE_SUBMITTED == true && $GUIDE_REMINDER_REASON == new_version && $GUIDE_NEXT_REMIND_AT == 1900604800 ]]' 'Historical successful users receive the next-version seven-day invitation'

reset_case legacy-submitted
write_legacy true 1999999999
check_prompt
check '[[ $GUIDE_SUBMITTED == true && -z $GUIDE_LAST_SUBMITTED_VERSION && $GUIDE_REMINDER_REASON == new_version && $GUIDE_NEXT_REMIND_AT == 1800604800 ]]' 'Legacy submitted=true is history, not current-version proof; overrides old cooldown'
reset_case legacy-first-use
write_legacy false 1800172800
before=$(< "$guide"); benchmark_startup_guide </dev/null > "$output"
check '[[ ! -s $output && $(< "$guide") == "$before" ]]' 'Legacy first-use retains the remaining 48-hour cooldown without migration writes'
test_now=1800172800; check_prompt
check '[[ $GUIDE_REMINDER_REASON == first_use ]]' 'Legacy unsubmitted migration remains first use'
reset_case corrupt
mkdir -p -- "$CONFIG_DIR/private"; printf broken > "$guide"
check_prompt
check '[[ $GUIDE_REMINDER_REASON == first_use ]]' 'Corrupt state safely recovers'
reset_case saved-json-export
S[json_output]=true
check_prompt
benchmark_submission_succeeded benchmark 1.3.0
benchmark_guide_read
check '[[ $GUIDE_LAST_SUBMITTED_VERSION == 1.3.0 ]]' 'Saved JSON export must not suppress either interactive invitations or real success records'
export REMINDER_TEST_VERSION=1.4.0
check_prompt
check '[[ $GUIDE_REMINDER_REASON == new_version ]]' 'Saved JSON export must not suppress the upgraded-version invitation'

hash=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
for receipt_version in unknown 1.2.0 1.3.0; do
    reset_case "receipt-$receipt_version"
    mkdir -p -- "$CONFIG_DIR/private/submission-receipts"
    extra=''; [[ $receipt_version == unknown ]] || extra=",\"engine_version\":\"$receipt_version\""
    printf '{"api_version":1,"kind":"benchmark","receipt_id":"rc_fixture","report_id":"fixture","file_sha256":"%s","payload_sha256":"%s","status":"forwarding","received_at":"2026-10-25T00:00:00Z"%s}\n' \
        "$hash" "$hash" "$extra" > "$CONFIG_DIR/private/submission-receipts/fixture.private-receipt.json"
    benchmark_startup_guide <<< 2 > "$output"
    benchmark_guide_read
    check '[[ $GUIDE_SUBMITTED == true ]]' 'A valid body receipt always proves historical success'
    if [[ $receipt_version == 1.3.0 ]]; then
        check '[[ ! -s $output && $GUIDE_LAST_SUBMITTED_VERSION == 1.3.0 ]]' 'A current-version receipt suppresses the invitation'
    else
        check '[[ -s $output && $GUIDE_LAST_SUBMITTED_VERSION != 1.3.0 && $GUIDE_REMINDER_REASON == new_version ]]' 'Unknown/old receipt cannot suppress current invitation'
        before=$(< "$guide"); test_now=1800000001
        benchmark_startup_guide </dev/null > "$output"
        check '[[ ! -s $output && $(< "$guide") == "$before" ]]' 'Receipt recovery does not reset the cooldown on every launch'
    fi
done

# Moving from the old user directory does not reset private history or cooldowns.
reset_case legacy-reminder-directory
LEGACY_CONFIG_DIR=$ROOT/legacy
mkdir -p "$LEGACY_CONFIG_DIR/private"
legacy_guide=$LEGACY_CONFIG_DIR/private/benchmark-first-submission.json
printf '{"schema_version":2,"submitted":true,"next_remind_at":1800604800,"last_prompt_version":"1.3.0","last_submitted_version":"1.2.0","reminder_reason":"new_version"}\n' > "$legacy_guide"
legacy_before=$(< "$legacy_guide")
benchmark_startup_guide </dev/null > "$output"
check '[[ ! -s $output && ! -e $guide && $(< "$legacy_guide") == "$legacy_before" ]]' 'Legacy private cooldown is a read-only fallback'
benchmark_submission_succeeded benchmark 1.3.0
benchmark_guide_read
check '[[ -f $guide && $GUIDE_LAST_SUBMITTED_VERSION == 1.3.0 && $(< "$legacy_guide") == "$legacy_before" ]]' 'A normal successful upload saves local state without modifying legacy history'
reset_case legacy-receipt-directory
LEGACY_CONFIG_DIR=$ROOT/legacy
mkdir -p "$LEGACY_CONFIG_DIR/private/submission-receipts"
legacy_receipt=$LEGACY_CONFIG_DIR/private/submission-receipts/fixture.private-receipt.json
printf '{"api_version":1,"kind":"benchmark","receipt_id":"rc_fixture","report_id":"fixture","file_sha256":"%s","payload_sha256":"%s","status":"forwarding","received_at":"2026-10-25T00:00:00Z","engine_version":"1.3.0"}\n' "$hash" "$hash" > "$legacy_receipt"
legacy_before=$(< "$legacy_receipt")
benchmark_startup_guide </dev/null > "$output"
benchmark_guide_read
check '[[ ! -s $output && $GUIDE_LAST_SUBMITTED_VERSION == 1.3.0 && $(< "$legacy_receipt") == "$legacy_before" ]]' 'Legacy receipts restore explicit current-version success without moving the receipt'

# Only the release-list fetch is mocked: selection, notes, choices and state are real.
fetch_url() {
    printf 'fetch\n' >> "$ROOT/network"
    [[ ${network_ok-true} == true ]] || return 1
    printf '[{"tag_name":"release-%s","draft":false,"prerelease":false,"body":"Short release notes\\nSecond line","assets":[{"name":"MoE4All-%s-v%s.tar.gz","browser_download_url":"https://example.invalid/package"},{"name":"MoE4All-%s-v%s.tar.gz.sha256","browser_download_url":"https://example.invalid/checksum"}]}]' \
        "$test_release" "$package_platform" "$test_release" "$package_platform" "$test_release" > "$2"
}
case $platform in linux) package_platform=Linux-x86_64 ;; macos) package_platform=macOS-arm64 ;; esac
download_update() { printf '%s\n' "${2-true}" >> "$ROOT/updates"; [[ ${updater_ok-true} == true ]]; }
open_submission_url() { printf '%s\n' "$1" >> "$ROOT/opened"; return 1; }
yes() { printf 'yes\n' >> "$ROOT/confirmed"; REPLY=true; }
reset_case update-skip
check_updates --automatic <<< skip > "$output"
check '[[ ! -e $update_state && ! -e $ROOT/updates ]]' 'Skip never stores a three-day snooze or downloads'
grep -Fq 'Short release notes' "$output"
grep -Fq 'Skip until next launch' "$output"
check_updates --automatic <<< 2 > "$output"
check 'grep -Fq "Skip until next launch" "$output"' 'Skip asks again on the very next launch'
check_updates --automatic <<< snooze > "$output"
update_reminder_read
check '[[ $UPDATE_REMINDER_TAG == release-1.4.0 && $UPDATE_REMINDER_NEXT == 1800259200 ]]' 'Snooze stores the same tag for exactly three days'
before=$(< "$update_state"); test_now=1800259199
check_updates --automatic </dev/null > "$output"
check '[[ ! -s $output && $(< "$update_state") == "$before" ]]' 'Same tag remains silent and unchanged before the boundary'
test_release=1.5.0
check_updates --automatic <<< skip > "$output"
check 'grep -Fq "v1.5.0" "$output"' 'A new tag bypasses the old tag snooze immediately'
check '[[ $(< "$update_state") == "$before" ]]' 'Skip does not add or replace a snooze'
test_release=1.4.0; test_now=1800259200
check_updates --automatic <<< 2 > "$output"
check 'grep -Fq "Skip until next launch" "$output"' 'Three-day boundary is inclusive'
check_updates --automatic <<< $'R\nskip\n' > "$output"
check '[[ $(< "$ROOT/opened") == https://github.com/Headmaster218/MoE4All/releases/tag/release-1.4.0 ]]' 'R opens full notes for the selected tag'
check 'grep -Fq "Open: https://github.com/Headmaster218/MoE4All/releases/tag/release-1.4.0" "$output"' 'No graphical browser prints the link and returns to choices'
if check_updates --automatic <<< 1 > "$output"; then exit 1; else update_result=$?; fi
check '[[ $update_result == 10 && $(< "$ROOT/updates") == false && ! -e $ROOT/confirmed ]]' 'Automatic Update hands off restart to the caller after applying the existing updater'

reset_case manual-update
check_updates --automatic <<< 3 > "$output"
check_updates --update </dev/null > "$output"
check '[[ -s $ROOT/updates && -s $ROOT/confirmed ]]' 'Explicit update bypasses active snooze and retains explicit confirmation'
reset_case check-only
check_updates --check-update </dev/null > "$output"
check '[[ ! -e $CONFIG_DIR && ! -e $ROOT/updates && ! -e $ROOT/confirmed ]]' 'Explicit check does not prompt, persist state or download'
check 'grep -Fq "Short release notes" "$output"' 'Check-only still displays release notes'
reset_case saved-json-update
S[json_output]=true
check_updates --automatic <<< skip > "$output"
check 'grep -Fq "Skip until next launch" "$output"' 'Saved JSON export must not suppress the interactive update choices'
reset_case offline
network_ok=false
check_updates --automatic </dev/null > "$output" 2>&1
check '[[ ! -e $CONFIG_DIR && ! -e $ROOT/updates ]]' 'Network failure is nonfatal and leaves reminders unchanged'
network_ok=true; updater_ok=false
check_updates --automatic <<< update > "$output" 2>&1
check 'grep -Fq "continuing startup" "$output"' 'Failed automatic update is nonfatal'
updater_ok=true

reset_case legacy-update-directory
LEGACY_CONFIG_DIR=$ROOT/legacy
mkdir -p "$LEGACY_CONFIG_DIR/private"
legacy_update=$LEGACY_CONFIG_DIR/private/update-reminder.json
printf '{"schema_version":1,"release_tag":"release-1.4.0","next_remind_at":1800259200}\n' > "$legacy_update"
legacy_before=$(< "$legacy_update")
check_updates --automatic </dev/null > "$output"
check '[[ ! -s $output && ! -e $update_state && $(< "$legacy_update") == "$legacy_before" ]]' 'Legacy update snooze remains read-only and effective for the same tag'
test_release=1.5.0
check_updates --automatic <<< snooze > "$output"
update_reminder_read
check '[[ -f $update_state && $UPDATE_REMINDER_TAG == release-1.5.0 && $(< "$legacy_update") == "$legacy_before" ]]' 'A new tag saves local snooze without overwriting the legacy reminder'

for mode in dry headless json ci noninteractive; do
    reset_case "guard-$mode"
    case $mode in dry) DRY_RUN=true ;; headless) HEADLESS=true ;; json) INTERACTIVE=false; S[json_output]=true ;; ci) CI=true ;; noninteractive) INTERACTIVE=false ;; esac
    benchmark_startup_guide </dev/null > "$output"
    benchmark_submission_succeeded benchmark
    benchmark_guide_write false 0
    update_reminder_snooze release-1.4.0
    check_updates --automatic </dev/null >> "$output"
    check '[[ ! -e $CONFIG_DIR && ! -e $ROOT/network && ! -e $ROOT/updates && ! -s $output ]]' "$mode must not prompt, fetch or persist reminder state"
    unset CI
done
printf 'PASS %s wizard reminders: %s checks (schema2 migration, version provenance, 48h/7d/3d clocks, choices, offline and automation guards)\n' "$platform" "$checks"
