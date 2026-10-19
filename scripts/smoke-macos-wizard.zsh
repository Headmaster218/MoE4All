#!/bin/zsh
emulate -LR zsh
setopt ERR_EXIT NO_UNSET PIPE_FAIL

repo=${0:A:h:h}
work=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/moe4all-macos-smoke.XXXXXX")
trap 'rm -rf -- "$work"' EXIT INT TERM

fail() { print -u2 -- "$*"; return 1; }
assert_contains() { [[ $1 == *$2* ]] || fail "missing '$2' in: $1"; }
make_engine() {
    local target=$1 version=$2
    mkdir -p -- "${target:h}"
    cat > "$target" <<EOF
#!/bin/zsh
case \${1-} in
  --version) print 'infr $version' ;;
  --help) print 'mock infr help' ;;
  *) print 'mock infr $version' ;;
esac
EOF
    chmod 755 "$target"
}

/bin/zsh -n "$repo/Start-INFR-Wizard-macOS.sh"
python3 -c 'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' "$repo/scripts/macos_release.py"
grep -Fq "trap 'launcher_exit \$?' EXIT" "$repo/Start-INFR-Wizard-macOS.sh"

root="$work/root"
home="$work/home"
mkdir -p -- "$root" "$home"
make_engine "$root/infr" 1.0.2
touch "$root/model.gguf" "$root/draft.gguf"

output=$(HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode run --model model.gguf --profile aggressive --dev metal --ram 48GiB)
assert_contains "$output" '--dev metal'
assert_contains "$output" 'device.auto_profile=aggressive'
assert_contains "$output" 'device.ram_budget=48GiB'
assert_contains "$output" "$root/model.gguf"

output=$(HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode serve --model model.gguf --profile conservative \
    --parallel 1 --addr 127.0.0.1:8080 --no-api-key)
assert_contains "$output" '--parallel 1'
[[ $output != *--mmproj* && $output != *--embedding-model* ]] || fail 'Metal serve exposed unsupported resources'

output=$(HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode bench --model model.gguf --profile aggressive \
    --bench-p 1024 --bench-n 32 --bench-d 4096)
assert_contains "$output" '--dev metal'
assert_contains "$output" '-p 1024'
assert_contains "$output" '-n 32'

output=$(HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode run --model model.gguf --profile aggressive \
    --mtp draft.gguf --mtp-k 4)
assert_contains "$output" 'spec.mtp=true'
assert_contains "$output" "spec.draft=$root/draft.gguf"

if HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode serve --model model.gguf --profile conservative --parallel 2 >/dev/null 2>&1; then
    fail 'Metal accepted unsupported multi-slot serving'
fi
if HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode serve --model model.gguf --profile conservative --mmproj image.gguf >/dev/null 2>&1; then
    fail 'Metal accepted unsupported vision projection'
fi
if HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode run --model model.gguf --profile aggressive --cpu-miss-cores 4 >/dev/null 2>&1; then
    fail 'Metal accepted Vulkan CPU-miss settings'
fi
if HOME="$home" /bin/zsh "$repo/Start-INFR-Wizard-macOS.sh" \
    --root "$root" --dry-run --mode run --model model.gguf --profile aggressive --vram 16GiB >/dev/null 2>&1; then
    fail 'Metal accepted a separate Vulkan VRAM budget'
fi

old_out="$work/old-out"
new_out="$work/new-out"
bad_out="$work/bad-out"
make_engine "$work/infr-old" 1.0.1
make_engine "$work/infr-new" 1.0.2
make_engine "$work/infr-bad" 9.9.9
python3 "$repo/scripts/macos_release.py" --root "$repo" --binary "$work/infr-old" --output "$old_out" --version 1.0.1 >/dev/null
python3 "$repo/scripts/macos_release.py" --root "$repo" --binary "$work/infr-new" --output "$new_out" --version 1.0.2 >/dev/null
python3 "$repo/scripts/macos_release.py" --root "$repo" --binary "$work/infr-bad" --output "$bad_out" --version 1.0.2 >/dev/null

MOE4ALL_WIZARD_LIBRARY=1 source "$repo/Start-INFR-Wizard-macOS.sh"

# The submission client uses only tools included with macOS. Exercise the
# browser-confirmed relay with local fixtures and verify that the private token
# is kept outside the public report tree.
submission_root="$work/submission-root"
submission_home="$work/submission-home"
submission_bin="$work/submission-bin"
mkdir -p -- "$submission_root/diagnostics/reports/v1/2026-10" "$submission_home" "$submission_bin"
report_id=20261019T010203.004Z-00112233445566778899aabbccddeeff
report_path="$submission_root/diagnostics/reports/v1/2026-10/$report_id.m4e"
printf 'MOE4ALL-SUBMISSION/1\nkind: error\nid: %s\n' "$report_id" > "$report_path"
file_sha256 "$report_path"; fixture_hash=$REPLY
cat > "$submission_bin/infr" <<'ENGINE'
#!/bin/zsh
if [[ ${1-} == submission && ${2-} == verify ]]; then
    exit 0
fi
if [[ ${1-} == submission && ${2-} == pack-benchmark ]]; then
    shift 2
    result_dir=''
    while (($#)); do
        case $1 in
            --result-dir) result_dir=$2; shift 2 ;;
            *) shift ;;
        esac
    done
    path="$result_dir/v1/2026-10/manual-fixture.m4b"
    mkdir -p -- "${path:h}"
    printf 'MOE4ALL-SUBMISSION/1\nkind: benchmark\nid: manual-fixture\n' > "$path"
    printf 'MOE4ALL_SUBMISSION_PATH=%s\n' "$path"
    exit 0
fi
exit 1
ENGINE
cat > "$submission_bin/open" <<'OPEN'
#!/bin/zsh
exit 0
OPEN
cat > "$submission_bin/curl" <<'CURL'
#!/bin/zsh
emulate -LR zsh
setopt KSH_ARRAYS
output='' url='' method=GET
while (($#)); do
    case $1 in
        --output) output=$2; shift 2 ;;
        --request) method=$2; shift 2 ;;
        --write-out|--header|--data-binary|--connect-timeout|--max-time) shift 2 ;;
        --silent|--show-error) shift ;;
        *) url=$1; shift ;;
    esac
done
if [[ $url == */upload-sessions && $method == POST ]]; then
    printf '{"api_version":1,"state":"awaiting_verification","session_id":"us_fixture","session_token":"st_fixture","verification_url":"http://127.0.0.1:18765/community/upload.html#session=fixture","max_upload_bytes":131072}' > "$output"
    printf 201
elif [[ $url == */upload-sessions/us_fixture && $method == GET ]]; then
    printf '{"api_version":1,"state":"ready","kind":"error","report_id":"%s","file_sha256":"%s","size_bytes":%s}' "$FIXTURE_REPORT_ID" "$FIXTURE_HASH" "$FIXTURE_SIZE" > "$output"
    printf 200
else
    printf '{"api_version":1,"receipt_id":"rc_fixture","receipt_token":"rt_fixture","kind":"error","report_id":"%s","file_sha256":"%s","payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","status":"forwarding","duplicate":false,"received_at":"2026-10-19T01:02:03Z","public_url":null}' "$FIXTURE_REPORT_ID" "$FIXTURE_HASH" > "$output"
    printf 202
fi
CURL
chmod 755 "$submission_bin/infr" "$submission_bin/open" "$submission_bin/curl"
ROOT=$submission_root
CONFIG_DIR="$submission_home/Library/Application Support/MoE4All"
STATE="$CONFIG_DIR/wizard-state.json"
BINARY=$submission_bin/infr
INTERACTIVE=true
HEADLESS=false
defaults
unset MOE4ALL_SUBMISSION_API_BASE MOE4ALL_SUBMISSION_SITE_ORIGIN
submission_api_config
[[ $SUBMISSION_API_BASE == https://api.moe4all.com/v1 && $SUBMISSION_SITE_ORIGIN == https://moe4all.com ]] || fail 'Default submission origins are inconsistent'
export FIXTURE_REPORT_ID=$report_id FIXTURE_HASH=$fixture_hash FIXTURE_SIZE=$(wc -c < "$report_path")
export MOE4ALL_SUBMISSION_API_BASE=http://127.0.0.1:18765/v1
old_path=$PATH
PATH="$submission_bin:$PATH"
website_submission_upload "$report_path" error > "$work/submission-output"
PATH=$old_path
unset FIXTURE_REPORT_ID FIXTURE_HASH FIXTURE_SIZE MOE4ALL_SUBMISSION_API_BASE
[[ -f "$CONFIG_DIR/private/submission-receipts/rc_fixture.private-receipt.json" ]] || fail 'Submission receipt was not saved'
assert_contains "$(< "$work/submission-output")" 'status: forwarding'
printf '[{"avg_ts":42.5}]\n' > "$work/manual-benchmark.json"
touch "$submission_root/model.gguf"
S[model]="$submission_root/model.gguf"
S[manual_bench_kind]=decode
S[prompt_tokens]=0
S[gen_tokens]=512
S[depth_mode]=real
S[depth_tokens]=30000
S[reps]=1
S[device]=metal
S[setup_mode]=aggressive
pack_manual_benchmark "$work/manual-benchmark.json" > "$work/manual-pack-output"
[[ -f $REPLY ]] || fail 'Manual benchmark submission was not packed'
assert_contains "$(< "$work/manual-pack-output")" 'Results / 结果'

install="$work/install"
extract_package "$old_out/MoE4All-macOS-arm64-v1.0.1.tar.gz" "$install" 1.0.1
stage="$work/stage"
extract_package "$new_out/MoE4All-macOS-arm64-v1.0.2.tar.gz" "$stage" 1.0.2
ROOT=$install
CONFIG_DIR="$home/Library/Application Support/MoE4All"
STATE="$CONFIG_DIR/wizard-state.json"
apply_update "$install" "$stage" 1.0.2
[[ $("$install/infr" --version) == 'infr 1.0.2' ]] || fail 'Updater did not install the new engine'

rollback="$work/rollback"
extract_package "$old_out/MoE4All-macOS-arm64-v1.0.1.tar.gz" "$rollback" 1.0.1
bad_stage="$work/bad-stage"
extract_package "$bad_out/MoE4All-macOS-arm64-v1.0.2.tar.gz" "$bad_stage" 1.0.2
if apply_update "$rollback" "$bad_stage" 1.0.2; then
    fail 'Updater accepted an executable with the wrong version'
fi
[[ $("$rollback/infr" --version) == 'infr 1.0.1' ]] || fail 'Failed update did not restore the old engine'

print 'macOS launcher/package smoke tests passed'
