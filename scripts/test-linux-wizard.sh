#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/Start-INFR-Wizard-Linux.sh"
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
ROOT=$tmp/package
CONFIG_DIR=$tmp/config/infr
STATE=$CONFIG_DIR/wizard-state.json
BINARY=$ROOT/infr
mkdir -p "$ROOT" "$CONFIG_DIR"
cp "$repo/Start-INFR-Wizard-Linux.sh" "$ROOT/"
cat > "$BINARY" <<'ENGINE'
#!/usr/bin/env bash
case ${1-} in
    --version) printf 'infr 0.10.0\n' ;;
    devices) printf '%s\n' '  Vulkan0: Software [cpu, 0 MiB]' '  Vulkan1: Test GPU [discrete, 24 GiB] <- default' '  Vulkan2: Other GPU [integrated, 4 GiB]' ;;
    *) printf 'STUB_LAUNCHED\n'; printf '<%s>\n' "$@"; printf 'KEY=%s\n' "${INFR_API_KEY-}" ;;
esac
ENGINE
chmod +x "$BINARY"
export XDG_CONFIG_HOME=$tmp/config
unset INFR_API_KEY
checks=0
pass() { ((checks+=1)); }
has_arg() { local arg; for arg in "${COMMAND[@]}"; do [[ $arg != "$1" ]] || return 0; done; return 1; }
reject() { if "$@" > "$tmp/reject.log" 2>&1; then printf 'Unexpected success: %s\n' "$*" >&2; exit 1; fi; pass; }
reset() { defaults; S[model]="$ROOT/model with spaces.gguf"; }

bash -n "$repo/Start-INFR-Wizard-Linux.sh"; pass
for file in image-codecs/libheif.so.1 image-codecs/libde265.so.0 image-codecs/libdav1d.so.7 image-codecs/SOURCES.txt; do
    managed "$file"; pass
done
reject managed image-codecs/evil.so
for mode in run serve bench; do
    output=$(bash "$ROOT/Start-INFR-Wizard-Linux.sh" --dry-run --mode "$mode" --model 'model with spaces.gguf' --profile aggressive </dev/null)
    [[ $output == *device.auto_profile=aggressive* && $output != *STUB_LAUNCHED* ]]; pass
done
[[ ! -f $STATE ]]; pass
reset; S[launch_mode]=serve; S[server_parallel]=2; S[mtp_enabled]=true; S[mtp_model]=head.gguf; S[mtp_verify_tokens]=2
choice mtp_verify_tokens 'MTP width' '4|4' '3|3' '2|2' <<< '' > "$tmp/numeric-choice"
[[ ${S[mtp_verify_tokens]} == 2 ]]; pass
choice mtp_verify_tokens 'MTP width' '4|4' '3|3' '2|2' <<< '2' > "$tmp/numeric-choice"
[[ ${S[mtp_verify_tokens]} == 3 ]]; pass
build_command; has_arg spec.k=4; has_arg --temp; pass
S[server_parallel]=3; reject build_command
reset; S[setup_mode]=manual; S[configure_memory]=true; S[vram_budget]=24g; S[ram_budget]=48g
S[submit_mode]=fixed; S[submit_cap]=256; S[ubatch]=3072; S[kv_overflow]=true; S[kv_overflow_vram_mb]=100
S[custom_sets]='paging.expert_prefetch=false; kernels.vulkan.cpu_miss_threads=99'
build_command; has_arg device.vram_budget=24g; has_arg device.ram_budget=48g; has_arg device.submit_dispatches=256; has_arg kv.overflow_vram_mb=100
has_arg kernels.vulkan.cpu_miss_threads=0; ! has_arg kernels.vulkan.cpu_miss_threads=99; pass
S[custom_sets]='serve.api_key=secret'; reject build_command
S[custom_sets]='malformed'; reject build_command
reset; S[cpu_miss_enabled]=true; S[cpu_miss_cores]=4; S[cpu_miss_max]=3
build_command; has_arg kernels.vulkan.cpu_miss_threads=4; has_arg kernels.vulkan.cpu_miss_max=3; pass
S[cpu_miss_max]=4; reject build_command
reset; S[launch_mode]=bench; S[bench_kind]=mixed; S[depth_mode]=synthetic; S[depth_tokens]=30000; S[json_output]=true
build_command; has_arg --pg; has_arg --synthetic-depth; has_arg --json; ! has_arg spec.mtp=false; pass

# Round-trip adversarial values as data, never as shell commands.
reset; S[model]=$'model "quote" \\ space\nline\t.g g u f'; S[mtp_model]='$(touch SENTINEL)'; S[think_mode]=think; S[reasoning_effort]=medium
save_state; original=${S[model]}; defaults; load_state
[[ ${S[model]} == "$original" && ${S[mtp_model]} == '$(touch SENTINEL)' && ! -e $ROOT/SENTINEL ]]; pass
printf '{"model":"\u4e2d\u6587 \ud83d\ude00.gguf", "unknown": [1,2]}' > "$STATE"
defaults; load_state; [[ ${S[model]} != *'\u'* && ${S[model]} == *.gguf ]]; pass
printf '{"model":"a", "model":"b"}' > "$STATE"; reject load_state
printf '{}' > "$STATE"; defaults; load_state; [[ ${S[setup_mode]} == conservative ]]; pass
printf '[]' > "$STATE"; reject load_state
printf '{"model":"\u0000"}' > "$STATE"; reject load_state
printf '{"model":123}' > "$STATE"; reject load_state
printf '{"server_auth":"false"}' > "$STATE"; reject load_state
printf '{"model":"broken"' > "$STATE"; reject load_state
printf '{"think_mode":"think", "reasoning_effort":"medium"}' > "$STATE"
defaults; load_state; [[ ${S[configure_thinking]} == true ]]; pass
rm "$STATE"
printf 'MODE=server\nMODEL=model.gguf\nPROFILE=aggressive\nUBATCH=3072\nMMPROJ=image.gguf\nEMBEDDING=emb.gguf\n' > "$CONFIG_DIR/wizard.conf"
defaults; load_state; [[ ${S[launch_mode]} == serve && ${S[server_vision]} == true && ${S[auto_overrides]} == *ubatch* ]]; pass
rm "$CONFIG_DIR/wizard.conf"

mkdir "$ROOT/models"
for name in flash-00001-of-00002.gguf flash-00002-of-00002.gguf mmproj-flash.gguf mtp-flash.gguf; do touch "$ROOT/models/$name"; done
model_candidates "$ROOT/models" llm; [[ ${#CANDIDATES[@]} == 1 && ${CANDIDATES[0]} == *-00001-of-00002.gguf ]]; pass
model_candidates "$ROOT/models" vision; [[ ${#CANDIDATES[@]} == 1 && ${CANDIDATES[0]} == *mmproj* ]]; pass
reset; model_path llm model <<< 'models' > "$tmp/model-choice"; [[ ${S[model]} == *-00001-of-00002.gguf ]]; pass
reset; choose_device <<< '' > "$tmp/device-choice"; [[ ${S[device]} == Vulkan1 ]]; ! grep -q Software "$tmp/device-choice"; pass
for address in 127.0.0.1:8080 localhost:8080 '[::1]:8080'; do valid_address "$address"; loopback "$address"; done; pass
reject valid_address localhost.evil:8080
reject valid_address 127.0.0.1:65536
reject valid_address 300.1.1.1:8080
reject valid_address '[::::]:8080'
reject valid_address '[1:2:3:4:5:6:7]:8080'
reject valid_address '[1:2:3:4:5:6:7:8::]:8080'
for address in '[::]:8080' '[2001:db8::1]:8080' '[1:2:3:4:5:6:7:8]:8080'; do valid_address "$address"; done; pass
! loopback localhost.evil:8080; pass

# Saved settings and explicit CLI disable switches have identical precedence to Windows.
reset; S[launch_mode]=serve; S[mtp_enabled]=true; S[mtp_model]=head.gguf; S[server_vision]=true; S[vision_projector]=image.gguf
S[server_embedding]=true; S[embedding_model]=emb.gguf; S[ubatch]=2048; save_state
before=$(sha256sum "$STATE")
output=$(INFR_API_KEY=fixture-secret bash "$ROOT/Start-INFR-Wizard-Linux.sh" --dry-run --model relative.gguf --ubatch 3072 --kv-k q8_0 --no-mtp --no-mmproj --no-embedding --no-api-key </dev/null)
[[ $output == *3072* && $output == *kv.type_k=q8_0* && $output != *fixture-secret* && $output != *spec.draft* && $output != *--mmproj* && $(sha256sum "$STATE") == "$before" ]]; pass
reject bash "$ROOT/Start-INFR-Wizard-Linux.sh" --mode serve --model model.gguf --addr 0.0.0.0:8080 --no-api-key
reject bash "$ROOT/Start-INFR-Wizard-Linux.sh" --mode serve --model model.gguf
output=$(INFR_API_KEY=fixture-secret bash "$ROOT/Start-INFR-Wizard-Linux.sh" --mode serve --model model.gguf --yes --no-mtp --no-mmproj --no-embedding </dev/null)
[[ $output == *KEY=fixture-secret* ]] && ! grep -q fixture-secret "$STATE"; pass
output=$(INFR_API_KEY=fixture-secret bash "$ROOT/Start-INFR-Wizard-Linux.sh" --mode serve --model model.gguf --yes --no-api-key </dev/null)
[[ $output == *KEY=* && $output != *fixture-secret* ]]; pass

# Tiny real GGUF metadata, including a skipped array before the template.
gguf_string() { local n=${#1}; printf '\\%03o' "$n" > /dev/null; printf '%b' "\\$(printf '%03o' "$n")\0\0\0\0\0\0\0"; printf '%s' "$1"; }
{
    printf 'GGUF\3\0\0\0\0\0\0\0\0\0\0\0\3\0\0\0\0\0\0\0'
    gguf_string general.architecture; printf '\10\0\0\0'; gguf_string qwen4exp
    gguf_string ignored.array; printf '\11\0\0\0\4\0\0\0\2\0\0\0\0\0\0\0\1\0\0\0\2\0\0\0'
    gguf_string tokenizer.chat_template; printf '\10\0\0\0'; gguf_string reasoning_effort
} > "$ROOT/tiny.gguf"
[[ $(reasoning_efforts "$ROOT/tiny.gguf") == 'low medium xhigh' ]]; pass
printf GGUF > "$ROOT/bad.gguf"; reject reasoning_efforts "$ROOT/bad.gguf"

# The release fixture contains only runtime products, not developer Python files.
make_package() {
    local base=$1 version=$2 name hash size sep=''
    mkdir -p "$base/documentation"
    printf '#!/usr/bin/env bash\nprintf "infr %s\\n"\n' "$version" > "$base/infr"
    cp "$repo/Start-INFR-Wizard-Linux.sh" "$base/"
    printf 'test guide\n' > "$base/documentation/README.md"
    chmod +x "$base/infr" "$base/Start-INFR-Wizard-Linux.sh"
    {
        printf '{"schema_version":1,"updater_protocol":1,"product":"moe4all-engine","platform":"Linux-x86_64","version":"%s","files":[' "$version"
        for name in infr Start-INFR-Wizard-Linux.sh documentation/README.md; do
            hash=$(sha256sum "$base/$name"); hash=${hash%% *}; size=$(wc -c < "$base/$name")
            printf '%s{"path":"%s","size":%s,"sha256":"%s"}' "$sep" "$name" "$size" "$hash"; sep=,
        done
        printf ']}\n'
    } > "$base/install-manifest.json"
}
make_package "$tmp/installed" 0.10.0
make_package "$tmp/stage" 0.11.0
validate_package "$tmp/stage" 0.11.0; pass
reject validate_package "$tmp/stage" 0.12.0
touch "$tmp/stage/unlisted"; reject validate_package "$tmp/stage" 0.11.0; rm "$tmp/stage/unlisted"
cp "$tmp/stage/infr" "$tmp/original-infr"; printf corrupt > "$tmp/stage/infr"; reject validate_package "$tmp/stage" 0.11.0; cp "$tmp/original-infr" "$tmp/stage/infr"
prefix=MoE4All-Linux-x86_64-v0.11.0
cp -r "$tmp/stage" "$tmp/$prefix"; tar -czf "$tmp/valid.tar.gz" -C "$tmp" "$prefix"
extract_package "$tmp/valid.tar.gz" "$tmp/extracted" 0.11.0; pass
printf private > "$tmp/installed/infr.toml"; mkdir "$tmp/installed/kv-sessions"; printf cache > "$tmp/installed/kv-sessions/private.infrkv"
apply_update "$tmp/installed" "$tmp/extracted" 0.11.0
[[ $("$tmp/installed/infr" --version) == 'infr 0.11.0' && $(< "$tmp/installed/infr.toml") == private && $(< "$tmp/installed/kv-sessions/private.infrkv") == cache ]]; pass
reject apply_update "$tmp/installed" "$tmp/stage" 0.11.0
make_package "$tmp/rollback-old" 0.10.0; make_package "$tmp/rollback-new" 0.11.0
# Version mismatch is internally consistent with the manifest, but fails the final executable probe.
printf '#!/usr/bin/env bash\nprintf "infr 9.9.9\\n"\n' > "$tmp/rollback-new/infr"
hash=$(sha256sum "$tmp/rollback-new/infr"); hash=${hash%% *}; size=$(wc -c < "$tmp/rollback-new/infr")
json_load "$tmp/rollback-new/install-manifest.json"; original_hash=${J['["files",0,"sha256"]']}; original_size=${J['["files",0,"size"]']}
sed -i "s/${original_hash:1:64}/$hash/;s/\"size\":$original_size,/\"size\":$size,/" "$tmp/rollback-new/install-manifest.json"
before=$(sha256sum "$tmp/rollback-old/infr" "$tmp/rollback-old/install-manifest.json")
reject apply_update "$tmp/rollback-old" "$tmp/rollback-new" 0.11.0
[[ $(sha256sum "$tmp/rollback-old/infr" "$tmp/rollback-old/install-manifest.json") == "$before" ]]; pass
reject fetch_url https://example.invalid/asset "$tmp/asset" 100

# Shell updater guards (not just the separate developer Python fixtures).
for name in 'documentation/../escape.md' 'documentation//escape.md' 'kv-sessions/private.infrkv' '/absolute' 'scripts/linux_wizard.py'; do reject managed "$name"; done
make_package "$tmp/link-package" 0.11.0
rm "$tmp/link-package/documentation/README.md"; ln -s /outside "$tmp/link-package/documentation/README.md"
reject validate_package "$tmp/link-package" 0.11.0
mkdir "$tmp/link-installed"; cp -r "$tmp/rollback-old/." "$tmp/link-installed/"
mv "$tmp/link-installed/documentation" "$tmp/outside-docs"; ln -s "$tmp/outside-docs" "$tmp/link-installed/documentation"
make_package "$tmp/link-stage" 0.11.0; reject apply_update "$tmp/link-installed" "$tmp/link-stage" 0.11.0
make_package "$tmp/running-old" 0.10.0; make_package "$tmp/running-new" 0.11.0
( engine_running() { return 0; }; reject apply_update "$tmp/running-old" "$tmp/running-new" 0.11.0 ); pass
mkdir "$tmp/running-old/.git"; reject apply_update "$tmp/running-old" "$tmp/running-new" 0.11.0
make_package "$tmp/mid-old" 0.10.0; make_package "$tmp/mid-new" 0.11.0
before=$(sha256sum "$tmp/mid-old/infr" "$tmp/mid-old/install-manifest.json")
(
    mv() { local arg; for arg in "$@"; do [[ $arg != *mid-new/Start-INFR-Wizard-Linux.sh ]] || return 1; done; command mv "$@"; }
    reject apply_update "$tmp/mid-old" "$tmp/mid-new" 0.11.0
)
[[ $(sha256sum "$tmp/mid-old/infr" "$tmp/mid-old/install-manifest.json") == "$before" ]]; pass

if [[ $(uname -s) == Linux ]]; then
    make_package "$tmp/fifo" 0.11.0; mkfifo "$tmp/fifo/unlisted-fifo"; reject validate_package "$tmp/fifo" 0.11.0
    mkdir -p "$tmp/archive-bad/$prefix"
    ln -s /outside "$tmp/archive-bad/$prefix/infr"
    tar -czf "$tmp/bad-link.tar.gz" -C "$tmp/archive-bad" "$prefix"
    reject extract_package "$tmp/bad-link.tar.gz" "$tmp/link-extract" 0.11.0
    rm "$tmp/archive-bad/$prefix/infr"; printf data > "$tmp/archive-bad/$prefix/infr"
    tar -czf "$tmp/bad-duplicate.tar.gz" -C "$tmp/archive-bad" "$prefix/infr" "$prefix/infr"
    reject extract_package "$tmp/bad-duplicate.tar.gz" "$tmp/duplicate-extract" 0.11.0
    tar -czf "$tmp/bad-traversal.tar.gz" --transform='s@infr$@../escape@' -C "$tmp/archive-bad" "$prefix/infr"
    reject extract_package "$tmp/bad-traversal.tar.gz" "$tmp/traversal-extract" 0.11.0

    # Full Shell download/checksum/extract/replace chain, all network replaced by local fixtures.
    make_package "$tmp/download-old" 0.10.0
    hash=$(sha256sum "$tmp/valid.tar.gz"); hash=${hash%% *}
    printf '%s  %s.tar.gz\n' "$hash" "$prefix" > "$tmp/download.sha256"
    (
        RELEASE_VERSION=0.11.0
        RELEASE_ARCHIVE="https://github.com/Headmaster218/MoE4All/releases/download/release-0.11.0/$prefix.tar.gz"
        RELEASE_CHECKSUM="$RELEASE_ARCHIVE.sha256"
        fetch_url() { if [[ $1 == *.sha256 ]]; then cp "$tmp/download.sha256" "$2"; else cp "$tmp/valid.tar.gz" "$2"; fi; }
        download_update "$tmp/download-old"
    ) > "$tmp/download-output"
    [[ $("$tmp/download-old/infr" --version) == 'infr 0.11.0' ]]; pass
    make_package "$tmp/bad-download-old" 0.10.0
    before=$(sha256sum "$tmp/bad-download-old/infr")
    (
        RELEASE_VERSION=0.11.0; RELEASE_ARCHIVE=archive; RELEASE_CHECKSUM=checksum
        fetch_url() { if [[ $1 == checksum ]]; then printf '%064d  %s.tar.gz\n' 0 "$prefix" > "$2"; else cp "$tmp/valid.tar.gz" "$2"; fi; }
        reject download_update "$tmp/bad-download-old"
    )
    [[ $(sha256sum "$tmp/bad-download-old/infr") == "$before" ]]; pass
fi

# Exercise full interactive configuration with stub devices, no TTY or GPU needed.
reset; S[model]="$ROOT/models/flash-00001-of-00002.gguf"
configure > "$tmp/interactive-default" <<'INPUT'











INPUT
[[ ${S[launch_mode]} == run && ${S[setup_mode]} == conservative && ${S[device]} == Vulkan1 && ${S[max_new]} == 65536 ]]; pass
reset; S[launch_mode]=serve; S[model]="$ROOT/models/flash-00001-of-00002.gguf"
S[setup_mode]=aggressive; S[mtp_model]="$ROOT/models/mtp-flash.gguf"; S[vision_projector]="$ROOT/models/mmproj-flash.gguf"
S[embedding_model]="$ROOT/models/mtp-flash.gguf"; S[server_parallel]=2
configure > "$tmp/interactive-combined" <<'INPUT'


y


y

y




n

n
n

150k
y
n

n
INPUT
build_command; has_arg --mmproj; has_arg --embedding-model; has_arg spec.k=4; has_arg kv.session_idle_secs=120; has_arg 150k; pass

version_newer 0.10.0 0.9.9; ! version_newer 0.10.0 0.10.0; ! version_newer 0.9.9 0.10.0; pass

# A runtime package works with Python and jq deliberately made unusable.
mkdir "$tmp/no-language-runtime"
for name in python python3 jq; do printf '#!/bin/sh\nexit 91\n' > "$tmp/no-language-runtime/$name"; chmod +x "$tmp/no-language-runtime/$name"; done
output=$(PATH="$tmp/no-language-runtime:$PATH" bash "$ROOT/Start-INFR-Wizard-Linux.sh" --dry-run --model model.gguf --no-mtp --no-mmproj --no-embedding --no-api-key </dev/null)
[[ $output != *STUB_LAUNCHED* && $output == *spec.mtp=false* ]]; pass

printf '[{"tag_name":"release-0.9.0","draft":false,"prerelease":false,"assets":[]},{"tag_name":"release-0.10.0","draft":false,"prerelease":false,"assets":[]},{"tag_name":"release-1.0.0","draft":false,"prerelease":true,"assets":[]},{"tag_name":"agent-9.0.0","draft":false,"prerelease":false,"assets":[]}]' > "$tmp/releases.json"
json_load "$tmp/releases.json"; select_release; [[ $RELEASE_VERSION == 0.10.0 && -z $RELEASE_ARCHIVE ]]; pass

printf '[{"tag_name":"release-0.11.0","draft":false,"prerelease":false,"assets":[{"name":"MoE4All-Linux-x86_64-v0.11.0.tar.gz","browser_download_url":"https://github.com/Headmaster218/MoE4All/releases/download/release-0.11.0/MoE4All-Linux-x86_64-v0.11.0.tar.gz"},{"name":"MoE4All-Linux-x86_64-v0.11.0.tar.gz.sha256","browser_download_url":"https://github.com/Headmaster218/MoE4All/releases/download/release-0.11.0/MoE4All-Linux-x86_64-v0.11.0.tar.gz.sha256"}]}]' > "$tmp/update-check.json"
(
    touch "$ROOT/install-manifest.json"; INTERACTIVE=true; HEADLESS=false
    fetch_url() { cp "$tmp/update-check.json" "$2"; }
    download_update() { touch "$tmp/UNEXPECTED_UPDATE"; return 1; }
    yes() { touch "$tmp/UNEXPECTED_PROMPT"; return 1; }
    check_updates --check-update
    [[ ! -f $tmp/UNEXPECTED_UPDATE && ! -f $tmp/UNEXPECTED_PROMPT ]]
    HEADLESS=true; check_updates --update
    [[ ! -f $tmp/UNEXPECTED_UPDATE && ! -f $tmp/UNEXPECTED_PROMPT ]]
    HEADLESS=false; yes() { REPLY=false; }; check_updates --update
    [[ ! -f $tmp/UNEXPECTED_UPDATE ]]
) > "$tmp/check-only-output"
pass
printf 'test-linux-wizard: %s checks passed\n' "$checks"
