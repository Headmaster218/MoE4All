#!/usr/bin/env bash
# Offline download configuration/catalog regression. Requires an explicit fresh output directory.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
work=${1:?Pass a fresh absolute directory under local-work/tmp/moe4all/<task>.}
mode=${2-full}
case $mode in full|--catalog-only) ;; *) printf 'Unknown test mode.\n' >&2; exit 2 ;; esac
case $work in /*) ;; *) printf 'Use an absolute fixture directory.\n' >&2; exit 1;; esac
[[ ! -e $work ]] || { printf 'Fixture directory must be new.\n' >&2; exit 1; }
source "$repo/Start-INFR-Wizard-Linux.sh"
mkdir -p "$work/install" "$work/config"
ROOT="$work/install"; CONFIG_DIR="$work/config"; STATE="$CONFIG_DIR/wizard-state.json"
LEGACY_STATE=''; LEGACY_CONFIG_DIR=''; BENCHMARK_GUIDE_ACTIVE=false
checks=0
pass() { ((checks+=1)); }
defaults
[[ ${S[download_dir]} == "$ROOT/models" ]]; pass
[[ $((44866868384+11258338528+38400184512)) == 94525391424 ]]; pass
for pair in qwen35-balanced:25624957632 flashnext-427:94525391424 flashnext-vision:904003840 qwen3-embedding-06b-q8:639150592 sensevoice-small-q8:291985952 qwen35-vision:902822624; do
    catalog_load "${pair%:*}"; [[ $CATALOG_BYTES == "${pair#*:}" ]]; pass
    [[ $CATALOG_NAME != *'('* && $CATALOG_NAME != *'shards'* ]]; pass
    catalog_size; [[ $REPLY == *GiB || $REPLY == *MiB ]]; pass
done
revision=f5b0b790eebb89033c48091c26abf74c1b385940
catalog_load flashnext-427
[[ $CATALOG_COUNT == 3 && $CATALOG_PRIMARY == *00001-of-00003.gguf && $CATALOG_INFO == "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/$revision" ]]; pass
for shard in 1 2 3; do
    catalog_file flashnext-427 "$shard"
    [[ $DOWNLOAD_URL == "https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/$revision/${DOWNLOAD_RELATIVE##*/}?download=true" ]]; pass
    [[ $DOWNLOAD_RELATIVE == *of-00003.gguf && $DOWNLOAD_RELATIVE != *of-00033* ]]; pass
done
if catalog_file flashnext-427 4 > "$work/invalid-shard.log" 2>&1; then exit 1; fi; pass
catalog_file flashnext-vision 1; [[ $DOWNLOAD_URL == 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true' ]]; pass
catalog_file qwen35-balanced 1; [[ $DOWNLOAD_URL == 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true' ]]; pass
catalog_file qwen3-embedding-06b-q8 1; [[ $DOWNLOAD_URL == 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true' ]]; pass
catalog_file sensevoice-small-q8 1; [[ $DOWNLOAD_URL == 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true' ]]; pass
[[ "${CATALOG_IDS[*]}" == 'qwen35-balanced flashnext-427 flashnext-vision qwen3-embedding-06b-q8 sensevoice-small-q8 qwen35-vision melotts-chinese-native' ]]; pass
catalog_load qwen35-vision
[[ $CATALOG_KIND == vision && $CATALOG_COUNT == 1 && $CATALOG_ACTION == download ]]; pass
[[ $CATALOG_NAME == 'Qwen3.6-35B-A3B · mmproj.gguf' && $CATALOG_PRIMARY == 'Qwen3.6-35B-A3B-APEX-I-Balanced/mmproj.gguf' ]]; pass
[[ $CATALOG_INFO == 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/mmproj.gguf' ]]; pass
catalog_file qwen35-vision 1
[[ $DOWNLOAD_RELATIVE == 'Qwen3.6-35B-A3B-APEX-I-Balanced/mmproj.gguf' && $DOWNLOAD_URL == 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/mmproj.gguf?download=true' ]]; pass
recommendations vision > "$work/vision-catalog.log"
grep -Fq '902822624 bytes' "$work/vision-catalog.log"; pass
grep -Fq 'mmproj-Qwen3.8-Flash-Next-F16' "$work/vision-catalog.log"; pass
grep -Fq 'Qwen3.6-35B-A3B · mmproj.gguf' "$work/vision-catalog.log"; pass
if [[ $mode == --catalog-only ]]; then
    printf 'PASS: %d pure catalog checks (no download, engine, config save or network)\n' "$checks"
    exit 0
fi

printf '%s\n' '{"model_history":"old-model-00001-of-00033.gguf","server_parallel":"1","future":{"keep":true}}' > "$STATE"
load_state
custom="$work/模型 downloads"
S[server_parallel]=8 # Changing only the download folder must not save this unaccepted draft.
choose_download_directory <<< "$custom" > "$work/choose-folder.log"
[[ ${S[download_dir]} == "$custom" ]]; pass
defaults; load_state
[[ ${S[download_dir]} == "$custom" && ${S[model_history]} == old-model-00001-of-00033.gguf && ${S[server_parallel]} == 1 ]]; pass
json_load "$STATE"; json_get '["future","keep"]'; [[ $REPLY == true ]]; pass
BENCHMARK_GUIDE_ACTIVE=true
choose_download_directory <<< '' > "$work/reuse-folder.log"
defaults; load_state
[[ ${S[download_dir]} == "$custom" && ${S[model_history]} == old-model-00001-of-00033.gguf ]]; pass
BENCHMARK_GUIDE_ACTIVE=false
model_download_file() {
    printf '%s\t%s\n' "$1" "$2" >> "$work/downloads.tsv"
    mkdir -p "${2%/*}"; printf GGUF-fixture > "$2"
}
catalog_load flashnext-427
old="$custom/${CATALOG_PRIMARY%/*}/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
mkdir -p "${old%/*}"; printf OLD-33-DO-NOT-TOUCH > "$old"
download_bundle flashnext-427 true > "$work/flash-download.log"
[[ $(wc -l < "$work/downloads.tsv") == 3 && $(< "$old") == OLD-33-DO-NOT-TOUCH ]]; pass
[[ ! -d $ROOT/models ]]; pass
CANDIDATES=(); catalog_candidates llm
[[ ${#CANDIDATES[@]} == 1 && ${CANDIDATES[0]} == "$custom/"*of-00003.gguf ]]; pass

# Vision D asks which matching projector to download; other single-item D paths retain the folder selector.
for pair in embedding:embedding_model:qwen3-embedding-06b-q8 audio:transcription_model:sensevoice-small-q8 vision:vision_projector:flashnext-vision; do
    kind=${pair%%:*}; tail=${pair#*:}; key=${tail%%:*}; id=${tail#*:}
    input=$'D\n\n1'; [[ $kind != vision ]] || input=$'D\n1\ny\n1'
    S[$key]=''; model_path "$kind" "$key" <<< "$input" > "$work/single-$kind.log"
    catalog_load "$id"; [[ ${S[$key]} == "$custom/$CATALOG_PRIMARY" && -s ${S[$key]} ]]; pass
done
S[vision_projector]=''; model_path vision vision_projector <<< $'D\n2\ny\n2' > "$work/vision-qwen35.log"
catalog_load qwen35-vision; [[ ${S[vision_projector]} == "$custom/$CATALOG_PRIMARY" && -s ${S[vision_projector]} ]]; pass
[[ ${S[model_history]} == old-model-00001-of-00033.gguf && $(< "$old") == OLD-33-DO-NOT-TOUCH ]]; pass
other="$work/other directory"
download_menu <<< "$(printf 'P\n%s\n1,4\ny\n' "$other")" > "$work/menu.log"
[[ ${S[download_dir]} == "$other" ]]; pass
catalog_load qwen35-balanced; [[ -s "$other/$CATALOG_PRIMARY" ]]; pass
catalog_load qwen3-embedding-06b-q8; [[ -s "$other/$CATALOG_PRIMARY" ]]; pass
defaults; load_state; [[ ${S[download_dir]} == "$other" && ${S[model_history]} == old-model-00001-of-00033.gguf ]]; pass
json_load "$STATE"; json_get '["future","keep"]'; [[ $REPLY == true ]]; pass
grep -Fq '94525391424 bytes' "$work/menu.log"; pass
grep -Fq '25624957632 bytes' "$work/menu.log"; pass
grep -Fq '904003840 bytes' "$work/menu.log"; pass
grep -Fq '639150592 bytes' "$work/menu.log"; pass
grep -Fq '291985952 bytes' "$work/menu.log"; pass
grep -Fq '902822624 bytes' "$work/menu.log"; pass
printf 'PASS: %d download directory/catalog/3-shard checks (tiny fixtures; no network or engine)\n' "$checks"
