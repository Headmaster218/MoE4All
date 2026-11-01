#!/usr/bin/env bash
# Run with bash for Linux, or zsh for macOS. Only source library functions; never main.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=${1:?Pass a task directory below MOE4ALL_WORK_ROOT/tmp/moe4all}
platform=${2:-linux}
case $work in "${MOE4ALL_WORK_ROOT:?}/tmp/moe4all/"*) ;; *) printf 'Invalid test output path\n' >&2; exit 1 ;; esac
case /$work/ in */../*|*/./*) exit 1 ;; esac
MOE4ALL_WIZARD_LIBRARY=1
case $platform in
    linux) source "$repo/Start-INFR-Wizard-Linux.sh" ;;
    macos) source "$repo/Start-INFR-Wizard-macOS.sh" ;;
    *) exit 1 ;;
esac
mkdir -p -- "$work"; work=$(mktemp -d "$work/run.XXXXXX")
mkdir -p -- "$work/package" "$work/state with spaces"
ROOT="$work/package"; CONFIG_DIR="$work/state with spaces"; STATE="$CONFIG_DIR/wizard-state.json"
LEGACY_STATE="$work/missing.json"; LEGACY_CONFIG_DIR="$work/legacy"; BINARY="$work/never-launch-engine"
printf '#!/bin/sh\nexit 99\n' > "$BINARY"
reset_audio() { defaults; S[model]='llm.gguf'; S[launch_mode]=serve; }
checks=0
has_arg() { local arg; for arg in "${COMMAND[@]}"; do [[ $arg != "$1" ]] || return 0; done; return 1; }
reject() { if ( "$@" ) > "$work/rejected.log" 2>&1; then printf 'Expected rejection\n' >&2; exit 1; fi; ((checks+=1)); }
reset_audio
[[ ${S[server_transcription]} == false && ${S[transcription_threads]} == 4 && ${S[transcription_idle_timeout]} == 300 ]]; ((checks+=1))
build_command; baseline=$(printf '<%s>\n' "${COMMAND[@]}")
S[transcription_model]='missing path'; S[transcription_threads]=bad; build_command
[[ $(printf '<%s>\n' "${COMMAND[@]}") == "$baseline" ]]; ((checks+=1))
configure_transcription <<< '' > "$work/disabled.log"; [[ ${S[server_transcription]} == false ]]; ((checks+=1))
for mode in run bench; do
    S[launch_mode]=$mode; S[server_transcription]=true
    configure_transcription < /dev/null; build_command
    ! has_arg --transcription-model; [[ -z $(show_transcription_api) ]]; ((checks+=1))
done
reset_audio
catalog_file sensevoice-small-q8 1
[[ $CATALOG_KIND == audio && $CATALOG_COUNT == 1 && $DOWNLOAD_RELATIVE == SenseVoice-Small-Q8_0/sense-voice-small-q8_0.gguf ]]; ((checks+=1))
[[ $DOWNLOAD_URL == 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true' ]]; ((checks+=1))
recommendations llm > "$work/llm.log"; ! grep -q 'SenseVoice' "$work/llm.log"; ((checks+=1))
mkdir -p -- "$ROOT/audio space's"
model_file="$ROOT/audio space's/voice.gguf"; printf fixture > "$model_file"
downloads=0
download_bundle() {
    [[ $1 == sensevoice-small-q8 ]]; ((downloads+=1))
    mkdir -p -- "$ROOT/models/SenseVoice-Small-Q8_0"
    printf fixture > "$ROOT/models/SenseVoice-Small-Q8_0/sense-voice-small-q8_0.gguf"
}
# Blank/R must never download; explicit local paths and invalid integers are simulated.
printf 'y\n\nR\n"%s"\n0\n4\n-1\n0\n%s\n' "$ROOT/audio space's" "$work/upload space's" > "$work/answers"
configure_transcription < "$work/answers" > "$work/enabled.log"
[[ ${S[server_transcription]} == true && ${S[transcription_model]} == "$model_file" && ${S[transcription_threads]} == 4 && ${S[transcription_idle_timeout]} == 0 && $downloads == 0 ]]; ((checks+=1))
S[transcription_upload_dir]="\"$work/upload space's\""; build_command
expected=(--transcription-model "$model_file" --transcription-dev cpu --transcription-threads 4 --transcription-idle-timeout 0 --transcription-upload-dir "$work/upload space's")
COMMAND=(); add_transcription_arguments
[[ $(printf '<%s>\n' "${COMMAND[@]}") == "$(printf '<%s>\n' "${expected[@]}")" ]]; ((checks+=1))
# %q output must also round-trip through the platform shell with spaces/apostrophes.
printf 'capture() { printf "<%%s>\\n" "$@"; }; capture ' > "$work/quoted.sh"
printf '%q ' "${COMMAND[@]}" >> "$work/quoted.sh"; printf '\n' >> "$work/quoted.sh"
if [[ $platform == macos ]]; then quoted=$(zsh "$work/quoted.sh"); else quoted=$(bash "$work/quoted.sh"); fi
[[ $quoted == "$(printf '<%s>\n' "${expected[@]}")" ]]; ((checks+=1))
show_transcription_api > "$work/api.log"; grep -q '/v1/audio/transcriptions (WAV/PCM, CPU)' "$work/api.log"; ((checks+=1))
save_state; defaults; load_state
[[ ${S[server_transcription]} == true && ${S[transcription_model]} == "$model_file" && ${S[transcription_threads]} == 4 && ${S[transcription_idle_timeout]} == 0 && ${S[transcription_upload_dir]} == "$work/upload space's" ]]; ((checks+=1))
S[server_transcription]=false; save_state; defaults; load_state
[[ ${S[server_transcription]} == false && ${S[transcription_model]} == "$model_file" ]]; ((checks+=1))
printf '{"server_transcription":"false"}\n' > "$STATE"; reject load_state
printf '{"server_transcription":null}\n' > "$STATE"; reject load_state
S[server_transcription]=true
for value in 0 -1 bad 257 1000000000; do S[transcription_threads]=$value; reject add_transcription_arguments; done
S[transcription_threads]=4; S[transcription_idle_timeout]=-1; reject add_transcription_arguments
S[transcription_idle_timeout]=300; S[transcription_model]='missing.gguf'; reject add_transcription_arguments
S[transcription_model]=$model_file; S[transcription_upload_dir]='""'; reject add_transcription_arguments
S[transcription_model]=''
printf 'D\n1\n' > "$work/download-answers"
model_path audio transcription_model < "$work/download-answers" > "$work/download.log"
[[ $downloads == 1 && ${S[transcription_model]} == "$ROOT/models/SenseVoice-Small-Q8_0/sense-voice-small-q8_0.gguf" ]]; ((checks+=1))
printf 'PASS: %d %s transcription checks (library/input simulation only)\n' "$checks" "$platform"
