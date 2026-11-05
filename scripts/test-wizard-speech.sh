#!/usr/bin/env bash
# bash ... WORK linux, or zsh ... WORK macos. Mock backend only; no network/GPU.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=${1:?Pass a task directory below MOE4ALL_WORK_ROOT/tmp/moe4all}
platform=${2:-linux}
case $work in "${MOE4ALL_WORK_ROOT:?}/tmp/moe4all/"*) ;; *) exit 1;; esac
case /$work/ in */../*|*/./*) exit 1;; esac
MOE4ALL_WIZARD_LIBRARY=1
case $platform in
    linux) launcher="$repo/Start-INFR-Wizard-Linux.sh"; shell=bash; device=Vulkan1 ;;
    macos) launcher="$repo/Start-INFR-Wizard-macOS.sh"; shell=zsh; device=metal ;;
    *) exit 1;;
esac
source "$launcher"
mkdir -p -- "$work"; work=$(mktemp -d "$work/speech.XXXXXX")
export TMPDIR="$work" TMP="$work" TEMP="$work"
ROOT="$work/package space's"; CONFIG_DIR="$ROOT/gui-data"; STATE="$CONFIG_DIR/wizard-state.json"
LEGACY_STATE="$work/missing.json"; LEGACY_CONFIG_DIR="$work/legacy"; BINARY="$ROOT/infr"
mkdir -p -- "$ROOT" "$CONFIG_DIR" "$work/assets space's"
export SPEECH_CAPTURE="$work/argv" SPEECH_DEVICE_PROBES="$work/device-probes"
export MOCK_ENGINE_VERSION=1.3.0 MOCK_VERSION_EXIT=0 MOCK_DISABLE_TTS=0
unset INFR_API_KEY
cat > "$BINARY" <<'MOCK'
#!/bin/sh
case ${1-} in
    --version) printf 'infr %s\n' "$MOCK_ENGINE_VERSION"; exit "$MOCK_VERSION_EXIT" ;;
    devices) printf probe >> "$SPEECH_DEVICE_PROBES"; exit 91 ;;
esac
if [ "$MOCK_DISABLE_TTS" = 1 ]; then
    printf 'native speech is unavailable in this build; rebuild infr-cli with --features tts\n' >&2
    exit 86
fi
printf '<%s>\n' "$@" > "$SPEECH_CAPTURE"
MOCK
chmod +x "$BINARY"
model="$ROOT/speech space's.gguf"; llm="$ROOT/llm.gguf"; embedding="$ROOT/embed.gguf"; audio="$ROOT/audio.gguf"
assets="$work/assets space's"
touch "$model" "$llm" "$embedding" "$audio"
for file in lexicon.txt tokens.txt config.json; do printf fixture > "$assets/$file"; done
checks=0
pass() { ((checks+=1)); }
has_arg() { local arg; for arg in "${COMMAND[@]}"; do [[ $arg != "$1" ]] || return 0; done; return 1; }
arg_value() { local i; for ((i=0;i<${#COMMAND[@]}-1;i++)); do if [[ ${COMMAND[i]} == "$1" ]]; then REPLY=${COMMAND[i+1]}; return 0; fi; done; return 1; }
reject() { if ( "$@" ) > "$work/reject.log" 2>&1; then printf 'Unexpected success: %s\n' "$*" >&2; exit 1; fi; pass; }
reset_speech() {
    defaults; OVERRIDES=(); S[launch_mode]=serve; S[device]=$device
    S[model]=$llm; S[embedding_model]=$embedding; S[transcription_model]=$audio
    S[transcription_upload_dir]="$work/uploads"; S[speech_model]=$model; S[speech_lexicon]=$assets
}
reset_speech
[[ ${S[server_speech]} == false && ${S[speech_threads]} == 4 && ${S[speech_idle_timeout]} == 300 && ${S[speech_max_output_seconds]} == 30 ]]; pass
build_command; baseline=$(printf '<%s>\n' "${COMMAND[@]}")
S[speech_model]=missing; S[speech_threads]=bad; build_command
[[ $(printf '<%s>\n' "${COMMAND[@]}") == "$baseline" ]]; pass
configure_speech < /dev/null
for mode in run bench; do reset_speech; S[launch_mode]=$mode; S[server_speech]=true; build_command; ! has_arg --speech-model; [[ -z $(show_speech_api) ]]; pass; done
reset_speech; S[server_llm_enabled]=false; reject build_command

# All 15 combinations: CPU speech never owns a GPU selection or positional LLM.
for l in false true; do for e in false true; do for a in false true; do for s in false true; do
    [[ $l == true || $e == true || $a == true || $s == true ]] || continue
    reset_speech; S[server_llm_enabled]=$l; S[server_embedding]=$e; S[server_transcription]=$a; S[server_speech]=$s
    build_command; has_arg serve
    if [[ $s == true ]]; then
        for flag in --speech-model --speech-lexicon --speech-threads --speech-idle-timeout --speech-max-output-seconds; do has_arg "$flag"; pass; done
        arg_value --speech-model; [[ $REPLY == "$model" ]]; pass
        arg_value --speech-lexicon; [[ $REPLY == "$assets" ]]; pass
        ! has_arg --speech-dev; pass
    else ! has_arg --speech-model; pass; fi
    if [[ $l == true ]]; then has_arg "$llm"; else ! has_arg "$llm"; ! has_arg --parallel; fi; pass
    if [[ $e == true ]]; then has_arg --embedding-model; else ! has_arg --embedding-model; fi; pass
    if [[ $a == true ]]; then has_arg --transcription-model; else ! has_arg --transcription-model; fi; pass
    arg_value --dev
    if [[ $l == false && $e == false ]]; then [[ $REPLY == cpu ]]; else [[ $REPLY == "$device" ]]; fi; pass
    "${COMMAND[@]}"; [[ $(cat "$SPEECH_CAPTURE") == "$(printf '<%s>\n' "${COMMAND[@]:1}")" ]]; pass
done; done; done; done
[[ ! -e $SPEECH_DEVICE_PROBES ]]; pass

reset_speech; S[server_llm_enabled]=false; S[server_speech]=true
for value in 0 -1 257 bad; do S[speech_threads]=$value; reject add_speech_arguments; done
S[speech_threads]=4; S[speech_idle_timeout]=-1; reject add_speech_arguments
S[speech_idle_timeout]=0
for value in 0 0.9 120.1 NaN inf -2 '1;exit'; do S[speech_max_output_seconds]=$value; reject add_speech_arguments; done
for value in 1 1.5 30 120; do S[speech_max_output_seconds]=$value; COMMAND=(); add_speech_arguments; pass; done
S[speech_max_output_seconds]=30; S[speech_lexicon]=''; reject add_speech_arguments
S[speech_lexicon]="$work/missing-assets"; reject add_speech_arguments
S[speech_lexicon]=$assets; S[speech_model]="$work/missing.gguf"; reject add_speech_arguments

# Dynamic route is based on the actual selected binary, not source/workspace version.
for version in 1.3.0 1.4.2 1.4.0-rc.1+build.7; do
    MOCK_ENGINE_VERSION=$version; speech_release_url
    [[ $REPLY == "https://github.com/Headmaster218/MoE4All/releases/tag/release-$version" ]]; pass
done
for version in '' unknown '1.3' '1.3.0/../../evil' '1.3.0 junk'; do MOCK_ENGINE_VERSION=$version; speech_release_url; [[ $REPLY == */releases/latest ]]; pass; done
MOCK_ENGINE_VERSION=1.3.0; MOCK_VERSION_EXIT=1; speech_release_url; [[ $REPLY == */releases/latest ]]; pass
MOCK_VERSION_EXIT=0
catalog_load melotts-chinese-native
[[ $CATALOG_KIND == speech && $CATALOG_BYTES == 155780558 && $CATALOG_COUNT == 0 && $CATALOG_ACTION == release-note && -z $CATALOG_PRIMARY ]]; pass
[[ $CATALOG_INFO == */releases/tag/release-1.3.0 ]]; pass
catalog_size; [[ $REPLY == '148.56 MiB' ]]; pass
CANDIDATES=(); catalog_candidates speech; [[ ${#CANDIDATES[@]} == 0 ]]; pass

open_submission_url() { printf '%s\n' "$1" >> "$work/opened-urls"; }
unset DISPLAY WAYLAND_DISPLAY
SSH_CONNECTION=fixture; export SSH_CONNECTION
reset_speech; S[server_llm_enabled]=false; S[server_speech]=true; save_state
before_state=$(cat "$STATE")
(
    choose_download_directory() { printf UNEXPECTED_DIRECTORY >&2; return 90; }
    model_download_file() { printf UNEXPECTED_DOWNLOAD >&2; return 91; }
    download_bundle melotts-chinese-native < /dev/null
) > "$work/headless-guide.log"
grep -Fq 'https://github.com/Headmaster218/MoE4All/releases/tag/release-1.3.0' "$work/headless-guide.log"; pass
[[ ! -e $work/opened-urls && ! -d $ROOT/models && $(cat "$STATE") == "$before_state" ]]; pass
if [[ $platform == linux ]]; then DISPLAY=:fixture; export DISPLAY; else unset SSH_CONNECTION; fi
show_speech_release_notes > "$work/gui-guide.log"
grep -Fxq 'https://github.com/Headmaster218/MoE4All/releases/tag/release-1.3.0' "$work/opened-urls"; pass
unset DISPLAY WAYLAND_DISPLAY; SSH_CONNECTION=fixture; export SSH_CONNECTION
index=0; for id in "${CATALOG_IDS[@]}"; do ((index+=1)); [[ $id != melotts-chinese-native ]] || break; done
download_menu <<< "$(printf '%s\ny\n' "$index")" > "$work/menu-guide.log"
grep -Fq '155780558 bytes' "$work/menu-guide.log"; pass
! grep -Fq 'MeloTTS-Chinese [已安装 / installed]' "$work/menu-guide.log"; pass
[[ $(cat "$STATE") == "$before_state" ]]; pass

# Explicit D is available for both model and asset prompts, without installing tools.
S[speech_model]=''; S[speech_lexicon]=''
printf 'D\nN\n%s\nD\n%s\n\n\nNaN\n12.5\n' "$model" "$assets" > "$work/answers"
configure_speech < "$work/answers" > "$work/configure.log"
[[ ${S[speech_model]} == "$model" && ${S[speech_lexicon]} == "$assets" && ${S[speech_threads]} == 4 && ${S[speech_idle_timeout]} == 300 && ${S[speech_max_output_seconds]} == 12.5 ]]; pass
[[ $(grep -c '^MeloTTS-Chinese' "$work/configure.log") == 2 ]]; pass
save_state; defaults; load_state; [[ ${S[server_speech]} == true && ${S[speech_model]} == "$model" && ${S[speech_max_output_seconds]} == 12.5 ]]; pass
for bad in '"false"' null 1; do printf '{"server_speech":%s}\n' "$bad" > "$STATE"; defaults; reject load_state; done

# Full saved-state TTS-only launch exercises the real shell against the mock backend.
reset_speech; S[server_llm_enabled]=false; S[server_speech]=true
S[device]='stale saved GPU'; S[setup_mode]='stale saved profile'; S[model]='old relative llm.gguf'; S[model_history]='old model history'
save_state
HOME="$work" "$shell" "$launcher" --root "$ROOT" --yes --skip-update-check > "$work/main-speech.log" < /dev/null
grep -Fxq '<cpu>' "$SPEECH_CAPTURE"; grep -Fxq '<--speech-model>' "$SPEECH_CAPTURE"; ! grep -Fq 'old relative llm.gguf' "$SPEECH_CAPTURE"; pass
[[ ! -e $SPEECH_DEVICE_PROBES ]]; pass
defaults; load_state; [[ ${S[device]} == 'stale saved GPU' && ${S[model_history]} == 'old model history' ]]; pass
MOCK_DISABLE_TTS=1; export MOCK_DISABLE_TTS
if HOME="$work" "$shell" "$launcher" --root "$ROOT" --yes --skip-update-check > "$work/no-tts.log" 2>&1 < /dev/null; then exit 1; fi
grep -Fq 'native speech is unavailable in this build' "$work/no-tts.log"; pass
printf 'PASS: %d %s speech/configuration/release-route checks (mock backend only; no network, browser, Python, GPU or live engine)\n' "$checks" "$platform"
