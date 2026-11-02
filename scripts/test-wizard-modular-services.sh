#!/usr/bin/env bash
# bash ... WORK linux, or zsh ... WORK macos. No real engine, GPU or downloads.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=${1:?Pass a task directory below MOE4ALL_WORK_ROOT/tmp/moe4all}
platform=${2:-linux}
case $work in "${MOE4ALL_WORK_ROOT:?}/tmp/moe4all/"*) ;; *) exit 1 ;; esac
case /$work/ in */../*|*/./*) exit 1 ;; esac
MOE4ALL_WIZARD_LIBRARY=1
case $platform in
    linux) launcher="$repo/Start-INFR-Wizard-Linux.sh"; shell=bash; device=Vulkan1 ;;
    macos) launcher="$repo/Start-INFR-Wizard-macOS.sh"; shell=zsh; device=metal ;;
    *) exit 1 ;;
esac
source "$launcher"
mkdir -p -- "$work"; work=$(mktemp -d "$work/modules.XXXXXX")
export TMPDIR="$work" TMP="$work" TEMP="$work"
ROOT="$work/package space's"; CONFIG_DIR="$ROOT/gui-data"; STATE="$CONFIG_DIR/wizard-state.json"
if [[ $platform == macos ]]; then CONFIG_DIR="$work/home/Library/Application Support/MoE4All"; STATE="$CONFIG_DIR/wizard-state.json"; fi
LEGACY_STATE="$work/missing.json"; LEGACY_CONFIG_DIR="$work/legacy"; BINARY="$ROOT/infr"
mkdir -p -- "$ROOT" "$CONFIG_DIR" "$work/home" "$work/cache"
export XDG_CONFIG_HOME="$work/home" XDG_CACHE_HOME="$work/cache"
export MODULAR_CAPTURE="$work/captured-argv" MODULAR_DEVICES="$work/device-probes"
unset INFR_API_KEY
cat > "$BINARY" <<'ENGINE'
#!/bin/sh
case ${1-} in
    devices)
        printf 'probe\n' >> "$MODULAR_DEVICES"
        printf '  Vulkan1: Fixture GPU [discrete, 24 GiB] <- default\n' ;;
    --version) printf 'infr 1.0.2\n' ;;
    *) printf '<%s>\n' "$@" > "$MODULAR_CAPTURE" ;;
esac
ENGINE
chmod +x "$BINARY"
llm="$ROOT/llm space's.gguf"; embedding="$ROOT/embed space's.gguf"; audio="$ROOT/audio space's.gguf"
touch "$llm" "$embedding" "$audio"
checks=0
pass() { ((checks+=1)); }
has_arg() { local arg; for arg in "${COMMAND[@]}"; do [[ $arg != "$1" ]] || return 0; done; return 1; }
reject() { if ( "$@" ) > "$work/reject.log" 2>&1; then printf 'Unexpected success: %s\n' "$*" >&2; exit 1; fi; pass; }
reset() {
    defaults; OVERRIDES=(); S[launch_mode]=serve; S[device]=$device
    S[model]=$llm; S[embedding_model]=$embedding; S[transcription_model]=$audio
    S[transcription_upload_dir]="$work/upload space's"
}
llm_keys=(setup_mode model model_history context ubatch threads config_path kv_preset kv_type_k kv_type_v
    configure_memory vram_budget vram_reserve expert_cache ram_budget host_dma pager_ring pager_ring_slots
    kv_overflow kv_overflow_vram_mb kv_overflow_reserve_mb submit_mode submit_cap configure_diagnostics
    pager_stats pager_profile stage_profile vram_profile think_mode configure_thinking reasoning_effort max_new
    configure_sampling temperature top_k top_p seed server_parallel server_session_cache session_cache_dir
    session_idle_secs session_cache_max session_cache_ttl_hours server_vision vision_projector mtp_enabled
    mtp_model mtp_verify_tokens cpu_miss_enabled cpu_miss_max cpu_miss_cores custom_sets auto_overrides)
snapshot_llm() { local key; for key in "${llm_keys[@]}"; do printf '<%s=%s>\n' "$key" "${S[$key]}"; done; }
saved_llm() {
    S[model]='old relative model.gguf'; S[model_history]='old history'; S[setup_mode]=aggressive
    S[kv_preset]=q8; S[kv_type_k]=q8_0; S[kv_type_v]=q8_0; S[context]=150k
    S[mtp_enabled]=true; S[mtp_model]='old relative draft.gguf'; S[mtp_verify_tokens]=2
    S[server_vision]=true; S[vision_projector]='old relative projector.gguf'
    S[server_session_cache]=true; S[session_cache_dir]='old relative kv'; S[server_parallel]=3
    S[config_path]='old relative config.toml'; S[custom_sets]='malformed saved LLM option'
    S[cpu_miss_enabled]=true; S[cpu_miss_cores]=bad; S[auto_overrides]='ubatch,ram_budget'
}

# JSON without the new key must retain the old LLM-enabled behavior.
printf '{"launch_mode":"server","model":"legacy model.gguf","setup_mode":"conservative"}\n' > "$STATE"
defaults; load_state
[[ ${S[server_llm_enabled]} == true && ${S[server_embedding]} == false && ${S[server_transcription]} == false ]]; pass
build_command; has_arg 'legacy model.gguf'; has_arg device.auto_profile=conservative; pass
for invalid in '"false"' null 0; do
    printf '{"server_llm_enabled":%s}\n' "$invalid" > "$STATE"; defaults; reject load_state
done
reset; S[server_llm_enabled]=false; reject build_command

# All seven combinations use serve; no LLM means no position model or LLM settings.
for l in false true; do for e in false true; do for a in false true; do
    [[ $l == true || $e == true || $a == true ]] || continue
    reset; S[server_llm_enabled]=$l; S[server_embedding]=$e; S[server_transcription]=$a
    build_command
    has_arg serve
    if [[ $l == true ]]; then has_arg "$llm"; has_arg --parallel; has_arg device.auto_profile=conservative
    else
        ! has_arg "$llm"; ! has_arg --parallel
        expected=("$BINARY" serve --dev "$device" --set serve.api_key= --addr 127.0.0.1:8080)
        if [[ $e == false ]]; then expected=("$BINARY" serve --dev cpu --set serve.api_key= --addr 127.0.0.1:8080); fi
        if [[ $e == true ]]; then expected+=(--embedding-model "$embedding" --embedding-idle-timeout 300); fi
        if [[ $a == true ]]; then expected+=(--transcription-model "$audio" --transcription-dev cpu --transcription-threads 4 --transcription-idle-timeout 300 --transcription-upload-dir "$work/upload space's"); fi
        [[ $(printf '<%s>\n' "${COMMAND[@]}") == "$(printf '<%s>\n' "${expected[@]}")" ]]
    fi
    if [[ $e == true ]]; then has_arg --embedding-model; else ! has_arg --embedding-model; fi
    if [[ $a == true ]]; then has_arg --transcription-model; has_arg --transcription-dev; else ! has_arg --transcription-model; fi
    [[ ${S[device]} == "$device" ]]; pass
    # Execute only the mock: verifies actual argv boundaries with spaces and apostrophes.
    "${COMMAND[@]}"
    expected_capture=$(printf '<%s>\n' "${COMMAND[@]:1}")
    [[ $(cat "$MODULAR_CAPTURE") == "$expected_capture" ]]; pass
done; done; done

# Real prompt/input path: all-off retries, three bools precede model/device prompts.
reset
configure_api_modules > "$work/module-prompts" <<'INPUT'
n
n
n
n
y
y
INPUT
[[ ${S[server_llm_enabled]} == false && ${S[server_embedding]} == true && ${S[server_transcription]} == true ]]; pass
grep -Fq 'Enable at least one API module.' "$work/module-prompts"; pass
for expected_text in 'Enable the language model API?' 'Enable the Embedding API?' 'Enable the local transcription API?'; do
    grep -Fq "$expected_text" "$work/module-prompts"; pass
done

# Disabled LLM settings survive configure, command migration, save/load AND main.
for aux in audio embedding both; do
    reset; saved_llm; before=$(snapshot_llm); previous_device=${S[device]}
    # Each prompt is answered explicitly; EOF would fail if a hidden LLM option appeared.
    {
        printf '\nn\n'
        if [[ $aux == audio ]]; then printf 'n\ny\n'; else printf 'y\n'; [[ $aux != embedding ]] || printf 'n\n'; [[ $aux != both ]] || printf 'y\n'; fi
        if [[ $aux != audio ]]; then printf '%s\n\n' "$embedding"; fi
        if [[ $aux != embedding ]]; then printf '%s\n\n\n\n' "$audio"; fi
        if [[ $aux != audio && $platform == linux ]]; then printf '\n'; fi
        printf '\n\n'
    } > "$work/answers-$aux"
    configure < "$work/answers-$aux" > "$work/prompts-$aux"
    [[ $(snapshot_llm) == "$before" && ${S[device]} == "$previous_device" ]]; pass
    if [[ $aux == audio ]]; then [[ ! -f $MODULAR_DEVICES ]]; fi
    # No LLM-owned options, model references or implicit migrations can leak.
    reset_legacy_mtp; build_command; remember_model
    [[ $(snapshot_llm) == "$before" ]]; pass
    save_state
    # Settings from another/newer wizard must survive an auxiliary-only save too.
    if [[ $aux == audio ]]; then
        { printf '{"future_llm":{"ctx":8192,"items":[true,"keep"]},\n'; tail -n +2 "$STATE"; } > "$work/extended-state"
        mv -- "$work/extended-state" "$STATE"
    fi
    save_state; defaults; load_state
    json_get '["future_llm","ctx"]'; [[ $REPLY == 8192 ]]; pass
    [[ $(snapshot_llm) == "$before" && ${S[device]} == "$previous_device" && ${S[server_llm_enabled]} == false ]]; pass
    HOME="$work/home" "$shell" "$launcher" --root "$ROOT" --yes --skip-update-check > "$work/main-$aux.log" < /dev/null
    defaults; load_state
    [[ $(snapshot_llm) == "$before" && ${S[device]} == "$previous_device" ]]; pass
    json_get '["future_llm","items",1]'; [[ $REPLY == keep ]]; pass
    # A valid mock launch must have happened, with no positional LLM.
    ! grep -Fq 'old relative model.gguf' "$MODULAR_CAPTURE"; pass
done

# Audio-alone never probes GPUs, even with an invalid/stale saved device/profile.
reset; saved_llm; S[server_llm_enabled]=false; S[server_transcription]=true
S[device]='remembered GPU'; S[setup_mode]='remembered profile'; save_state
HOME="$work/home" "$shell" "$launcher" --root "$ROOT" --yes --skip-update-check > "$work/main-audio-stale.log" < /dev/null
defaults; load_state
[[ ${S[device]} == 'remembered GPU' && ${S[setup_mode]} == 'remembered profile' ]]; pass
grep -Fxq '<cpu>' "$MODULAR_CAPTURE"; pass
[[ ! -f $MODULAR_DEVICES || $(wc -l < "$MODULAR_DEVICES") == 2 ]]; pass

# Defaults retained for chat/bench even when the API-only LLM switch is false.
for mode in run bench; do
    reset; S[launch_mode]=$mode; S[server_llm_enabled]=false; S[server_embedding]=true; S[server_transcription]=true
    build_command; has_arg "$llm"; ! has_arg --embedding-model; ! has_arg --transcription-model; pass
done

# Explicit D remains usable in an audio-only API flow. No real network calls.
reset; S[model]=''; S[server_llm_enabled]=false; S[transcription_model]=''
downloads=0
download_bundle() {
    [[ $1 == sensevoice-small-q8 ]]; ((downloads+=1))
    mkdir -p -- "$ROOT/models/SenseVoice-Small-Q8_0"
    touch "$ROOT/models/SenseVoice-Small-Q8_0/sense-voice-small-q8_0.gguf"
}
printf '\nn\nn\ny\nD\n1\n\n\n\n\n\n' > "$work/download-answers"
configure < "$work/download-answers" > "$work/audio-download.log"
[[ $downloads == 1 && ${S[model]} == '' ]]; build_command; has_arg --transcription-model; pass
grep -Fq '4 threads by default; the loaded model uses about 300 MiB RAM.' "$work/audio-download.log"; pass
grep -Fq 'loads on the first request; no extra GPU weights or external runtime are needed.' "$work/audio-download.log"; pass
grep -Fq 'First release: WAV/PCM' "$work/audio-download.log"; pass
printf 'PASS: %d %s modular-service checks (mock engine only)\n' "$checks" "$platform"
