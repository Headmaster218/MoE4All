#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${MOE4ALL_PYTHON:-python3}"
"$python" "$root/scripts/test_linux_tools.py"
bash "$root/scripts/test-linux-wizard.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/package" "$tmp/settings"
cp "$root/Start-INFR-Wizard-Linux.sh" "$tmp/package/"
printf '#!/bin/sh\nprintf "STUB_LAUNCHED\\n"\n' > "$tmp/package/infr"
chmod +x "$tmp/package/infr"
export XDG_CONFIG_HOME="$tmp/settings"
unset INFR_API_KEY
for mode in run serve bench; do
    out="$(bash "$tmp/package/Start-INFR-Wizard-Linux.sh" --dry-run --mode "$mode" --model '/models/model with spaces.gguf' --profile aggressive </dev/null)"
    [[ "$out" == *"$mode"* && "$out" == *device.auto_profile=aggressive* && "$out" != *STUB_LAUNCHED* ]]
done
out="$(bash "$tmp/package/Start-INFR-Wizard-Linux.sh" --dry-run --mode serve --model model.gguf --mtp head.gguf --parallel 2 --mtp-k 2 </dev/null)"
[[ "$out" == *spec.k=4* && "$out" == *"--temp 0"* ]]
if bash "$tmp/package/Start-INFR-Wizard-Linux.sh" --mode serve --model model.gguf --addr 0.0.0.0:8080 --no-api-key </dev/null >"$tmp/rejected" 2>&1; then
    printf 'Unauthenticated EOF launch was not rejected\n' >&2
    exit 1
fi
! grep -q STUB_LAUNCHED "$tmp/rejected"
if [ "$(uname -s)" = Linux ]; then
    bash "$tmp/package/Start-INFR-Wizard-Linux.sh" --mode serve --model model.gguf --no-api-key --yes </dev/null >"$tmp/launched"
    grep -q STUB_LAUNCHED "$tmp/launched"
    # The real CI packager and the single-SH runtime must agree on the manifest/archive.
    "$python" "$root/scripts/linux_release.py" package --root "$root" --binary "$tmp/package/infr" --output "$tmp/dist" --version 0.10.0 > "$tmp/archive-path"
    source "$root/Start-INFR-Wizard-Linux.sh"
    extract_package "$(< "$tmp/archive-path")" "$tmp/extracted" 0.10.0
    ! find "$tmp/extracted" -name '*.py' -print -quit | grep -q .
    bash "$tmp/extracted/Start-INFR-Wizard-Linux.sh" --dry-run --mode serve --model model.gguf --profile aggressive > "$tmp/extracted-command"
    grep -q device.auto_profile=aggressive "$tmp/extracted-command"
fi
printf 'smoke-linux-wizard: all checks passed\n'
