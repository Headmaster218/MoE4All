#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${MOE4ALL_PYTHON:-python3}"
"$python" "$root/scripts/test_linux_tools.py"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/package/scripts" "$tmp/settings"
cp "$root/Start-INFR-Wizard-Linux.sh" "$tmp/package/"
cp "$root/scripts/linux_wizard.py" "$root/scripts/linux_release.py" "$tmp/package/scripts/"
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
fi
printf 'smoke-linux-wizard: all checks passed\n'
