#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${MOE4ALL_PYTHON:-python3}"
command -v "$python" >/dev/null || { printf 'Python 3 is required (no pip packages).\n' >&2; exit 1; }
exec "$python" "$root/scripts/linux_wizard.py" --root "$root" "$@"
