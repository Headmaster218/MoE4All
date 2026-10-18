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
