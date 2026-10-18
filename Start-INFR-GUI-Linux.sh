#!/usr/bin/env bash
# MoE4All 网页控制台 (infr-gui) 启动器  —— 由 systemd --user 托管，不怕会话回收
#   用法: ./Start-INFR-GUI-Linux.sh single | dual     (默认 single)
#   打开: http://127.0.0.1:8180
#   停:   systemctl --user stop infr-gui
set -euo pipefail
case "${1:-single}" in
  single) ep=Vulkan0 ;;            # 只喂 7900XTX #1
  dual)   ep=Vulkan0,Vulkan2 ;;    # 两张卡做 EP（Vulkan1 是 llvmpipe=CPU，别用）
  *)      echo "用法: $0 single|dual" >&2; exit 2 ;;
esac

cat > "$HOME/.config/infr-gui.env" <<EOF
INFR_EXPERT_PARALLEL=$ep
INFR_AUTO_PROFILE=aggressive
DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1
INFR_EP_HOST_BUDGET=24GiB
INFR_EP_POOL_EXTRA_SLOTS=96
EOF

systemctl --user restart infr-gui
sleep 2
systemctl --user is-active infr-gui
echo "→ 浏览器打开 http://127.0.0.1:8180   (日志: \$HOME/.local/share/infr-gui.log)"
