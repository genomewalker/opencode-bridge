#!/usr/bin/env bash
# Install chitta-bridge as a Codex plugin.
# Usage: ./install.sh [--uninstall]
set -euo pipefail

PLUGIN_NAME="chitta-bridge"
MARKETPLACE="local"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
PLUGIN_DIR="${CODEX_HOME}/plugins/cache/${MARKETPLACE}/${PLUGIN_NAME}/local"
CONFIG="${CODEX_HOME}/config.toml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

_uninstall() {
    if [ -d "$PLUGIN_DIR" ]; then
        rm -rf "$PLUGIN_DIR"
        echo "Removed plugin: $PLUGIN_DIR"
    else
        echo "Plugin not installed."
    fi
    # Remove config entries
    if [ -f "$CONFIG" ] && grep -q "chitta-bridge@local" "$CONFIG"; then
        sed -i '/plugins\."chitta-bridge@local"/,/^$/d' "$CONFIG"
        echo "Removed plugin config from $CONFIG"
    fi
    echo "Uninstall complete."
}

_install() {
    # Check chitta-bridge is available
    if ! command -v chitta-bridge &>/dev/null; then
        echo "Error: chitta-bridge not found in PATH."
        echo "Install with: pip install chitta-bridge  (or:  uv pip install chitta-bridge)"
        exit 1
    fi

    # Copy plugin files
    mkdir -p "$PLUGIN_DIR"
    cp -r "${SCRIPT_DIR}/.codex-plugin" "$PLUGIN_DIR/"
    cp -r "${SCRIPT_DIR}/.mcp.json" "$PLUGIN_DIR/"
    cp -r "${SCRIPT_DIR}/skills" "$PLUGIN_DIR/"
    echo "Installed plugin to: $PLUGIN_DIR"

    # Resolve chitta-bridge to absolute path in .mcp.json
    CB_PATH="$(command -v chitta-bridge)"
    cat > "${PLUGIN_DIR}/.mcp.json" <<EOF
{
  "mcpServers": {
    "chitta-bridge": {
      "command": "${CB_PATH}",
      "args": []
    }
  }
}
EOF
    echo "MCP server: $CB_PATH"

    # Enable plugin in config.toml
    mkdir -p "$(dirname "$CONFIG")"
    touch "$CONFIG"

    # Ensure features.plugins = true
    if ! grep -q '^\[features\]' "$CONFIG" 2>/dev/null; then
        printf '\n[features]\nplugins = true\n' >> "$CONFIG"
    elif ! grep -q 'plugins\s*=\s*true' "$CONFIG"; then
        sed -i '/^\[features\]/a plugins = true' "$CONFIG"
    fi

    # Add plugin entry if missing
    if ! grep -q "chitta-bridge@local" "$CONFIG" 2>/dev/null; then
        printf '\n[plugins."chitta-bridge@local"]\nenabled = true\n' >> "$CONFIG"
    fi

    echo ""
    echo "chitta-bridge plugin installed for Codex."
    echo "Skills: /review, /rescue, /room, /soul"
    echo "Tools: mcp__chitta_bridge__* (soul memory, rooms, web, codex jobs)"
}

if [ "${1:-}" = "--uninstall" ]; then
    _uninstall
else
    _install
fi
