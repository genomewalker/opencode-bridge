#!/usr/bin/env python3
"""Install/uninstall chitta-bridge for Claude Code and/or Codex CLI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_NAME = "chitta-bridge"
MARKETPLACE = "local"


# ── helpers ──────────────────────────────────────────────────────────

def _chitta_bridge_path() -> str:
    """Resolve absolute path to chitta-bridge binary."""
    return shutil.which("chitta-bridge") or "chitta-bridge"


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))


def _plugin_dir() -> Path:
    return _codex_home() / "plugins" / "cache" / MARKETPLACE / PLUGIN_NAME / "local"


def _codex_config() -> Path:
    return _codex_home() / "config.toml"


def _plugin_source_dir() -> Path:
    """Find the codex-plugin source directory.

    Checks in order:
      1. Alongside this package (editable / source install)
      2. Wheel shared-data location (pip install)
    """
    # Source / editable install
    src = Path(__file__).resolve().parent.parent / "codex-plugin"
    if src.is_dir():
        return src
    # Wheel shared-data
    import sysconfig
    data = Path(sysconfig.get_path("data")) / "share" / PLUGIN_NAME / "codex-plugin"
    if data.is_dir():
        return data
    return src  # fallback — will fail with clear error


# ── Claude Code ──────────────────────────────────────────────────────

def _install_claude_code():
    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "--transport", "stdio", "--scope", "user",
             "chitta-bridge", "--", "chitta-bridge"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("  Claude Code: registered")
        elif "already exists" in result.stderr.lower():
            print("  Claude Code: already registered")
        else:
            print(f"  Claude Code: failed — {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  Claude Code: 'claude' CLI not found — skipping")
        return False
    return True


def _uninstall_claude_code():
    try:
        result = subprocess.run(
            ["claude", "mcp", "remove", "chitta-bridge"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("  Claude Code: removed")
        elif "not found" in result.stderr.lower():
            print("  Claude Code: not registered")
        else:
            print(f"  Claude Code: failed — {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  Claude Code: 'claude' CLI not found — skipping")
        return False
    return True


# ── Codex CLI ────────────────────────────────────────────────────────

def _install_codex():
    source = _plugin_source_dir()
    if not source.is_dir():
        print(f"  Codex: plugin source not found at {source}")
        return False

    dest = _plugin_dir()
    dest.mkdir(parents=True, exist_ok=True)

    # Copy plugin manifest
    codex_plugin = source / ".codex-plugin"
    if codex_plugin.is_dir():
        shutil.copytree(codex_plugin, dest / ".codex-plugin", dirs_exist_ok=True)

    # Copy skills
    skills = source / "skills"
    if skills.is_dir():
        shutil.copytree(skills, dest / "skills", dirs_exist_ok=True)

    # Write .mcp.json with resolved absolute path
    cb_path = _chitta_bridge_path()
    mcp_json = {"mcpServers": {"chitta-bridge": {"command": cb_path, "args": []}}}
    (dest / ".mcp.json").write_text(json.dumps(mcp_json, indent=2) + "\n")

    # Enable plugin in config.toml
    config = _codex_config()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.touch(exist_ok=True)
    text = config.read_text()

    if "[features]" not in text:
        text += "\n[features]\nplugins = true\n"
    elif "plugins" not in text.split("[features]", 1)[1].split("\n[", 1)[0]:
        text = text.replace("[features]", "[features]\nplugins = true")

    if "chitta-bridge@local" not in text:
        text += f'\n[plugins."{PLUGIN_NAME}@{MARKETPLACE}"]\nenabled = true\n'

    config.write_text(text)

    print(f"  Codex: installed to {dest}")
    print(f"  Codex: MCP server → {cb_path}")
    print("  Codex: skills — /review, /rescue, /room, /soul")
    return True


def _uninstall_codex():
    dest = _plugin_dir()
    if dest.is_dir():
        shutil.rmtree(dest)
        print(f"  Codex: removed {dest}")
    else:
        print("  Codex: not installed")

    config = _codex_config()
    if config.is_file():
        text = config.read_text()
        if "chitta-bridge@local" in text:
            # Remove the plugin section
            lines = text.split("\n")
            out, skip = [], False
            for line in lines:
                if f'plugins."{PLUGIN_NAME}@{MARKETPLACE}"' in line:
                    skip = True
                    continue
                if skip and (line.startswith("[") or not line.strip()):
                    if not line.strip():
                        continue
                    skip = False
                if skip:
                    continue
                out.append(line)
            config.write_text("\n".join(out))
            print("  Codex: removed config entry")
    return True


# ── CLI ──────────────────────────────────────────────────────────────

TARGETS = {
    "claude-code": (_install_claude_code, _uninstall_claude_code),
    "codex": (_install_codex, _uninstall_codex),
}

USAGE = """usage: chitta-bridge-install [claude-code|codex|all]
       chitta-bridge-uninstall [claude-code|codex|all]

Targets:
  claude-code   Register MCP server with Claude Code
  codex         Install Codex CLI plugin (skills + MCP)
  all           Both (default)"""


def _parse_target(args: list[str]) -> list[str]:
    if not args or args[0] == "all":
        return list(TARGETS.keys())
    if args[0] in ("--help", "-h"):
        print(USAGE)
        sys.exit(0)
    if args[0] in TARGETS:
        return [args[0]]
    print(f"Unknown target: {args[0]}\n")
    print(USAGE)
    sys.exit(1)


def install():
    targets = _parse_target(sys.argv[1:])
    print("chitta-bridge install:")
    ok = True
    for t in targets:
        if not TARGETS[t][0]():
            ok = False
    if not ok:
        sys.exit(1)
    print("\ndone.")


def uninstall():
    targets = _parse_target(sys.argv[1:])
    print("chitta-bridge uninstall:")
    for t in targets:
        TARGETS[t][1]()
    print("\ndone.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        sys.argv.pop(1)
        uninstall()
    else:
        install()
