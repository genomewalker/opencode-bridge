#!/usr/bin/env python3
"""Install/uninstall opencode-bridge MCP server with Claude Code."""

import subprocess
import sys


def install():
    """Register opencode-bridge as an MCP server with Claude Code."""
    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "--transport", "stdio", "--scope", "user",
             "opencode-bridge", "--", "opencode-bridge"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("opencode-bridge registered with Claude Code")
            print(result.stdout)
        else:
            if "already exists" in result.stderr.lower():
                print("opencode-bridge already registered")
            else:
                print(f"Failed to register: {result.stderr}")
                sys.exit(1)
    except FileNotFoundError:
        print("Claude Code CLI not found. Install from: https://claude.ai/download")
        sys.exit(1)


def uninstall():
    """Remove opencode-bridge MCP server from Claude Code."""
    try:
        result = subprocess.run(
            ["claude", "mcp", "remove", "opencode-bridge"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("opencode-bridge removed from Claude Code")
            print(result.stdout)
        else:
            if "not found" in result.stderr.lower():
                print("opencode-bridge not registered")
            else:
                print(f"Failed to remove: {result.stderr}")
                sys.exit(1)
    except FileNotFoundError:
        print("Claude Code CLI not found")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall()
    else:
        install()
