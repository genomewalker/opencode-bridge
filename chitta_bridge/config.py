"""Module-level configuration, binary discovery, and session-state helpers."""

import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from chitta_bridge.io_utils import _path_write_lock, _atomic_write_text

__all__ = [
    "Config",
    "find_codex",
    "_get_ppid_chain",
    "_chitta_sql",
    "_get_claude_session_id",
    "_chitta_session_alive",
    "_strip_startup_warnings",
    "_migrate_persisted",
    # constants also exported for callers that import from here
    "DEFAULT_MODEL",
    "DEFAULT_AGENT",
    "DEFAULT_VARIANT",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_SANDBOX",
    "CODEX_BIN",
    "CLAUDE_BIN",
    "PERSISTED_SCHEMA_VERSION",
    "_STARTUP_WARNING_PREFIXES",
    "_CHITTA_MIND_DIR",
    "_UUID_RE",
]

# Default configuration
DEFAULT_MODEL = "openai/gpt-5.3-codex"
DEFAULT_AGENT = "plan"
DEFAULT_VARIANT = "medium"

# Codex defaults
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_SANDBOX = "danger-full-access"


@dataclass
class Config:
    # OpenCode settings
    model: str = DEFAULT_MODEL
    agent: str = DEFAULT_AGENT
    variant: str = DEFAULT_VARIANT
    # Codex settings
    codex_model: str = DEFAULT_CODEX_MODEL
    codex_sandbox: str = DEFAULT_CODEX_SANDBOX

    @classmethod
    def load(cls) -> "Config":
        config = cls()

        # Load from config file
        config_path = Path.home() / ".chitta-bridge" / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                config.model = data.get("model", config.model)
                config.agent = data.get("agent", config.agent)
                config.variant = data.get("variant", config.variant)
                config.codex_model = data.get("codex_model", config.codex_model)
                config.codex_sandbox = data.get("codex_sandbox", config.codex_sandbox)
            except Exception:
                pass

        # Environment variables override config file
        config.model = os.environ.get("OPENCODE_MODEL", config.model)
        config.agent = os.environ.get("OPENCODE_AGENT", config.agent)
        config.variant = os.environ.get("OPENCODE_VARIANT") or config.variant
        config.codex_model = os.environ.get("CODEX_MODEL", config.codex_model)
        config.codex_sandbox = os.environ.get("CODEX_SANDBOX", config.codex_sandbox)

        return config

    def save(self):
        config_dir = Path.home() / ".chitta-bridge"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        data = {
            "model": self.model,
            "agent": self.agent,
            "variant": self.variant,
            "codex_model": self.codex_model,
            "codex_sandbox": self.codex_sandbox,
        }
        with _path_write_lock(config_path):
            _atomic_write_text(config_path, json.dumps(data, indent=2))


def find_codex() -> Optional[Path]:
    """Find codex binary."""
    # Check common locations
    paths = [
        Path.home() / ".codex" / "bin" / "codex",
        Path("/usr/local/bin/codex"),
        Path("/usr/bin/codex"),
    ]
    for p in paths:
        if p.exists():
            return p
    # Check PATH
    which = shutil.which("codex")
    if which:
        return Path(which)
    return None


CODEX_BIN = find_codex()
CLAUDE_BIN = shutil.which("claude")

_STARTUP_WARNING_PREFIXES = (
    "WARNING: failed to clean up stale",
)

_CHITTA_MIND_DIR = Path.home() / ".claude" / "mind"


def _get_ppid_chain() -> list[int]:
    """Return PIDs from current process up to init."""
    pids = []
    pid = os.getpid()
    for _ in range(15):
        pids.append(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    pid = int(line.split()[1])
                    break
            else:
                break
        except OSError:
            break
        if pid <= 1:
            break
    return pids


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _chitta_sql(query: str, timeout: int = 5) -> Optional[str]:
    """Run a chitta sql_query and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["chitta", "sql_query", "--query", query],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def _get_claude_session_id() -> Optional[str]:
    """Look up the current Claude Code session ID from chitta's session_registry.

    Chitta stores session_id keyed by Claude's PID in the session_registry DuckDB
    table. We walk up the process tree and ask chitta for a match.
    """
    pids = _get_ppid_chain()
    pid_list = ",".join(str(p) for p in pids)
    output = _chitta_sql(
        f"SELECT session_id FROM session_registry WHERE pid IN ({pid_list}) AND status='active' ORDER BY last_heartbeat DESC LIMIT 1"
    )
    if output:
        for line in output.splitlines():
            candidate = line.strip().strip("|").strip()
            if _UUID_RE.match(candidate):
                return candidate
    return os.environ.get("CLAUDE_SESSION_ID")


def _chitta_session_alive(claude_session_id: str) -> Optional[bool]:
    """Check if a Claude Code session is still active in chitta's registry.

    Returns True if active, False if dead/missing, None if chitta unavailable.
    """
    if not _UUID_RE.match(claude_session_id):
        return None
    output = _chitta_sql(
        f"SELECT COUNT(*) FROM session_registry WHERE session_id='{claude_session_id}' AND status='active'"
    )
    if output is None:
        return None
    for line in output.splitlines():
        candidate = line.strip().strip("|").strip()
        if candidate.isdigit():
            return int(candidate) > 0
    return None


def _strip_startup_warnings(text: str) -> str:
    """Remove known benign startup warnings emitted to stderr by OpenCode/Codex binaries."""
    lines = [line for line in text.splitlines() if not line.startswith(_STARTUP_WARNING_PREFIXES)]
    return "\n".join(lines).strip()


# Persisted state schema version. Bump whenever an on-disk shape changes
# in a way that needs migration. Absence in a JSON file means v0 (pre-versioning).
PERSISTED_SCHEMA_VERSION = 1


def _migrate_persisted(data: dict, kind: str) -> dict:
    """Lazily migrate a loaded JSON dict to PERSISTED_SCHEMA_VERSION.

    kind: "session" | "codex_session" | "codex_job" | "room" — reserved for
    per-kind migration steps. Currently v0→v1 is a no-op (load() already
    tolerates missing fields via .get with defaults), so we just stamp the
    version. Newer-than-current files are left untouched and surfaced by
    chitta-bridge-doctor as WARN.
    """
    version = data.get("schema_version", 0)
    if version > PERSISTED_SCHEMA_VERSION:
        return data  # forward-compat: don't downgrade, doctor will warn
    # v0 → v1: no field shape changes, just stamp.
    data["schema_version"] = PERSISTED_SCHEMA_VERSION
    # Room migration: retry_counts was keyed by turn_key ("r{N}:{name}"); normalize to name.
    if kind == "room" and "retry_counts" in data:
        old = data["retry_counts"]
        migrated: dict = {}
        for k, v in old.items():
            if k.startswith("r") and ":" in k:
                try:
                    int(k[1:k.index(":")])  # verify numeric round prefix
                    name = k[k.index(":") + 1:]
                    migrated[name] = max(migrated.get(name, 0), v)
                    continue
                except ValueError:
                    pass
            migrated[k] = max(migrated.get(k, 0), v)
        data["retry_counts"] = migrated
    return data
