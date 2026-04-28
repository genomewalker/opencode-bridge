"""Diagnostic subcommand for chitta-bridge.

Surfaces install-time and runtime footguns: missing CLIs, broken GPU URL
discovery dirs, malformed session/job/room JSON, persisted enum values
that the policy layer would now reject. Exits non-zero if any FAIL.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

CONFIG_DIR = Path.home() / ".chitta-bridge"
STATE_DIRS = {
    "opencode-sessions": CONFIG_DIR / "sessions",
    "codex-sessions": CONFIG_DIR / "codex-sessions",
    "codex-jobs": CONFIG_DIR / "codex-jobs",
    "rooms": CONFIG_DIR / "rooms",
}
DEFAULT_GPU_URL_DIR = str(CONFIG_DIR / "endpoints")
CODEX_EFFORTS = {"low", "medium", "high", "xhigh"}
CODEX_SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}
CURRENT_SCHEMA_VERSION = 1

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
SYMBOL = {PASS: "✓", WARN: "!", FAIL: "✗"}


def _print(level: str, name: str, detail: str = "") -> None:
    line = f"  {SYMBOL[level]} [{level:4}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_clis() -> list[str]:
    print("\nCLI presence:")
    results = []
    for cli in ("codex", "opencode", "ollama"):
        path = shutil.which(cli)
        if path:
            _print(PASS, cli, path)
            results.append(PASS)
        else:
            level = WARN  # WARN, not FAIL — bridge can run without all three
            _print(level, cli, "not on PATH (corresponding tools will be unusable)")
            results.append(level)
    return results


def check_gpu_url_dir() -> list[str]:
    print("\nGPU URL discovery dir:")
    env = os.environ.get("CHITTA_BRIDGE_URL_DIR")
    url_dir = Path(env or DEFAULT_GPU_URL_DIR)
    using_default = env is None
    if not url_dir.exists():
        if using_default:
            _print(WARN, "CHITTA_BRIDGE_URL_DIR",
                   f"default '{DEFAULT_GPU_URL_DIR}' missing; set CHITTA_BRIDGE_URL_DIR for local_discover")
        else:
            _print(FAIL, "CHITTA_BRIDGE_URL_DIR", f"'{url_dir}' does not exist")
        return [FAIL if not using_default else WARN]
    urls = sorted(url_dir.glob("ollama-server-*.url"))
    if not urls:
        _print(WARN, "ollama-server-*.url", f"none in {url_dir}; start an Ollama serve job")
        return [WARN]
    _print(PASS, "ollama-server-*.url", f"{len(urls)} endpoint file(s) in {url_dir}")
    return [PASS]


def _scan_json_dir(label: str, path: Path,
                   enum_fields: Iterable[tuple[str, set[str]]]) -> list[str]:
    if not path.exists():
        _print(PASS, label, "(no state yet)")
        return [PASS]
    bad = []
    bad_enum = []
    future_schema = []
    total = 0
    for f in sorted(path.glob("*.json")):
        total += 1
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            bad.append((f.name, str(e)))
            continue
        for field, allowed in enum_fields:
            value = data.get(field)
            if value is not None and value not in allowed:
                bad_enum.append((f.name, field, value))
        sv = data.get("schema_version", 0)
        if isinstance(sv, int) and sv > CURRENT_SCHEMA_VERSION:
            future_schema.append((f.name, sv))
    if bad:
        for fname, err in bad[:5]:
            _print(FAIL, f"{label}/{fname}", f"unparseable: {err}")
        if len(bad) > 5:
            print(f"    … and {len(bad) - 5} more unparseable")
        return [FAIL]
    levels: list[str] = []
    if bad_enum:
        for fname, field, value in bad_enum[:5]:
            _print(WARN, f"{label}/{fname}", f"unknown {field}={value!r}")
        if len(bad_enum) > 5:
            print(f"    … and {len(bad_enum) - 5} more")
        levels.append(WARN)
    if future_schema:
        for fname, sv in future_schema[:5]:
            _print(WARN, f"{label}/{fname}",
                   f"schema_version={sv} > {CURRENT_SCHEMA_VERSION} (newer chitta-bridge wrote this)")
        if len(future_schema) > 5:
            print(f"    … and {len(future_schema) - 5} more future-schema")
        levels.append(WARN)
    if levels:
        return levels
    _print(PASS, label, f"{total} file(s) parse cleanly")
    return [PASS]


def check_state() -> list[str]:
    print(f"\nPersisted state under {CONFIG_DIR}:")
    if not CONFIG_DIR.exists():
        _print(PASS, "config dir", "(not yet created)")
        return [PASS]
    results: list[str] = []
    enum_specs = {
        "opencode-sessions": (("sandbox", CODEX_SANDBOXES),),
        "codex-sessions": (("sandbox", CODEX_SANDBOXES),),
        "codex-jobs": (("effort", CODEX_EFFORTS), ("sandbox", CODEX_SANDBOXES)),
        "rooms": (),
    }
    for label, path in STATE_DIRS.items():
        results.extend(_scan_json_dir(label, path, enum_specs[label]))
    return results


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="chitta-bridge-doctor",
        description="Diagnose chitta-bridge install and state.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on WARN as well as FAIL (useful for CI).",
    )
    args = parser.parse_args()

    print(f"chitta-bridge doctor — diagnostic for {CONFIG_DIR}")
    results = check_clis() + check_gpu_url_dir() + check_state()
    fails = results.count(FAIL)
    warns = results.count(WARN)
    passes = results.count(PASS)
    print(f"\nSummary: {passes} pass, {warns} warn, {fails} fail"
          + (" (strict)" if args.strict else ""))
    if fails:
        return 1
    if args.strict and warns:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
