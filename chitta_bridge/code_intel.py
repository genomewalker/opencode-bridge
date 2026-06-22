"""
Code-intel helpers extracted from server.py.

Provides symbol-body caching, handle-based addressing, outline utilities,
linting, post-write reindex, composite code analysis, and file reading helpers.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

from chitta_bridge.io_utils import _content_hash
from chitta_bridge.symbols import _locate_symbol
from chitta_bridge.soul import SoulClient

__all__ = [
    "_cache_get",
    "_cache_put",
    "_cache_get_fresh",
    "_cache_pop",
    "_cache_pop_file",
    "_make_handle",
    "_resolve_handle",
    "_parse_outline_symbols",
    "_outline_diff",
    "_run_lint",
    "_post_write_refresh",
    "_code_intel",
    "_read_range",
    "_read_outline",
]

# ---------------------------------------------------------------------------
# Symbol-body cache
# Key: (session_id, file_resolved, symbol_name). Invalidated on mtime_ns change.
# Size-bounded LRU (~64 entries) to avoid unbounded growth in long sessions.
# ---------------------------------------------------------------------------

_symbol_body_cache: dict = None  # type: ignore
_SYMBOL_CACHE_MAX = 64


def _cache_get():
    global _symbol_body_cache
    if _symbol_body_cache is None:
        import collections as _c
        _symbol_body_cache = _c.OrderedDict()
    return _symbol_body_cache


def _current_session_id() -> str:
    """Best-effort session id. Falls back to 'default' if not available."""
    return os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("CODEX_SESSION_ID") or "default"


def _cache_put(file: Path, symbol: str, body: str,
               line_start: int, line_end: int) -> None:
    try:
        mtime = file.stat().st_mtime_ns
    except OSError:
        return
    key = (_current_session_id(), str(file.resolve()), symbol)
    cache = _cache_get()
    cache[key] = {"mtime_ns": mtime, "body": body,
                  "line_start": line_start, "line_end": line_end}
    cache.move_to_end(key)
    while len(cache) > _SYMBOL_CACHE_MAX:
        cache.popitem(last=False)


def _cache_get_fresh(file_hint: Optional[str], symbol: str) -> Optional[dict]:
    """Return cached entry if (session, file, symbol) matches AND file mtime
    is unchanged. If file_hint is None, accept only if there's exactly one
    session-local cached hit for this symbol (avoids name collisions)."""
    cache = _cache_get()
    sid = _current_session_id()
    candidates = []
    for (s, f, sym), v in cache.items():
        if s == sid and sym == symbol:
            if file_hint is None or Path(f) == Path(file_hint).resolve():
                candidates.append((f, v))
    if not candidates:
        return None
    if file_hint is None and len(candidates) > 1:
        return None
    f, v = candidates[0]
    try:
        if Path(f).stat().st_mtime_ns != v["mtime_ns"]:
            cache.pop((sid, f, symbol), None)
            return None
    except OSError:
        return None
    return {**v, "file": f}


def _cache_pop(file: Path, symbol: str) -> None:
    """Invalidate a cached symbol entry across all sessions for this file."""
    cache = _cache_get()
    try:
        resolved = str(file.resolve())
    except OSError:
        resolved = str(file)
    for key in list(cache.keys()):
        if key[1] == resolved and key[2] == symbol:
            cache.pop(key, None)


def _cache_pop_file(file: Path) -> None:
    """Invalidate ALL cached symbol entries for a file (e.g. after file_patch)."""
    cache = _cache_get()
    try:
        resolved = str(file.resolve())
    except OSError:
        resolved = str(file)
    for key in list(cache.keys()):
        if key[1] == resolved:
            cache.pop(key, None)


# ---------------------------------------------------------------------------
# Handle-based addressing
# ---------------------------------------------------------------------------

_handle_store: dict[str, dict] = {}
_HANDLE_STORE_MAX = 256


def _make_handle(file_resolved: str, symbol: str, body_hash: str) -> str:
    sid = _current_session_id()
    hid = hashlib.sha256((sid + file_resolved + symbol + body_hash).encode()).hexdigest()[:12]
    _handle_store[hid] = {"file": file_resolved, "symbol": symbol, "body_hash": body_hash}
    if len(_handle_store) > _HANDLE_STORE_MAX:
        for k in list(_handle_store.keys())[:len(_handle_store) - _HANDLE_STORE_MAX]:
            _handle_store.pop(k, None)
    return hid


def _resolve_handle(handle: str):
    """Return (file_str, symbol) or an error string."""
    rec = _handle_store.get(handle)
    if not rec:
        return "Error: unknown handle — re-read symbol first"
    p = Path(rec["file"])
    if not p.is_file():
        return "Error: stale handle — file gone, re-read symbol first"
    content = p.read_text(encoding="utf-8", errors="replace")
    loc = _locate_symbol(p, rec["symbol"], content)
    if loc is None or isinstance(loc, str):
        return "Error: stale handle — symbol not found, re-read symbol first"
    s, e, _ = loc
    if _content_hash(content[s:e]) != rec["body_hash"]:
        return "Error: stale handle — body changed, re-read symbol first"
    return rec["file"], rec["symbol"]


# ---------------------------------------------------------------------------
# Outline diff helpers
# ---------------------------------------------------------------------------

_OUTLINE_LINE_RE = re.compile(r'^\s*L\s*(\d+)\s+(.*\S)\s*$')


def _parse_outline_symbols(outline: str) -> dict:
    out: dict[str, int] = {}
    for ln in outline.splitlines():
        m = _OUTLINE_LINE_RE.match(ln)
        if m:
            out[m.group(2).strip()] = int(m.group(1))
    return out


def _outline_diff(before: str, after: str) -> str:
    b, a = _parse_outline_symbols(before), _parse_outline_symbols(after)
    added   = [f"+ {n} (L{a[n]})" for n in a if n not in b]
    removed = [f"- {n}" for n in b if n not in a]
    moved   = [f"~ {n} (L{b[n]}→L{a[n]})" for n in a if n in b and a[n] != b[n]]
    parts   = added + moved + removed
    return "\n\n**Symbol changes:** " + "  ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Linter-to-symbol mapping
# ---------------------------------------------------------------------------

def _run_lint(filepath: str) -> str:
    import shutil as _shutil
    import subprocess as _sp
    import json as _json
    p = Path(filepath)
    if p.suffix.lower() not in (".py", ".pyx"):
        return ""
    ruff = _shutil.which("ruff")
    if not ruff:
        return ""
    try:
        proc = _sp.run(
            [ruff, "check", "--output-format=json", str(p)],
            capture_output=True, text=True, timeout=5,
        )
        data = _json.loads(proc.stdout or "[]")
    except Exception:
        return ""
    if not data:
        return ""
    pairs: list[tuple[int, str]] = []
    for ln in _read_outline(str(p)).splitlines():
        m = _OUTLINE_LINE_RE.match(ln)
        if m:
            pairs.append((int(m.group(1)), m.group(2).strip().split()[-1]))
    pairs.sort()

    def sym_for(row: int) -> str:
        name = "?"
        for lno, nm in pairs:
            if lno <= row:
                name = nm
            else:
                break
        return name

    out = []
    for e in data[:5]:
        code = e.get("code", "?")
        row  = (e.get("location") or {}).get("row", 0)
        msg  = e.get("message", "")
        out.append(f"{code} in `{sym_for(row)}` L{row}: {msg}")
    return "\n\n**Lint:** " + "  ".join(out) if out else ""


def _post_write_refresh(paths) -> None:
    """Fast-path reindex after a bridge-owned write. Bypasses the
    5-min rate limiter in file-changed-hook.sh because these writes
    come from our own MCP tools. Best-effort — short timeout; failure
    is logged but never blocks the patch."""
    if not paths:
        return
    if isinstance(paths, (str, Path)):
        paths = [paths]
    seen_dirs = set()
    for p in paths:
        try:
            path_obj = Path(p) if not isinstance(p, Path) else p
            dir_path = str(path_obj.parent)
            if dir_path in seen_dirs:
                continue
            seen_dirs.add(dir_path)
            SoulClient.learn_codebase(dir_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Composite code analysis
# ---------------------------------------------------------------------------

def _code_intel(symbol: str = "", path: str = "", realm: Optional[str] = None) -> str:
    """Composite code analysis: structure + call graph + imports + chitta memory.

    Smarter than tldr: fuses static analysis with chitta's knowledge graph so
    you get callers, callees, imports, and every memory chitta holds about this
    symbol — in one call, zero extra tokens spent navigating.
    """
    parts: list[str] = []

    # 1. File-level structure
    if path:
        ctx = SoulClient._call("code_context", {"path": path})
        if ctx:
            parts.append(f"## Structure: {path}\n{ctx}")
        imports = SoulClient._call("file_imports", {"path": path})
        if imports:
            parts.append(f"## Imports\n{imports}")

    # 2. Symbol call graph
    if symbol:
        source = SoulClient._call("read_symbol", {"name": symbol})
        if source:
            parts.append(f"## Source: {symbol}\n{source}")
        callers = SoulClient._call("symbol_callers", {"name": symbol})
        if callers:
            parts.append(f"## Callers → {symbol}\n{callers}")
        callees = SoulClient._call("symbol_callees", {"name": symbol})
        if callees:
            parts.append(f"## {symbol} → Callees\n{callees}")

    # 3. Chitta memory recall — what the system remembers about this symbol/file
    query = " ".join(filter(None, [symbol, path]))
    if query:
        mem = SoulClient.hybrid_recall(query, limit=5, realm=realm)
        if mem:
            parts.append(f"## Memory\n{mem}")

    if not parts:
        return "(no symbol or path provided)"
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------

def _read_range(filepath: str, start_line: int, end_line: int) -> str:
    """Read a line range (1-based, inclusive) from any file."""
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error reading {p.name}: {e}"
    total = len(lines)
    start_line = max(1, start_line)
    end_line = min(total, end_line)
    if start_line > end_line:
        return f"Error: start_line ({start_line}) > end_line ({end_line}) for {p.name} ({total} lines)"
    snippet = lines[start_line - 1:end_line]
    header = f"# {p.name}  L{start_line}–{end_line} / {total}\n"
    return header + "\n".join(f"{i + start_line:6d}\t{ln}" for i, ln in enumerate(snippet))


def _read_outline(filepath: str) -> str:
    """List symbols with line numbers via regex scan (Python/Rust/Go/JS/C)."""
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error reading {p.name}: {e}"
    ext = p.suffix.lower()
    if ext in (".py", ".pyx"):
        pat = re.compile(r'^(\s*)(class|async def|def)\s+(\w+)')
    elif ext in (".rs",):
        pat = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(fn|struct|enum|impl|trait|mod)\s+(\w+)')
    elif ext in (".go",):
        pat = re.compile(r'^(?:func\s+(?:\([^)]+\)\s+)?(\w+)|type\s+(\w+)\s+(?:struct|interface))')
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        pat = re.compile(r'^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|(const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\()')
    elif ext in (".c", ".cpp", ".cc", ".h", ".hpp"):
        pat = re.compile(r'^(?:[\w:*&<>\s]+\s+)?(\w+)\s*\([^;]*$')
    else:
        pat = re.compile(r'^(?:pub\s+)?(?:async\s+)?(fn|class|struct|enum|def|function)\s+(\w+)')
    out = [f"# Outline: {p.name} ({len(lines)} lines)"]
    for i, ln in enumerate(lines, 1):
        m = pat.match(ln)
        if not m:
            continue
        if ext in (".py", ".pyx"):
            indent_depth = len(m.group(1)) // 4
            out.append(f"  L{i:5d}  {'  ' * indent_depth}{m.group(2)} {m.group(3)}")
        else:
            name = next((g for g in m.groups() if g), "?")
            kind = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
            out.append(f"  L{i:5d}  {kind} {name}".rstrip())
    if len(out) == 1:
        out.append("  (no symbols found — unsupported file type or empty file)")
    return "\n".join(out)
