"""Symbol-level file-editing helpers extracted from server.py.

These functions locate, patch, edit, delete, rename, move, and insert
named symbols inside source files. They rely on chitta's tree-sitter index
for non-Python files and fall back to indent-aware regex for .py/.pyx.
"""

from __future__ import annotations

import re
from pathlib import Path

from chitta_bridge.io_utils import (
    _reject_sensitive_path,
    _path_write_lock,
    _atomic_write_text,
    _content_hash,
)

# Late imports from server to avoid circular dependency at module load time.
# These are resolved on first function call, not at import time.
def _srv():
    import chitta_bridge.server as _s
    return _s


__all__ = [
    "_apply_file_patch",
    "_find_symbol_range",
    "_merge_delta",
    "_apply_symbol_patch",
    "_apply_symbol_edit",
    "_locate_symbol",
    "_apply_symbol_delete",
    "_apply_symbol_rename",
    "_apply_symbol_rename_project",
    "_apply_symbol_move",
    "_apply_symbol_insert_child",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DELTA_MARKERS = frozenset({
    '# ... existing code ...',
    '// ... existing code ...',
    '# ...',
    '// ...',
})

_IDENT_RE_TMPL = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])"

_INSERT_POSITIONS = (
    "start", "end", "before_return", "after_last_import", "after_docstring",
)


# ---------------------------------------------------------------------------
# _apply_file_patch
# ---------------------------------------------------------------------------

def _apply_file_patch(filepath: str, old_str: str, new_str: str) -> str:
    """Apply a search-replace patch. Returns compact diff summary on success."""
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    blocked = _reject_sensitive_path(p)
    if blocked:
        return f"Error: {blocked}"

    s = _srv()
    outline_before = s._read_outline(str(p))

    with _path_write_lock(p):
        try:
            content = p.read_text(encoding="utf-8")
            pre_hash = _content_hash(content)
        except OSError as e:
            return f"Error reading {p.name}: {e}"

        count = content.count(old_str)
        if count == 0:
            preview = old_str[:80].replace('\n', '↵')
            first_frag = old_str.split('\n')[0].strip()[:40]
            candidates = []
            if first_frag:
                for i, ln in enumerate(content.splitlines(), 1):
                    if first_frag[:20] in ln:
                        candidates.append(f"  L{i}: {ln[:100]}")
                        if len(candidates) >= 3:
                            break
            hint = ("\nNearest lines containing first fragment:\n" + "\n".join(candidates)) if candidates else ""
            return f"Error: old_str not found in {p.name}\nSearched for: {preview!r}{hint}"
        if count > 1:
            match_lines = []
            pos = 0
            while True:
                idx = content.find(old_str, pos)
                if idx == -1:
                    break
                match_lines.append(f"L{content[:idx].count(chr(10)) + 1}")
                pos = idx + 1
            return f"Error: old_str matches {count} locations in {p.name} at {', '.join(match_lines)} — make it more specific"

        line_num = content[:content.index(old_str)].count('\n') + 1
        old_lines = old_str.count('\n') + 1
        new_lines = new_str.count('\n') + (1 if new_str else 0)
        delta = new_lines - old_lines

        try:
            if _content_hash(p.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {p.name} changed on disk since read — retry"
            _atomic_write_text(p, content.replace(old_str, new_str, 1))
        except OSError as e:
            return f"Error writing {p.name}: {e}"

    s._post_write_refresh(p)
    s._cache_pop_file(p)
    sign = "+" if delta >= 0 else ""
    msg = f"✓ {p.name} patched @ L{line_num} (+{new_lines}/-{old_lines} lines, net {sign}{delta})"
    msg += s._outline_diff(outline_before, s._read_outline(str(p)))
    msg += s._run_lint(str(p))
    return msg


# ---------------------------------------------------------------------------
# _find_symbol_range
# ---------------------------------------------------------------------------

def _find_symbol_range(content: str, symbol: str, ext: str):
    """Return (start, end) byte range of a named symbol for .py/.pyx files only.

    The brace-language fallback is intentionally absent — it silently corrupts
    ranges when braces appear inside strings or comments. Callers must hard-fail
    for non-Python files when tree-sitter is unavailable.

    Includes preceding @decorator lines in the returned range.
    """
    if ext not in (".py", ".pyx"):
        return None
    patterns = [
        rf"^(\s*)(async\s+def\s+{re.escape(symbol)}\s*[\(:])",
        rf"^(\s*)(def\s+{re.escape(symbol)}\s*[\(:])",
        rf"^(\s*)(class\s+{re.escape(symbol)}\s*[\(:])",
    ]
    for pat in patterns:
        for m in re.finditer(pat, content, re.MULTILINE):
            indent = len(m.group(1))
            start = m.start()
            # Walk backward to include @decorator lines at the same indent
            while start > 0:
                prev_nl = content.rfind('\n', 0, start - 1)
                prev_start = prev_nl + 1 if prev_nl >= 0 else 0
                prev_line = content[prev_start:start]
                stripped = prev_line.lstrip()
                if len(prev_line) - len(stripped) == indent and stripped.startswith('@'):
                    start = prev_start
                else:
                    break
            # Find end: next non-blank line at same or lower indent after body
            rest = content[m.end():]
            end = len(content)
            seen_body = False
            pos = m.end()
            for line in rest.split('\n'):
                if line.strip():
                    line_indent = len(line) - len(line.lstrip())
                    if seen_body and line_indent <= indent:
                        end = pos
                        break
                    seen_body = True
                pos += len(line) + 1
            return start, end
    return None


# ---------------------------------------------------------------------------
# _merge_delta
# ---------------------------------------------------------------------------

def _merge_delta(original_body: str, delta: str) -> str:
    """Merge compact delta with `... existing code ...` markers into original_body.

    Each marker is replaced with original lines bracketed by the surrounding
    context anchors in the delta. Falls back to full replacement if no markers.

    Chitta advantage over FastEdit: purely deterministic, no model, works in-process.
    """
    orig_lines = original_body.splitlines(keepends=True)
    delta_lines = delta.splitlines(keepends=True)

    marker_positions = [i for i, ln in enumerate(delta_lines) if ln.strip() in _DELTA_MARKERS]
    if not marker_positions:
        return delta  # No markers — full replacement (backward compat)

    result: list[str] = []
    orig_cursor = 0
    delta_cursor = 0

    for mi in marker_positions:
        result.extend(delta_lines[delta_cursor:mi])

        # Locate orig_start via longest-common-prefix match:
        # Walk delta lines before the marker and advance orig_cursor
        # for each line that matches the original. This handles the case
        # where the pre-context introduces new constructs (try:, if ...) not
        # yet in the original — we skip past lines that DO match (e.g. the
        # function signature) and set orig_start to just after them.
        before_delta = delta_lines[delta_cursor:mi]
        orig_start = orig_cursor
        for dl in before_delta:
            if orig_start < len(orig_lines) and orig_lines[orig_start].strip() == dl.strip():
                orig_start += 1

        # Post-anchor: first non-empty delta line after marker (before next marker)
        next_mi = (marker_positions[marker_positions.index(mi) + 1]
                   if mi != marker_positions[-1] else len(delta_lines))
        post_anchor = None
        for j in range(mi + 1, next_mi):
            s = delta_lines[j].strip()
            if s:
                post_anchor = s
                break

        # Locate orig_end: original line matching post_anchor
        orig_end = len(orig_lines)
        post_found = False
        if post_anchor:
            for j in range(orig_start, len(orig_lines)):
                if orig_lines[j].strip() == post_anchor:
                    orig_end = j
                    post_found = True
                    break

        if not post_found:
            if mi != marker_positions[-1]:
                # A non-final marker with an unmatchable post-anchor would absorb
                # the entire remaining body, leaving later markers nothing to
                # preserve and duplicating code. Refuse instead of corrupting.
                raise ValueError(
                    f"delta merge: context line {post_anchor!r} after a "
                    f"`... existing code ...` marker not found in the original — "
                    f"provide the full body or fix the anchor"
                )
            # post_anchor missing from original (new code added after marker, or
            # brace-language closing delimiter): trim original tail lines that
            # also appear in the delta suffix to avoid duplicating braces/dedents.
            delta_suffix = [ln.strip() for ln in delta_lines[mi + 1:next_mi] if ln.strip()]
            while orig_end > orig_start and delta_suffix:
                if orig_lines[orig_end - 1].strip() == delta_suffix[-1]:
                    orig_end -= 1
                    delta_suffix.pop()
                else:
                    break

        # Reindent preserved lines to match marker indentation
        marker_indent = len(delta_lines[mi]) - len(delta_lines[mi].lstrip())
        preserved = list(orig_lines[orig_start:orig_end])
        if preserved:
            non_empty = [ln for ln in preserved if ln.strip()]
            if non_empty:
                orig_indent = len(non_empty[0]) - len(non_empty[0].lstrip())
                diff = marker_indent - orig_indent
                if diff != 0:
                    adjusted = []
                    for ln in preserved:
                        if ln.strip():
                            cur = len(ln) - len(ln.lstrip())
                            adjusted.append(' ' * max(0, cur + diff) + ln.lstrip())
                        else:
                            adjusted.append(ln)
                    preserved = adjusted

        result.extend(preserved)
        orig_cursor = orig_end
        delta_cursor = mi + 1

    result.extend(delta_lines[delta_cursor:])
    return ''.join(result)


# ---------------------------------------------------------------------------
# _apply_symbol_patch
# ---------------------------------------------------------------------------

def _apply_symbol_patch(filepath: str, symbol: str, new_body: str) -> str:
    """Replace a named function/class/method. No old_str needed — finds by name.

    new_body may contain `# ... existing code ...` (or `// ...`) markers.
    Markers are replaced with the original lines bracketed by surrounding context.
    """
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    blocked = _reject_sensitive_path(p)
    if blocked:
        return f"Error: {blocked}"

    s = _srv()
    outline_before = s._read_outline(str(p))

    with _path_write_lock(p):
        try:
            content = p.read_text(encoding="utf-8")
            pre_hash = _content_hash(content)
        except OSError as e:
            return f"Error reading {p.name}: {e}"

        ext = p.suffix.lower()

        # Prefer chitta tree-sitter index; fall back to regex
        ts_loc = s.SoulClient.find_symbol_location(str(p), symbol)
        if ts_loc is not None:
            ls, le = ts_loc
            lines = content.splitlines(keepends=True)
            # Clamp daemon-reported line numbers — a stale index can point past EOF.
            ls = max(1, min(ls, len(lines)))
            le = min(le, len(lines))
            start = sum(len(lines[i]) for i in range(ls - 1))
            end = min(sum(len(lines[i]) for i in range(le)), len(content))
            # Snap start to the beginning of its line — tree-sitter may point to
            # the `fn` keyword, leaving `pub ` / `pub(crate) ` in content[:start].
            line_begin = content.rfind('\n', 0, start)
            start = line_begin + 1 if line_begin >= 0 else 0
            # Walk back to include attribute (#[...], @decorator) and doc lines above symbol.
            while start > 0:
                prev_nl = content.rfind('\n', 0, start - 1)
                prev_line = content[prev_nl + 1 if prev_nl >= 0 else 0:start].strip()
                if (prev_line.startswith('#[') or prev_line.startswith('@')
                        or prev_line.startswith('///') or prev_line.startswith('/**')
                        or prev_line.startswith('*')):
                    start = prev_nl + 1 if prev_nl >= 0 else 0
                else:
                    break
            line_num = content[:start].count('\n') + 1
        else:
            if ext not in (".py", ".pyx"):
                return (
                    f"Error: symbol '{symbol}' not found via tree-sitter in {p.name} — "
                    f"chitta daemon unavailable for {ext or 'unknown'} files. "
                    f"Start chittad or use file_patch for exact-string replacement."
                )
            result = _find_symbol_range(content, symbol, ext)
            if result is None:
                return f"Error: symbol '{symbol}' not found in {p.name}"
            start, end = result
            line_num = content[:start].count('\n') + 1
        old_lines = content[start:end].count('\n') + 1

        original_body = content[start:end]
        try:
            merged = _merge_delta(original_body, new_body)
        except ValueError as e:
            return f"Error: {e}"
        body = merged if merged.endswith('\n') else merged + '\n'
        new_lines = body.count('\n')
        delta = new_lines - old_lines

        try:
            if _content_hash(p.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {p.name} changed on disk since read — retry"
            _atomic_write_text(p, content[:start] + body + content[end:])
        except OSError as e:
            return f"Error writing {p.name}: {e}"

    used_delta = any(ln.strip() in _DELTA_MARKERS for ln in new_body.splitlines())
    mode = " [compact-delta]" if used_delta else ""
    sign = "+" if delta >= 0 else ""
    s._post_write_refresh(p)
    try:
        s._cache_put(p, symbol, body, line_num, line_num + new_lines - 1)
    except Exception:
        pass
    s.SoulClient.remember(
        f"[edit] {p.name}::{symbol} patched{mode} @ L{line_num} (+{new_lines}/-{old_lines})",
        kind="episode", tags="file-edit,symbol-patch", confidence=0.7,
    )
    msg = f"✓ {p.name}::{symbol} patched{mode} @ L{line_num} (+{new_lines}/-{old_lines} lines, net {sign}{delta})"
    msg += s._outline_diff(outline_before, s._read_outline(str(p)))
    msg += s._run_lint(str(p))
    return msg


# ---------------------------------------------------------------------------
# _locate_symbol
# ---------------------------------------------------------------------------

def _locate_symbol(p: Path, symbol: str, content: str):
    """Shared symbol lookup: tree-sitter first, Python-indent fallback for .py/.pyx only.

    Returns (start, end, line_num) on success, None on Python miss, or an error
    string when tree-sitter is unavailable for a non-Python file. Callers must
    check isinstance(result, str) before unpacking.
    """
    s = _srv()
    ts_loc = s.SoulClient.find_symbol_location(str(p), symbol)
    if ts_loc is not None:
        ls, le = ts_loc
        lines = content.splitlines(keepends=True)
        # Clamp daemon-reported line numbers — a stale index can point past EOF.
        ls = max(1, min(ls, len(lines)))
        le = min(le, len(lines))
        start = sum(len(lines[i]) for i in range(ls - 1))
        end = min(sum(len(lines[i]) for i in range(le)), len(content))
        return start, end, ls
    ext = p.suffix.lower()
    if ext not in (".py", ".pyx"):
        return (
            f"Error: symbol '{symbol}' not found via tree-sitter in {p.name} — "
            f"chitta daemon unavailable for {ext or 'unknown'} files. "
            f"Start chittad or use file_patch for exact-string replacement."
        )
    r = _find_symbol_range(content, symbol, ext)
    if r is None:
        return None
    sv, e = r
    return sv, e, content[:sv].count("\n") + 1


# ---------------------------------------------------------------------------
# _apply_symbol_edit
# ---------------------------------------------------------------------------

def _apply_symbol_edit(filepath: str, symbol: str, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str inside a named symbol's body.

    Uniqueness is scoped to the symbol — old_str must match exactly once
    *within the symbol body*, not within the whole file. This makes
    old_str short and stable (no need to pad for file-wide uniqueness).
    Fails closed on 0 or >1 matches in-scope.
    """
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    blocked = _reject_sensitive_path(p)
    if blocked:
        return f"Error: {blocked}"
    if not old_str:
        return "Error: old_str is empty"

    s = _srv()
    outline_before = s._read_outline(str(p))

    with _path_write_lock(p):
        try:
            content = p.read_text(encoding="utf-8")
            pre_hash = _content_hash(content)
        except OSError as e:
            return f"Error reading {p.name}: {e}"

        loc = _locate_symbol(p, symbol, content)
        if loc is None:
            return f"Error: symbol '{symbol}' not found in {p.name}"
        if isinstance(loc, str):
            return loc
        sym_start, sym_end, sym_line = loc
        body = content[sym_start:sym_end]
        count = body.count(old_str)
        if count == 0:
            return f"Error: old_str not found inside {symbol} (in {p.name})"
        if count > 1:
            return f"Error: old_str matches {count} locations inside {symbol} — make it more specific"

        local_idx = body.index(old_str)
        abs_idx = sym_start + local_idx
        line_num = content[:abs_idx].count("\n") + 1
        old_lines = old_str.count("\n") + 1
        new_lines = new_str.count("\n") + (1 if new_str else 0)
        delta = new_lines - old_lines

        new_body_content = body[:local_idx] + new_str + body[local_idx + len(old_str):]
        new_content = content[:sym_start] + new_body_content + content[sym_end:]

        try:
            if _content_hash(p.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {p.name} changed on disk since read — retry"
            _atomic_write_text(p, new_content)
        except OSError as e:
            return f"Error writing {p.name}: {e}"

    s._post_write_refresh(p)
    try:
        s._cache_put(p, symbol, new_body_content, sym_line,
                     sym_line + new_body_content.count("\n"))
    except Exception:
        pass
    s.SoulClient.remember(
        f"[edit] {p.name}::{symbol} edited @ L{line_num} (+{new_lines}/-{old_lines})",
        kind="episode", tags="file-edit,symbol-edit", confidence=0.7,
    )
    sign = "+" if delta >= 0 else ""
    msg = f"✓ {p.name}::{symbol} edited @ L{line_num} (+{new_lines}/-{old_lines} lines, net {sign}{delta})"
    msg += s._outline_diff(outline_before, s._read_outline(str(p)))
    msg += s._run_lint(str(p))
    return msg


# ---------------------------------------------------------------------------
# _apply_symbol_delete
# ---------------------------------------------------------------------------

def _apply_symbol_delete(filepath: str, symbol: str) -> str:
    """Delete a named function/class/method by symbol name.

    Uses tree-sitter (via chitta) to locate the symbol, then splices its
    range plus up to one trailing blank line. No old_str required.
    """
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    blocked = _reject_sensitive_path(p)
    if blocked:
        return f"Error: {blocked}"

    s = _srv()
    outline_before = s._read_outline(str(p))

    with _path_write_lock(p):
        try:
            content = p.read_text(encoding="utf-8")
            pre_hash = _content_hash(content)
        except OSError as e:
            return f"Error reading {p.name}: {e}"

        loc = _locate_symbol(p, symbol, content)
        if loc is None:
            return f"Error: symbol '{symbol}' not found in {p.name}"
        if isinstance(loc, str):
            return loc
        start, end, line_num = loc

        # Swallow one trailing blank line so the file doesn't grow orphan gaps
        tail_end = end
        if tail_end < len(content) and content[tail_end] == "\n":
            nxt = content.find("\n", tail_end + 1)
            line = content[tail_end + 1:nxt if nxt >= 0 else len(content)]
            if line.strip() == "":
                tail_end = (nxt + 1) if nxt >= 0 else len(content)

        removed = content[start:tail_end].count("\n")

        try:
            if _content_hash(p.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {p.name} changed on disk since read — retry"
            _atomic_write_text(p, content[:start] + content[tail_end:])
        except OSError as e:
            return f"Error writing {p.name}: {e}"

    s._post_write_refresh(p)
    try:
        s._cache_pop(p, symbol)
    except Exception:
        pass
    s.SoulClient.remember(
        f"[edit] {p.name}::{symbol} deleted @ L{line_num} (-{removed} lines)",
        kind="episode", tags="file-edit,symbol-delete", confidence=0.7,
    )
    msg = f"✓ {p.name}::{symbol} deleted @ L{line_num} (-{removed} lines)"
    msg += s._outline_diff(outline_before, s._read_outline(str(p)))
    msg += s._run_lint(str(p))
    return msg


# ---------------------------------------------------------------------------
# _apply_symbol_rename
# ---------------------------------------------------------------------------

def _apply_symbol_rename(filepath: str, old_name: str, new_name: str) -> str:
    """Rename every occurrence of an identifier in a single file.

    Uses a word-boundary regex — won't touch substrings of larger
    identifiers, won't edit string literals if they happen to embed the
    name (simple heuristic: skip lines that are pure comments/strings is
    out of scope; callers should review diff).

    For cross-file rename use symbol_callers + batch rename (TODO).
    """
    if not old_name or not new_name or old_name == new_name:
        return f"Error: invalid rename {old_name!r} → {new_name!r}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", new_name):
        return f"Error: {new_name!r} is not a valid identifier"

    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    blocked = _reject_sensitive_path(p)
    if blocked:
        return f"Error: {blocked}"

    s = _srv()

    with _path_write_lock(p):
        try:
            content = p.read_text(encoding="utf-8")
            pre_hash = _content_hash(content)
        except OSError as e:
            return f"Error reading {p.name}: {e}"

        pattern = re.compile(_IDENT_RE_TMPL % re.escape(old_name))
        new_content, n = pattern.subn(new_name, content)
        if n == 0:
            return f"Error: identifier '{old_name}' not found in {p.name}"

        try:
            if _content_hash(p.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {p.name} changed on disk since read — retry"
            _atomic_write_text(p, new_content)
        except OSError as e:
            return f"Error writing {p.name}: {e}"

    s._post_write_refresh(p)
    try:
        s._cache_pop(p, old_name)
        s._cache_pop(p, new_name)
    except Exception:
        pass
    s.SoulClient.remember(
        f"[edit] {p.name} rename {old_name}→{new_name} ({n} sites)",
        kind="episode", tags="file-edit,symbol-rename", confidence=0.7,
    )
    return f"✓ {p.name}: renamed {old_name} → {new_name} at {n} site(s)"


# ---------------------------------------------------------------------------
# _apply_symbol_rename_project
# ---------------------------------------------------------------------------

def _apply_symbol_rename_project(filepath: str, old_name: str, new_name: str) -> str:
    """Rename an identifier across all project files (git-repo-aware).

    Discovers all files containing old_name via grep, snapshots content hashes,
    validates no concurrent edits, then writes atomically per file.
    """
    import re as _re
    import subprocess as _sp
    import shutil as _sh
    if not old_name or not new_name or old_name == new_name:
        return f"Error: invalid rename {old_name!r} → {new_name!r}"
    if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", new_name):
        return f"Error: {new_name!r} is not a valid identifier"

    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"

    try:
        root_bytes = _sp.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(p.parent), stderr=_sp.DEVNULL,
        )
        root = Path(root_bytes.decode().strip())
    except Exception:
        root = p.parent

    grep = _sh.which("grep") or "grep"
    try:
        raw = _sp.check_output(
            [grep, "-rl",
             "--include=*.py", "--include=*.pyx",
             "--include=*.ts", "--include=*.js",
             "--include=*.go", "--include=*.rs",
             "--include=*.c", "--include=*.cpp",
             "--include=*.h", "--include=*.hpp",
             old_name, str(root)],
            stderr=_sp.DEVNULL, timeout=15,
        ).decode(errors="replace")
        candidate_files = [Path(ln.strip()) for ln in raw.splitlines() if ln.strip()]
    except _sp.CalledProcessError:
        candidate_files = []
    except Exception as exc:
        return f"Error scanning repo: {exc}"

    if p not in candidate_files:
        candidate_files.insert(0, p)

    pattern = _re.compile(_IDENT_RE_TMPL % _re.escape(old_name))

    snapshots: list[tuple[Path, str, str]] = []
    for fp in candidate_files:
        try:
            content = fp.read_text(encoding="utf-8")
        except OSError:
            continue
        if not pattern.search(content):
            continue
        snapshots.append((fp, content, _content_hash(content)))

    if not snapshots:
        return f"Error: identifier '{old_name}' not found in any project file"

    for fp, content, pre_hash in snapshots:
        blocked = _reject_sensitive_path(fp)
        if blocked:
            return f"Error: {blocked}"
        try:
            if _content_hash(fp.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {fp.name} changed on disk during scan — retry"
        except OSError as e:
            return f"Error re-reading {fp.name}: {e}"

    s = _srv()
    results = []
    for fp, content, pre_hash in snapshots:
        new_content, n = pattern.subn(new_name, content)
        with _path_write_lock(fp):
            try:
                if _content_hash(fp.read_text(encoding="utf-8")) != pre_hash:
                    return (
                        f"Error: {fp.name} changed on disk — partial rename applied "
                        f"to {len(results)} file(s)"
                    )
                _atomic_write_text(fp, new_content)
            except OSError as e:
                return f"Error writing {fp.name}: {e}"
        s._post_write_refresh(fp)
        s._cache_pop_file(fp)
        results.append(f"{fp.name} ({n} site{'s' if n != 1 else ''})")

    s.SoulClient.remember(
        f"[edit] project rename {old_name}→{new_name} in {len(results)} file(s): "
        + ", ".join(r.split()[0] for r in results[:5]),
        kind="episode", tags="file-edit,symbol-rename,project-rename", confidence=0.7,
    )
    return (
        f"✓ Renamed {old_name} → {new_name} in {len(results)} file(s):\n"
        + "\n".join(f"  {r}" for r in results)
    )


# ---------------------------------------------------------------------------
# _apply_symbol_move
# ---------------------------------------------------------------------------

def _apply_symbol_move(filepath: str, symbol: str, dest_filepath: str) -> str:
    """Move a named symbol from one file to another.

    Extracts the symbol range from `filepath`, appends it to
    `dest_filepath` (creating the file if needed), and removes the
    original. Atomic per-file writes with mtime guards on both sides.
    """
    src = Path(filepath).expanduser().resolve()
    dst = Path(dest_filepath).expanduser().resolve()
    if not src.is_file():
        return f"Error: source file not found: {filepath}"
    for bad in (_reject_sensitive_path(src), _reject_sensitive_path(dst)):
        if bad:
            return f"Error: {bad}"
    if src == dst:
        return "Error: source and destination are the same file"

    with _path_write_lock(src):
        try:
            src_content = src.read_text(encoding="utf-8")
            src_pre_hash = _content_hash(src_content)
        except OSError as e:
            return f"Error reading {src.name}: {e}"

        loc = _locate_symbol(src, symbol, src_content)
        if loc is None:
            return f"Error: symbol '{symbol}' not found in {src.name}"
        if isinstance(loc, str):
            return loc
        start, end, line_num = loc
        block = src_content[start:end]
        if not block.endswith("\n"):
            block += "\n"

        # Swallow one trailing blank line on the cut so we don't leave a double-blank
        cut_end = end
        if cut_end < len(src_content) and src_content[cut_end] == "\n":
            nxt = src_content.find("\n", cut_end + 1)
            line = src_content[cut_end + 1:nxt if nxt >= 0 else len(src_content)]
            if line.strip() == "":
                cut_end = (nxt + 1) if nxt >= 0 else len(src_content)

        with _path_write_lock(dst):
            try:
                dst_content = dst.read_text(encoding="utf-8") if dst.exists() else ""
                dst_pre_hash = _content_hash(dst_content)
            except OSError as e:
                return f"Error reading {dst.name}: {e}"

            separator = "" if (not dst_content or dst_content.endswith("\n\n")) else (
                "\n" if dst_content.endswith("\n") else "\n\n"
            )
            new_dst = dst_content + separator + block

            try:
                if _content_hash(src.read_text(encoding="utf-8")) != src_pre_hash:
                    return f"Error: {src.name} changed on disk since read — retry"
                if dst.exists() and _content_hash(dst.read_text(encoding="utf-8")) != dst_pre_hash:
                    return f"Error: {dst.name} changed on disk since read — retry"
                _atomic_write_text(dst, new_dst)
                _atomic_write_text(src, src_content[:start] + src_content[cut_end:])
            except OSError as e:
                return f"Error writing: {e}"

    moved_lines = block.count("\n")
    s = _srv()
    s._post_write_refresh([src, dst])
    try:
        s._cache_pop(src, symbol)
        s._cache_pop(dst, symbol)
    except Exception:
        pass
    s.SoulClient.remember(
        f"[edit] moved {symbol}: {src.name}→{dst.name} ({moved_lines} lines)",
        kind="episode", tags="file-edit,symbol-move", confidence=0.7,
    )
    return f"✓ moved {symbol} from {src.name} (L{line_num}) → {dst.name} ({moved_lines} lines)"


# ---------------------------------------------------------------------------
# _apply_symbol_insert_child
# ---------------------------------------------------------------------------

def _apply_symbol_insert_child(
    filepath: str, parent: str, position: str, new_body: str,
) -> str:
    """Insert a block inside a parent symbol at a named position.

    position one of:
      start, end, before_return, after_last_import, after_docstring,
      before:<child>, after:<child>

    new_body is inserted with the parent's body indent level. Parent
    `"__module__"` targets the top-level scope (for after_last_import,
    after_docstring at module).
    """
    p = Path(filepath).expanduser().resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    blocked = _reject_sensitive_path(p)
    if blocked:
        return f"Error: {blocked}"

    with _path_write_lock(p):
        try:
            content = p.read_text(encoding="utf-8")
            pre_hash = _content_hash(content)
        except OSError as e:
            return f"Error reading {p.name}: {e}"

        # Determine parent range
        if parent == "__module__":
            start, end = 0, len(content)
            body_indent = ""
            header_end = 0
        else:
            loc = _locate_symbol(p, parent, content)
            if loc is None:
                return f"Error: parent '{parent}' not found in {p.name}"
            if isinstance(loc, str):
                return loc
            start, end, _ = loc
            block = content[start:end]
            # body indent = indent of first non-empty line after header
            m = re.match(r"^[^\n]*\n([ \t]*)\S", block)
            body_indent = m.group(1) if m else "    "
            # header_end = position after the `def/class foo(...):` line
            header_match = re.search(r":\s*\n", block)
            header_end = start + (header_match.end() if header_match else 0)

        # Reindent new_body to parent's body indent
        body_lines = new_body.splitlines()
        # Strip common leading indent so we can re-apply
        nonempty = [ln for ln in body_lines if ln.strip()]
        common = min((len(ln) - len(ln.lstrip()) for ln in nonempty), default=0)
        reindented = "\n".join(
            (body_indent + ln[common:]) if ln.strip() else ""
            for ln in body_lines
        )
        if not reindented.endswith("\n"):
            reindented += "\n"

        # Resolve insertion offset
        insert_at = None
        if position == "start":
            insert_at = header_end
        elif position == "end":
            insert_at = end
            # Back off any trailing whitespace so we land flush with last stmt
            while insert_at > header_end and content[insert_at - 1] in " \t":
                insert_at -= 1
            if insert_at > 0 and content[insert_at - 1] != "\n":
                reindented = "\n" + reindented
        elif position == "after_docstring":
            # Search for a triple-quoted string right after header_end
            rest = content[header_end:end]
            ds = re.match(r"\s*([\"']{3}).*?\1\s*\n", rest, re.S)
            if not ds:
                return f"Error: no docstring found in {parent}"
            insert_at = header_end + ds.end()
        elif position == "after_last_import":
            rest = content[start:end]
            last = None
            for m in re.finditer(r"^[ \t]*(?:from |import )[^\n]*\n", rest, re.M):
                last = m
            if last is None:
                return f"Error: no imports found in {parent}"
            insert_at = start + last.end()
        elif position == "before_return":
            rest = content[header_end:end]
            last = None
            for m in re.finditer(r"^[ \t]+return\b[^\n]*\n", rest, re.M):
                last = m
            if last is None:
                return f"Error: no return statement in {parent}"
            insert_at = header_end + last.start()
        elif position.startswith("before:") or position.startswith("after:"):
            mode, _, child = position.partition(":")
            if not child:
                return f"Error: position '{position}' needs a child name"
            child_re = re.compile(
                rf"^([ \t]*)(?:async\s+)?(?:def|class)\s+{re.escape(child)}\b", re.M,
            )
            m = child_re.search(content, start, end)
            if not m:
                return f"Error: child '{child}' not found in {parent}"
            if mode == "before":
                insert_at = m.start()
            else:
                # after = end of child's block
                child_loc = _locate_symbol(p, child, content)
                if child_loc is None or isinstance(child_loc, str):
                    return f"Error: cannot locate end of child '{child}'"
                insert_at = child_loc[1]
                if insert_at > 0 and content[insert_at - 1] != "\n":
                    reindented = "\n" + reindented
        else:
            return (
                f"Error: unknown position '{position}'. "
                f"Valid: {', '.join(_INSERT_POSITIONS)}, before:<child>, after:<child>"
            )

        new_content = content[:insert_at] + reindented + content[insert_at:]
        inserted_lines = reindented.count("\n")
        line_num = content[:insert_at].count("\n") + 1

        try:
            if _content_hash(p.read_text(encoding="utf-8")) != pre_hash:
                return f"Error: {p.name} changed on disk since read — retry"
            _atomic_write_text(p, new_content)
        except OSError as e:
            return f"Error writing {p.name}: {e}"

    s = _srv()
    s._post_write_refresh(p)
    try:
        s._cache_pop(p, parent)
    except Exception:
        pass
    s.SoulClient.remember(
        f"[edit] {p.name}::{parent} +child @ {position} L{line_num} (+{inserted_lines} lines)",
        kind="episode", tags="file-edit,symbol-insert", confidence=0.7,
    )
    return f"✓ {p.name}::{parent} inserted {inserted_lines} lines @ {position} (L{line_num})"
