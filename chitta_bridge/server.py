#!/usr/bin/env python3
"""
OpenCode Bridge - MCP server for continuous OpenCode and Codex sessions.

Features:
- Continuous discussion sessions with conversation history
- Access to OpenCode models (GPT-5, Claude, Gemini, etc.)
- Access to Codex CLI (OpenAI's agentic coding assistant)
- Agent support (plan, build, explore, general)
- Session continuation
- File attachment for code review

Configuration:
- OPENCODE_MODEL: Default model for OpenCode
- OPENCODE_AGENT: Default agent (plan, build, explore, general)
- CODEX_MODEL: Default model for Codex (e.g., o3, gpt-4.1)
- ~/.chitta-bridge/config.json: Persistent config
"""

import os
import re
import sys
import json
import asyncio
import shutil
import socket
import tempfile
import glob as _glob
import html as _html
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict, fields as dc_fields

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability

from chitta_bridge import __version__


_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent path traversal."""
    if Path(session_id).name != session_id:
        raise ValueError("Invalid session ID: path separators not allowed")
    if not _SAFE_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session ID: must be alphanumeric, hyphens, underscores only")
    return session_id


# File size thresholds
SMALL_FILE = 500        # lines
MEDIUM_FILE = 1500      # lines
LARGE_FILE = 5000       # lines

# Chunked processing thresholds
CHUNK_THRESHOLD = 2000   # lines — files above this get chunked
CHUNK_SIZE = 800         # lines per chunk
CHUNK_OVERLAP = 20       # overlap between adjacent chunks
MAX_PARALLEL_CHUNKS = 6  # concurrency limit
MAX_TOTAL_CHUNKS = 20    # safety cap

# Language detection by extension
LANG_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript/React",
    ".jsx": "JavaScript/React", ".go": "Go", ".rs": "Rust", ".java": "Java",
    ".c": "C", ".cpp": "C++", ".h": "C/C++ Header", ".hpp": "C++ Header",
    ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".kt": "Kotlin", ".scala": "Scala", ".sh": "Shell", ".bash": "Bash",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".yaml": "YAML", ".yml": "YAML", ".json": "JSON", ".toml": "TOML",
    ".xml": "XML", ".md": "Markdown", ".r": "R", ".lua": "Lua",
    ".zig": "Zig", ".nim": "Nim", ".ex": "Elixir", ".erl": "Erlang",
    ".clj": "Clojure", ".hs": "Haskell", ".ml": "OCaml", ".vue": "Vue",
    ".svelte": "Svelte", ".dart": "Dart", ".proto": "Protocol Buffers",
}


_file_info_cache: dict[str, dict] = {}

# OpenCode snapshot directory (issue #6845: tmp_* files can grow unbounded)
_OPENCODE_SNAPSHOT_PACK_DIR = Path.home() / ".local" / "share" / "opencode" / "snapshot" / "global" / "objects" / "pack"

# All tmp_* prefixes created by OpenCode's git pack operations
_SNAPSHOT_TMP_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_mtimes_", "tmp_rev_")


def cleanup_opencode_snapshot() -> str:
    """Remove stale tmp_* files from the OpenCode snapshot pack directory.

    These files are created during git pack operations but never cleaned up
    when OpenCode crashes or is force-killed, causing unbounded disk growth
    (see: https://github.com/anomalyco/opencode/issues/6845).
    """
    pack_dir = _OPENCODE_SNAPSHOT_PACK_DIR
    if not pack_dir.exists():
        return "Snapshot pack directory does not exist — nothing to clean."

    removed = []
    errors = []
    for f in pack_dir.iterdir():
        if not any(f.name.startswith(p) for p in _SNAPSHOT_TMP_PREFIXES):
            continue
        try:
            size = f.stat().st_size
            f.unlink()
            removed.append(f"{f.name} ({size / 1024:.0f} KB)")
        except OSError as e:
            errors.append(f"{f.name}: {e}")

    if not removed and not errors:
        return "No stale tmp_* files found."

    lines = []
    if removed:
        lines.append(f"Removed {len(removed)} stale file(s):")
        lines.extend(f"  - {r}" for r in removed)
    if errors:
        lines.append(f"Failed to remove {len(errors)} file(s):")
        lines.extend(f"  - {e}" for e in errors)
    return "\n".join(lines)

MAX_READ_SIZE = 10 * 1024 * 1024  # 10MB - above this, estimate lines from size


def _apply_file_patch(filepath: str, old_str: str, new_str: str) -> str:
    """Apply a search-replace patch. Returns compact diff summary on success."""
    p = Path(filepath).resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    try:
        content = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading {p.name}: {e}"

    count = content.count(old_str)
    if count == 0:
        preview = old_str[:80].replace('\n', '↵')
        return f"Error: old_str not found in {p.name}\nSearched for: {preview!r}"
    if count > 1:
        return f"Error: old_str matches {count} locations in {p.name} — make it more specific"

    line_num = content[:content.index(old_str)].count('\n') + 1
    old_lines = old_str.count('\n') + 1
    new_lines = new_str.count('\n') + (1 if new_str else 0)
    delta = new_lines - old_lines

    try:
        p.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
    except OSError as e:
        return f"Error writing {p.name}: {e}"

    sign = "+" if delta >= 0 else ""
    return f"✓ {p.name} patched @ L{line_num} (+{new_lines}/-{old_lines} lines, net {sign}{delta})"


def _find_symbol_range(content: str, symbol: str, ext: str):
    """Return (start, end) byte range of a named symbol. Returns None if not found."""
    import re
    if ext in (".py", ".pyx"):
        # Python: indent-based (def/async def/class)
        patterns = [
            rf"^(\s*)(async\s+def\s+{re.escape(symbol)}\s*[\(:])",
            rf"^(\s*)(def\s+{re.escape(symbol)}\s*[\(:])",
            rf"^(\s*)(class\s+{re.escape(symbol)}\s*[\(:])",
        ]
        for pat in patterns:
            for m in re.finditer(pat, content, re.MULTILINE):
                indent = len(m.group(1))
                start = m.start()
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
    else:
        # Brace-based: Rust, C, C++, JS, Go, Java…
        kws = ["fn ", "function ", "async function ", "def ", "class ", "impl ", "struct ", "enum "]
        for kw in kws:
            needle = kw + symbol
            idx = content.find(needle)
            while idx != -1:
                before = content[:idx]
                after = content[idx + len(needle):]
                # Word boundary check
                prev_char = before[-1] if before else ' '
                next_char = after[0] if after else ' '
                if not prev_char.isalnum() and prev_char != '_' and next_char in ('(', '<', '{', ' ', '\n', ':'):
                    brace_pos = content.find('{', idx)
                    if brace_pos == -1:
                        break
                    depth = 0
                    for i, c in enumerate(content[brace_pos:]):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                end = brace_pos + i + 1
                                if end < len(content) and content[end] == '\n':
                                    end += 1
                                return idx, end
                idx = content.find(needle, idx + 1)
    return None


def _apply_symbol_patch(filepath: str, symbol: str, new_body: str) -> str:
    """Replace a named function/class/method. No old_str needed — finds by name."""
    p = Path(filepath).resolve()
    if not p.is_file():
        return f"Error: file not found: {filepath}"
    try:
        content = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading {p.name}: {e}"

    ext = p.suffix.lower()
    result = _find_symbol_range(content, symbol, ext)
    if result is None:
        return f"Error: symbol '{symbol}' not found in {p.name}"

    start, end = result
    line_num = content[:start].count('\n') + 1
    old_lines = content[start:end].count('\n') + 1
    body = new_body if new_body.endswith('\n') else new_body + '\n'
    new_lines = body.count('\n')
    delta = new_lines - old_lines

    try:
        p.write_text(content[:start] + body + content[end:], encoding="utf-8")
    except OSError as e:
        return f"Error writing {p.name}: {e}"

    sign = "+" if delta >= 0 else ""
    return f"✓ {p.name}::{symbol} patched @ L{line_num} (+{new_lines}/-{old_lines} lines, net {sign}{delta})"


def get_file_info(filepath: str) -> dict:
    """Get metadata about a file: size, lines, language, etc. Results are cached per path."""
    filepath = str(Path(filepath).resolve())
    if filepath in _file_info_cache:
        cached = _file_info_cache[filepath]
        try:
            st = Path(filepath).stat()
            if st.st_mtime == cached.get("_mtime") and st.st_size == cached.get("size_bytes"):
                return cached
        except OSError:
            pass
        # Stale — fall through to re-compute

    p = Path(filepath)
    if not p.is_file():
        return {}
    try:
        stat = p.stat()
        ext = p.suffix.lower()

        # Count lines efficiently: stream for large files, estimate for huge ones
        if stat.st_size > MAX_READ_SIZE:
            # Estimate: ~40 bytes per line for code files
            line_count = stat.st_size // 40
        else:
            # Stream line counting without loading full content into memory
            line_count = 0
            with open(p, "r", errors="replace") as f:
                for _ in f:
                    line_count += 1

        result = {
            "path": filepath,
            "name": p.name,
            "size_bytes": stat.st_size,
            "size_human": _human_size(stat.st_size),
            "lines": line_count,
            "language": LANG_MAP.get(ext, ext.lstrip(".").upper() if ext else "Unknown"),
            "ext": ext,
            "category": (
                "small" if line_count <= SMALL_FILE
                else "medium" if line_count <= MEDIUM_FILE
                else "large" if line_count <= LARGE_FILE
                else "very large"
            ),
            "_mtime": stat.st_mtime,
        }
        _file_info_cache[filepath] = result
        return result
    except Exception:
        return {"path": filepath, "name": p.name}


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f}{unit}" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _expand_paths(paths: list[str]) -> list[str]:
    """Expand directories to contained files; keep plain file paths as-is."""
    result: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            result.extend(str(f) for f in sorted(path.rglob("*")) if f.is_file())
        elif path.is_file():
            result.append(str(path))
    return result


def _embed_files_in_prompt(message: str, files: list[str]) -> str:
    """Embed file content inline for backends that don't support --file args."""
    if not files:
        return message
    parts = []
    for f in files:
        p = Path(f)
        if p.is_file():
            try:
                content = p.read_text(errors="replace")
                parts.append(f"### File: {p.name}\n```\n{content}\n```")
            except OSError:
                pass
    if not parts:
        return message
    return "\n\n".join(parts) + "\n\n" + message


def build_file_context(file_paths: list[str]) -> str:
    """Build a context block describing attached files."""
    if not file_paths:
        return ""
    infos = [info for f in file_paths if (info := get_file_info(f))]
    if not infos:
        return ""

    parts = ["## Attached Files\n"]
    for info in infos:
        line = f"- **{info.get('name', '?')}**"
        details = []
        if "language" in info:
            details.append(info["language"])
        if "lines" in info:
            details.append(f"{info['lines']} lines")
        if "size_human" in info:
            details.append(info["size_human"])
        if "category" in info:
            details.append(info["category"])
        if details:
            line += f" ({', '.join(details)})"
        parts.append(line)

    total_lines = sum(i.get("lines", 0) for i in infos)
    if total_lines > LARGE_FILE:
        parts.append(f"\n> Total: {total_lines} lines across {len(infos)} file(s) — this is a large review.")
        parts.append("> Focus on the most critical issues first. Use a structured, section-by-section approach.")

    return "\n".join(parts)


def build_review_prompt(file_infos: list[dict], focus: str) -> str:
    """Build an adaptive review prompt based on file size and type."""
    total_lines = sum(i.get("lines", 0) for i in file_infos)

    # Base review instructions
    prompt_parts = [f"Please review the attached code, focusing on: **{focus}**\n"]

    # Add file context
    if file_infos:
        prompt_parts.append("### Files to review:")
        for info in file_infos:
            prompt_parts.append(f"- {info.get('name', '?')} ({info.get('language', '?')}, {info.get('lines', '?')} lines)")
        prompt_parts.append("")

    # Adapt strategy to file size
    if total_lines > LARGE_FILE:
        prompt_parts.append("""### Review Strategy (Large File)
This is a large codebase review. Use this structured approach:

1. **Architecture Overview**: Describe the overall structure, main components, and data flow
2. **Critical Issues**: Security vulnerabilities, bugs, race conditions, memory leaks
3. **Design Concerns**: Architectural problems, tight coupling, missing abstractions
4. **Code Quality**: Naming, duplication, complexity hotspots (focus on the worst areas)
5. **Key Recommendations**: Top 5 most impactful improvements, prioritized

Do NOT try to comment on every line. Focus on patterns and the most impactful findings.""")
    elif total_lines > MEDIUM_FILE:
        prompt_parts.append("""### Review Strategy (Medium File)
Provide a structured review:

1. **Summary**: What does this code do? Overall assessment
2. **Issues Found**: Bugs, security concerns, edge cases, error handling gaps
3. **Design Feedback**: Structure, patterns, abstractions
4. **Specific Suggestions**: Concrete improvements with code examples where helpful""")
    else:
        prompt_parts.append("""### Review Guidelines
Provide a thorough review covering:
- Correctness and edge cases
- Error handling
- Code clarity and naming
- Any security concerns
- Concrete suggestions for improvement""")

    return "\n".join(prompt_parts)


def build_message_prompt(message: str, file_paths: list[str]) -> str:
    """Build a smart prompt that includes file context and instructions."""
    parts = []

    # Add file context if files are attached
    user_files = [f for f in file_paths if not Path(f).name.startswith("opencode_msg_")]
    if user_files:
        file_context = build_file_context(user_files)
        if file_context:
            parts.append(file_context)
            parts.append("")

        total_lines = sum(get_file_info(f).get("lines", 0) for f in user_files)
        if total_lines > LARGE_FILE:
            parts.append("**Note:** Large file(s) attached. Read through the full content carefully before responding. "
                         "If asked to analyze or review, use a structured section-by-section approach.")
            parts.append("")

    parts.append("## Request")
    parts.append("Respond to the user's request in the attached message file. "
                 "Read all attached files completely before responding.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Companion System — Auto-Framing
# ---------------------------------------------------------------------------


def build_companion_prompt(
    message: str,
    files: Optional[list[str]] = None,
    domain_override: Optional[str] = None,
    is_followup: bool = False,
) -> str:
    """Assemble a companion prompt that auto-detects the domain.

    The LLM identifies the domain and adopts an appropriate expert persona.
    An optional *domain_override* hint biases the framing toward a specific field.
    """
    # Follow-up: lightweight prompt
    if is_followup:
        return "\n".join([
            "## Continuing Our Discussion",
            "",
            message,
            "",
            "Remember: challenge assumptions, consider alternatives, be explicit about trade-offs.",
        ])

    # --- Full initial prompt ---
    parts = []

    # File context
    user_files = [f for f in (files or []) if not Path(f).name.startswith("opencode_msg_")]
    if user_files:
        file_context = build_file_context(user_files)
        if file_context:
            parts.append("## Context")
            parts.append(file_context)
            parts.append("")

    # Domain hint
    domain_hint = ""
    if domain_override:
        domain_hint = (
            f"\n\nNote: the user has indicated this is about **{domain_override}** — "
            "frame your expertise accordingly."
        )

    parts.append("## Discussion Setup")
    parts.append(
        "Determine the **specific domain of expertise** this question belongs to "
        "(e.g., distributed systems, metagenomics, compiler design, quantitative finance, "
        "DevOps, security, database design, or any other field).\n"
        "\n"
        "Then adopt the persona of a **senior practitioner with deep, hands-on "
        "experience** in that domain. You have:\n"
        "- Years of practical experience solving real problems in this field\n"
        "- Deep knowledge of the key frameworks, methods, and trade-offs\n"
        "- Strong opinions loosely held — you recommend but explain why\n"
        "\n"
        "Briefly state what domain you identified and what expert lens you're "
        f"applying (one line at the top is enough).{domain_hint}"
    )
    parts.append("")

    parts.append("## Collaborative Ground Rules")
    parts.append("- Think out loud, share your reasoning step by step")
    parts.append("- Challenge questionable assumptions — including mine")
    parts.append("- Lay out trade-offs explicitly: what we gain, what we lose")
    parts.append("- Name the key analytical frameworks or methods relevant to this domain")
    parts.append("- Propose at least one alternative I haven't considered")
    parts.append("")

    parts.append("## Your Approach")
    parts.append("1. Identify the domain and the core question")
    parts.append("2. Apply domain-specific frameworks and best practices")
    parts.append("3. Analyze trade-offs with concrete reasoning")
    parts.append("4. Provide a clear recommendation")
    parts.append("")

    parts.append("## The Question")
    parts.append(message)
    parts.append("")

    parts.append("## Synthesize")
    parts.append("1. Your recommendation with rationale")
    parts.append("2. Key trade-offs")
    parts.append("3. Risks or blind spots")
    parts.append("4. Open questions worth exploring")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chunked Processing — map-reduce for large files
# ---------------------------------------------------------------------------

# Regex for natural code boundaries (language-agnostic)
_BOUNDARY_RE = re.compile(
    r"^(?:\s*$"
    r"|(?:(?:async )?def |class |(?:export )?(?:default )?function |func |fn |pub fn |impl |module |package )"
    r"|(?:(?:export )?(?:const|let|var|interface|struct|enum|trait) )"
    r"|(?:})\s*$"
    r"|(?://|#|/\*|\*/).{0,80}$"
    r")",
    re.MULTILINE,
)


def chunk_file(
    filepath: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split a file into overlapping chunks with boundary snapping.

    Returns a list of dicts with keys:
        chunk_index, total_chunks, start_line, end_line, content, filepath
    """
    p = Path(filepath)
    try:
        lines = p.read_text(errors="replace").splitlines(keepends=True)
    except Exception:
        return []

    total = len(lines)
    if total == 0:
        return []
    if total <= chunk_size:
        return [{
            "chunk_index": 0,
            "total_chunks": 1,
            "start_line": 1,
            "end_line": total,
            "content": "".join(lines),
            "filepath": str(p),
        }]

    chunks: list[dict] = []
    pos = 0
    while pos < total:
        end = min(pos + chunk_size, total)

        # Snap to a natural boundary within ±50 lines of the cut point
        if end < total:
            best = end
            scan_start = max(end - 50, pos + chunk_size // 2)
            scan_end = min(end + 50, total)
            for i in range(scan_start, scan_end):
                if _BOUNDARY_RE.match(lines[i]):
                    best = i + 1  # include the boundary line in this chunk
                    break
            end = best

        chunk_content = "".join(lines[pos:end])
        chunks.append({
            "chunk_index": len(chunks),
            "total_chunks": -1,  # filled in below
            "start_line": pos + 1,  # 1-indexed
            "end_line": end,
            "content": chunk_content,
            "filepath": str(p),
        })

        # Advance: overlap with previous chunk, but stop if we've reached the end
        if end >= total:
            break
        pos = max(end - overlap, pos + 1)

    # Fill in total_chunks
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


def build_chunk_prompt(
    user_prompt: str,
    chunk_info: dict,
    file_info: dict,
    mode: str = "discuss",
) -> str:
    """Build a focused prompt for analyzing a single file chunk."""
    name = file_info.get("name", Path(chunk_info["filepath"]).name)
    language = file_info.get("language", "Unknown")
    total_lines = file_info.get("lines", "?")
    idx = chunk_info["chunk_index"] + 1
    total = chunk_info["total_chunks"]
    start = chunk_info["start_line"]
    end = chunk_info["end_line"]

    parts = [
        f"You are analyzing **chunk {idx} of {total}** from `{name}` "
        f"({language}, {total_lines} total lines).",
        f"This chunk covers **lines {start}–{end}**.",
        "",
        "## Task",
        user_prompt,
        "",
        "## Instructions",
        "- Focus ONLY on the code in this chunk",
        "- Note any references to code that might exist outside this chunk",
        "- Be concise — your output will be combined with analyses of other chunks",
        "- Include line numbers for any issues found",
    ]

    if mode == "review":
        parts.append("- Categorize findings as: bug, security, design, performance, or style")

    return "\n".join(parts)


def build_synthesis_prompt(
    user_prompt: str,
    chunk_results: list[dict],
    file_infos: list[dict],
    mode: str = "discuss",
) -> str:
    """Build a prompt that merges chunk analyses into one coherent response."""
    file_desc = ", ".join(
        f"`{i.get('name', '?')}` ({i.get('lines', '?')} lines)"
        for i in file_infos
    )
    n = len(chunk_results)

    parts = [
        f"You analyzed a large file in **{n} chunks**. "
        "Synthesize the chunk analyses below into one coherent response.",
        "",
        "## Original Request",
        user_prompt,
        "",
        "## Files Analyzed",
        file_desc,
        "",
        "## Chunk Analyses",
    ]

    for cr in sorted(chunk_results, key=lambda c: c.get("chunk_index", 0)):
        idx = cr.get("chunk_index", 0) + 1
        fp = Path(cr.get("file", "")).name
        response = cr.get("response", "[analysis failed]")
        if cr.get("error"):
            response = f"[analysis failed: {cr['error']}]"
        parts.append(f"\n### Chunk {idx} — `{fp}`")
        parts.append(response)

    parts.extend([
        "",
        "## Instructions",
        "- Combine findings and remove duplicates (chunks overlap slightly)",
        "- Organize by importance, not by chunk order",
        "- Preserve line number references from the original analyses",
        "- Provide an overall assessment at the top",
    ])

    if mode == "review":
        parts.append("- Group findings by category: bugs, security, design, performance, style")

    return "\n".join(parts)


# Default configuration
DEFAULT_MODEL = "openai/gpt-5.3-codex"
DEFAULT_AGENT = "plan"
DEFAULT_VARIANT = "medium"

# Codex defaults
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_SANDBOX = "workspace-write"


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
        config_path.write_text(json.dumps(data, indent=2))


def find_opencode() -> Optional[Path]:
    """Find opencode binary."""
    # Check common locations
    paths = [
        Path.home() / ".opencode" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
        Path("/usr/bin/opencode"),
    ]
    for p in paths:
        if p.exists():
            return p
    # Check PATH
    which = shutil.which("opencode")
    if which:
        return Path(which)
    return None


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


OPENCODE_BIN = find_opencode()
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
        import subprocess
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


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Session:
    """Session for OpenCode backend."""
    id: str
    model: str
    agent: str
    variant: str = DEFAULT_VARIANT
    opencode_session_id: Optional[str] = None
    claude_session_ids: list = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def save(self, path: Path):
        data = {
            "id": self.id,
            "model": self.model,
            "agent": self.agent,
            "variant": self.variant,
            "opencode_session_id": self.opencode_session_id,
            "claude_session_ids": self.claude_session_ids,
            "created": self.created,
            "messages": [asdict(m) for m in self.messages]
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Session":
        data = json.loads(path.read_text())
        session = cls(
            id=data["id"],
            model=data["model"],
            agent=data.get("agent", DEFAULT_AGENT),
            variant=data.get("variant", DEFAULT_VARIANT),
            opencode_session_id=data.get("opencode_session_id"),
            claude_session_ids=data.get("claude_session_ids", []),
            created=data.get("created", datetime.now().isoformat())
        )
        for m in data.get("messages", []):
            session.messages.append(Message(**m))
        return session


@dataclass
class CodexSession:
    """Session for Codex backend."""
    id: str
    model: str
    sandbox: str = DEFAULT_CODEX_SANDBOX
    full_auto: bool = True
    codex_session_id: Optional[str] = None
    working_dir: Optional[str] = None
    claude_session_ids: list = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def save(self, path: Path):
        data = {
            "id": self.id,
            "model": self.model,
            "sandbox": self.sandbox,
            "full_auto": self.full_auto,
            "codex_session_id": self.codex_session_id,
            "working_dir": self.working_dir,
            "claude_session_ids": self.claude_session_ids,
            "created": self.created,
            "messages": [asdict(m) for m in self.messages]
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CodexSession":
        data = json.loads(path.read_text())
        session = cls(
            id=data["id"],
            model=data["model"],
            sandbox=data.get("sandbox", DEFAULT_CODEX_SANDBOX),
            full_auto=data.get("full_auto", True),
            codex_session_id=data.get("codex_session_id"),
            working_dir=data.get("working_dir"),
            claude_session_ids=data.get("claude_session_ids", []),
            created=data.get("created", datetime.now().isoformat())
        )
        for m in data.get("messages", []):
            session.messages.append(Message(**m))
        return session


@dataclass
class CodexJob:
    """Background Codex task with persistent status tracking."""
    id: str
    task: str
    model: str
    working_dir: str
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"  # running | completed | failed | cancelled
    effort: Optional[str] = None
    sandbox: Optional[str] = None
    resume_from: Optional[str] = None
    started: Optional[str] = None
    finished: Optional[str] = None
    result: Optional[str] = None
    codex_session_id: Optional[str] = None

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CodexJob":
        data = json.loads(path.read_text())
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


class OpenCodeBridge:
    def __init__(self):
        self.start_time = datetime.now()
        self.config = Config.load()
        self.sessions: dict[str, Session] = {}
        self.active_session: Optional[str] = None
        self.sessions_dir = Path.home() / ".chitta-bridge" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.available_models: list[str] = []
        self.available_agents: list[str] = []
        self._load_sessions()

    def _load_sessions(self):
        for path in self.sessions_dir.glob("*.json"):
            try:
                session = Session.load(path)
                self.sessions[session.id] = session
            except Exception as e:
                print(f"Warning: skipping corrupted session {path.name}: {e}", file=sys.stderr)

    async def _run_opencode(self, *args, timeout: int = 120, stall_timeout: int = 120) -> tuple[str, int]:
        """Run opencode CLI command with streaming stdout and stall detection.

        timeout: max total seconds before giving up.
        stall_timeout: max seconds of silence (no output) before declaring the model hung.
        """
        global OPENCODE_BIN
        # Lazy retry: if binary wasn't found at startup, try again
        if not OPENCODE_BIN:
            OPENCODE_BIN = find_opencode()
        if not OPENCODE_BIN:
            return "OpenCode not installed. Install from: https://opencode.ai", 1

        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(OPENCODE_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            proc.stdin.close()

            # Drain stderr concurrently so a full stderr pipe never blocks stdout.
            stderr_task = asyncio.ensure_future(proc.stderr.read())

            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + timeout
            first_line = True

            # Read stdout line by line — detect stalls between lines.
            # stall_timeout only applies after the first line; initial response
            # uses the full remaining budget so slow-thinking models aren't killed.
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Timed out after {timeout}s", 1
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=read_timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s", 1
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            out = "".join(stdout_parts).strip()
            if proc.returncode == 0:
                err = _strip_startup_warnings(stderr_raw.decode(errors="replace")).strip()
            else:
                err = stderr_raw.decode(errors="replace").strip()
            output = out if out else err
            if out and err and proc.returncode:
                output = f"{out}\n\nStderr:\n{err}"
            return output, proc.returncode or 0
        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Command timed out after {timeout}s", 1
        except Exception as e:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}", 1

    @staticmethod
    def _parse_opencode_response(output: str) -> tuple[str, Optional[str]]:
        """Parse JSON-lines output from opencode CLI.

        Returns (reply_text, session_id).
        """
        reply_parts: list[str] = []
        session_id: Optional[str] = None
        for line in output.split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                if not session_id and "sessionID" in event:
                    session_id = event["sessionID"]
                if event.get("type") == "text":
                    text = event.get("part", {}).get("text", "")
                    if text:
                        reply_parts.append(text)
            except json.JSONDecodeError:
                continue
        return "".join(reply_parts), session_id

    async def _run_chunk(
        self,
        chunk_info: dict,
        file_info: dict,
        user_prompt: str,
        session: "Session",
        mode: str = "discuss",
    ) -> dict:
        """Process a single file chunk through OpenCode (stateless)."""
        result = {
            "chunk_index": chunk_info["chunk_index"],
            "file": chunk_info["filepath"],
            "response": "",
            "error": None,
        }

        # Write chunk to a temp file preserving the original extension
        ext = Path(chunk_info["filepath"]).suffix or ".txt"
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=ext, delete=False, prefix="opencode_chunk_"
            )
            tmp.write(chunk_info["content"])
            tmp.close()

            prompt = build_chunk_prompt(user_prompt, chunk_info, file_info, mode)

            args = [
                "run", prompt,
                "--model", session.model,
                "--agent", session.agent,
                "--file", tmp.name,
                "--format", "json",
            ]
            if session.variant:
                args.extend(["--variant", session.variant])

            chunk_lines = chunk_info.get("line_count", CHUNK_SIZE)
            stall_timeout = min(300, max(120, chunk_lines // 10))
            output, code = await self._run_opencode(*args, timeout=300, stall_timeout=stall_timeout)

            if code != 0:
                result["error"] = output[:500]
                return result

            reply, _ = self._parse_opencode_response(output)
            result["response"] = reply or "[no response]"

        except Exception as e:
            result["error"] = str(e)
        finally:
            if tmp:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        return result

    async def _run_chunked(
        self,
        user_prompt: str,
        files: list[str],
        session: "Session",
        mode: str = "discuss",
    ) -> str:
        """Map-reduce orchestrator: chunk large files, process in parallel, synthesize."""
        small_files: list[str] = []
        all_chunks: list[tuple[dict, dict]] = []  # (chunk_info, file_info)

        for f in files:
            info = get_file_info(f)
            line_count = info.get("lines", 0)
            if line_count > CHUNK_THRESHOLD:
                chunks = chunk_file(f, CHUNK_SIZE, CHUNK_OVERLAP)
                for c in chunks:
                    all_chunks.append((c, info))
            else:
                small_files.append(f)

        # Safety: if too many chunks, increase chunk size and re-chunk
        if len(all_chunks) > MAX_TOTAL_CHUNKS:
            all_chunks = []
            bigger = CHUNK_SIZE * 2
            for f in files:
                info = get_file_info(f)
                if info.get("lines", 0) > CHUNK_THRESHOLD:
                    chunks = chunk_file(f, bigger, CHUNK_OVERLAP)
                    for c in chunks:
                        all_chunks.append((c, info))
                # small_files already collected above

        if not all_chunks:
            return "No chunks to process."

        # --- Map phase: run chunks in parallel ---
        sem = asyncio.Semaphore(MAX_PARALLEL_CHUNKS)

        async def _limited(chunk_info: dict, file_info: dict) -> dict:
            async with sem:
                return await self._run_chunk(chunk_info, file_info, user_prompt, session, mode)

        tasks = [_limited(ci, fi) for ci, fi in all_chunks]
        chunk_results: list[dict] = await asyncio.gather(*tasks)

        # Check failure rate
        failed = sum(1 for cr in chunk_results if cr.get("error"))
        if failed > len(chunk_results) / 2:
            return (
                f"Chunked analysis failed: {failed}/{len(chunk_results)} chunks errored. "
                "Try with a smaller file or increase the chunk size."
            )

        # --- Reduce phase: synthesize ---
        file_infos = []
        seen_paths: set[str] = set()
        for _, fi in all_chunks:
            fp = fi.get("path", "")
            if fp not in seen_paths:
                seen_paths.add(fp)
                file_infos.append(fi)

        synthesis_prompt = build_synthesis_prompt(user_prompt, chunk_results, file_infos, mode)

        # Attach small files for reference context (not the large ones)
        args = [
            "run", synthesis_prompt,
            "--model", session.model,
            "--agent", session.agent,
            "--format", "json",
        ]
        if session.variant:
            args.extend(["--variant", session.variant])
        for sf in small_files:
            args.extend(["--file", sf])

        # Longer timeout for synthesis
        output, code = await self._run_opencode(*args, timeout=600)

        if code != 0:
            # Fallback: concatenate raw chunk results
            parts = ["*Synthesis failed — showing raw chunk analyses:*\n"]
            for cr in sorted(chunk_results, key=lambda c: c.get("chunk_index", 0)):
                idx = cr.get("chunk_index", 0) + 1
                fp = Path(cr.get("file", "")).name
                parts.append(f"\n### Chunk {idx} — `{fp}`")
                if cr.get("error"):
                    parts.append(f"[error: {cr['error']}]")
                else:
                    parts.append(cr.get("response", "[no response]"))
            return "\n".join(parts)

        reply, _ = self._parse_opencode_response(output)
        return reply or "No response from synthesis."

    async def list_models(self, provider: Optional[str] = None) -> str:
        """List available models from OpenCode."""
        args = ["models"]
        if provider:
            args.append(provider)

        output, code = await self._run_opencode(*args)
        if code != 0:
            return f"Error listing models: {output}"

        self.available_models = [line.strip() for line in output.split("\n") if line.strip()]

        # Group by provider
        providers: dict[str, list[str]] = {}
        for model in self.available_models:
            if "/" in model:
                prov, name = model.split("/", 1)
            else:
                prov, name = "other", model
            providers.setdefault(prov, []).append(name)

        lines = ["Available models:"]
        for prov in sorted(providers.keys()):
            lines.append(f"\n**{prov}:**")
            for name in sorted(providers[prov]):
                full = f"{prov}/{name}"
                lines.append(f"  - {full}")

        return "\n".join(lines)

    async def list_agents(self) -> str:
        """List available agents from OpenCode."""
        output, code = await self._run_opencode("agent", "list")
        if code != 0:
            return f"Error listing agents: {output}"

        # Parse agent names from output
        agents = []
        for line in output.split("\n"):
            line = line.strip()
            if line and "(" in line:
                name = line.split("(")[0].strip()
                agents.append(name)

        self.available_agents = agents
        return "Available agents:\n" + "\n".join(f"  - {a}" for a in agents)

    async def start_session(
        self,
        session_id: str,
        model: Optional[str] = None,
        agent: Optional[str] = None,
        variant: Optional[str] = None
    ) -> str:
        session_id = _sanitize_session_id(session_id)

        if session_id in self.sessions:
            return f"Session '{session_id}' already exists. Use a different ID or end it first."

        # Use config defaults if not specified
        model = model or self.config.model
        agent = agent or self.config.agent
        variant = variant or self.config.variant

        claude_session_id = _get_claude_session_id()

        session = Session(
            id=session_id,
            model=model,
            agent=agent,
            variant=variant,
            claude_session_ids=[claude_session_id] if claude_session_id else []
        )
        self.sessions[session_id] = session
        self.active_session = session_id
        session.save(self.sessions_dir / f"{session_id}.json")

        # Warmup: fire a trivial message so opencode pre-initializes and we capture
        # the session ID. All subsequent calls use --session and skip cold start.
        warmup_args = [
            "run", ".",
            "--model", model,
            "--agent", agent,
            "--format", "json",
        ]
        if variant:
            warmup_args.extend(["--variant", variant])
        warmup_out, _ = await self._run_opencode(*warmup_args, timeout=60, stall_timeout=60)
        _, oc_session_id = self._parse_opencode_response(warmup_out)
        if oc_session_id:
            session.opencode_session_id = oc_session_id
            session.save(self.sessions_dir / f"{session_id}.json")

        result = f"Session '{session_id}' started\n  Model: {model}\n  Agent: {agent}"
        if variant:
            result += f"\n  Variant: {variant}"
        if oc_session_id:
            result += f"\n  OpenCode session: {oc_session_id} (warmed up)"
        if claude_session_id:
            result += f"\n  Claude session: {claude_session_id}"
        return result

    def get_config(self) -> str:
        """Get current configuration."""
        return f"""Current configuration:
  Model: {self.config.model}
  Agent: {self.config.agent}
  Variant: {self.config.variant}

Set via:
  - ~/.chitta-bridge/config.json
  - OPENCODE_MODEL, OPENCODE_AGENT, OPENCODE_VARIANT env vars
  - opencode_configure tool"""

    def set_config(self, model: Optional[str] = None, agent: Optional[str] = None, variant: Optional[str] = None) -> str:
        """Update and persist configuration."""
        changes = []
        if model:
            self.config.model = model
            changes.append(f"model: {model}")
        if agent:
            self.config.agent = agent
            changes.append(f"agent: {agent}")
        if variant:
            self.config.variant = variant
            changes.append(f"variant: {variant}")

        if changes:
            self.config.save()
            return "Configuration updated:\n  " + "\n  ".join(changes)
        return "No changes made."

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[list[str]] = None,
        domain_override: Optional[str] = None,
        _raw: bool = False,
    ) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session. Use opencode_start first."

        session = self.sessions[sid]
        session.add_message("user", message)
        # Save immediately so user messages aren't lost if OpenCode fails
        session.save(self.sessions_dir / f"{sid}.json")

        # Always write message to temp file to avoid shell escaping issues
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, prefix='opencode_msg_'
        )
        temp_file.write(message)
        temp_file.close()
        files = (files or []) + [temp_file.name]

        try:
            # --- Chunking gate: large user files get map-reduce processing ---
            user_files = [f for f in files if not Path(f).name.startswith("opencode_msg_")]
            file_line_counts = [get_file_info(f).get("lines", 0) for f in user_files]
            total_lines = sum(file_line_counts)
            needs_chunking = any(n > CHUNK_THRESHOLD for n in file_line_counts)

            if needs_chunking:
                reply = await self._run_chunked(message, user_files, session, mode="discuss")
                if reply:
                    session.add_message("assistant", reply)
                    session.save(self.sessions_dir / f"{sid}.json")
                return reply or "No response received"

            # --- Normal (non-chunked) path ---

            # Build prompt: companion system unless _raw is set
            if _raw:
                run_prompt = build_message_prompt(message, files)
            else:
                is_followup = len(session.messages) > 1
                run_prompt = build_companion_prompt(
                    message, files, domain_override=domain_override,
                    is_followup=is_followup,
                )

            args = ["run", run_prompt]

            args.extend(["--model", session.model])
            args.extend(["--agent", session.agent])

            # Add variant if specified
            if session.variant:
                args.extend(["--variant", session.variant])

            # Continue session if we have an opencode session ID
            if session.opencode_session_id:
                args.extend(["--session", session.opencode_session_id])

            # Attach files
            if files:
                for f in files:
                    args.extend(["--file", f])

            # Use JSON format to get session ID
            args.extend(["--format", "json"])

            # Scale timeout based on attached file size (total_lines computed above in chunking gate)
            # Base 300s, +60s per 1000 lines above threshold, capped at 900s
            timeout = min(900, 300 + max(0, (total_lines - MEDIUM_FILE) * 60 // 1000))

            # stall_timeout: gpt-5.4/high variant can take 2+ min before first token
            stall_timeout = min(300, max(240, total_lines // 10))
            output, code = await self._run_opencode(*args, timeout=timeout, stall_timeout=stall_timeout)

            if code != 0:
                return f"Error: {output}"

            # Parse JSON events for session ID and text
            reply, new_session_id = self._parse_opencode_response(output)
            if new_session_id and not session.opencode_session_id:
                session.opencode_session_id = new_session_id

            if reply:
                session.add_message("assistant", reply)

            # Save if we got a reply or captured a new session ID
            if reply or session.opencode_session_id:
                session.save(self.sessions_dir / f"{sid}.json")

            return reply or "No response received"
        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

    async def plan(
        self,
        task: str,
        session_id: Optional[str] = None,
        files: Optional[list[str]] = None
    ) -> str:
        """Start a planning discussion using the plan agent."""
        sid = session_id or self.active_session

        # If no active session, create one for planning
        if not sid or sid not in self.sessions:
            sid = f"plan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="plan")

        # Switch to plan agent if not already
        session = self.sessions[sid]
        if session.agent != "plan":
            session.agent = "plan"
            session.save(self.sessions_dir / f"{sid}.json")

        return await self.send_message(task, sid, files)

    async def brainstorm(
        self,
        topic: str,
        session_id: Optional[str] = None
    ) -> str:
        """Open-ended brainstorming discussion — routes through companion system."""
        sid = session_id or self.active_session

        if not sid or sid not in self.sessions:
            sid = f"brainstorm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="build")

        return await self.send_message(f"Let's brainstorm about: {topic}", sid)

    async def review_code(
        self,
        code_or_file: str,
        focus: str = "correctness, efficiency, and potential bugs",
        session_id: Optional[str] = None
    ) -> str:
        """Review code for issues and improvements."""
        sid = session_id or self.active_session

        if not sid or sid not in self.sessions:
            sid = f"review-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="build")

        # Check if it's a file path (could be multiple, comma or space separated)
        files = None
        file_paths = []

        # Try splitting by comma first, then check each part
        candidates = [c.strip() for c in code_or_file.replace(",", " ").split() if c.strip()]
        for candidate in candidates:
            if Path(candidate).is_file():
                file_paths.append(candidate)

        if file_paths:
            files = file_paths
            file_infos = [get_file_info(f) for f in file_paths]
            file_infos = [i for i in file_infos if i]
            total_lines = sum(i.get("lines", 0) for i in file_infos)

            # Chunking gate for large reviews
            if any(i.get("lines", 0) > CHUNK_THRESHOLD for i in file_infos):
                prompt = build_review_prompt(file_infos, focus)
                session = self.sessions[sid]
                session.add_message("user", f"[code review] {focus}")
                session.save(self.sessions_dir / f"{sid}.json")
                reply = await self._run_chunked(prompt, file_paths, session, mode="review")
                if reply:
                    session.add_message("assistant", reply)
                    session.save(self.sessions_dir / f"{sid}.json")
                return reply

            prompt = build_review_prompt(file_infos, focus)

            # Increase timeout for large files
            if total_lines > LARGE_FILE:
                # Use variant=high for large reviews if not already high+
                session = self.sessions[sid]
                if session.variant in ("minimal", "low", "medium"):
                    prompt += "\n\n> *Auto-escalated to thorough review due to file size.*"
        else:
            # Inline code snippet
            prompt = f"""Please review this code, focusing on: **{focus}**

```
{code_or_file}
```

Provide:
- Issues found (bugs, edge cases, security)
- Design feedback
- Concrete improvement suggestions"""

        return await self.send_message(prompt, sid, files, _raw=True)

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No sessions found."

        lines = ["Sessions:"]
        for sid, session in self.sessions.items():
            active = " (active)" if sid == self.active_session else ""
            msg_count = len(session.messages)
            variant_str = f", variant={session.variant}" if session.variant else ""
            cc_ids = f", claude={','.join(session.claude_session_ids)}" if session.claude_session_ids else ""
            lines.append(f"  - {sid}: {session.model} [{session.agent}{variant_str}], {msg_count} messages{cc_ids}{active}")
        return "\n".join(lines)

    def attach_claude_session(self, session_id: str, claude_session_id: str) -> str:
        """Register a Claude Code session ID as using this OpenCode session."""
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "Session not found."
        session = self.sessions[sid]
        if claude_session_id not in session.claude_session_ids:
            session.claude_session_ids.append(claude_session_id)
            session.save(self.sessions_dir / f"{sid}.json")
        return f"Attached Claude session '{claude_session_id}' to OpenCode session '{sid}'."

    def detach_claude_session(self, session_id: str, claude_session_id: str) -> str:
        """Remove a Claude Code session ID from an OpenCode session."""
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "Session not found."
        session = self.sessions[sid]
        if claude_session_id in session.claude_session_ids:
            session.claude_session_ids.remove(claude_session_id)
            session.save(self.sessions_dir / f"{sid}.json")
            return f"Detached Claude session '{claude_session_id}' from '{sid}'."
        return f"Claude session '{claude_session_id}' was not attached to '{sid}'."

    def end_unattached(self) -> str:
        """End all OpenCode sessions with no live Claude Code session IDs.

        A session is kept if any attached ID is confirmed alive (True) or unknown (None).
        Only sessions where all IDs are confirmed dead (False) are ended.
        """
        targets = []
        for sid, s in self.sessions.items():
            if not s.claude_session_ids:
                targets.append(sid)
            else:
                statuses = [_chitta_session_alive(csid) for csid in s.claude_session_ids]
                if any(st is True or st is None for st in statuses):
                    continue  # keep: at least one alive or status unknown
                targets.append(sid)
        if not targets:
            return "All sessions have live attached Claude Code IDs — nothing to end."
        for sid in targets:
            del self.sessions[sid]
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                path.unlink()
            if self.active_session == sid:
                self.active_session = None
        cleanup_opencode_snapshot()
        return f"Ended {len(targets)} unattached session(s): {', '.join(targets)}"

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        variant_str = f", Variant: {session.variant}" if session.variant else ""
        lines = [f"Session: {sid}", f"Model: {session.model}, Agent: {session.agent}{variant_str}", "---"]

        for msg in session.messages[-last_n:]:
            role = "You" if msg.role == "user" else "OpenCode"
            lines.append(f"\n**{role}:**\n{msg.content}")

        return "\n".join(lines)

    def set_active(self, session_id: str) -> str:
        session_id = _sanitize_session_id(session_id)
        if session_id not in self.sessions:
            return f"Session '{session_id}' not found."
        self.active_session = session_id
        session = self.sessions[session_id]
        variant_str = f", variant={session.variant}" if session.variant else ""
        return f"Active session: '{session_id}' ({session.model}, {session.agent}{variant_str})"

    def set_model(self, model: str, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        old_model = session.model
        session.model = model
        session.save(self.sessions_dir / f"{sid}.json")

        return f"Model changed: {old_model} -> {model}"

    def set_agent(self, agent: str, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        old_agent = session.agent
        session.agent = agent
        session.save(self.sessions_dir / f"{sid}.json")

        return f"Agent changed: {old_agent} -> {agent}"

    def set_variant(self, variant: Optional[str], session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        old_variant = session.variant or "none"
        session.variant = variant
        session.save(self.sessions_dir / f"{sid}.json")

        new_variant = variant or "none"
        return f"Variant changed: {old_variant} -> {new_variant}"

    def end_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if sid:
            sid = _sanitize_session_id(sid)
        if not sid or sid not in self.sessions:
            return "No active session to end."

        del self.sessions[sid]
        session_path = self.sessions_dir / f"{sid}.json"
        if session_path.exists():
            session_path.unlink()

        if self.active_session == sid:
            self.active_session = None

        cleanup_opencode_snapshot()
        return f"Session '{sid}' ended."

    def end_all(self, session_ids: Optional[list] = None, exclude_model: Optional[str] = None) -> str:
        """End all sessions, or only the sessions named in session_ids.

        exclude_model: if set, sessions using this model are kept; all others are ended.
        """
        if session_ids:
            candidates = [_sanitize_session_id(s) for s in session_ids if s in self.sessions]
            not_found = [s for s in session_ids if s not in self.sessions]
        else:
            candidates = list(self.sessions.keys())
            not_found = []

        if exclude_model:
            targets = [s for s in candidates if self.sessions[s].model != exclude_model]
            skipped = [s for s in candidates if self.sessions[s].model == exclude_model]
        else:
            targets = candidates
            skipped = []

        if not targets:
            msg = "No matching sessions to end."
            if skipped:
                msg += f" Kept {len(skipped)} session(s) with model '{exclude_model}'."
            if not_found:
                msg += f" Not found: {', '.join(not_found)}"
            return msg

        for sid in targets:
            del self.sessions[sid]
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                path.unlink()
            if self.active_session == sid:
                self.active_session = None

        cleanup_opencode_snapshot()
        lines = [f"Ended {len(targets)} session(s): {', '.join(targets)}"]
        if skipped:
            lines.append(f"Kept {len(skipped)} session(s) with model '{exclude_model}': {', '.join(skipped)}")
        if not_found:
            lines.append(f"Not found: {', '.join(not_found)}")
        return "\n".join(lines)

    def export_session(self, session_id: Optional[str] = None, export_format: str = "markdown") -> str:
        """Export a session as markdown or JSON."""
        sid = session_id or self.active_session
        if sid:
            sid = _sanitize_session_id(sid)
        if not sid or sid not in self.sessions:
            return "No active session to export."

        session = self.sessions[sid]

        if export_format not in ("markdown", "json"):
            return f"Unsupported export format: '{export_format}'. Use 'markdown' or 'json'."

        if export_format == "json":
            data = {
                "id": session.id,
                "model": session.model,
                "agent": session.agent,
                "variant": session.variant,
                "created": session.created,
                "messages": [asdict(m) for m in session.messages]
            }
            return json.dumps(data, indent=2)

        # Markdown format
        lines = [
            f"# Session: {session.id}",
            f"**Model:** {session.model} | **Agent:** {session.agent} | **Variant:** {session.variant}",
            f"**Created:** {session.created}",
            f"**Messages:** {len(session.messages)}",
            "",
            "---",
            "",
        ]
        for msg in session.messages:
            role = "User" if msg.role == "user" else "OpenCode"
            lines.append(f"## {role}")
            lines.append(f"*{msg.timestamp}*\n")
            lines.append(msg.content)
            lines.append("\n---\n")

        return "\n".join(lines)

    def health_check(self) -> dict:
        """Return server health status."""
        uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
        return {
            "status": "ok",
            "sessions": len(self.sessions),
            "uptime": uptime_seconds
        }

    async def ping(self, session_id: Optional[str] = None) -> str:
        """Send a minimal request to verify the model is reachable and responding.

        Uses the active session's model if available, otherwise falls back to config model.
        Reports response latency so slow models are visible before committing to large tasks.
        """
        if session_id and session_id not in self.sessions:
            return f"Session '{session_id}' not found."
        sid = session_id or self.active_session
        session = self.sessions.get(sid) if sid else None
        model = (session.model if session else None) or self.config.model
        variant = session.variant if session else None

        t0 = asyncio.get_event_loop().time()
        output, code = await self._run_opencode(
            "run", "Reply with only the word: OK", "--model", model, "--format", "json",
            timeout=30, stall_timeout=15
        )
        elapsed = asyncio.get_event_loop().time() - t0

        label = f"{model}" + (f" [{variant}]" if variant else "")
        latency = f"{elapsed:.1f}s"
        speed = " ⚠️ slow" if elapsed > 10 else ""

        if code != 0:
            return f"Model unreachable ({label}, {latency}): {output[:300]}"
        reply, _ = self._parse_opencode_response(output)
        if reply:
            return f"Model reachable ({label}) — {latency}{speed}. Response: {reply.strip()[:100]}"
        return f"Model responded but returned no text ({label}, {latency}{speed})."


class CodexBridge:
    """Bridge for Codex CLI interactions with session management."""

    def __init__(self):
        self.start_time = datetime.now()
        self.config = Config.load()
        self.sessions: dict[str, CodexSession] = {}
        self.active_session: Optional[str] = None
        self.sessions_dir = Path.home() / ".chitta-bridge" / "codex-sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._load_sessions()
        self.jobs: dict[str, CodexJob] = {}
        self._job_tasks: dict[str, "asyncio.Task"] = {}
        self.jobs_dir = Path.home() / ".chitta-bridge" / "codex-jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def _load_sessions(self):
        for path in self.sessions_dir.glob("*.json"):
            try:
                session = CodexSession.load(path)
                self.sessions[session.id] = session
            except Exception:
                pass

    def _load_jobs(self):
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = CodexJob.load(path)
                # Jobs that were "running" at startup are now orphaned
                if job.status == "running":
                    job.status = "failed"
                    job.result = "Server restarted while job was running"
                    job.finished = datetime.now().isoformat()
                    job.save(path)
                self.jobs[job.id] = job
            except Exception:
                pass

    async def _run_codex(self, *args, timeout: int = 120, stall_timeout: int = 120, cwd: Optional[str] = None) -> tuple[str, int]:
        """Run codex CLI command with streaming stdout and stall detection."""
        if not CODEX_BIN:
            return "Codex not installed. Install from: https://github.com/openai/codex", 1

        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(CODEX_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            proc.stdin.close()

            # Drain stderr concurrently so a full stderr pipe never blocks stdout.
            stderr_task = asyncio.ensure_future(proc.stderr.read())

            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + timeout
            first_line = True

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Timed out after {timeout}s", 1
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=read_timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s", 1
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            out = "".join(stdout_parts).strip()
            if proc.returncode == 0:
                err = _strip_startup_warnings(stderr_raw.decode(errors="replace")).strip()
            else:
                err = stderr_raw.decode(errors="replace").strip()
            output = out if out else err
            return output, proc.returncode or 0
        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return "Command timed out", 1
        except Exception as e:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}", 1

    async def _run_codex_exec_stdin(
        self,
        args: list,
        stdin_data: str,
        cwd: str,
        timeout: int = 300,
        stall_timeout: int = 90,
    ) -> tuple[str, int]:
        """Run a codex exec command with stdin data; returns (raw_output, returncode)."""
        if not CODEX_BIN:
            return "Codex not installed.", 1
        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(CODEX_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            proc.stdin.write(stdin_data.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            stderr_task = asyncio.ensure_future(proc.stderr.read())
            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + timeout
            first_line = True

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Timed out after {timeout}s", 1
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=read_timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s", 1
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            out = "".join(stdout_parts)
            if proc.returncode != 0:
                err = stderr_raw.decode(errors="replace").strip()
                return err or out, proc.returncode or 1
            return out, 0
        except asyncio.CancelledError:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            raise
        except Exception as e:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}", 1

    @staticmethod
    def _parse_codex_jsonl(output: str) -> tuple[str, Optional[str]]:
        """Extract reply text and thread_id from Codex JSONL output."""
        reply_parts = []
        thread_id = None
        for line in output.split("\n"):
            if not line or line.startswith("WARNING:"):
                continue
            try:
                event = json.loads(line)
                if not thread_id and event.get("thread_id"):
                    thread_id = event["thread_id"]
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            reply_parts.append(text)
            except json.JSONDecodeError:
                continue
        return "\n".join(reply_parts), thread_id

    async def _run_rescue_background(self, job_id: str):
        """Coroutine that runs a rescue job in the background and updates its state."""
        job = self.jobs[job_id]
        job.started = datetime.now().isoformat()
        job.save(self.jobs_dir / f"{job_id}.json")

        args = ["exec"]
        if job.resume_from:
            args.extend(["resume", job.resume_from])
        if job.model:
            args.extend(["--model", job.model])
        if job.effort:
            args.extend(["--effort", job.effort])
        if job.sandbox:
            args.extend(["--sandbox", job.sandbox])
        args.extend(["--full-auto", "--json", "-"])

        try:
            output, code = await self._run_codex_exec_stdin(
                args, job.task, job.working_dir, timeout=1800, stall_timeout=120
            )
            reply, thread_id = self._parse_codex_jsonl(output)
            job.status = "completed" if code == 0 else "failed"
            job.result = reply or output or "(no output)"
            job.codex_session_id = thread_id
            job.finished = datetime.now().isoformat()
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.result = "(cancelled)"
            job.finished = datetime.now().isoformat()
        finally:
            job.save(self.jobs_dir / f"{job_id}.json")
            self._job_tasks.pop(job_id, None)

    async def start_session(
        self,
        session_id: str,
        model: Optional[str] = None,
        sandbox: Optional[str] = None,
        full_auto: bool = True,
        working_dir: Optional[str] = None
    ) -> str:
        model = model or self.config.codex_model
        sandbox = sandbox or self.config.codex_sandbox

        if not CODEX_BIN:
            return "Codex not installed — session not started. Install from: https://github.com/openai/codex"

        claude_session_id = _get_claude_session_id()

        session = CodexSession(
            id=session_id,
            model=model,
            sandbox=sandbox,
            full_auto=full_auto,
            working_dir=working_dir or os.getcwd(),
            claude_session_ids=[claude_session_id] if claude_session_id else []
        )
        self.sessions[session_id] = session
        self.active_session = session_id
        session.save(self.sessions_dir / f"{session_id}.json")

        result = f"Codex session '{session_id}' started\n  Model: {model}\n  Sandbox: {sandbox}"
        if full_auto:
            result += "\n  Mode: full-auto"
        if working_dir:
            result += f"\n  Working dir: {working_dir}"
        if claude_session_id:
            result += f"\n  Claude session: {claude_session_id}"
        return result

    def get_config(self) -> str:
        """Get current Codex configuration."""
        return f"""Codex configuration:
  Model: {self.config.codex_model}
  Sandbox: {self.config.codex_sandbox}

Set via:
  - ~/.chitta-bridge/config.json
  - CODEX_MODEL, CODEX_SANDBOX env vars
  - codex_configure tool"""

    def set_config(self, model: Optional[str] = None, sandbox: Optional[str] = None) -> str:
        """Update and persist Codex configuration."""
        changes = []
        if model:
            self.config.codex_model = model
            changes.append(f"model: {model}")
        if sandbox:
            self.config.codex_sandbox = sandbox
            changes.append(f"sandbox: {sandbox}")

        if changes:
            self.config.save()
            return "Codex configuration updated:\n  " + "\n  ".join(changes)
        return "No changes made."

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        images: Optional[list[str]] = None
    ) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session. Use codex_start first."

        session = self.sessions[sid]
        session.add_message("user", message)

        # Build args for codex exec (or resume if we have a session)
        if session.codex_session_id:
            # Resume existing conversation
            args = ["exec", "resume", session.codex_session_id]
        else:
            # Start new conversation
            args = ["exec"]

        # Add model only if explicitly set (otherwise use codex config default)
        if session.model:
            args.extend(["--model", session.model])

        # Add sandbox mode (for new sessions or as override)
        if session.full_auto:
            args.append("--full-auto")
        elif not session.codex_session_id:
            # Only set sandbox on first call; resume inherits
            args.extend(["--sandbox", session.sandbox])

        # Add images if provided
        if images:
            for img in images:
                args.extend(["--image", img])

        # Use JSON output for parsing
        args.append("--json")

        # Add the prompt (read from stdin via -)
        args.append("-")

        # Run codex with message as stdin
        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(CODEX_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=session.working_dir
            )
            proc.stdin.write(message.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            # Drain stderr concurrently so a full stderr pipe never blocks stdout.
            stderr_task = asyncio.ensure_future(proc.stderr.read())

            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + 300
            stall_timeout = 90
            first_line = True

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return "Timed out after 300s"
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=read_timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s"
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            output = "".join(stdout_parts)
            if proc.returncode != 0:
                err = stderr_raw.decode(errors="replace").strip()
                return f"Error: {err or output}"
        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return "Command timed out"
        except Exception as e:
            if proc:
                proc.kill()
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}"

        # Parse JSON output (Codex JSONL format)
        reply_parts = []
        for line in output.split("\n"):
            if not line or line.startswith("WARNING:"):
                continue
            try:
                event = json.loads(line)
                # Capture thread ID as session ID
                if not session.codex_session_id and event.get("thread_id"):
                    session.codex_session_id = event["thread_id"]
                # Extract text from item.completed events
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            reply_parts.append(text)
            except json.JSONDecodeError:
                continue

        reply = "\n".join(reply_parts)
        if reply:
            session.add_message("assistant", reply)
            session.save(self.sessions_dir / f"{sid}.json")

        return reply or "No response received"

    async def run_task(
        self,
        task: str,
        working_dir: Optional[str] = None,
        model: Optional[str] = None,
        full_auto: bool = True,
        effort: Optional[str] = None,
        sandbox: Optional[str] = None,
    ) -> str:
        """Run a one-off task without session management."""
        args = self._build_exec_args(model, effort, sandbox=sandbox, full_auto=full_auto)


        cwd = working_dir or os.getcwd()
        output, code = await self._run_codex_exec_stdin(args, task, cwd)
        if code != 0:
            return f"Error: {output}"

        reply, thread_id = self._parse_codex_jsonl(output)
        result = reply or output or "No response received"
        if thread_id:
            result += f"\n\n(Codex session: {thread_id} — resume with: codex resume {thread_id})"
        return result

    async def review_code(
        self,
        working_dir: Optional[str] = None,
        model: Optional[str] = None,
        mode: str = "normal",
        focus: Optional[str] = None,
        base: Optional[str] = None,
        effort: Optional[str] = None,
        background: bool = False,
        sandbox: Optional[str] = None,
    ) -> str:
        """Run Codex code review. mode='adversarial' pressure-tests design decisions."""
        model = model or self.config.codex_model
        cwd = working_dir or os.getcwd()

        if mode == "adversarial":
            focus_clause = f"\n\nSpecific focus area: {focus}" if focus else ""
            task = (
                "You are a senior adversarial code reviewer. Your job is NOT to find obvious bugs — "
                "it is to challenge the design decisions, architecture, and tradeoffs in this code.\n\n"
                "Review the uncommitted changes (or the full repo if no changes) and:\n"
                "1. Question whether the chosen approach was the right one at all\n"
                "2. Identify hidden assumptions that could break under load or edge cases\n"
                "3. Pressure-test failure modes: what happens when X fails, Y is slow, Z is empty?\n"
                "4. Challenge the architecture: would a different design be safer/simpler?\n"
                "5. Flag race conditions, data loss risks, rollback gaps, reliability holes\n"
                "6. Propose at least one alternative approach and explain the tradeoff"
                f"{focus_clause}\n\n"
                "Be direct and hard to satisfy. Do not praise good code — focus exclusively on risks."
            )
            if base:
                task += f"\n\nReview changes relative to base: {base}"
            if background:
                return await self._launch_rescue(task, model=model, effort=effort, cwd=cwd, sandbox=sandbox)
            output, code = await self._run_codex_exec_stdin(
                self._build_exec_args(model, effort, sandbox=sandbox), task, cwd, timeout=600
            )
            if code != 0:
                return f"Error: {output}"
            reply, thread_id = self._parse_codex_jsonl(output)
            result = reply or output or "Review complete"
            if thread_id:
                result += f"\n\n(Codex session: {thread_id} — resume with: codex resume {thread_id})"
            return result
        else:
            # Normal review via `codex exec review`
            args = ["exec", "review", "--model", model, "--json"]
            if base:
                args.extend(["--base", base])
            if effort:
                args.extend(["--effort", effort])
            if sandbox:
                args.extend(["--sandbox", sandbox])
            if background:
                task = f"Run a code review{f' vs {base}' if base else ''}"
                return await self._launch_rescue(task, model=model, effort=effort, cwd=cwd, sandbox=sandbox)
            output, code = await self._run_codex(*args, cwd=cwd, timeout=600)
            if code != 0:
                return f"Error: {output}"
            return output or "Review complete"

    def _build_exec_args(
        self,
        model: Optional[str],
        effort: Optional[str],
        sandbox: Optional[str] = None,
        full_auto: bool = True,
    ) -> list:
        args = ["exec", "--skip-git-repo-check"]
        if model:
            args.extend(["--model", model])
        if effort:
            args.extend(["--effort", effort])
        if sandbox:
            args.extend(["--sandbox", sandbox])
            if full_auto:
                args.append("--full-auto")
        elif full_auto:
            args.append("--full-auto")
        else:
            args.extend(["--sandbox", self.config.codex_sandbox])
        args.extend(["--json", "-"])
        return args

    async def _launch_rescue(
        self,
        task: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        cwd: Optional[str] = None,
        resume_from: Optional[str] = None,
        sandbox: Optional[str] = None,
    ) -> str:
        """Start a background rescue job and return its ID immediately."""
        job_id = f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        job = CodexJob(
            id=job_id,
            task=task,
            model=model or self.config.codex_model,
            working_dir=cwd or os.getcwd(),
            effort=effort,
            sandbox=sandbox,
            resume_from=resume_from,
        )
        self.jobs[job_id] = job
        job.save(self.jobs_dir / f"{job_id}.json")
        t = asyncio.create_task(self._run_rescue_background(job_id))
        self._job_tasks[job_id] = t
        return (
            f"Rescue job started in background.\n"
            f"  Job ID: {job_id}\n"
            f"  Model: {job.model}{f', effort: {effort}' if effort else ''}"
            f"{f', resume: {resume_from}' if resume_from else ''}\n"
            f"  Use codex_job_status to check progress.\n"
            f"  Use codex_job_result {job_id} when done."
        )

    async def rescue(
        self,
        task: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        working_dir: Optional[str] = None,
        background: bool = True,
        resume_from: Optional[str] = None,
        fresh: bool = False,
        sandbox: Optional[str] = None,
    ) -> str:
        """Delegate a task to Codex. Supports background execution and session resume."""
        cwd = working_dir or os.getcwd()
        if not resume_from and not fresh:
            recent = [j for j in self.jobs.values() if j.codex_session_id and j.status == "completed"]
            if recent:
                latest = max(recent, key=lambda j: j.finished or "")
                resume_from = latest.codex_session_id

        if background:
            return await self._launch_rescue(task, model=model, effort=effort, cwd=cwd, resume_from=resume_from, sandbox=sandbox)

        # Foreground execution
        args = self._build_exec_args(model or self.config.codex_model, effort, sandbox=sandbox)
        if resume_from:
            args = ["exec", "resume", resume_from] + args[1:]
        output, code = await self._run_codex_exec_stdin(args, task, cwd, timeout=600)
        if code != 0:
            return f"Error: {output}"
        reply, thread_id = self._parse_codex_jsonl(output)
        result = reply or output or "No response received"
        if thread_id:
            result += f"\n\n(Codex session: {thread_id} — resume with: codex rescue --resume {thread_id})"
        return result

    def job_status(self, job_id: Optional[str] = None) -> str:
        """Show status of one or all rescue jobs."""
        if not self.jobs:
            return "No rescue jobs found."
        if job_id:
            job_id = _sanitize_session_id(job_id) if re.fullmatch(r'[\w\-]+', job_id) else None
            if not job_id or job_id not in self.jobs:
                return f"Job '{job_id}' not found."
            job = self.jobs[job_id]
            lines = [
                f"Job: {job.id}",
                f"  Status:  {job.status}",
                f"  Task:    {job.task[:120]}{'…' if len(job.task) > 120 else ''}",
                f"  Model:   {job.model}{f' ({job.effort})' if job.effort else ''}",
                f"  Created: {job.created}",
            ]
            if job.started:
                lines.append(f"  Started: {job.started}")
            if job.finished:
                lines.append(f"  Finished: {job.finished}")
            if job.codex_session_id:
                lines.append(f"  Codex session: {job.codex_session_id}")
            return "\n".join(lines)

        # All jobs — show most recent 10
        jobs_sorted = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:10]
        lines = [f"Rescue Jobs ({len(self.jobs)} total, showing latest 10):"]
        for job in jobs_sorted:
            marker = {"running": "⏳", "completed": "✓", "failed": "✗", "cancelled": "⊘"}.get(job.status, "?")
            age = job.created[:19].replace("T", " ")
            lines.append(f"  {marker} {job.id}  [{job.status}]  {age}  {job.task[:60]}{'…' if len(job.task) > 60 else ''}")
        return "\n".join(lines)

    def job_result(self, job_id: Optional[str] = None) -> str:
        """Get the final output of a completed rescue job."""
        if not self.jobs:
            return "No rescue jobs found."
        if not job_id:
            completed = [j for j in self.jobs.values() if j.status == "completed"]
            if not completed:
                return "No completed jobs found."
            job = max(completed, key=lambda j: j.finished or "")
        else:
            if job_id not in self.jobs:
                return f"Job '{job_id}' not found."
            job = self.jobs[job_id]

        if job.status == "running":
            return f"Job '{job.id}' is still running. Check back with codex_job_status."
        lines = [
            f"Job: {job.id}  [{job.status}]",
            f"Task: {job.task[:200]}{'…' if len(job.task) > 200 else ''}",
            "",
            job.result or "(no output)",
        ]
        if job.codex_session_id:
            lines.append(f"\nCodex session: {job.codex_session_id}")
            lines.append(f"Resume in Codex: codex resume {job.codex_session_id}")
        return "\n".join(lines)

    def job_cancel(self, job_id: Optional[str] = None) -> str:
        """Cancel a running rescue job."""
        if not job_id:
            running = [j for j in self.jobs.values() if j.status == "running"]
            if not running:
                return "No running jobs to cancel."
            if len(running) > 1:
                return f"Multiple running jobs: {', '.join(j.id for j in running)}. Specify job_id."
            job_id = running[0].id

        if job_id not in self.jobs:
            return f"Job '{job_id}' not found."
        job = self.jobs[job_id]
        if job.status != "running":
            return f"Job '{job_id}' is not running (status: {job.status})."

        task = self._job_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return f"Cancellation requested for job '{job_id}'."
        # Already done but status not updated yet
        job.status = "cancelled"
        job.finished = datetime.now().isoformat()
        job.save(self.jobs_dir / f"{job_id}.json")
        return f"Job '{job_id}' marked as cancelled."

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No Codex sessions found."

        lines = ["Codex Sessions:"]
        for sid, session in self.sessions.items():
            active = " (active)" if sid == self.active_session else ""
            msg_count = len(session.messages)
            mode = "full-auto" if session.full_auto else session.sandbox
            cc_ids = f", claude={','.join(session.claude_session_ids)}" if session.claude_session_ids else ""
            lines.append(f"  - {sid}: {session.model} [{mode}], {msg_count} messages{cc_ids}{active}")
        return "\n".join(lines)

    def attach_claude_session(self, session_id: str, claude_session_id: str) -> str:
        """Register a Claude Code session ID as using this Codex session."""
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "Codex session not found."
        session = self.sessions[sid]
        if claude_session_id not in session.claude_session_ids:
            session.claude_session_ids.append(claude_session_id)
            session.save(self.sessions_dir / f"{sid}.json")
        return f"Attached Claude session '{claude_session_id}' to Codex session '{sid}'."

    def detach_claude_session(self, session_id: str, claude_session_id: str) -> str:
        """Remove a Claude Code session ID from a Codex session."""
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "Codex session not found."
        session = self.sessions[sid]
        if claude_session_id in session.claude_session_ids:
            session.claude_session_ids.remove(claude_session_id)
            session.save(self.sessions_dir / f"{sid}.json")
            return f"Detached Claude session '{claude_session_id}' from '{sid}'."
        return f"Claude session '{claude_session_id}' was not attached to '{sid}'."

    def end_unattached(self) -> str:
        """End all Codex sessions with no live Claude Code session IDs.

        A session is kept if any attached ID is confirmed alive (True) or unknown (None).
        Only sessions where all IDs are confirmed dead (False) are ended.
        """
        targets = []
        for sid, s in self.sessions.items():
            if not s.claude_session_ids:
                targets.append(sid)
            else:
                statuses = [_chitta_session_alive(csid) for csid in s.claude_session_ids]
                if any(st is True or st is None for st in statuses):
                    continue  # keep: at least one alive or status unknown
                targets.append(sid)
        if not targets:
            return "All Codex sessions have live attached Claude Code IDs — nothing to end."
        for sid in targets:
            del self.sessions[sid]
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                path.unlink()
            if self.active_session == sid:
                self.active_session = None
        cleanup_opencode_snapshot()
        return f"Ended {len(targets)} unattached Codex session(s): {', '.join(targets)}"

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session."

        session = self.sessions[sid]
        mode = "full-auto" if session.full_auto else session.sandbox
        lines = [f"Codex Session: {sid}", f"Model: {session.model}, Mode: {mode}", "---"]

        for msg in session.messages[-last_n:]:
            role = "You" if msg.role == "user" else "Codex"
            lines.append(f"\n**{role}:**\n{msg.content}")

        return "\n".join(lines)

    def set_active(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return f"Codex session '{session_id}' not found."
        self.active_session = session_id
        session = self.sessions[session_id]
        mode = "full-auto" if session.full_auto else session.sandbox
        return f"Active Codex session: '{session_id}' ({session.model}, {mode})"

    def set_model(self, model: str, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session."

        session = self.sessions[sid]
        old_model = session.model
        session.model = model
        session.save(self.sessions_dir / f"{sid}.json")

        return f"Codex model changed: {old_model} -> {model}"

    def end_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session to end."

        del self.sessions[sid]
        session_path = self.sessions_dir / f"{sid}.json"
        if session_path.exists():
            session_path.unlink()

        if self.active_session == sid:
            self.active_session = None

        cleanup_opencode_snapshot()
        return f"Codex session '{sid}' ended."

    def end_all(self, session_ids: Optional[list] = None, exclude_model: Optional[str] = None) -> str:
        """End all Codex sessions, or only the sessions named in session_ids.

        exclude_model: if set, sessions using this model are kept; all others are ended.
        """
        if session_ids:
            candidates = [_sanitize_session_id(s) for s in session_ids if s in self.sessions]
            not_found = [s for s in session_ids if s not in self.sessions]
        else:
            candidates = list(self.sessions.keys())
            not_found = []

        if exclude_model:
            targets = [s for s in candidates if self.sessions[s].model != exclude_model]
            skipped = [s for s in candidates if self.sessions[s].model == exclude_model]
        else:
            targets = candidates
            skipped = []

        if not targets:
            msg = "No matching Codex sessions to end."
            if skipped:
                msg += f" Kept {len(skipped)} session(s) with model '{exclude_model}'."
            if not_found:
                msg += f" Not found: {', '.join(not_found)}"
            return msg

        for sid in targets:
            del self.sessions[sid]
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                path.unlink()
            if self.active_session == sid:
                self.active_session = None

        cleanup_opencode_snapshot()
        lines = [f"Ended {len(targets)} Codex session(s): {', '.join(targets)}"]
        if skipped:
            lines.append(f"Kept {len(skipped)} session(s) with model '{exclude_model}': {', '.join(skipped)}")
        if not_found:
            lines.append(f"Not found: {', '.join(not_found)}")
        return "\n".join(lines)

    def health_check(self) -> dict:
        """Return Codex bridge health status."""
        uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
        return {
            "status": "ok" if CODEX_BIN else "codex not found",
            "codex_installed": CODEX_BIN is not None,
            "sessions": len(self.sessions),
            "uptime": uptime_seconds
        }


# ---------------------------------------------------------------------------
# GPU Node Auto-Discovery
# ---------------------------------------------------------------------------

# Default port for Ollama / vLLM (OpenAI-compatible)
_LOCAL_LLM_PORT = 11434

# URL cache files written by slurm-serve-ollama.sh
_OLLAMA_URL_GLOB = "/tmp/ollama-server-*.url"


class GpuNodeDiscovery:
    """Discover GPU nodes reachable via Slurm or direct hostname and probe for Ollama/vLLM."""

    # Well-known node hostnames to probe (can be extended via environment variable
    # CHITTA_BRIDGE_GPU_NODES=node1,node2,...)
    _ENV_NODES_VAR = "CHITTA_BRIDGE_GPU_NODES"

    @staticmethod
    def _probe_ollama(base_url: str, timeout: int = 4) -> Optional[list[str]]:
        """Return list of available model names at base_url, or None if unreachable."""
        # base_url ends with /v1; Ollama's tag endpoint is at the parent /api/tags
        tags_url = base_url.rstrip("/v1").rstrip("/") + "/api/tags"
        try:
            req = urllib.request.urlopen(tags_url, timeout=timeout)
            data = json.loads(req.read().decode())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return None

    @classmethod
    def _cached_urls(cls) -> list[tuple[str, str]]:
        """Return list of (model_hint, base_url) from /tmp/ollama-server-*.url files."""
        results = []
        for path in _glob.glob(_OLLAMA_URL_GLOB):
            try:
                url = Path(path).read_text().strip()
                if url:
                    # Extract model hint from filename: /tmp/ollama-server-<model>.url
                    hint = Path(path).stem.removeprefix("ollama-server-")
                    results.append((hint, url))
            except OSError:
                pass
        return results

    @classmethod
    def _slurm_gpu_nodes(cls) -> list[str]:
        """Return hostnames of running Slurm jobs that allocated GPU resources."""
        if not shutil.which("squeue"):
            return []
        try:
            import subprocess
            out = subprocess.check_output(
                ["squeue", "--format=%T %N %b", "--noheader", "--me"],
                timeout=8,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            nodes = []
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "RUNNING" and "gpu" in parts[2].lower():
                    nodes.append(parts[1])
            return nodes
        except Exception:
            return []

    @classmethod
    def _env_nodes(cls) -> list[str]:
        """Return nodes from the CHITTA_BRIDGE_GPU_NODES env variable."""
        val = os.environ.get(cls._ENV_NODES_VAR, "")
        return [n.strip() for n in val.split(",") if n.strip()]

    @classmethod
    def discover(cls) -> list[dict]:
        """
        Return a list of reachable LLM endpoints:
          [{"base_url": "http://node:11434/v1", "node": "nodename", "models": [...], "source": "..."}]
        """
        seen: dict[str, dict] = {}  # base_url -> entry

        # 1. Cached URL files (highest priority — already health-checked at launch time)
        for hint, base_url in cls._cached_urls():
            models = cls._probe_ollama(base_url)
            if models is not None:
                node = base_url.split("//")[-1].split(":")[0]
                seen[base_url] = {"base_url": base_url, "node": node, "models": models, "source": "cached"}

        # 2. Slurm running GPU jobs (my own jobs that allocated a GPU)
        for node in cls._slurm_gpu_nodes():
            base_url = f"http://{node}:{_LOCAL_LLM_PORT}/v1"
            if base_url not in seen:
                models = cls._probe_ollama(base_url)
                if models is not None:
                    seen[base_url] = {"base_url": base_url, "node": node, "models": models, "source": "slurm"}

        # 3. Env-configured nodes
        for node in cls._env_nodes():
            base_url = f"http://{node}:{_LOCAL_LLM_PORT}/v1"
            if base_url not in seen:
                models = cls._probe_ollama(base_url)
                if models is not None:
                    seen[base_url] = {"base_url": base_url, "node": node, "models": models, "source": "env"}

        # 4. Localhost fallback
        local_url = f"http://localhost:{_LOCAL_LLM_PORT}/v1"
        if local_url not in seen:
            models = cls._probe_ollama(local_url)
            if models is not None:
                seen[local_url] = {"base_url": local_url, "node": "localhost", "models": models, "source": "local"}

        return list(seen.values())


# ---------------------------------------------------------------------------
# Local Model Bridge (OpenAI-compatible: Ollama / vLLM)
# ---------------------------------------------------------------------------


@dataclass
class LocalSession:
    id: str
    endpoint: str          # e.g. http://node:11434/v1
    model: str
    messages: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    attached_claude_sessions: list = field(default_factory=list)


class LocalModelBridge:
    """Chat with local LLMs (Ollama/vLLM) running on GPU nodes via OpenAI-compatible API."""

    def __init__(self):
        self.sessions: dict[str, LocalSession] = {}
        self._active_id: Optional[str] = None
        self._start_time = datetime.now()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, session_id: str, model: str, endpoint: str) -> str:
        _sanitize_session_id(session_id)
        if session_id in self.sessions:
            return f"Session '{session_id}' already exists."
        s = LocalSession(id=session_id, endpoint=endpoint.rstrip("/"), model=model)
        self.sessions[session_id] = s
        self._active_id = session_id
        return f"Started local session '{session_id}' → {endpoint} model={model}"

    def _active(self) -> Optional[LocalSession]:
        if self._active_id and self._active_id in self.sessions:
            return self.sessions[self._active_id]
        return None

    def set_active(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return f"Session '{session_id}' not found."
        self._active_id = session_id
        s = self.sessions[session_id]
        return f"Switched to local session '{session_id}' ({s.model} @ {s.endpoint})"

    def end_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self._active_id
        if not sid or sid not in self.sessions:
            return "No session to end."
        del self.sessions[sid]
        if self._active_id == sid:
            self._active_id = next(iter(self.sessions), None)
        return f"Ended local session '{sid}'."

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No local model sessions."
        lines = []
        for sid, s in self.sessions.items():
            marker = " [active]" if sid == self._active_id else ""
            lines.append(f"  {sid}{marker} — {s.model} @ {s.endpoint} ({len(s.messages)} messages)")
        return "\n".join(lines)

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self._active_id
        if not sid or sid not in self.sessions:
            return "No session found."
        s = self.sessions[sid]
        msgs = s.messages[-last_n:]
        if not msgs:
            return "No messages yet."
        return "\n".join(f"[{m['role']}]: {m['content'][:300]}" for m in msgs)

    def get_config(self) -> str:
        s = self._active()
        if not s:
            return "No active local session."
        return f"Session: {s.id}\nEndpoint: {s.endpoint}\nModel: {s.model}\nMessages: {len(s.messages)}"

    def health_check(self) -> dict:
        uptime = int((datetime.now() - self._start_time).total_seconds())
        return {"status": "ok", "sessions": len(self.sessions), "uptime": uptime}

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        sid = session_id or self._active_id
        if not sid or sid not in self.sessions:
            return "Error: no active local session."
        s = self.sessions[sid]
        s.messages.append({"role": "user", "content": message})

        payload: dict = {
            "model": s.model,
            "messages": s.messages.copy(),
            "stream": False,
        }
        if system_prompt:
            payload["messages"] = [{"role": "system", "content": system_prompt}] + payload["messages"]

        try:
            reply = await asyncio.get_event_loop().run_in_executor(
                None, self._post_completion, s.endpoint, payload
            )
        except Exception as e:
            s.messages.pop()  # roll back user message on error
            return f"Error calling local model: {e}"

        s.messages.append({"role": "assistant", "content": reply})
        return reply

    @staticmethod
    def _post_completion(endpoint: str, payload: dict, timeout: int = 300) -> str:
        """POST to /v1/chat/completions with retries for model-loading connection drops."""
        import http.client
        url = f"{endpoint}/chat/completions"
        data = json.dumps(payload).encode()
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(4):
            if attempt:
                import time
                time.sleep(10 * attempt)  # 10s, 20s, 30s back-off while model loads
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                # Retry on 500/502/503 (model loading, GPU contention)
                if e.code in (500, 502, 503):
                    last_exc = e
                    continue
                raise
            except (http.client.RemoteDisconnected, ConnectionResetError, urllib.error.URLError) as e:
                last_exc = e
                continue
        raise last_exc

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_models_at(endpoint: str, timeout: int = 8) -> list[str]:
        tags_url = endpoint.rstrip("/v1").rstrip("/") + "/api/tags"
        try:
            req = urllib.request.urlopen(tags_url, timeout=timeout)
            data = json.loads(req.read().decode())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Web Search (DuckDuckGo – no API key, stdlib only)
# ---------------------------------------------------------------------------

class WebSearch:
    """Search the web via DuckDuckGo HTML and return parsed results."""

    _DDG_URL = "https://html.duckduckgo.com/html/"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    _RESULT_RE = re.compile(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?<a class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    @classmethod
    def search(cls, query: str, max_results: int = 8, timeout: int = 10) -> list[dict]:
        data = urllib.parse.urlencode({"q": query}).encode()
        req = urllib.request.Request(cls._DDG_URL, data=data, headers=cls._HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        results = []
        for url, title, snippet in cls._RESULT_RE.findall(body):
            if "/y.js?" in url:
                # DuckDuckGo redirect — extract actual URL
                m = re.search(r"uddg=([^&]+)", url)
                if m:
                    url = urllib.parse.unquote(m.group(1))
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            title = _html.unescape(title)
            snippet = _html.unescape(snippet)
            if url and title:
                results.append({"url": url, "title": title, "snippet": snippet})
            if len(results) >= max_results:
                break
        return results

    @classmethod
    def search_formatted(cls, query: str, max_results: int = 8) -> str:
        results = cls.search(query, max_results)
        if not results:
            return f"No results found for: {query}"
        lines = [f"Web search: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def fetch_page(cls, url: str, max_chars: int = 12000, timeout: int = 15) -> str:
        req = urllib.request.Request(url, headers=cls._HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = _html.unescape(text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"
        return text


# ---------------------------------------------------------------------------
# Soul Integration (chittad Unix socket — bidirectional memory bridge)
# ---------------------------------------------------------------------------

class SoulClient:
    """Connect to chittad daemon for memory recall and storage."""

    @staticmethod
    def _djb2_hash(s: str) -> int:
        h = 5381
        for c in s:
            h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
        return h

    @classmethod
    def _socket_path(cls) -> str:
        home = os.environ.get("HOME", "")
        mind_path = os.path.join(home, ".claude", "mind")
        hash_val = cls._djb2_hash(mind_path)
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg and os.access(xdg, os.W_OK):
            base = os.path.join(xdg, "chitta")
        elif home:
            base = os.path.join(home, ".cache", "chitta")
        else:
            base = "/tmp"
        return os.path.join(base, f"chitta-{hash_val}.sock")

    @classmethod
    def _call(cls, method: str, arguments: dict, timeout: float = 5.0) -> Optional[str]:
        path = cls._socket_path()
        if not os.path.exists(path):
            return None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(path)
            req = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": method, "arguments": arguments},
            })
            sock.sendall((req + "\n").encode())
            response = b""
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                response += chunk
                if b"\n" in response:
                    break
            sock.close()
            data = json.loads(response.decode().strip())
            result = data.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return str(result)
        except Exception:
            return None

    @classmethod
    def recall(cls, query: str, limit: int = 5, realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if realm:
            args["realm"] = realm
        return cls._call("recall", args)

    @classmethod
    def smart_context(cls, task: str, realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"task": task}
        if realm:
            args["realm"] = realm
        return cls._call("smart_context", args, timeout=10.0)

    @classmethod
    def remember(cls, content: str, kind: str = "episode",
                 tags: str = "", confidence: float = 0.8,
                 realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"content": content, "type": kind, "confidence": confidence}
        if tags:
            args["tags"] = tags
        if realm:
            args["realm"] = realm
        return cls._call("remember", args)

    @classmethod
    def hybrid_recall(cls, query: str, limit: int = 5, realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if realm:
            args["realm"] = realm
        return cls._call("hybrid_recall", args)

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls._socket_path())


class Orchestrator:
    """Multi-agent orchestration for complex workflows."""

    def __init__(self, opencode_bridge: OpenCodeBridge, codex_bridge: CodexBridge):
        self.opencode = opencode_bridge
        self.codex = codex_bridge

    async def multi_consult(
        self,
        question: str,
        backends: list[str] = None,
        files: list[str] = None,
        synthesize: bool = True,
    ) -> str:
        """Fan-out a question to multiple backends in parallel, optionally synthesize results.

        Args:
            question: The question/task to send to all backends
            backends: List of backends to consult ["opencode", "codex"] (default: both)
            files: Files to attach (OpenCode only)
            synthesize: Whether to synthesize results into a unified response
        """
        backends = backends or ["opencode", "codex"]
        results: dict[str, str] = {}
        errors: dict[str, str] = {}

        async def run_opencode():
            try:
                # Create temporary session
                sid = f"multi-{datetime.now().strftime('%Y%m%d-%H%M%S')}-oc"
                await self.opencode.start_session(sid)
                result = await self.opencode.send_message(question, sid, files)
                self.opencode.end_session(sid)
                return result
            except Exception as e:
                return f"[OpenCode error: {e}]"

        async def run_codex():
            try:
                # Use stateless run for multi-consult
                result = await self.codex.run_task(question)
                return result
            except Exception as e:
                return f"[Codex error: {e}]"

        # Run backends in parallel
        tasks = []
        task_names = []
        if "opencode" in backends:
            tasks.append(run_opencode())
            task_names.append("opencode")
        if "codex" in backends:
            tasks.append(run_codex())
            task_names.append("codex")

        if not tasks:
            return "No backends specified."

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for name, response in zip(task_names, responses):
            if isinstance(response, Exception):
                errors[name] = str(response)
            else:
                results[name] = response

        # Format output
        if not synthesize or len(results) == 1:
            parts = []
            for name, response in results.items():
                parts.append(f"## {name.upper()}\n\n{response}")
            for name, error in errors.items():
                parts.append(f"## {name.upper()} (error)\n\n{error}")
            return "\n\n---\n\n".join(parts)

        # Synthesize using OpenCode
        if results:
            synthesis_prompt = f"""Synthesize these responses to the question: "{question}"

"""
            for name, response in results.items():
                synthesis_prompt += f"### {name.upper()} Response:\n{response}\n\n"

            synthesis_prompt += """### Instructions:
- Identify areas of agreement and disagreement
- Highlight unique insights from each perspective
- Provide a unified recommendation that considers all viewpoints
- Note any caveats or areas needing further investigation"""

            try:
                sid = f"synth-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                await self.opencode.start_session(sid, agent="build")
                synthesis = await self.opencode.send_message(synthesis_prompt, sid, _raw=True)
                self.opencode.end_session(sid)
                return f"## SYNTHESIS\n\n{synthesis}\n\n---\n\n## Individual Responses\n\n" + \
                    "\n\n---\n\n".join(f"### {n.upper()}\n{r}" for n, r in results.items())
            except Exception as e:
                # Fallback to non-synthesized output
                parts = [f"[Synthesis failed: {e}]"]
                for name, response in results.items():
                    parts.append(f"## {name.upper()}\n\n{response}")
                return "\n\n---\n\n".join(parts)

        return "All backends failed: " + ", ".join(f"{k}: {v}" for k, v in errors.items())

    async def chain(
        self,
        steps: list[dict],
    ) -> str:
        """Execute a chain of agent steps, passing results forward.

        Each step is a dict with:
            - backend: "opencode" or "codex"
            - task: The task/prompt (can include {previous} placeholder)
            - model: Optional model override
            - agent: Optional agent override (OpenCode only)

        Example:
            [
                {"backend": "opencode", "task": "Plan how to implement X", "agent": "plan"},
                {"backend": "codex", "task": "Implement the plan: {previous}"},
                {"backend": "opencode", "task": "Review this implementation: {previous}"}
            ]
        """
        if not steps:
            return "No steps provided."

        results = []
        previous = ""

        for i, step in enumerate(steps, 1):
            backend = step.get("backend", "opencode")
            task = step.get("task", "")
            model = step.get("model")
            agent = step.get("agent")

            # Substitute {previous} placeholder
            if "{previous}" in task and previous:
                task = task.replace("{previous}", previous)

            step_header = f"## Step {i}: {backend.upper()}"
            if model:
                step_header += f" (model={model})"
            if agent:
                step_header += f" (agent={agent})"

            try:
                if backend == "opencode":
                    sid = f"chain-{i}-{datetime.now().strftime('%H%M%S')}"
                    await self.opencode.start_session(sid, model=model, agent=agent)
                    result = await self.opencode.send_message(task, sid, _raw=True)
                    self.opencode.end_session(sid)
                elif backend == "codex":
                    result = await self.codex.run_task(task, model=model)
                else:
                    result = f"Unknown backend: {backend}"

                previous = result
                results.append(f"{step_header}\n\n{result}")

            except Exception as e:
                error_msg = f"Step {i} failed: {e}"
                results.append(f"{step_header}\n\n**Error:** {error_msg}")
                # Continue chain even if a step fails
                previous = f"[Previous step failed: {e}]"

        return "\n\n---\n\n".join(results)

    async def delegate_to_codex(
        self,
        task: str,
        working_dir: str = None,
        model: str = None,
        return_to_opencode: bool = False,
        opencode_followup: str = None,
    ) -> str:
        """Delegate a task to Codex, optionally return result to OpenCode for review.

        This enables: Claude -> OpenCode -> Codex -> OpenCode flow

        Args:
            task: Task for Codex to execute
            working_dir: Working directory for Codex
            model: Codex model to use
            return_to_opencode: Whether to send Codex result back to OpenCode
            opencode_followup: Custom prompt for OpenCode followup (default: review)
        """
        # Run task in Codex
        codex_result = await self.codex.run_task(task, working_dir=working_dir, model=model)

        if not return_to_opencode:
            return f"## Codex Result\n\n{codex_result}"

        # Send to OpenCode for review/followup
        followup = opencode_followup or f"""Review this Codex output and provide feedback:

## Original Task
{task}

## Codex Result
{codex_result}

## Instructions
- Evaluate the correctness and completeness
- Identify any issues or improvements needed
- Provide a final assessment"""

        try:
            sid = f"delegate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.opencode.start_session(sid, agent="build")
            review = await self.opencode.send_message(followup, sid, _raw=True)
            self.opencode.end_session(sid)

            return f"""## Codex Execution

{codex_result}

---

## OpenCode Review

{review}"""
        except Exception as e:
            return f"""## Codex Execution

{codex_result}

---

## OpenCode Review (failed)

Error: {e}"""

    async def parallel_agents(
        self,
        tasks: list[dict],
    ) -> str:
        """Run multiple agent tasks in parallel across backends.

        Each task is a dict with:
            - backend: "opencode" or "codex"
            - task: The task/prompt
            - name: Optional name for the task
            - model: Optional model override

        All tasks run concurrently, results returned together.
        """
        if not tasks:
            return "No tasks provided."

        async def run_task(task_def: dict, index: int):
            backend = task_def.get("backend", "opencode")
            task = task_def.get("task", "")
            name = task_def.get("name", f"Task {index}")
            model = task_def.get("model")

            try:
                if backend == "opencode":
                    sid = f"parallel-{index}-{datetime.now().strftime('%H%M%S')}"
                    await self.opencode.start_session(sid, model=model)
                    result = await self.opencode.send_message(task, sid, _raw=True)
                    self.opencode.end_session(sid)
                elif backend == "codex":
                    result = await self.codex.run_task(task, model=model)
                else:
                    result = f"Unknown backend: {backend}"

                return {"name": name, "backend": backend, "result": result, "error": None}
            except Exception as e:
                return {"name": name, "backend": backend, "result": None, "error": str(e)}

        # Run all tasks in parallel
        coros = [run_task(t, i) for i, t in enumerate(tasks, 1)]
        results = await asyncio.gather(*coros)

        # Format output
        parts = []
        for r in results:
            header = f"## {r['name']} ({r['backend']})"
            if r["error"]:
                parts.append(f"{header}\n\n**Error:** {r['error']}")
            else:
                parts.append(f"{header}\n\n{r['result']}")

        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Discussion Room — async multi-agent roundtable
# ---------------------------------------------------------------------------

@dataclass
class AgentSoul:
    """Identity and capabilities for a room participant — the agent's 'soul'."""
    system_prompt: str             # markdown body: expertise, personality, rules
    realm: str = ""                # chitta memory namespace, e.g. "agent:critic"
    tools: list = field(default_factory=list)  # ["recall", "remember", "web_search", ...]
    max_tool_turns: int = 3        # max tool-use iterations per response
    max_rounds: int = 0            # max discussion rounds (0 = unlimited)
    response_format: str = ""      # structured output template
    challenge_bias: float = 0.5    # 0=agreeable, 1=devil's advocate


# Tool definitions for the mediated tool-calling loop (Ollama native + XML fallback)
# Organized by category matching Claude Code's agent tools, plus chitta-specific extras.

def _tool(name: str, desc: str, props: dict, required: list) -> dict:
    """Helper to build an OpenAI function-calling tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }

AGENT_TOOL_DEFINITIONS = [
    # ── Memory (core) ──────────────────────────────────────────────────
    _tool("recall", "Semantic search over your memory. Returns the most similar memories.",
          {"query": {"type": "string", "description": "What to search for"},
           "limit": {"type": "integer", "description": "Max results (default 5)"}},
          ["query"]),
    _tool("remember", "Store an important insight or fact in your memory for future recall.",
          {"content": {"type": "string", "description": "What to remember"},
           "tags": {"type": "string", "description": "Comma-separated tags"}},
          ["content"]),
    _tool("smart_context", "Get contextually relevant memories, code symbols, and graph connections for a task.",
          {"task": {"type": "string", "description": "Describe the task or topic"}},
          ["task"]),

    # ── Memory (extended) ──────────────────────────────────────────────
    _tool("recall_keyword", "BM25 keyword search over memory. Best when you know exact terms.",
          {"query": {"type": "string", "description": "Keywords to search for"},
           "limit": {"type": "integer", "description": "Max results (default 5)"}},
          ["query"]),
    _tool("recall_temporal", "Search memories from a specific time range.",
          {"query": {"type": "string", "description": "What to search for"},
           "since": {"type": "string", "description": "Start time (ISO 8601 or relative like '2h', '7d')"},
           "until": {"type": "string", "description": "End time (ISO 8601 or 'now')"},
           "limit": {"type": "integer", "description": "Max results (default 5)"}},
          ["query"]),
    _tool("hybrid_recall", "Combined vector + BM25 keyword search. Best general-purpose recall.",
          {"query": {"type": "string", "description": "What to search for"},
           "limit": {"type": "integer", "description": "Max results (default 5)"}},
          ["query"]),
    _tool("5w_search", "Structured who/what/when/where/why search over memory.",
          {"who": {"type": "string", "description": "Person or entity"},
           "what": {"type": "string", "description": "Action or event"},
           "when": {"type": "string", "description": "Time reference"},
           "where": {"type": "string", "description": "Location or context"},
           "why": {"type": "string", "description": "Reason or cause"}},
          []),
    _tool("forget", "Remove a memory by query. Use when information is wrong or outdated.",
          {"query": {"type": "string", "description": "Memory to forget (matched by similarity)"}},
          ["query"]),

    # ── Web ────────────────────────────────────────────────────────────
    _tool("web_search", "Search the web for current information via DuckDuckGo.",
          {"query": {"type": "string", "description": "Search query"},
           "max_results": {"type": "integer", "description": "Max results (default 5)"}},
          ["query"]),
    _tool("web_fetch", "Fetch a web page and return its text content (HTML stripped).",
          {"url": {"type": "string", "description": "URL to fetch"},
           "max_chars": {"type": "integer", "description": "Max characters to return (default 8000)"}},
          ["url"]),

    # ── File operations ────────────────────────────────────────────────
    _tool("read_file", "Read a file's contents with line numbers. Handles text, PDF, Jupyter notebooks, and images.",
          {"path": {"type": "string", "description": "Absolute or relative file path"},
           "offset": {"type": "integer", "description": "Start line (0-based, default 0)"},
           "limit": {"type": "integer", "description": "Max lines to read (default 200, max 500)"},
           "pages": {"type": "string", "description": "Page range for PDF files (e.g. '1-5', '3')"}},
          ["path"]),
    _tool("write_file", "Create or overwrite a file with new content. Must read_file first for existing files.",
          {"path": {"type": "string", "description": "File path to write"},
           "content": {"type": "string", "description": "Content to write"}},
          ["path", "content"]),
    _tool("edit_file", "Replace a specific string in a file. Shows match locations if ambiguous, unified diff on success.",
          {"path": {"type": "string", "description": "File path to edit"},
           "old_string": {"type": "string", "description": "Exact text to find"},
           "new_string": {"type": "string", "description": "Replacement text"},
           "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"}},
          ["path", "old_string", "new_string"]),
    _tool("glob", "Find files matching a glob pattern. Returns paths with size and age, sorted by mtime.",
          {"pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')"},
           "path": {"type": "string", "description": "Base directory (default: cwd)"}},
          ["pattern"]),
    _tool("grep", "Search file contents for a regex pattern. Supports multiline, output modes, pagination.",
          {"pattern": {"type": "string", "description": "Regex pattern to search for"},
           "path": {"type": "string", "description": "File or directory to search (default: cwd)"},
           "glob": {"type": "string", "description": "Glob filter for files (e.g., '*.py')"},
           "type": {"type": "string", "description": "File type filter (e.g., 'py', 'js', 'rust')"},
           "context": {"type": "integer", "description": "Lines of context around matches (default 2)"},
           "multiline": {"type": "boolean", "description": "Enable multiline matching (default false)"},
           "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"],
                           "description": "Output mode (default: content)"},
           "offset": {"type": "integer", "description": "Skip first N results (default 0)"},
           "head_limit": {"type": "integer", "description": "Max results to return (default 50)"}},
          ["pattern"]),

    # ── Shell ──────────────────────────────────────────────────────────
    _tool("bash", "Execute a shell command. Sandboxed: no network, persistent cwd per participant.",
          {"command": {"type": "string", "description": "Shell command to execute"},
           "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 60)"},
           "description": {"type": "string", "description": "What this command does (for audit trail)"},
           "background": {"type": "boolean", "description": "Run in background, return immediately (default false)"}},
          ["command"]),

    # ── Code intelligence (via chitta) ─────────────────────────────────
    _tool("read_function", "Read a specific function's source code by name (uses chitta symbol index).",
          {"name": {"type": "string", "description": "Function or method name to read"}},
          ["name"]),
    _tool("read_symbol", "Read any code symbol (class, function, variable) by name.",
          {"name": {"type": "string", "description": "Symbol name to look up"}},
          ["name"]),
    _tool("search_symbols", "Search for code symbols matching a query.",
          {"query": {"type": "string", "description": "Search query for symbols"},
           "limit": {"type": "integer", "description": "Max results (default 10)"}},
          ["query"]),
    _tool("codebase_overview", "Get a high-level overview of the codebase structure.",
          {},
          []),

    # ── Task tracking ──────────────────────────────────────────────────
    _tool("todo_add", "Add a task to your personal todo list for this discussion.",
          {"task": {"type": "string", "description": "Task description"},
           "priority": {"type": "string", "description": "low, medium, high (default: medium)"}},
          ["task"]),
    _tool("todo_list", "List your current todo items.",
          {},
          []),
    _tool("todo_done", "Mark a todo item as complete by its number.",
          {"number": {"type": "integer", "description": "Todo item number (1-based)"}},
          ["number"]),
]

# XML fallback instruction block for models that don't support native tool calling
TOOL_XML_INSTRUCTIONS = """## Available Tools

You can request tool calls by outputting EXACTLY this XML format:

<tool_call>
{"tool": "recall", "args": {"query": "your search query", "limit": 5}}
</tool_call>

Wait for the result before continuing. You may make multiple tool calls.
When done with tools, output your final response inside:

<final_response>
Your contribution to the discussion goes here.
</final_response>

Available tools:
- recall: Search your memory. Args: query (string, required), limit (int, default 5)
- remember: Store a memory. Args: content (string, required), tags (string, optional)
- web_search: Search the web. Args: query (string, required), max_results (int, default 5)
- smart_context: Get relevant context for a task. Args: task (string, required)
"""


@dataclass
class DiscussionRoom:
    """Shared message board where multiple agents post and read asynchronously."""
    id: str
    topic: str
    participants: list  # [{name, backend, session_id, soul?}]
    messages: list = field(default_factory=list)  # [{name, content, ts}]
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    turn_counts: dict = field(default_factory=dict)  # {name: int} per-participant round count
    challenge_mode: bool = False
    files: list = field(default_factory=list)


class RoomManager:
    """Manage discussion rooms and run async multi-agent conversations."""

    def __init__(self, opencode_bridge: "OpenCodeBridge", codex_bridge: "CodexBridge", local_bridge: "LocalModelBridge"):
        self.opencode = opencode_bridge
        self.codex = codex_bridge
        self.local = local_bridge
        self.rooms: dict[str, DiscussionRoom] = {}

    def create(self, room_id: str, topic: str, participants: list[dict],
               files: Optional[list[str]] = None) -> str:
        if room_id in self.rooms:
            return f"Room '{room_id}' already exists."
        expanded = _expand_paths(files or [])
        room = DiscussionRoom(id=room_id, topic=topic, participants=participants, files=expanded)
        room.messages.append({"name": "TOPIC", "content": topic, "ts": datetime.now().isoformat()})
        # Inject soul context if chittad is running (filter code symbols)
        if SoulClient.is_available():
            ctx = SoulClient.hybrid_recall(topic, limit=5)
            if ctx and len(ctx.strip()) > 20:
                code_markers = ["[code]", "[symbol]", "function ", "class ", "method "]
                if not any(m in ctx[:200] for m in code_markers):
                    room.messages.append({
                        "name": "CONTEXT",
                        "content": f"[Relevant memories]\n{ctx}",
                        "ts": datetime.now().isoformat(),
                    })
        self.rooms[room_id] = room
        names = ", ".join(p["name"] for p in participants)
        soul_tag = " (with soul context)" if len(room.messages) > 1 else ""
        return f"Room '{room_id}' created with {len(participants)} participants: {names}{soul_tag}"

    def add_participant(self, room_id: str, participant: dict) -> str:
        if room_id not in self.rooms:
            return f"Room '{room_id}' not found."
        room = self.rooms[room_id]
        room.participants.append(participant)
        return f"Added '{participant['name']}' to room '{room_id}'. Now {len(room.participants)} participants."

    def read(self, room_id: str) -> str:
        if room_id not in self.rooms:
            return f"Room '{room_id}' not found."
        room = self.rooms[room_id]
        lines = [f"# Discussion Room: {room_id}", f"**Topic:** {room.topic}", ""]
        for msg in room.messages:
            ts = msg["ts"][11:19]  # HH:MM:SS
            lines.append(f"**[{ts}] {msg['name']}:**")
            lines.append(msg["content"])
            lines.append("")
        return "\n".join(lines)

    async def synthesize(self, room_id: str, synthesizer: Optional[dict] = None) -> str:
        """Run a final synthesis pass over the full transcript — distills all responses into one answer."""
        if room_id not in self.rooms:
            return f"Room '{room_id}' not found."
        room = self.rooms[room_id]

        transcript = self.read(room_id)
        prompt = (
            f"You are a neutral synthesizer reviewing a multi-agent discussion.\n\n"
            f"{transcript}\n\n"
            f"## Synthesis Task\n"
            f"Resolve any contradictions between participants, then distill the discussion into a single, coherent answer:\n"
            f"1. **Core consensus** — what all participants agreed on\n"
            f"2. **Key disagreements** — where they diverged and why\n"
            f"3. **Best answer** — your integrated recommendation, drawing on the strongest points\n"
            f"4. **Open questions** — what remains unresolved\n"
        )

        # Use synthesizer config or infer backend from room participants
        if synthesizer:
            synth = synthesizer
        else:
            # Infer backend from participants — if all use the same backend, reuse it
            backends = [p.get("backend", "claude") for p in room.participants]
            inferred = backends[0] if backends and len(set(backends)) == 1 else "claude"
            synth = {"name": "Synthesizer", "backend": inferred}
        synth_name = synth.get("name", "Synthesizer")
        backend = synth.get("backend", "claude")
        sid = synth.get("session_id")

        try:
            if backend == "claude":
                reply = await self._run_claude_p(prompt)
            elif backend == "local":
                base_url = synth.get("base_url") or synth.get("endpoint")
                model = synth.get("model", "")
                if not base_url:
                    nodes = await asyncio.get_event_loop().run_in_executor(None, GpuNodeDiscovery.discover)
                    if nodes:
                        base_url = nodes[0]["base_url"]
                        if not model and nodes[0]["models"]:
                            model = nodes[0]["models"][0]
                if base_url:
                    tmp = f"synth-{room_id}"
                    self.local.start_session(tmp, model=model or "default", endpoint=base_url)
                    reply = await self.local.send_message(prompt, tmp)
                    self.local.end_session(tmp)
                else:
                    reply = "[error: no local endpoint found for synthesis]"
            elif backend == "codex":
                if sid and sid in self.codex.sessions:
                    reply = await self.codex.send_message(prompt, sid)
                else:
                    reply = await self.codex.run_task(prompt)
            else:  # opencode
                if sid and sid in self.opencode.sessions:
                    reply = await self.opencode.send_message(prompt, sid, _raw=True)
                else:
                    tmp = f"synth-{room_id}"
                    await self.opencode.start_session(tmp, model=synth.get("model"))
                    reply = await self.opencode.send_message(prompt, tmp, _raw=True)
                    self.opencode.end_session(tmp)
        except Exception as e:
            reply = f"[synthesis error: {e}]"

        room.messages.append({"name": f"⟳ {synth_name}", "content": reply, "ts": datetime.now().isoformat()})
        # Store synthesis back to soul memory
        if SoulClient.is_available():
            participants = ", ".join(p["name"] for p in room.participants)
            # Extract key terms from topic for tags
            topic_words = re.sub(r"[^\w\s]", "", room.topic.lower()).split()
            stop = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "and", "or", "of", "in", "to", "for", "with", "on", "at", "by", "from", "do", "does"}
            tags = ",".join(dict.fromkeys(w for w in topic_words if w not in stop and len(w) > 2))[:200]
            memory = (
                f"[room-synthesis:{room_id}] {room.topic}\n"
                f"Participants: {participants} | Synthesizer: {synth_name}\n\n"
                f"{reply[:2000]}"
            )
            SoulClient.remember(memory, kind="wisdom", tags=f"room,synthesis,{tags}", confidence=0.85)
        return f"## Synthesis by {synth_name}\n\n{reply}"

    # ------------------------------------------------------------------
    # Soul-aware context building
    # ------------------------------------------------------------------

    def _parse_soul(self, participant: dict) -> Optional[AgentSoul]:
        """Parse soul from participant dict, if present."""
        raw = participant.get("soul")
        if not raw:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return AgentSoul(system_prompt=raw)
        name_slug = re.sub(r"[^a-z0-9]+", "-", participant["name"].lower()).strip("-")
        return AgentSoul(
            system_prompt=raw.get("system_prompt", raw.get("prompt", "")),
            realm=raw.get("realm", f"agent:{name_slug}"),
            tools=raw.get("tools", []),
            max_tool_turns=raw.get("max_tool_turns", 3),
            max_rounds=raw.get("max_rounds", 0),
            response_format=raw.get("response_format", ""),
            challenge_bias=raw.get("challenge_bias", 0.5),
        )

    def _build_thread_context(self, room: DiscussionRoom, participant: dict) -> tuple[str, str]:
        """Build (system_prompt, user_message) for a participant.

        If the participant has a soul, the system prompt contains their identity,
        loaded memories, and tool instructions. Otherwise falls back to the
        generic prompt used before.
        """
        name = participant["name"]
        soul = self._parse_soul(participant)

        # -- Build discussion transcript --
        transcript_parts = []
        for msg in room.messages:
            if msg["name"] == "TOPIC":
                continue
            transcript_parts.append(f"**{msg['name']}:** {msg['content']}")
            transcript_parts.append("")
        transcript = "\n".join(transcript_parts)

        # -- System prompt (the soul) --
        if soul and soul.system_prompt:
            sys_parts = [soul.system_prompt]

            # Load relevant memories from the agent's realm
            if soul.realm and SoulClient.is_available():
                memories = SoulClient.hybrid_recall(room.topic, limit=5, realm=soul.realm)
                if memories and len(memories.strip()) > 20:
                    sys_parts.append(f"\n## Your Memories\n{memories}")
                # Also check global memories — but filter to non-code types only
                global_mem = SoulClient.hybrid_recall(room.topic, limit=3)
                if global_mem and len(global_mem.strip()) > 20:
                    # Skip if it's mostly code symbols (not useful for discussions)
                    code_markers = ["[code]", "[symbol]", "function ", "class ", "method "]
                    if not any(m in global_mem[:200] for m in code_markers):
                        sys_parts.append(f"\n## Shared Knowledge\n{global_mem}")

            # Tool instructions (XML fallback — always included for models that
            # don't support native tool calling)
            if soul.tools:
                available = [t for t in AGENT_TOOL_DEFINITIONS
                             if t["function"]["name"] in soul.tools]
                if available:
                    tool_lines = []
                    for t in available:
                        fn = t["function"]
                        params = fn["parameters"]["properties"]
                        param_desc = ", ".join(
                            f'{k} ({v.get("type", "string")}'
                            f'{", required" if k in fn["parameters"].get("required", []) else ""})'
                            for k, v in params.items()
                        )
                        tool_lines.append(f"- **{fn['name']}**: {fn['description']}. Args: {param_desc}")
                    sys_parts.append(TOOL_XML_INSTRUCTIONS.replace(
                        "Available tools:\n- recall: Search your memory. Args: query (string, required), limit (int, default 5)\n- remember: Store a memory. Args: content (string, required), tags (string, optional)\n- web_search: Search the web. Args: query (string, required), max_results (int, default 5)\n- smart_context: Get relevant context for a task. Args: task (string, required)",
                        "Available tools:\n" + "\n".join(tool_lines),
                    ))

            # Response format
            if soul.response_format:
                sys_parts.append(f"\n## Response Format\n{soul.response_format}")

            # Challenge bias instruction
            if soul.challenge_bias > 0.6:
                sys_parts.append(
                    "\n## Critical Thinking Directive\n"
                    "You are a rigorous critic. When other participants make claims, "
                    "ACTIVELY challenge them. Ask for evidence. Point out logical gaps. "
                    "Do NOT agree just to be polite. If something sounds wrong or "
                    "unsubstantiated, say so directly."
                )

            system_prompt = "\n".join(sys_parts)
        else:
            system_prompt = (
                f"You are **{name}**, a specialist participant in a multi-agent discussion. "
                f"Contribute your distinct expertise to the topic. Be analytical, specific, "
                f"and direct. React to other participants' arguments — challenge, extend, or "
                f"correct them as warranted."
            )

        # -- User message --
        user_parts = [
            f"**Topic:** {room.topic}",
            "",
            "## Discussion so far",
            transcript if transcript else "(No messages yet — you are first to respond.)",
            "",
            "## Your turn",
            f"You are {name}. Read the full discussion above and contribute your perspective.",
            "Be direct and specific. React to what others said — agree, challenge, or add something new.",
        ]
        if not (soul and soul.tools):
            user_parts.append("Keep it to 2-4 paragraphs.")

        return system_prompt, "\n".join(user_parts)

    # ------------------------------------------------------------------
    # Tool execution for room participants
    # ------------------------------------------------------------------

    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # Fallback: bare JSON with "tool" key — greedy enough for nested args
    _BARE_TOOL_RE = re.compile(
        r'(\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\})', re.DOTALL,
    )
    _FINAL_RESPONSE_RE = re.compile(r"<final_response>(.*?)</final_response>", re.DOTALL)

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """Extract a tool call from model output.

        Tries <tool_call> XML first, then falls back to bare JSON with
        "tool" key — many local models output the JSON without XML wrappers.
        """
        # Try XML-wrapped first
        m = self._TOOL_CALL_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if "tool" in parsed:
                    return {"tool": parsed["tool"], "args": parsed.get("args", {})}
            except json.JSONDecodeError:
                pass

        # Fallback: bare JSON tool call (models often skip XML tags)
        m = self._BARE_TOOL_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if "tool" in parsed:
                    return {"tool": parsed["tool"], "args": parsed.get("args", {})}
            except json.JSONDecodeError:
                pass

        # Last resort: try to find any JSON object with "tool" and "args" keys
        # (handles extra whitespace, markdown code blocks, etc.)
        stripped = text.strip()
        # Strip markdown code fences
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner = "\n".join(
                ln for ln in lines if not ln.strip().startswith("```")
            ).strip()
            if inner:
                stripped = inner
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "tool" in parsed:
                return {"tool": parsed["tool"], "args": parsed.get("args", {})}
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _extract_final_response(self, text: str) -> Optional[str]:
        """Extract <final_response> content, if present."""
        m = self._FINAL_RESPONSE_RE.search(text)
        return m.group(1).strip() if m else None

    async def _execute_agent_tool(self, tool_name: str, args: dict,
                                   realm: Optional[str] = None,
                                   participant_name: str = "") -> str:
        """Execute a tool on behalf of a room participant.

        Categories:
          - Memory (core): recall, remember, smart_context
          - Memory (extended): recall_keyword, recall_temporal, hybrid_recall,
                               5w_search, forget
          - Web: web_search, web_fetch
          - File: read_file, write_file, edit_file, glob, grep
          - Shell: bash
          - Code intelligence: read_function, read_symbol, search_symbols,
                               codebase_overview
          - Task tracking: todo_add, todo_list, todo_done
        """
        try:
            # ── Memory (core) ──────────────────────────────────────────
            if tool_name == "recall":
                result = SoulClient.recall(
                    query=args.get("query", ""),
                    limit=int(args.get("limit", 5)),
                    realm=realm,
                )
                return result or "(no memories found)"

            elif tool_name == "remember":
                result = SoulClient.remember(
                    content=args.get("content", ""),
                    kind=args.get("kind", "wisdom"),
                    tags=args.get("tags", ""),
                    confidence=float(args.get("confidence", 0.8)),
                    realm=realm,
                )
                return result or "(stored)"

            elif tool_name == "smart_context":
                result = SoulClient.smart_context(
                    task=args.get("task", ""),
                    realm=realm,
                )
                return result or "(no context found)"

            # ── Memory (extended) ──────────────────────────────────────
            elif tool_name == "recall_keyword":
                a: dict[str, Any] = {"query": args.get("query", ""),
                                     "limit": int(args.get("limit", 5))}
                if realm:
                    a["realm"] = realm
                return SoulClient._call("recall_keyword", a) or "(no results)"

            elif tool_name == "recall_temporal":
                a = {"query": args.get("query", ""),
                     "limit": int(args.get("limit", 5))}
                if args.get("since"):
                    a["since"] = args["since"]
                if args.get("until"):
                    a["until"] = args["until"]
                if realm:
                    a["realm"] = realm
                return SoulClient._call("recall_temporal", a) or "(no results)"

            elif tool_name == "hybrid_recall":
                result = SoulClient.hybrid_recall(
                    query=args.get("query", ""),
                    limit=int(args.get("limit", 5)),
                    realm=realm,
                )
                return result or "(no results)"

            elif tool_name == "5w_search":
                a = {}
                for k in ("who", "what", "when", "where", "why"):
                    if args.get(k):
                        a[k] = args[k]
                if not a:
                    return "(provide at least one of: who, what, when, where, why)"
                if realm:
                    a["realm"] = realm
                return SoulClient._call("5w_search", a) or "(no results)"

            elif tool_name == "forget":
                a = {"query": args.get("query", "")}
                if realm:
                    a["realm"] = realm
                return SoulClient._call("forget", a) or "(forgotten)"

            # ── Web ────────────────────────────────────────────────────
            elif tool_name == "web_search":
                results = WebSearch.search(
                    query=args.get("query", ""),
                    max_results=int(args.get("max_results", 5)),
                )
                if not results:
                    return "(no web results)"
                lines = []
                for r in results:
                    lines.append(f"**{r.get('title', '')}**")
                    lines.append(f"  {r.get('url', '')}")
                    lines.append(f"  {r.get('snippet', '')}")
                return "\n".join(lines)

            elif tool_name == "web_fetch":
                url = args.get("url", "")
                if not url:
                    return "(no URL provided)"
                max_chars = int(args.get("max_chars", 8000))
                text = WebSearch.fetch_page(url, max_chars=max_chars)
                return text if text else "(failed to fetch page)"

            # ── File operations ────────────────────────────────────────
            elif tool_name == "read_file":
                return self._tool_read_file(args, participant_name=participant_name)

            elif tool_name == "write_file":
                return self._tool_write_file(args, participant_name=participant_name)

            elif tool_name == "edit_file":
                return self._tool_edit_file(args)

            elif tool_name == "glob":
                return self._tool_glob(args)

            elif tool_name == "grep":
                return await self._tool_grep(args)

            # ── Shell ──────────────────────────────────────────────────
            elif tool_name == "bash":
                return await self._tool_bash(args, participant_name=participant_name)

            # ── Code intelligence ──────────────────────────────────────
            elif tool_name == "read_function":
                return SoulClient._call("read_function", {"name": args.get("name", "")}) or "(not found)"

            elif tool_name == "read_symbol":
                return SoulClient._call("read_symbol", {"name": args.get("name", "")}) or "(not found)"

            elif tool_name == "search_symbols":
                a = {"query": args.get("query", ""), "limit": int(args.get("limit", 10))}
                return SoulClient._call("search_symbols", a) or "(no symbols found)"

            elif tool_name == "codebase_overview":
                return SoulClient._call("codebase_overview", {}) or "(no overview available)"

            # ── Task tracking ──────────────────────────────────────────
            elif tool_name == "todo_add":
                return self._tool_todo_add(args, participant_name)

            elif tool_name == "todo_list":
                return self._tool_todo_list(participant_name)

            elif tool_name == "todo_done":
                return self._tool_todo_done(args, participant_name)

            else:
                return f"(unknown tool: {tool_name})"
        except Exception as e:
            return f"(tool error: {e})"

    # ------------------------------------------------------------------
    # File tool implementations — each explains why it beats Claude Code's
    # ------------------------------------------------------------------

    # Track which files each participant has read (for write safety)
    _read_files: dict = {}  # class-level: {participant: {path: True}}

    @staticmethod
    def _is_binary(path: Path, check_bytes: int = 8192) -> bool:
        """Detect binary files by checking for null bytes and high-byte ratio."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(check_bytes)
            if b"\x00" in chunk:
                return True
            # High ratio of non-text bytes = binary
            non_text = sum(1 for b in chunk if b > 127 or (b < 32 and b not in (9, 10, 13)))
            return len(chunk) > 0 and non_text / len(chunk) > 0.3
        except Exception:
            return False

    @staticmethod
    def _format_size(n: int) -> str:
        if n > 1_048_576:
            return f"{n / 1_048_576:.1f}MB"
        if n > 1024:
            return f"{n / 1024:.1f}KB"
        return f"{n}B"

    # Image extensions for metadata detection
    _IMAGE_EXTS = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
        ".webp", ".ico", ".svg", ".heic", ".heif", ".avif",
    })

    def _tool_read_file(self, args: dict, participant_name: str = "") -> str:
        """Read a file — handles text, PDF, Jupyter notebooks, and images."""
        path = Path(args.get("path", "")).expanduser().resolve()
        str_path = str(path)
        # Block sensitive system paths
        blocked_prefixes = ("/proc", "/sys", "/dev")
        blocked_exact = ("/etc/shadow", "/etc/gshadow", "/etc/master.passwd")
        # Block sensitive dotfiles/dirs (credentials, keys, tokens)
        blocked_dotpaths = (
            "/.ssh/", "/.gnupg/", "/.aws/", "/.azure/", "/.gcloud/",
            "/.config/gh/", "/.docker/config.json", "/.kube/config",
            "/.netrc", "/.env", "/.npmrc",
        )
        if any(str_path.startswith(b) for b in blocked_prefixes):
            return f"(blocked: cannot read {path})"
        if str_path in blocked_exact:
            return f"(blocked: cannot read {path})"
        if any(bp in str_path for bp in blocked_dotpaths):
            return f"(blocked: sensitive file — {path})"
        if not path.exists():
            return f"(file not found: {path})"
        if not path.is_file():
            return f"(not a file: {path})"
        size = path.stat().st_size
        suffix = path.suffix.lower()

        # Track this read for write-safety
        key = participant_name or "_global"
        if key not in RoomManager._read_files:
            RoomManager._read_files[key] = {}
        RoomManager._read_files[key][str(path)] = True

        # ── Image metadata ────────────────────────────────────────────
        if suffix in self._IMAGE_EXTS:
            info = f"(image: {path}, {self._format_size(size)}, type: {suffix})"
            # Try to get dimensions
            try:
                import struct
                with open(path, "rb") as f:
                    head = f.read(32)
                if suffix == ".png" and head[:8] == b"\x89PNG\r\n\x1a\n":
                    w, h = struct.unpack(">II", head[16:24])
                    info = f"(image: {path}, {w}x{h} PNG, {self._format_size(size)})"
                elif suffix in (".jpg", ".jpeg"):
                    # JPEG: scan for SOF marker
                    with open(path, "rb") as f:
                        data = f.read(min(size, 65536))
                    i = 0
                    while i < len(data) - 9:
                        if data[i] == 0xFF and data[i + 1] in (0xC0, 0xC2):
                            h, w = struct.unpack(">HH", data[i + 5:i + 9])
                            info = f"(image: {path}, {w}x{h} JPEG, {self._format_size(size)})"
                            break
                        i += 1
                elif suffix == ".gif" and head[:6] in (b"GIF87a", b"GIF89a"):
                    w, h = struct.unpack("<HH", head[6:10])
                    info = f"(image: {path}, {w}x{h} GIF, {self._format_size(size)})"
                elif suffix == ".svg":
                    # SVG is text — fall through to text reading
                    pass
                else:
                    pass
            except Exception:
                pass
            if suffix != ".svg":
                return info

        # ── PDF extraction via pdftotext ──────────────────────────────
        if suffix == ".pdf":
            pages_arg = args.get("pages", "")
            try:
                import subprocess as _sp
                cmd = ["pdftotext", "-layout"]
                if pages_arg:
                    # Parse "3" or "1-5"
                    parts = pages_arg.split("-")
                    if len(parts) == 2:
                        cmd += ["-f", parts[0].strip(), "-l", parts[1].strip()]
                    elif len(parts) == 1:
                        cmd += ["-f", parts[0].strip(), "-l", parts[0].strip()]
                cmd += [str(path), "-"]
                result = _sp.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout
                    lines = text.splitlines()
                    total = len(lines)
                    limit = min(int(args.get("limit", 200)), 500)
                    offset = int(args.get("offset", 0))
                    selected = lines[offset:offset + limit]
                    numbered = [f"{i + offset + 1:>5}\t{line}" for i, line in enumerate(selected)]
                    pg = f", pages {pages_arg}" if pages_arg else ""
                    header = f"# {path} (PDF{pg}, {total} text lines, {self._format_size(size)})"
                    if total > offset + limit:
                        header += f" — showing {offset + 1}-{offset + len(selected)}"
                    return header + "\n" + "\n".join(numbered)
            except (FileNotFoundError, _sp.TimeoutExpired):
                pass
            return f"(PDF file: {path}, {self._format_size(size)} — install pdftotext to read)"

        # ── Jupyter notebook (.ipynb) ─────────────────────────────────
        if suffix == ".ipynb":
            try:
                import json as _json
                nb = _json.loads(path.read_bytes())
                cells = nb.get("cells", [])
                parts = []
                for ci, cell in enumerate(cells):
                    ctype = cell.get("cell_type", "code")
                    src = "".join(cell.get("source", []))
                    tag = f"[{ctype} cell {ci + 1}]"
                    parts.append(f"{'=' * 60}\n{tag}")
                    parts.append(src)
                    # Show outputs for code cells
                    outputs = cell.get("outputs", [])
                    for out in outputs:
                        otype = out.get("output_type", "")
                        if otype == "stream":
                            parts.append("[output]\n" + "".join(out.get("text", [])))
                        elif otype in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            if "text/plain" in data:
                                parts.append("[result]\n" + "".join(data["text/plain"]))
                            if "image/png" in data:
                                parts.append("[image: embedded PNG]")
                        elif otype == "error":
                            parts.append("[error] " + out.get("ename", "") + ": " + out.get("evalue", ""))
                text = "\n".join(parts)
                lines = text.splitlines()
                total = len(lines)
                offset = int(args.get("offset", 0))
                limit = min(int(args.get("limit", 200)), 500)
                selected = lines[offset:offset + limit]
                numbered = [f"{i + offset + 1:>5}\t{line}" for i, line in enumerate(selected)]
                kernel = nb.get("metadata", {}).get("kernelspec", {}).get("display_name", "?")
                header = f"# {path} (Jupyter notebook, {len(cells)} cells, kernel: {kernel}, {self._format_size(size)})"
                if total > offset + limit:
                    header += f" — showing {offset + 1}-{offset + len(selected)}"
                return header + "\n" + "\n".join(numbered)
            except Exception as exc:
                return f"(notebook parse error: {exc})"

        # ── Binary detection ──────────────────────────────────────────
        if self._is_binary(path):
            return f"(binary file: {path}, {self._format_size(size)}, type: {suffix or 'unknown'})"

        offset = int(args.get("offset", 0))
        limit = min(int(args.get("limit", 200)), 500)
        try:
            raw = path.read_bytes()
            # Detect encoding
            encoding = "utf-8"
            if raw[:3] == b"\xef\xbb\xbf":
                encoding = "utf-8-sig"
            elif raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                encoding = "utf-16"
            text = raw.decode(encoding, errors="replace")
            lines = text.splitlines()
            total = len(lines)
            selected = lines[offset:offset + limit]
            numbered = [f"{i + offset + 1:>5}\t{line}" for i, line in enumerate(selected)]
            header = f"# {path} ({total} lines, {self._format_size(size)}, {encoding})"
            if total > offset + limit:
                header += f" — showing {offset + 1}-{offset + len(selected)}"
            return header + "\n" + "\n".join(numbered)
        except Exception as e:
            return f"(read error: {e})"

    def _tool_write_file(self, args: dict, participant_name: str = "") -> str:
        """Write a file. Beats Claude Code's Write:
        CC: overwrites without checking if file was read, no backup.
        Ours: requires read-before-overwrite for existing files (prevents
        blind clobbering), creates .bak backup of existing content,
        auto-creates parent dirs, shows diff summary.
        """
        path = Path(args.get("path", "")).expanduser().resolve()
        content = args.get("content", "")
        str_path = str(path)
        blocked_prefixes = ("/proc", "/sys", "/dev", "/etc")
        blocked_dotpaths = (
            "/.ssh/", "/.gnupg/", "/.aws/", "/.azure/", "/.gcloud/",
            "/.config/gh/", "/.docker/config.json", "/.kube/config",
            "/.netrc", "/.env", "/.npmrc",
        )
        if any(str_path.startswith(b) for b in blocked_prefixes):
            return f"(blocked: cannot write to {path})"
        if any(bp in str_path for bp in blocked_dotpaths):
            return f"(blocked: sensitive path — {path})"

        # Read-before-overwrite check
        key = participant_name or "_global"
        read_set = RoomManager._read_files.get(key, {})
        if path.exists() and str(path) not in read_set:
            return (
                f"(safety: must read_file '{path}' before overwriting it. "
                f"This prevents accidentally clobbering existing content.)"
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Backup existing file
            old_content = ""
            if path.exists():
                old_content = path.read_text(errors="replace")
                bak = path.with_suffix(path.suffix + ".bak")
                bak.write_text(old_content)

            path.write_text(content)
            new_lines = len(content.splitlines())

            if old_content:
                old_lines = len(old_content.splitlines())
                added = max(0, new_lines - old_lines)
                removed = max(0, old_lines - new_lines)
                return (
                    f"(wrote {len(content)} bytes to {path} — "
                    f"{new_lines} lines, +{added}/-{removed} vs previous, "
                    f"backup at {path.with_suffix(path.suffix + '.bak')})"
                )
            return f"(created {path} — {len(content)} bytes, {new_lines} lines)"
        except Exception as e:
            return f"(write error: {e})"

    @staticmethod
    def _tool_edit_file(args: dict) -> str:
        """Edit a file. Beats Claude Code's Edit:
        CC: fails if old_string not unique — but only tells you "not unique".
        Ours: fails if not unique AND shows all match locations so the model
        can add context to disambiguate. Also shows unified diff of the
        change, and supports replace_all flag.
        """
        path = Path(args.get("path", "")).expanduser().resolve()
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        replace_all = args.get("replace_all", False)
        if not old:
            return "(old_string is empty)"
        if old == new:
            return "(old_string and new_string are identical)"
        if not path.exists():
            return f"(file not found: {path})"
        try:
            text = path.read_text(errors="replace")
            count = text.count(old)
            if count == 0:
                # Help the model: show similar lines
                old_first_line = old.splitlines()[0].strip() if old.strip() else old
                lines = text.splitlines()
                near = [
                    f"  {i + 1}: {line.rstrip()}"
                    for i, line in enumerate(lines)
                    if old_first_line[:30] in line
                ][:5]
                hint = ""
                if near:
                    hint = "\nSimilar lines found:\n" + "\n".join(near)
                return f"(old_string not found in {path}){hint}"

            if count > 1 and not replace_all:
                # Show all match locations to help disambiguate
                lines = text.splitlines()
                old_first = old.splitlines()[0] if old.splitlines() else old
                locations = [
                    f"  line {i + 1}: {line.rstrip()}"
                    for i, line in enumerate(lines)
                    if old_first in line
                ][:10]
                return (
                    f"(old_string matches {count} locations in {path} — "
                    f"add surrounding context to make it unique, "
                    f"or set replace_all=true)\n"
                    + "\n".join(locations)
                )

            # Apply edit
            if replace_all:
                updated = text.replace(old, new)
                replaced = count
            else:
                updated = text.replace(old, new, 1)
                replaced = 1
            path.write_text(updated)

            # Show unified diff of the change
            old_lines = old.splitlines(keepends=True)
            new_lines = new.splitlines(keepends=True)
            import difflib
            diff = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile="before", tofile="after", lineterm="",
            ))
            diff_str = "\n".join(diff[:20])  # cap diff output

            # Find line number of edit
            pre_edit = text[:text.index(old)]
            line_num = pre_edit.count("\n") + 1

            return (
                f"(replaced {replaced} occurrence{'s' if replaced > 1 else ''} "
                f"at line {line_num} in {path})\n{diff_str}"
            )
        except Exception as e:
            return f"(edit error: {e})"

    @staticmethod
    def _tool_glob(args: dict) -> str:
        """Find files. Beats Claude Code's Glob:
        CC: returns paths only, sorted by mtime.
        Ours: shows file sizes, line counts for text files, mtime,
        groups by directory for readability, caps at 50.
        """
        import glob as glob_mod
        pattern = args.get("pattern", "")
        base = args.get("path", ".")
        try:
            matches = glob_mod.glob(os.path.join(base, pattern), recursive=True)
            # Filter to files only (skip directories)
            matches = [m for m in matches if os.path.isfile(m)]
            if not matches:
                return f"(no files match '{pattern}' in {base})"
            matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
            total = len(matches)
            capped = matches[:50]
            lines = []
            for m in capped:
                try:
                    stat = os.stat(m)
                    sz = stat.st_size
                    size_str = RoomManager._format_size(sz)
                    # Show age
                    import time
                    age_s = time.time() - stat.st_mtime
                    if age_s < 3600:
                        age = f"{int(age_s / 60)}m ago"
                    elif age_s < 86400:
                        age = f"{int(age_s / 3600)}h ago"
                    elif age_s < 604800:
                        age = f"{int(age_s / 86400)}d ago"
                    else:
                        age = f"{int(age_s / 604800)}w ago"
                    lines.append(f"  {m}  ({size_str}, {age})")
                except OSError:
                    lines.append(f"  {m}")
            header = f"# {total} files matching '{pattern}'"
            if total > 50:
                header += " (showing 50 most recent)"
            return header + "\n" + "\n".join(lines)
        except Exception as e:
            return f"(glob error: {e})"

    @staticmethod
    async def _tool_grep(args: dict) -> str:
        """Search files — multiline, output modes, type filter, pagination."""
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        file_glob = args.get("glob", "")
        file_type = args.get("type", "")
        context = min(int(args.get("context", 2)), 5)
        multiline = args.get("multiline", False)
        output_mode = args.get("output_mode", "content")
        skip = int(args.get("offset", 0))
        head_limit = int(args.get("head_limit", 50))
        if not pattern:
            return "(no pattern provided)"

        import shutil
        rg = shutil.which("rg")

        # Build command based on output mode
        if rg:
            cmd = [rg, "--color=never"]
            if output_mode == "files_with_matches":
                cmd += ["--files-with-matches"]
            elif output_mode == "count":
                cmd += ["--count"]
            else:
                cmd += ["--no-heading", "--line-number", f"--context={context}",
                        f"--max-count={head_limit + skip}"]
            if multiline:
                cmd += ["-U", "--multiline-dotall"]
            if file_glob:
                cmd += [f"--glob={file_glob}"]
            if file_type:
                cmd += [f"--type={file_type}"]
            cmd += [pattern, path]
        else:
            # Fallback to grep (no multiline or type support)
            cmd = ["grep", "-rn", "--color=never"]
            if output_mode == "files_with_matches":
                cmd += ["-l"]
            elif output_mode == "count":
                cmd += ["-c"]
            else:
                cmd += [f"--context={context}", "-m", str(head_limit + skip)]
            if file_glob:
                cmd += [f"--include={file_glob}"]
            cmd += [pattern, path]

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                ),
                timeout=15,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = stdout.decode(errors="replace").strip()
            if not output:
                return f"(no matches for /{pattern}/ in {path})"

            all_lines = output.splitlines()

            # ── files_with_matches mode ───────────────────────────────
            if output_mode == "files_with_matches":
                files = all_lines[skip:skip + head_limit]
                total = len(all_lines)
                header = f"# {total} files match /{pattern}/"
                if skip > 0:
                    header += f" (offset {skip})"
                if total > skip + head_limit:
                    header += f" — showing {len(files)}"
                return header + "\n" + "\n".join(f"  {f}" for f in files)

            # ── count mode ────────────────────────────────────────────
            if output_mode == "count":
                entries = all_lines[skip:skip + head_limit]
                total_matches = 0
                for entry in entries:
                    if ":" in entry:
                        try:
                            total_matches += int(entry.rsplit(":", 1)[1])
                        except ValueError:
                            pass
                header = f"# {total_matches} matches across {len(entries)} file(s)"
                return header + "\n" + "\n".join(f"  {e}" for e in entries)

            # ── content mode (default) ────────────────────────────────
            # Extract match entries (groups separated by --)
            match_lines = [ln for ln in all_lines if ln and not ln.startswith("--")]
            files_seen = set()
            for ln in match_lines:
                if ":" in ln:
                    files_seen.add(ln.split(":")[0])

            # Apply offset/limit on entries
            if skip > 0 or head_limit < len(all_lines):
                # Split output into entry groups
                groups: list[list[str]] = []
                current: list[str] = []
                for ln in all_lines:
                    if ln == "--":
                        if current:
                            groups.append(current)
                            current = []
                    else:
                        current.append(ln)
                if current:
                    groups.append(current)
                selected = groups[skip:skip + head_limit]
                output = "\n--\n".join("\n".join(g) for g in selected)

            header = f"# {len(match_lines)} matches in {len(files_seen)} file(s)"

            # Truncate at match boundary
            if len(output) > 4000:
                lines = output.splitlines()
                truncated = []
                total_len = 0
                for line in lines:
                    if total_len + len(line) > 3800:
                        break
                    truncated.append(line)
                    total_len += len(line) + 1
                output = "\n".join(truncated)
                remaining = len(match_lines) - len([ln for ln in truncated if ln and not ln.startswith("--")])
                output += f"\n... ({remaining} more matches)"

            return header + "\n" + output
        except asyncio.TimeoutError:
            return "(search timed out after 15s)"
        except Exception as e:
            return f"(grep error: {e})"

    # Per-participant persistent working directory and background tasks
    _agent_cwd: dict[str, str] = {}   # {participant: cwd_path}
    _bg_tasks: dict[str, dict] = {}   # {task_id: {proc, command, started, participant}}

    async def _tool_bash(self, args: dict, participant_name: str = "") -> str:
        """Execute a command — persistent cwd, background support, structural safety."""
        command = args.get("command", "")
        timeout = min(int(args.get("timeout", 30)), 60)
        background = args.get("background", False)
        if not command:
            return "(no command provided)"

        # ── Safety checks ─────────────────────────────────────────────
        import shlex
        normalized = " ".join(command.split())
        lower = normalized.lower()

        if any(lower.startswith(p) for p in ("sudo ", "su ", "su\n", "doas ")):
            return "(blocked: privilege escalation)"

        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        if tokens and tokens[0] in ("rm", "/bin/rm", "/usr/bin/rm"):
            flags = set()
            paths = []
            for t in tokens[1:]:
                if t.startswith("-"):
                    flags.update(c for c in t[1:] if c.isalpha())
                    if t in ("--recursive", "--force", "--no-preserve-root"):
                        flags.add(t)
                else:
                    paths.append(t)
            is_recursive = "r" in flags or "R" in flags or "--recursive" in flags
            is_force = "f" in flags or "--force" in flags
            has_root = any(p in ("/", "/*", "/.", "/..") for p in paths)
            if is_recursive and is_force and has_root:
                return "(blocked: recursive forced deletion of root)"

        bomb_patterns = [
            ":(){ :", "|:&", "fork()", "./$0|./$0",
            "dd if=/dev/zero of=/dev/sd", "mkfs.", "> /dev/sd",
            "chmod -R 777 /", "chown -R",
        ]
        for bp in bomb_patterns:
            if bp in normalized:
                return "(blocked: dangerous pattern detected)"

        # Block encoding/indirection bypasses (base64 decode | bash, hex, python os.system)
        bypass_patterns = [
            r"base64\s.*\|\s*(ba)?sh",            # base64 -d | bash
            r"printf\s+['\"]\\x",                  # printf '\x72\x6d' hex encoding
            r"python[23]?\s+-c\s+.*os\.system",    # python -c "os.system(...)"
            r"python[23]?\s+-c\s+.*subprocess",     # python -c "subprocess..."
            r"perl\s+-e\s+.*system",                # perl -e 'system(...)'
            r"ruby\s+-e\s+.*system",                # ruby -e 'system(...)'
            r"\$\(\s*echo\s+.*\|\s*(ba)?sh",       # $(echo ... | bash)
            r"wget\s.*\|\s*(ba)?sh",               # wget ... | bash
            r"curl\s.*\|\s*(ba)?sh",               # curl ... | bash
        ]
        for bp in bypass_patterns:
            if re.search(bp, normalized, re.IGNORECASE):
                return "(blocked: encoding/indirection bypass detected)"

        if re.search(r'\beval\s', command) or re.search(r'\bexec\s', command):
            inner = command.split("eval", 1)[-1] if "eval" in command else ""
            inner += command.split("exec", 1)[-1] if "exec" in command else ""
            if any(d in inner.lower() for d in ("rm ", "dd ", "mkfs", "/dev/")):
                return "(blocked: eval/exec wrapping dangerous command)"

        # ── Working directory persistence ─────────────────────────────
        key = participant_name or "_global"
        cwd = RoomManager._agent_cwd.get(key, os.getcwd())

        # Detect cd commands and update persistent cwd
        cd_match = re.match(r'^cd\s+(.+?)(?:\s*&&|\s*;|\s*$)', command)
        if cd_match:
            target = cd_match.group(1).strip().strip("'\"")
            target_path = Path(target).expanduser()
            if not target_path.is_absolute():
                target_path = Path(cwd) / target_path
            target_path = target_path.resolve()
            if target_path.is_dir():
                RoomManager._agent_cwd[key] = str(target_path)
                cwd = str(target_path)
                # If bare "cd <dir>", just update cwd
                if re.match(r'^cd\s+\S+\s*$', command):
                    return f"(cwd: {cwd})"

        # ── Build subprocess ──────────────────────────────────────────
        import shutil
        env = os.environ.copy()
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

        use_unshare = shutil.which("unshare") is not None
        if use_unshare:
            shell_cmd = ["unshare", "--net", "--", "bash", "-c", command]
        else:
            # Fail-closed: without network isolation, only allow safe commands
            safe_prefixes = (
                "ls", "cat", "head", "tail", "wc", "sort", "uniq", "cut",
                "grep", "find", "file", "stat", "du", "df", "echo", "printf",
                "pwd", "date", "whoami", "uname", "which", "env", "id",
                "diff", "md5sum", "sha256sum", "tr", "sed", "awk",
                "python", "python3", "pip", "rg", "fd", "jq",
            )
            first_cmd = tokens[0] if tokens else ""
            base_cmd = os.path.basename(first_cmd)
            if base_cmd not in safe_prefixes:
                return f"(blocked: '{base_cmd}' not allowed without network isolation — unshare not available)"
            shell_cmd = ["bash", "-c", command]

        # ── Background execution ──────────────────────────────────────
        if background:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env, cwd=cwd,
                )
                from datetime import datetime
                task_id = f"bg-{proc.pid}"
                RoomManager._bg_tasks[task_id] = {
                    "proc": proc, "command": command,
                    "started": datetime.now().isoformat(),
                    "participant": participant_name,
                }
                return f"(started background task {task_id}: {command[:60]})"
            except Exception as e:
                return f"(background start error: {e})"

        # ── Foreground execution ──────────────────────────────────────
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env, cwd=cwd,
                ),
                timeout=5,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                return f"(command killed after {timeout}s timeout)"

            out = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()
            parts = []
            if out:
                parts.append(out)
            if err:
                parts.append(f"[stderr] {err}")
            if proc.returncode != 0:
                parts.append(f"[exit code: {proc.returncode}]")
            result = "\n".join(parts) if parts else "(no output)"
            if len(result) > 4000:
                lines = result.splitlines()
                truncated = []
                total_len = 0
                for line in lines:
                    if total_len + len(line) > 3800:
                        break
                    truncated.append(line)
                    total_len += len(line) + 1
                result = "\n".join(truncated) + "\n... (truncated)"
            return result
        except asyncio.TimeoutError:
            return "(failed to start command within 5s)"
        except Exception as e:
            return f"(bash error: {e})"

    # ------------------------------------------------------------------
    # Todo tracking (per-participant, in-memory)
    # ------------------------------------------------------------------

    _agent_todos: dict = {}  # class-level: {participant_name: [{task, priority, done}]}

    def _tool_todo_add(self, args: dict, name: str) -> str:
        key = name or "anonymous"
        if key not in RoomManager._agent_todos:
            RoomManager._agent_todos[key] = []
        task = args.get("task", "")
        priority = args.get("priority", "medium")
        RoomManager._agent_todos[key].append({"task": task, "priority": priority, "done": False})
        n = len(RoomManager._agent_todos[key])
        return f"(added todo #{n}: {task} [{priority}])"

    def _tool_todo_list(self, name: str) -> str:
        key = name or "anonymous"
        todos = RoomManager._agent_todos.get(key, [])
        if not todos:
            return "(no todos)"
        lines = []
        for i, t in enumerate(todos, 1):
            mark = "x" if t["done"] else " "
            lines.append(f"  [{mark}] {i}. [{t['priority']}] {t['task']}")
        return "\n".join(lines)

    def _tool_todo_done(self, args: dict, name: str) -> str:
        key = name or "anonymous"
        todos = RoomManager._agent_todos.get(key, [])
        num = int(args.get("number", 0))
        if num < 1 or num > len(todos):
            return f"(invalid todo number: {num})"
        todos[num - 1]["done"] = True
        return f"(completed: {todos[num - 1]['task']})"

    # ------------------------------------------------------------------
    # Backend dispatch + tool-use loop
    # ------------------------------------------------------------------

    async def _send_to_backend(self, participant: dict, message: str,
                                system_prompt: Optional[str] = None,
                                tools: Optional[list] = None,
                                files: Optional[list[str]] = None) -> str:
        """Send a message to a participant's backend, returning the raw reply."""
        name = participant["name"]
        backend = participant["backend"]
        sid = participant.get("session_id")

        if backend == "claude":
            full_prompt = f"{system_prompt}\n\n{message}" if system_prompt else message
            return await self._run_claude_p(full_prompt, files=files, model=participant.get("model"))

        elif backend == "local":
            base_url = participant.get("base_url") or participant.get("endpoint")
            model = participant.get("model", "")
            if not base_url:
                nodes = await asyncio.get_event_loop().run_in_executor(None, GpuNodeDiscovery.discover)
                if not nodes:
                    return "[error: no local model endpoint found]"
                node = nodes[0]
                base_url = node["base_url"]
                if not model and node["models"]:
                    model = node["models"][0]
            if base_url:
                msg_with_files = _embed_files_in_prompt(message, files or [])
                if sid and sid in self.local.sessions:
                    return await self.local.send_message(msg_with_files, sid, system_prompt=system_prompt)
                else:
                    tmp = f"room-{participant.get('_room_id', 'r')}-{name.lower().replace(' ', '-')}"
                    if tmp not in self.local.sessions:
                        self.local.start_session(tmp, model=model or "default", endpoint=base_url)
                    participant["session_id"] = tmp
                    return await self.local.send_message(msg_with_files, tmp, system_prompt=system_prompt)
            return "[error: no endpoint]"

        elif backend == "codex":
            full_prompt = f"{system_prompt}\n\n{message}" if system_prompt else message
            full_prompt = _embed_files_in_prompt(full_prompt, files or [])
            if sid and sid in self.codex.sessions:
                return await self.codex.send_message(full_prompt, sid)
            return await self.codex.run_task(full_prompt)

        else:  # opencode
            full_prompt = f"{system_prompt}\n\n{message}" if system_prompt else message
            if sid and sid in self.opencode.sessions:
                return await self.opencode.send_message(full_prompt, sid, files=files, _raw=True)
            tmp = f"room-{participant.get('_room_id', 'r')}-{name.lower().replace(' ', '-')}"
            await self.opencode.start_session(tmp, model=participant.get("model"))
            reply = await self.opencode.send_message(full_prompt, tmp, files=files, _raw=True)
            self.opencode.end_session(tmp)
            return reply

    async def _run_claude_p(self, prompt: str, timeout: int = 300,
                             files: Optional[list[str]] = None,
                             model: Optional[str] = None) -> str:
        """Run `claude -p` with prompt via stdin and return the text response."""
        global CLAUDE_BIN
        if not CLAUDE_BIN:
            CLAUDE_BIN = shutil.which("claude")
        if not CLAUDE_BIN:
            return "[error: claude binary not found]"
        try:
            # Embed files inline — avoids CLAUDE_CODE_SESSION_ACCESS_TOKEN requirement
            full_prompt = _embed_files_in_prompt(prompt, files or [])
            cmd = [CLAUDE_BIN, "-p"]
            if model:
                cmd.extend(["--model", model])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(full_prompt.encode()), timeout=timeout
            )
            if proc.returncode == 0 and stdout:
                return stdout.decode(errors="replace").strip()
            return f"[error: {stderr.decode(errors='replace').strip() or 'empty response'}]"
        except asyncio.TimeoutError:
            return f"[error: claude -p timed out after {timeout}s]"
        except Exception as e:
            return f"[error: {e}]"

    async def _participant_respond(self, room: DiscussionRoom, participant: dict) -> dict:
        """Get one participant's response with optional tool-use loop."""
        name = participant["name"]
        soul = self._parse_soul(participant)
        participant["_room_id"] = room.id

        # Seed realm on first turn if empty
        if soul and soul.realm and SoulClient.is_available():
            count = room.turn_counts.get(name, 0)
            if count == 0:
                existing = SoulClient.recall("identity role expertise", limit=1, realm=soul.realm)
                if not existing or len(existing.strip()) < 20:
                    SoulClient.remember(
                        content=f"I am {name}. {soul.system_prompt[:300]}",
                        kind="identity",
                        tags="identity,role,seed",
                        confidence=0.95,
                        realm=soul.realm,
                    )
                    SoulClient.remember(
                        content=f"Discussion topic: {room.topic}",
                        kind="episode",
                        tags="topic,room,seed",
                        confidence=0.8,
                        realm=soul.realm,
                    )

        # Check per-participant round limits
        if soul and soul.max_rounds > 0:
            count = room.turn_counts.get(name, 0)
            if count >= soul.max_rounds:
                return {"name": name, "content": "(max rounds reached — sitting out)",
                        "ts": datetime.now().isoformat()}

        system_prompt, user_msg = self._build_thread_context(room, participant)
        max_tool_turns = soul.max_tool_turns if soul and soul.tools else 0
        realm = soul.realm if soul else None
        allowed_tools = set(soul.tools) if soul else set()

        room_files = room.files or None
        reply = ""
        for turn in range(max_tool_turns + 1):
            try:
                if turn == 0:
                    reply = await self._send_to_backend(participant, user_msg, system_prompt, files=room_files)
                else:
                    reply = await self._send_to_backend(participant, user_msg, system_prompt)
            except Exception as e:
                reply = f"[error: {e}]"
                break

            # Check for tool call in the response
            tool_req = self._extract_tool_call(reply)
            if tool_req is None or turn >= max_tool_turns:
                break

            # Validate tool is allowed
            if tool_req["tool"] not in allowed_tools:
                break

            # Execute the tool
            tool_result = await self._execute_agent_tool(
                tool_req["tool"], tool_req["args"], realm=realm,
                participant_name=name,
            )

            # Inject result and re-prompt
            user_msg = (
                f"{reply}\n\n"
                f"<tool_result>\n{tool_result[:2000]}\n</tool_result>\n\n"
                f"Continue. You may make another tool call or provide your final response."
            )

        # Extract final response if wrapped in tags, otherwise use raw reply
        final = self._extract_final_response(reply) or reply

        # Store the participant's contribution as a memory in their realm
        if soul and soul.realm and SoulClient.is_available() and len(final) > 50:
            SoulClient.remember(
                content=f"[room:{room.id}] My contribution on '{room.topic[:80]}':\n{final[:500]}",
                kind="episode",
                tags=f"room,discussion,{room.id}",
                confidence=0.7,
                realm=soul.realm,
            )

        # Track round count
        room.turn_counts[name] = room.turn_counts.get(name, 0) + 1

        return {"name": name, "content": final, "ts": datetime.now().isoformat()}

    # ------------------------------------------------------------------
    # Challenge round support
    # ------------------------------------------------------------------

    def _extract_claims(self, messages: list[dict]) -> list[str]:
        """Extract substantive claims from recent messages for challenge rounds."""
        claims = []
        seen = set()
        # Match full sentences containing assertion verbs
        assertion_re = re.compile(
            r'([A-Z][^.!?\n]{20,}(?:is |are |should |must |requires |causes |'
            r'leads to |results in |provides |ensures |enables |produces |'
            r'can be |will |has been |have been )[^.!?\n]{10,}[.!?])',
        )
        # Skip lines that are headers, bullet markers, or code blocks
        skip_re = re.compile(r'^(?:\s*[-*#>|`]|```|\|)')
        for msg in messages:
            if msg["name"] in ("TOPIC", "CONTEXT", "MODERATOR"):
                continue
            for line in msg["content"].split("\n"):
                if skip_re.match(line):
                    continue
                for m in assertion_re.finditer(line):
                    claim = m.group(1).strip()
                    # Deduplicate by first 50 chars
                    key = claim[:50].lower()
                    if key not in seen and 40 < len(claim) < 300:
                        seen.add(key)
                        claims.append(f"[{msg['name']}]: {claim}")
        # Return top 5 most substantive (longest) claims
        claims.sort(key=lambda c: len(c), reverse=True)
        return claims[:5]

    async def run_rounds(self, room_id: str, rounds: int = 2,
                          challenge: bool = False) -> str:
        """Run N rounds of async discussion — all participants respond in parallel each round."""
        if room_id not in self.rooms:
            return f"Room '{room_id}' not found."
        room = self.rooms[room_id]
        room.challenge_mode = challenge

        for round_num in range(1, rounds + 1):
            # Inject challenge prompt between rounds if enabled
            if challenge and round_num > 1:
                prev_round_msgs = room.messages[-(len(room.participants)):]
                claims = self._extract_claims(prev_round_msgs)
                if claims:
                    challenge_text = (
                        "**[Challenge Round]** The following key claims were made in the previous round. "
                        "Each participant MUST: (1) identify at least one claim you disagree with or find incomplete, "
                        "(2) provide specific evidence or reasoning for your disagreement, "
                        "(3) propose a concrete refinement. Do NOT simply agree with everything.\n\n"
                        + "\n".join(f"- {c}" for c in claims)
                    )
                    room.messages.append({
                        "name": "MODERATOR",
                        "content": challenge_text,
                        "ts": datetime.now().isoformat(),
                    })

            # Filter participants who haven't hit their round limit
            active = []
            for p in room.participants:
                soul = self._parse_soul(p)
                if soul and soul.max_rounds > 0:
                    if room.turn_counts.get(p["name"], 0) >= soul.max_rounds:
                        continue
                active.append(p)

            if not active:
                break

            # Detect if participants use different local models on the same endpoint.
            # If so, run sequentially to avoid GPU model-loading contention.
            local_models = set()
            local_endpoints = set()
            for p in active:
                if p.get("backend") == "local":
                    local_models.add(p.get("model", ""))
                    ep = p.get("base_url") or p.get("endpoint") or ""
                    if p.get("session_id") and p["session_id"] in self.local.sessions:
                        s = self.local.sessions[p["session_id"]]
                        local_models.add(s.model)
                        ep = s.endpoint
                    local_endpoints.add(ep)

            needs_sequential = len(local_models) > 1 and len(local_endpoints) <= 1

            if needs_sequential:
                # Sequential: different models on same GPU — avoid model swap thrashing
                responses = []
                for p in active:
                    resp = await self._participant_respond(room, p)
                    responses.append(resp)
            else:
                # Parallel: same model or different endpoints
                coros = [self._participant_respond(room, p) for p in active]
                responses = await asyncio.gather(*coros)

            for resp in responses:
                room.messages.append(resp)

        return self.read(room_id)


# MCP Server setup
bridge = OpenCodeBridge()
codex_bridge = CodexBridge()
local_bridge = LocalModelBridge()
orchestrator = Orchestrator(bridge, codex_bridge)
rooms = RoomManager(bridge, codex_bridge, local_bridge)
server = Server("chitta-bridge")

# Checked once at startup — used to suppress tools for missing backends
_HAS_CODEX = find_codex() is not None
_HAS_OPENCODE = find_opencode() is not None

# Tools hidden from tools/list to save context tokens.
# All are still callable directly or via the `advanced` gateway.
HIDDEN_TOOLS = {
    # Session lifecycle — prefer reuse over start/end
    "opencode_start", "opencode_end", "opencode_end_all",
    "opencode_history", "opencode_model", "opencode_agent", "opencode_variant",
    "opencode_config", "opencode_configure", "opencode_export", "opencode_health",
    "codex_start", "codex_end", "codex_end_all",
    "codex_switch", "codex_sessions", "codex_history",
    "codex_model", "codex_config", "codex_configure",
    "codex_review", "codex_rescue", "codex_health",
    # Local model management
    "local_start", "local_end", "local_switch",
    "local_sessions", "local_history", "local_models",
    "local_discover", "local_health", "local_discuss",
    # Orchestration (complex, rarely needed)
    "multi_consult", "agent_chain", "delegate_codex", "parallel_agents",
    # Rooms (multi-agent discussion)
    "room_create", "room_run", "room_synthesize", "room_read",
    # Status/health
    "soul_status",
}


def handle_advanced(arguments: dict) -> str:
    """Gateway to hidden chitta-bridge tools.

    Actions:
    - list: Show all hidden tools by category
    - call a hidden tool: {"tool": "<name>", "arguments": {...}}

    Examples:
      {"action": "list"}
      {"tool": "opencode_start", "arguments": {"session_id": "main"}}
    """
    tool_name = arguments.get("tool", "")
    action = arguments.get("action", "")

    if tool_name:
        if tool_name not in HIDDEN_TOOLS:
            return f"Unknown hidden tool: {tool_name}\nUse action='list' to see available tools."

    # List hidden tools by category
    categories = {
        "Session lifecycle (opencode)": [t for t in sorted(HIDDEN_TOOLS) if t.startswith("opencode_")],
        "Session lifecycle (codex)":    [t for t in sorted(HIDDEN_TOOLS) if t.startswith("codex_")],
        "Local models":                 [t for t in sorted(HIDDEN_TOOLS) if t.startswith("local_")],
        "Orchestration":                [t for t in sorted(HIDDEN_TOOLS) if t in {"multi_consult", "agent_chain", "delegate_codex", "parallel_agents"}],
        "Rooms":                        [t for t in sorted(HIDDEN_TOOLS) if t.startswith("room_")],
        "Misc":                         [t for t in sorted(HIDDEN_TOOLS) if not any(t.startswith(p) for p in ("opencode_", "codex_", "local_", "room_")) and t not in {"multi_consult", "agent_chain", "delegate_codex", "parallel_agents"}],
    }
    lines = ["Hidden chitta-bridge tools (callable via advanced gateway or directly):\n"]
    for cat, tools in categories.items():
        if tools:
            lines.append(f"{cat}:")
            lines.extend(f"  • {t}" for t in tools)
    lines.append(f"\nTotal: {len(HIDDEN_TOOLS)} hidden tools")
    lines.append('\nUsage: {"tool": "<name>", "arguments": {...}}')
    return "\n".join(lines)


@server.list_tools()
async def list_tools():
    _tools = [
        Tool(
            name="opencode_start",
            description="Start a new discussion session with OpenCode",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for this session"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: openai/gpt-5.2-codex)"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent to use: plan, build, explore, general (default: plan)"
                    },
                    "variant": {
                        "type": "string",
                        "description": "Reasoning effort: minimal, low, medium, high, xhigh, max"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="opencode_discuss",
            description="Send a message to OpenCode. Auto-detects domain; use 'domain' to override.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or question"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to attach for context"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Hint the domain of expertise (e.g., 'security', 'metagenomics', 'quantitative finance')"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="opencode_plan",
            description="Start a planning discussion with the plan agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to plan"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant file paths"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="opencode_review",
            description="Review code for issues. Accepts file paths (space/comma separated) or code snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_or_file": {
                        "type": "string",
                        "description": "Code snippet, file path, or multiple file paths (space/comma separated)"
                    },
                    "focus": {
                        "type": "string",
                        "description": "What to focus on (default: correctness, efficiency, bugs)"
                    }
                },
                "required": ["code_or_file"]
            }
        ),
        Tool(
            name="opencode_model",
            description="Change the model for the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "New model"}
                },
                "required": ["model"]
            }
        ),
        Tool(
            name="opencode_agent",
            description="Change the agent for the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "New agent (plan, build, explore, general)"}
                },
                "required": ["agent"]
            }
        ),
        Tool(
            name="opencode_variant",
            description="Change the model variant (reasoning effort) for the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "variant": {"type": "string", "description": "New variant: minimal, low, medium, high, xhigh, max"}
                },
                "required": ["variant"]
            }
        ),
        Tool(
            name="opencode_history",
            description="Get conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID (default: active session)"},
                    "last_n": {"type": "integer", "description": "Number of messages (default: 20)"}
                }
            }
        ),
        Tool(
            name="opencode_sessions",
            description="List all sessions",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_switch",
            description="Switch to a different session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to switch to"}
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="opencode_end",
            description="End the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to end (default: active)"}
                }
            }
        ),
        Tool(
            name="opencode_config",
            description="Get current configuration (default model, agent, variant)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_configure",
            description="Set default model, agent, and/or variant (persisted)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Default model"},
                    "agent": {"type": "string", "description": "Default agent"},
                    "variant": {"type": "string", "description": "Default variant: minimal, low, medium, high, xhigh, max"}
                }
            }
        ),
        Tool(
            name="opencode_export",
            description="Export a session transcript as markdown or JSON",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to export (default: active)"},
                    "format": {"type": "string", "description": "Export format: markdown or json (default: markdown)", "enum": ["markdown", "json"]}
                }
            }
        ),
        Tool(
            name="opencode_health",
            description="Health check: returns server status, session count, and uptime",
            inputSchema={"type": "object", "properties": {}}
        ),
        # Codex tools
        Tool(
            name="codex_start",
            description="Start a new Codex session (OpenAI's agentic coding assistant)",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for this session"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: o3). Options: o3, o4-mini, gpt-4.1"
                    },
                    "sandbox": {
                        "type": "string",
                        "description": "Sandbox mode: read-only, workspace-write, danger-full-access (default: workspace-write)"
                    },
                    "full_auto": {
                        "type": "boolean",
                        "description": "Enable full-auto mode for low-friction execution (default: true)"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for Codex (default: current directory)"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="codex_discuss",
            description="Send a message to Codex. Use for coding tasks, file operations, debugging.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or coding task"
                    },
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Image file paths to attach"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="codex_run",
            description="Run a one-off Codex task (stateless). Returns result + session ID for resuming.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The coding task to perform"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: current)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: o3)"
                    },
                    "full_auto": {
                        "type": "boolean",
                        "description": "Enable full-auto mode (default: true)"
                    },
                    "effort": {
                        "type": "string",
                        "description": "Effort: low, medium, high, xhigh"
                    },
                    "sandbox": {
                        "type": "string",
                        "enum": ["read-only", "workspace-write", "danger-full-access"],
                        "description": "Sandbox: read-only, workspace-write, danger-full-access"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="codex_review",
            description="Run Codex code review. adversarial mode pressure-tests design decisions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "working_dir": {
                        "type": "string",
                        "description": "Repository directory to review (default: current)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for review"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "adversarial"],
                        "description": "Review mode: 'normal' (default) or 'adversarial' (challenges design, architecture, tradeoffs)"
                    },
                    "focus": {
                        "type": "string",
                        "description": "For adversarial mode: specific risk area to challenge (e.g. 'auth flow', 'race conditions')"
                    },
                    "base": {
                        "type": "string",
                        "description": "Git ref to compare against (e.g. 'main', 'HEAD~3'). Reviews only changes since that ref."
                    },
                    "effort": {
                        "type": "string",
                        "description": "Effort: low, medium, high, xhigh"
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run in background and return job ID immediately (default: false)"
                    },
                    "sandbox": {
                        "type": "string",
                        "enum": ["read-only", "workspace-write", "danger-full-access"],
                        "description": "Sandbox: read-only, workspace-write, danger-full-access"
                    }
                }
            }
        ),
        Tool(
            name="codex_rescue",
            description="Delegate a long task to Codex with background execution and session resume.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to delegate to Codex (investigate, fix, implement)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: configured default)"
                    },
                    "effort": {
                        "type": "string",
                        "description": "Effort: low, medium, high, xhigh"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: current)"
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run in background (default: true)"
                    },
                    "resume_from": {
                        "type": "string",
                        "description": "Codex session ID to resume"
                    },
                    "fresh": {
                        "type": "boolean",
                        "description": "Start fresh — do not auto-resume the latest completed job (default: false)"
                    },
                    "sandbox": {
                        "type": "string",
                        "enum": ["read-only", "workspace-write", "danger-full-access"],
                        "description": "Sandbox: read-only, workspace-write, danger-full-access"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="codex_model",
            description="Change the model for the current Codex session",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "New model (o3, o4-mini, gpt-4.1)"}
                },
                "required": ["model"]
            }
        ),
        Tool(
            name="codex_history",
            description="Get Codex conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of messages (default: 20)"}
                }
            }
        ),
        Tool(
            name="codex_sessions",
            description="List all Codex sessions",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="codex_switch",
            description="Switch to a different Codex session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to switch to"}
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="codex_end",
            description="End the current Codex session",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="codex_config",
            description="Get current Codex configuration",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="codex_configure",
            description="Set default Codex model and sandbox mode (persisted)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Default model (o3, o4-mini, gpt-4.1)"},
                    "sandbox": {"type": "string", "description": "Default sandbox mode"}
                }
            }
        ),
        Tool(
            name="codex_health",
            description="Codex health check: returns status and installation info",
            inputSchema={"type": "object", "properties": {}}
        ),
        # Orchestration tools
        Tool(
            name="multi_consult",
            description="Fan-out a question to multiple backends (OpenCode + Codex) in parallel, optionally synthesize results",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question/task to send to all backends"
                    },
                    "backends": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["opencode", "codex"]},
                        "description": "Backends to consult (default: both)"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to attach (OpenCode only)"
                    },
                    "synthesize": {
                        "type": "boolean",
                        "description": "Whether to synthesize results into unified response (default: true)"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="agent_chain",
            description="Execute agent steps sequentially, passing results forward (e.g. OpenCode → Codex → OpenCode).",
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "backend": {"type": "string", "enum": ["opencode", "codex"]},
                                "task": {"type": "string", "description": "Task prompt. Use {previous} to include result from previous step"},
                                "model": {"type": "string", "description": "Optional model override"},
                                "agent": {"type": "string", "description": "Optional agent override (OpenCode only)"}
                            },
                            "required": ["backend", "task"]
                        },
                        "description": "List of steps to execute sequentially"
                    }
                },
                "required": ["steps"]
            }
        ),
        Tool(
            name="delegate_codex",
            description="Delegate to Codex, optionally send result to OpenCode for review.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task for Codex to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for Codex"
                    },
                    "model": {
                        "type": "string",
                        "description": "Codex model to use"
                    },
                    "return_to_opencode": {
                        "type": "boolean",
                        "description": "Send Codex result to OpenCode for review (default: false)"
                    },
                    "opencode_followup": {
                        "type": "string",
                        "description": "Custom prompt for OpenCode followup"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="parallel_agents",
            description="Run multiple agent tasks in parallel across backends. All tasks run concurrently.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "backend": {"type": "string", "enum": ["opencode", "codex"]},
                                "task": {"type": "string"},
                                "name": {"type": "string", "description": "Optional name for this task"},
                                "model": {"type": "string", "description": "Optional model override"}
                            },
                            "required": ["backend", "task"]
                        },
                        "description": "List of tasks to run in parallel"
                    }
                },
                "required": ["tasks"]
            }
        ),
        Tool(
            name="room_create",
            description="Create a multi-agent discussion room. Participants post async and see each other's messages. Each can have a soul (system_prompt, tools, realm).",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Unique room identifier"},
                    "topic": {"type": "string", "description": "The discussion topic or opening question"},
                    "participants": {
                        "type": "string",
                        "description": 'JSON array: [{"name":"...","backend":"claude|opencode|codex|local","session_id":"...","model":"...",'
                                       '"soul":{"system_prompt":"...","realm":"...","tools":["recall","web_search"],'
                                       '"max_tool_turns":3,"challenge_bias":0.5,"max_rounds":0}}]. backend defaults to "claude" if omitted — set explicitly to avoid unexpected API spend.'
                    },
                    "files": {
                        "type": "string",
                        "description": 'JSON array of file or directory paths to attach to all participants: ["/path/to/file.py", "/path/to/dir"]. Directories are expanded recursively. Files are passed via --file to opencode/claude, embedded inline for codex/local.'
                    }
                },
                "required": ["room_id", "topic", "participants"]
            }
        ),
        Tool(
            name="room_run",
            description="Run N rounds in a room. Participants respond in parallel. challenge=true injects adversarial claims between rounds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to run"},
                    "rounds": {"type": "integer", "description": "Number of discussion rounds (default: 2)"},
                    "challenge": {"type": "boolean", "description": "Enable challenge rounds — auto-extract claims and ask participants to verify/challenge them (default: false)"},
                    "prompt": {"type": "string", "description": "Discussion prompt to inject as a MODERATOR message before running rounds"},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "File paths to attach to the room for this run"}
                },
                "required": ["room_id"]
            }
        ),
        Tool(
            name="room_synthesize",
            description="Synthesize a room's transcript into consensus, disagreements, and best answer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to synthesize"},
                    "synthesizer": {
                        "type": "string",
                        "description": 'Optional JSON: {"name":"...","backend":"claude|opencode|codex|local","model":"..."}. Defaults to the backend used by room participants (inferred); falls back to claude if mixed.'
                    }
                },
                "required": ["room_id"]
            }
        ),
        Tool(
            name="room_read",
            description="Read the full transcript of a discussion room.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to read"}
                },
                "required": ["room_id"]
            }
        ),
        # Local model tools (Ollama / vLLM on GPU nodes)
        Tool(
            name="local_discover",
            description="Discover GPU nodes running Ollama/vLLM. Checks cache files, Slurm jobs, CHITTA_BRIDGE_GPU_NODES, and localhost.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="local_start",
            description="Start a session with a local model (Ollama/vLLM) on a GPU node.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Unique session identifier"},
                    "model": {"type": "string", "description": "Model name (e.g. llama3.3:70b, qwen3:30b-a3b)"},
                    "endpoint": {"type": "string", "description": "Base URL of the OpenAI-compatible server (e.g. http://node:11434/v1). Auto-discovered if omitted."}
                },
                "required": ["session_id", "model"]
            }
        ),
        Tool(
            name="local_discuss",
            description="Send a message to the active local model session and get a response.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to send"},
                    "session_id": {"type": "string", "description": "Session ID (defaults to active session)"},
                    "system_prompt": {"type": "string", "description": "Optional system prompt to prepend"}
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="local_sessions",
            description="List all active local model sessions.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="local_switch",
            description="Switch the active local model session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to activate"}
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="local_end",
            description="End a local model session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to end (defaults to active)"}
                }
            }
        ),
        Tool(
            name="local_history",
            description="Show conversation history for a local model session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID (defaults to active)"},
                    "last_n": {"type": "integer", "description": "Number of messages to show (default: 20)"}
                }
            }
        ),
        Tool(
            name="local_models",
            description="List models available at a local model endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "Base URL (e.g. http://node:11434/v1). Auto-discovers if omitted."}
                }
            }
        ),
        Tool(
            name="local_health",
            description="Health check for local model sessions.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_end_all",
            description="End all OpenCode sessions, or a specific list of named sessions. "
                        "Use exclude_model to keep sessions of one model and kill the rest.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of session IDs to end. Omit to target ALL sessions."
                    },
                    "exclude_model": {
                        "type": "string",
                        "description": "Keep sessions using this model; end all others. E.g. 'gpt-5.4'."
                    }
                }
            }
        ),
        Tool(
            name="codex_end_all",
            description="End all Codex sessions, or a specific list of named sessions. "
                        "Use exclude_model to keep sessions of one model and kill the rest.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of session IDs to end. Omit to target ALL sessions."
                    },
                    "exclude_model": {
                        "type": "string",
                        "description": "Keep sessions using this model; end all others."
                    }
                }
            }
        ),

        # ── Web Search ─────────────────────────────────────────────
        Tool(
            name="web_search",
            description="Search the web via DuckDuckGo. Returns titles, URLs, and snippets. No API key needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 8)",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="web_fetch",
            description="Fetch a web page and return its text (HTML stripped).",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 12000)",
                        "default": 12000
                    }
                },
                "required": ["url"]
            }
        ),

        # ── Soul Memory ────────────────────────────────────────────
        Tool(
            name="soul_recall",
            description="Recall memories from the soul (chittad).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="soul_remember",
            description="Store a memory in the soul (chittad).",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Memory content to store"
                    },
                    "kind": {
                        "type": "string",
                        "description": "Memory kind: episode, wisdom, correction, symbol (default: episode)",
                        "default": "episode"
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags for searchability (e.g. 'room,metagenomics,decay')",
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="soul_context",
            description="Get smart context (memories + code symbols + graph) for a task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task or question to get context for"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="soul_status",
            description="Check if the soul (chittad daemon) is available.",
            inputSchema={"type": "object", "properties": {}}
        ),

        # ── Token-efficient file editing ───────────────────────────
        Tool(
            name="file_patch",
            description=(
                "Apply a search-replace patch to a file. ~10-50x cheaper than Read+Edit "
                "because only the changed strings are sent, not the full file. "
                "Returns a compact summary: filename, line number, +added/-removed lines."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Absolute path to the file to patch"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact string to find (must match exactly once)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string (empty string to delete)"
                    }
                },
                "required": ["file", "old_str", "new_str"]
            }
        ),
        Tool(
            name="symbol_patch",
            description=(
                "Replace an entire function, class, or method by name — no old_str needed. "
                "Finds the symbol in the file and replaces its full definition. "
                "Supports Python (def/class) and brace-based languages (Rust, JS, Go, C). "
                "Returns compact summary: file::symbol, line, +added/-removed lines."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name (function, class, method) to replace"
                    },
                    "new_body": {
                        "type": "string",
                        "description": "Complete new definition (including def/fn/class line)"
                    }
                },
                "required": ["file", "symbol", "new_body"]
            }
        ),
    ]
    if not _HAS_CODEX:
        _tools = [t for t in _tools if not t.name.startswith("codex_")]
    if not _HAS_OPENCODE:
        _tools = [t for t in _tools if not t.name.startswith("opencode_")]
    _tools = [t for t in _tools if t.name not in HIDDEN_TOOLS]
    _tools.append(Tool(
        name="advanced",
        description=(
            "Gateway to hidden chitta-bridge tools (session lifecycle, orchestration, rooms, local models). "
            "Use action='list' to see all hidden tools, or tool='<name>' with arguments to call one."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Use 'list' to enumerate all hidden tools"
                },
                "tool": {
                    "type": "string",
                    "description": "Name of the hidden tool to call"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the tool"
                }
            }
        }
    ))
    return _tools


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "advanced":
            # List mode
            if "tool" not in arguments:
                result = handle_advanced(arguments)
            else:
                # Re-dispatch to the hidden tool
                hidden_name = arguments["tool"]
                hidden_args = arguments.get("arguments", {})
                if hidden_name not in HIDDEN_TOOLS:
                    result = f"Unknown hidden tool: {hidden_name}\nUse action='list' to see available tools."
                else:
                    return await call_tool(hidden_name, hidden_args)
        elif name == "opencode_models":
            result = await bridge.list_models(arguments.get("provider"))
        elif name == "opencode_agents":
            result = await bridge.list_agents()
        elif name == "opencode_start":
            result = await bridge.start_session(
                session_id=arguments["session_id"],
                model=arguments.get("model"),
                agent=arguments.get("agent"),
                variant=arguments.get("variant")
            )
        elif name == "opencode_discuss":
            result = await bridge.send_message(
                message=arguments["message"],
                files=arguments.get("files"),
                domain_override=arguments.get("domain"),
            )
        elif name == "opencode_plan":
            result = await bridge.plan(
                task=arguments["task"],
                files=arguments.get("files")
            )
        elif name == "opencode_brainstorm":
            result = await bridge.brainstorm(arguments["topic"])
        elif name == "opencode_review":
            result = await bridge.review_code(
                code_or_file=arguments["code_or_file"],
                focus=arguments.get("focus", "correctness, efficiency, and potential bugs")
            )
        elif name == "opencode_model":
            result = bridge.set_model(arguments["model"])
        elif name == "opencode_agent":
            result = bridge.set_agent(arguments["agent"])
        elif name == "opencode_variant":
            result = bridge.set_variant(arguments["variant"])
        elif name == "opencode_history":
            result = bridge.get_history(
                session_id=arguments.get("session_id"),
                last_n=arguments.get("last_n", 20)
            )
        elif name == "opencode_sessions":
            result = bridge.list_sessions()
        elif name == "opencode_switch":
            result = bridge.set_active(arguments["session_id"])
        elif name == "opencode_end":
            result = bridge.end_session(session_id=arguments.get("session_id"))
        elif name == "opencode_config":
            result = bridge.get_config()
        elif name == "opencode_configure":
            result = bridge.set_config(
                model=arguments.get("model"),
                agent=arguments.get("agent"),
                variant=arguments.get("variant")
            )
        elif name == "opencode_export":
            result = bridge.export_session(
                session_id=arguments.get("session_id"),
                export_format=arguments.get("format", "markdown")
            )
        elif name == "opencode_health":
            health = bridge.health_check()
            result = f"Status: {health['status']}\nSessions: {health['sessions']}\nUptime: {health['uptime']}s"
        # Codex tools
        elif name == "codex_start":
            result = await codex_bridge.start_session(
                session_id=arguments["session_id"],
                model=arguments.get("model"),
                sandbox=arguments.get("sandbox"),
                full_auto=arguments.get("full_auto", True),
                working_dir=arguments.get("working_dir")
            )
        elif name == "codex_discuss":
            result = await codex_bridge.send_message(
                message=arguments["message"],
                images=arguments.get("images")
            )
        elif name == "codex_run":
            result = await codex_bridge.run_task(
                task=arguments["task"],
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
                full_auto=arguments.get("full_auto", True),
                effort=arguments.get("effort"),
                sandbox=arguments.get("sandbox"),
            )
        elif name == "codex_review":
            result = await codex_bridge.review_code(
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
                mode=arguments.get("mode", "normal"),
                focus=arguments.get("focus"),
                base=arguments.get("base"),
                effort=arguments.get("effort"),
                background=arguments.get("background", False),
                sandbox=arguments.get("sandbox"),
            )
        elif name == "codex_rescue":
            result = await codex_bridge.rescue(
                task=arguments["task"],
                model=arguments.get("model"),
                effort=arguments.get("effort"),
                working_dir=arguments.get("working_dir"),
                background=arguments.get("background", True),
                resume_from=arguments.get("resume_from"),
                fresh=arguments.get("fresh", False),
                sandbox=arguments.get("sandbox"),
            )
        elif name == "codex_job_status":
            result = codex_bridge.job_status(arguments.get("job_id"))
        elif name == "codex_job_result":
            result = codex_bridge.job_result(arguments.get("job_id"))
        elif name == "codex_job_cancel":
            result = codex_bridge.job_cancel(arguments.get("job_id"))
        elif name == "codex_model":
            result = codex_bridge.set_model(arguments["model"])
        elif name == "codex_history":
            result = codex_bridge.get_history(last_n=arguments.get("last_n", 20))
        elif name == "codex_sessions":
            result = codex_bridge.list_sessions()
        elif name == "codex_switch":
            result = codex_bridge.set_active(arguments["session_id"])
        elif name == "codex_end":
            result = codex_bridge.end_session()
        elif name == "codex_config":
            result = codex_bridge.get_config()
        elif name == "codex_configure":
            result = codex_bridge.set_config(
                model=arguments.get("model"),
                sandbox=arguments.get("sandbox")
            )
        elif name == "codex_health":
            health = codex_bridge.health_check()
            result = f"Status: {health['status']}\nCodex installed: {health['codex_installed']}\nSessions: {health['sessions']}\nUptime: {health['uptime']}s"
        # Orchestration tools
        elif name == "multi_consult":
            result = await orchestrator.multi_consult(
                question=arguments["question"],
                backends=arguments.get("backends"),
                files=arguments.get("files"),
                synthesize=arguments.get("synthesize", True)
            )
        elif name == "agent_chain":
            result = await orchestrator.chain(steps=arguments["steps"])
        elif name == "delegate_codex":
            result = await orchestrator.delegate_to_codex(
                task=arguments["task"],
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
                return_to_opencode=arguments.get("return_to_opencode", False),
                opencode_followup=arguments.get("opencode_followup")
            )
        elif name == "parallel_agents":
            result = await orchestrator.parallel_agents(tasks=arguments["tasks"])
        elif name == "room_create":
            participants = arguments["participants"]
            if isinstance(participants, str):
                participants = json.loads(participants)
            # Normalize string participants into dicts:
            #   "local-gpu/model" → backend=local, "codex/model" → backend=codex,
            #   "claude" or "claude/model" → backend=claude,
            #   bare string → check existing sessions (local, codex, opencode) by ID,
            #   else → backend=opencode
            normalized = []
            for p in participants:
                if isinstance(p, dict):
                    normalized.append(p)
                else:
                    s = str(p)
                    if s.startswith("local-gpu/") or s.startswith("local/"):
                        model = s.split("/", 1)[1]
                        normalized.append({"name": model, "backend": "local", "model": model})
                    elif s.startswith("codex/"):
                        model = s.split("/", 1)[1]
                        normalized.append({"name": f"Codex ({model})", "backend": "codex", "model": model})
                    elif s == "claude" or s.startswith("claude/"):
                        model = s.split("/", 1)[1] if "/" in s else None
                        d = {"name": "Claude", "backend": "claude"}
                        if model:
                            d["model"] = model
                        normalized.append(d)
                    elif s in local_bridge.sessions:
                        sess = local_bridge.sessions[s]
                        normalized.append({"name": s, "backend": "local", "session_id": s, "model": sess.model})
                    elif s in codex_bridge.sessions:
                        sess = codex_bridge.sessions[s]
                        normalized.append({"name": s, "backend": "codex", "session_id": s, "model": sess.model})
                    elif s in bridge.sessions:
                        normalized.append({"name": s, "backend": "opencode", "session_id": s})
                    else:
                        normalized.append({"name": s, "backend": "opencode", "model": s})
            participants = normalized
            files_arg = arguments.get("files")
            if isinstance(files_arg, str):
                files_arg = json.loads(files_arg)
            result = rooms.create(
                room_id=arguments["room_id"],
                topic=arguments["topic"],
                participants=participants,
                files=files_arg,
            )
        elif name == "room_add_participant":
            p = arguments["participant"]
            if isinstance(p, str):
                p = json.loads(p)
            result = rooms.add_participant(room_id=arguments["room_id"], participant=p)
        elif name == "room_run":
            rid = arguments["room_id"]
            prompt = arguments.get("prompt")
            if prompt and rid in rooms.rooms:
                room = rooms.rooms[rid]
                room.messages.append({
                    "name": "MODERATOR",
                    "content": prompt,
                    "ts": datetime.now().isoformat(),
                })
            files_arg = arguments.get("files")
            if files_arg:
                if isinstance(files_arg, str):
                    files_arg = json.loads(files_arg)
                if rid in rooms.rooms:
                    expanded = _expand_paths(files_arg)
                    existing = set(rooms.rooms[rid].files or [])
                    rooms.rooms[rid].files = list(existing | set(expanded))
            result = await rooms.run_rounds(
                room_id=rid,
                rounds=arguments.get("rounds", 2),
                challenge=arguments.get("challenge", False),
            )
        elif name == "room_read":
            result = rooms.read(room_id=arguments["room_id"])
        elif name == "room_synthesize":
            synth = arguments.get("synthesizer")
            if isinstance(synth, str):
                synth = json.loads(synth)
            result = await rooms.synthesize(room_id=arguments["room_id"], synthesizer=synth)
        # Local model tools
        elif name == "local_discover":
            nodes = await asyncio.get_event_loop().run_in_executor(None, GpuNodeDiscovery.discover)
            if not nodes:
                result = "No local model endpoints found.\n\nTo make a GPU node discoverable:\n" \
                         "  1. Run slurm-serve-ollama.sh <model> to start Ollama on a Slurm GPU node\n" \
                         "  2. Or set CHITTA_BRIDGE_GPU_NODES=node1,node2 env var\n" \
                         "  3. Or run Ollama locally (localhost:11434)"
            else:
                lines = ["Available local model endpoints:\n"]
                for n in nodes:
                    lines.append(f"  [{n['source']}] {n['node']} — {n['base_url']}")
                    if n["models"]:
                        lines.append(f"    Models: {', '.join(n['models'])}")
                    else:
                        lines.append("    Models: (none loaded)")
                result = "\n".join(lines)
        elif name == "local_start":
            endpoint = arguments.get("endpoint")
            if not endpoint:
                nodes = await asyncio.get_event_loop().run_in_executor(None, GpuNodeDiscovery.discover)
                if not nodes:
                    result = "No local endpoint found. Run local_discover or specify endpoint."
                else:
                    endpoint = nodes[0]["base_url"]
                    result = local_bridge.start_session(
                        session_id=arguments["session_id"],
                        model=arguments["model"],
                        endpoint=endpoint,
                    )
            else:
                result = local_bridge.start_session(
                    session_id=arguments["session_id"],
                    model=arguments["model"],
                    endpoint=endpoint,
                )
        elif name == "local_discuss":
            result = await local_bridge.send_message(
                message=arguments["message"],
                session_id=arguments.get("session_id"),
                system_prompt=arguments.get("system_prompt"),
            )
        elif name == "local_sessions":
            result = local_bridge.list_sessions()
        elif name == "local_switch":
            result = local_bridge.set_active(arguments["session_id"])
        elif name == "local_end":
            result = local_bridge.end_session(arguments.get("session_id"))
        elif name == "local_history":
            result = local_bridge.get_history(
                session_id=arguments.get("session_id"),
                last_n=arguments.get("last_n", 20),
            )
        elif name == "local_models":
            endpoint = arguments.get("endpoint")
            if not endpoint:
                nodes = await asyncio.get_event_loop().run_in_executor(None, GpuNodeDiscovery.discover)
                if not nodes:
                    result = "No local endpoint found. Run local_discover or specify endpoint."
                else:
                    models = nodes[0]["models"]
                    result = f"Models at {nodes[0]['base_url']}:\n" + "\n".join(f"  - {m}" for m in models) if models else "No models loaded."
            else:
                models = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: LocalModelBridge.list_models_at(endpoint)
                )
                result = f"Models at {endpoint}:\n" + "\n".join(f"  - {m}" for m in models) if models else "No models found or endpoint unreachable."
        elif name == "local_health":
            h = local_bridge.health_check()
            result = f"Status: {h['status']}\nSessions: {h['sessions']}\nUptime: {h['uptime']}s"
        elif name == "opencode_cleanup":
            result = cleanup_opencode_snapshot()
        elif name == "opencode_ping":
            result = await bridge.ping(session_id=arguments.get("session_id"))
        elif name == "opencode_attach":
            result = bridge.attach_claude_session(
                session_id=arguments["session_id"],
                claude_session_id=arguments["claude_session_id"]
            )
        elif name == "opencode_detach":
            result = bridge.detach_claude_session(
                session_id=arguments["session_id"],
                claude_session_id=arguments["claude_session_id"]
            )
        elif name == "opencode_end_unattached":
            result = bridge.end_unattached()
        elif name == "opencode_end_all":
            result = bridge.end_all(
                session_ids=arguments.get("session_ids"),
                exclude_model=arguments.get("exclude_model")
            )
        elif name == "codex_attach":
            result = codex_bridge.attach_claude_session(
                session_id=arguments["session_id"],
                claude_session_id=arguments["claude_session_id"]
            )
        elif name == "codex_detach":
            result = codex_bridge.detach_claude_session(
                session_id=arguments["session_id"],
                claude_session_id=arguments["claude_session_id"]
            )
        elif name == "codex_end_unattached":
            result = codex_bridge.end_unattached()
        elif name == "codex_end_all":
            result = codex_bridge.end_all(
                session_ids=arguments.get("session_ids"),
                exclude_model=arguments.get("exclude_model")
            )
        elif name == "web_search":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: WebSearch.search_formatted(
                    arguments["query"],
                    arguments.get("max_results", 8),
                ),
            )
        elif name == "web_fetch":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: WebSearch.fetch_page(
                    arguments["url"],
                    arguments.get("max_chars", 12000),
                ),
            )
        elif name == "soul_recall":
            r = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: SoulClient.recall(arguments["query"], arguments.get("limit", 5)),
            )
            result = r or "Soul not available (chittad not running)"
        elif name == "soul_remember":
            r = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: SoulClient.remember(
                    arguments["content"],
                    arguments.get("kind", "episode"),
                    arguments.get("tags", ""),
                ),
            )
            result = r or "Soul not available (chittad not running)"
        elif name == "soul_context":
            r = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: SoulClient.smart_context(arguments["task"]),
            )
            result = r or "Soul not available (chittad not running)"
        elif name == "soul_status":
            available = SoulClient.is_available()
            if available:
                r = SoulClient._call("health_check", {})
                result = f"Soul: connected\n{r}" if r else "Soul: socket exists but no response"
            else:
                result = "Soul: not available (chittad not running)"
        elif name == "file_patch":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_file_patch(
                    arguments["file"],
                    arguments["old_str"],
                    arguments["new_str"],
                ),
            )
        elif name == "symbol_patch":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_patch(
                    arguments["file"],
                    arguments["symbol"],
                    arguments["new_body"],
                ),
            )
        else:
            result = f"Unknown tool: {name}"

        # Truncate large responses to reduce token cost. Export/history tools are exempt.
        _no_truncate = {"opencode_export", "opencode_history", "codex_history", "local_history"}
        _max_chars = 12_000
        if name not in _no_truncate and isinstance(result, str) and len(result) > _max_chars:
            result = result[:_max_chars] + f"\n\n[truncated — {len(result) - _max_chars:,} chars omitted]"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def _run_exec_mode() -> None:
    """Single-shot exec mode: read JSON from stdin, call backend, write JSON to stdout.

    Input (stdin):
        {"backend": "opencode"|"claude"|"codex", "model": "...",
         "system": "...", "message": "...", "session_id": "..." (optional)}

    Output (stdout):
        {"content": "...", "error": null}
    """
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"content": "", "error": f"invalid JSON: {e}"}))
        return

    backend = req.get("backend")
    if not backend:
        print(json.dumps({"content": "", "error": "missing required field: backend (claude|opencode|codex|local)"}))
        return
    model = req.get("model")
    system = req.get("system", "")
    message = req.get("message", "")
    session_id = req.get("session_id")

    full_prompt = f"{system}\n\n{message}" if system else message
    base_url = req.get("base_url")

    try:
        if backend == "claude":
            content = await bridge._run_claude_p(full_prompt)
        elif backend in ("opencode", "chitta-bridge"):
            sid = session_id or f"exec-{os.getpid()}"
            ephemeral = session_id is None
            if sid not in bridge.sessions:
                await bridge.start_session(sid, model=model)
            content = await bridge.send_message(full_prompt, sid, _raw=True)
            if ephemeral:
                bridge.end_session(sid)
        elif backend == "codex":
            content = await codex_bridge.run_task(full_prompt)
        elif backend == "local":
            endpoint = base_url or "http://localhost:11434/v1"
            sid = session_id or f"exec-local-{os.getpid()}"
            ephemeral = session_id is None
            if sid not in local_bridge.sessions:
                local_bridge.start_session(sid, model=model or "default", endpoint=endpoint)
            content = await local_bridge.send_message(
                message, sid, system_prompt=system or None
            )
            if ephemeral:
                local_bridge.end_session(sid)
        else:
            content = f"[error: unknown backend '{backend}']"
        print(json.dumps({"content": content}))
    except Exception as e:
        print(json.dumps({"content": "", "error": str(e)}))


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exec", action="store_true",
                        help="Single-shot mode: read JSON from stdin, write JSON to stdout")
    args, _ = parser.parse_known_args()

    if args.exec:
        asyncio.run(_run_exec_mode())
        return

    # Clean up stale snapshot files on every startup (issue #6845)
    cleanup_opencode_snapshot()

    async def run():
        init_options = InitializationOptions(
            server_name="chitta-bridge",
            server_version=__version__,
            capabilities=ServerCapabilities(tools=ToolsCapability()),
            instructions=(
                "## Session Reuse — CRITICAL\n"
                "Never call opencode_start or codex_start unless the user explicitly asks for a "
                "new session or a specific model. Always reuse the active session:\n"
                "1. Call opencode_sessions to see what is running\n"
                "2. If a session exists, use opencode_switch then opencode_discuss to continue\n"
                "3. Only call opencode_start when the user says 'new session', 'start fresh', "
                "or requests a specific model/id\n"
                "The active session retains full conversation context — starting a new one "
                "destroys that context. Same rule applies to Codex: prefer codex_discuss / "
                "codex_switch over codex_start.\n\n"
                "## File Attachments — CRITICAL\n"
                "The 'files' parameter in opencode_discuss, opencode_review, and similar tools "
                "MUST be an array, even for a single file.\n"
                "WRONG: files: \"/path/to/file.hpp\"\n"
                "CORRECT: files: [\"/path/to/file.hpp\"]"
            )
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)

    asyncio.run(run())


if __name__ == "__main__":
    main()
