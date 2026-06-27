"""
Prompt-building utilities for chitta-bridge.

All functions construct prompt strings or extract file metadata for use
in OpenCode / Codex sessions.  No I/O side-effects beyond reading files
for content embedding.
"""

import re
from pathlib import Path
from typing import Optional

__all__ = [
    "get_file_info",
    "_human_size",
    "_expand_paths",
    "_embed_files_in_prompt",
    "build_file_context",
    "build_review_prompt",
    "build_message_prompt",
    "build_companion_prompt",
    "chunk_file",
    "build_chunk_prompt",
    "build_synthesis_prompt",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMALL_FILE = 500        # lines
MEDIUM_FILE = 1500      # lines
LARGE_FILE = 5000       # lines

CHUNK_SIZE = 800         # lines per chunk
CHUNK_OVERLAP = 20       # overlap between adjacent chunks

MAX_READ_SIZE = 10 * 1024 * 1024  # 10MB - above this, estimate lines from size

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

_BOUNDARY_RE = re.compile(
    r"^\s*$"
    r"|^(def|class|async\s+def)\s"
    r"|^(function|const|export)\s"
    r"|^(fn|pub\s+fn|impl|struct|mod)\s"
    r"|^#{1,6}\s"
    r"|^\s*[}\])]\s*$"
)

_file_info_cache: dict[str, dict] = {}

# Hone candidate-2 prompt — optimized for haiku bug-fix tasks (+20pp on unseen challenges)
# Source: github.com/twaldin/hone writeup/2026-04-18-haiku-20train-9holdout.md
_HAIKU_CODING_PREAMBLE = (
    "You are an AI coding agent fixing a bug in an open-source project.\n\n"
    "Follow this process:\n\n"
    "1. **Read ALL failing tests first.** Read test files completely. Run the suite — "
    "note every failing case, not just the first. Group failures by type.\n\n"
    "2. **Find the root cause.** Trace each failure to specific source lines. "
    "Check if failures share a root cause or need separate fixes. Check git log if unclear.\n\n"
    "3. **Fix root cause, not symptom.** Minimal change to pass failing tests without "
    "breaking others. If the same error appears in multiple places, fix all of them.\n\n"
    "4. **Handle edge cases.** Empty/null, special chars, numeric bounds, nested structures, "
    "encoding, array notation, option flags. For configurable libraries, check option paths.\n\n"
    "5. **Verify all tests pass.** Keep iterating until every originally-failing test passes "
    "and no regressions. If some still fail, re-read them and revise.\n\n"
    "6. **Persist through partial fixes.** Partial progress is not success. Check for "
    "second locations needing the same fix.\n\n"
    "Keep changes minimal. Do not refactor unrelated code or add new tests.\n\n"
)

# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------


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


def _expand_paths(paths: list) -> list:
    """Expand directories to contained files; keep plain file paths as-is.

    Accepts both strings and {path, description} dicts — dicts are passed
    through with their description preserved; only the path is expanded.
    """
    result: list = []
    for p in paths:
        if isinstance(p, dict):
            raw = p.get("path", "")
            desc = p.get("description", "")
            path = Path(raw)
            if path.is_dir():
                result.extend({"path": str(f), "description": desc} for f in sorted(path.rglob("*")) if f.is_file())
            elif path.is_file():
                result.append({"path": str(path), "description": desc})
        else:
            path = Path(p)
            if path.is_dir():
                result.extend(str(f) for f in sorted(path.rglob("*")) if f.is_file())
            elif path.is_file():
                result.append(str(path))
    return result


_BINARY_SUFFIXES = frozenset({
    ".gz", ".bz2", ".zst", ".xz", ".bam", ".cram", ".bcf",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar",
})
_EMBED_SIZE_LIMIT = 100_000  # bytes — skip files larger than this


def _embed_files_in_prompt(message: str, files: list) -> str:
    """Embed small text file content inline; list path+description for large/binary files.

    Accepts either flat strings or {path, description} dicts.
    Binary files and files over 100KB are never embedded — only listed with their description.
    """
    if not files:
        return message
    embedded, listed = [], []
    for f in files:
        if isinstance(f, dict):
            raw_path = f.get("path", "")
            desc = f.get("description", "")
        else:
            raw_path, desc = str(f), ""
        p = Path(raw_path)
        if not p.is_file():
            continue
        label = f"{p.name}" + (f" — {desc}" if desc else "")
        if p.suffix.lower() in _BINARY_SUFFIXES or p.stat().st_size > _EMBED_SIZE_LIMIT:
            listed.append(f"- **{label}** (binary/large — read via bash/read_file, path: `{p}`)")
        else:
            try:
                content = p.read_text(errors="replace")
                embedded.append(f"### File: {label}\n```\n{content}\n```")
            except OSError:
                listed.append(f"- **{label}** (unreadable, path: `{p}`)")
    parts = []
    if embedded:
        parts.extend(embedded)
    if listed:
        parts.append("### Reference files (too large or binary — access via bash/read_file)\n" + "\n".join(listed))
    if not parts:
        return message
    return "\n\n".join(parts) + "\n\n" + message


# ---------------------------------------------------------------------------
# Context and review prompts
# ---------------------------------------------------------------------------


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


def build_companion_prompt(
    message: str,
    files: Optional[list[str]] = None,
    domain_override: Optional[str] = None,
    is_followup: bool = False,
    model: Optional[str] = None,
) -> str:
    user_files = [f for f in (files or []) if not Path(f).name.startswith("opencode_msg_")]

    # Haiku: skip discussion scaffolding, inject bug-fix methodology
    if model and "haiku" in model.lower():
        parts = [_HAIKU_CODING_PREAMBLE]
        if user_files:
            file_context = build_file_context(user_files)
            if file_context:
                parts.extend(["## Context", file_context, ""])
        parts.append(message)
        return "\n".join(parts)

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

    if user_files:
        file_context = build_file_context(user_files)
        if file_context:
            parts.append("## Context")
            parts.append(file_context)
            parts.append("")

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
# Chunked-file processing
# ---------------------------------------------------------------------------


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
