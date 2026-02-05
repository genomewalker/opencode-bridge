#!/usr/bin/env python3
"""
OpenCode Bridge - MCP server for continuous OpenCode sessions.

Features:
- Continuous discussion sessions with conversation history
- Access to all OpenCode models (GPT-5, Claude, Gemini, etc.)
- Agent support (plan, build, explore, general)
- Session continuation
- File attachment for code review

Configuration:
- OPENCODE_MODEL: Default model (e.g., openai/gpt-5.2-codex)
- OPENCODE_AGENT: Default agent (plan, build, explore, general)
- ~/.opencode-bridge/config.json: Persistent config
"""

import os
import re
import json
import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability


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

MAX_READ_SIZE = 10 * 1024 * 1024  # 10MB - above this, estimate lines from size


def get_file_info(filepath: str) -> dict:
    """Get metadata about a file: size, lines, language, etc. Results are cached per path."""
    filepath = str(Path(filepath).resolve())
    if filepath in _file_info_cache:
        return _file_info_cache[filepath]

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
    r"^(?:\s*$"                     # blank line
    r"|(?:def |class |function |func |fn |pub fn |impl |module |package )"  # definitions
    r"|(?:})\s*$"                   # closing brace on its own line
    r"|(?://|#|/\*|\*/).{0,80}$"   # comment lines
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
DEFAULT_MODEL = "openai/gpt-5.2-codex"
DEFAULT_AGENT = "plan"
DEFAULT_VARIANT = "medium"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    agent: str = DEFAULT_AGENT
    variant: str = DEFAULT_VARIANT

    @classmethod
    def load(cls) -> "Config":
        config = cls()

        # Load from config file
        config_path = Path.home() / ".opencode-bridge" / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                config.model = data.get("model", config.model)
                config.agent = data.get("agent", config.agent)
                config.variant = data.get("variant", config.variant)
            except Exception:
                pass

        # Environment variables override config file
        config.model = os.environ.get("OPENCODE_MODEL", config.model)
        config.agent = os.environ.get("OPENCODE_AGENT", config.agent)
        config.variant = os.environ.get("OPENCODE_VARIANT") or config.variant

        return config

    def save(self):
        config_dir = Path.home() / ".opencode-bridge"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        data = {"model": self.model, "agent": self.agent, "variant": self.variant}
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


OPENCODE_BIN = find_opencode()


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Session:
    id: str
    model: str
    agent: str
    variant: str = DEFAULT_VARIANT
    opencode_session_id: Optional[str] = None
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
            created=data.get("created", datetime.now().isoformat())
        )
        for m in data.get("messages", []):
            session.messages.append(Message(**m))
        return session


class OpenCodeBridge:
    def __init__(self):
        self.start_time = datetime.now()
        self.config = Config.load()
        self.sessions: dict[str, Session] = {}
        self.active_session: Optional[str] = None
        self.sessions_dir = Path.home() / ".opencode-bridge" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.available_models: list[str] = []
        self.available_agents: list[str] = []
        self._load_sessions()

    def _load_sessions(self):
        for path in self.sessions_dir.glob("*.json"):
            try:
                session = Session.load(path)
                self.sessions[session.id] = session
            except Exception:
                pass

    async def _run_opencode(self, *args, timeout: int = 300) -> tuple[str, int]:
        """Run opencode CLI command and return output (async)."""
        global OPENCODE_BIN
        # Lazy retry: if binary wasn't found at startup, try again
        if not OPENCODE_BIN:
            OPENCODE_BIN = find_opencode()
        if not OPENCODE_BIN:
            return "OpenCode not installed. Install from: https://opencode.ai", 1

        try:
            proc = await asyncio.create_subprocess_exec(
                str(OPENCODE_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=b''),
                timeout=timeout
            )
            # Combine stdout+stderr so errors aren't silently lost
            out = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()
            output = out if out else err
            # If both exist and return code indicates error, include stderr
            if out and err and proc.returncode:
                output = f"{out}\n\nStderr:\n{err}"
            return output, proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Command timed out after {timeout}s", 1
        except Exception as e:
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

            output, code = await self._run_opencode(*args, timeout=300)

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
        # Use config defaults if not specified
        model = model or self.config.model
        agent = agent or self.config.agent
        variant = variant or self.config.variant

        session = Session(
            id=session_id,
            model=model,
            agent=agent,
            variant=variant
        )
        self.sessions[session_id] = session
        self.active_session = session_id
        session.save(self.sessions_dir / f"{session_id}.json")

        result = f"Session '{session_id}' started\n  Model: {model}\n  Agent: {agent}"
        if variant:
            result += f"\n  Variant: {variant}"
        return result

    def get_config(self) -> str:
        """Get current configuration."""
        return f"""Current configuration:
  Model: {self.config.model}
  Agent: {self.config.agent}
  Variant: {self.config.variant}

Set via:
  - ~/.opencode-bridge/config.json
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

        # --- Chunking gate: large user files get map-reduce processing ---
        user_files = [f for f in files if not Path(f).name.startswith("opencode_msg_")]
        needs_chunking = any(
            get_file_info(f).get("lines", 0) > CHUNK_THRESHOLD
            for f in user_files
        )

        if needs_chunking:
            reply = await self._run_chunked(message, user_files, session, mode="discuss")
            # Cleanup temp file
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
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

        # Scale timeout based on attached file size
        total_lines = sum(get_file_info(f).get("lines", 0) for f in user_files)
        # Base 300s, +60s per 1000 lines above threshold, capped at 900s
        timeout = min(900, 300 + max(0, (total_lines - MEDIUM_FILE) * 60 // 1000))

        output, code = await self._run_opencode(*args, timeout=timeout)

        # Cleanup temp file
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

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
            if total_lines > CHUNK_THRESHOLD:
                prompt = build_review_prompt(file_infos, focus)
                return await self._run_chunked(prompt, file_paths, self.sessions[sid], mode="review")

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
            lines.append(f"  - {sid}: {session.model} [{session.agent}{variant_str}], {msg_count} messages{active}")
        return "\n".join(lines)

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
        if not sid or sid not in self.sessions:
            return "No active session to end."

        del self.sessions[sid]
        session_path = self.sessions_dir / f"{sid}.json"
        if session_path.exists():
            session_path.unlink()

        if self.active_session == sid:
            self.active_session = None

        return f"Session '{sid}' ended."

    def export_session(self, session_id: Optional[str] = None, format: str = "markdown") -> str:
        """Export a session as markdown or JSON."""
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session to export."

        session = self.sessions[sid]

        if format == "json":
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


# MCP Server setup
bridge = OpenCodeBridge()
server = Server("opencode-bridge")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="opencode_models",
            description="List available models from OpenCode (GPT-5, Claude, Gemini, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Filter by provider (openai, github-copilot, anthropic)"
                    }
                }
            }
        ),
        Tool(
            name="opencode_agents",
            description="List available agents (plan, build, explore, general)",
            inputSchema={"type": "object", "properties": {}}
        ),
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
                        "description": "Model variant for reasoning effort: minimal, low, medium, high, xhigh, max (default: medium)"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="opencode_discuss",
            description="Send a message to OpenCode. Use for code review, architecture, brainstorming. "
                        "Auto-detects discussion domain and frames OpenCode as a specialized expert. "
                        "Use 'domain' to override detection.",
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
            name="opencode_brainstorm",
            description="Open-ended brainstorming on a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to brainstorm about"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="opencode_review",
            description="Review code for issues and improvements. Supports large files with adaptive review strategies. Can accept multiple file paths (space or comma separated).",
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
            inputSchema={"type": "object", "properties": {}}
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
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "opencode_models":
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
            result = bridge.end_session()
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
                format=arguments.get("format", "markdown")
            )
        elif name == "opencode_health":
            health = bridge.health_check()
            result = f"Status: {health['status']}\nSessions: {health['sessions']}\nUptime: {health['uptime']}s"
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def main():
    import asyncio

    async def run():
        init_options = InitializationOptions(
            server_name="opencode-bridge",
            server_version="0.1.0",
            capabilities=ServerCapabilities(tools=ToolsCapability())
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)

    asyncio.run(run())


if __name__ == "__main__":
    main()
