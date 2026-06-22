"""Multi-agent orchestration and room participant identity.

Extracted from server.py: the Orchestrator class (fan-out / chain / parallel
delegation across backends), the AgentSoul dataclass (room participant identity),
and the _tool helper for building OpenAI function-calling tool definitions.
"""

import asyncio
from dataclasses import dataclass, field

from chitta_bridge.backends.codex import CodexBridge

__all__ = ["Orchestrator", "AgentSoul", "AGENT_TOOL_DEFINITIONS", "TOOL_XML_INSTRUCTIONS"]


class Orchestrator:
    """Multi-agent orchestration for complex workflows."""

    def __init__(self, codex_bridge: CodexBridge):
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
            backends: List of backends to consult ["codex"] (default: codex)
            files: Files to attach
            synthesize: Whether to synthesize results into a unified response
        """
        backends = backends or ["codex"]
        results: dict[str, str] = {}
        errors: dict[str, str] = {}

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
        parts = []
        for name, response in results.items():
            parts.append(f"## {name.upper()}\n\n{response}")
        for name, error in errors.items():
            parts.append(f"## {name.upper()} (error)\n\n{error}")
        return "\n\n---\n\n".join(parts)

    async def chain(
        self,
        steps: list[dict],
    ) -> str:
        """Execute a chain of agent steps, passing results forward.

        Each step is a dict with:
            - backend: "codex"
            - task: The task/prompt (can include {previous} placeholder)
            - model: Optional model override

        Example:
            [
                {"backend": "codex", "task": "Implement X"},
                {"backend": "codex", "task": "Review this implementation: {previous}"}
            ]
        """
        if not steps:
            return "No steps provided."

        results = []
        previous = ""

        for i, step in enumerate(steps, 1):
            backend = step.get("backend", "codex")
            task = step.get("task", "")
            model = step.get("model")

            # Substitute {previous} placeholder
            if "{previous}" in task and previous:
                task = task.replace("{previous}", previous)

            step_header = f"## Step {i}: {backend.upper()}"
            if model:
                step_header += f" (model={model})"

            try:
                if backend == "codex":
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
    ) -> str:
        """Delegate a task to Codex.

        Args:
            task: Task for Codex to execute
            working_dir: Working directory for Codex
            model: Codex model to use
        """
        codex_result = await self.codex.run_task(task, working_dir=working_dir, model=model)
        return f"## Codex Result\n\n{codex_result}"

    async def parallel_agents(
        self,
        tasks: list[dict],
    ) -> str:
        """Run multiple agent tasks in parallel across backends.

        Each task is a dict with:
            - backend: "codex"
            - task: The task/prompt
            - name: Optional name for the task
            - model: Optional model override

        All tasks run concurrently, results returned together.
        """
        if not tasks:
            return "No tasks provided."

        async def run_task(task_def: dict, index: int):
            backend = task_def.get("backend", "codex")
            task = task_def.get("task", "")
            name = task_def.get("name", f"Task {index}")
            model = task_def.get("model")

            try:
                if backend == "codex":
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
    _tool("paper_fetch", "Fetch academic paper metadata + discover supplements/data/code. "
          "Bypasses Cloudflare on bioRxiv/medRxiv/arXiv via their open APIs. "
          "Use full_text=true to extract full text (auto-finds local PDF by DOI, or provide pdf_path).",
          {"url": {"type": "string", "description": "Paper URL (bioRxiv, arXiv, DOI, PubMed) or bare DOI (10.xxx/...)"},
           "pdf_path": {"type": "string", "description": "Local PDF path for full text extraction and supplement URL scanning"},
           "doi": {"type": "string", "description": "Bare DOI as alternative to url"},
           "full_text": {"type": "boolean", "description": "Auto-find local PDF by DOI and extract full text. Gives download instructions if PDF not cached locally."}},
          []),
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
    _tool("pdf_read", "Read a PDF file with high-fidelity text extraction (PyMuPDF). Supports page ranges, "
          "metadata, table detection, and optional chitta ingestion for later recall.",
          {"path": {"type": "string", "description": "Absolute or relative path to the PDF file"},
           "pages": {"type": "string", "description": "Page range: '3', '1-5', 'all', or 'info' for metadata only. "
                     "Default: first max_pages pages."},
           "max_pages": {"type": "integer", "description": "Max pages to return when pages='all' (default 30)"},
           "ingest": {"type": "boolean", "description": "Auto-ingest extracted text into chitta memory (default false)"}},
          ["path"]),
    _tool("doc_read", "Read Office documents: .docx (Word), .xlsx (Excel), .pptx (PowerPoint), .odt/.ods/.odp (LibreOffice). "
          "Extracts text, tables, slide notes, and sheet data. Optional chitta ingestion.",
          {"path": {"type": "string", "description": "Absolute or relative path to the document"},
           "sheets": {"type": "string", "description": "For xlsx/ods: sheet name or index (e.g. 'Sheet1', '0'). Default: all sheets."},
           "ingest": {"type": "boolean", "description": "Auto-ingest extracted text into chitta memory (default false)"}},
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
    _tool("code_intel",
          "Memory-aware code analysis: symbol source + call graph (callers/callees) + file imports + chitta memory recall. One call replaces read_symbol + symbol_callers + symbol_callees + file_imports + recall.",
          {"symbol": {"type": "string", "description": "Symbol name (function/class/method)"},
           "path":   {"type": "string", "description": "File path for structure + imports"}},
          []),
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
