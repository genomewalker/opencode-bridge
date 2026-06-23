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
import signal as _signal
import asyncio
import socket
import uuid
import threading as _threading
from datetime import datetime
from pathlib import Path

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability

# MCP cancel-response suppression.
# Default RequestResponder.cancel() sends ErrorData(code=0, "Request cancelled")
# for the cancelled request id. Claude Code's stdio client already treats the
# request as cancelled locally (code -32001) and, on receiving that late
# response, logs "Received a response for an unknown message ID" and closes
# the transport, killing the whole MCP subprocess. We cancel the scope +
# mark completed WITHOUT emitting the duplicate response; our handlers
# propagate CancelledError, and the SDK's respond() path is already guarded
# by `if not self.cancelled`, so no response goes out either way.
try:
    from mcp.shared.session import RequestResponder as _RequestResponder

    async def _cancel_without_response(self):
        if not self._entered:
            raise RuntimeError("RequestResponder must be used as a context manager")
        if not self._cancel_scope:
            raise RuntimeError("No active cancel scope")
        self._cancel_scope.cancel()
        self._completed = True

    _RequestResponder.cancel = _cancel_without_response
except Exception:
    pass

from chitta_bridge import __version__

# --- extracted modules (re-exported for backward compat) ---
# ruff: noqa: F405  -- star-import facade; symbols come from extracted modules
from chitta_bridge.io_utils import *  # noqa: F401,F403
from chitta_bridge.cost import *  # noqa: F401,F403
from chitta_bridge.models import *  # noqa: F401,F403
from chitta_bridge.config import *  # noqa: F401,F403
from chitta_bridge.discovery import *  # noqa: F401,F403
from chitta_bridge.symbols import *  # noqa: F401,F403
from chitta_bridge.prompts import *  # noqa: F401,F403
from chitta_bridge.soul import *  # noqa: F401,F403
from chitta_bridge.backends.codex import *  # noqa: F401,F403
from chitta_bridge.backends.local import *  # noqa: F401,F403
from chitta_bridge.search.web import *  # noqa: F401,F403
from chitta_bridge.search.lit import *  # noqa: F401,F403
from chitta_bridge.reflib import *  # noqa: F401,F403
from chitta_bridge.code_intel import *  # noqa: F401,F403
from chitta_bridge.ingest import *  # noqa: F401,F403
from chitta_bridge.orchestrator import *  # noqa: F401,F403
from chitta_bridge.rooms import *  # noqa: F401,F403
from chitta_bridge.registry import REGISTRY, register
# Explicit imports so ruff can resolve star-import symbols used in this file
from chitta_bridge.config import CLAUDE_BIN, CODEX_BIN, DEFAULT_CODEX_MODEL, find_codex
from chitta_bridge.discovery import _discover_claude_shorthands, _discover_codex_shorthands, _infer_backend, _normalize_participant_shorthands
from chitta_bridge.symbols import _apply_file_patch, _apply_symbol_delete, _apply_symbol_edit, _apply_symbol_insert_child, _apply_symbol_move, _apply_symbol_patch, _apply_symbol_rename, _apply_symbol_rename_project, _locate_symbol
from chitta_bridge.code_intel import _cache_get_fresh, _make_handle, _read_outline, _read_range
from chitta_bridge.ingest import chitta_ingest, _doc_ingest, distill_event
from chitta_bridge.prompts import _expand_paths
from chitta_bridge.io_utils import _content_hash
from chitta_bridge.soul import SoulClient
from chitta_bridge.backends.codex import CodexBridge
from chitta_bridge.backends.local import GpuNodeDiscovery, LocalModelBridge
from chitta_bridge.search.web import WebSearch
from chitta_bridge.search.lit import LitSearch
from chitta_bridge.reflib import RefLib
from chitta_bridge.orchestrator import Orchestrator
from chitta_bridge.rooms import RoomManager, _resolve_preamble, ROOM_PREAMBLES, _ULTRACODE_KEYWORDS


_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Natural chunk-boundary markers used by chunk_file() to snap cuts to readable
# spots: blank lines, top-level defs/classes, markdown headings, closing braces.
_BOUNDARY_RE = re.compile(
    r"^\s*$"
    r"|^(def|class|async\s+def)\s"
    r"|^(function|const|export)\s"
    r"|^(fn|pub\s+fn|impl|struct|mod)\s"
    r"|^#{1,6}\s"
    r"|^\s*[}\])]\s*$"
)



# MCP Server setup
codex_bridge = CodexBridge()
local_bridge = LocalModelBridge()
orchestrator = Orchestrator(codex_bridge)
rooms = RoomManager(codex_bridge, local_bridge)
server = Server("chitta-bridge")

# ── Registry-backed tool handlers ───────────────────────────────────────────
# Single source of truth for schema + handler. list_tools() iterates REGISTRY;
# call_tool() fast-paths through REGISTRY before the legacy if/elif fallback.
# A registered handler owns the full response contract: it must return
# list[TextContent], applying the same truncation the legacy chain does.
_REGISTRY_CODEX_DEFAULT = _discover_codex_shorthands().get("gpt", "gpt-5.5")


def _display_name_for(shorthand: str) -> str:
    """Derive a short display name from a backend:model[:effort] shorthand.
    "claude:opus:high" → "Opus", "codex:gpt-5.5" → "GPT-5.5", "local:llama" → "Llama"
    Used by conductor_fusion so 'sees' entries match participant names naturally.
    """
    parts = shorthand.split(":")
    if parts[0] not in ("claude", "codex", "local") or len(parts) < 2:
        return shorthand
    model = parts[1]
    if parts[0] == "claude":
        fam = model.lower()
        if fam.startswith("claude-"):
            fam = fam[7:]
        return fam.split("-")[0].capitalize()
    if parts[0] == "codex":
        if model.lower().startswith("gpt"):
            return "GPT" + model[3:] if len(model) > 3 else "GPT"
        return model[0].upper() + model[1:]
    return model.split("-")[0].capitalize()


_NO_TRUNCATE = {"codex_history", "local_history", "pdf_read", "paper_fetch",
                "lit_search_arxiv", "lit_search_biorxiv", "lit_search_europepmc",
                "lit_search_openalex", "reflib_export"}


def _finalize(name: str, result: str) -> list:
    _max_chars = 12_000
    if name not in _NO_TRUNCATE and isinstance(result, str) and len(result) > _max_chars:
        result = result[:_max_chars] + f"\n\n[truncated — {len(result) - _max_chars:,} chars omitted]"
    return [TextContent(type="text", text=result)]


async def _h_discuss(arguments: dict) -> list:
    result = await codex_bridge.send_message(
        message=arguments["message"],
        images=arguments.get("files"),
    )
    _threading.Thread(target=distill_event, args=("checkpoint", result, {}), daemon=True).start()
    return _finalize("discuss", result)


_h_discuss.__doc__ = f"Ask a question or start a discussion. Routes to {_REGISTRY_CODEX_DEFAULT} by default."
register("discuss", {
    "type": "object",
    "properties": {
        "message":  {"type": "string",  "description": "Your message or question"},
        "files":    {"type": "array", "items": {"type": "string"}, "description": "File paths to attach"},
        "domain":   {"type": "string",  "description": "Domain hint (e.g. 'bioinformatics', 'security')"},
        "model":    {"type": "string",  "description": f"Model override (default: {_REGISTRY_CODEX_DEFAULT})"},
        "effort":   {"type": "string",  "description": "Effort: low, medium, high, xhigh (default: xhigh)"},
        "backend":  {"type": "string",  "description": "codex (default)"},
    },
    "required": ["message"],
})(_h_discuss)


async def _h_run(arguments: dict) -> list:
    result = await codex_bridge.run_task(
        task=arguments["task"],
        working_dir=arguments.get("working_dir"),
        model=arguments.get("model"),
        full_auto=True,
        effort=arguments.get("effort", "xhigh"),
        sandbox=arguments.get("sandbox", "danger-full-access"),
    )
    return _finalize("run", result)


_h_run.__doc__ = (
    f"Run a one-off task via {_REGISTRY_CODEX_DEFAULT} (full-auto, danger-full-access). "
    f"For questions use discuss; for long background tasks use codex_rescue."
)
register("run", {
    "type": "object",
    "properties": {
        "task":        {"type": "string",  "description": "Task or question"},
        "working_dir": {"type": "string",  "description": "Working directory (default: current)"},
        "model":       {"type": "string",  "description": f"Model override (default: {_REGISTRY_CODEX_DEFAULT})"},
        "effort":      {"type": "string",  "description": "Effort (default: xhigh)"},
        "sandbox":     {"type": "string",  "enum": ["read-only", "workspace-write", "danger-full-access"], "description": "Sandbox (default: danger-full-access)"},
    },
    "required": ["task"],
})(_h_run)


@register("codex_review", {
    "type": "object",
    "properties": {
        "working_dir": {"type": "string", "description": "Repo directory (default: current)"},
        "model":       {"type": "string", "description": "Model override"},
        "mode":        {"type": "string", "enum": ["normal", "adversarial"], "description": "normal (default) or adversarial"},
        "focus":       {"type": "string", "description": "What to focus on"},
        "base":        {"type": "string", "description": "Git ref to diff against (e.g. 'main')"},
        "effort":      {"type": "string", "description": "Effort"},
        "background":  {"type": "boolean", "description": "Run in background and return a job id (default: false)"},
        "sandbox":     {"type": "string", "enum": ["read-only", "workspace-write", "danger-full-access"], "description": "Sandbox mode"},
    },
})
async def _h_codex_review(arguments: dict) -> list:
    """Review code for bugs, issues, or design problems via Codex. Set background=true for long reviews."""
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
    return _finalize("codex_review", result)


@register("room_read", {
    "type": "object",
    "properties": {
        "room_id": {"type": "string", "description": "Room identifier"},
        "last_n":  {"type": "integer", "description": "Only return the last N messages"},
    },
    "required": ["room_id"],
})
async def _h_room_read(arguments: dict) -> list:
    """Read the transcript of a multi-model room (e.g. a fusion or discussion room)."""
    result = rooms.read(room_id=arguments.get("room_id", ""), last_n=arguments.get("last_n"))
    return _finalize("room_read", result)


@register("soul_recall", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to recall from memory"},
        "limit": {"type": "integer", "description": "Max memories to return (default: 5)"},
    },
    "required": ["query"],
})
async def _h_soul_recall(arguments: dict) -> list:
    """Recall memories from the soul (chittad) for the given query."""
    r = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: SoulClient.recall(arguments["query"], arguments.get("limit", 5)),
    )
    result = r or "Soul not available (chittad not running)"
    return _finalize("soul_recall", result)
# ── End registry-backed handlers ────────────────────────────────────────────

# Checked once at startup — used to suppress tools for missing backends
_HAS_CODEX = find_codex() is not None

# Tools hidden from tools/list to save context tokens.
# All are still callable directly or via the `advanced` gateway.
HIDDEN_TOOLS = {
    # Session lifecycle — prefer reuse over start/end
    "codex_start", "codex_end", "codex_end_all",
    "codex_switch", "codex_sessions", "codex_history",
    "codex_model", "codex_config", "codex_configure",
    "codex_rescue", "codex_health",
    "codex_job_status", "codex_job_result", "codex_job_cancel",
    # Local model management
    "local_start", "local_end", "local_switch",
    "local_sessions", "local_history", "local_models",
    "local_discover", "local_health", "local_discuss",
    # Orchestration (complex, rarely needed)
    "multi_consult", "agent_chain", "delegate_codex", "parallel_agents",
    # Rooms (multi-agent discussion — lifecycle only; core tools promoted to visible)
    "room_challenge", "room_cost",
    "room_inject", "room_fork", "room_add_participant", "room_set_preamble", "room_set_visibility",
    "scheduler_list", "scheduler_run_now", "scheduler_pause", "scheduler_resume", "scheduler_history",
    "room_status", "room_suggest_participants",
    # Status/health
    "soul_status",
    # Reference library — mutation ops (search/export remain visible)
    "reflib_remove", "reflib_tag",
}


def handle_advanced(arguments: dict) -> str:
    """Gateway to hidden chitta-bridge tools.

    Actions:
    - list: Show all hidden tools by category
    - call a hidden tool: {"tool": "<name>", "arguments": {...}}

    Examples:
      {"action": "list"}
      {"tool": "codex_start", "arguments": {"session_id": "main"}}
    """
    tool_name = arguments.get("tool", "")

    if tool_name:
        if tool_name not in HIDDEN_TOOLS:
            return f"Unknown hidden tool: {tool_name}\nUse action='list' to see available tools."

    # List hidden tools by category
    categories = {
        "Session lifecycle (codex)":    [t for t in sorted(HIDDEN_TOOLS) if t.startswith("codex_")],
        "Local models":                 [t for t in sorted(HIDDEN_TOOLS) if t.startswith("local_")],
        "Orchestration":                [t for t in sorted(HIDDEN_TOOLS) if t in {"multi_consult", "agent_chain", "delegate_codex", "parallel_agents"}],
        "Rooms":                        [t for t in sorted(HIDDEN_TOOLS) if t.startswith("room_")],
        "Misc":                         [t for t in sorted(HIDDEN_TOOLS) if not any(t.startswith(p) for p in ("codex_", "local_", "room_")) and t not in {"multi_consult", "agent_chain", "delegate_codex", "parallel_agents"}],
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
    _codex_default = _discover_codex_shorthands().get("gpt", "gpt-5.5")
    _tools = [
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
                        "description": "Sandbox mode: read-only, workspace-write, danger-full-access (default: danger-full-access — full host access; specify workspace-write for safer operation)"
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
        # ── Intent-shaped tools (backend is an internal detail) ─────────────────
        Tool(
            name="discuss",
            description=f"Ask a question or start a discussion. Routes to {_codex_default} by default.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message":  {"type": "string",  "description": "Your message or question"},
                    "files":    {"type": "array", "items": {"type": "string"}, "description": "File paths to attach"},
                    "domain":   {"type": "string",  "description": "Domain hint (e.g. 'bioinformatics', 'security')"},
                    "model":    {"type": "string",  "description": f"Model override (default: {_codex_default})"},
                    "effort":   {"type": "string",  "description": "Effort: low, medium, high, xhigh (default: xhigh)"},
                    "backend":  {"type": "string",  "description": "codex (default)"},
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="review",
            description="Review code for bugs, issues, or design problems. Routes to Codex by default.",
            inputSchema={
                "type": "object",
                "properties": {
                    "working_dir":   {"type": "string", "description": "Repo directory (default: current)"},
                    "code_or_file":  {"type": "string", "description": "Code snippet or file path(s)"},
                    "focus":         {"type": "string", "description": "What to focus on"},
                    "mode":          {"type": "string", "enum": ["normal", "adversarial"], "description": "normal (default) or adversarial"},
                    "base":          {"type": "string", "description": "Git ref to diff against (e.g. 'main')"},
                    "effort":        {"type": "string", "description": "Effort (default: xhigh)"},
                    "model":         {"type": "string", "description": f"Model override (default: {_codex_default})"},
                    "backend":       {"type": "string", "description": "codex (default)"},
                },
            },
        ),
        Tool(
            name="run",
            description=f"Run a one-off task via {_codex_default} (full-auto, danger-full-access). For questions use discuss; for long background tasks use codex_rescue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task":        {"type": "string",  "description": "Task or question"},
                    "working_dir": {"type": "string",  "description": "Working directory (default: current)"},
                    "model":       {"type": "string",  "description": f"Model override (default: {_codex_default})"},
                    "effort":      {"type": "string",  "description": "Effort (default: xhigh)"},
                    "sandbox":     {"type": "string",  "enum": ["read-only", "workspace-write", "danger-full-access"], "description": "Sandbox (default: danger-full-access)"},
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="fusion",
            description=(
                f"Dispatch a prompt to a panel of models in parallel (sparse topology — no cross-contamination), "
                f"then synthesize a single fused answer with a judge model. "
                f"Default panel: opus + {_codex_default}. "
                f"Fusion rooms are ephemeral; their transcript is readable via room_read."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The question or task to run through the panel.",
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "'backend:model[:effort]' strings for each panel member. "
                            "Default: [\"claude:opus:xhigh\", \"codex:gpt:xhigh\"]"
                        ),
                    },
                    "judge": {
                        "type": "string",
                        "description": (
                            "Model that synthesizes panel responses. "
                            "Default: \"claude:opus:max\""
                        ),
                    },
                    "topic": {
                        "type": "string",
                        "description": "Short label (defaults to first 120 chars of prompt).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["quick", "deep"],
                        "description": (
                            "Preset: \"quick\" = rounds=1, adversarial=false (default, fast). "
                            "\"deep\" = rounds=3, adversarial=true (full DRACO-style research). "
                            "Explicit rounds/adversarial params override the preset."
                        ),
                    },
                    "rounds": {
                        "type": "integer",
                        "description": "Number of independent sampling rounds per participant (sparse — no cross-contamination). More rounds = more answer-space coverage for the judge. Overrides mode preset. (default: 1)",
                    },
                    "adversarial": {
                        "type": "boolean",
                        "description": "If true, synthesis produces majority + minority reading + decision bet. Overrides mode preset. (default: false)",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Absolute paths to files or directories to include as context. "
                            "Directories are walked recursively (skips .git, __pycache__, node_modules, "
                            "binaries, etc.; 100KB per file / 600KB total budget). "
                            "Each file is prepended to the preamble as a fenced code block. "
                            "Use for codebase review, diffs, or any file-grounded analysis."
                        ),
                    },
                    "preamble": {
                        "type": "string",
                        "description": (
                            "Free-text framing prepended to every participant's system prompt. "
                            "Appended after any file content from the 'files' parameter. "
                            "Use to set domain context, constraints, or review focus."
                        ),
                    },
                    "minority_filter": {
                        "type": "boolean",
                        "description": (
                            "Show the judge only dissenting traces + a compressed majority summary "
                            "instead of the full transcript. Reduces input tokens; same synthesis quality "
                            "(SOTA: arXiv:2605.29116). Requires ≥3 participants to be effective; "
                            "pairs well with self_moa. Default: false."
                        ),
                    },
                    "cross_attend": {
                        "type": "boolean",
                        "description": (
                            "Ask the judge to first note what is unique to each proposal before synthesizing "
                            "(Attention-MoA prompt variant; SOTA: arXiv:2601.16596, +2.59pp AlpacaEval 2.0). Default: false."
                        ),
                    },
                    "adaptive_stop": {
                        "type": "boolean",
                        "description": (
                            "Use a haiku convergence score per round to stop early instead of the keyword heuristic. "
                            "Stops when score ≥ adaptive_threshold for adaptive_k consecutive rounds. Default: false."
                        ),
                    },
                    "adaptive_threshold": {
                        "type": "number",
                        "description": "Convergence score threshold for adaptive_stop (default: 0.85).",
                    },
                    "adaptive_k": {
                        "type": "integer",
                        "description": "Consecutive rounds above threshold before adaptive_stop triggers (default: 2).",
                    },
                    "min_quality": {
                        "type": "boolean",
                        "description": (
                            "Warn via MODERATOR message if any panel model is a known-weak proposer. "
                            "Does not block execution. Default: false."
                        ),
                    },
                    "self_moa": {
                        "type": "boolean",
                        "description": (
                            "Replace the panel with N repeated samples of the strongest participant "
                            "(Self-MoA). Cheaper than heterogeneous panels and often better when "
                            "one model is clearly strongest (SOTA: arXiv:2502.00674). Default: false."
                        ),
                    },
                    "self_moa_n": {
                        "type": "integer",
                        "description": "Number of Self-MoA copies (default: number of participants).",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="dual_fusion",
            description=(
                "Run the same prompt through BOTH a sparse room (independent sampling, no cross-talk) "
                "AND a dense room (full debate, participants see each other), then cross-synthesize with "
                "a judge model. Produces three categories: double-confirmed claims (highest confidence), "
                "sparse-only insights (lost to debate pressure), and dense-only refinements (emerged from challenge). "
                "Default panel: opus + gpt. Default judge: claude:opus:max."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The question or task."},
                    "participants": {
                        "type": "array", "items": {"type": "string"},
                        "description": "'backend:model[:effort]' strings. Default: [\"claude:opus:xhigh\", \"codex:gpt:xhigh\"]",
                    },
                    "judge": {
                        "type": "string",
                        "description": "Model for cross-synthesis. Default: \"claude:opus:max\"",
                    },
                    "dense_rounds": {
                        "type": "integer",
                        "description": "Rounds in the dense (debate) room (default: 2).",
                    },
                    "challenge": {
                        "type": "boolean",
                        "description": "Enable challenge rounds in the dense room (default: false).",
                    },
                    "adversarial": {
                        "type": "boolean",
                        "description": "Include majority/minority split in final synthesis (default: false).",
                    },
                    "topic": {"type": "string", "description": "Short label."},
                    "files": {
                        "type": "array", "items": {"type": "string"},
                        "description": "File/directory paths to include as context (same budget as fusion).",
                    },
                    "preamble": {"type": "string", "description": "Framing prepended to all participant prompts."},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="conductor_fusion",
            description=(
                "Conductor-style orchestration (arXiv:2512.04388): assign each agent a different subtask "
                "and explicit visibility over peers, then synthesize. Each workflow step specifies which "
                "agent runs, what sub-question it answers, and which other agents it can see. "
                "Compiles to preambles + visibility matrix automatically — no manual room setup needed. "
                "Use for complex tasks that benefit from information asymmetry: Thinker proposes, "
                "Workers execute on different angles, Verifiers check without cross-contamination."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": (
                            'JSON array of steps: [{"agent": "claude:opus:xhigh", '
                            '"subtask": "Propose the solution.", "sees": ["AgentA"]}]. '
                            "'agent' uses backend:model[:effort] shorthands (same as fusion); "
                            "default per step: claude:opus:xhigh. "
                            "'subtask' is injected as this agent's preamble. "
                            "'sees' is a list of agent names, or 'all'/'none'. "
                            "Duplicate base names get auto-numbered (Opus#1, Opus#2). "
                            "Omit workflow entirely to use the default TRINITY panel: "
                            "claude:opus:xhigh (Thinker, blind) + codex:gpt-5.5 (Worker, sees Opus) "
                            "+ claude:opus:high (Verifier, sees all)."
                        ),
                    },
                    "topic": {"type": "string", "description": "Short label for the room."},
                    "judge": {"type": "string", "description": "Model for final synthesis. Default: 'claude:opus:max'."},
                    "rounds": {"type": "integer", "description": "Discussion rounds per step (default: 1)."},
                    "adversarial": {"type": "boolean", "description": "Adversarial synthesis (default: false)."},
                    "files": {
                        "type": "array", "items": {"type": "string"},
                        "description": "File/directory paths to attach as shared context.",
                    },
                    "preamble": {"type": "string", "description": "Shared preamble prepended to all agents."},
                },
                "required": [],
            },
        ),
        Tool(
            name="room_set_visibility",
            description=(
                "Update the per-round visibility matrix on an existing room. "
                "Use to change what participants see mid-run without recreating the room. "
                "Keys are round numbers (integers), values map participant name to 'all', 'none', or [names]."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string"},
                    "visibility": {
                        "type": "string",
                        "description": 'JSON object: {"1": {"Alice": "none", "Bob": ["Alice"]}, "2": {"Alice": "all"}}.'
                    },
                },
                "required": ["room_id", "visibility"],
            },
        ),
        # ── Backend-specific tools (kept for direct control) ─────────────────
        Tool(
            name="codex_discuss",
            description=f"Send a message to Codex ({_codex_default}). Use for coding tasks, questions, debugging.",
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
            description=f"Run a one-off task via the Codex CLI ({_codex_default} by default). Use this to query {_codex_default} directly. Returns the result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: current)"
                    },
                    "model": {
                        "type": "string",
                        "description": f"Model to use (default: {_codex_default})"
                    },
                    "effort": {
                        "type": "string",
                        "description": "Effort: low, medium, high, xhigh (default: xhigh)"
                    },
                    "sandbox": {
                        "type": "string",
                        "enum": ["read-only", "workspace-write", "danger-full-access"],
                        "description": "Sandbox mode (default: danger-full-access)"
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
                    "model": {"type": "string", "description": f"New model (e.g. {_codex_default}, o3, o4-mini)"}
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
        Tool(
            name="codex_job_status",
            description="Check status of a background Codex job. Omit job_id to see all jobs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID (omit for all jobs)"}
                }
            }
        ),
        Tool(
            name="codex_job_result",
            description="Retrieve the result of a completed Codex background job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to retrieve result for"}
                }
            }
        ),
        Tool(
            name="codex_job_cancel",
            description="Cancel a running Codex background job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to cancel"}
                },
                "required": []
            }
        ),
        # Orchestration tools
        Tool(
            name="multi_consult",
            description="Fan-out a question to backends (Codex) in parallel, optionally synthesize results",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question/task to send to all backends (alias: prompt)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Alias for question"
                    },
                    "backends": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["codex"]},
                        "description": "Backends to consult (default: codex). Alias: participants. NOTE: model suffix (e.g. 'claude:claude-opus-4-7') is accepted for compatibility but ignored — use room_create for per-participant model control."
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Alias for backends. Accepts room-style shorthands like 'claude:claude-opus-4-7' or 'codex:gpt-5.5' but model suffix is ignored (routing only). Use room_create for per-model control."
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to attach"
                    },
                    "synthesize": {
                        "type": "boolean",
                        "description": "Whether to synthesize results into unified response (default: true). Alias: synthesis"
                    },
                    "synthesis": {
                        "type": "boolean",
                        "description": "Alias for synthesize"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="agent_chain",
            description="Execute agent steps sequentially, passing results forward.",
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "backend": {"type": "string", "enum": ["codex"]},
                                "task": {"type": "string", "description": "Task prompt. Use {previous} to include result from previous step"},
                                "model": {"type": "string", "description": "Optional model override"}
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
            description="Delegate a task to Codex.",
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
                                "backend": {"type": "string", "enum": ["codex"]},
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
                        "description": (
                            'JSON array: [{"name":"...","backend":"claude|codex|local",'
                            '"session_id":"...","model":"...","effort":"low|medium|high|xhigh|max",'
                            '"quarantine":"reader|actor",'
                            '"soul":{"system_prompt":"...","realm":"...","tools":["recall","web_search"],'
                            '"max_tool_turns":3,"challenge_bias":0.5,"max_rounds":0}}]. '
                            'backend defaults to "claude" if omitted. '
                            'effort: codex=low/medium/high/xhigh; claude=low/medium/xhigh/max '
                            '(NOTE: high is NOT valid for claude-opus-4-7 — use xhigh instead). '
                            'quarantine: "reader" = read-only tools only (web/search/recall), cannot write/run; '
                            '"actor" = action tools only, validates reader findings before acting. '
                            'Shorthand strings: "codex:gpt-5.5:high", "claude:claude-opus-4-7:xhigh".'
                        )
                    },
                    "files": {
                        "type": "string",
                        "description": 'JSON array of file or directory paths to attach to all participants: ["/path/to/file.py", "/path/to/dir"]. Directories are expanded recursively. Files are passed via --file to claude, embedded inline for codex/local.'
                    },
                    "roles": {
                        "type": "string",
                        "description": 'JSON object mapping participant name → epistemic role. Valid roles: "skeptic", "empiricist", "advocate", "devils_advocate". E.g. {"Claude-A": "skeptic", "Claude-B": "devils_advocate"}. Role text is injected into every turn prompt so it persists across rounds.'
                    },
                    "clean": {
                        "type": "boolean",
                        "description": "Clean room: participants only see injected CONTEXT/MODERATOR messages, not each other's accumulated history. Use for doc review, independent evaluation, or any task where you want explicit context injection rather than accumulated transcript. (default: false)"
                    },
                    "verbatim_rounds": {
                        "type": "integer",
                        "description": "Keep last N rounds verbatim; compress older rounds to haiku-generated summaries to bound context growth. Default: 2. Set 0 to disable compression (old behaviour)."
                    },
                    "preamble": {
                        "type": "string",
                        "description": "Opt-in framing prepended to every participant's system prompt — front-loads honest research context so a benign-but-sensitive domain is legible before the task, lowering false-positive safety flags. Pass a named preset (e.g. 'ancient-dna') or literal text. Lowers a base rate; does not guarantee a prompt clears. Not auto-applied by keyword."
                    },
                    "preambles": {
                        "type": "string",
                        "description": "JSON object: {participant_name: preamble_text}. Per-participant preamble overrides the shared preamble for that agent — enables Conductor-style per-agent framing, subtask context injection, or role-specific research context."
                    },
                    "visibility": {
                        "type": "string",
                        "description": "JSON object: {round_num: {participant_name: 'all'|'none'|[names]}}. Per-round per-participant access matrix (Conductor arXiv:2512.04388 access_list). 'all'=see everyone, 'none'=blind, [names]=see only listed participants. Round keys are integers. Falls back to sparse_topology/blind_first_round when absent."
                    },
                    "participant_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Tools each participant may call during their turn. "
                            "Pass [\"all\"] to grant access to all configured MCP tools (web_search, paper_fetch, chitta recall, etc.). "
                            "Or pass specific tool names: [\"mcp__chitta-bridge__web_search\", \"mcp__chitta__recall\"]. "
                            "Default: [] (no tools — text-only responses). "
                            "Applies to claude backend; codex has tools natively via full-auto."
                        )
                    }
                },
                "required": ["room_id", "topic", "participants"]
            }
        ),
        Tool(
            name="room_run",
            description=(
                "Run discussion rounds in a room. Participants respond in parallel each round.\n\n"
                "**Follow-up messages**: pass `prompt='...'` to inject a MODERATOR message before "
                "inference runs — this is the ONLY correct way to send follow-up questions. "
                "Never call room_run without prompt and expect a queued message to appear. "
                "When prompt is given, rounds defaults to 1 (each participant answers once); "
                "for a fresh start, omit prompt and rounds defaults to 2.\n\n"
                "Other options: challenge=true injects adversarial claims between rounds; "
                "blind_first_round=true hides peer responses in round 1; "
                "sparse_topology=true keeps ALL rounds blind; "
                "stop_early=true halts when disagreement resolves."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to run"},
                    "rounds": {"type": "integer", "description": "Number of discussion rounds (default: 1 when prompt given, 2 otherwise)"},
                    "challenge": {"type": "boolean", "description": "Enable challenge rounds — auto-extract claims and ask participants to verify/challenge them (default: false)"},
                    "blind_first_round": {"type": "boolean", "description": "If true, participants in round 1 see only the topic/context/moderator messages, not each other's prior outputs. Prevents first-round anchoring. (default: false)"},
                    "sparse_topology": {"type": "boolean", "description": "If true, ALL rounds are blind — participants never see each other's responses, only topic/context/moderator. Preserves statistical independence across all rounds; the synthesizer is the only node that sees the full transcript. Stronger than blind_first_round. (default: false)"},
                    "stop_early": {"type": "boolean", "description": "If true, stop before exhausting all rounds when the latest round shows no disagreement language and at least one cited response — i.e. the discussion has converged on evidence. (default: false)"},
                    "prompt": {"type": "string", "description": "Discussion prompt to inject as a MODERATOR message before running rounds"},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "File paths to attach to the room for this run"}
                },
                "required": ["room_id"]
            }
        ),
        Tool(
            name="room_synthesize",
            description="Synthesize a room's transcript into consensus, disagreements, and best answer. adversarial=true produces a majority reading + strongest-minority reading + decision bet field.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to synthesize"},
                    "adversarial": {"type": "boolean", "description": "If true, produce two competing readings (majority + strongest-minority) plus a 'decision bet' naming the critical unverified assumption. If no coherent minority can be constructed, the discussion is genuinely converged. (default: false)"},
                    "verify_citations": {"type": "boolean", "description": "If true, the synthesizer fetches and verifies each cited URL/arXiv/DOI before finalizing the synthesis. Flags unverifiable or misquoted references. (default: false)"},
                    "synthesizer": {
                        "type": "string",
                        "description": 'Optional JSON: {"name":"...","backend":"claude|codex|local","model":"..."}. Defaults to the backend used by room participants (inferred); falls back to claude if mixed.'
                    }
                },
                "required": ["room_id"]
            }
        ),
        Tool(
            name="room_challenge",
            description="Fork a completed room into a challenge round. Participants respond blind to the minority reading + decision bet from an adversarial synthesis — without seeing each other or re-litigating the majority. Run room_synthesize with adversarial=true first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "ID of the completed room to challenge"},
                    "minority_reading": {"type": "string", "description": "The strongest-minority reading from adversarial synthesis. Must be non-empty — room_challenge refuses to fabricate dissent."},
                    "decision_bet": {"type": "string", "description": "The critical unverified assumption from the adversarial synthesis decision bet field."},
                    "blind": {"type": "boolean", "description": "If true (default), participants form their challenge responses independently without seeing each other."}
                },
                "required": ["room_id", "minority_reading", "decision_bet"]
            }
        ),
        Tool(
            name="room_read",
            description="Read a discussion room transcript. Use last_n to get only the most recent N messages (avoids hitting length caps in long rooms).",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to read"},
                    "last_n": {"type": "integer", "description": "Return only the last N messages. Omit for full transcript."}
                },
                "required": ["room_id"]
            }
        ),
        Tool(
            name="room_status",
            description="Show per-round per-participant turn state for a room: success, retryable-absent, terminal-poison, or pending. Also shows retry_counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID"}
                },
                "required": ["room_id"]
            }
        ),
        Tool(
            name="room_fork",
            description=(
                "Fork a completed room into a new room seeded with a haiku-generated summary of the previous discussion. "
                "Rooms are single-use — use room_fork to continue a discussion without accumulating transcript cost. "
                "The new room starts clean with only the summary as context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "ID of the room to fork from"},
                    "new_room_id": {"type": "string", "description": "ID for the new room"},
                    "topic": {"type": "string", "description": "New topic (defaults to same as original)"},
                    "participants": {"type": "string", "description": "JSON array of participants (defaults to same as original)"},
                },
                "required": ["room_id", "new_room_id"]
            }
        ),
        Tool(
            name="room_inject",
            description=(
                "Inject explicit context into a room as CONTEXT messages. "
                "Use before room_run to add files, text, or memories without relying on the accumulated transcript. "
                "Essential for clean rooms (clean=true). Also useful to refresh context mid-discussion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string", "description": "Room ID to inject into"},
                    "items": {
                        "type": "array",
                        "description": (
                            'List of context items to inject. Each item: '
                            '{"type": "file", "path": "/abs/path", "label": "optional label"} — reads file, truncates to 8k chars; '
                            '{"type": "text", "content": "...", "label": "label"} — injects text directly; '
                            '{"type": "memory", "query": "search query", "limit": 5} — recalls from chitta memory.'
                        ),
                        "items": {"type": "object"}
                    }
                },
                "required": ["room_id", "items"]
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

        # ── Reference library (mutation ops) ──────────────────────
        Tool(
            name="reflib_remove",
            description="Remove a paper from the reference library by DOI or title substring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi_or_title": {"type": "string", "description": "DOI (exact) or title substring to match"}
                },
                "required": ["doi_or_title"],
            }
        ),
        Tool(
            name="reflib_tag",
            description="Add tags or notes to one or more papers in the reference library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi_or_title": {"type": "string", "description": "DOI (exact) or title substring to match"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add"
                    },
                    "notes": {"type": "string", "description": "Note to set on matched entries"},
                    "replace": {"type": "boolean", "description": "Replace existing tags instead of appending (default false)"},
                },
                "required": ["doi_or_title", "tags"],
            }
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
        Tool(
            name="symbol_delete",
            description=(
                "Delete an entire function, class, or method by name. AST-aware "
                "via tree-sitter; no old_str needed. Cheaper than Read+Edit for "
                "removing obsolete code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file":   {"type": "string", "description": "Absolute path to the file"},
                    "symbol": {"type": "string", "description": "Symbol name to delete"},
                },
                "required": ["file", "symbol"],
            },
        ),
        Tool(
            name="symbol_rename",
            description=(
                "Rename every occurrence of an identifier in a single file using "
                "word-boundary matching. Won't touch substrings of larger names. "
                "For cross-file rename use symbol_rename_project."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file":     {"type": "string", "description": "Absolute path to the file"},
                    "old_name": {"type": "string", "description": "Current identifier"},
                    "new_name": {"type": "string", "description": "New identifier (must be valid)"},
                },
                "required": ["file", "old_name", "new_name"],
            },
        ),
        Tool(
            name="symbol_rename_project",
            description=(
                "Rename an identifier across ALL files in the project. "
                "Discovers candidate files via grep from the git repo root, "
                "snapshots content hashes, validates no concurrent edits, "
                "then writes atomically per file. Covers .py .ts .js .go .rs .c .cpp .h files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file":     {"type": "string", "description": "Any file in the target git repo (used to find repo root)"},
                    "old_name": {"type": "string", "description": "Current identifier"},
                    "new_name": {"type": "string", "description": "New identifier (must be valid)"},
                },
                "required": ["file", "old_name", "new_name"],
            },
        ),
        Tool(
            name="symbol_move",
            description=(
                "Move a named function, class, or method from one file to another. "
                "AST-aware via tree-sitter. Appends to destination (creating it if "
                "needed) and removes from source atomically."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file":   {"type": "string", "description": "Source file (absolute path)"},
                    "symbol": {"type": "string", "description": "Symbol name to move"},
                    "dest":   {"type": "string", "description": "Destination file (absolute path)"},
                },
                "required": ["file", "symbol", "dest"],
            },
        ),
        Tool(
            name="symbol_edit",
            description=(
                "Replace old_str with new_str inside a named symbol's body. "
                "Uniqueness scoped to the symbol (not whole file) — old_str "
                "can be short and stable. Fails if old_str is missing or "
                "matches multiple times within the symbol."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file":    {"type": "string", "description": "Absolute path to the file"},
                    "symbol":  {"type": "string", "description": "Containing symbol name"},
                    "old_str": {"type": "string", "description": "Exact string inside symbol body (must be unique in scope)"},
                    "new_str": {"type": "string", "description": "Replacement string"},
                },
                "required": ["file", "symbol", "old_str", "new_str"],
            },
        ),
        Tool(
            name="symbol_callers",
            description=(
                "List the call sites of a symbol (who calls it) from the chitta "
                "code-graph daemon. Answer 'what depends on this' before changing a "
                "signature — structural, not guessed."
            ),
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Symbol name"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="symbol_callees",
            description=(
                "List the symbols a function calls (its callees) from the chitta "
                "code-graph daemon. Answer 'does this function touch X?' from the "
                "call graph instead of speculating about dataflow."
            ),
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Symbol name"}},
                "required": ["name"],
            },
        ),
        Tool(
            name="symbol_insert_child",
            description=(
                "Insert a block inside a parent function/class at a named position. "
                "AST-aware via tree-sitter. Auto-reindents body to parent's indent level. "
                "position: 'start', 'end', 'before_return', 'after_last_import', "
                "'after_docstring', 'before:<child>', 'after:<child>'. "
                "parent='__module__' targets top-level scope."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file":     {"type": "string", "description": "Absolute path to the file"},
                    "parent":   {"type": "string", "description": "Parent symbol name, or '__module__' for top-level"},
                    "position": {"type": "string", "description": "start|end|before_return|after_last_import|after_docstring|before:<child>|after:<child>"},
                    "new_body": {"type": "string", "description": "Block to insert (will be reindented)"},
                },
                "required": ["file", "parent", "position", "new_body"],
            },
        ),
        Tool(
            name="chitta_ingest",
            description=(
                "Manually ingest text into soul memory via regex extraction. "
                "Extracts SSL triplets, corrections, decisions, and review comments from text "
                "and writes each as an episodic memory tagged bridge-ingest. "
                "Returns count of memories written. Useful for testing or manual ingestion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract and ingest into soul memory"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="doc_ingest",
            description=(
                "Extract structured knowledge records from a document (PDF, URL, or text file) "
                "using a frontier LLM and write them as atomic, searchable chitta memories. "
                "dry_run=true (default) returns extracted JSON for review without writing. "
                "dry_run=false writes each record to chitta with provenance tracking. "
                "Use this to ingest papers, reports, or documentation into persistent memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Absolute path to a PDF/text file, or a URL"
                    },
                    "realm": {
                        "type": "string",
                        "description": "Chitta realm to write memories to (default: research)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to attach to all extracted records"
                    },
                    "model": {
                        "type": "string",
                        "description": "Extraction model (default: gpt-5.5)"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true (default), return extracted JSON without writing to chitta"
                    },
                    "max_memories": {
                        "type": "integer",
                        "description": "Max records to extract (default: 50)"
                    }
                },
                "required": ["source"]
            }
        ),
        Tool(
            name="lit_search_arxiv",
            description=(
                "Search arXiv for preprints and papers. No auth required. "
                "Returns title, authors, date, abstract snippet, and URL for each result."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (arXiv query syntax supported)"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 10, max 50)"},
                    "sort_by": {"type": "string", "enum": ["relevance", "lastUpdatedDate", "submittedDate"], "description": "Sort order (default: relevance)"},
                },
                "required": ["query"],
            }
        ),
        Tool(
            name="lit_search_biorxiv",
            description=(
                "Search bioRxiv or medRxiv preprints by keyword within a date range. "
                "Date range is required (API limitation). Client-side keyword filtering applied."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to filter results (AND logic)"},
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    "server": {"type": "string", "enum": ["biorxiv", "medrxiv"], "description": "Which server (default: biorxiv)"},
                    "max_results": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["query", "start_date", "end_date"],
            }
        ),
        Tool(
            name="lit_search_europepmc",
            description=(
                "Search Europe PMC for peer-reviewed literature. Open access filter on by default. "
                "Supports full PMC query syntax. Returns PMID, title, authors, journal, DOI."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (Europe PMC syntax supported)"},
                    "max_results": {"type": "integer", "description": "Max results (default 20, max 100)"},
                    "open_access_only": {"type": "boolean", "description": "Restrict to open access papers (default true)"},
                },
                "required": ["query"],
            }
        ),
        Tool(
            name="lit_search_openalex",
            description=(
                "Search OpenAlex for works, authors, institutions, topics, and more. "
                "Free API — uses polite pool without key, set OPENALEX_API_KEY env var for higher rate limits."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Full-text search query"},
                    "entity_type": {"type": "string", "enum": ["works", "authors", "sources", "institutions", "topics"], "description": "Entity type to search (default: works)"},
                    "max_results": {"type": "integer", "description": "Max results (default 20, max 100)"},
                    "filters": {"type": "string", "description": "OpenAlex filter string e.g. 'publication_year:2024,open_access.is_oa:true'"},
                },
                "required": ["query"],
            }
        ),
        Tool(
            name="paper_fetch",
            description=(
                "Fetch academic paper metadata and discover supplementary resources. "
                "Bypasses Cloudflare on bioRxiv/medRxiv/arXiv/PubMed via official open APIs. "
                "Also handles Zenodo, Figshare, GitHub URLs directly. "
                "Searches Zenodo/Figshare for supplement deposits and scans local PDFs for resource URLs. "
                "Provide url OR doi (url takes precedence when both are given). "
                "full_text=true auto-searches local scratch dirs for a cached PDF; use pdf_path to point at a known local file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Paper URL (bioRxiv, arXiv, PubMed, Zenodo, Figshare, GitHub) or bare DOI. Takes precedence over doi."},
                    "doi": {"type": "string", "description": "Bare DOI (e.g. '10.1038/s41586-021-03405-w') — used when url is not provided."},
                    "pdf_path": {"type": "string", "description": "Absolute path to a local PDF for full-text extraction and supplement URL scanning."},
                    "full_text": {"type": "boolean", "description": "If true, auto-search local scratch dirs for a cached PDF by DOI and extract all pages. No truncation applied."},
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="reflib_add",
            description=(
                "Add one or more papers to the persistent reference library. "
                "Accepts a DOI, URL (bioRxiv, arXiv, PubMed, DOI link), or whitespace/comma-separated list. "
                "Fetches structured metadata automatically. Deduplicates by DOI. "
                "Storage: $CHITTA_REFLIB or ~/.chitta/reflib.jsonl (git-backed)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url_or_doi": {
                        "type": "string",
                        "description": "One or more DOIs or URLs (space/comma/newline-separated)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to apply to all added papers (e.g. ['key-paper', 'metagenomics'])"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Free-text note to attach to all added papers"
                    },
                },
                "required": ["url_or_doi"],
            }
        ),
        Tool(
            name="reflib_search",
            description=(
                "Search the local reference library by keyword (title, abstract, authors, notes, tags). "
                "All keywords must match (AND logic). Optionally filter by tag. "
                "Returns DOI, title, authors, journal, tags, notes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search (AND logic). Empty string lists all entries."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter to entries that have this tag (case-insensitive)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 20)"
                    },
                },
                "required": [],
            }
        ),
        Tool(
            name="reflib_export",
            description=(
                "Export the reference library as markdown, bibtex, or jsonl. "
                "Optional tag/query filter. Markdown groups by year."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fmt": {
                        "type": "string",
                        "enum": ["markdown", "bibtex", "jsonl"],
                        "description": "Output format (default: markdown)"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Only export entries with this tag"
                    },
                    "query": {
                        "type": "string",
                        "description": "Keyword filter (AND logic) applied before export"
                    },
                },
                "required": [],
            }
        ),
        Tool(
            name="pdf_read",
            description=(
                "Read a PDF file with high-fidelity text and table extraction (pdfplumber + pypdf). "
                "Supports page ranges, metadata inspection, and optional chitta ingestion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the PDF file"},
                    "pages": {"type": "string", "description": "'info' for metadata, '1-5' for range, '3' for single page, 'all' for full doc"},
                    "max_pages": {"type": "integer", "description": "Max pages when pages='all' (default 30)"},
                    "ingest": {"type": "boolean", "description": "Auto-ingest extracted text into chitta memory"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="doc_read",
            description=(
                "Read Office/LibreOffice documents: .docx (Word), .xlsx (Excel), .pptx (PowerPoint), "
                ".odt/.ods/.odp (LibreOffice). Extracts text, tables, slide notes, sheet data. "
                "Optional chitta ingestion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the document"},
                    "sheets": {"type": "string", "description": "xlsx/ods only: sheet name or index (default: all)"},
                    "ingest": {"type": "boolean", "description": "Auto-ingest extracted text into chitta memory"}
                },
                "required": ["path"]
            }
        ),
    ]
    if not _HAS_CODEX:
        _tools = [t for t in _tools if not t.name.startswith("codex_")]
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
    # Registry is the source of truth for migrated tools; legacy _tools is the
    # fallback for everything not yet migrated. Registry wins on name clash.
    _registry_tools = [t.as_mcp_tool() for t in REGISTRY.values() if not t.hidden]
    _legacy_tools = [t for t in _tools if t.name not in REGISTRY]
    return _registry_tools + _legacy_tools


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name in REGISTRY:
            return await REGISTRY[name].handler(arguments)
        # existing if/elif chain continues below as fallback
        if name == "advanced":
            # List mode
            if "tool" not in arguments:
                result = handle_advanced(arguments)
            else:
                # Re-dispatch to the hidden tool
                hidden_name = arguments["tool"]
                hidden_args = arguments.get("arguments") or {
                    k: v for k, v in arguments.items()
                    if k not in ("tool", "action", "arguments")
                }
                if hidden_name not in HIDDEN_TOOLS:
                    result = f"Unknown hidden tool: {hidden_name}\nUse action='list' to see available tools."
                else:
                    return await call_tool(hidden_name, hidden_args)
        # Intent-shaped tools — route internally
        elif name == "discuss":
            result = await codex_bridge.send_message(
                message=arguments["message"],
                images=arguments.get("files"),
            )
            _threading.Thread(target=distill_event, args=("checkpoint", result, {}), daemon=True).start()
        elif name == "review":
            result = await codex_bridge.review_code(
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
                mode=arguments.get("mode", "normal"),
                focus=arguments.get("focus"),
                base=arguments.get("base"),
                effort=arguments.get("effort", "xhigh"),
                background=arguments.get("background", False),
                sandbox=arguments.get("sandbox"),
            )
            _threading.Thread(target=distill_event, args=("checkpoint", result, {}), daemon=True).start()
        elif name == "run":
            result = await codex_bridge.run_task(
                task=arguments["task"],
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
                full_auto=True,
                effort=arguments.get("effort", "xhigh"),
                sandbox=arguments.get("sandbox", "danger-full-access"),
            )
        elif name == "fusion":
            _fuse_prompt = arguments["prompt"]
            _fuse_topic = arguments.get("topic") or _fuse_prompt[:120]
            _fuse_parts_raw = arguments.get("participants") or ["claude:opus:xhigh", "codex:gpt:xhigh"]
            _fuse_judge_raw = arguments.get("judge", "claude:opus:max")
            _fuse_mode = arguments.get("mode", "quick")
            _fuse_adversarial = arguments.get("adversarial", _fuse_mode == "deep")
            _fuse_rounds = max(1, int(arguments.get("rounds", 3 if _fuse_mode == "deep" else 1)))
            if isinstance(_fuse_parts_raw, str):
                _fuse_parts_raw = json.loads(_fuse_parts_raw)
            _fuse_norm = _normalize_participant_shorthands(_fuse_parts_raw)
            _judge_norm = _normalize_participant_shorthands([_fuse_judge_raw])
            _fuse_judge = _judge_norm[0] if _judge_norm else {"name": "Synthesizer", "backend": "claude"}
            _fuse_room_id = f"fusion-{uuid.uuid4().hex[:8]}"

            # ceiling: static allowlist; upgrade: pull from a scored registry
            _STRONG_PROPOSERS = {"opus", "gpt", "sonnet", "gemini", "o3", "o4", "llama"}

            if arguments.get("self_moa") and _fuse_norm:
                _base = next(
                    (p for p in _fuse_norm if any(s in p.get("model", "").lower() for s in _STRONG_PROPOSERS)),
                    _fuse_norm[0],
                )
                _n = int(arguments.get("self_moa_n") or len(_fuse_norm))
                _fuse_norm = [
                    {**_base, "name": f"{_base['name']}#{i + 1}"} for i in range(max(1, _n))
                ]

            # Build preamble: file manifest (agents read on demand) + explicit preamble text
            _fuse_preamble_parts: list[str] = []
            if arguments.get("min_quality") and _fuse_norm:
                _weak = [p["name"] for p in _fuse_norm
                         if not any(s in p.get("model", "").lower() for s in _STRONG_PROPOSERS)]
                if _weak:
                    _fuse_preamble_parts.append(
                        f"[MODERATOR] Weak proposers detected: {', '.join(_weak)}. "
                        "Consider self_moa=true or substituting stronger models."
                    )
            _file_paths = arguments.get("files") or []
            if _file_paths:
                _manifest_lines: list[str] = []
                for _fp in _file_paths:
                    _afp = os.path.abspath(_fp)
                    if os.path.isdir(_afp):
                        _manifest_lines.append(f"- {_afp}/")
                    elif os.path.exists(_afp):
                        _manifest_lines.append(f"- {_afp}")
                if _manifest_lines:
                    _fuse_preamble_parts.append(
                        "## Files available for reading\n"
                        "Use read_file, sqz_read_file, find, or ls to read these paths on demand:\n\n"
                        + "\n".join(_manifest_lines)
                    )
            if arguments.get("preamble"):
                _fuse_preamble_parts.append(arguments["preamble"])
            _fuse_preamble = "\n\n".join(_fuse_preamble_parts)
            # Soul routing memory: recall historical performance for this topic
            # and reorder proposers by descending citation score.
            if SoulClient.is_available() and not arguments.get("self_moa"):
                _routing_mem = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: SoulClient.hybrid_recall(
                        f"[routing-memory] {_fuse_topic[:80]}", limit=10,
                    ),
                )
                if _routing_mem and len(_routing_mem.strip()) > 20:
                    import re as _re
                    # Extract model→citation_score pairs from memory text
                    _scores: dict[str, float] = {}
                    for _line in _routing_mem.splitlines():
                        _m = _re.search(r'model:([\w.:-]+).*citation_score:([0-9.]+)', _line)
                        if _m:
                            _scores[_m.group(1)] = float(_m.group(2))
                    if _scores:
                        _fuse_norm.sort(
                            key=lambda p: _scores.get(p.get("model", ""), 0.0),
                            reverse=True,
                        )
                        _fuse_preamble_parts.insert(0,
                            f"[MODERATOR] Participant order optimised by soul routing memory "
                            f"({len(_scores)} prior runs). Top model: {_fuse_norm[0].get('model','?')}.")
                        _fuse_preamble = "\n\n".join(_fuse_preamble_parts)
            await rooms.create(room_id=_fuse_room_id, topic=_fuse_topic, participants=_fuse_norm,
                               participant_tools=["all"], preamble=_fuse_preamble)
            await rooms.run_rounds(
                room_id=_fuse_room_id, rounds=_fuse_rounds, sparse_topology=True,
                adaptive_stop=bool(arguments.get("adaptive_stop")),
                adaptive_threshold=float(arguments.get("adaptive_threshold", 0.85)),
                adaptive_k=int(arguments.get("adaptive_k", 2)),
            )
            result = await rooms.synthesize(
                room_id=_fuse_room_id, synthesizer=_fuse_judge, adversarial=_fuse_adversarial,
                minority_filter=bool(arguments.get("minority_filter")),
                cross_attend=bool(arguments.get("cross_attend")),
            )
            _threading.Thread(target=distill_event, args=("room_synth", result, {}), daemon=True).start()
            # Soul routing memory: persist per-participant citation scores for future recall.
            if SoulClient.is_available():
                _fuse_room = rooms.rooms.get(_fuse_room_id)
                if _fuse_room:
                    _part_scores: dict[str, list[float]] = {}
                    for _msg in _fuse_room.messages:
                        _pname = _msg.get("name", "")
                        if _pname in {p["name"] for p in _fuse_norm}:
                            _cs = _msg.get("citation_score", 0)
                            _part_scores.setdefault(_pname, []).append(_cs)
                    _majority, _minority, _ = rooms._detect_plurality(_fuse_room)
                    _minority_names = {m["name"] for m in _minority}
                    _n_rounds = rooms._committed_rounds(_fuse_room)
                    for _p in _fuse_norm:
                        _pn = _p["name"]
                        _avg_cit = sum(_part_scores.get(_pn, [0])) / max(1, len(_part_scores.get(_pn, [1])))
                        _mem_content = (
                            f"[routing-memory] topic:{_fuse_topic[:60]} "
                            f"participant:{_pn} model:{_p.get('model','?')} "
                            f"backend:{_p.get('backend','?')} "
                            f"citation_score:{_avg_cit:.2f} "
                            f"was_minority:{_pn in _minority_names} "
                            f"round_count:{_n_rounds}"
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda c=_mem_content: SoulClient.remember(
                                c, "signal", "routing-memory,fusion,bridge",
                            ),
                        )
            result = f"[fusion:{_fuse_room_id}]\n{result}"
        elif name == "dual_fusion":
            _df_prompt    = arguments["prompt"]
            _df_topic     = arguments.get("topic") or _df_prompt[:120]
            _df_parts_raw = arguments.get("participants") or ["claude:opus:xhigh", "codex:gpt:xhigh"]
            _df_judge_raw = arguments.get("judge", "claude:opus:max")
            _df_dense_rounds = max(1, int(arguments.get("dense_rounds", 2)))
            _df_challenge = bool(arguments.get("challenge", False))
            _df_adversarial = bool(arguments.get("adversarial", False))
            if isinstance(_df_parts_raw, str):
                _df_parts_raw = json.loads(_df_parts_raw)
            _df_norm  = _normalize_participant_shorthands(_df_parts_raw)
            _df_judge = (_normalize_participant_shorthands([_df_judge_raw]) or [{}])[0]
            _df_base  = f"dual-{uuid.uuid4().hex[:8]}"
            _df_sparse_id = f"{_df_base}-sparse"
            _df_dense_id  = f"{_df_base}-dense"

            # File manifest — agents read on demand via tools
            _df_preamble_parts: list[str] = []
            _df_file_paths = arguments.get("files") or []
            if _df_file_paths:
                _df_manifest: list[str] = []
                for _fp in _df_file_paths:
                    _afp = os.path.abspath(_fp)
                    if os.path.isdir(_afp):
                        _df_manifest.append(f"- {_afp}/")
                    elif os.path.exists(_afp):
                        _df_manifest.append(f"- {_afp}")
                if _df_manifest:
                    _df_preamble_parts.append(
                        "## Files available for reading\n"
                        "Use read_file, sqz_read_file, find, or ls to read these paths on demand:\n\n"
                        + "\n".join(_df_manifest)
                    )
            if arguments.get("preamble"):
                _df_preamble_parts.append(arguments["preamble"])
            _df_preamble = "\n\n".join(_df_preamble_parts)

            # Create both rooms, run in parallel
            await rooms.create(room_id=_df_sparse_id, topic=_df_topic,
                               participants=_df_norm, participant_tools=["all"], preamble=_df_preamble)
            await rooms.create(room_id=_df_dense_id,  topic=_df_topic,
                               participants=_df_norm, participant_tools=["all"], preamble=_df_preamble)
            await asyncio.gather(
                rooms.run_rounds(_df_sparse_id, rounds=1, sparse_topology=True),
                rooms.run_rounds(_df_dense_id,  rounds=_df_dense_rounds, challenge=_df_challenge),
            )

            # Build combined transcript for cross-synthesis
            _sparse_room = rooms.rooms[_df_sparse_id]
            _dense_room  = rooms.rooms[_df_dense_id]
            _sparse_tx   = rooms._build_annotated_transcript(_sparse_room)
            _dense_tx    = rooms._build_annotated_transcript(_dense_room)
            _adversarial_block = (
                "\nAfter completing the five sections, add:\n"
                "### Minority reading\nThe strongest alternative conclusion a reasonable reader could "
                "reach from the combined evidence.\n"
                "### Decision bet\nThe single most critical unverified assumption both rooms share.\n"
            ) if _df_adversarial else ""
            _df_synth_prompt = (
                f"You are a cross-room synthesizer. Two rooms ran the same prompt with the same participants "
                f"but different topologies.\n\n"
                f"**SPARSE ROOM** (`{_df_sparse_id}`) — independent sampling, no cross-talk:\n"
                f"Participants answered without seeing each other. Positions are statistically independent.\n\n"
                f"{_sparse_tx}\n\n"
                f"---\n\n"
                f"**DENSE ROOM** (`{_df_dense_id}`) — full debate topology, {_df_dense_rounds} round(s):\n"
                f"Participants read each other's responses and could challenge, refine, or concede.\n\n"
                f"{_dense_tx}\n\n"
                f"---\n\n"
                f"## Cross-Room Synthesis\n\n"
                f"### 1. Double-confirmed claims\n"
                f"Claims present in BOTH rooms — independently reached AND debate-tested. "
                f"Highest epistemic weight. Distinguish grounded (cited) from asserted.\n\n"
                f"### 2. Sparse-only insights\n"
                f"Positions from the sparse room that debate pressure crowded out or drove to premature consensus. "
                f"What did cross-talk lose?\n\n"
                f"### 3. Dense-only refinements\n"
                f"Claims sharpened or discovered through challenge that neither participant would have reached alone. "
                f"What did debate add?\n\n"
                f"### 4. Final integrated answer\n"
                f"The strongest single answer drawing on all three categories.\n\n"
                f"### 5. Open questions\n"
                f"What remains unresolved across both rooms?"
                f"{_adversarial_block}"
            )
            _df_backend = _df_judge.get("backend", "claude")
            try:
                if _df_backend == "claude":
                    _df_reply = await rooms._run_claude_p(_df_synth_prompt, model=_df_judge.get("model"))
                elif _df_backend == "codex":
                    _df_reply = await codex_bridge.run_task(_df_synth_prompt)
                else:
                    _df_reply = f"[error: unknown judge backend {_df_backend!r}]"
            except Exception as _dfe:
                _df_reply = f"[cross-synthesis error: {_dfe}]"
            _threading.Thread(target=distill_event, args=("room_synth", _df_reply, {}), daemon=True).start()
            result = (
                f"[dual_fusion:{_df_base}]\n"
                f"Sparse room: {_df_sparse_id} · Dense room: {_df_dense_id}\n\n"
                f"## Cross-Room Synthesis by {_df_judge.get('name', _df_judge_raw)}\n\n"
                f"{_df_reply}"
            )
        elif name == "conductor_fusion":
            _cf_workflow_raw = arguments.get("workflow", "[]")
            if isinstance(_cf_workflow_raw, str):
                _cf_workflow = json.loads(_cf_workflow_raw)
            else:
                _cf_workflow = _cf_workflow_raw
            _cf_topic     = arguments.get("topic") or "Conductor task"
            _cf_judge_raw = arguments.get("judge", "claude:opus:max")
            _cf_rounds    = max(1, int(arguments.get("rounds", 1)))
            _cf_adversarial = bool(arguments.get("adversarial", False))
            _cf_judge = (_normalize_participant_shorthands([_cf_judge_raw]) or [{}])[0]
            _cf_room_id = f"conductor-{uuid.uuid4().hex[:8]}"
            # Default panel when workflow is empty: TRINITY (Thinker / Worker / Verifier)
            # Opus:xhigh proposes blind (strongest independent reasoning caps the chain);
            # GPT-5.5 extends (backend diversity in the Worker slot breaks Opus echo chamber);
            # Opus:high verifies with strong critical reasoning over both.
            if not _cf_workflow:
                _cf_workflow = [
                    {"agent": "claude:opus:xhigh",  "subtask": "Thinker: propose a complete answer independently.", "sees": "none"},
                    {"agent": "codex:gpt-5.5",      "subtask": "Worker: build on or extend what the Thinker proposes.", "sees": ["Opus"]},
                    {"agent": "claude:opus:high",   "subtask": "Verifier: critique both above responses. Cite at least 2 specific issues or confirmations.", "sees": ["Opus", "GPT-5.5"]},
                ]

            # Compile workflow into participants, preambles, and visibility matrix.
            # Display names ("Opus", "GPT-5.5") are derived so sees entries match naturally.
            # Explicit "name" in each step takes priority over the derived name.
            _cf_name_counts: dict[str, int] = {}
            _cf_participants = []
            _cf_preambles: dict[str, str] = {}
            _cf_vis_per_step: list[dict] = []  # [{name: sees}] per step
            for _step in _cf_workflow:
                _agent_raw = _step.get("agent", "claude:opus:xhigh")
                _norm = _normalize_participant_shorthands([_agent_raw])
                _p = _norm[0] if _norm else {"name": _agent_raw, "backend": "claude"}
                _base_name = _step.get("name") or _display_name_for(_agent_raw)
                _p["_agent_raw"] = _agent_raw
                _cf_name_counts[_base_name] = _cf_name_counts.get(_base_name, 0) + 1
                _cnt = _cf_name_counts[_base_name]
                _p["name"] = f"{_base_name}#{_cnt}" if _cnt > 1 else _base_name
                _cf_participants.append(_p)
                _cf_preambles[_p["name"]] = _step.get("subtask", "")
                _sees = _step.get("sees", "all")
                _cf_vis_per_step.append({_p["name"]: _sees})
            # Build case-insensitive lookup so sees entries match regardless of form
            # ("Opus", "opus", "claude:opus", "claude:opus:high" all resolve correctly).
            _cf_name_lookup: dict[str, str] = {}
            for _p in _cf_participants:
                _nm = _p["name"]
                _cf_name_lookup[_nm.lower()] = _nm
                _cf_name_lookup.setdefault(_nm.split("#")[0].lower(), _nm)
                _raw = _p.pop("_agent_raw", "")
                if _raw:
                    _cf_name_lookup[_raw.lower()] = _nm
                    _raw_parts = _raw.split(":")
                    if len(_raw_parts) > 1:
                        _cf_name_lookup.setdefault(_raw_parts[1].lower(), _nm)
            def _resolve_sees_cf(_s):
                if not isinstance(_s, list):
                    return _s
                return [_cf_name_lookup.get(_e.lower(), _e) for _e in _s]
            _cf_vis_per_step = [{k: _resolve_sees_cf(v) for k, v in d.items()} for d in _cf_vis_per_step]
            # Apply same visibility to all rounds
            _cf_visibility: dict[int, dict] = {
                r: {k: v for d in _cf_vis_per_step for k, v in d.items()}
                for r in range(1, _cf_rounds + 1)
            }
            # Shared preamble: file manifest (agents read on demand) + explicit preamble text
            _cf_preamble_parts: list[str] = []
            _cf_file_paths = arguments.get("files") or []
            if _cf_file_paths:
                _cf_manifest: list[str] = []
                for _fp in _cf_file_paths:
                    _afp = os.path.abspath(_fp)
                    if os.path.isdir(_afp):
                        _cf_manifest.append(f"- {_afp}/")
                    elif os.path.exists(_afp):
                        _cf_manifest.append(f"- {_afp}")
                if _cf_manifest:
                    _cf_preamble_parts.append(
                        "## Files available for reading\n"
                        "Use read_file, sqz_read_file, find, or ls to read these paths on demand:\n\n"
                        + "\n".join(_cf_manifest)
                    )
            if arguments.get("preamble"):
                _cf_preamble_parts.append(arguments["preamble"])
            _cf_shared_preamble = "\n\n".join(_cf_preamble_parts)
            await rooms.create(
                room_id=_cf_room_id, topic=_cf_topic,
                participants=_cf_participants,
                preamble=_cf_shared_preamble,
                preambles=_cf_preambles,
                visibility=_cf_visibility,
                participant_tools=["all"],
            )
            await rooms.run_rounds(_cf_room_id, rounds=_cf_rounds)
            result = await rooms.synthesize(
                _cf_room_id, synthesizer=_cf_judge, adversarial=_cf_adversarial,
            )
            _threading.Thread(target=distill_event, args=("room_synth", result, {}), daemon=True).start()
            result = f"[conductor_fusion:{_cf_room_id}]\n{result}"

        elif name == "room_set_visibility":
            _rv_id = arguments.get("room_id", "")
            if _rv_id not in rooms.rooms:
                rooms._try_load_room(_rv_id)
            if _rv_id not in rooms.rooms:
                result = f"Room '{_rv_id}' not found."
            else:
                _rv_raw = arguments.get("visibility", "{}")
                _rv_parsed = json.loads(_rv_raw) if isinstance(_rv_raw, str) else _rv_raw
                # Normalise string keys to int
                _rv_norm = {int(k): v for k, v in _rv_parsed.items()}
                rooms.rooms[_rv_id].visibility = _rv_norm
                rooms._save_room(_rv_id)
                result = f"Visibility matrix updated on '{_rv_id}': {_rv_norm}"

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
            _threading.Thread(target=distill_event, args=("checkpoint", result, {}), daemon=True).start()
        elif name == "codex_run":
            result = await codex_bridge.run_task(
                task=arguments["task"],
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
                full_auto=arguments.get("full_auto", True),
                effort=arguments.get("effort", "xhigh"),
                sandbox=arguments.get("sandbox", "danger-full-access"),
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
            _mc_q = arguments.get("question") or arguments.get("prompt", "")
            _mc_syn = arguments.get("synthesize", arguments.get("synthesis", True))
            _mc_backends = arguments.get("backends")
            if not _mc_backends and "participants" in arguments:
                _mc_backends = ["codex"]
            result = await orchestrator.multi_consult(
                question=_mc_q,
                backends=_mc_backends,
                files=arguments.get("files"),
                synthesize=_mc_syn,
            )
        elif name == "agent_chain":
            result = await orchestrator.chain(steps=arguments["steps"])
        elif name == "delegate_codex":
            result = await orchestrator.delegate_to_codex(
                task=arguments["task"],
                working_dir=arguments.get("working_dir"),
                model=arguments.get("model"),
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
            #   bare string → check existing sessions (local, codex) by ID,
            #   else → backend=claude
            _EFFORT_VALUES = {"low", "medium", "high", "xhigh", "max"}
            _CLAUDE_SHORTHANDS = _discover_claude_shorthands()
            _CODEX_SHORTHANDS  = _discover_codex_shorthands()
            normalized = []
            for p in participants:
                if isinstance(p, dict):
                    normalized.append(p)
                else:
                    s = str(p)
                    if ":" in s and s.split(":", 1)[0] in ("codex", "claude", "local"):
                        parts = s.split(":")
                        backend_hint = parts[0]
                        sid_or_model = parts[1] if len(parts) > 1 else ""
                        effort_hint = parts[2].lower() if len(parts) > 2 and parts[2].lower() in _EFFORT_VALUES else None
                        if backend_hint == "claude":
                            d = {"name": s, "backend": "claude"}
                            if sid_or_model:
                                d["model"] = _CLAUDE_SHORTHANDS.get(sid_or_model.lower(), sid_or_model)
                            if effort_hint:
                                d["effort"] = effort_hint
                            normalized.append(d)
                        elif backend_hint == "local":
                            if sid_or_model in local_bridge.sessions:
                                sess = local_bridge.sessions[sid_or_model]
                                normalized.append({"name": s, "backend": "local", "session_id": sid_or_model, "model": sess.model})
                            else:
                                normalized.append({"name": s, "backend": "local", "model": sid_or_model})
                        elif backend_hint == "codex":
                            d: dict = {"name": s, "backend": "codex"}
                            if sid_or_model in codex_bridge.sessions:
                                sess = codex_bridge.sessions[sid_or_model]
                                d["session_id"] = sid_or_model
                                d["model"] = sess.model
                            else:
                                d["model"] = _CODEX_SHORTHANDS.get(sid_or_model.lower(), sid_or_model)
                            if effort_hint:
                                d["effort"] = effort_hint
                            normalized.append(d)
                        continue
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
                    else:
                        try:
                            inferred = _infer_backend(s)
                        except ValueError:
                            inferred = "claude"
                        if inferred == "claude":
                            model_id = _CLAUDE_SHORTHANDS.get(s.lower(), s)
                        elif inferred == "codex":
                            model_id = _CODEX_SHORTHANDS.get(s.lower(), s)
                        else:
                            model_id = s
                        normalized.append({"name": s, "backend": inferred, "model": model_id})
            participants = normalized
            # Resolve backend at create time — never silently at dispatch
            unresolved = None
            for p in participants:
                if not p.get("backend"):
                    try:
                        p["backend"] = _infer_backend(p.get("name", ""), p.get("model"))
                    except ValueError as e:
                        unresolved = f"Error resolving backend for '{p.get('name', '?')}': {e}"
                        break
            if unresolved:
                result = unresolved
            else:
                files_arg = arguments.get("files")
                if isinstance(files_arg, str):
                    files_arg = json.loads(files_arg)
                room_id = arguments.get("room_id") or f"room-{uuid.uuid4().hex[:8]}"
                roles_arg = arguments.get("roles")
                if isinstance(roles_arg, str):
                    roles_arg = json.loads(roles_arg)
                result = await rooms.create(
                    room_id=room_id,
                    topic=arguments.get("topic", ""),
                    participants=participants,
                    files=files_arg,
                    roles=roles_arg,
                    clean=bool(arguments.get("clean", False)),
                    verbatim_rounds=int(arguments.get("verbatim_rounds", 2)),
                    preamble=arguments.get("preamble", ""),
                    preambles=(json.loads(arguments["preambles"]) if isinstance(arguments.get("preambles"), str) else arguments.get("preambles") or {}),
                    visibility=(json.loads(arguments["visibility"]) if isinstance(arguments.get("visibility"), str) else arguments.get("visibility") or {}),
                    participant_tools=arguments.get("participant_tools") or [],
                )
        elif name == "room_set_preamble":
            rid = arguments.get("room_id", "")
            if rid not in rooms.rooms:
                rooms._try_load_room(rid)
            if rid not in rooms.rooms:
                result = f"Room '{rid}' not found."
            else:
                val = arguments.get("preamble", "")
                rooms.rooms[rid].preamble = val or ""
                rooms._save_room(rid)
                resolved = _resolve_preamble(val)
                if not val:
                    result = f"Cleared framing preamble on '{rid}'."
                else:
                    named = " (named: " + val + ")" if val.strip().lower() in ROOM_PREAMBLES else ""
                    result = (f"Set framing preamble on '{rid}'{named} — "
                              f"{len(resolved)} chars, prepended to every participant's "
                              f"system prompt from the next round.")
        elif name == "room_add_participant":
            p = arguments.get("participant")
            if not p:
                result = "Error: 'participant' is required"
            else:
                if isinstance(p, str):
                    p = json.loads(p)
                rid = arguments.get("room_id")
                result = await rooms.add_participant(room_id=rid, participant=p) if rid else "Error: 'room_id' is required"
        elif name == "room_run":
            rid = arguments.get("room_id", "")
            # Ensure room is loaded from disk (survives process restart)
            if rid not in rooms.rooms:
                rooms._try_load_room(rid)
            prompt = arguments.get("prompt")
            # Ultracode detection — elevate effort + rounds when keyword present
            ultracode_mode = bool(prompt and _ULTRACODE_KEYWORDS.search(prompt))
            if ultracode_mode and rid in rooms.rooms:
                room_obj = rooms.rooms[rid]
                for p in room_obj.participants:
                    if p.get("backend") == "claude" and p.get("effort") not in ("xhigh", "max"):
                        p["effort"] = "xhigh"
                rooms.rooms[rid].messages.append({
                    "name": "SYSTEM",
                    "content": "[ultracode] Extended reasoning activated — xhigh effort, adversarial verification enabled.",
                    "ts": datetime.now().isoformat(),
                })
                rooms._save_room(rid)
            if prompt and rid in rooms.rooms:
                msgs = rooms.rooms[rid].messages
                last = msgs[-1] if msgs else {}
                if not (last.get("name") == "MODERATOR" and last.get("content") == prompt):
                    msgs.append({
                        "name": "MODERATOR",
                        "content": prompt,
                        "ts": datetime.now().isoformat(),
                        "op_id": str(uuid.uuid4()),  # idempotency key — dedup retries
                    })
                    rooms._save_room(rid)
            files_arg = arguments.get("files")
            if files_arg:
                if isinstance(files_arg, str):
                    files_arg = json.loads(files_arg)
                if rid in rooms.rooms:
                    expanded = _expand_paths(files_arg)
                    existing = set(rooms.rooms[rid].files or [])
                    rooms.rooms[rid].files = list(existing | set(expanded))
                    rooms._save_room(rid)
            # Default to 1 round for follow-ups, 2 for initial runs, 3 for ultracode
            default_rounds = 3 if ultracode_mode else (1 if prompt else 2)
            result = await rooms.run_rounds(
                room_id=rid,
                rounds=int(arguments.get("rounds", default_rounds)),
                challenge=arguments.get("challenge", ultracode_mode),
                blind_first_round=arguments.get("blind_first_round", ultracode_mode),
                sparse_topology=arguments.get("sparse_topology", False),
                stop_early=arguments.get("stop_early", False),
            )
        elif name == "room_fork":
            old_id = arguments.get("room_id", "")
            new_id = arguments.get("new_room_id", "")
            if not old_id or not new_id:
                result = "Error: room_id and new_room_id are required"
            else:
                if old_id not in rooms.rooms:
                    rooms._try_load_room(old_id)
                topic_arg = arguments.get("topic")
                parts_arg = arguments.get("participants")
                if isinstance(parts_arg, str):
                    parts_arg = json.loads(parts_arg) if parts_arg else None
                if parts_arg:
                    parts_arg = _normalize_participant_shorthands(parts_arg)
                result = await rooms.fork(
                    old_room_id=old_id, new_room_id=new_id,
                    topic=topic_arg, participants=parts_arg,
                )
        elif name == "room_inject":
            rid = arguments.get("room_id", "")
            if rid not in rooms.rooms:
                rooms._try_load_room(rid)
            if rid not in rooms.rooms:
                result = f"Room '{rid}' not found."
            else:
                room_obj = rooms.rooms[rid]
                items = arguments.get("items", [])
                if isinstance(items, str):
                    items = json.loads(items)
                injected = 0
                total_chars = 0
                for item in items:
                    itype = item.get("type", "text")
                    label = item.get("label", itype)
                    if itype == "file":
                        path = item.get("path", "")
                        try:
                            content = Path(path).read_text(errors="replace")[:8192]
                            msg_content = f"[{label}: {path}]\n{content}"
                        except Exception as e:
                            msg_content = f"[{label}: {path} — read error: {e}]"
                    elif itype == "memory":
                        if SoulClient.is_available():
                            query = item.get("query", "")
                            limit = int(item.get("limit", 5))
                            recalled = SoulClient.hybrid_recall(query, limit=limit)
                            msg_content = f"[Memory: {query}]\n{recalled or '(no results)'}"
                        else:
                            msg_content = "[Memory: chitta not available]"
                    else:
                        msg_content = f"[{label}]\n{item.get('content', '')}"
                    room_obj.messages.append({
                        "name": "CONTEXT",
                        "content": msg_content,
                        "ts": datetime.now().isoformat(),
                    })
                    injected += 1
                    total_chars += len(msg_content)
                rooms._save_room(rid)
                result = f"Injected {injected} context item(s) into '{rid}' ({total_chars:,} chars total)."
        elif name == "room_read":
            result = rooms.read(room_id=arguments.get("room_id", ""), last_n=arguments.get("last_n"))
        elif name == "room_status":
            rid = arguments.get("room_id", "")
            if rid not in rooms.rooms:
                rooms._try_load_room(rid)
            if rid not in rooms.rooms:
                result = f"Room '{rid}' not found."
            else:
                room = rooms.rooms[rid]
                participants = [p["name"] for p in room.participants]
                # Collect all committed turn_keys and their states
                committed: dict[str, dict] = {}  # turn_key → {name, round, poison}
                existing_rounds: set[int] = set()
                for m in room.messages:
                    tk = m.get("turn_key", "")
                    if tk and tk.startswith("r") and ":" in tk:
                        try:
                            rnum = int(tk[1:tk.index(":")])
                            existing_rounds.add(rnum)
                            committed[tk] = {
                                "participant": tk[tk.index(":")+1:],
                                "round": rnum,
                                "poison": m.get("poison", False),
                            }
                        except ValueError:
                            pass
                max_round = max(existing_rounds) if existing_rounds else 0
                lines = [f"# Room status: {rid}", f"Participants: {', '.join(participants)}", ""]
                for rnum in range(1, max_round + 1):
                    lines.append(f"## Round {rnum}")
                    for pname in participants:
                        tk = f"r{rnum}:{pname}"
                        retries = room.retry_counts.get(pname, 0)
                        if tk in committed:
                            state = "poison" if committed[tk]["poison"] else "success"
                        elif retries > 0:
                            state = f"retryable-absent (retried {retries}×)"
                        else:
                            state = "pending"
                        lines.append(f"  {pname}: **{state}**")
                    lines.append("")
                # Show any open retryable-absent in the next round (skip poisoned)
                poisoned_names = {
                    m["turn_key"][m["turn_key"].index(":") + 1:]
                    for m in room.messages
                    if m.get("poison") and ":" in m.get("turn_key", "")
                }
                next_round = max_round + 1
                pending = []
                for pname in participants:
                    if pname in poisoned_names:
                        continue
                    tk = f"r{next_round}:{pname}"
                    retries = room.retry_counts.get(pname, 0)
                    if retries > 0:
                        pending.append(f"{pname} (retried {retries}×)")
                if pending:
                    lines.append(f"## Round {next_round} (in progress)")
                    for p in pending:
                        lines.append(f"  {p}: retryable-absent")
                result = "\n".join(lines)
        elif name == "room_suggest_participants":
            # Query live backends and return the best participant strings for the task.
            task_type = arguments.get("task_type", "design")  # design|coding|review|fast
            lines = ["## Suggested participants (live query)\n"]
            # Claude: probe via `claude --version` output or fall back to CLAUDE.md constants
            claude_models = {}
            try:
                import subprocess as _sp
                raw = _sp.run(["claude", "models", "--json"], capture_output=True, text=True, timeout=8)
                if raw.returncode == 0:
                    import json as _json
                    data = _json.loads(raw.stdout)
                    for m in data if isinstance(data, list) else data.get("models", []):
                        mid = m if isinstance(m, str) else m.get("id", "")
                        if "opus" in mid:
                            claude_models["opus"] = mid
                        elif "sonnet" in mid:
                            claude_models["sonnet"] = mid
                        elif "haiku" in mid:
                            claude_models["haiku"] = mid
            except Exception:
                pass
            # Fallback: read from CLAUDE.md model table if CLI query failed
            if not claude_models:
                import re as _re
                claude_md = Path.home() / ".claude" / "CLAUDE.md"
                if claude_md.exists():
                    text = claude_md.read_text()
                    for mid in _re.findall(r'`(claude-[a-z0-9-]+)`', text):
                        if "opus" in mid and "opus" not in claude_models:
                            claude_models["opus"] = mid
                        elif "sonnet" in mid and "sonnet" not in claude_models:
                            claude_models["sonnet"] = mid
                        elif "haiku" in mid and "haiku" not in claude_models:
                            claude_models["haiku"] = mid
            opus = claude_models.get("opus", "claude-opus-4-8")
            sonnet = claude_models.get("sonnet", "claude-sonnet-4-6")
            # Codex: use current configured model
            codex_model = codex_bridge.config.codex_model if codex_bridge else DEFAULT_CODEX_MODEL
            if task_type == "fast":
                lines.append(f"claude:{sonnet}:medium   — fast synthesis")
                lines.append(f"codex:{codex_model}:medium  — fast reasoning")
            elif task_type == "review":
                lines.append(f"claude:{opus}:xhigh    — adversarial review")
                lines.append(f"codex:{codex_model}:xhigh   — independent perspective")
            else:  # design / default
                lines.append(f"claude:{opus}:xhigh    — architecture, extended thinking")
                lines.append(f"codex:{codex_model}:xhigh   — extended reasoning")
            lines.append(f"\n_Queried live. Claude opus={opus}, sonnet={sonnet}, codex={codex_model}_")
            result = "\n".join(lines)
        elif name == "room_cost":
            rid = arguments.get("room_id", "")
            import json as _j

            def _summarise_cost_records(records: list, header: str) -> str:
                total_in  = sum(r.get("in_tok", 0) for r in records)
                total_out = sum(r.get("out_tok", 0) for r in records)
                total_cw  = sum(r.get("cache_write_tok", 0) for r in records)
                total_cr  = sum(r.get("cache_read_tok", 0) for r in records)
                total_usd = sum(r.get("est_usd", 0.0) for r in records)
                lines = [header]
                by_participant: dict = {}
                for r in records:
                    p = r.get("participant", "?")
                    by_participant.setdefault(p, {"in": 0, "out": 0, "cw": 0, "cr": 0, "usd": 0.0, "rounds": 0})
                    by_participant[p]["in"]     += r.get("in_tok", 0)
                    by_participant[p]["out"]    += r.get("out_tok", 0)
                    by_participant[p]["cw"]     += r.get("cache_write_tok", 0)
                    by_participant[p]["cr"]     += r.get("cache_read_tok", 0)
                    by_participant[p]["usd"]    += r.get("est_usd", 0.0)
                    by_participant[p]["rounds"] += 1
                for p, s in sorted(by_participant.items(), key=lambda x: -x[1]["usd"]):
                    lines.append(f"  **{p}** — {s['rounds']} turns  in {s['in']:,} · out {s['out']:,} · cw {s['cw']:,} · cr {s['cr']:,} · **${s['usd']:.4f}**")
                lines.append(f"\n  **Total** — in {total_in:,} · out {total_out:,} · cw {total_cw:,} · cr {total_cr:,} · **${total_usd:.4f}**")
                lines.append(f"  _$200/mo Agent SDK credit — {total_usd / 200 * 100:.2f}% used_")
                return "\n".join(lines)

            if rid:
                cost_path = rooms.rooms_dir / f"{rid}.costs.jsonl"
                if not cost_path.exists():
                    result = f"No cost data for room '{rid}' (no claude: participants or room not found)."
                else:
                    records = [_j.loads(line) for line in cost_path.read_text().splitlines() if line.strip()]
                    result = _summarise_cost_records(records, f"# Room cost: {rid}\n")
            else:
                # All rooms summary
                all_files = sorted(rooms.rooms_dir.glob("*.costs.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
                if not all_files:
                    result = "No room cost data found."
                else:
                    all_records = []
                    room_lines = ["# All rooms cost summary\n"]
                    for cf in all_files:
                        recs = [_j.loads(line) for line in cf.read_text().splitlines() if line.strip()]
                        if not recs:
                            continue
                        room_total = sum(r.get("est_usd", 0.0) for r in recs)
                        room_rounds = len(set(f"r{r.get('round',0)}:{r.get('participant','')}" for r in recs))
                        room_lines.append(f"  **{cf.stem}** — {room_rounds} turns  ${room_total:.4f}")
                        all_records.extend(recs)
                    grand_total = sum(r.get("est_usd", 0.0) for r in all_records)
                    room_lines.append(f"\n**Grand total across {len(all_files)} room(s): ${grand_total:.4f}**")
                    room_lines.append(f"_$200/mo Agent SDK credit — {grand_total / 200 * 100:.2f}% used_")
                    result = "\n".join(room_lines)
        elif name in ("scheduler_list", "scheduler_run_now", "scheduler_pause",
                       "scheduler_resume", "scheduler_history"):
            # Lazy import so stdio-mode sessions don't pay the import cost
            try:
                from chitta_bridge.scheduler import SchedulerService  # noqa: F401
                # Retrieve the running scheduler instance from the HTTP mode global
                import chitta_bridge.server as _self_mod
                _sched = getattr(_self_mod, "_active_scheduler", None)
                if _sched is None:
                    result = "Scheduler not running (start bridge with --http to enable)."
                elif name == "scheduler_list":
                    result = _sched.list_jobs()
                elif name == "scheduler_run_now":
                    result = await _sched.run_now(
                        arguments.get("job_id", ""),
                        dry_run=arguments.get("dry_run", False),
                    )
                elif name == "scheduler_pause":
                    result = _sched.pause(arguments.get("job_id", ""))
                elif name == "scheduler_resume":
                    result = _sched.resume(arguments.get("job_id", ""))
                elif name == "scheduler_history":
                    result = _sched.job_history(arguments.get("job_id", ""))
            except Exception as e:
                result = f"[scheduler error: {e}]"
        elif name == "room_synthesize":
            synth = arguments.get("synthesizer")
            if isinstance(synth, str):
                synth = json.loads(synth)
            result = await rooms.synthesize(room_id=arguments.get("room_id", ""), synthesizer=synth,
                                            adversarial=arguments.get("adversarial", False),
                                            verify_citations=arguments.get("verify_citations", False))
            _threading.Thread(target=distill_event, args=("room_synth", result, {}), daemon=True).start()
        elif name == "room_challenge":
            result = await rooms.challenge(
                room_id=arguments.get("room_id", ""),
                minority_reading=arguments.get("minority_reading", ""),
                decision_bet=arguments.get("decision_bet", ""),
                blind=arguments.get("blind", True),
            )
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
            result = await asyncio.to_thread(codex_bridge.end_unattached)
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
            # ── DEDUP NOTE: thin alias of mcp__chitta__remember ──────────────
            # mcp__chitta-bridge__soul_remember and mcp__chitta__remember both
            # terminate at the SAME chittad daemon. SoulClient._socket_path
            # (server.py:5460-5474) derives the socket from a djb2 hash of
            # ~/.claude/mind, so both clients dial the identical unix socket and
            # invoke the daemon's "remember" tool. This branch is a thin alias:
            # it adds NO storage, dedup, or transformation of its own. Prefer
            # mcp__chitta__remember directly; this exists only for in-room tool
            # parity. Do not add divergent behaviour here without mirroring it in
            # the chitta MCP path or the two surfaces will silently disagree.
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
        elif name == "symbol_delete":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_delete(arguments["file"], arguments["symbol"]),
            )
        elif name == "symbol_rename":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_rename(
                    arguments["file"], arguments["old_name"], arguments["new_name"],
                ),
            )
        elif name == "symbol_rename_project":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_rename_project(
                    arguments["file"], arguments["old_name"], arguments["new_name"],
                ),
            )
        elif name == "symbol_move":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_move(
                    arguments["file"], arguments["symbol"], arguments["dest"],
                ),
            )
        elif name == "symbol_edit":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_edit(
                    arguments["file"], arguments["symbol"],
                    arguments["old_str"], arguments["new_str"],
                ),
            )
        elif name == "read_symbol":
            sym = arguments.get("name", "")
            file_hint = arguments.get("file")
            offset = int(arguments.get("offset", 0))
            max_chars = int(arguments.get("max_chars", 8000))
            cached = _cache_get_fresh(file_hint, sym) if sym else None
            if cached:
                body = f"[cache] {cached['file']}:{cached['line_start']}\n{cached['body']}"
                # Emit a fresh handle for cache hits too
                try:
                    fp = Path(cached["file"])
                    raw_body = cached["body"]
                    hid = _make_handle(str(fp.resolve()), sym, _content_hash(raw_body))
                    body += f"\n\n[handle: cbh:{hid}]"
                except Exception:
                    pass
            else:
                body = SoulClient._call("read_symbol", {"name": sym}) or "(not found)"
                # Attempt to emit a handle by re-locating the symbol on disk
                if file_hint and body != "(not found)":
                    try:
                        fp = Path(file_hint).expanduser().resolve()
                        content = fp.read_text(encoding="utf-8", errors="replace")
                        loc = _locate_symbol(fp, sym, content)
                        if loc and not isinstance(loc, str):
                            s, e, _ = loc
                            hid = _make_handle(str(fp), sym, _content_hash(content[s:e]))
                            body += f"\n\n[handle: cbh:{hid}]"
                    except Exception:
                        pass
            body = body[offset:] if offset > 0 else body
            if len(body) > max_chars:
                remaining = len(body) - max_chars
                result = body[:max_chars] + f"\n\n[…{remaining:,} chars remaining — call again with offset={offset + max_chars}]"
            else:
                result = body
        elif name == "read_range":
            result = _read_range(
                arguments.get("file", ""),
                int(arguments.get("start_line", 1)),
                int(arguments.get("end_line", 50)),
            )
        elif name == "read_outline":
            result = _read_outline(arguments.get("file", ""))
        elif name == "symbol_callers":
            result = SoulClient._call("symbol_callers", {"name": arguments["name"]}) or "(no callers found)"
        elif name == "symbol_callees":
            result = SoulClient._call("symbol_callees", {"name": arguments["name"]}) or "(no callees found)"
        elif name == "symbol_insert_child":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _apply_symbol_insert_child(
                    arguments["file"], arguments["parent"],
                    arguments["position"], arguments["new_body"],
                ),
            )
        elif name == "pdf_read":
            result = rooms._tool_pdf_read(arguments)
        elif name == "doc_read":
            result = rooms._tool_doc_read(arguments)
        elif name == "lit_search_arxiv":
            result = LitSearch.arxiv(
                query=arguments["query"],
                max_results=int(arguments.get("max_results", 10)),
                sort_by=arguments.get("sort_by", "relevance"),
            )
        elif name == "lit_search_biorxiv":
            result = LitSearch.biorxiv(
                query=arguments["query"],
                start_date=arguments["start_date"],
                end_date=arguments["end_date"],
                server=arguments.get("server", "biorxiv"),
                max_results=int(arguments.get("max_results", 20)),
            )
        elif name == "lit_search_europepmc":
            result = LitSearch.europepmc(
                query=arguments["query"],
                max_results=int(arguments.get("max_results", 20)),
                open_access_only=bool(arguments.get("open_access_only", True)),
            )
        elif name == "lit_search_openalex":
            result = LitSearch.openalex(
                query=arguments["query"],
                entity_type=arguments.get("entity_type", "works"),
                max_results=int(arguments.get("max_results", 20)),
                filters=arguments.get("filters", ""),
            )
        elif name == "paper_fetch":
            result = WebSearch.paper_fetch(
                url_or_doi=arguments.get("url", arguments.get("doi", "")),
                pdf_path=arguments.get("pdf_path", ""),
                full_text=bool(arguments.get("full_text", False)),
            )
        elif name == "chitta_ingest":
            n = chitta_ingest(arguments["text"])
            result = f"chitta_ingest: wrote {n} memories"
        elif name == "doc_ingest":
            result = await _doc_ingest(
                source=arguments["source"],
                realm=arguments.get("realm", "research"),
                tags=arguments.get("tags") or [],
                model=arguments.get("model", "gpt-5.5"),
                dry_run=bool(arguments.get("dry_run", True)),
                max_memories=int(arguments.get("max_memories", 50)),
            )
        elif name == "reflib_add":
            result = RefLib.add(
                url_or_doi=arguments["url_or_doi"],
                tags=arguments.get("tags"),
                notes=arguments.get("notes", ""),
            )
        elif name == "reflib_search":
            result = RefLib.search(
                query=arguments.get("query", ""),
                tag=arguments.get("tag", ""),
                limit=int(arguments.get("limit", 20)),
            )
        elif name == "reflib_export":
            result = RefLib.export(
                fmt=arguments.get("fmt", "markdown"),
                tag=arguments.get("tag", ""),
                query=arguments.get("query", ""),
            )
        elif name == "reflib_remove":
            result = RefLib.remove(doi_or_title=arguments["doi_or_title"])
        elif name == "reflib_tag":
            result = RefLib.tag(
                doi_or_title=arguments["doi_or_title"],
                tags=arguments.get("tags", []),
                notes=arguments.get("notes", ""),
                replace=bool(arguments.get("replace", False)),
            )
        else:
            result = f"Unknown tool: {name}"

        # Truncate large responses to reduce token cost. Export/history tools are exempt.
        _no_truncate = {"codex_history", "local_history", "pdf_read", "paper_fetch",
                        "lit_search_arxiv", "lit_search_biorxiv", "lit_search_europepmc", "lit_search_openalex",
                        "reflib_export"}
        _max_chars = 12_000
        if name not in _no_truncate and isinstance(result, str) and len(result) > _max_chars:
            result = result[:_max_chars] + f"\n\n[truncated — {len(result) - _max_chars:,} chars omitted]"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def _run_exec_mode() -> None:
    """Single-shot exec mode: read JSON from stdin, call backend, write JSON to stdout.

    Input (stdin):
        {"backend": "claude"|"codex"|"local", "model": "...",
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
        print(json.dumps({"content": "", "error": "missing required field: backend (claude|codex|local)"}))
        return
    model = req.get("model")
    system = req.get("system", "")
    message = req.get("message", "")
    session_id = req.get("session_id")

    full_prompt = f"{system}\n\n{message}" if system else message
    base_url = req.get("base_url")

    try:
        if backend == "claude":
            content = await rooms._run_claude_p(full_prompt)
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


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>chitta-bridge · rooms</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0a0c10;--surface:#111520;--card:#161c28;--card-h:#1a2234;
  --border:#1e2940;--border-h:#2d3f5e;
  --accent:#4f8ef7;--accent-dim:#1a2d5a;
  --green:#34d399;--green-dim:#0d2e20;
  --yellow:#fbbf24;--yellow-dim:#2a1f07;
  --red:#f87171;--red-dim:#2a0d0d;
  --purple:#a78bfa;--purple-dim:#1e1540;
  --muted:#4a5568;--muted2:#64748b;--text:#dde4f0;--text2:#94a3b8;
  --rail-active:var(--green);--rail-stale:var(--yellow);
  --rail-synth:var(--purple);--rail-empty:#2d3748;
  --font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;
  --radius:10px;--radius-sm:6px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:dark}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;min-height:100vh;overflow-x:hidden}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-h);border-radius:3px}

/* ── Header ── */
header{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 20px;height:52px;display:flex;align-items:center;gap:12px;
  position:sticky;top:0;z-index:100;
}
.logo{font-weight:700;font-size:15px;letter-spacing:-.03em;white-space:nowrap}
.logo span{color:var(--accent)}
.logo sub{font-size:10px;font-weight:400;color:var(--muted2);vertical-align:middle;margin-left:4px}

#search-wrap{flex:1;max-width:320px;position:relative}
#search{
  width:100%;background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:6px 10px 6px 30px;
  color:var(--text);font-size:13px;font-family:var(--font);outline:none;
  transition:border-color .15s;
}
#search:focus{border-color:var(--accent)}
#search-wrap::before{
  content:'⌕';position:absolute;left:9px;top:50%;transform:translateY(-50%);
  color:var(--muted2);font-size:15px;pointer-events:none;
}

.filter-chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{
  background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:3px 10px;font-size:11px;font-weight:500;cursor:pointer;
  color:var(--muted2);transition:all .15s;white-space:nowrap;
}
.chip:hover,.chip.on{border-color:var(--accent);color:var(--text);background:var(--accent-dim)}
.chip.s-active.on{border-color:var(--green);color:var(--green);background:var(--green-dim)}
.chip.s-stale.on{border-color:var(--yellow);color:var(--yellow);background:var(--yellow-dim)}
.chip.s-synth.on{border-color:var(--purple);color:var(--purple);background:var(--purple-dim)}
.chip.s-empty.on{border-color:var(--muted2);color:var(--muted2)}

.hstats{margin-left:auto;display:flex;gap:16px;font-size:12px;color:var(--muted2);white-space:nowrap}
.hstats b{color:var(--text2)}

#sse-status{
  width:8px;height:8px;border-radius:50%;background:var(--muted);
  flex-shrink:0;transition:background .3s;
}
#sse-status.live{background:var(--green);animation:pulse 2.5s infinite}
#sse-status.error{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* ── Grid ── */
main{padding:20px;max-width:1600px;margin:0 auto}
#rooms-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:12px;
}

/* ── Card ── */
.room-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:0;cursor:pointer;transition:border-color .15s,transform .1s,box-shadow .15s;
  display:flex;position:relative;overflow:hidden;outline:none;
  border-left:none;
}
.room-card:hover,.room-card:focus{
  border-color:var(--border-h);transform:translateY(-1px);
  box-shadow:0 4px 20px #0006;
}
.room-card.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.card-rail{width:3px;flex-shrink:0;border-radius:var(--radius) 0 0 var(--radius)}
.s-active  .card-rail{background:var(--rail-active)}
.s-stale   .card-rail{background:var(--rail-stale)}
.s-synth   .card-rail{background:var(--rail-synth)}
.s-empty   .card-rail{background:var(--rail-empty)}
.card-body{padding:12px 14px;flex:1;min-width:0;display:flex;flex-direction:column;gap:6px}

.card-row1{display:flex;align-items:flex-start;gap:8px}
.card-topic{font-weight:600;font-size:13px;flex:1;line-height:1.4;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.status-dot{
  width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;
}
.s-active  .status-dot{background:var(--green)}
.s-stale   .status-dot{background:var(--yellow)}
.s-synth   .status-dot{background:var(--purple)}
.s-empty   .status-dot{background:var(--rail-empty)}

.card-chips{display:flex;flex-wrap:wrap;gap:4px}
.p-chip{
  border-radius:20px;padding:1px 7px;font-size:10px;font-weight:500;
  border:1px solid transparent;
}
.be-codex{background:#0d1f3c;border-color:#1e3a6e;color:#60a5fa}
.be-claude{background:#1a0b35;border-color:#3d1a7a;color:#c084fc}
.be-local{background:#1a0e00;border-color:#5c3300;color:#fb923c}
.be-opencode{background:#001a10;border-color:#005a30;color:#34d399}

.card-preview{
  font-size:11.5px;color:var(--muted2);line-height:1.45;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}
.card-verdict{
  font-size:11px;color:var(--purple);line-height:1.4;font-style:italic;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}
.card-meta{display:flex;gap:10px;font-size:11px;color:var(--muted);align-items:center}
.card-meta .dot{color:var(--border-h)}

/* ── Empty / loading states ── */
.state-msg{
  grid-column:1/-1;text-align:center;padding:60px 20px;color:var(--muted2);
}
.state-msg h2{font-size:15px;margin-bottom:6px;color:var(--text2)}
.state-msg p{font-size:13px}

/* ── Detail panel ── */
#detail{
  position:fixed;top:0;right:0;width:min(680px,100vw);height:100vh;
  background:var(--surface);border-left:1px solid var(--border);
  z-index:200;display:flex;flex-direction:column;
  transform:translateX(100%);transition:transform .2s ease;
}
#detail.open{transform:translateX(0)}
#detail-header{
  padding:14px 16px;border-bottom:1px solid var(--border);
  display:flex;align-items:flex-start;gap:10px;
  background:var(--surface);position:sticky;top:0;z-index:10;flex-shrink:0;
}
#detail-title-wrap{flex:1;min-width:0}
#detail-title{font-weight:700;font-size:14px;line-height:1.4;word-break:break-word}
#detail-subtitle{font-size:11px;color:var(--muted2);margin-top:2px}
#close-btn{
  background:none;border:1px solid var(--border);color:var(--muted2);
  border-radius:var(--radius-sm);padding:4px 8px;cursor:pointer;font-size:12px;
  flex-shrink:0;line-height:1.2;
}
#close-btn:hover{border-color:var(--text2);color:var(--text)}
#close-btn:focus{outline:2px solid var(--accent);outline-offset:2px}

#detail-body{overflow-y:auto;flex:1;padding:16px;display:flex;flex-direction:column;gap:16px}

.section-label{
  font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin-bottom:8px;
}

/* Synthesis verdict pinned */
.verdict-pin{
  background:var(--purple-dim);border:1px solid var(--purple);
  border-radius:var(--radius);padding:12px 14px;
}
.verdict-pin .verdict-label{
  font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--purple);margin-bottom:6px;display:flex;align-items:center;gap:6px;
}
.verdict-pin .verdict-text{font-size:13px;color:#d4b8ff;line-height:1.5}
.verdict-jump{
  font-size:11px;color:var(--purple);cursor:pointer;text-decoration:underline;
  background:none;border:none;padding:0;margin-top:4px;display:inline-block;
}

/* Participants */
.p-cards{display:flex;flex-direction:column;gap:8px}
.p-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:10px 12px;
}
.p-card-top{display:flex;align-items:baseline;gap:8px}
.p-name{font-weight:600;font-size:13px}
.p-meta-row{font-size:11px;color:var(--muted2);display:flex;gap:6px;flex-wrap:wrap}
.p-soul{
  margin-top:8px;font-size:11.5px;color:var(--text2);line-height:1.55;
  white-space:pre-wrap;border-top:1px solid var(--border);padding-top:8px;
  max-height:100px;overflow-y:auto;font-family:var(--font);
}
.p-soul-toggle{
  font-size:11px;color:var(--accent);cursor:pointer;background:none;
  border:none;padding:0;margin-top:4px;
}

/* Transcript */
#transcript-wrap{position:relative}
.msg{display:flex;gap:10px;animation:fadeUp .18s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
.msg+.msg{margin-top:10px}
.msg-av{
  width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:9px;font-weight:700;flex-shrink:0;
  text-transform:uppercase;margin-top:1px;
}
.av-codex{background:#0d1f3c;color:#93c5fd}
.av-claude{background:#1a0b35;color:#d8b4fe}
.av-local{background:#1a0e00;color:#fed7aa}
.av-opencode{background:#001a10;color:#6ee7b7}
.av-synth{background:var(--purple-dim);color:var(--purple)}
.av-system{background:#141e30;color:var(--muted2)}

.msg-right{flex:1;min-width:0}
.msg-who{display:flex;align-items:baseline;gap:6px;margin-bottom:3px}
.who-name{font-size:12px;font-weight:700;color:var(--text2)}
.who-ts{font-size:10px;color:var(--muted)}

.msg-content{font-size:12.5px;line-height:1.65;color:var(--text)}
.msg-content code{
  font-family:var(--mono);font-size:11.5px;background:#0d1829;
  border:1px solid var(--border);border-radius:3px;padding:1px 4px;
}
.msg-content pre{
  background:#0a1020;border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:10px 12px;overflow-x:auto;margin:6px 0;
}
.msg-content pre code{background:none;border:none;padding:0;font-size:11px;line-height:1.5}
.msg-content strong{color:#e8edf8;font-weight:600}
.msg-content em{color:var(--text2)}

/* Special message types */
.msg-moderator .msg-content{color:var(--yellow)}
.msg-moderator .who-name{color:var(--yellow)}
.msg-synth{
  background:var(--purple-dim);border:1px solid #5b3fa0;border-radius:var(--radius);
  padding:12px 14px;
}
.msg-synth .who-name{color:var(--purple)}
.msg-synth .msg-content{color:#d4b8ff}

/* New messages pill */
#new-pill{
  position:sticky;bottom:16px;left:50%;transform:translateX(-50%);
  background:var(--accent);color:#fff;border-radius:20px;padding:5px 14px;
  font-size:12px;font-weight:600;cursor:pointer;display:none;width:fit-content;
  box-shadow:0 4px 16px #0008;z-index:20;border:none;
}

/* Typing indicator */
.msg-typing .msg-content{display:flex;align-items:center;gap:3px;height:18px}
.typing-dot{
  width:5px;height:5px;border-radius:50%;background:var(--muted2);
  animation:typing 1.2s infinite;
}
.typing-dot:nth-child(2){animation-delay:.2s}
.typing-dot:nth-child(3){animation-delay:.4s}
@keyframes typing{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-4px)}}

/* Overlay */
#overlay{
  display:none;position:fixed;inset:0;background:#0009;z-index:190;
}
#overlay.show{display:block}
</style>
</head>
<body>
<header>
  <div id="sse-status" title="SSE disconnected" aria-label="Connection status: disconnected"></div>
  <div class="logo">chitta<span>-bridge</span> <sub>rooms</sub></div>
  <div id="search-wrap">
    <label for="search" class="sr-only">Search rooms</label>
    <input id="search" type="search" placeholder="Search rooms…" autocomplete="off" aria-label="Search rooms"/>
  </div>
  <div class="filter-chips" role="group" aria-label="Filter by status">
    <button class="chip s-active" data-filter="status:active">active</button>
    <button class="chip s-stale"  data-filter="status:stale">stale</button>
    <button class="chip s-synth"  data-filter="status:synthesized">synthesized</button>
    <button class="chip s-empty"  data-filter="status:empty">empty</button>
    <button class="chip" data-filter="be:codex">codex</button>
    <button class="chip" data-filter="be:claude">claude</button>
    <button class="chip" data-filter="be:opencode">opencode</button>
  </div>
  <div class="hstats" aria-live="polite">
    <span><b id="s-total">0</b> rooms</span>
    <span><b id="s-active">0</b> active</span>
    <span><b id="s-synth">0</b> synth</span>
  </div>
</header>

<main>
  <div id="rooms-grid" role="list" aria-label="Discussion rooms">
    <div class="state-msg"><h2>Loading rooms…</h2><p>Connecting to live stream</p></div>
  </div>
</main>

<div id="overlay" aria-hidden="true"></div>
<div id="detail" role="dialog" aria-modal="true" aria-labelledby="detail-title" hidden>
  <div id="detail-header">
    <div id="detail-title-wrap">
      <div id="detail-title">—</div>
      <div id="detail-subtitle"></div>
    </div>
    <button id="close-btn" aria-label="Close detail panel">✕</button>
  </div>
  <div id="detail-body">
    <div class="state-msg"><p>Loading…</p></div>
  </div>
</div>

<script>
'use strict';

// ── Markdown (escape-first, transform-second — no HTML injection) ──────────
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') }

function md(raw){
  if(!raw) return '';
  const lines = String(raw).split('\n');
  const out = [];
  let fence = null, fenceLang = '', fenceBuf = [];

  for(let i=0;i<lines.length;i++){
    const line = lines[i];
    if(fence){
      if(line.trimStart().startsWith(fence)){
        const lang = /^[a-z0-9_-]{1,24}$/i.test(fenceLang) ? ` class="language-${esc(fenceLang)}"` : '';
        out.push(`<pre><code${lang}>${fenceBuf.map(esc).join('\n')}</code></pre>`);
        fence=null; fenceLang=''; fenceBuf=[];
      } else { fenceBuf.push(line) }
      continue;
    }
    const fm = line.match(/^(`{3,}|~{3,})(\S*)/);
    if(fm){ fence=fm[1]; fenceLang=fm[2]; fenceBuf=[]; continue; }
    out.push(inlineMd(line));
  }
  if(fenceBuf.length){
    out.push(`<pre><code>${fenceBuf.map(esc).join('\n')}</code></pre>`);
  }
  return out.join('\n');
}

function inlineMd(line){
  // Tokenize backticks first so they are never re-processed
  const parts = [];
  let rest = line, m;
  const codeRe = /`([^`]+)`/g;
  let last = 0;
  codeRe.lastIndex = 0;
  while((m = codeRe.exec(line)) !== null){
    parts.push({t:'text', v: line.slice(last, m.index)});
    parts.push({t:'code', v: m[1]});
    last = m.index + m[0].length;
  }
  parts.push({t:'text', v: line.slice(last)});

  return parts.map(p => {
    if(p.t==='code') return `<code>${esc(p.v)}</code>`;
    // bold, italic on escaped text
    let s = esc(p.v);
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
    return s;
  }).join('');
}

// ── Constants ─────────────────────────────────────────────────────────────
const SYS = new Set(['TOPIC','CONTEXT','MODERATOR','⟳ Synthesizer']);
const BE_CLS = {codex:'be-codex',claude:'be-claude',local:'be-local',opencode:'be-opencode'};
const AV_CLS = n => {
  const l = n.toLowerCase();
  if(n==='⟳ Synthesizer') return 'av-synth';
  if(l.startsWith('codex')||l.startsWith('gpt')||l.startsWith('openai')) return 'av-codex';
  if(l.startsWith('claude')||l.startsWith('opus')) return 'av-claude';
  if(l.startsWith('local')||l.startsWith('ollama')) return 'av-local';
  if(l.startsWith('opencode')) return 'av-opencode';
  return 'av-system';
};

// ── State ─────────────────────────────────────────────────────────────────
let roomsById = {};
let roomOrder = [];
let hydratedTranscript = {};  // roomId → messages[]
let openRoomId = null;
let activeFilters = new Set();
let searchQ = '';
let stickyBottom = true;
let pendingNew = 0;
let tsInterval;

// ── Relative time ──────────────────────────────────────────────────────────
function relTime(ts){
  if(!ts) return '';
  const epoch = typeof ts==='number' ? ts*1000 : new Date(ts).getTime();
  if(!epoch) return '';
  const sec = (Date.now()-epoch)/1000;
  if(sec<60) return 'just now';
  if(sec<3600) return `${Math.floor(sec/60)}m ago`;
  if(sec<86400) return `${Math.floor(sec/3600)}h ago`;
  return `${Math.floor(sec/86400)}d ago`;
}
function fmtTs(ts){
  if(!ts) return '';
  try{ return new Date(ts).toLocaleString('en',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) }catch{ return String(ts) }
}
function initials(name){ const p=name.split(':'); return p[p.length-1].slice(0,2).toUpperCase() }

// ── Filter/search ──────────────────────────────────────────────────────────
function matchesFilter(r){
  if(activeFilters.size){
    let ok=false;
    for(const f of activeFilters){
      if(f.startsWith('status:') && r.status===f.slice(7)){ ok=true; break }
      if(f.startsWith('be:') && (r.participants||[]).some(p=>p.backend===f.slice(3))){ ok=true; break }
    }
    if(!ok) return false;
  }
  if(!searchQ) return true;
  const q=searchQ.toLowerCase();
  if((r.topic||'').toLowerCase().includes(q)) return true;
  if((r.id||'').toLowerCase().includes(q)) return true;
  if((r.participants||[]).some(p=>p.name.toLowerCase().includes(q))) return true;
  if((r.verdict||'').toLowerCase().includes(q)) return true;
  if((r.preview||'').toLowerCase().includes(q)) return true;
  return false;
}

function visibleRooms(){ return roomOrder.map(id=>roomsById[id]).filter(r=>r&&matchesFilter(r)) }

// ── Card rendering ─────────────────────────────────────────────────────────
function makeCardEl(r){
  const el = document.createElement('article');
  el.className = `room-card s-${r.status==='synthesized'?'synth':r.status}`;
  el.setAttribute('role','button');
  el.setAttribute('tabindex','0');
  el.setAttribute('aria-label', r.topic||r.id);
  el.dataset.roomId = r.id;

  const chips = (r.participants||[]).map(p=>`<span class="p-chip ${BE_CLS[p.backend]||''}">${esc(p.name.split(':')[0])}</span>`).join('');
  const preview = r.verdict
    ? `<div class="card-verdict">⟳ ${esc(r.verdict)}</div>`
    : r.preview ? `<div class="card-preview">${esc(r.preview)}</div>` : '';

  el.innerHTML = `
    <div class="card-rail"></div>
    <div class="card-body">
      <div class="card-row1">
        <div class="card-topic">${esc(r.topic||r.id)}</div>
        <div class="status-dot" title="${esc(r.status)}"></div>
      </div>
      <div class="card-chips">${chips}</div>
      ${preview}
      <div class="card-meta">
        <span>${r.turns||0} turns</span>
        <span class="dot">·</span>
        <span class="rel-ts" data-ts="${r.last_activity||r.created||''}">${relTime(r.last_activity||0)||fmtTs(r.created)}</span>
      </div>
    </div>`;
  return el;
}

function patchCard(r){
  const old = document.querySelector(`[data-room-id="${CSS.escape(r.id)}"]`);
  const el = makeCardEl(r);
  if(old){ old.replaceWith(el) } else { $grid.prepend(el) }
}

// ── Grid render ────────────────────────────────────────────────────────────
const $grid = document.getElementById('rooms-grid');

function renderGrid(){
  const rooms = visibleRooms();
  const active = roomOrder.map(id=>roomsById[id]).filter(r=>r&&r.status==='active').length;
  const synth  = roomOrder.map(id=>roomsById[id]).filter(r=>r&&r.status==='synthesized').length;
  document.getElementById('s-total').textContent = roomOrder.length;
  document.getElementById('s-active').textContent = active;
  document.getElementById('s-synth').textContent = synth;

  if(!rooms.length){
    $grid.innerHTML = `<div class="state-msg"><h2>${searchQ||activeFilters.size?'No matching rooms':'No rooms yet'}</h2><p>${searchQ?'Try a different search.':'Create a room with room_create.'}</p></div>`;
    return;
  }
  // Rebuild — needed for reorder; we do this only on filter/search changes
  // For live updates we use patchCard() to avoid blowing away all cards
  const frag = document.createDocumentFragment();
  rooms.forEach(r=>frag.appendChild(makeCardEl(r)));
  $grid.innerHTML='';
  $grid.appendChild(frag);
  if(openRoomId){
    const sel = document.querySelector(`[data-room-id="${CSS.escape(openRoomId)}"]`);
    if(sel) sel.classList.add('selected');
  }
}

// ── Detail panel ──────────────────────────────────────────────────────────
const $detail = document.getElementById('detail');
const $overlay = document.getElementById('overlay');
const $detailBody = document.getElementById('detail-body');

async function openDetail(id){
  openRoomId = id;
  location.hash = '#room/'+encodeURIComponent(id);
  document.querySelectorAll('.room-card').forEach(c=>c.classList.remove('selected'));
  const card = document.querySelector(`[data-room-id="${CSS.escape(id)}"]`);
  if(card) card.classList.add('selected');

  $detail.hidden=false;
  $detail.removeAttribute('hidden');
  $detail.classList.add('open');
  $overlay.classList.add('show');
  $overlay.setAttribute('aria-hidden','false');

  document.getElementById('detail-title').textContent = roomsById[id]?.topic||id;
  document.getElementById('detail-subtitle').textContent =
    `${id} · created ${fmtTs(roomsById[id]?.created)}`;
  $detailBody.innerHTML = '<div class="state-msg"><p>Loading…</p></div>';

  // Cold hydrate from /api/rooms/{id}
  if(!hydratedTranscript[id]){
    try{
      const r = await fetch('/api/rooms/'+encodeURIComponent(id)).then(x=>x.json());
      hydratedTranscript[id] = r.messages||[];
      // Merge slim data in case SSE hasn't caught up
      if(!roomsById[id]) roomsById[id] = {id};
      Object.assign(roomsById[id], {
        topic:r.topic, created:r.created,
        participants:r.participants, files:r.files, file_count:(r.files||[]).length
      });
    }catch(e){
      $detailBody.innerHTML=`<div class="state-msg"><h2>Failed to load</h2><p>${esc(String(e))}</p></div>`;
      return;
    }
  }
  renderDetail(id);
  // Restore hash-based open after render
  requestAnimationFrame(()=>scrollToBottom(true));
}

function closeDetail(){
  openRoomId = null;
  location.hash = '';
  $detail.classList.remove('open');
  $overlay.classList.remove('show');
  $overlay.setAttribute('aria-hidden','true');
  document.querySelectorAll('.room-card').forEach(c=>c.classList.remove('selected'));
  setTimeout(()=>{ $detail.hidden=true }, 200);
}

function renderDetail(id){
  const r = roomsById[id]||{};
  const msgs = hydratedTranscript[id]||[];
  const synth = msgs.find(m=>m.name==='⟳ Synthesizer');
  const ps = r.participants||[];
  const files = r.files||[];

  document.getElementById('detail-title').textContent = r.topic||id;
  document.getElementById('detail-subtitle').textContent =
    `${id} · ${(r.turns||0)} turns · created ${fmtTs(r.created)}`;

  const verdictHtml = synth ? `
    <div class="verdict-pin">
      <div class="verdict-label">⟳ Synthesis</div>
      <div class="verdict-text">${md(synth.content)}</div>
    </div>` : '';

  const pHtml = ps.map(p=>{
    const soul = p.soul||{};
    const prompt = typeof soul==='string'?soul:(soul.system_prompt||'');
    const bias = soul.challenge_bias!=null ? ` · challenge_bias ${soul.challenge_bias}` : '';
    const effort = p.effort ? ` · effort:${p.effort}` : '';
    return `<div class="p-card">
      <div class="p-card-top">
        <span class="p-name">${esc(p.name)}</span>
        <span class="p-chip ${BE_CLS[p.backend]||''}">${esc(p.backend||'')}</span>
      </div>
      <div class="p-meta-row"><span>${esc(p.model||'')}</span>${effort?`<span>${esc(effort)}</span>`:''}</div>
      ${prompt?`<div class="p-soul">${esc(prompt)}${bias?`\n\n${esc(bias)}`:''}</div>`:''}
    </div>`;
  }).join('');

  const filesHtml = files.length ? `
    <div>
      <div class="section-label">Attached files (${files.length})</div>
      <div style="display:flex;flex-wrap:wrap;gap:4px">
        ${files.map(f=>`<code style="font-size:11px;padding:2px 6px;background:var(--card);border:1px solid var(--border);border-radius:4px">${esc(f.split('/').pop())}</code>`).join('')}
      </div>
    </div>` : '';

  const nonCtx = msgs.filter(m=>m.name!=='CONTEXT'&&m.name!=='TOPIC');
  const transcriptHtml = nonCtx.length
    ? nonCtx.map(renderMsg).join('')
    : '<div class="state-msg" style="padding:20px 0"><p>No messages yet.</p></div>';

  $detailBody.innerHTML = `
    ${verdictHtml}
    <div>
      <div class="section-label">Participants (${ps.length})</div>
      <div class="p-cards">${pHtml}</div>
    </div>
    ${filesHtml}
    <div>
      <div class="section-label">Transcript · ${msgs.filter(m=>!SYS.has(m.name)).length} turns</div>
      <div id="transcript-wrap">
        ${transcriptHtml}
        <button id="new-pill" onclick="scrollToBottom(true)">↓ new messages</button>
      </div>
    </div>`;
  stickyBottom=true; pendingNew=0;
  $detailBody.addEventListener('scroll', onDetailScroll, {passive:true});
}

function renderMsg(m){
  const isSynth = m.name==='⟳ Synthesizer';
  const isMod = m.name==='MODERATOR';
  const cls = isSynth?'msg-synth':isMod?'msg-moderator':'';
  const av = AV_CLS(m.name);
  return `<div class="msg ${cls}">
    <div class="msg-av ${av}" aria-hidden="true">${esc(initials(m.name))}</div>
    <div class="msg-right">
      <div class="msg-who">
        <span class="who-name">${esc(m.name)}</span>
        <span class="who-ts" title="${esc(fmtTs(m.ts))}">${relTime(m.ts||0)||fmtTs(m.ts)}</span>
      </div>
      <div class="msg-content">${md(m.content||'')}</div>
    </div>
  </div>`;
}

// ── Scroll management ──────────────────────────────────────────────────────
function onDetailScroll(){
  const el = $detailBody;
  stickyBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 60;
  if(stickyBottom){ pendingNew=0; updatePill() }
}
function scrollToBottom(force=false){
  if(force||stickyBottom){
    $detailBody.scrollTop = $detailBody.scrollHeight;
    pendingNew=0; stickyBottom=true; updatePill();
  }
}
function updatePill(){
  const pill = document.getElementById('new-pill');
  if(pill){ pill.style.display = (!stickyBottom&&pendingNew>0)?'block':'none';
    pill.textContent = `↓ ${pendingNew} new`; }
}

function appendMsgToDetail(msg){
  const wrap = document.getElementById('transcript-wrap');
  if(!wrap) return;
  const div = document.createElement('div');
  div.innerHTML = renderMsg(msg);
  const pill = document.getElementById('new-pill');
  wrap.insertBefore(div, pill);
  if(stickyBottom){ scrollToBottom(true) }
  else { pendingNew++; updatePill() }
}

// ── SSE ────────────────────────────────────────────────────────────────────
let _es = null;
function connect(){
  if(_es){ _es.close() }
  const dot = document.getElementById('sse-status');
  dot.className='';
  _es = new EventSource('/events');
  _es.onopen = () => { dot.className='live'; dot.setAttribute('aria-label','Connection status: live') };
  _es.onmessage = e => {
    let data;
    try{ data=JSON.parse(e.data) }catch{ return }
    switch(data.type){
      case 'snapshot':
        roomsById={};roomOrder=[];
        (data.rooms||[]).forEach(r=>{ roomsById[r.id]=r; roomOrder.push(r.id) });
        renderGrid();
        checkHash();
        break;
      case 'room_new':
        if(!roomsById[data.room.id]) roomOrder.unshift(data.room.id);
        roomsById[data.room.id]=data.room;
        patchCard(data.room); updateStats();
        break;
      case 'room_meta':
        if(roomsById[data.room_id])
          Object.assign(roomsById[data.room_id], data.meta);
        patchCard(roomsById[data.room_id]); updateStats();
        break;
      case 'message':
        if(!hydratedTranscript[data.room_id]) break; // not open yet — skip
        if(data.index >= (hydratedTranscript[data.room_id].length||0)){
          hydratedTranscript[data.room_id].push(data.message);
          if(openRoomId===data.room_id) appendMsgToDetail(data.message);
        }
        break;
      case 'room_reset':
        delete hydratedTranscript[data.room_id];
        roomsById[data.room_id]=data.room;
        patchCard(data.room);
        if(openRoomId===data.room_id){ renderDetail(data.room_id) }
        break;
      case 'turn_start':
        if(openRoomId===data.room_id) showTyping(data.name);
        break;
      case 'turn_end':
        if(openRoomId===data.room_id) hideTyping();
        break;
      // legacy compat
      case 'update':
        if(!roomsById[data.room.id]) roomOrder.unshift(data.room.id);
        roomsById[data.room.id]=data.room;
        patchCard(data.room); updateStats();
        break;
    }
  };
  _es.onerror = () => {
    dot.className='error'; dot.setAttribute('aria-label','Connection status: disconnected');
    _es.close(); setTimeout(connect,3000);
  };
}

function updateStats(){
  document.getElementById('s-total').textContent=roomOrder.length;
  document.getElementById('s-active').textContent=roomOrder.filter(id=>roomsById[id]?.status==='active').length;
  document.getElementById('s-synth').textContent=roomOrder.filter(id=>roomsById[id]?.status==='synthesized').length;
}

function showTyping(name){
  let row = document.getElementById('typing-row');
  if(!row){
    const wrap=document.getElementById('transcript-wrap'); if(!wrap) return;
    row=document.createElement('div'); row.id='typing-row';
    row.className='msg msg-typing';
    row.innerHTML=`<div class="msg-av ${AV_CLS(name)}">${esc(initials(name))}</div>
      <div class="msg-right">
        <div class="msg-who"><span class="who-name">${esc(name)}</span></div>
        <div class="msg-content"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>
      </div>`;
    const pill=document.getElementById('new-pill');
    wrap.insertBefore(row, pill);
    scrollToBottom();
  }
}
function hideTyping(){ const r=document.getElementById('typing-row'); if(r) r.remove() }

// ── Hash routing ───────────────────────────────────────────────────────────
function checkHash(){
  const m = location.hash.match(/^#room\/(.+)/);
  if(m){ const id=decodeURIComponent(m[1]); if(roomsById[id]) openDetail(id) }
}
window.addEventListener('hashchange', checkHash);

// ── Keyboard nav ───────────────────────────────────────────────────────────
function focusedCardIndex(){
  const cards=[...document.querySelectorAll('.room-card')];
  return cards.indexOf(document.activeElement);
}
document.addEventListener('keydown', e=>{
  if(e.key==='Escape'){ closeDetail(); return }
  if(e.target===document.getElementById('search')) return;
  const cards=[...document.querySelectorAll('.room-card')];
  if(!cards.length) return;
  const idx=focusedCardIndex();
  if(e.key==='ArrowDown'){ e.preventDefault(); cards[Math.min(idx+1,cards.length-1)]?.focus() }
  if(e.key==='ArrowUp'){ e.preventDefault(); cards[Math.max(idx-1,0)]?.focus() }
  if(e.key==='Enter'&&idx>=0){ openDetail(cards[idx].dataset.roomId) }
});

// ── Delegated click / keyboard ─────────────────────────────────────────────
document.getElementById('rooms-grid').addEventListener('click', e=>{
  const card=e.target.closest('.room-card');
  if(card) openDetail(card.dataset.roomId);
});
document.getElementById('rooms-grid').addEventListener('keydown', e=>{
  if(e.key===' '||e.key==='Enter'){
    const card=e.target.closest('.room-card');
    if(card){ e.preventDefault(); openDetail(card.dataset.roomId) }
  }
});
document.getElementById('close-btn').addEventListener('click', closeDetail);
document.getElementById('overlay').addEventListener('click', closeDetail);

// ── Filters ────────────────────────────────────────────────────────────────
document.querySelectorAll('.chip[data-filter]').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const f=btn.dataset.filter;
    if(activeFilters.has(f)){ activeFilters.delete(f); btn.classList.remove('on') }
    else { activeFilters.add(f); btn.classList.add('on') }
    renderGrid();
  });
});

// ── Search ─────────────────────────────────────────────────────────────────
let searchDebounce;
document.getElementById('search').addEventListener('input', e=>{
  clearTimeout(searchDebounce);
  searchDebounce=setTimeout(()=>{ searchQ=e.target.value.trim(); renderGrid() }, 150);
});

// ── Relative timestamp refresh ─────────────────────────────────────────────
function refreshTs(){
  document.querySelectorAll('.rel-ts[data-ts]').forEach(el=>{
    const ts=el.dataset.ts;
    if(ts) el.textContent=relTime(parseFloat(ts)||ts)||fmtTs(ts);
  });
  // also update detail timestamps
  if(openRoomId){
    document.querySelectorAll('.who-ts[title]').forEach(el=>{
      const title=el.getAttribute('title');
      if(title) el.textContent=relTime(title)||title;
    });
  }
}
setInterval(refreshTs, 30000);

// ── Bootstrap ─────────────────────────────────────────────────────────────
connect();
// Fallback: if SSE never fires, load from /api/rooms
setTimeout(()=>{
  if(!roomOrder.length){
    fetch('/api/rooms').then(r=>r.json()).then(rooms=>{
      if(roomOrder.length) return; // SSE beat us
      rooms.forEach(r=>{ roomsById[r.id]=r; roomOrder.push(r.id) });
      renderGrid(); checkHash();
    }).catch(()=>{});
  }
}, 3000);
</script>
</body>
</html>"""


async def _start_dashboard(port: int = 7680) -> None:
    """Serve the rooms dashboard on http://localhost:{port}.

    SSE protocol — slim snapshot + incremental deltas:
      {type:'snapshot',   rooms:[<slim>]}
      {type:'room_new',   room:<slim>}
      {type:'message',    room_id, index, message}
      {type:'room_meta',  room_id, meta}
      {type:'room_reset', room_id, room:<slim>}
      {type:'turn_start', room_id, name}
      {type:'turn_end',   room_id}
    Full transcript via /api/rooms/{id} only (cold hydration on first open).
    """
    from aiohttp import web
    import asyncio
    import json as _json
    import time

    rooms_dir = Path.home() / ".chitta-bridge" / "rooms"
    _sse_queues: list[asyncio.Queue] = []
    _SYS = {"TOPIC", "CONTEXT", "MODERATOR", "⟳ Synthesizer"}
    STALE_AFTER = 600.0

    def _load_room(path: Path) -> dict | None:
        try:
            with open(path) as f:
                d = _json.load(f)
            d.setdefault("id", path.stem)
            return d
        except Exception:
            return None

    def _ts_epoch(ts) -> float:
        if ts is None:
            return 0.0
        if isinstance(ts, (int, float)):
            return float(ts)
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    def _msgs(room: dict) -> list:
        m = room.get("messages")
        return m if isinstance(m, list) else []

    def _turns(room: dict) -> list:
        return [m for m in _msgs(room) if m.get("name") not in _SYS]

    def _last_activity(room: dict) -> float:
        epochs = [_ts_epoch(m.get("ts")) for m in _msgs(room)]
        epochs.append(_ts_epoch(room.get("created")))
        return max(epochs) if epochs else 0.0

    def _verdict(room: dict) -> str:
        for m in reversed(_msgs(room)):
            if m.get("name") == "⟳ Synthesizer":
                lines = (m.get("content") or "").strip().splitlines()
                return (lines[0] if lines else "").replace("**", "")[:200]
        return ""

    def _status(room: dict) -> str:
        msgs = _msgs(room)
        if any(m.get("name") == "⟳ Synthesizer" for m in msgs):
            return "synthesized"
        if not _turns(room):
            return "empty"
        if (time.time() - _last_activity(room)) > STALE_AFTER:
            return "stale"
        return "active"

    def _slim(room: dict) -> dict:
        turns = _turns(room)
        last = turns[-1] if turns else None
        preview = ""
        if last:
            preview = (last.get("content") or "").replace("**", "").strip()[:140]
        parts = [
            {"name": p.get("name", ""), "backend": p.get("backend", ""),
             "model": p.get("model", ""), "effort": p.get("effort", "")}
            if isinstance(p, dict) else {"name": str(p), "backend": "", "model": "", "effort": ""}
            for p in (room.get("participants") or [])
        ]
        return {
            "id": room.get("id"),
            "topic": room.get("topic") or room.get("id"),
            "created": room.get("created"),
            "participants": parts,
            "file_count": len(room.get("files") or []),
            "turns": len(turns),
            "messages": len(_msgs(room)),
            "status": _status(room),
            "verdict": _verdict(room),
            "preview": preview,
            "last_activity": _last_activity(room),
            "last_author": (last.get("name") if last else ""),
        }

    def _meta(slim: dict) -> dict:
        return {k: slim[k] for k in ("status", "verdict", "preview", "turns",
                                      "messages", "last_activity", "last_author")}

    def _all_slim() -> list[dict]:
        out = []
        for p in sorted(rooms_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            r = _load_room(p)
            if r:
                out.append(_slim(r))
        return out

    async def _emit(payload: dict) -> None:
        if not _sse_queues:
            return
        msg = _json.dumps(payload)
        for q in list(_sse_queues):
            await q.put(msg)

    async def _watch_rooms():
        seen_mtime: dict[str, float] = {}
        seen_count: dict[str, int] = {}
        seen_meta: dict[str, dict] = {}
        seen_pending: dict[str, str] = {}
        for p in rooms_dir.glob("*.json"):
            r = _load_room(p)
            if not r:
                continue
            rid = r["id"]
            seen_mtime[str(p)] = p.stat().st_mtime
            seen_count[rid] = len(_msgs(r))
            seen_meta[rid] = _meta(_slim(r))
            seen_pending[rid] = r.get("pending") or ""
        while True:
            await asyncio.sleep(2)
            try:
                for p in rooms_dir.glob("*.json"):
                    mtime = p.stat().st_mtime
                    if seen_mtime.get(str(p)) == mtime:
                        continue
                    seen_mtime[str(p)] = mtime
                    r = _load_room(p)
                    if not r:
                        continue
                    rid = r["id"]
                    slim = _slim(r)
                    msgs = _msgs(r)
                    prev = seen_count.get(rid)
                    if prev is None:
                        await _emit({"type": "room_new", "room": slim})
                    elif len(msgs) < prev:
                        await _emit({"type": "room_reset", "room_id": rid, "room": slim})
                    elif len(msgs) > prev:
                        for i in range(prev, len(msgs)):
                            await _emit({"type": "message", "room_id": rid,
                                         "index": i, "message": msgs[i]})
                    pending = r.get("pending") or ""
                    if pending != seen_pending.get(rid, ""):
                        if pending:
                            await _emit({"type": "turn_start", "room_id": rid, "name": pending})
                        else:
                            await _emit({"type": "turn_end", "room_id": rid})
                        seen_pending[rid] = pending
                    meta = _meta(slim)
                    if meta != seen_meta.get(rid):
                        await _emit({"type": "room_meta", "room_id": rid, "meta": meta})
                        seen_meta[rid] = meta
                    seen_count[rid] = len(msgs)
            except Exception:
                pass

    async def handle_index(request):
        return web.Response(text=_DASHBOARD_HTML, content_type="text/html")

    async def handle_rooms(request):
        return web.Response(text=_json.dumps(_all_slim()), content_type="application/json")

    async def handle_room(request):
        rid = request.match_info["id"]
        path = rooms_dir / f"{rid}.json"
        r = _load_room(path) if path.exists() else None
        if r is None:
            return web.Response(status=404, text="not found")
        return web.Response(text=_json.dumps(r), content_type="application/json")

    async def handle_events(request):
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })
        await resp.prepare(request)
        await resp.write(
            f"data: {_json.dumps({'type': 'snapshot', 'rooms': _all_slim()})}\n\n".encode())
        q: asyncio.Queue = asyncio.Queue()
        _sse_queues.append(q)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25)
                    await resp.write(f"data: {msg}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": ping\n\n")
        except Exception:
            pass
        finally:
            if q in _sse_queues:
                _sse_queues.remove(q)
        return resp

    _dash_token = _http_token()

    @web.middleware
    async def _auth_middleware(request, handler):
        import hmac as _hmac
        bearer = ""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            bearer = auth[7:]
        qp = request.query.get("token", "")
        cookie = request.cookies.get("cb_token", "")
        if not any(c and _hmac.compare_digest(c, _dash_token) for c in (bearer, qp, cookie)):
            return web.Response(
                status=401,
                text="Unauthorized — open /?token=<token> (token in ~/.chitta-bridge/token)",
            )
        resp = await handler(request)
        if qp and not cookie:
            # First visit via ?token= — set a cookie so the page's own
            # /api and /events requests authenticate without the query param.
            try:
                resp.set_cookie("cb_token", _dash_token, httponly=True, samesite="Strict")
            except Exception:
                pass
        return resp

    app = web.Application(middlewares=[_auth_middleware])
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/rooms", handle_rooms)
    app.router.add_get("/api/rooms/{id}", handle_room)
    app.router.add_get("/events", handle_events)

    async def _try_bind() -> bool:
        nonlocal runner
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        try:
            await site.start()
            return True
        except OSError as _e:
            await runner.cleanup()
            if _e.errno != 98:
                raise
            return False

    runner = None
    if not await _try_bind():
        # Port in use — if the HTTP daemon owns it, defer to it and skip
        if not _evict_port(port, allow_http=True):
            return  # held by HTTP daemon or unrelated process — skip silently
        await asyncio.sleep(1.2)
        if not await _try_bind():
            return  # still can't bind — give up silently

    asyncio.create_task(_watch_rooms())


def _evict_port(port: int, *, allow_http: bool = True) -> bool:
    """SIGTERM any chitta-bridge process holding *port*.

    If allow_http=True (default), never evicts the persistent HTTP daemon
    (identified by '--http' in its cmdline). Returns True only if something
    was actually evicted.
    """
    import subprocess
    import os
    try:
        pids = [
            int(p) for p in
            subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True).stdout.split()
            if p.strip().isdigit()
        ]
    except Exception:
        return False
    evicted = False
    for pid in pids:
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\x00", " ")
            # Match this server's own entrypoint, not the bare substring
            # "chitta" — that would also kill chittad, the chitta CLI, or any
            # other user tool with "chitta" in its path.
            if "chitta-bridge" not in cmd and "chitta_bridge" not in cmd:
                continue
            if allow_http and "--http" in cmd:
                continue  # never kill the persistent HTTP daemon
            os.kill(pid, 15)
            evicted = True
        except Exception:
            pass
    return evicted


def _make_init_options() -> "InitializationOptions":
    return InitializationOptions(
        server_name="chitta-bridge",
        server_version=__version__,
        capabilities=ServerCapabilities(tools=ToolsCapability()),
        instructions=(
            "## Multi-model discussions — use rooms\n"
            "For any discussion involving multiple models (e.g. GPT + Claude), always use "
            "room_create (via advanced gateway) with participant shorthands:\n"
            "  codex:<model-id>               — Codex backend, default effort\n"
            "  codex:<model-id>:medium        — Codex backend, medium reasoning effort\n"
            "  codex:<model-id>:xhigh         — Codex backend, extended reasoning\n"
            "  claude:<model-id>              — Claude API, default effort\n"
            "  claude:<model-id>:xhigh        — Claude with extended thinking (xhigh/max only)\n"
            "Effort: codex=low/medium/high/xhigh; claude=low/medium/xhigh/max. "
            "NOTE: 'high' is NOT valid for claude backends — use xhigh.\n"
            "Use current model IDs from CLAUDE.md or ask the user — never hardcode versions here.\n"
            "Then run with room_run.\n\n"
            "## room_run always needs prompt= — CRITICAL\n"
            "Every room_run call — initial AND follow-up — MUST include prompt=. "
            "Without it the room produces 0 responses.\n"
            "  room_run(room_id='room-xxx', prompt='Your full brief or question here')\n"
            "DO NOT call room_run without prompt hoping a prior message was queued. "
            "rounds defaults to 1 for follow-ups.\n\n"
            "## Room → Workflow pattern (design then execute)\n"
            "Rooms design. Workflows execute. Never describe a workflow in prose — call Workflow().\n"
            "  1. room_create + room_run(prompt=<full brief>) — get the design\n"
            "  2. Synthesize room output into a JS Workflow script\n"
            "  3. Workflow(script=<the script>) — actually makes the changes\n"
            "Skipping step 3 and describing the workflow in text is a failure mode.\n\n"
            "## Codex session reuse\n"
            "Prefer codex_discuss over codex_start when a session already exists. "
            "Never call codex_start unless the user asks for a new session or specific model.\n\n"
            "## File Attachments — CRITICAL\n"
            "The 'files' parameter in discuss, review, and similar tools "
            "MUST be an array, even for a single file.\n"
            "WRONG: files: \"/path/to/file.hpp\"\n"
            "CORRECT: files: [\"/path/to/file.hpp\"]"
        ),
    )


def _write_private(path: Path, text: str) -> None:
    """Write `text` to `path` with mode 0600 atomically from creation.

    write_text()+chmod() leaves a window where the file is world/group
    readable under the prevailing umask — on a shared NFS home that exposes
    the bearer token. O_CREAT|O_EXCL-style creation with mode 0600 closes it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    # Re-assert mode in case the file pre-existed with looser perms.
    os.chmod(str(path), 0o600)


def _http_token() -> str:
    """Load or create the shared bearer token for HTTP mode."""
    import secrets as _sec
    token_path = Path.home() / ".chitta-bridge" / "token"
    env = os.environ.get("CHITTA_BRIDGE_TOKEN", "").strip()
    if env:
        return env
    if token_path.exists():
        t = token_path.read_text().strip()
        if t:
            return t
    t = _sec.token_urlsafe(32)
    _write_private(token_path, t)
    return t


async def _run_http_mode(mcp_port: int = 7681, dashboard_port: int = 7680) -> None:
    """Run MCP over SSE (shared persistent server) + dashboard, both evicting stale bridges."""
    import hmac as _hmac
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import Response, PlainTextResponse

    _token = _http_token()
    # Propagate token to subprocesses (Codex, OpenCode) so they can connect back
    # to this bridge's HTTP SSE endpoint without spawning a new stdio bridge.
    os.environ["CHITTA_BRIDGE_TOKEN"] = _token

    def _auth_ok(request) -> bool:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return _hmac.compare_digest(auth[7:], _token)
        # Also allow token as query param for SSE clients that can't set headers
        return _hmac.compare_digest(request.query_params.get("token", ""), _token)

    init_options = _make_init_options()
    sse_transport = SseServerTransport("/messages/")

    # Streamable HTTP transport (MCP ≥ 1.0 preferred transport — Codex uses this)
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    session_manager = StreamableHTTPSessionManager(
        app=server, stateless=True, json_response=False,
    )

    async def handle_sse(request):
        if not _auth_ok(request):
            return PlainTextResponse("Unauthorized", status_code=401)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
        return Response()

    class _MCPApp:
        # Non-function Route endpoints are used as raw ASGI apps (no
        # request_response wrapper) — the session manager sends the full
        # response itself; a wrapped endpoint returning Response() afterwards
        # double-sends and logs "Unexpected ASGI message ... already completed".
        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return
            if not _auth_ok(_Request(scope, receive)):
                await PlainTextResponse("Unauthorized", status_code=401)(scope, receive, send)
                return
            await session_manager.handle_request(scope, receive, send)

    async def lifespan(app):
        async with session_manager.run():
            yield

    from starlette.requests import Request as _Request

    async def _messages_guard(scope, receive, send):
        # POST-back channel for SSE clients. The session_id is an unguessable
        # UUID disclosed only over the authenticated SSE stream, so it acts as
        # the capability; still reject explicit bad credentials and requests
        # that carry neither a token nor a session_id.
        if scope["type"] == "http":
            req = _Request(scope, receive)
            auth = req.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                if not _hmac.compare_digest(auth[7:], _token):
                    await PlainTextResponse("Unauthorized", status_code=401)(scope, receive, send)
                    return
            elif not _auth_ok(req) and not req.query_params.get("session_id"):
                await PlainTextResponse("Unauthorized", status_code=401)(scope, receive, send)
                return
        await sse_transport.handle_post_message(scope, receive, send)

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/mcp", endpoint=_MCPApp(), methods=["GET", "POST", "DELETE"]),
            Mount("/messages/", app=_messages_guard),
        ],
        lifespan=lifespan,
    )

    # Evict stale bridge on MCP port if needed
    for port, label in ((mcp_port, "MCP"), (dashboard_port, "dashboard")):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        in_use = sock.connect_ex(("127.0.0.1", port)) == 0
        sock.close()
        if in_use:
            if _evict_port(port):
                await asyncio.sleep(1.5)

    # Write port file so other tools can discover us (token included for auth).
    # 0600 from creation — it carries the bearer token.
    port_file = Path.home() / ".chitta-bridge" / "http.ports"
    _write_private(
        port_file,
        f"mcp={mcp_port}\ndashboard={dashboard_port}\npid={os.getpid()}\ntoken={_token}\n",
    )

    # Start scheduler daemon
    import chitta_bridge.server as _self_mod
    from chitta_bridge.scheduler import SchedulerService, JOBS_YAML
    _scheduler = SchedulerService(
        jobs_yaml=JOBS_YAML,
        bridge_tools={
            "codex_bin": str(CODEX_BIN) if CODEX_BIN else "codex",
            "claude_bin": str(CLAUDE_BIN) if CLAUDE_BIN else "claude",
            "room_manager": rooms,
            "bridge_url": f"http://127.0.0.1:{mcp_port}",
            "bridge_token": _token,
        },
        slack_fn=None,   # wire Slack MCP here when available
        chitta_fn=SoulClient.remember if SoulClient.is_available() else None,
    )
    await _scheduler.start()
    _self_mod._active_scheduler = _scheduler

    # Start dashboard and MCP SSE concurrently
    await _start_dashboard(port=dashboard_port)

    config = uvicorn.Config(starlette_app, host="127.0.0.1", port=mcp_port,
                            log_level="warning", access_log=False)
    userver = uvicorn.Server(config)
    print(f"chitta-bridge HTTP mode: MCP SSE on :{mcp_port}, dashboard on :{dashboard_port}",
          flush=True)

    # Use _serve() directly to avoid uvicorn's signal handler installation,
    # which would exit the process on SIGTERM from unrelated bridge instances.
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _on_signal():
        userver.should_exit = True
        stop.set()

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    try:
        await userver._serve()
    finally:
        await _scheduler.stop()
        for sig in (_signal.SIGTERM, _signal.SIGINT):
            loop.remove_signal_handler(sig)


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exec", action="store_true",
                        help="Single-shot mode: read JSON from stdin, write JSON to stdout")
    parser.add_argument("--http", action="store_true",
                        help="HTTP mode: shared persistent MCP SSE server (no stdio)")
    parser.add_argument("--mcp-port", type=int, default=7681,
                        help="MCP SSE port in --http mode (default: 7681)")
    parser.add_argument("--dashboard-port", type=int, default=7680,
                        help="Dashboard port (default: 7680)")
    args, _ = parser.parse_known_args()

    if args.exec:
        asyncio.run(_run_exec_mode())
        return

    if args.http:
        asyncio.run(_run_http_mode(mcp_port=args.mcp_port, dashboard_port=args.dashboard_port))
        return

    # Stdio mode (default) — one bridge per Claude session
    async def run():
        await _start_dashboard(port=args.dashboard_port)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, _make_init_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
