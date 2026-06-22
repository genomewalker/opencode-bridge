"""Acceptance tests for the MCP tool dispatch table.

Fix 3 assertion: there must be exactly one write-capable soul_remember surface
registered in the tool list (i.e. a single Tool entry named 'soul_remember').
"""

import asyncio

import pytest


class TestSoulToolRegistry:
    """Verify tool dispatch table entries for soul memory tools."""

    def _get_tools(self):
        # list_tools() is an async MCP handler; invoke it directly.
        import chitta_bridge.server as srv
        # The handler is registered via @server.list_tools(); retrieve it by
        # calling the coroutine that was decorated.
        return asyncio.run(srv.list_tools())

    def test_exactly_one_soul_remember(self):
        tools = self._get_tools()
        names = [t.name for t in tools]
        remember_entries = [n for n in names if n == "soul_remember"]
        assert len(remember_entries) == 1, (
            f"Expected exactly 1 soul_remember tool, found {len(remember_entries)}: {remember_entries}"
        )

    def test_exactly_one_soul_recall(self):
        tools = self._get_tools()
        names = [t.name for t in tools]
        recall_entries = [n for n in names if n == "soul_recall"]
        assert len(recall_entries) == 1, (
            f"Expected exactly 1 soul_recall tool, found {len(recall_entries)}: {recall_entries}"
        )

    def test_soul_remember_requires_content(self):
        tools = self._get_tools()
        soul_remember = next((t for t in tools if t.name == "soul_remember"), None)
        assert soul_remember is not None
        required = soul_remember.inputSchema.get("required", [])
        assert "content" in required, "soul_remember must require 'content' parameter"

    def test_soul_remember_is_write_capable(self):
        """soul_remember must NOT appear in _READER_TOOLS (read-only set)."""
        from chitta_bridge.rooms import _READER_TOOLS
        assert "soul_remember" not in _READER_TOOLS, (
            "soul_remember is write-capable and must not appear in _READER_TOOLS"
        )

    def test_no_duplicate_tool_names(self):
        tools = self._get_tools()
        names = [t.name for t in tools]
        seen = set()
        dupes = []
        for n in names:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        assert not dupes, f"Duplicate tool names in registry: {dupes}"
