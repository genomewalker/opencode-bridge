"""Acceptance tests for room audit ledger (_append_room_audit fixes)."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chitta_bridge.server import (
    DiscussionRoom,
    RoomManager,
    _append_room_audit,
)


# ---------------------------------------------------------------------------
# Direct _append_room_audit tests
# ---------------------------------------------------------------------------

class TestAppendRoomAudit:
    def test_creates_audit_jsonl(self, tmp_path):
        record = {
            "audit_id": "abc123",
            "room_id": "r1",
            "round_num": 1,
            "participant": "Alice",
            "backend": "opencode",
            "model": "gpt-4.1",
            "timestamp": 1234567890.0,
            "system_prompt_sha256": "deadbeef",
            "user_msg_sha256": "cafebabe",
            "tool_calls": [],
            "memory_injection": False,
            "unsourced": False,
            "usage": {},
        }
        _append_room_audit(tmp_path, "r1", "Alice", 1, record)
        audit_path = tmp_path / "r1.audit.jsonl"
        assert audit_path.exists(), "audit.jsonl file must be created"
        lines = [l for l in audit_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["audit_id"] == "abc123"
        assert row["participant"] == "Alice"
        assert row["backend"] == "opencode"

    def test_appends_multiple_records(self, tmp_path):
        for i in range(3):
            _append_room_audit(tmp_path, "r1", "Bob", i + 1, {"seq": i})
        lines = (tmp_path / "r1.audit.jsonl").read_text().splitlines()
        assert len(lines) == 3

    def test_required_fields_present(self, tmp_path):
        record = {
            "audit_id": "xyz",
            "participant": "Charlie",
            "backend": "codex",
            "model": "o4",
            "round_num": 1,
        }
        _append_room_audit(tmp_path, "room42", "Charlie", 1, record)
        row = json.loads((tmp_path / "room42.audit.jsonl").read_text().strip())
        for field in ("audit_id", "participant", "backend"):
            assert field in row, f"audit record must contain '{field}'"


# ---------------------------------------------------------------------------
# Integration: participant_respond writes audit file + unsourced flag
# ---------------------------------------------------------------------------

def _make_room_manager(rooms_dir: Path) -> RoomManager:
    """Build a RoomManager with all bridges mocked out."""
    opencode = MagicMock()
    codex = MagicMock()
    local = MagicMock()
    rm = RoomManager.__new__(RoomManager)
    rm.opencode = opencode
    rm.codex = codex
    rm.local = local
    rm.rooms = {}
    rm.rooms_dir = rooms_dir
    rm._room_locks = {}
    rm._endpoint_locks = {}
    return rm


def _make_participant(name: str = "Alice", backend: str = "opencode") -> dict:
    return {"name": name, "backend": backend, "model": "test-model"}


def _make_room(room_id: str, participant_name: str = "Alice") -> DiscussionRoom:
    return DiscussionRoom(
        id=room_id,
        topic="What is 2+2?",
        participants=[_make_participant(participant_name)],
    )


class TestParticipantRespondAudit:
    """_participant_respond must write a well-formed audit record."""

    def test_audit_file_created_after_round(self, tmp_path):
        rm = _make_room_manager(tmp_path)
        room = _make_room("test-room-1")
        participant = _make_participant()

        with patch.object(rm, "_send_to_backend", new=AsyncMock(return_value="The answer is 4.")), \
             patch.object(rm, "_build_thread_context", return_value=("system", "user")), \
             patch.object(rm, "_extract_tool_call", return_value=None), \
             patch.object(rm, "_extract_final_response", return_value="The answer is 4."), \
             patch.object(rm, "_parse_soul", return_value=None):
            result = asyncio.get_event_loop().run_until_complete(
                rm._participant_respond(room, participant, round_num=1)
            )

        audit_path = tmp_path / "test-room-1.audit.jsonl"
        assert audit_path.exists(), "audit.jsonl must be created after a completed round"
        row = json.loads(audit_path.read_text().strip())
        assert "audit_id" in row
        assert row["participant"] == "Alice"
        assert row["backend"] == "opencode"

    def test_unsourced_flag_set_when_path_present_no_tool_calls(self, tmp_path):
        rm = _make_room_manager(tmp_path)
        room = _make_room("test-room-2")
        participant = _make_participant()

        # Response contains a file path pattern — no tool calls
        fabricated = "See /path/to/file for details."

        with patch.object(rm, "_send_to_backend", new=AsyncMock(return_value=fabricated)), \
             patch.object(rm, "_build_thread_context", return_value=("system", "user")), \
             patch.object(rm, "_extract_tool_call", return_value=None), \
             patch.object(rm, "_extract_final_response", return_value=fabricated), \
             patch.object(rm, "_parse_soul", return_value=None):
            asyncio.get_event_loop().run_until_complete(
                rm._participant_respond(room, participant, round_num=1)
            )

        audit_path = tmp_path / "test-room-2.audit.jsonl"
        row = json.loads(audit_path.read_text().strip())
        assert row["unsourced"] is True, (
            "unsourced must be True when response contains path-like text but no tool calls"
        )

    def test_unsourced_false_for_plain_response(self, tmp_path):
        rm = _make_room_manager(tmp_path)
        room = _make_room("test-room-3")
        participant = _make_participant()

        plain = "The answer is four."

        with patch.object(rm, "_send_to_backend", new=AsyncMock(return_value=plain)), \
             patch.object(rm, "_build_thread_context", return_value=("system", "user")), \
             patch.object(rm, "_extract_tool_call", return_value=None), \
             patch.object(rm, "_extract_final_response", return_value=plain), \
             patch.object(rm, "_parse_soul", return_value=None):
            asyncio.get_event_loop().run_until_complete(
                rm._participant_respond(room, participant, round_num=1)
            )

        audit_path = tmp_path / "test-room-3.audit.jsonl"
        row = json.loads(audit_path.read_text().strip())
        assert row["unsourced"] is False
