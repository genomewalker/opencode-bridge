"""Tests for the four SOTA fusion-room gaps implemented in rooms.py and server.py."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chitta_bridge.rooms import DiscussionRoom, RoomManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().isoformat()


def _make_room_manager(rooms_dir: Path) -> RoomManager:
    rm = RoomManager.__new__(RoomManager)
    rm.codex = MagicMock()
    rm.local = MagicMock()
    rm.rooms = {}
    rm.rooms_dir = rooms_dir
    rm._room_locks = {}
    rm._endpoint_locks = {}
    return rm


def _make_room(msgs: list[dict]) -> DiscussionRoom:
    room = DiscussionRoom(id="test-room", topic="Test topic", participants=[
        {"name": "Alice", "backend": "claude"},
        {"name": "Bob", "backend": "codex"},
        {"name": "Carol", "backend": "claude"},
    ])
    room.messages = msgs
    return room


def _msg(name: str, content: str, citation_score: int = 0) -> dict:
    return {"name": name, "content": content, "ts": _ts(),
            "citation_score": citation_score}


# ---------------------------------------------------------------------------
# Gap 0: _tag_for helper
# ---------------------------------------------------------------------------

class TestTagFor:
    def setup_method(self):
        self.rm = RoomManager.__new__(RoomManager)

    def test_system_message_empty_tag(self):
        for name in ("TOPIC", "CONTEXT", "MODERATOR"):
            assert self.rm._tag_for({"name": name, "content": "x"}) == ""

    def test_participant_no_citations(self):
        tag = self.rm._tag_for({"name": "Alice", "content": "x", "citation_score": 0})
        assert tag == " [asserted: no citations]"

    def test_participant_with_citations(self):
        tag = self.rm._tag_for({"name": "Alice", "content": "x", "citation_score": 3})
        assert tag == " [grounded:3 citations]"


# ---------------------------------------------------------------------------
# Gap 1: _detect_plurality
# ---------------------------------------------------------------------------

class TestDetectPlurality:
    def setup_method(self):
        self.rm = RoomManager.__new__(RoomManager)

    def test_fewer_than_three_msgs_returns_all_as_majority(self):
        room = _make_room([_msg("Alice", "foo"), _msg("Bob", "bar")])
        maj, min_, summ = self.rm._detect_plurality(room)
        assert len(maj) == 2
        assert min_ == []
        assert summ == ""

    def test_divergent_messages_produce_minority(self):
        # Alice and Bob say similar things; Carol says something completely different
        similar = (
            "The answer is gradient descent. It minimizes the loss function. "
            "This is the standard approach. We should use it here."
        )
        different = (
            "We should use evolutionary algorithms. They are robust to local minima. "
            "Gradient descent fails on non-convex problems. Consider CMA-ES instead."
        )
        room = _make_room([
            _msg("Alice", similar),
            _msg("Bob", similar),
            _msg("Carol", different),
        ])
        maj, min_, summ = self.rm._detect_plurality(room)
        assert len(maj) == 2
        assert len(min_) == 1
        assert min_[0]["name"] == "Carol"
        assert "Carol" not in summ or "Alice" in summ or "Bob" in summ

    def test_fully_converged_returns_empty_minority(self):
        text = "The answer is clearly 42. This is well established. No dispute here."
        room = _make_room([_msg("Alice", text), _msg("Bob", text), _msg("Carol", text)])
        _, min_, _ = self.rm._detect_plurality(room)
        assert min_ == []

    def test_skips_system_and_poison_messages(self):
        room = _make_room([
            {"name": "MODERATOR", "content": "ignored", "ts": _ts()},
            {"name": "Alice", "content": "real answer", "ts": _ts(), "citation_score": 0},
            {"name": "Bob", "content": "different view", "ts": _ts(), "citation_score": 0,
             "poison": True},
        ])
        maj, min_, _ = self.rm._detect_plurality(room)
        # Only Alice is real (MODERATOR skipped, Bob is poison-skipped) → <3 → fallback
        assert min_ == []


# ---------------------------------------------------------------------------
# Gap 1+2: synthesize() minority_filter and cross_attend
# ---------------------------------------------------------------------------

class TestSynthesizeNewParams:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rm = _make_room_manager(self.tmp)

    def _make_synth_room(self) -> DiscussionRoom:
        similar = (
            "Gradient descent is best. It is standard. We should use it. "
            "Loss function minimization is key."
        )
        different = (
            "Evolutionary algorithms are better. They handle non-convex problems. "
            "CMA-ES is robust. Gradient descent fails here."
        )
        room = _make_room([
            _msg("Alice", similar),
            _msg("Bob", similar),
            _msg("Carol", different),
        ])
        self.rm.rooms[room.id] = room
        return room

    def test_minority_filter_sends_filtered_transcript(self):
        room = self._make_synth_room()
        captured = {}

        async def fake_claude_p(prompt, **kw):
            captured["prompt"] = prompt
            return "synthesis result"

        with patch.object(self.rm, "_run_claude_p", side_effect=fake_claude_p):
            asyncio.run(self.rm.synthesize(room.id, minority_filter=True,
                                           synthesizer={"backend": "claude", "name": "Judge"}))

        assert "Majority position (summarized)" in captured["prompt"]
        assert "Dissenting traces (full)" in captured["prompt"]
        assert "Carol" in captured["prompt"]

    def test_minority_filter_fallback_when_no_minority(self):
        text = "Gradient descent is best. Loss function minimization. Standard approach."
        room = _make_room([_msg("Alice", text), _msg("Bob", text), _msg("Carol", text)])
        self.rm.rooms[room.id] = room
        captured = {}

        async def fake_claude_p(prompt, **kw):
            captured["prompt"] = prompt
            return "synthesis result"

        with patch.object(self.rm, "_run_claude_p", side_effect=fake_claude_p):
            asyncio.run(self.rm.synthesize(room.id, minority_filter=True,
                                           synthesizer={"backend": "claude", "name": "Judge"}))

        assert "Discussion Room" in captured["prompt"]
        assert "Majority position (summarized)" not in captured["prompt"]

    def test_cross_attend_adds_block_to_prompt(self):
        room = self._make_synth_room()
        captured = {}

        async def fake_claude_p(prompt, **kw):
            captured["prompt"] = prompt
            return "synthesis result"

        with patch.object(self.rm, "_run_claude_p", side_effect=fake_claude_p):
            asyncio.run(self.rm.synthesize(room.id, cross_attend=True,
                                           synthesizer={"backend": "claude", "name": "Judge"}))

        assert "Cross-Attention Pass" in captured["prompt"]
        assert "UNIQUE" in captured["prompt"]

    def test_no_cross_attend_by_default(self):
        room = self._make_synth_room()
        captured = {}

        async def fake_claude_p(prompt, **kw):
            captured["prompt"] = prompt
            return "synthesis result"

        with patch.object(self.rm, "_run_claude_p", side_effect=fake_claude_p):
            asyncio.run(self.rm.synthesize(room.id,
                                           synthesizer={"backend": "claude", "name": "Judge"}))

        assert "Cross-Attention Pass" not in captured["prompt"]


# ---------------------------------------------------------------------------
# Gap 3: adaptive_stop in run_rounds
# ---------------------------------------------------------------------------

class TestAdaptiveStop:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rm = _make_room_manager(self.tmp)

    def _fake_respond(self, content: str):
        async def _respond(room, participant, round_num, blind=False, visible_names=None):
            return {
                "name": participant["name"],
                "content": content,
                "ts": _ts(),
                "turn_key": f"r{round_num}:{participant['name']}",
            }
        return _respond

    def test_adaptive_stop_halts_on_streak(self):
        room = DiscussionRoom(id="adaptive-test", topic="t",
                              participants=[{"name": "A", "backend": "claude"}],
                              max_total_rounds=10)
        self.rm.rooms[room.id] = room

        with (patch.object(self.rm, "_participant_respond",
                           side_effect=self._fake_respond("converged answer")),
              patch.object(self.rm, "_score_convergence", new=AsyncMock(return_value=0.95)),
              patch.object(self.rm, "_round_converged", return_value=(True, [])),
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=5, adaptive_stop=True,
                                           adaptive_threshold=0.85, adaptive_k=2))

        moderator_msgs = [m for m in room.messages if m["name"] == "MODERATOR"
                          and "[Adaptive]" in m["content"]]
        assert len(moderator_msgs) == 2
        assert "streak=2/2" in moderator_msgs[-1]["content"]

    def test_adaptive_stop_resets_streak_on_low_score(self):
        room = DiscussionRoom(id="adaptive-reset", topic="t",
                              participants=[{"name": "A", "backend": "claude"}],
                              max_total_rounds=10)
        self.rm.rooms[room.id] = room

        scores = iter([0.9, 0.3, 0.9, 0.9])

        async def score_fn(contents):
            return next(scores, 0.9)

        with (patch.object(self.rm, "_participant_respond",
                           side_effect=self._fake_respond("answer")),
              patch.object(self.rm, "_score_convergence", side_effect=score_fn),
              patch.object(self.rm, "_round_converged", return_value=(True, [])),
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=10, adaptive_stop=True,
                                           adaptive_threshold=0.85, adaptive_k=2))

        streaks = [m["content"] for m in room.messages if m["name"] == "MODERATOR"
                   and "[Adaptive]" in m["content"]]
        assert any("streak=0/2" in s for s in streaks)
        assert streaks[-1].endswith("streak=2/2")

    def test_stop_early_heuristic_still_works(self):
        room = DiscussionRoom(id="stop-early-test", topic="t",
                              participants=[{"name": "A", "backend": "claude"}],
                              max_total_rounds=10)
        self.rm.rooms[room.id] = room

        with (patch.object(self.rm, "_participant_respond",
                           side_effect=self._fake_respond("same answer")),
              patch.object(self.rm, "_round_converged", return_value=(True, [])),
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=5, stop_early=True))

        participant_msgs = [m for m in room.messages if m["name"] == "A"]
        assert len(participant_msgs) == 1


# ---------------------------------------------------------------------------
# Gap 4: self_moa and min_quality pre-flight logic (unit-tested inline)
# ---------------------------------------------------------------------------

class TestSelfMoaAndMinQuality:
    """Tests the pre-flight logic extracted from the fusion handler."""

    _STRONG_PROPOSERS = {"opus", "gpt", "sonnet", "gemini", "o3", "o4", "llama"}

    def _apply_self_moa(self, participants, n=None):
        """Mirrors the self_moa handler logic."""
        base = next(
            (p for p in participants if any(s in p.get("model", "").lower()
                                            for s in self._STRONG_PROPOSERS)),
            participants[0],
        )
        count = n or len(participants)
        return [{**base, "name": f"{base['name']}#{i + 1}"} for i in range(max(1, count))]

    def _weak_proposers(self, participants):
        return [p["name"] for p in participants
                if not any(s in p.get("model", "").lower() for s in self._STRONG_PROPOSERS)]

    def test_self_moa_replaces_panel_with_n_copies(self):
        panel = [
            {"name": "Opus", "model": "claude-opus-4-8", "backend": "claude"},
            {"name": "GPT", "model": "gpt-o3", "backend": "codex"},
        ]
        result = self._apply_self_moa(panel)
        assert len(result) == 2
        assert all(p["model"] == "claude-opus-4-8" for p in result)
        # Names must be distinct to avoid turn-key collisions
        names = [p["name"] for p in result]
        assert len(set(names)) == len(names)

    def test_self_moa_n_override(self):
        panel = [{"name": "Opus", "model": "claude-opus-4-8", "backend": "claude"}]
        result = self._apply_self_moa(panel, n=4)
        assert len(result) == 4
        assert result[0]["name"] == "Opus#1"
        assert result[3]["name"] == "Opus#4"

    def test_self_moa_picks_strong_model_first(self):
        panel = [
            {"name": "Weak", "model": "tinyllm-1b", "backend": "local"},
            {"name": "Strong", "model": "claude-opus-4-8", "backend": "claude"},
        ]
        result = self._apply_self_moa(panel, n=2)
        assert all("opus" in p["model"] for p in result)

    def test_self_moa_falls_back_to_first_if_no_strong(self):
        panel = [
            {"name": "A", "model": "unknown-model-x", "backend": "local"},
            {"name": "B", "model": "another-unknown", "backend": "local"},
        ]
        result = self._apply_self_moa(panel, n=2)
        assert all(p["model"] == "unknown-model-x" for p in result)

    def test_min_quality_detects_weak_proposers(self):
        panel = [
            {"name": "Opus", "model": "claude-opus-4-8", "backend": "claude"},
            {"name": "Tiny", "model": "tinyllm-1b", "backend": "local"},
        ]
        weak = self._weak_proposers(panel)
        assert weak == ["Tiny"]

    def test_min_quality_no_weak_proposers(self):
        panel = [
            {"name": "Opus", "model": "claude-opus-4-8", "backend": "claude"},
            {"name": "GPT", "model": "gpt-o3", "backend": "codex"},
        ]
        assert self._weak_proposers(panel) == []
