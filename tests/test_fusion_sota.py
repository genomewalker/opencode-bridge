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
        # "cited", not "grounded": the score counts citation-shaped tokens, unverified.
        tag = self.rm._tag_for({"name": "Alice", "content": "x", "citation_score": 3})
        assert tag == " [cited:3]"

    def test_summary_message_empty_tag(self):
        # SUMMARY is system context, not a participant claim — must not be tagged
        # [asserted], which would launder the grounded turns it compressed.
        assert self.rm._tag_for({"name": "SUMMARY", "content": "x", "citation_score": 0}) == ""


# ---------------------------------------------------------------------------
# Gap 1: _detect_plurality
# ---------------------------------------------------------------------------

class TestDetectPlurality:
    """Semantic (LLM-grouped) plurality; the classifier pass is mocked."""
    def setup_method(self):
        self.rm = RoomManager.__new__(RoomManager)

    def _run(self, room):
        return asyncio.run(self.rm._detect_plurality(room))

    def test_fewer_than_three_msgs_returns_all_as_majority(self):
        # returns before the LLM call — no mock needed
        room = _make_room([_msg("Alice", "foo"), _msg("Bob", "bar")])
        maj, min_, summ = self._run(room)
        assert len(maj) == 2 and min_ == [] and summ == ""

    def test_semantic_grouping_maps_names(self):
        room = _make_room([_msg("Alice", "use A"), _msg("Bob", "A is right"),
                           _msg("Carol", "no, use B")])
        self.rm._cheap_llm_call = AsyncMock(
            return_value='{"majority": ["Alice", "Bob"], "minority": ["Carol"]}')
        maj, min_, summ = self._run(room)
        assert {m["name"] for m in maj} == {"Alice", "Bob"}
        assert [m["name"] for m in min_] == ["Carol"]
        assert "Carol" not in summ

    def test_unanimous_returns_empty_minority(self):
        room = _make_room([_msg("Alice", "42"), _msg("Bob", "42"), _msg("Carol", "42")])
        self.rm._cheap_llm_call = AsyncMock(
            return_value='{"majority": ["Alice", "Bob", "Carol"], "minority": []}')
        _, min_, _ = self._run(room)
        assert min_ == []

    def test_classifier_error_falls_back_safely(self):
        room = _make_room([_msg("Alice", "x"), _msg("Bob", "y"), _msg("Carol", "z")])
        self.rm._cheap_llm_call = AsyncMock(side_effect=RuntimeError("no daemon"))
        maj, min_, summ = self._run(room)
        assert len(maj) == 3 and min_ == [] and summ == ""  # never breaks a fusion

    def test_skips_system_and_poison_messages(self):
        room = _make_room([
            {"name": "MODERATOR", "content": "ignored", "ts": _ts()},
            {"name": "Alice", "content": "real answer", "ts": _ts(), "citation_score": 0},
            {"name": "Bob", "content": "different view", "ts": _ts(), "citation_score": 0,
             "poison": True},
        ])
        maj, min_, _ = self._run(room)
        # Only Alice real (MODERATOR skipped, Bob poison) → <3 → fallback, no LLM call
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

        # _detect_plurality is now an async LLM pass — mock it to split the real room
        # (Alice+Bob majority, Carol minority) so the filter path is exercised.
        by_name = {m["name"]: m for m in room.messages}
        self.rm._detect_plurality = AsyncMock(return_value=(
            [by_name["Alice"], by_name["Bob"]],
            [by_name["Carol"]],
            "[Alice] majority view",
        ))
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


class TestInteractionTaxNote:
    """Advisory fires only for a heterogeneous panel in dense/debate topology."""
    _hetero = [{"backend": "claude"}, {"backend": "codex"}]
    _homo = [{"backend": "claude"}, {"backend": "claude"}]

    def test_fires_hetero_dense_multiround(self):
        note = RoomManager._interaction_tax_note(self._hetero, sparse_topology=False,
                                                 rounds=2, challenge=False)
        assert note and "interaction-tax" in note and "claude" in note and "codex" in note

    def test_fires_hetero_challenge(self):
        assert RoomManager._interaction_tax_note(self._hetero, False, 1, challenge=True)

    def test_silent_when_sparse(self):
        assert RoomManager._interaction_tax_note(self._hetero, sparse_topology=True,
                                                 rounds=3, challenge=True) is None

    def test_silent_when_homogeneous(self):
        assert RoomManager._interaction_tax_note(self._homo, False, 3, challenge=True) is None

    def test_silent_single_round_no_challenge(self):
        assert RoomManager._interaction_tax_note(self._hetero, False, 1, challenge=False) is None


class TestDisagreeStance:
    """_DISAGREE_RE keys on stance, not bare connectives."""
    def test_connective_agreement_is_not_disagreement(self):
        # bare "however"/"but" must no longer flag agreement as dissent
        assert not RoomManager._DISAGREE_RE.search("However, I fully agree with that.")
        assert not RoomManager._DISAGREE_RE.search("Good point, but also worth noting.")

    def test_real_disagreement_still_detected(self):
        for s in ("I disagree with that claim.",
                  "That is incorrect.",
                  "The measurement contradicts the hypothesis.",
                  "This reasoning is flawed."):
            assert RoomManager._DISAGREE_RE.search(s), s


class TestResolutionTierScoping:
    """resolution_tier is scoped to the question, not the whole message."""
    def test_question_without_own_citation_is_unresolvable(self):
        rm = RoomManager.__new__(RoomManager)
        msg = {"name": "A", "content": (
            "We cite https://ex.io/a for the method. "
            "The effect of temperature on the reaction rate remains unresolved.")}
        qs = rm._extract_open_questions([msg])
        # question itself has no citation → unresolvable, despite the URL earlier in the turn
        assert qs and qs[0]["resolution_tier"] == "unresolvable"
