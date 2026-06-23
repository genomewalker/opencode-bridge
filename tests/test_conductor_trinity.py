"""Tests for conductor_fusion, room_set_visibility, ROLE_PRESETS, dual-track stop, and soul routing."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from chitta_bridge.rooms import DiscussionRoom, RoomManager, ROLE_PRESETS, ROLE_PROMPTS


def _ts():
    return datetime.now().isoformat()


def _make_rm(rooms_dir):
    rm = RoomManager.__new__(RoomManager)
    rm.codex = MagicMock()
    rm.local = MagicMock()
    rm.rooms = {}
    rm.rooms_dir = rooms_dir
    rm._room_locks = {}
    rm._endpoint_locks = {}
    return rm


def _make_room(participants, roles=None, visibility=None):
    room = DiscussionRoom(
        id="test", topic="t",
        participants=participants,
        roles=roles or {},
        visibility=visibility or {},
    )
    return room


def _msg(name, content, citation_score=0):
    return {"name": name, "content": content, "ts": _ts(), "citation_score": citation_score}


# ---------------------------------------------------------------------------
# ROLE_PRESETS
# ---------------------------------------------------------------------------

class TestRolePresets:
    def test_all_trinity_roles_defined(self):
        assert "thinker" in ROLE_PRESETS
        assert "worker" in ROLE_PRESETS
        assert "verifier" in ROLE_PRESETS

    def test_each_preset_has_prompt_and_scope(self):
        for key, preset in ROLE_PRESETS.items():
            assert "prompt" in preset, f"{key} missing prompt"
            assert "visibility_scope" in preset, f"{key} missing visibility_scope"

    def test_thinker_scope_is_all(self):
        assert ROLE_PRESETS["thinker"]["visibility_scope"] == "all"

    def test_worker_scope_is_role_list(self):
        scope = ROLE_PRESETS["worker"]["visibility_scope"]
        assert isinstance(scope, list)
        assert any("role:thinker" in s for s in scope)

    def test_verifier_scope_excludes_verifiers(self):
        scope = ROLE_PRESETS["verifier"]["visibility_scope"]
        assert "all_except_role:verifier" in scope

    def test_legacy_roles_still_valid(self):
        for key in ("skeptic", "empiricist", "advocate", "devils_advocate"):
            assert key in ROLE_PROMPTS


# ---------------------------------------------------------------------------
# _resolve_vis — visibility matrix + role presets
# ---------------------------------------------------------------------------

class TestResolveVis:
    def setup_method(self):
        self.rm = RoomManager.__new__(RoomManager)

    def _room(self, roles=None, visibility=None, participants=None):
        r = DiscussionRoom(id="x", topic="t",
                           participants=participants or [
                               {"name": "A"}, {"name": "B"}, {"name": "C"}
                           ],
                           roles=roles or {}, visibility=visibility or {})
        return r

    def test_no_matrix_no_flags_returns_none(self):
        room = self._room()
        assert self.rm._resolve_vis(room, "A", 1, False, False, 0) is None

    def test_sparse_topology_returns_blind(self):
        room = self._room()
        assert self.rm._resolve_vis(room, "A", 1, True, False, 0) == frozenset()

    def test_blind_first_round_only_round_0(self):
        room = self._room()
        assert self.rm._resolve_vis(room, "A", 1, False, True, 0) == frozenset()
        assert self.rm._resolve_vis(room, "A", 2, False, True, 1) is None

    def test_explicit_matrix_overrides_flags(self):
        room = self._room(visibility={1: {"A": "all"}})
        assert self.rm._resolve_vis(room, "A", 1, True, True, 0) is None  # matrix wins

    def test_explicit_matrix_none_means_blind(self):
        room = self._room(visibility={1: {"A": "none"}})
        assert self.rm._resolve_vis(room, "A", 1, False, False, 0) == frozenset()

    def test_explicit_matrix_list(self):
        room = self._room(visibility={1: {"A": ["B"]}})
        result = self.rm._resolve_vis(room, "A", 1, False, False, 0)
        assert result == frozenset({"B"})

    def test_thinker_role_returns_none(self):
        room = self._room(roles={"A": "thinker"})
        assert self.rm._resolve_vis(room, "A", 1, False, False, 0) is None

    def test_worker_sees_only_thinkers(self):
        parts = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        room = self._room(roles={"A": "thinker", "B": "worker", "C": "worker"},
                          participants=parts)
        result = self.rm._resolve_vis(room, "B", 1, False, False, 0)
        assert result == frozenset({"A"})

    def test_verifier_excludes_other_verifiers(self):
        parts = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        room = self._room(roles={"A": "worker", "B": "verifier", "C": "verifier"},
                          participants=parts)
        result = self.rm._resolve_vis(room, "B", 1, False, False, 0)
        assert "C" not in result
        assert "A" in result
        assert "B" in result  # sees itself (own messages in transcript)


# ---------------------------------------------------------------------------
# Per-participant preamble
# ---------------------------------------------------------------------------

class TestPerParticipantPreamble:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rm = _make_rm(self.tmp)

    def test_preamble_override_injected_for_named_participant(self):
        room = DiscussionRoom(id="p-test", topic="t",
                              participants=[{"name": "Alice", "backend": "claude"}],
                              preamble="shared preamble",
                              preambles={"Alice": "alice specific"})
        self.rm.rooms[room.id] = room
        captured = {}

        async def fake_backend(participant, user_msg, system_prompt=None, **kw):
            captured["system"] = system_prompt or user_msg
            return "response"

        with patch.object(self.rm, "_send_to_backend", side_effect=fake_backend):
            asyncio.run(self.rm._participant_respond(room, {"name": "Alice", "backend": "claude"}, round_num=1))

        assert "system" in captured
        assert "alice specific" in captured["system"]
        assert "shared preamble" not in captured["system"]

    def test_shared_preamble_used_when_no_override(self):
        room = DiscussionRoom(id="p-test2", topic="t",
                              participants=[{"name": "Bob", "backend": "claude"}],
                              preamble="shared",
                              preambles={"Alice": "alice specific"})
        self.rm.rooms[room.id] = room
        captured = {}

        async def fake_backend(participant, user_msg, system_prompt=None, **kw):
            captured["system"] = system_prompt or user_msg
            return "response"

        with patch.object(self.rm, "_send_to_backend", side_effect=fake_backend):
            asyncio.run(self.rm._participant_respond(room, {"name": "Bob", "backend": "claude"}, round_num=1))

        assert "system" in captured
        assert "shared" in captured["system"]


# ---------------------------------------------------------------------------
# Dual-track stop: both ledger AND haiku score must agree
# ---------------------------------------------------------------------------

class TestDualTrackStop:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rm = _make_rm(self.tmp)

    def _fake_respond(self, content):
        async def _r(room, participant, round_num, blind=False, visible_names=None):
            return {"name": participant["name"], "content": content,
                    "ts": _ts(), "turn_key": f"r{round_num}:{participant['name']}"}
        return _r

    def test_adaptive_stop_and_stop_early_together_raise(self):
        room = DiscussionRoom(id="conflict-test", topic="t",
                              participants=[{"name": "A", "backend": "claude"}])
        self.rm.rooms[room.id] = room
        import pytest as _pytest
        with _pytest.raises(ValueError, match="mutually exclusive"):
            asyncio.run(self.rm.run_rounds(room.id, rounds=2,
                                           adaptive_stop=True, stop_early=True))

    def test_scorer_error_logs_err_and_does_not_advance_streak(self):
        """score=None (scorer crash) → MODERATOR logs ERR, streak stays 0."""
        room = DiscussionRoom(id="err-test", topic="t",
                              participants=[{"name": "A", "backend": "claude"}],
                              max_total_rounds=10, verbatim_rounds=0)
        self.rm.rooms[room.id] = room

        with (patch.object(self.rm, "_participant_respond", side_effect=self._fake_respond("x")),
              patch.object(self.rm, "_score_convergence", new=AsyncMock(return_value=None)),
              patch.object(self.rm, "_round_converged", return_value=(True, [])),
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=3,
                                           adaptive_stop=True, adaptive_threshold=0.85, adaptive_k=2))

        adaptive_msgs = [m["content"] for m in room.messages
                         if m["name"] == "MODERATOR" and "[Adaptive]" in m["content"]]
        assert all("convergence=ERR" in m for m in adaptive_msgs)
        assert all("streak=0/" in m for m in adaptive_msgs)
        assert len([m for m in room.messages if m.get("name") == "A"]) == 3  # all rounds ran

    def test_dual_track_requires_both_signals(self):
        """High haiku score alone (converged=False from ledger) should NOT stop early — streak never advances."""
        room = DiscussionRoom(id="dual-test", topic="t",
                              participants=[{"name": "A", "backend": "claude"}],
                              max_total_rounds=10, verbatim_rounds=0)  # disable compression
        self.rm.rooms[room.id] = room

        async def score_fn(contents):
            return 0.95  # haiku says converged every round

        with (patch.object(self.rm, "_participant_respond", side_effect=self._fake_respond("new claim each round")),
              patch.object(self.rm, "_score_convergence", side_effect=score_fn),
              patch.object(self.rm, "_round_converged", return_value=(False, [])),  # ledger says NOT converged
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=4,
                                           adaptive_stop=True, adaptive_threshold=0.85, adaptive_k=2))

        # Streak should never advance past 0 (ledger never converged)
        adaptive_msgs = [m["content"] for m in room.messages
                         if m["name"] == "MODERATOR" and "[Adaptive]" in m["content"]]
        assert all("streak=0/" in m for m in adaptive_msgs)
        # All 4 rounds ran (no compression, so all turn_keys present)
        participant_msgs = [m for m in room.messages if m.get("name") == "A"]
        assert len(participant_msgs) == 4

    def test_dual_track_stops_when_both_agree(self):
        """Both haiku high AND ledger converged → streak advances → stop."""
        room = DiscussionRoom(id="dual-both", topic="t",
                              participants=[{"name": "A", "backend": "claude"}],
                              max_total_rounds=10)
        self.rm.rooms[room.id] = room

        with (patch.object(self.rm, "_participant_respond", side_effect=self._fake_respond("same")),
              patch.object(self.rm, "_score_convergence", new=AsyncMock(return_value=0.95)),
              patch.object(self.rm, "_round_converged", return_value=(True, [])),
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=10,
                                           adaptive_stop=True, adaptive_threshold=0.85, adaptive_k=2))

        moderator_adaptive = [m for m in room.messages
                              if m["name"] == "MODERATOR" and "[Adaptive]" in m["content"]]
        assert any("streak=2/2" in m["content"] for m in moderator_adaptive)
        assert len([m for m in room.messages if m.get("name") == "A"]) == 2


# ---------------------------------------------------------------------------
# Adaptive role reassignment
# ---------------------------------------------------------------------------

class TestAdaptiveRoleReassignment:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rm = _make_rm(self.tmp)

    def test_low_neff_reassigns_weakest_to_verifier(self):
        parts = [{"name": "A", "backend": "claude"}, {"name": "B", "backend": "claude"}]
        room = DiscussionRoom(id="adapt-role", topic="t", participants=parts, max_total_rounds=10)
        self.rm.rooms[room.id] = room
        call_n = {"n": 0}

        async def fake_respond(r, participant, round_num, blind=False, visible_names=None):
            call_n["n"] += 1
            return {"name": participant["name"], "content": "same content",
                    "ts": _ts(), "turn_key": f"r{round_num}:{participant['name']}",
                    "citation_score": 0}

        low_div = {"N_eff": 1.2, "warning": "low", "claim_overlap": 0.9,
                   "backend_collisions": [], "N_participants": 2}

        with (patch.object(self.rm, "_participant_respond", side_effect=fake_respond),
              patch.object(self.rm, "_compute_diversity", return_value=low_div),
              patch.object(self.rm, "_save_room", return_value=None)):
            asyncio.run(self.rm.run_rounds(room.id, rounds=2))

        reassignment_msgs = [m for m in room.messages
                             if m["name"] == "MODERATOR" and "[Adaptive Role]" in m["content"]]
        assert len(reassignment_msgs) >= 1
        assert any(p["name"] for p in parts if room.roles.get(p["name"]) == "verifier")


# ---------------------------------------------------------------------------
# conductor_fusion workflow compilation (unit-tested inline)
# ---------------------------------------------------------------------------

class TestConductorFusionCompilation:
    """Test the workflow→preambles+visibility compilation logic, extracted inline."""

    def _compile(self, workflow, rounds=1):
        from chitta_bridge.server import _normalize_participant_shorthands, _display_name_for
        name_counts = {}
        participants = []
        preambles = {}
        vis_per_step = []
        for step in workflow:
            agent_raw = step.get("agent", "claude:sonnet")
            norm = _normalize_participant_shorthands([agent_raw])
            p = norm[0] if norm else {"name": agent_raw, "backend": "claude"}
            base = step.get("name") or _display_name_for(agent_raw)
            p["_agent_raw"] = agent_raw
            name_counts[base] = name_counts.get(base, 0) + 1
            cnt = name_counts[base]
            p["name"] = f"{base}#{cnt}" if cnt > 1 else base
            participants.append(p)
            preambles[p["name"]] = step.get("subtask", "")
            sees = step.get("sees", "all")
            vis_per_step.append({p["name"]: sees})
        name_lookup: dict[str, str] = {}
        for p in participants:
            nm = p["name"]
            name_lookup[nm.lower()] = nm
            name_lookup.setdefault(nm.split("#")[0].lower(), nm)
            raw = p.pop("_agent_raw", "")
            if raw:
                name_lookup[raw.lower()] = nm
                raw_parts = raw.split(":")
                if len(raw_parts) > 1:
                    name_lookup.setdefault(raw_parts[1].lower(), nm)

        def _resolve(s):
            if not isinstance(s, list):
                return s
            return [name_lookup.get(e.lower(), e) for e in s]

        vis_per_step = [{k: _resolve(v) for k, v in d.items()} for d in vis_per_step]
        visibility = {r: {k: v for d in vis_per_step for k, v in d.items()}
                      for r in range(1, rounds + 1)}
        return participants, preambles, visibility

    def test_unique_agents_get_clean_names(self):
        wf = [
            {"agent": "claude:opus", "subtask": "Propose", "sees": "none"},
            {"agent": "claude:sonnet", "subtask": "Implement", "sees": ["Opus"]},
        ]
        parts, preambles, vis = self._compile(wf)
        names = [p["name"] for p in parts]
        assert len(set(names)) == 2
        assert preambles[names[0]] == "Propose"
        assert preambles[names[1]] == "Implement"

    def test_duplicate_agents_get_numbered(self):
        wf = [
            {"agent": "claude:sonnet", "subtask": "Angle A", "sees": "none"},
            {"agent": "claude:sonnet", "subtask": "Angle B", "sees": "none"},
        ]
        parts, preambles, vis = self._compile(wf)
        names = [p["name"] for p in parts]
        assert names[0].endswith("#1") or "#" not in names[0]
        assert "#2" in names[1]

    def test_visibility_applied_to_all_rounds(self):
        wf = [
            {"agent": "claude:opus", "subtask": "S1", "sees": "none"},
            {"agent": "claude:sonnet", "subtask": "S2", "sees": ["claude:opus"]},
        ]
        parts, _, vis = self._compile(wf, rounds=3)
        opus_name = parts[0]["name"]  # whatever normalization produces
        assert set(vis.keys()) == {1, 2, 3}
        for r in range(1, 4):
            assert opus_name in vis[r]
            assert vis[r][opus_name] == "none"

    def test_sees_all_string_preserved(self):
        wf = [{"agent": "claude:haiku", "subtask": "Verify", "sees": "all"}]
        _, _, vis = self._compile(wf)
        agent_name = list(vis[1].keys())[0]
        assert vis[1][agent_name] == "all"
