"""Regression tests for room cross-project contamination fixes.

Covers the reported incident: a "Verifier"-role participant grounding a claim
in an unrelated repo's memory/code-intel results because realm and code-intel
lookups were not scoped to the room's actual project. A second incident (same
session) showed the first fix's warning-banner fallback wasn't enough — a room
with no `files` attached (target repo described in prose, not attached paths)
still let cross-repo code-intel results through with just a caveat, which the
participant ignored. These tests pin the fail-closed replacement.
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

from chitta_bridge.rooms import DiscussionRoom, RoomManager, _derive_project
from chitta_bridge.code_intel import _code_intel, _symbol_in_project


def _make_room_manager(rooms_dir: Path = None) -> RoomManager:
    rm = RoomManager.__new__(RoomManager)
    rm.rooms = {}
    if rooms_dir is not None:
        rm.rooms_dir = rooms_dir
    return rm


def _make_participant(name: str = "Alice") -> dict:
    return {"name": name, "backend": "opencode", "model": "test-model"}


def _make_git_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src" / "foo.py").write_text("def foo(): pass\n")
    return repo


class TestDeriveProject:
    def test_no_files_gives_no_scope(self):
        assert _derive_project([]) == ("", [])

    def test_walks_up_to_git_root(self, tmp_path):
        repo = _make_git_repo(tmp_path, "libtaph-fqdup")
        slug, roots = _derive_project([str(repo / "src" / "foo.py")])
        assert slug == "libtaph-fqdup"
        assert roots == [str(repo)]

    def test_two_different_repos_get_different_slugs(self, tmp_path):
        repo_a = _make_git_repo(tmp_path, "ancient-synthdata")
        repo_b = _make_git_repo(tmp_path, "libtaph-fqdup")
        slug_a, _ = _derive_project([str(repo_a / "src" / "foo.py")])
        slug_b, _ = _derive_project([str(repo_b / "src" / "foo.py")])
        assert slug_a != slug_b


class TestRealmScoping:
    """The actual reported bug: a role name like 'Verifier' is reused verbatim
    across unrelated rooms (see ROLE_PRESETS), so a realm keyed only on the
    role name collides across projects. project must be folded in."""

    def test_same_role_name_different_projects_get_different_realms(self):
        rm = _make_room_manager()
        participant = {"name": "Verifier", "soul": {"system_prompt": "verify things"}}
        soul_a = rm._parse_soul(participant, project="libtaph-fqdup")
        soul_b = rm._parse_soul(dict(participant), project="ancient-synthdata")
        assert soul_a.realm != soul_b.realm
        assert "libtaph-fqdup" in soul_a.realm
        assert "ancient-synthdata" in soul_b.realm

    def test_explicit_realm_override_wins(self):
        rm = _make_room_manager()
        participant = {"name": "Verifier", "soul": {"system_prompt": "x", "realm": "pinned"}}
        soul = rm._parse_soul(participant, project="some-project")
        assert soul.realm == "pinned"

    def test_no_project_falls_back_to_agent_name(self):
        rm = _make_room_manager()
        participant = {"name": "Verifier", "soul": {"system_prompt": "x"}}
        soul = rm._parse_soul(participant, project="")
        assert soul.realm == "agent:verifier"


class TestPathInProject:
    def test_path_under_root_is_in_scope(self):
        assert RoomManager._path_in_project("/repos/libtaph-fqdup/src/foo.cpp",
                                             ["/repos/libtaph-fqdup"])

    def test_path_outside_root_is_out_of_scope(self):
        assert not RoomManager._path_in_project(
            "/repos/ancient-synthdata/ancient_fraction.cpp",
            ["/repos/libtaph-fqdup"],
        )

    def test_no_roots_matches_nothing(self):
        # Callers gate on "not project_roots" before calling this (showing the
        # no-scope warning instead) — this just pins that an empty root list
        # can't accidentally allow-list everything if that guard is skipped.
        assert not RoomManager._path_in_project("/anywhere/foo.cpp", [])

    def test_code_intel_helper_matches_same_rule(self):
        assert _symbol_in_project("/repos/libtaph-fqdup/src/foo.cpp", ["/repos/libtaph-fqdup"])
        assert not _symbol_in_project("/repos/other-repo/foo.cpp", ["/repos/libtaph-fqdup"])


class TestExplicitProjectRootsOverride:
    """The second incident: target repo's real paths were described in the
    topic/preamble text, not attached via `files` — _derive_project correctly
    found nothing to derive from. room_create/conductor_fusion must accept an
    explicit project_roots override for exactly this case."""

    def test_create_accepts_explicit_project_roots_without_files(self, tmp_path):
        rm = _make_room_manager(tmp_path)
        asyncio.run(rm.create(
            room_id="r1", topic="t", participants=[_make_participant()],
            project_roots=["/repos/libtaph-fqdup"],
        ))
        room = rm.rooms["r1"]
        assert room.project_roots == ["/repos/libtaph-fqdup"]
        assert room.project == "libtaph-fqdup"

    def test_explicit_project_slug_overrides_derived_name(self, tmp_path):
        rm = _make_room_manager(tmp_path)
        asyncio.run(rm.create(
            room_id="r2", topic="t", participants=[_make_participant()],
            project="fqdup", project_roots=["/repos/libtaph-fqdup"],
        ))
        assert rm.rooms["r2"].project == "fqdup"

    def test_no_files_and_no_project_roots_leaves_scope_empty(self, tmp_path):
        rm = _make_room_manager(tmp_path)
        asyncio.run(rm.create(room_id="r3", topic="t", participants=[_make_participant()]))
        room = rm.rooms["r3"]
        assert room.project == ""
        assert room.project_roots == []


class TestFailClosedOnNoScope:
    """The actual reported regression: an unscoped room (no files, no explicit
    project_roots — the conductor_fusion incident) must REFUSE code-intel
    results outright, not return them behind a warning banner the model can
    ignore. Also asserts the daemon is never even queried when scope is
    missing, since there is no way to validate what comes back."""

    def _room(self, project_roots=None):
        return DiscussionRoom(
            id="r", topic="t", participants=[_make_participant()],
            project="libtaph-fqdup" if project_roots else "",
            project_roots=project_roots or [],
        )

    def test_search_symbols_refuses_without_scope(self):
        rm = _make_room_manager()
        room = self._room(project_roots=[])
        with patch("chitta_bridge.rooms.SoulClient._call_full") as mock_call:
            result = asyncio.run(rm._execute_agent_tool(
                "search_symbols", {"query": "damaged_fraction_pi"}, room=room,
            ))
        mock_call.assert_not_called()
        assert "refusing" in result.lower()

    def test_read_symbol_refuses_without_scope(self):
        rm = _make_room_manager()
        room = self._room(project_roots=[])
        with patch("chitta_bridge.rooms.SoulClient._call_full") as mock_call:
            result = asyncio.run(rm._execute_agent_tool(
                "read_symbol", {"name": "d_max"}, room=room,
            ))
        mock_call.assert_not_called()
        assert "refusing" in result.lower()

    def test_read_function_refuses_without_scope(self):
        rm = _make_room_manager()
        room = self._room(project_roots=[])
        with patch("chitta_bridge.rooms.SoulClient._call_full") as mock_call:
            result = asyncio.run(rm._execute_agent_tool(
                "read_function", {"name": "cmd_sample_damage"}, room=room,
            ))
        mock_call.assert_not_called()
        assert "refusing" in result.lower()

    def test_search_symbols_proceeds_with_scope(self):
        rm = _make_room_manager()
        room = self._room(project_roots=["/repos/libtaph-fqdup"])
        with patch(
            "chitta_bridge.rooms.SoulClient._call_full",
            return_value=("Found 1 symbols", {"symbols": [
                {"kind": "field", "name": "damaged_fraction_pi",
                 "file": "/repos/libtaph-fqdup/src/ancient_fraction.cpp", "line_start": 10},
            ]}),
        ):
            result = asyncio.run(rm._execute_agent_tool(
                "search_symbols", {"query": "damaged_fraction_pi"}, room=room,
            ))
        assert "damaged_fraction_pi" in result
        assert "refusing" not in result.lower()

    def test_code_intel_refuses_symbol_lookup_without_scope(self):
        # _call_full is also used by the (separately realm-scoped) memory-recall
        # step at the end of _code_intel, so assert on which RPC methods were
        # actually invoked rather than "never called at all".
        with patch("chitta_bridge.code_intel.SoulClient._call_full", return_value=("", {})) as mock_call:
            result = _code_intel(symbol="d_max", project_roots=[])
        called_methods = {c.args[0] for c in mock_call.call_args_list}
        assert called_methods.isdisjoint({"read_symbol", "symbol_callers", "symbol_callees"})
        assert "refusing" in result.lower()
