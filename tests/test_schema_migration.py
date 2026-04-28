"""Round-trip tests for schema_version + lazy migration on persisted dataclasses."""

import json

import pytest

from chitta_bridge.server import (
    CodexJob,
    CodexSession,
    DiscussionRoom,
    PERSISTED_SCHEMA_VERSION,
    Session,
    _migrate_persisted,
)


class TestMigrateHelper:
    def test_v0_stamped_to_current(self):
        data = {"id": "x", "model": "m"}
        out = _migrate_persisted(data, "session")
        assert out["schema_version"] == PERSISTED_SCHEMA_VERSION

    def test_current_unchanged(self):
        data = {"id": "x", "schema_version": PERSISTED_SCHEMA_VERSION}
        out = _migrate_persisted(data, "session")
        assert out["schema_version"] == PERSISTED_SCHEMA_VERSION

    def test_future_left_alone(self):
        future = PERSISTED_SCHEMA_VERSION + 7
        data = {"id": "x", "schema_version": future}
        out = _migrate_persisted(data, "session")
        assert out["schema_version"] == future


class TestSessionRoundTrip:
    def test_v0_loads_and_stamps(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({
            "id": "abc", "model": "gpt-5.4", "agent": "build",
        }))
        s = Session.load(path)
        assert s.id == "abc"
        assert s.schema_version == PERSISTED_SCHEMA_VERSION

    def test_save_includes_version(self, tmp_path):
        s = Session(id="abc", model="m", agent="build")
        path = tmp_path / "s.json"
        s.save(path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == PERSISTED_SCHEMA_VERSION

    def test_round_trip_preserves_fields(self, tmp_path):
        s = Session(id="abc", model="m", agent="build", variant="default")
        s.add_message("user", "hi")
        path = tmp_path / "s.json"
        s.save(path)
        loaded = Session.load(path)
        assert loaded.id == s.id
        assert loaded.model == s.model
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "hi"


class TestCodexSessionRoundTrip:
    def test_v0_loads(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"id": "j", "model": "gpt-5.4"}))
        s = CodexSession.load(path)
        assert s.id == "j"
        assert s.schema_version == PERSISTED_SCHEMA_VERSION


class TestCodexJobRoundTrip:
    def test_v0_loads(self, tmp_path):
        path = tmp_path / "j.json"
        path.write_text(json.dumps({
            "id": "j1", "task": "do x", "model": "gpt-5.4",
            "working_dir": "/tmp", "created": "2024-01-01T00:00:00",
        }))
        j = CodexJob.load(path)
        assert j.id == "j1"
        assert j.schema_version == PERSISTED_SCHEMA_VERSION

    def test_save_includes_version(self, tmp_path):
        j = CodexJob(id="j1", task="x", model="m", working_dir="/tmp")
        path = tmp_path / "j.json"
        j.save(path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == PERSISTED_SCHEMA_VERSION


class TestDiscussionRoomRoundTrip:
    def test_v0_loads(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps({
            "id": "r1", "topic": "test", "participants": [],
        }))
        r = DiscussionRoom.load(path)
        assert r.id == "r1"
        assert r.schema_version == PERSISTED_SCHEMA_VERSION
