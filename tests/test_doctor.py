"""Tests for chitta-bridge-doctor exit codes and check helpers.

Uses isolated CONFIG_DIR via monkeypatch to avoid touching the real ~/.chitta-bridge.
"""

import json
import sys
from pathlib import Path

import pytest

from chitta_bridge import doctor


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(doctor, "STATE_DIRS", {
        "opencode-sessions": tmp_path / "sessions",
        "codex-sessions": tmp_path / "codex-sessions",
        "codex-jobs": tmp_path / "codex-jobs",
        "rooms": tmp_path / "rooms",
    })
    monkeypatch.setattr(doctor, "DEFAULT_GPU_URL_DIR", str(tmp_path / "endpoints"))
    monkeypatch.delenv("CHITTA_BRIDGE_URL_DIR", raising=False)
    return tmp_path


class TestScanJsonDir:
    def test_missing_dir_passes(self, tmp_path):
        result = doctor._scan_json_dir("test", tmp_path / "nope", [])
        assert result == [doctor.PASS]

    def test_clean_dir_passes(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({"id": "a", "schema_version": 1}))
        result = doctor._scan_json_dir("test", d, [])
        assert result == [doctor.PASS]

    def test_corrupt_json_fails(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        (d / "bad.json").write_text("{not json")
        result = doctor._scan_json_dir("test", d, [])
        assert doctor.FAIL in result

    def test_unknown_enum_warns(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({"effort": "ultra"}))
        result = doctor._scan_json_dir("test", d, [("effort", {"low", "high"})])
        assert doctor.WARN in result

    def test_future_schema_warns(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "id": "a", "schema_version": doctor.CURRENT_SCHEMA_VERSION + 5,
        }))
        result = doctor._scan_json_dir("test", d, [])
        assert doctor.WARN in result

    def test_current_schema_passes(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({
            "id": "a", "schema_version": doctor.CURRENT_SCHEMA_VERSION,
        }))
        result = doctor._scan_json_dir("test", d, [])
        assert result == [doctor.PASS]


class TestExitCodes:
    def test_clean_returns_zero(self, isolated_config, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["chitta-bridge-doctor"])
        # Pre-create state dirs so they exist but are empty
        for p in doctor.STATE_DIRS.values():
            p.mkdir(parents=True)
        rc = doctor.main()
        assert rc == 0

    def test_strict_promotes_warn(self, isolated_config, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["chitta-bridge-doctor", "--strict"])
        # Plant a future-schema file to force a WARN
        sessions = doctor.STATE_DIRS["opencode-sessions"]
        sessions.mkdir(parents=True)
        (sessions / "a.json").write_text(json.dumps({
            "id": "a", "schema_version": doctor.CURRENT_SCHEMA_VERSION + 99,
        }))
        rc = doctor.main()
        assert rc == 1

    def test_corrupt_state_fails(self, isolated_config, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["chitta-bridge-doctor"])
        sessions = doctor.STATE_DIRS["opencode-sessions"]
        sessions.mkdir(parents=True)
        (sessions / "broken.json").write_text("{nope")
        rc = doctor.main()
        assert rc == 1
