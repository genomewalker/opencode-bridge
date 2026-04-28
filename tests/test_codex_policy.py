"""Tests for CodexBridge._apply_codex_policy — the single chokepoint enforcing
Codex effort defaults and Fast-variant rejection."""

import os

import pytest

from chitta_bridge.server import CodexBridge


class TestEffortDefault:
    def test_none_becomes_high(self):
        _, effort = CodexBridge._apply_codex_policy("gpt-5.4", None)
        assert effort == "high"

    def test_empty_becomes_high(self):
        _, effort = CodexBridge._apply_codex_policy("gpt-5.4", "")
        assert effort == "high"

    @pytest.mark.parametrize("e", ["low", "medium", "high", "xhigh"])
    def test_valid_efforts_pass_through(self, e):
        _, effort = CodexBridge._apply_codex_policy("gpt-5.4", e)
        assert effort == e

    @pytest.mark.parametrize("e", ["xxhigh", "HIGH", "ultra", "max"])
    def test_invalid_efforts_rejected(self, e):
        with pytest.raises(ValueError, match="Invalid Codex effort"):
            CodexBridge._apply_codex_policy("gpt-5.4", e)


class TestFastVariantRejection:
    def test_fast_in_name_rejected(self, monkeypatch):
        monkeypatch.delenv("CODEX_ALLOW_FAST", raising=False)
        with pytest.raises(ValueError, match="Fast Codex variant"):
            CodexBridge._apply_codex_policy("gpt-5.5-fast", "high")

    def test_fast_case_insensitive(self, monkeypatch):
        monkeypatch.delenv("CODEX_ALLOW_FAST", raising=False)
        with pytest.raises(ValueError):
            CodexBridge._apply_codex_policy("gpt-5.5-FAST", "high")

    def test_opt_in_allows_fast(self, monkeypatch):
        monkeypatch.setenv("CODEX_ALLOW_FAST", "1")
        model, effort = CodexBridge._apply_codex_policy("gpt-5.5-fast", "high")
        assert model == "gpt-5.5-fast"
        assert effort == "high"

    def test_non_fast_passes(self, monkeypatch):
        monkeypatch.delenv("CODEX_ALLOW_FAST", raising=False)
        model, _ = CodexBridge._apply_codex_policy("gpt-5.4", "high")
        assert model == "gpt-5.4"

    def test_none_model_passes(self, monkeypatch):
        monkeypatch.delenv("CODEX_ALLOW_FAST", raising=False)
        model, effort = CodexBridge._apply_codex_policy(None, None)
        assert model is None
        assert effort == "high"
