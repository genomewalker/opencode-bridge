"""Tests for the auto-framing companion prompt system."""

from opencode_bridge.server import build_companion_prompt


# ---------------------------------------------------------------------------
# Initial prompt structure
# ---------------------------------------------------------------------------

class TestInitialPrompt:
    def test_has_all_sections(self):
        prompt = build_companion_prompt("How should we handle auth?")
        assert "## Discussion Setup" in prompt
        assert "## Collaborative Ground Rules" in prompt
        assert "## Your Approach" in prompt
        assert "## The Question" in prompt
        assert "## Synthesize" in prompt

    def test_contains_message(self):
        msg = "Should we use event sourcing for orders?"
        prompt = build_companion_prompt(msg)
        assert msg in prompt

    def test_instructs_domain_identification(self):
        prompt = build_companion_prompt("How do we price a barrier option?")
        assert "specific domain of expertise" in prompt
        assert "senior practitioner" in prompt

    def test_instructs_trade_off_analysis(self):
        prompt = build_companion_prompt("Should we use Redis or Memcached?")
        assert "trade-offs" in prompt.lower()
        assert "challenge" in prompt.lower()

    def test_works_for_software_topics(self):
        prompt = build_companion_prompt("Should we use microservices or a monolith?")
        assert "## The Question" in prompt
        assert "microservices" in prompt

    def test_works_for_science_topics(self):
        prompt = build_companion_prompt(
            "Should we use co-assembly or per-sample binning for ancient DNA metagenomes?"
        )
        assert "## The Question" in prompt
        assert "co-assembly" in prompt

    def test_works_for_finance_topics(self):
        prompt = build_companion_prompt(
            "How should we price a European barrier option with jump diffusion?"
        )
        assert "## The Question" in prompt
        assert "barrier option" in prompt


# ---------------------------------------------------------------------------
# Domain override hint
# ---------------------------------------------------------------------------

class TestDomainOverride:
    def test_override_included_in_prompt(self):
        prompt = build_companion_prompt("Tell me about caching", domain_override="security")
        assert "security" in prompt.lower()

    def test_override_free_form(self):
        prompt = build_companion_prompt(
            "How do we handle this?", domain_override="metagenomics"
        )
        assert "metagenomics" in prompt

    def test_no_override_no_hint(self):
        prompt = build_companion_prompt("Tell me about caching")
        assert "user has indicated" not in prompt


# ---------------------------------------------------------------------------
# Follow-up prompts
# ---------------------------------------------------------------------------

class TestFollowup:
    def test_followup_is_lightweight(self):
        full = build_companion_prompt("How should we handle auth?")
        followup = build_companion_prompt("What about JWT?", is_followup=True)
        assert "Continuing Our Discussion" in followup
        assert len(followup) < len(full)

    def test_followup_does_not_have_full_sections(self):
        followup = build_companion_prompt("What about JWT?", is_followup=True)
        assert "## Discussion Setup" not in followup
        assert "## Your Approach" not in followup

    def test_followup_contains_message(self):
        msg = "What about JWT vs sessions?"
        followup = build_companion_prompt(msg, is_followup=True)
        assert msg in followup

    def test_followup_has_collaborative_reminder(self):
        followup = build_companion_prompt("What next?", is_followup=True)
        assert "challenge assumptions" in followup


# ---------------------------------------------------------------------------
# File context
# ---------------------------------------------------------------------------

class TestFileContext:
    def test_no_crash_with_files(self):
        prompt = build_companion_prompt("Review this", files=["/tmp/test.py"])
        assert "## The Question" in prompt

    def test_temp_files_excluded_from_context(self):
        prompt = build_companion_prompt(
            "Review this", files=["/tmp/opencode_msg_abc.md"]
        )
        # Temp message files should not appear in file context
        assert "opencode_msg" not in prompt.split("## The Question")[0]
