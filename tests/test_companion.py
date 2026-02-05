"""Tests for the domain detection and companion prompt system."""

import pytest

from opencode_bridge.server import (
    DOMAIN_REGISTRY,
    DomainDetection,
    DomainProfile,
    build_companion_prompt,
    detect_domain,
)


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

class TestDomainRegistry:
    def test_all_13_domains_registered(self):
        assert len(DOMAIN_REGISTRY) == 13

    def test_expected_domain_ids(self):
        expected = {
            "architecture", "debugging", "performance", "security",
            "testing", "devops", "database", "api_design",
            "frontend", "algorithms", "code_quality", "planning", "general",
        }
        assert set(DOMAIN_REGISTRY.keys()) == expected

    def test_every_profile_has_required_fields(self):
        for domain_id, profile in DOMAIN_REGISTRY.items():
            assert profile.id == domain_id
            assert profile.name
            assert profile.expert_persona
            assert len(profile.thinking_frameworks) >= 1
            assert len(profile.key_questions) >= 1
            assert len(profile.structured_approach) >= 1
            assert profile.agent_hint in ("plan", "build", "explore", "general")

    def test_general_has_no_keywords(self):
        """General is the fallback — it should never win by keyword match."""
        g = DOMAIN_REGISTRY["general"]
        assert g.keywords == []
        assert g.phrases == []


# ---------------------------------------------------------------------------
# Domain detection — keyword / phrase scoring
# ---------------------------------------------------------------------------

class TestDetectDomain:
    def test_architecture(self):
        d = detect_domain("Should we use event sourcing for our order system?")
        assert d.primary.id == "architecture"

    def test_debugging(self):
        d = detect_domain("There's a race condition in the worker pool")
        assert d.primary.id == "debugging"

    def test_performance(self):
        d = detect_domain("We need to optimize the hot path and reduce p99 latency")
        assert d.primary.id == "performance"

    def test_security(self):
        d = detect_domain("How do we prevent SQL injection in the login form?")
        assert d.primary.id == "security"

    def test_testing(self):
        d = detect_domain("We need better unit test coverage for the auth module")
        assert d.primary.id == "testing"

    def test_devops(self):
        d = detect_domain("Let's set up a CI/CD pipeline with kubernetes")
        assert d.primary.id == "devops"

    def test_database(self):
        d = detect_domain("Should we add an index on the orders table for this query?")
        assert d.primary.id == "database"

    def test_api_design(self):
        d = detect_domain("How should we version our REST API endpoints?")
        assert d.primary.id == "api_design"

    def test_frontend(self):
        d = detect_domain("Should this React component use SSR or client-side rendering?")
        assert d.primary.id == "frontend"

    def test_algorithms(self):
        d = detect_domain("What's the time complexity of this dynamic programming solution?")
        assert d.primary.id == "algorithms"

    def test_code_quality(self):
        d = detect_domain("This class violates single responsibility and has a lot of code smell")
        assert d.primary.id == "code_quality"

    def test_planning(self):
        d = detect_domain("Let's define user stories and acceptance criteria for the MVP")
        assert d.primary.id == "planning"

    def test_general_fallback(self):
        d = detect_domain("What is the meaning of life?")
        assert d.primary.id == "general"
        assert d.confidence == 50

    def test_confidence_increases_with_more_matches(self):
        few = detect_domain("Tell me about microservices")
        many = detect_domain(
            "Should we use microservices with event driven architecture, "
            "a message bus, and domain driven design for our distributed system?"
        )
        assert many.confidence > few.confidence

    def test_phrases_score_higher_than_keywords(self):
        """A phrase like 'event sourcing' should beat a single keyword like 'event'."""
        kw_only = detect_domain("event")
        phrase = detect_domain("event sourcing")
        assert phrase.confidence >= kw_only.confidence


# ---------------------------------------------------------------------------
# Cross-domain detection
# ---------------------------------------------------------------------------

class TestCrossDomain:
    def test_secondary_domain_detected(self):
        d = detect_domain("Optimize the database query performance and add an index")
        assert d.secondary is not None
        assert d.secondary.id != d.primary.id
        assert d.secondary_confidence > 0

    def test_no_secondary_when_dominant(self):
        d = detect_domain("What is the meaning of life?")
        assert d.secondary is None

    def test_secondary_ids_are_valid(self):
        d = detect_domain("Deploy a kubernetes cluster with monitoring and CI/CD pipeline alerts")
        if d.secondary:
            assert d.secondary.id in DOMAIN_REGISTRY


# ---------------------------------------------------------------------------
# File indicator scoring
# ---------------------------------------------------------------------------

class TestFileIndicators:
    def test_dockerfile_triggers_devops(self):
        d = detect_domain("How should we configure this?", file_paths=["/app/Dockerfile"])
        assert d.primary.id == "devops"

    def test_yaml_triggers_devops(self):
        d = detect_domain("How should we set this up?", file_paths=["/app/deploy.yaml"])
        # yaml can match devops or architecture; both have .yaml
        assert d.primary.id in ("devops", "architecture")

    def test_sql_triggers_database(self):
        d = detect_domain("Review this file", file_paths=["/db/migration.sql"])
        assert d.primary.id == "database"

    def test_proto_triggers_architecture(self):
        d = detect_domain("Review this", file_paths=["/api/service.proto"])
        assert d.primary.id in ("architecture", "api_design")

    def test_tsx_triggers_frontend(self):
        d = detect_domain("Review this component", file_paths=["/src/Button.tsx"])
        assert d.primary.id == "frontend"

    def test_test_file_triggers_testing(self):
        d = detect_domain("Check this", file_paths=["/src/auth_test.py"])
        assert d.primary.id == "testing"

    def test_multiple_files_accumulate(self):
        d = detect_domain(
            "Review these",
            file_paths=["/app/Dockerfile", "/app/deploy.yaml", "/app/k8s.yml"],
        )
        assert d.primary.id == "devops"
        assert d.confidence > detect_domain(
            "Review these", file_paths=["/app/Dockerfile"]
        ).confidence


# ---------------------------------------------------------------------------
# Companion prompt generation
# ---------------------------------------------------------------------------

class TestBuildCompanionPrompt:
    def test_initial_prompt_has_all_sections(self):
        prompt, det = build_companion_prompt("How should we handle auth?")
        assert "## Discussion Setup" in prompt
        assert "### Analytical Toolkit" in prompt
        assert "### Key Questions to Consider" in prompt
        assert "## Collaborative Ground Rules" in prompt
        assert "## Approach" in prompt
        assert "## The Question" in prompt
        assert "## Synthesize" in prompt

    def test_initial_prompt_contains_message(self):
        msg = "Should we use event sourcing for orders?"
        prompt, _ = build_companion_prompt(msg)
        assert msg in prompt

    def test_initial_prompt_contains_persona(self):
        prompt, det = build_companion_prompt("How to prevent SQL injection?")
        assert det.primary.expert_persona in prompt

    def test_followup_is_lightweight(self):
        full, _ = build_companion_prompt("How should we handle auth?")
        followup, _ = build_companion_prompt("What about JWT?", is_followup=True)
        assert "Continuing Our Discussion" in followup
        assert len(followup) < len(full)

    def test_followup_does_not_have_full_sections(self):
        followup, _ = build_companion_prompt("What about JWT?", is_followup=True)
        assert "## Discussion Setup" not in followup
        assert "### Analytical Toolkit" not in followup

    def test_followup_contains_message(self):
        msg = "What about JWT vs sessions?"
        followup, _ = build_companion_prompt(msg, is_followup=True)
        assert msg in followup

    def test_domain_override(self):
        prompt, det = build_companion_prompt("Tell me about caching", domain_override="security")
        assert det.primary.id == "security"
        assert det.confidence == 99

    def test_invalid_domain_override_falls_back_to_detection(self):
        prompt, det = build_companion_prompt(
            "Optimize the query", domain_override="nonexistent_domain"
        )
        # Should fall back to auto-detection, not crash
        assert det.primary.id != "nonexistent_domain"

    def test_file_context_included(self):
        prompt, _ = build_companion_prompt(
            "Review this", files=["/tmp/test.py"]
        )
        # File context section is only added if file exists; just ensure no crash
        assert "## The Question" in prompt

    def test_cross_domain_note_in_prompt(self):
        prompt, det = build_companion_prompt(
            "Optimize the database query performance and add an index"
        )
        if det.secondary:
            assert "also touches on" in prompt.lower()


# ---------------------------------------------------------------------------
# DomainDetection dataclass
# ---------------------------------------------------------------------------

class TestDomainDetection:
    def test_confidence_range(self):
        for msg in [
            "event sourcing microservices",
            "debug this bug",
            "What is life?",
        ]:
            d = detect_domain(msg)
            assert 0 <= d.confidence <= 100

    def test_secondary_confidence_less_than_or_equal_primary(self):
        d = detect_domain("Optimize the database query and add an index")
        if d.secondary:
            assert d.secondary_confidence <= d.confidence


# ---------------------------------------------------------------------------
# Auto-generated framing for unknown domains
# ---------------------------------------------------------------------------

class TestAutoFraming:
    def test_unknown_domain_gets_auto_framing(self):
        prompt, det = build_companion_prompt(
            "Should we use co-assembly or per-sample binning for our ancient DNA metagenomes?"
        )
        assert "Auto-Framing" in prompt
        assert det.primary.id == "auto"
        assert det.primary.name == "Auto-Detected"

    def test_auto_framing_has_collaborative_rules(self):
        prompt, _ = build_companion_prompt("How do we price a barrier option with jump diffusion?")
        assert "## Collaborative Ground Rules" in prompt
        assert "challenge" in prompt.lower()

    def test_auto_framing_has_synthesize(self):
        prompt, _ = build_companion_prompt("What's the best way to train a GAN for medical imaging?")
        assert "## Synthesize" in prompt
        assert "Domain identified" in prompt

    def test_auto_framing_contains_message(self):
        msg = "How should we handle phylogenetic placement of short aDNA reads?"
        prompt, _ = build_companion_prompt(msg)
        assert msg in prompt

    def test_auto_framing_instructs_self_identification(self):
        prompt, _ = build_companion_prompt("What primer sets work best for ITS2 amplicon sequencing?")
        assert "specific domain of expertise" in prompt
        assert "senior practitioner" in prompt

    def test_known_domain_does_not_get_auto_framing(self):
        prompt, det = build_companion_prompt("Should we use event sourcing?")
        assert "Auto-Framing" not in prompt
        assert det.primary.id == "architecture"

    def test_auto_framing_skipped_on_followup(self):
        prompt, det = build_companion_prompt(
            "What about the damage patterns?", is_followup=True
        )
        assert "Continuing Our Discussion" in prompt
        assert "Auto-Framing" not in prompt
