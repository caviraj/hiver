"""Unit tests for Hiver tag derivation and application logic.

Phase: M6.P6.1.F1
Tests:
- Known intent mapping
- Composite key overrides ({intent}:{tier})
- Automatic 'Urgent' tag injection for critical and urgent tiers
- Deduplication of 'Urgent' tag if already configured
- Unknown intent label fallback to fallback_tag with logged warning (dynamic taxonomy safety)
- Case insensitivity in escalation tier checks
- Empty and populated tag application via client
"""

from __future__ import annotations

import logging
import pytest

from src.integrations.hiver.client import MockHiverClient
from src.integrations.hiver.schema import TagConfig
from src.integrations.hiver.tagging import apply_triage_tags, derive_tags


@pytest.fixture
def sample_tag_config() -> TagConfig:
    """Fixture providing representative tag mappings."""
    return TagConfig(
        mapping={
            "Billing_Issue": ["Billing", "Finance"],
            "Billing_Issue:critical": ["Billing", "Finance", "Executive_Escalation"],
            "Technical_Support": ["Tech_Support", "L2"],
            "Security_Vulnerability": ["Security", "Urgent"],  # Already contains Urgent
            "Feature_Request": ["Product", "Backlog"],
        },
        fallback_tag="Uncategorized",
    )


def test_derive_tags_known_intent_standard_tier(sample_tag_config: TagConfig):
    """Known intent with standard tier should return mapped tags without Urgent tag."""
    tags = derive_tags(
        intent_label="Billing_Issue",
        escalation_tier="standard",
        config=sample_tag_config,
    )
    assert tags == ["Billing", "Finance"]
    assert "Urgent" not in tags


def test_derive_tags_known_intent_none_tier(sample_tag_config: TagConfig):
    """Known intent with none tier should return mapped tags without Urgent tag."""
    tags = derive_tags(
        intent_label="Feature_Request",
        escalation_tier="none",
        config=sample_tag_config,
    )
    assert tags == ["Product", "Backlog"]
    assert "Urgent" not in tags


def test_derive_tags_composite_key_override(sample_tag_config: TagConfig):
    """Composite key '{intent}:{tier}' should take precedence and append Urgent for critical."""
    tags = derive_tags(
        intent_label="Billing_Issue",
        escalation_tier="critical",
        config=sample_tag_config,
    )
    # Composite mapping gives ["Billing", "Finance", "Executive_Escalation"],
    # plus "Urgent" appended because tier is critical
    assert tags == ["Billing", "Finance", "Executive_Escalation", "Urgent"]


def test_derive_tags_urgent_tier_appends_urgent(sample_tag_config: TagConfig):
    """Urgent escalation tier should always append 'Urgent' tag."""
    tags = derive_tags(
        intent_label="Technical_Support",
        escalation_tier="urgent",
        config=sample_tag_config,
    )
    assert tags == ["Tech_Support", "L2", "Urgent"]


def test_derive_tags_urgent_deduplication(sample_tag_config: TagConfig):
    """If 'Urgent' is already in mapping, it should not be duplicated when tier is critical."""
    tags = derive_tags(
        intent_label="Security_Vulnerability",
        escalation_tier="critical",
        config=sample_tag_config,
    )
    assert tags == ["Security", "Urgent"]
    assert tags.count("Urgent") == 1


def test_derive_tags_tier_case_insensitivity(sample_tag_config: TagConfig):
    """Escalation tier checks should be case-insensitive."""
    tags_upper = derive_tags(
        intent_label="Technical_Support",
        escalation_tier="CRITICAL",
        config=sample_tag_config,
    )
    assert "Urgent" in tags_upper

    tags_mixed = derive_tags(
        intent_label="Technical_Support",
        escalation_tier="UrGeNt",
        config=sample_tag_config,
    )
    assert "Urgent" in tags_mixed


def test_derive_tags_unknown_intent_fallback_with_warning(
    sample_tag_config: TagConfig, caplog: pytest.LogCaptureFixture
):
    """Unknown intent (e.g. from newly induced dynamic taxonomy) must fallback gracefully and log warning."""
    with caplog.at_level(logging.WARNING):
        tags = derive_tags(
            intent_label="Newly_Discovered_Edge_Case_Intent",
            escalation_tier="standard",
            config=sample_tag_config,
        )

    assert tags == ["Uncategorized"]
    assert any(
        "Unrecognized intent_label 'Newly_Discovered_Edge_Case_Intent'" in record.message
        for record in caplog.records
    )


def test_derive_tags_unknown_intent_with_critical_tier_appends_urgent(
    sample_tag_config: TagConfig, caplog: pytest.LogCaptureFixture
):
    """Unknown intent with critical tier should receive fallback tag AND 'Urgent'."""
    with caplog.at_level(logging.WARNING):
        tags = derive_tags(
            intent_label="Unmapped_Dynamic_Intent",
            escalation_tier="critical",
            config=sample_tag_config,
        )

    assert tags == ["Uncategorized", "Urgent"]
    assert any("Unrecognized intent_label 'Unmapped_Dynamic_Intent'" in r.message for r in caplog.records)


def test_derive_tags_empty_mapping_uses_fallback():
    """Empty mapping should default to fallback_tag."""
    config = TagConfig(mapping={}, fallback_tag="Triage_Needed")
    tags = derive_tags("Any_Intent", "standard", config)
    assert tags == ["Triage_Needed"]


def test_apply_triage_tags_empty_tags_no_op():
    """Applying empty tag sequence should return True without calling client."""
    client = MockHiverClient()
    result = apply_triage_tags("th_123", [], client)
    assert result is True
    assert len(client.calls) == 0


def test_apply_triage_tags_delegates_to_client():
    """Applying populated tag sequence should delegate to client.apply_tags."""
    client = MockHiverClient()
    tags = ["Billing", "Urgent"]
    result = apply_triage_tags("th_999", tags, client)

    assert result is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["method"] == "apply_tags"
    assert call["thread_id"] == "th_999"
    assert call["args"]["tags"] == ["Billing", "Urgent"]


def test_apply_triage_tags_returns_false_on_client_failure():
    """If client fails, apply_triage_tags should return False."""
    client = MockHiverClient(fail_tagging=True)
    result = apply_triage_tags("th_fail", ["Tag1"], client)
    assert result is False
