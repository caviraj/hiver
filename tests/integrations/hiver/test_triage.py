"""Unit tests for Hiver triage orchestration.

Phase: M6.P6.1.F1
Tests:
- load_hiver_config YAML loading and error handling
- run_triage happy path (both tagging and SLA timer succeed)
- run_triage partial failure: tagging succeeds, SLA fails
- run_triage partial failure: tagging fails, SLA succeeds
- run_triage total failure: both tagging and SLA fail
- Exception resilience: unexpected exceptions during tagging or SLA do not crash triage
- Strict SLA ConfigurationError caught gracefully without blocking tagging
- Dynamic taxonomy safety: unmapped intent label tags with fallback tag and succeeds
- Invariant: run_triage never raises uncaught exceptions to caller
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.integrations.hiver.client import MockHiverClient
from src.integrations.hiver.schema import ConfigurationError, SLAConfig, TagConfig
from src.integrations.hiver.triage import load_hiver_config, run_triage


@pytest.fixture
def test_tag_config() -> TagConfig:
    return TagConfig(
        mapping={
            "Billing_Issue": ["Billing", "Finance"],
            "Account_Access": ["Accounts", "Security"],
        },
        fallback_tag="General_Support",
    )


@pytest.fixture
def test_sla_config() -> SLAConfig:
    return SLAConfig(
        durations_minutes={
            "critical": 15,
            "urgent": 60,
            "standard": 240,
            "none": 1440,
        }
    )


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


def test_load_hiver_config_success():
    """Verify loading from actual config/hiver_config.yaml."""
    tag_cfg, sla_cfg = load_hiver_config("config/hiver_config.yaml")

    assert isinstance(tag_cfg, TagConfig)
    assert isinstance(sla_cfg, SLAConfig)
    assert tag_cfg.fallback_tag == "Uncategorized"
    assert "Billing_Issue" in tag_cfg.mapping
    assert sla_cfg.durations_minutes.get("critical") == 15
    assert sla_cfg.durations_minutes.get("urgent") == 60
    assert sla_cfg.durations_minutes.get("standard") == 240
    assert sla_cfg.durations_minutes.get("none") == 1440


def test_load_hiver_config_missing_file():
    """Missing config file should raise ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_hiver_config("config/non_existent_config_file_123.yaml")
    assert "not found" in str(exc_info.value)


def test_load_hiver_config_malformed_yaml():
    """Malformed YAML content should raise ConfigurationError."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write("invalid: yaml: content: [unclosed")
        tmp_path = tmp.name

    try:
        with pytest.raises(ConfigurationError) as exc_info:
            load_hiver_config(tmp_path)
        assert "Failed to parse Hiver configuration YAML" in str(exc_info.value)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Triage Orchestration Tests
# ---------------------------------------------------------------------------


def test_run_triage_happy_path(test_tag_config: TagConfig, test_sla_config: SLAConfig):
    """Full success: both tagging and SLA timer succeed."""
    client = MockHiverClient()
    result = run_triage(
        thread_id="th_ok_001",
        intent_label="Billing_Issue",
        escalation_tier="standard",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.thread_id == "th_ok_001"
    assert result.tags_applied == ["Billing", "Finance"]
    assert result.sla_minutes == 240
    assert result.tagging_success is True
    assert result.sla_success is True
    assert result.errors == []
    assert len(client.calls) == 2


def test_run_triage_critical_tier_appends_urgent(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Critical tier should append 'Urgent' and set 15m SLA timer."""
    client = MockHiverClient()
    result = run_triage(
        thread_id="th_crit_002",
        intent_label="Account_Access",
        escalation_tier="critical",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tags_applied == ["Accounts", "Security", "Urgent"]
    assert result.sla_minutes == 15
    assert result.tagging_success is True
    assert result.sla_success is True
    assert result.errors == []


def test_run_triage_partial_failure_tagging_succeeds_sla_fails(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Tagging succeeds, but SLA timer start fails on client."""
    client = MockHiverClient(fail_sla=True)
    result = run_triage(
        thread_id="th_part_001",
        intent_label="Billing_Issue",
        escalation_tier="urgent",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tagging_success is True
    assert result.sla_success is False
    assert result.tags_applied == ["Billing", "Finance", "Urgent"]
    assert result.sla_minutes == 60
    assert len(result.errors) == 1
    assert "Client failed to start SLA timer (60m)" in result.errors[0]


def test_run_triage_partial_failure_tagging_fails_sla_succeeds(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Tagging fails on client, but SLA timer succeeds."""
    client = MockHiverClient(fail_tagging=True)
    result = run_triage(
        thread_id="th_part_002",
        intent_label="Billing_Issue",
        escalation_tier="standard",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tagging_success is False
    assert result.sla_success is True
    assert result.tags_applied == ["Billing", "Finance"]
    assert result.sla_minutes == 240
    assert len(result.errors) == 1
    assert "Client failed to apply tags" in result.errors[0]


def test_run_triage_total_failure_both_fail(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Both operations fail on client; both errors are recorded."""
    client = MockHiverClient(fail_tagging=True, fail_sla=True)
    result = run_triage(
        thread_id="th_total_fail",
        intent_label="Billing_Issue",
        escalation_tier="standard",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tagging_success is False
    assert result.sla_success is False
    assert len(result.errors) == 2
    assert any("Client failed to apply tags" in err for err in result.errors)
    assert any("Client failed to start SLA timer" in err for err in result.errors)


def test_run_triage_tagging_exception_handled_cleanly(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Unexpected exception in client during tagging must not raise; SLA still runs."""
    client = MockHiverClient(tagging_exception=RuntimeError("Connection reset by peer"))
    result = run_triage(
        thread_id="th_exc_tag",
        intent_label="Billing_Issue",
        escalation_tier="standard",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tagging_success is False
    assert result.sla_success is True  # SLA still executed independently!
    assert result.sla_minutes == 240
    assert len(result.errors) == 1
    assert "Connection reset by peer" in result.errors[0]


def test_run_triage_sla_exception_handled_cleanly(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Unexpected exception in client during SLA must not raise; tagging is preserved."""
    client = MockHiverClient(sla_exception=RuntimeError("Database lock error"))
    result = run_triage(
        thread_id="th_exc_sla",
        intent_label="Billing_Issue",
        escalation_tier="urgent",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tagging_success is True
    assert result.sla_success is False
    assert result.tags_applied == ["Billing", "Finance", "Urgent"]
    assert len(result.errors) == 1
    assert "Database lock error" in result.errors[0]


def test_run_triage_missing_sla_tier_handled_without_crash(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Unconfigured escalation tier raises ConfigurationError internally, captured in errors."""
    client = MockHiverClient()
    result = run_triage(
        thread_id="th_bad_tier",
        intent_label="Billing_Issue",
        escalation_tier="unrecognized_tier",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    # Tagging succeeds normally
    assert result.tagging_success is True
    assert result.tags_applied == ["Billing", "Finance"]
    # SLA fails due to config error
    assert result.sla_success is False
    assert result.sla_minutes is None
    assert len(result.errors) == 1
    assert "SLA configuration error for tier 'unrecognized_tier'" in result.errors[0]


def test_run_triage_unknown_intent_label_uses_fallback(
    test_tag_config: TagConfig, test_sla_config: SLAConfig
):
    """Unmapped intent label falls back to fallback_tag without failing triage."""
    client = MockHiverClient()
    result = run_triage(
        thread_id="th_dyn_intent",
        intent_label="Novel_AI_Induced_Intent",
        escalation_tier="urgent",
        client=client,
        tag_config=test_tag_config,
        sla_config=test_sla_config,
    )

    assert result.tagging_success is True
    assert result.sla_success is True
    assert result.tags_applied == ["General_Support", "Urgent"]
    assert result.sla_minutes == 60
    assert result.errors == []
