"""Unit tests for Hiver SLA derivation and timer initiation logic.

Phase: M6.P6.1.F1
Tests:
- Strict SLA duration lookup across all configured tiers
- Case insensitivity in tier resolution
- Strict ConfigurationError raised on unconfigured tier (no silent fallback)
- Non-positive duration rejection
- SLA timer delegation to HiverClient
- Failure propagation from HiverClient
"""

from __future__ import annotations

import logging
import pytest

from src.integrations.hiver.client import MockHiverClient
from src.integrations.hiver.schema import ConfigurationError, SLAConfig
from src.integrations.hiver.sla import derive_sla_duration, start_sla_for_thread


@pytest.fixture
def sample_sla_config() -> SLAConfig:
    """Fixture providing standard tier-to-duration mappings."""
    return SLAConfig(
        durations_minutes={
            "critical": 15,
            "urgent": 60,
            "standard": 240,
            "none": 1440,
        }
    )


@pytest.mark.parametrize(
    "tier,expected_minutes",
    [
        ("critical", 15),
        ("urgent", 60),
        ("standard", 240),
        ("none", 1440),
    ],
)
def test_derive_sla_duration_valid_tiers(
    sample_sla_config: SLAConfig, tier: str, expected_minutes: int
):
    """Each valid tier must resolve to its exact configured duration."""
    duration = derive_sla_duration(tier, sample_sla_config)
    assert duration == expected_minutes


@pytest.mark.parametrize(
    "raw_tier,expected_minutes",
    [
        ("CRITICAL", 15),
        ("Urgent", 60),
        ("  standard  ", 240),
        ("NONE", 1440),
    ],
)
def test_derive_sla_duration_case_and_whitespace_insensitivity(
    sample_sla_config: SLAConfig, raw_tier: str, expected_minutes: int
):
    """Tier lookup should handle uppercase and surrounding whitespace cleanly."""
    duration = derive_sla_duration(raw_tier, sample_sla_config)
    assert duration == expected_minutes


def test_derive_sla_duration_missing_tier_raises_configuration_error(sample_sla_config: SLAConfig):
    """Unconfigured tier MUST raise ConfigurationError and list available tiers."""
    with pytest.raises(ConfigurationError) as exc_info:
        derive_sla_duration("unprecedented_tier", sample_sla_config)

    error_msg = str(exc_info.value)
    assert "Missing SLA duration configuration for escalation tier 'unprecedented_tier'" in error_msg
    assert "critical" in error_msg
    assert "urgent" in error_msg
    assert "standard" in error_msg
    assert "none" in error_msg


def test_derive_sla_duration_empty_config_raises_configuration_error():
    """Empty durations_minutes dict should immediately raise ConfigurationError."""
    empty_config = SLAConfig(durations_minutes={})
    with pytest.raises(ConfigurationError) as exc_info:
        derive_sla_duration("standard", empty_config)

    assert "Missing SLA duration configuration" in str(exc_info.value)


def test_start_sla_for_thread_delegates_to_client():
    """start_sla_for_thread should invoke client.start_sla_timer with exact parameters."""
    client = MockHiverClient()
    success = start_sla_for_thread("th_sla_100", 60, client)

    assert success is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["method"] == "start_sla_timer"
    assert call["thread_id"] == "th_sla_100"
    assert call["args"]["duration_minutes"] == 60


def test_start_sla_for_thread_rejects_non_positive_duration(caplog: pytest.LogCaptureFixture):
    """Non-positive SLA duration should log a warning and return False without invoking client."""
    client = MockHiverClient()

    with caplog.at_level(logging.WARNING):
        res_zero = start_sla_for_thread("th_zero", 0, client)
        res_neg = start_sla_for_thread("th_neg", -15, client)

    assert res_zero is False
    assert res_neg is False
    assert len(client.calls) == 0
    assert any("non-positive duration (0 min)" in r.message for r in caplog.records)
    assert any("non-positive duration (-15 min)" in r.message for r in caplog.records)


def test_start_sla_for_thread_returns_false_on_client_failure():
    """Client failure should return False cleanly."""
    client = MockHiverClient(fail_sla=True)
    success = start_sla_for_thread("th_sla_fail", 30, client)
    assert success is False
    assert len(client.calls) == 1
