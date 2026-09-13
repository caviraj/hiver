"""
Unit tests for HITL Value-per-Cost Metric Computations
Milestone 7 - Phase 7.1
"""

from datetime import datetime, timezone
import pytest

from src.telemetry.hitl_metrics import (
    compute_error_value,
    compute_hitl_ratio,
    compute_operational_cost,
)
from src.telemetry.schema import (
    ConfigurationError,
    HITLReviewEvent,
)


@pytest.fixture
def severity_weights():
    return {
        "low": 5.0,
        "medium": 15.0,
        "high": 50.0,
        "critical": 200.0,
        "unknown": 15.0,
    }


def make_event(
    reviewer_minutes: float = 10.0,
    sla_target_minutes: int = 15,
    actual_response_delay_minutes: float = 10.0,
    error_caught: bool = True,
    error_severity: str = "medium",
    escalation_tier: str = "tier_1",
    thread_id: str = "th_test_1",
) -> HITLReviewEvent:
    return HITLReviewEvent(
        thread_id=thread_id,
        escalation_tier=escalation_tier,
        reviewer_minutes=reviewer_minutes,
        sla_target_minutes=sla_target_minutes,
        actual_response_delay_minutes=actual_response_delay_minutes,
        error_caught=error_caught,
        error_severity=error_severity,
        timestamp=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    )


class TestOperationalCost:
    def test_compute_operational_cost_no_sla_penalty(self):
        # Actual delay (10 min) <= SLA target (15 min) -> penalty is 0.0
        # Labor cost: 10 * 0.75 = 7.50
        event = make_event(
            reviewer_minutes=10.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=10.0,
        )
        cost = compute_operational_cost(
            event=event,
            minute_cost_rate=0.75,
            sla_penalty_per_minute=1.50,
        )
        assert cost == 7.50

    def test_compute_operational_cost_with_sla_penalty(self):
        # Actual delay (25 min) > SLA target (15 min) -> 10 excess min
        # Labor cost: 10 * 0.75 = 7.50
        # SLA penalty: 10 * 1.50 = 15.00
        # Total cost: 22.50
        event = make_event(
            reviewer_minutes=10.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=25.0,
        )
        cost = compute_operational_cost(
            event=event,
            minute_cost_rate=0.75,
            sla_penalty_per_minute=1.50,
        )
        assert cost == 22.50

    def test_compute_operational_cost_invalid_minute_cost_rate(self):
        event = make_event()
        with pytest.raises(ConfigurationError, match="minute_cost_rate must be positive"):
            compute_operational_cost(
                event=event,
                minute_cost_rate=0.0,
                sla_penalty_per_minute=1.50,
            )

        with pytest.raises(ConfigurationError, match="minute_cost_rate must be positive"):
            compute_operational_cost(
                event=event,
                minute_cost_rate=-0.5,
                sla_penalty_per_minute=1.50,
            )

    def test_compute_operational_cost_invalid_sla_penalty_rate(self):
        event = make_event()
        with pytest.raises(ConfigurationError, match="sla_penalty_per_minute must be positive"):
            compute_operational_cost(
                event=event,
                minute_cost_rate=0.75,
                sla_penalty_per_minute=0.0,
            )

        with pytest.raises(ConfigurationError, match="sla_penalty_per_minute must be positive"):
            compute_operational_cost(
                event=event,
                minute_cost_rate=0.75,
                sla_penalty_per_minute=-1.0,
            )


class TestErrorValue:
    def test_compute_error_value_caught(self, severity_weights):
        for sev, expected_val in [
            ("low", 5.0),
            ("medium", 15.0),
            ("high", 50.0),
            ("critical", 200.0),
        ]:
            event = make_event(error_caught=True, error_severity=sev)
            val = compute_error_value(event=event, severity_weights=severity_weights)
            assert val == expected_val

    def test_compute_error_value_not_caught(self, severity_weights):
        event = make_event(error_caught=False, error_severity="critical")
        val = compute_error_value(event=event, severity_weights=severity_weights)
        assert val == 0.0

    def test_compute_error_value_fallback_to_unknown(self, severity_weights):
        # error_caught=True, but error_severity=None
        event = HITLReviewEvent(
            thread_id="th_missing_sev",
            escalation_tier="tier_1",
            reviewer_minutes=5.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=5.0,
            error_caught=True,
            error_severity=None,
            timestamp=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
        )
        val = compute_error_value(event=event, severity_weights=severity_weights)
        assert val == 15.0  # unknown fallback


class TestHITLRatio:
    def test_compute_hitl_ratio_normal(self, severity_weights):
        # 10 min * $0.75 = $7.50 cost, medium error = $15.00 value
        # Ratio = 15.00 / 7.50 = 2.0
        event = make_event(
            reviewer_minutes=10.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=10.0,
            error_caught=True,
            error_severity="medium",
        )
        result = compute_hitl_ratio(
            event=event,
            minute_cost_rate=0.75,
            sla_penalty_per_minute=1.50,
            severity_weights=severity_weights,
        )
        assert result.thread_id == "th_test_1"
        assert result.operational_cost == 7.50
        assert result.error_value == 15.00
        assert result.ratio == 2.0

    def test_compute_hitl_ratio_zero_cost(self, severity_weights):
        # 0 reviewer minutes, delay <= sla target -> cost = 0.0
        event = make_event(
            reviewer_minutes=0.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=10.0,
            error_caught=True,
            error_severity="high",
        )
        result = compute_hitl_ratio(
            event=event,
            minute_cost_rate=0.75,
            sla_penalty_per_minute=1.50,
            severity_weights=severity_weights,
        )
        assert result.operational_cost == 0.0
        assert result.error_value == 50.0
        assert result.ratio is None

    def test_compute_hitl_ratio_no_error_caught(self, severity_weights):
        # 10 min * $0.75 = $7.50 cost, no error caught = $0.0 value
        # Ratio = 0.0 / 7.50 = 0.0
        event = make_event(
            reviewer_minutes=10.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=10.0,
            error_caught=False,
            error_severity=None,
        )
        result = compute_hitl_ratio(
            event=event,
            minute_cost_rate=0.75,
            sla_penalty_per_minute=1.50,
            severity_weights=severity_weights,
        )
        assert result.operational_cost == 7.50
        assert result.error_value == 0.0
        assert result.ratio == 0.0
