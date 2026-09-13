"""
Unit tests for HITL Dashboard Data Aggregator
Milestone 7 - Phase 7.1
"""

from datetime import datetime, timezone
import pytest

from src.telemetry.dashboard_data import (
    aggregate_hitl_ratio,
    load_hitl_config,
)
from src.telemetry.schema import (
    ConfigurationError,
    HITLReviewEvent,
)


@pytest.fixture
def hitl_config():
    return load_hitl_config("config/hitl_config.yaml")


def make_event(
    idx: int,
    tier: str = "tier_1",
    reviewer_minutes: float = 10.0,
    sla_target: int = 15,
    actual_delay: float = 10.0,
    error_caught: bool = True,
    severity: str = "medium",
) -> HITLReviewEvent:
    return HITLReviewEvent(
        thread_id=f"th_{idx}",
        escalation_tier=tier,
        reviewer_minutes=reviewer_minutes,
        sla_target_minutes=sla_target,
        actual_response_delay_minutes=actual_delay,
        error_caught=error_caught,
        error_severity=severity,
        timestamp=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    )


class TestLoadHitlConfig:
    def test_load_hitl_config_valid(self):
        config = load_hitl_config("config/hitl_config.yaml")
        assert config["minute_cost_rate"] == 0.75
        assert config["sla_penalty_per_minute"] == 1.50
        assert config["min_sample_size"] == 10
        assert "severity_weights" in config
        assert config["severity_weights"]["critical"] == 200.0

    def test_load_hitl_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_hitl_config("config/non_existent_config.yaml")

    def test_load_hitl_config_missing_required_key(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("minute_cost_rate: 0.75\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Missing required configuration key"):
            load_hitl_config(str(bad_yaml))

    def test_load_hitl_config_invalid_rates(self, tmp_path):
        bad_yaml = tmp_path / "bad_rates.yaml"
        bad_yaml.write_text(
            """
minute_cost_rate: -0.5
sla_penalty_per_minute: 1.50
min_sample_size: 10
severity_weights: {low: 5.0}
""",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="minute_cost_rate must be a positive number"):
            load_hitl_config(str(bad_yaml))


class TestAggregateHitlRatio:
    def test_aggregate_empty_events(self, hitl_config):
        summary = aggregate_hitl_ratio(
            events=[],
            minute_cost_rate=hitl_config["minute_cost_rate"],
            sla_penalty_per_minute=hitl_config["sla_penalty_per_minute"],
            severity_weights=hitl_config["severity_weights"],
            window_label="24h",
            min_sample_size=hitl_config["min_sample_size"],
        )
        assert summary.window == "24h"
        assert summary.n_events == 0
        assert summary.total_cost == 0.0
        assert summary.total_value == 0.0
        assert summary.mean_ratio is None
        assert summary.low_confidence is True
        assert summary.breakdown_by_tier == {}

    def test_aggregate_low_confidence_flag(self, hitl_config):
        # 5 events < min_sample_size (10) -> low_confidence=True
        events = [make_event(i) for i in range(5)]
        summary = aggregate_hitl_ratio(
            events=events,
            minute_cost_rate=hitl_config["minute_cost_rate"],
            sla_penalty_per_minute=hitl_config["sla_penalty_per_minute"],
            severity_weights=hitl_config["severity_weights"],
            window_label="24h",
            min_sample_size=10,
        )
        assert summary.n_events == 5
        assert summary.low_confidence is True
        assert summary.mean_ratio is not None

    def test_aggregate_high_confidence_flag(self, hitl_config):
        # 12 events >= min_sample_size (10) -> low_confidence=False
        events = [make_event(i) for i in range(12)]
        summary = aggregate_hitl_ratio(
            events=events,
            minute_cost_rate=hitl_config["minute_cost_rate"],
            sla_penalty_per_minute=hitl_config["sla_penalty_per_minute"],
            severity_weights=hitl_config["severity_weights"],
            window_label="7d",
            min_sample_size=10,
        )
        assert summary.n_events == 12
        assert summary.low_confidence is False

    def test_aggregate_tier_breakdown(self, hitl_config):
        events = [
            make_event(1, tier="tier_1", reviewer_minutes=5.0, severity="low"),
            make_event(2, tier="tier_1", reviewer_minutes=10.0, severity="medium"),
            make_event(3, tier="tier_2", reviewer_minutes=20.0, severity="high"),
            make_event(4, tier="tier_3", reviewer_minutes=30.0, severity="critical"),
        ]
        summary = aggregate_hitl_ratio(
            events=events,
            minute_cost_rate=hitl_config["minute_cost_rate"],
            sla_penalty_per_minute=hitl_config["sla_penalty_per_minute"],
            severity_weights=hitl_config["severity_weights"],
            window_label="30d",
            min_sample_size=10,
            include_tier_breakdown=True,
        )
        assert summary.n_events == 4
        assert "tier_1" in summary.breakdown_by_tier
        assert "tier_2" in summary.breakdown_by_tier
        assert "tier_3" in summary.breakdown_by_tier

        t1_summary = summary.breakdown_by_tier["tier_1"]
        assert t1_summary.n_events == 2
        assert t1_summary.breakdown_by_tier == {}  # No nested recursion

        t3_summary = summary.breakdown_by_tier["tier_3"]
        assert t3_summary.n_events == 1
        assert t3_summary.total_value == 200.0

    def test_aggregate_all_zero_cost_events(self, hitl_config):
        # 10 events with 0 minutes and 0 SLA delay -> cost = 0.0
        events = [
            make_event(i, reviewer_minutes=0.0, sla_target=15, actual_delay=5.0)
            for i in range(10)
        ]
        summary = aggregate_hitl_ratio(
            events=events,
            minute_cost_rate=hitl_config["minute_cost_rate"],
            sla_penalty_per_minute=hitl_config["sla_penalty_per_minute"],
            severity_weights=hitl_config["severity_weights"],
            window_label="7d",
            min_sample_size=10,
        )
        assert summary.n_events == 10
        assert summary.total_cost == 0.0
        assert summary.mean_ratio is None
