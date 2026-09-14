"""
Tests for Unified Telemetry Dashboard Orchestration.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.2 (M7.P7.2.F1)
"""

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from src.telemetry.schema import (
    ConfigurationError,
    ContainmentEvent,
    HITLReviewEvent,
    TelemetryConfig,
    TelemetryDashboard,
)
from src.telemetry.telemetry_dashboard import (
    build_telemetry_dashboard,
    load_telemetry_config,
)


class TestLoadTelemetryConfig:
    """Test suite for loading and validating telemetry config."""

    def test_load_default_config(self):
        """Loads default config/telemetry_config.yaml successfully."""
        config = load_telemetry_config()
        assert isinstance(config, TelemetryConfig)
        assert config.csat_threshold == 4.0
        assert config.repeat_contact_window_hours == 48
        assert config.intent_similarity_threshold == 0.7

    def test_load_custom_config(self, tmp_path):
        """Loads custom YAML config file."""
        custom_file = tmp_path / "custom_telemetry.yaml"
        custom_file.write_text(
            "csat_threshold: 4.5\n"
            "repeat_contact_window_hours: 72\n"
            "intent_similarity_threshold: 0.85\n",
            encoding="utf-8",
        )
        config = load_telemetry_config(str(custom_file))
        assert config.csat_threshold == 4.5
        assert config.repeat_contact_window_hours == 72
        assert config.intent_similarity_threshold == 0.85

    def test_missing_explicit_file_raises_error(self, tmp_path):
        """Non-existent explicit config path raises FileNotFoundError."""
        missing_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_telemetry_config(str(missing_path))

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        """Syntax-invalid YAML raises ConfigurationError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("csat_threshold: [unclosed_bracket", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_telemetry_config(str(bad_yaml))

    def test_invalid_values_raise_configuration_error(self, tmp_path):
        """Out-of-range schema validation error raises ConfigurationError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("csat_threshold: 10.0\n", encoding="utf-8")  # csat max is 5.0
        with pytest.raises(ConfigurationError):
            load_telemetry_config(str(invalid_yaml))


class TestBuildTelemetryDashboard:
    """Test suite for build_telemetry_dashboard."""

    @pytest.fixture
    def sample_containment_events(self):
        base_time = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
        return [
            # 1. Contained, resolved via CSAT 5.0
            ContainmentEvent(
                thread_id="t-1",
                customer_id="c-1",
                intent="billing",
                first_response_at=base_time,
                is_contained=True,
                escalated_to_human=False,
                csat_score=5.0,
            ),
            # 2. Contained, unresolved via repeat contact
            ContainmentEvent(
                thread_id="t-2",
                customer_id="c-2",
                intent="refund",
                first_response_at=base_time,
                is_contained=True,
                escalated_to_human=False,
                csat_score=None,
                repeat_contact_within_window=True,
                window_elapsed=True,
            ),
            # 3. Contained, pending verification (window unelapsed, no csat)
            ContainmentEvent(
                thread_id="t-3",
                customer_id="c-3",
                intent="login",
                first_response_at=base_time,
                is_contained=True,
                escalated_to_human=False,
                csat_score=None,
                repeat_contact_within_window=None,
                window_elapsed=False,
            ),
            # 4. Escalated to human (not contained)
            ContainmentEvent(
                thread_id="t-4",
                customer_id="c-4",
                intent="account_takeover",
                first_response_at=base_time,
                is_contained=False,
                escalated_to_human=True,
            ),
        ]

    @pytest.fixture
    def sample_hitl_events(self):
        return [
            HITLReviewEvent(
                review_id="r-1",
                agent_id="agt-1",
                thread_id="t-10",
                duration_seconds=120,
                status="approved",
                severity_level="high",
                sla_target_seconds=300,
            ),
            HITLReviewEvent(
                review_id="r-2",
                agent_id="agt-2",
                thread_id="t-11",
                duration_seconds=180,
                status="rejected",
                severity_level="medium",
                sla_target_seconds=300,
            ),
        ]

    def test_full_dashboard_assembly(self, sample_containment_events, sample_hitl_events):
        """Unified dashboard calculates both headline and secondary metrics correctly."""
        dashboard = build_telemetry_dashboard(
            containment_events=sample_containment_events,
            hitl_events=sample_hitl_events,
            window_label="Last 24 Hours",
        )

        assert isinstance(dashboard, TelemetryDashboard)
        assert dashboard.window_label == "Last 24 Hours"

        # Raw containment rate: 3 contained out of 4 total = 0.75
        assert dashboard.raw_containment_rate == 0.75

        # Verified contained: t-1 (resolved), t-2 (unresolved). t-3 is pending.
        # Verified resolution rate: 1 / 2 = 0.50
        assert dashboard.resolution_rate_within_containment == 0.50
        assert dashboard.verified_count == 2
        assert dashboard.pending_verification_count == 1

        # HITL summary attached and populated
        assert dashboard.hitl_summary is not None
        assert dashboard.hitl_summary.total_reviews == 2

    def test_all_pending_verification_discipline(self, sample_hitl_events):
        """When all contained events are pending, resolution rate is None, never 0.0."""
        base_time = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            ContainmentEvent(
                thread_id="t-pending",
                customer_id="c-1",
                intent="billing",
                first_response_at=base_time,
                is_contained=True,
                escalated_to_human=False,
                csat_score=None,
                repeat_contact_within_window=None,
                window_elapsed=False,
            )
        ]
        dashboard = build_telemetry_dashboard(
            containment_events=events,
            hitl_events=sample_hitl_events,
        )

        assert dashboard.raw_containment_rate == 1.0
        assert dashboard.resolution_rate_within_containment is None
        assert dashboard.verified_count == 0
        assert dashboard.pending_verification_count == 1

    def test_empty_events_handling(self):
        """Empty inputs safely produce None rates without division-by-zero errors."""
        dashboard = build_telemetry_dashboard(
            containment_events=[],
            hitl_events=[],
        )
        assert dashboard.raw_containment_rate is None
        assert dashboard.resolution_rate_within_containment is None
        assert dashboard.verified_count == 0
        assert dashboard.pending_verification_count == 0
        assert dashboard.hitl_summary.total_reviews == 0

    def test_dashboard_is_immutable(self, sample_containment_events, sample_hitl_events):
        """TelemetryDashboard is frozen and prevents runtime mutation."""
        dashboard = build_telemetry_dashboard(
            containment_events=sample_containment_events,
            hitl_events=sample_hitl_events,
        )
        with pytest.raises(ValidationError):
            dashboard.raw_containment_rate = 0.99
