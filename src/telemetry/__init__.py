"""
Business Telemetry & HITL Dynamics Package
Milestone 7: Business Telemetry & HITL Dynamics (Phase 7.1 & Phase 7.2)
"""

from src.telemetry.containment_metrics import (
    compute_containment_rate,
    compute_resolution_rate_within_containment,
    verify_resolution,
)
from src.telemetry.dashboard_data import aggregate_hitl_ratio, load_hitl_config
from src.telemetry.hitl_metrics import (
    compute_error_value,
    compute_hitl_ratio,
    compute_operational_cost,
)
from src.telemetry.repeat_contact_tracker import (
    compute_intent_similarity,
    has_repeat_contact_within_window,
)
from src.telemetry.schema import (
    ConfigurationError,
    ContainmentEvent,
    HITLDashboardSummary,
    HITLRatioResult,
    HITLReviewEvent,
    TelemetryConfig,
    TelemetryDashboard,
    ThreadRecord,
)
from src.telemetry.telemetry_dashboard import (
    build_telemetry_dashboard,
    load_telemetry_config,
)

__all__ = [
    # Schema & Configurations
    "ConfigurationError",
    "HITLReviewEvent",
    "HITLRatioResult",
    "HITLDashboardSummary",
    "ContainmentEvent",
    "ThreadRecord",
    "TelemetryConfig",
    "TelemetryDashboard",
    # HITL Value-per-Cost Functions (P7.1)
    "compute_operational_cost",
    "compute_error_value",
    "compute_hitl_ratio",
    "load_hitl_config",
    "aggregate_hitl_ratio",
    # Containment & Resolution Functions (P7.2)
    "compute_containment_rate",
    "verify_resolution",
    "compute_resolution_rate_within_containment",
    "compute_intent_similarity",
    "has_repeat_contact_within_window",
    # Unified Dashboard (P7.2)
    "load_telemetry_config",
    "build_telemetry_dashboard",
]
