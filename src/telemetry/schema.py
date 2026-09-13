"""
Schema definitions for HITL Value-per-Cost Telemetry.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.1
"""

from datetime import datetime
from typing import Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ConfigurationError(ValueError):
    """Raised when configuration parameters are missing, malformed, or out of valid bounds."""
    pass


class HITLReviewEvent(BaseModel):
    """Structured event capturing the operational details and outcome of a human-in-the-loop review."""

    model_config = ConfigDict(frozen=True)

    thread_id: str = Field(description="Unique identifier for the conversation thread")
    escalation_tier: str = Field(description="Tier level of human reviewer, e.g. 'tier_1', 'tier_2', 'tier_3'")
    reviewer_minutes: float = Field(ge=0.0, description="Active minutes spent by human reviewer reviewing the thread")
    sla_target_minutes: int = Field(ge=0, description="Target SLA window in minutes for first/resolution response")
    actual_response_delay_minutes: float = Field(ge=0.0, description="Actual delay in minutes from escalation until resolution")
    error_caught: bool = Field(description="Whether the human reviewer identified and corrected an AI error/hallucination")
    error_severity: Optional[Literal["low", "medium", "high", "critical"]] = Field(
        default=None,
        description="Severity of caught error if error_caught=True, else None"
    )
    timestamp: datetime = Field(description="Timestamp when review completed")


class HITLRatioResult(BaseModel):
    """Calculated operational cost, error value, and value-per-cost ratio for an individual HITL event."""

    model_config = ConfigDict(frozen=True)

    thread_id: str = Field(description="Unique identifier for the conversation thread")
    operational_cost: float = Field(ge=0.0, description="Total operational cost in dollars ($)")
    error_value: float = Field(ge=0.0, description="Assigned value in dollars ($) for error caught")
    ratio: Optional[float] = Field(
        default=None,
        description="Value-per-cost ratio (value / cost). None if operational_cost is 0.0 (undefined ratio)"
    )


class HITLDashboardSummary(BaseModel):
    """Time-windowed aggregate summary of HITL metrics with tier breakdown and confidence flags."""

    model_config = ConfigDict(frozen=True)

    window: str = Field(description="Time window label, e.g. '24h', '7d', '30d'")
    mean_ratio: Optional[float] = Field(
        default=None,
        description="Mean value-per-cost ratio across events with non-zero cost. None if no non-zero cost events"
    )
    total_cost: float = Field(ge=0.0, description="Aggregate operational cost in dollars across all events")
    total_value: float = Field(ge=0.0, description="Aggregate value in dollars across all events")
    n_events: int = Field(ge=0, description="Total number of review events in the window")
    low_confidence: bool = Field(description="True if n_events is below the statistical confidence threshold (e.g. <10)")
    breakdown_by_tier: Dict[str, "HITLDashboardSummary"] = Field(
        default_factory=dict,
        description="Nested summary breakdown keyed by escalation tier"
    )
