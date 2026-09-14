"""
Schema definitions for HITL Value-per-Cost Telemetry & Containment Resolution.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.1 & 7.2 (M7.P7.2.F1)
"""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigurationError(ValueError):
    """Raised when configuration parameters are missing, malformed, or out of valid bounds."""
    pass


class HITLReviewEvent(BaseModel):
    """Structured event capturing the operational details and outcome of a human-in-the-loop review."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    thread_id: str = Field(description="Unique identifier for the conversation thread")
    review_id: Optional[str] = None
    agent_id: Optional[str] = None
    escalation_tier: str = Field(default="tier_1", description="Tier level of human reviewer, e.g. 'tier_1', 'tier_2', 'tier_3'")
    reviewer_minutes: float = Field(default=0.0, ge=0.0, description="Active minutes spent by human reviewer reviewing the thread")
    sla_target_minutes: int = Field(default=0, ge=0, description="Target SLA window in minutes for first/resolution response")
    actual_response_delay_minutes: float = Field(default=0.0, ge=0.0, description="Actual delay in minutes from escalation until resolution")
    error_caught: bool = Field(default=False, description="Whether the human reviewer identified and corrected an AI error/hallucination")
    error_severity: Optional[Literal["low", "medium", "high", "critical"]] = Field(
        default=None,
        description="Severity of caught error if error_caught=True, else None"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when review completed")
    duration_seconds: Optional[float] = None
    status: Optional[str] = None
    severity_level: Optional[str] = None
    sla_target_seconds: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _map_review_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "duration_seconds" in data:
                mins = float(data["duration_seconds"]) / 60.0
                if "reviewer_minutes" not in data:
                    data["reviewer_minutes"] = mins
                if "actual_response_delay_minutes" not in data:
                    data["actual_response_delay_minutes"] = mins
            if "sla_target_seconds" in data and "sla_target_minutes" not in data:
                data["sla_target_minutes"] = int(data["sla_target_seconds"]) // 60
            if "severity_level" in data and "error_severity" not in data:
                data["error_severity"] = data["severity_level"]
            if "status" in data and "error_caught" not in data:
                data["error_caught"] = data["status"] in ("rejected", "error")
        return data


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

    @property
    def total_reviews(self) -> int:
        return self.n_events

    @property
    def overall_ratio(self) -> Optional[float]:
        return self.mean_ratio


class ContainmentEvent(BaseModel):
    """Structured event capturing containment status and verification signals."""

    model_config = ConfigDict(frozen=False, extra="ignore")

    thread_id: str = Field(description="Unique identifier for the conversation thread")
    customer_id: str = Field(description="Unique identifier for the customer")
    intent: Optional[str] = Field(default=None, description="Detected customer intent or category")
    contained: bool = Field(default=True, description="True when escalation_tier == 'none', False if escalated")
    csat_score: Optional[float] = Field(default=None, description="Customer satisfaction score, if submitted")
    had_repeat_contact_within_48h: Optional[bool] = Field(default=None, description="Whether repeat contact occurred within 48h")
    conflicting_signals: bool = Field(default=False, description="True if CSAT and repeat contact signals disagree")
    first_response_at: datetime = Field(description="Timestamp of first AI response")
    verification_deadline_at: Optional[datetime] = Field(default=None, description="Timestamp when verification window closes")
    escalated_to_human: Optional[bool] = None
    window_elapsed: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_containment_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "is_contained" in data and "contained" not in data:
                data["contained"] = data.pop("is_contained")
            if "escalated_to_human" in data and "contained" not in data:
                data["contained"] = not data["escalated_to_human"]
            if "repeat_contact_within_window" in data and "had_repeat_contact_within_48h" not in data:
                data["had_repeat_contact_within_48h"] = data.pop("repeat_contact_within_window")
        return data

    @property
    def is_contained(self) -> bool:
        return self.contained

    @property
    def repeat_contact_within_window(self) -> Optional[bool]:
        return self.had_repeat_contact_within_48h


class ThreadRecord(BaseModel):
    """Record of a customer interaction thread for repeat contact analysis."""

    model_config = ConfigDict(frozen=True)

    thread_id: str = Field(description="Unique identifier for the thread")
    customer_id: str = Field(description="Customer identifier")
    intent: str = Field(description="Detected customer intent or category")
    created_at: datetime = Field(description="Timestamp when thread was initiated")


class TelemetryConfig(BaseModel):
    """Configuration parameters for containment resolution and repeat contact tracking."""

    model_config = ConfigDict(frozen=True)

    csat_threshold: float = Field(default=4.0, ge=1.0, le=5.0, description="Minimum CSAT score to count as verified resolved")
    repeat_contact_window_hours: int = Field(default=48, ge=1, description="Window in hours to monitor for repeat contact")
    intent_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold for matching intents")


class TelemetryDashboard(BaseModel):
    """Final unified telemetry dashboard combining containment resolution and HITL value-per-cost metrics.
    
    Resolution rate within containment is structurally positioned as the primary headline metric,
    with raw containment rate presented as secondary contextual telemetry.
    """

    model_config = ConfigDict(frozen=True)

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the dashboard was generated"
    )
    window_label: str = Field(description="Time window or cohort label, e.g. '30d', 'all'")
    resolution_rate_within_containment: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Primary headline metric: rate of contained issues verified resolved (None if 0 verified)"
    )
    raw_containment_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Secondary contextual metric: fraction of total contacts not escalated (None if empty events)"
    )
    verified_count: int = Field(ge=0, description="Number of contained events with verified resolution status")
    pending_verification_count: int = Field(ge=0, description="Number of contained events awaiting verification")
    hitl_summary: HITLDashboardSummary = Field(description="Aggregated HITL value-per-cost summary from P7.1")

