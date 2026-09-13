"""Pydantic schemas and configuration models for Hiver integration.

Phase: M6.P6.1.F1
Defines TagConfig, SLAConfig, TriageResult, and custom exceptions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ConfigurationError(Exception):
    """Raised when configuration values (such as required SLA duration tiers) are missing or invalid."""
    pass


class TagConfig(BaseModel):
    """Configuration model for intent/tier to tag mapping."""

    mapping: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Tag lookup mapping. Can be keyed by '{intent_label}:{escalation_tier}' "
            "for specific tier overrides or '{intent_label}' as general fallback."
        ),
    )
    fallback_tag: str = Field(
        default="Uncategorized",
        description="Fallback tag applied when an intent label is not recognized in mapping.",
    )


class SLAConfig(BaseModel):
    """Configuration model for escalation tier SLA durations in minutes."""

    durations_minutes: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of escalation_tier string (e.g. 'critical', 'urgent', 'standard', 'none') to SLA minutes.",
    )


class TriageResult(BaseModel):
    """Result of running triage (tagging and SLA timers) on a thread."""

    thread_id: str = Field(..., description="Unique identifier of the Hiver thread.")
    tags_applied: List[str] = Field(default_factory=list, description="Tags attempted/applied to the thread.")
    sla_minutes: Optional[int] = Field(default=None, description="SLA timer duration in minutes that was requested.")
    tagging_success: bool = Field(default=False, description="Whether the apply_tags operation succeeded.")
    sla_success: bool = Field(default=False, description="Whether the start_sla_timer operation succeeded.")
    errors: List[str] = Field(default_factory=list, description="Error messages from failed sub-operations.")


class AssignmentResult(BaseModel):
    """Result of running auto-assignment and escalation routing on an escalated thread."""

    thread_id: str = Field(..., description="Unique identifier of the Hiver thread.")
    assignee: Optional[str] = Field(default=None, description="Assigned agent ID or email, if assigned.")
    pool_name: Optional[str] = Field(default=None, description="Name of the agent pool used for assignment.")
    strategy_used: Optional[str] = Field(default=None, description="Strategy used: 'round_robin' or 'skill_based'.")
    success: bool = Field(default=False, description="Whether the assignment operation succeeded.")
    bypassed: bool = Field(default=True, description="Whether the decision warranted escalation bypass (False if non-escalated).")
    errors: List[str] = Field(default_factory=list, description="Error or warning messages encountered during assignment.")


class HandoffNoteContent(BaseModel):
    """Structured contents for an internal handoff note assembled upon escalation."""

    thread_id: str = Field(..., description="Unique identifier of the Hiver thread.")
    escalation_reason: str = Field(..., description="Reason for escalation or bypass.")
    escalation_tier: str = Field(..., description="Target escalation tier (e.g. tier1, tier2, critical).")
    predicted_intent: Optional[str] = Field(default=None, description="Predicted intent label if classification ran, else None.")
    confidence: Optional[float] = Field(default=None, description="Confidence score if classification ran, else None.")
    retrieved_chunks: List[str] = Field(
        default_factory=list,
        description="Top retrieved resolution_text snippets (truncated for scannability).",
    )
    draft_response_if_any: Optional[str] = Field(
        default=None,
        description="Unsent AI draft response if generated prior to escalation, else None.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the handoff note was assembled.",
    )

