"""Pydantic schemas and configuration models for Hiver integration.

Phase: M6.P6.1.F1
Defines TagConfig, SLAConfig, TriageResult, and custom exceptions.
"""

from __future__ import annotations

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
