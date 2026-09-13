"""SLA timer derivation and initiation logic for Hiver threads.

Phase: M6.P6.1.F1
Derives SLA countdown durations strictly by escalation tier and invokes client timers.
"""

from __future__ import annotations

import logging

from src.integrations.hiver.client import HiverClient
from src.integrations.hiver.schema import ConfigurationError, SLAConfig

logger = logging.getLogger(__name__)


def derive_sla_duration(escalation_tier: str, config: SLAConfig) -> int:
    """Derive response SLA duration in minutes for a given escalation tier.

    Strict Contract:
        If escalation_tier is missing from config.durations_minutes, a ConfigurationError
        is raised immediately. SLA timers have contractual and operational compliance
        implications and MUST NOT silently fallback to an assumed default.

    Args:
        escalation_tier: Escalation tier from M2.P2.3 (e.g., "critical", "urgent", "standard", "none").
        config: SLAConfig specifying duration mappings.

    Returns:
        Duration in minutes (e.g. 15, 60, 240, 1440).

    Raises:
        ConfigurationError: If escalation_tier is not configured in durations_minutes.
    """
    durations = config.durations_minutes or {}
    norm_tier = (escalation_tier or "").strip().lower()

    # Check both normalized tier and raw escalation_tier
    if norm_tier in durations:
        return durations[norm_tier]
    if escalation_tier in durations:
        return durations[escalation_tier]

    valid_tiers = list(durations.keys())
    raise ConfigurationError(
        f"Missing SLA duration configuration for escalation tier '{escalation_tier}'. "
        f"Available configured tiers are: {valid_tiers}. SLA durations must be explicitly defined."
    )


def start_sla_for_thread(thread_id: str, duration_minutes: int, client: HiverClient) -> bool:
    """Initiate an SLA timer on a Hiver thread via the client interface.

    Args:
        thread_id: Target Hiver thread identifier.
        duration_minutes: Duration of the SLA countdown in minutes.
        client: HiverClient instance (API or Mock).

    Returns:
        True if the SLA timer was successfully initiated, False otherwise.
    """
    if duration_minutes <= 0:
        logger.warning(
            "Attempted to initiate SLA timer with non-positive duration (%d min) on thread %s",
            duration_minutes,
            thread_id,
        )
        return False
    return client.start_sla_timer(thread_id, duration_minutes)
