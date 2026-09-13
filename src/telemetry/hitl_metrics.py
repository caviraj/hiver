"""
HITL Value-per-Cost Metric Computations
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.1

========================================================================================
IMPORTANT CAVEAT & DERIVATION NOTICE:
The PRD presents the HITL Value-per-Cost metric's exact formula as an embedded image,
which is not extractable as machine-readable text.

The implementation below is DERIVED from the PRD's written description:
"the operational cost of human review, calculated by reviewer minutes and the SLA penalty
of delayed response, is justified by the severity of the errors caught by the reviewers."

Formula derivation:
- Operational Cost = (reviewer_minutes * minute_cost_rate)
                     + max(0.0, actual_response_delay_minutes - sla_target_minutes) * sla_penalty_per_minute
- Error Value = severity_weights[error_severity] if error_caught else 0.0
- Ratio = Error Value / Operational Cost (when Operational Cost > 0, else None)

This is a reasonable-faith reconstruction, not a verified transcription of the original formula.
If you have access to the original PRD image asset, verify and adjust these computations if needed.
========================================================================================
"""

import logging
from typing import Dict

from src.telemetry.schema import ConfigurationError, HITLRatioResult, HITLReviewEvent

logger = logging.getLogger(__name__)


def compute_operational_cost(
    event: HITLReviewEvent,
    minute_cost_rate: float,
    sla_penalty_per_minute: float,
) -> float:
    """
    Calculate the operational cost of human review.

    Args:
        event: The review event containing reviewer minutes, SLA target, and actual delay.
        minute_cost_rate: Dollar cost per minute of reviewer effort (> 0.0).
        sla_penalty_per_minute: Dollar penalty per minute for SLA breaches (> 0.0).

    Returns:
        float: Calculated operational cost in dollars ($).

    Raises:
        ConfigurationError: If minute_cost_rate or sla_penalty_per_minute is non-positive.
    """
    if minute_cost_rate <= 0:
        raise ConfigurationError(
            f"minute_cost_rate must be positive (> 0), got: {minute_cost_rate}"
        )
    if sla_penalty_per_minute <= 0:
        raise ConfigurationError(
            f"sla_penalty_per_minute must be positive (> 0), got: {sla_penalty_per_minute}"
        )

    # Base labor cost
    labor_cost = event.reviewer_minutes * minute_cost_rate

    # SLA breach penalty (clamped at 0.0 if delay <= target)
    delay_excess = max(0.0, event.actual_response_delay_minutes - float(event.sla_target_minutes))
    sla_penalty = delay_excess * sla_penalty_per_minute

    total_cost = labor_cost + sla_penalty
    return round(total_cost, 4)


def compute_error_value(
    event: HITLReviewEvent,
    severity_weights: Dict[str, float],
) -> float:
    """
    Calculate the business value of an error caught by human review.

    Args:
        event: The review event indicating whether an error was caught and its severity.
        severity_weights: Mapping of severity level ('low', 'medium', 'high', 'critical', 'unknown') to dollar value.

    Returns:
        float: Calculated business value in dollars ($). 0.0 if no error was caught.
    """
    if not event.error_caught:
        return 0.0

    severity = event.error_severity
    if severity is not None and severity in severity_weights:
        return float(severity_weights[severity])

    # Fallback to 'unknown' weight with a warning log
    logger.warning(
        "HITL event for thread %s caught an error but has missing or unrecognized severity '%s'. "
        "Falling back to 'unknown' weight.",
        event.thread_id,
        severity,
    )
    return float(severity_weights.get("unknown", 15.0))


def compute_hitl_ratio(
    event: HITLReviewEvent,
    minute_cost_rate: float,
    sla_penalty_per_minute: float,
    severity_weights: Dict[str, float],
) -> HITLRatioResult:
    """
    Compute operational cost, caught error value, and value-per-cost ratio for a review event.

    Args:
        event: The review event to evaluate.
        minute_cost_rate: Cost per reviewer minute.
        sla_penalty_per_minute: Penalty rate for SLA delays.
        severity_weights: Value weights per error severity.

    Returns:
        HITLRatioResult: Result containing cost, value, and ratio (or None if cost == 0.0).
    """
    cost = compute_operational_cost(
        event=event,
        minute_cost_rate=minute_cost_rate,
        sla_penalty_per_minute=sla_penalty_per_minute,
    )
    value = compute_error_value(
        event=event,
        severity_weights=severity_weights,
    )

    if cost == 0.0:
        logger.warning(
            "HITL event for thread %s resulted in zero operational cost. "
            "Value-per-cost ratio is undefined (None).",
            event.thread_id,
        )
        ratio = None
    else:
        ratio = round(value / cost, 4)

    return HITLRatioResult(
        thread_id=event.thread_id,
        operational_cost=cost,
        error_value=value,
        ratio=ratio,
    )
