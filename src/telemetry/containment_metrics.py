"""
Containment and Resolution Metrics.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.2 (M7.P7.2.F1)

Per the PRD, raw Containment Rate is explicitly recognized as the most misleading
metric in AI customer support because customer abandonment caused by hallucination
or unhelpful answers is technically "contained" but entirely unresolved.

This module computes the mandated replacement headline metric:
Resolution Rate WITHIN Containment, verified by either authoritative direct customer
feedback (CSAT >= threshold) or behavioral proxy (absence of repeat contact within 48h).
"""

import logging
from typing import List, Optional, Tuple

from src.telemetry.schema import ContainmentEvent

logger = logging.getLogger(__name__)


def compute_containment_rate(events: List[ContainmentEvent]) -> float:
    """Compute the raw, contextual containment rate: count(contained) / total.
    
    NOTE: Per the PRD mandate, this metric is de-prioritized and must NEVER be
    presented as a headline metric on its own. It is retained solely for contextual
    and comparative visibility.
    """
    if not events:
        return 0.0
    contained_count = sum(1 for e in events if e.contained)
    return round(contained_count / len(events), 4)


def verify_resolution(
    event: ContainmentEvent,
    csat_threshold: float = 4.0
) -> Optional[bool]:
    """Verify whether a contained conversation was genuinely resolved.
    
    Verification Hierarchy:
    1. Direct customer feedback (CSAT score): Authoritative signal.
       Scores >= csat_threshold count as resolved.
    2. Behavioral proxy (absence of repeat contact within 48h): Fallback signal.
       had_repeat_contact_within_48h == False counts as resolved.
    3. Conflicting signals: When both CSAT and repeat contact signals are present
       and DISAGREE (CSAT satisfied but repeat contact occurred, or CSAT dissatisfied
       without repeat contact), CSAT remains authoritative, but event.conflicting_signals
       is set to True and logged.
    4. Undetermined: If neither signal is available yet (window hasn't elapsed and
       no CSAT submitted), returns None. This is never coerced to True or False.
    """
    has_csat = event.csat_score is not None
    has_repeat = event.had_repeat_contact_within_48h is not None

    if has_csat and has_repeat:
        csat_resolved = event.csat_score >= csat_threshold
        repeat_resolved = not event.had_repeat_contact_within_48h

        if csat_resolved != repeat_resolved:
            event.conflicting_signals = True
            logger.warning(
                "Conflicting telemetry signals for thread %s: CSAT (score=%.2f, resolved=%s) "
                "disagrees with repeat contact proxy (had_repeat=%s, resolved=%s). "
                "CSAT signal takes precedence.",
                event.thread_id,
                event.csat_score,
                csat_resolved,
                event.had_repeat_contact_within_48h,
                repeat_resolved,
            )
        return csat_resolved

    if has_csat:
        return event.csat_score >= csat_threshold

    if has_repeat:
        return not event.had_repeat_contact_within_48h

    # Neither signal available yet (window still active, no CSAT)
    return None


def compute_resolution_rate_within_containment(
    events: List[ContainmentEvent],
    csat_threshold: float = 4.0
) -> Tuple[Optional[float], int, int]:
    """Compute the primary headline metric: Resolution Rate WITHIN Containment.
    
    Filters to contained=True events, evaluates verify_resolution on each,
    and excludes pending (None) events entirely from the rate calculation.
    
    Returns:
        Tuple of (rate, verified_count, pending_count)
        - rate: resolved_count / verified_count, or None if verified_count == 0.
        - verified_count: number of contained events with definitive verification.
        - pending_count: number of contained events still awaiting verification signals.
        
    Strict None-vs-0.0 Discipline:
        If verified_count == 0 (zero contained events, or all contained events
        still pending verification), rate is None (NOT 0.0). This prevents
        zero-data states from being misinterpreted as confident failure.
    """
    contained_events = [e for e in events if e.contained]
    if not contained_events:
        return (None, 0, 0)

    resolved_count = 0
    verified_count = 0
    pending_count = 0

    for event in contained_events:
        status = verify_resolution(event, csat_threshold=csat_threshold)
        if status is None:
            pending_count += 1
        else:
            verified_count += 1
            if status is True:
                resolved_count += 1

    if verified_count == 0:
        return (None, 0, pending_count)

    rate = round(resolved_count / verified_count, 4)
    return (rate, verified_count, pending_count)
