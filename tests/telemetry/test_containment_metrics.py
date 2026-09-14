"""
Unit Tests for Containment and Resolution Metrics
Milestone 7 - Phase 7.2 (M7.P7.2.F1)
"""

from datetime import datetime, timezone
import pytest

from src.telemetry.containment_metrics import (
    compute_containment_rate,
    compute_resolution_rate_within_containment,
    verify_resolution,
)
from src.telemetry.schema import ContainmentEvent


def make_containment_event(
    thread_id: str = "th_1",
    customer_id: str = "cust_1",
    contained: bool = True,
    first_response_at: datetime = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc),
    csat_score: float = None,
    had_repeat_contact_within_48h: bool = None,
    conflicting_signals: bool = False,
    intent: str = "billing",
) -> ContainmentEvent:
    return ContainmentEvent(
        thread_id=thread_id,
        customer_id=customer_id,
        contained=contained,
        first_response_at=first_response_at,
        csat_score=csat_score,
        had_repeat_contact_within_48h=had_repeat_contact_within_48h,
        conflicting_signals=conflicting_signals,
        intent=intent,
    )


class TestRawContainmentRate:
    def test_empty_events(self):
        assert compute_containment_rate([]) == 0.0

    def test_all_contained(self):
        events = [
            make_containment_event(thread_id=f"th_{i}", contained=True)
            for i in range(5)
        ]
        assert compute_containment_rate(events) == 1.0

    def test_mixed_containment(self):
        events = [
            make_containment_event(thread_id="th_1", contained=True),
            make_containment_event(thread_id="th_2", contained=True),
            make_containment_event(thread_id="th_3", contained=False),
            make_containment_event(thread_id="th_4", contained=False),
        ]
        assert compute_containment_rate(events) == 0.5


class TestVerifyResolution:
    def test_csat_authoritative_pass(self):
        event = make_containment_event(csat_score=5.0)
        assert verify_resolution(event, csat_threshold=4.0) is True
        assert event.conflicting_signals is False

    def test_csat_authoritative_fail(self):
        event = make_containment_event(csat_score=2.5)
        assert verify_resolution(event, csat_threshold=4.0) is False
        assert event.conflicting_signals is False

    def test_csat_exact_threshold_boundary(self):
        event_at_threshold = make_containment_event(csat_score=4.0)
        assert verify_resolution(event_at_threshold, csat_threshold=4.0) is True

        event_below_threshold = make_containment_event(csat_score=3.99)
        assert verify_resolution(event_below_threshold, csat_threshold=4.0) is False

    def test_repeat_contact_fallback_no_repeat(self):
        # CSAT is None; no repeat contact indicates genuine resolution
        event = make_containment_event(csat_score=None, had_repeat_contact_within_48h=False)
        assert verify_resolution(event) is True
        assert event.conflicting_signals is False

    def test_repeat_contact_fallback_with_repeat(self):
        # CSAT is None; repeat contact indicates unresolved issue
        event = make_containment_event(csat_score=None, had_repeat_contact_within_48h=True)
        assert verify_resolution(event) is False
        assert event.conflicting_signals is False

    def test_conflicting_signals_csat_high_repeat_true(self, caplog):
        # Customer rated 5.0, but contacted again within 48h (e.g. follow-up or edge-case).
        # Direct CSAT is authoritative (True), but conflict is flagged.
        event = make_containment_event(csat_score=5.0, had_repeat_contact_within_48h=True)
        result = verify_resolution(event, csat_threshold=4.0)
        assert result is True
        assert event.conflicting_signals is True
        assert "Conflicting telemetry signals" in caplog.text

    def test_conflicting_signals_csat_low_repeat_false(self, caplog):
        # Customer rated 2.0 (unhappy/abandoned), but never contacted again.
        # Direct CSAT is authoritative (False), and conflict is flagged.
        event = make_containment_event(csat_score=2.0, had_repeat_contact_within_48h=False)
        result = verify_resolution(event, csat_threshold=4.0)
        assert result is False
        assert event.conflicting_signals is True
        assert "Conflicting telemetry signals" in caplog.text

    def test_undetermined_both_signals_absent(self):
        # 48h window has not elapsed yet, no CSAT submitted
        event = make_containment_event(csat_score=None, had_repeat_contact_within_48h=None)
        assert verify_resolution(event) is None
        assert event.conflicting_signals is False


class TestResolutionRateWithinContainment:
    def test_zero_contained_events_returns_none(self):
        # No events at all
        assert compute_resolution_rate_within_containment([]) == (None, 0, 0)

        # Events exist but all were escalated (contained=False)
        escalated_only = [
            make_containment_event(thread_id="th_esc_1", contained=False),
            make_containment_event(thread_id="th_esc_2", contained=False),
        ]
        assert compute_resolution_rate_within_containment(escalated_only) == (None, 0, 0)

    def test_all_contained_events_pending_returns_none_strict_discipline(self):
        # 3 contained events, but all awaiting verification (window active, no CSAT)
        events = [
            make_containment_event(thread_id=f"th_pend_{i}", contained=True, csat_score=None, had_repeat_contact_within_48h=None)
            for i in range(3)
        ]
        rate, verified_count, pending_count = compute_resolution_rate_within_containment(events)
        assert rate is None  # Strict discipline: MUST NOT be 0.0!
        assert verified_count == 0
        assert pending_count == 3

    def test_standard_computation_with_mixed_verification(self):
        events = [
            # Contained & Resolved via CSAT (True)
            make_containment_event(thread_id="th_1", contained=True, csat_score=4.5),
            # Contained & Resolved via No Repeat (True)
            make_containment_event(thread_id="th_2", contained=True, csat_score=None, had_repeat_contact_within_48h=False),
            # Contained & Unresolved via Repeat Contact (False)
            make_containment_event(thread_id="th_3", contained=True, csat_score=None, had_repeat_contact_within_48h=True),
            # Contained & Unresolved via CSAT (False)
            make_containment_event(thread_id="th_4", contained=True, csat_score=2.0),
            # Contained & Pending (None)
            make_containment_event(thread_id="th_5", contained=True, csat_score=None, had_repeat_contact_within_48h=None),
            # Escalated (Ignored in resolution within containment)
            make_containment_event(thread_id="th_6", contained=False, csat_score=5.0),
        ]

        # Contained total = 5
        # Pending = 1 (th_5)
        # Verified = 4 (th_1, th_2, th_3, th_4)
        # Resolved = 2 (th_1, th_2)
        # Rate = 2 / 4 = 0.5
        rate, verified_count, pending_count = compute_resolution_rate_within_containment(events)
        assert rate == 0.5
        assert verified_count == 4
        assert pending_count == 1
