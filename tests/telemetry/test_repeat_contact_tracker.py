"""
Tests for Repeat Contact Tracking and Intent Similarity Matching.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.2 (M7.P7.2.F1)
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.telemetry.repeat_contact_tracker import (
    compute_intent_similarity,
    has_repeat_contact_within_window,
)
from src.telemetry.schema import ThreadRecord


class TestIntentSimilarity:
    """Test suite for compute_intent_similarity."""

    def test_exact_match(self):
        """Identical intent strings yield 1.0 similarity."""
        assert compute_intent_similarity("billing_inquiry", "billing_inquiry") == 1.0
        assert compute_intent_similarity("  PASSWORD_RESET  ", "password_reset") == 1.0

    def test_empty_or_none(self):
        """Empty or falsy intent strings return 0.0."""
        assert compute_intent_similarity("", "billing") == 0.0
        assert compute_intent_similarity("billing", "") == 0.0
        assert compute_intent_similarity("", "") == 0.0

    def test_token_jaccard_overlap(self):
        """Bag-of-words token overlap handles word-order variation."""
        score = compute_intent_similarity(
            "invoice billing payment question",
            "payment question invoice billing",
        )
        assert score == 1.0

    def test_sequence_matcher_typos(self):
        """SequenceMatcher catches minor typos and morphological stems."""
        score = compute_intent_similarity("refund_request", "refund_reqest")
        assert score >= 0.85

    def test_dissimilar_intents(self):
        """Unrelated intents produce similarity below threshold."""
        score = compute_intent_similarity("billing_inquiry", "feature_request_dark_mode")
        assert score < 0.4


class TestRepeatContactWithinWindow:
    """Test suite for has_repeat_contact_within_window with boundary and isolation rules."""

    @pytest.fixture
    def base_time(self):
        return datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_true_positive_repeat_contact(self, base_time):
        """Same customer, new thread, within 48h, matching intent -> True."""
        candidate = ThreadRecord(
            thread_id="thread-2",
            customer_id="cust-100",
            intent="billing_inquiry_invoice",
            created_at=base_time + timedelta(hours=24),
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[candidate],
            intent_similarity_threshold=0.7,
        )
        assert result is True

    def test_different_customer_ignored(self, base_time):
        """Candidate matching intent and time window from a different customer is ignored."""
        candidate = ThreadRecord(
            thread_id="thread-2",
            customer_id="cust-999",  # Different customer
            intent="billing_inquiry",
            created_at=base_time + timedelta(hours=10),
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[candidate],
            intent_similarity_threshold=0.7,
        )
        assert result is False

    def test_same_thread_ignored(self, base_time):
        """Replies within the same thread are not repeat contacts."""
        candidate = ThreadRecord(
            thread_id="thread-1",  # Same thread
            customer_id="cust-100",
            intent="billing_inquiry",
            created_at=base_time + timedelta(hours=5),
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[candidate],
            intent_similarity_threshold=0.7,
        )
        assert result is False

    def test_boundary_prior_to_or_at_first_response(self, base_time):
        """Interactions at or before first_response_at are strictly excluded."""
        past_candidate = ThreadRecord(
            thread_id="thread-0",
            customer_id="cust-100",
            intent="billing_inquiry",
            created_at=base_time - timedelta(hours=1),
        )
        exact_first_response_candidate = ThreadRecord(
            thread_id="thread-00",
            customer_id="cust-100",
            intent="billing_inquiry",
            created_at=base_time,
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[past_candidate, exact_first_response_candidate],
            intent_similarity_threshold=0.7,
        )
        assert result is False

    def test_boundary_exact_window_end_included(self, base_time):
        """A candidate created exactly at first_response_at + 48 hours is included."""
        boundary_candidate = ThreadRecord(
            thread_id="thread-boundary",
            customer_id="cust-100",
            intent="billing_inquiry",
            created_at=base_time + timedelta(hours=48),
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[boundary_candidate],
            intent_similarity_threshold=0.7,
        )
        assert result is True

    def test_boundary_after_window_excluded(self, base_time):
        """A candidate created strictly after the window (e.g. at 48h 1s or 49h) is excluded."""
        out_of_window_candidate = ThreadRecord(
            thread_id="thread-late",
            customer_id="cust-100",
            intent="billing_inquiry",
            created_at=base_time + timedelta(hours=48, seconds=1),
        )
        candidate_49h = ThreadRecord(
            thread_id="thread-49h",
            customer_id="cust-100",
            intent="billing_inquiry",
            created_at=base_time + timedelta(hours=49),
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[out_of_window_candidate, candidate_49h],
            intent_similarity_threshold=0.7,
        )
        assert result is False

    def test_dissimilar_intent_excluded(self, base_time):
        """A customer contacting within window with an unrelated intent is not a repeat contact."""
        candidate = ThreadRecord(
            thread_id="thread-unrelated",
            customer_id="cust-100",
            intent="how_to_export_csv_reports",  # completely different intent
            created_at=base_time + timedelta(hours=12),
        )
        result = has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[candidate],
            intent_similarity_threshold=0.7,
        )
        assert result is False

    def test_empty_candidates(self, base_time):
        """No candidates returns False."""
        assert has_repeat_contact_within_window(
            customer_id="cust-100",
            original_thread_id="thread-1",
            original_intent="billing_inquiry",
            first_response_at=base_time,
            window_hours=48,
            candidate_threads=[],
        ) is False
