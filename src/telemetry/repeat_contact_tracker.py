"""
Repeat Contact Tracking and Intent Similarity Matching.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.2 (M7.P7.2.F1)

Detects whether a customer initiated a repeat contact within a defined time window
(typically 48 hours) regarding the same or highly similar issue/intent.
This serves as the behavioral proxy for unresolved containment.
"""

from datetime import datetime, timedelta
import difflib
import logging
import re
from typing import List

from src.telemetry.schema import ThreadRecord

logger = logging.getLogger(__name__)


# ==============================================================================
# Architectural Note on Search Complexity:
# ------------------------------------------------------------------------------
# In a naive implementation, scanning all candidate threads for every contained
# conversation requires O(N * M) ~ O(N^2) comparisons, where N is the number of
# contained events and M is the total thread volume across the contact window.
#
# Production Scalability Recommendation:
# For production scale handling tens of thousands of conversations daily:
# 1. Candidate threads must be indexed by (customer_id, created_at) in the data store.
# 2. Query only candidate threads where customer_id matches and
#    first_response_at < created_at <= first_response_at + window_hours.
# 3. This reduces the search space per contained event from O(M) to O(K), where K
#    is the small number of interactions by that specific customer within the window.
# ==============================================================================


def compute_intent_similarity(intent_a: str, intent_b: str) -> float:
    """Compute lexical and token-overlap similarity between two customer intent strings.
    
    Returns a float between 0.0 and 1.0 based on the maximum of:
    1. Token-level Jaccard similarity (order-independent bag-of-words overlap).
    2. Character-level SequenceMatcher ratio (handles slight typos / stems).
    
    Args:
        intent_a: First intent string (e.g. "billing_inquiry").
        intent_b: Second intent string (e.g. "billing_question").
        
    Returns:
        Similarity score between 0.0 and 1.0 (rounded to 4 decimal places).
    """
    if not intent_a or not intent_b:
        return 0.0

    norm_a = intent_a.lower().strip()
    norm_b = intent_b.lower().strip()

    if norm_a == norm_b:
        return 1.0

    tokens_a = set(re.findall(r"\w+", norm_a))
    tokens_b = set(re.findall(r"\w+", norm_b))

    jaccard = 0.0
    if tokens_a and tokens_b:
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        jaccard = len(intersection) / len(union)

    seq_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    return round(max(jaccard, seq_ratio), 4)


def has_repeat_contact_within_window(
    customer_id: str,
    original_thread_id: str,
    original_intent: str,
    first_response_at: datetime,
    window_hours: int,
    candidate_threads: List[ThreadRecord],
    intent_similarity_threshold: float = 0.7,
) -> bool:
    """Determine whether a customer initiated a repeat contact within the defined window.
    
    Criteria for a repeat contact:
    1. Customer match: candidate.customer_id == customer_id.
    2. Distinct thread: candidate.thread_id != original_thread_id.
    3. Window boundary: first_response_at < candidate.created_at <= first_response_at + window_hours.
       (Threads created prior to or at the original first response, or strictly after the
       window boundary e.g. at 49h, are strictly excluded).
    4. Intent similarity: compute_intent_similarity(original_intent, candidate.intent) >= threshold.
       (Unrelated inquiries within the window, such as a feature request following a billing
       question, do not count as repeat contact for the original issue).
       
    Args:
        customer_id: Unique customer identifier.
        original_thread_id: ID of the contained thread being evaluated.
        original_intent: Intent detected in the original thread.
        first_response_at: Datetime of the first AI response in the original thread.
        window_hours: Duration in hours for the repeat contact window (e.g. 48).
        candidate_threads: List of customer interaction ThreadRecords.
        intent_similarity_threshold: Minimum similarity score to consider intents matching.
        
    Returns:
        True if an intent-matched repeat contact occurred within the window, False otherwise.
    """
    window_end = first_response_at + timedelta(hours=window_hours)

    for candidate in candidate_threads:
        # 1. Must be the same customer
        if candidate.customer_id != customer_id:
            continue

        # 2. Must be a different thread
        if candidate.thread_id == original_thread_id:
            continue

        # 3. Must be strictly within the time window: (first_response_at, window_end]
        if candidate.created_at <= first_response_at:
            continue
        if candidate.created_at > window_end:
            continue

        # 4. Must match intent similarity threshold
        similarity = compute_intent_similarity(original_intent, candidate.intent)
        if similarity >= intent_similarity_threshold:
            logger.info(
                "Repeat contact detected for customer %s (original thread %s, repeat thread %s): "
                "intent '%s' vs '%s' (similarity=%.2f >= %.2f)",
                customer_id,
                original_thread_id,
                candidate.thread_id,
                original_intent,
                candidate.intent,
                similarity,
                intent_similarity_threshold,
            )
            return True

    return False
