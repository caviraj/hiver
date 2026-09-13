"""Unit tests for Hiver contextual handoff note assembly, formatting, and posting (M6.P6.4.F1).

Validates:
- Full handoff assembly with all fields present
- Severity-bypass path with no classification (predicted_intent=None, empty retrieval)
- Missing draft_response completely omitting the AI draft section
- Present draft_response clearly labeled as unsent for reference only
- Stale/missing doc_id reference producing a graceful placeholder without crashing
- Snippet truncation at the 200-character limit with trailing '...'
- Empty retrieved chunks producing an explicit 'none found' message
- Special characters and formatting robustness (newlines, quotes, unicode)
- post_handoff_note successful execution with MockHiverClient
- post_handoff_note append failure logging CRITICAL on exhaustion
- post_handoff_note client exception logging CRITICAL and returning False
"""

from __future__ import annotations

import logging
from typing import List
import pytest

from src.integrations.hiver.client import MockHiverClient
from src.integrations.hiver.handoff_note import (
    AI_DRAFT_HEADER,
    NO_RESOLUTIONS_FOUND,
    SEVERITY_BYPASS_TEXT,
    SNIPPET_MAX_LEN,
    UNAVAILABLE_SNIPPET,
    assemble_handoff_content,
    format_note_text,
    post_handoff_note,
)
from src.integrations.hiver.schema import HandoffNoteContent
from src.models.intent.escalation_router import EscalationDecision
from src.retrieval.schema import RetrievalDocument, RetrievalResult


@pytest.fixture
def sample_decision() -> EscalationDecision:
    """Fixture providing a standard EscalationDecision."""
    return EscalationDecision(
        thread_id="th_12345",
        tweet_id="tw_98765",
        bypassed=False,
        bypass_reason="Confidence below threshold",
        predicted_intent="Billing_Issue",
        confidence=0.68,
        escalation_tier="standard",
    )


@pytest.fixture
def sample_corpus() -> List[RetrievalDocument]:
    """Fixture providing a sample grounding corpus."""
    return [
        RetrievalDocument(
            doc_id="doc_1",
            query_text="How do I get a refund for an overcharge?",
            resolution_text="We have reviewed your account and refunded the $25 fee to your primary card.",
        ),
        RetrievalDocument(
            doc_id="doc_2",
            query_text="Why was my card billed twice?",
            resolution_text="Duplicate charges are reversed within 3-5 business days. Please monitor your statement.",
        ),
    ]


@pytest.fixture
def sample_retrieval_results() -> List[RetrievalResult]:
    """Fixture providing sample retrieval results."""
    return [
        RetrievalResult(doc_id="doc_1", score=0.95, rank=1),
        RetrievalResult(doc_id="doc_2", score=0.82, rank=2),
    ]


def test_assemble_handoff_content_full(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
):
    """Verify handoff assembly correctly maps fields when all artifacts are present."""
    draft = "Thank you for reaching out. We are processing your request."
    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=sample_retrieval_results,
        corpus=sample_corpus,
        draft_response=draft,
    )

    assert isinstance(content, HandoffNoteContent)
    assert content.thread_id == "th_12345"
    assert content.escalation_tier == "standard"
    assert content.escalation_reason == "Confidence below threshold"
    assert content.predicted_intent == "Billing_Issue"
    assert content.confidence == 0.68
    assert len(content.retrieved_chunks) == 2
    assert "refunded the $25 fee" in content.retrieved_chunks[0]
    assert "Duplicate charges are reversed" in content.retrieved_chunks[1]
    assert content.draft_response_if_any == draft


def test_format_note_text_full(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
):
    """Verify formatted plain-text note produces clean, labeled sections."""
    draft = "Please let us know your account number so we can investigate."
    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=sample_retrieval_results,
        corpus=sample_corpus,
        draft_response=draft,
    )
    note_text = format_note_text(content)

    assert "=== AI HANDOFF SUMMARY ===" in note_text
    assert "Thread ID: th_12345" in note_text
    assert "Escalation Tier: STANDARD" in note_text
    assert "Reason: Confidence below threshold" in note_text
    assert "Timestamp:" in note_text

    assert "=== CLASSIFICATION & INTENT ===" in note_text
    assert "Predicted Intent: Billing_Issue (Confidence: 0.68)" in note_text

    assert "=== HISTORICAL CONTEXT ===" in note_text
    assert "[1] We have reviewed your account and refunded the $25 fee" in note_text
    assert "[2] Duplicate charges are reversed within 3-5 business days" in note_text

    assert AI_DRAFT_HEADER in note_text
    assert draft in note_text


def test_severity_bypass_with_no_classification():
    """Verify severity bypass path formats correctly without classification data."""
    decision = EscalationDecision(
        thread_id="th_bypass_critical",
        tweet_id="tw_critical_1",
        bypassed=True,
        bypass_reason="Keyword match: legal lawsuit imminent",
        predicted_intent=None,
        confidence=None,
        escalation_tier="critical",
    )

    content = assemble_handoff_content(
        decision=decision,
        retrieval_results=[],
        corpus=[],
        draft_response=None,
    )
    note_text = format_note_text(content)

    # Must clearly explain bypass rather than misleading "Intent: None"
    assert SEVERITY_BYPASS_TEXT in note_text
    assert "Classification skipped — routed directly on severity signal" in note_text
    assert "Intent: None" not in note_text
    assert "Predicted Intent: None" not in note_text

    # Historical context should state none found
    assert NO_RESOLUTIONS_FOUND in note_text

    # AI Draft section must be completely absent
    assert AI_DRAFT_HEADER not in note_text
    assert "AI DRAFT" not in note_text


def test_missing_draft_response_omits_section_entirely(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
):
    """Verify that omitting draft_response leaves no trace of draft header or placeholder."""
    # Test with None
    content_none = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=sample_retrieval_results,
        corpus=sample_corpus,
        draft_response=None,
    )
    note_none = format_note_text(content_none)
    assert AI_DRAFT_HEADER not in note_none
    assert "AI DRAFT" not in note_none

    # Test with empty/whitespace string
    content_blank = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=sample_retrieval_results,
        corpus=sample_corpus,
        draft_response="   \n\t  ",
    )
    note_blank = format_note_text(content_blank)
    assert AI_DRAFT_HEADER not in note_blank
    assert "AI DRAFT" not in note_blank


def test_present_draft_response_clearly_labeled(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
):
    """Verify that when a draft exists, it is explicitly labeled as unsent for reference only."""
    draft = "We have escalated this ticket to our senior tier."
    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=sample_retrieval_results,
        corpus=sample_corpus,
        draft_response=draft,
    )
    note_text = format_note_text(content)

    assert "=== AI DRAFT — NOT SENT — for reference only ===" in note_text
    assert draft in note_text


def test_stale_doc_id_reference_graceful_placeholder(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
):
    """Verify a stale or ungrounded doc_id results in a placeholder rather than a crash."""
    results = [
        RetrievalResult(doc_id="doc_1", score=0.9, rank=1),
        RetrievalResult(doc_id="doc_nonexistent_999", score=0.5, rank=2),
    ]

    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=results,
        corpus=sample_corpus,
    )

    assert len(content.retrieved_chunks) == 2
    assert "refunded the $25 fee" in content.retrieved_chunks[0]
    assert content.retrieved_chunks[1] == UNAVAILABLE_SNIPPET

    note_text = format_note_text(content)
    assert f"[2] {UNAVAILABLE_SNIPPET}" in note_text


def test_empty_resolution_text_in_corpus_uses_placeholder(sample_decision: EscalationDecision):
    """Verify that a document with empty/whitespace resolution_text uses the unavailable placeholder."""
    empty_doc = RetrievalDocument(
        doc_id="doc_empty",
        query_text="Sample query",
        resolution_text="    ",
    )
    results = [RetrievalResult(doc_id="doc_empty", score=0.8, rank=1)]

    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=results,
        corpus=[empty_doc],
    )

    assert content.retrieved_chunks == [UNAVAILABLE_SNIPPET]


def test_snippet_truncation_at_limit(sample_decision: EscalationDecision):
    """Verify long historical resolution snippets are truncated to SNIPPET_MAX_LEN with '...'."""
    long_resolution = "A" * (SNIPPET_MAX_LEN + 100)
    short_resolution = "B" * 50

    corpus = [
        RetrievalDocument(doc_id="doc_long", query_text="Long query", resolution_text=long_resolution),
        RetrievalDocument(doc_id="doc_short", query_text="Short query", resolution_text=short_resolution),
    ]
    results = [
        RetrievalResult(doc_id="doc_long", score=0.9, rank=1),
        RetrievalResult(doc_id="doc_short", score=0.8, rank=2),
    ]

    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=results,
        corpus=corpus,
    )

    # Long snippet should be truncated to 200 chars + "..."
    assert len(content.retrieved_chunks[0]) == SNIPPET_MAX_LEN + 3
    assert content.retrieved_chunks[0] == ("A" * SNIPPET_MAX_LEN) + "..."

    # Short snippet should remain untruncated without "..."
    assert len(content.retrieved_chunks[1]) == 50
    assert content.retrieved_chunks[1] == "B" * 50
    assert not content.retrieved_chunks[1].endswith("...")


def test_empty_retrieved_chunks_produces_explicit_message(sample_decision: EscalationDecision):
    """Verify empty retrieval outputs an explicit message rather than a blank section."""
    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=[],
        corpus=[],
    )
    note_text = format_note_text(content)

    assert "=== HISTORICAL CONTEXT ===" in note_text
    assert NO_RESOLUTIONS_FOUND in note_text


def test_special_characters_formatting_robustness(sample_decision: EscalationDecision):
    """Verify special characters, embedded newlines, quotes, and unicode format cleanly."""
    complex_resolution = (
        'Customer stated: "My bill is $50 too high!"\n'
        'Resolution: Credited $50.00 — verified via Ledger ID #9988.\n'
        'Notes: <Customer verified via SMS: "Code 1234"> 🎉'
    )
    complex_draft = (
        'Hello "Valued Customer",\n\n'
        'We\'ve adjusted your bill by -$50.00.\n'
        'Reference ID: \\REF#4321\\.\n'
        'Best regards,\nCustomer Support Team'
    )

    corpus = [
        RetrievalDocument(
            doc_id="doc_special",
            query_text="Complex billing inquiry",
            resolution_text=complex_resolution,
        )
    ]
    results = [RetrievalResult(doc_id="doc_special", score=0.91, rank=1)]

    content = assemble_handoff_content(
        decision=sample_decision,
        retrieval_results=results,
        corpus=corpus,
        draft_response=complex_draft,
    )
    note_text = format_note_text(content)

    assert complex_resolution in note_text
    assert complex_draft in note_text
    assert "🎉" in note_text
    assert "\\REF#4321\\" in note_text


def test_post_handoff_note_success(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
):
    """Verify post_handoff_note successfully calls client and stores note."""
    client = MockHiverClient()
    success = post_handoff_note(
        decision=sample_decision,
        retrieval_results=sample_retrieval_results,
        corpus=sample_corpus,
        client=client,
        draft_response="Draft response for testing.",
    )

    assert success is True
    assert len(client.internal_notes["th_12345"]) == 1
    stored_note = client.internal_notes["th_12345"][0]
    assert "=== AI HANDOFF SUMMARY ===" in stored_note
    assert "=== AI DRAFT — NOT SENT — for reference only ===" in stored_note

    # Verify client call history
    call = client.calls[-1]
    assert call["method"] == "append_internal_note"
    assert call["thread_id"] == "th_12345"
    assert "note" in call


def test_post_handoff_note_failure_logs_critical(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
    caplog: pytest.LogCaptureFixture,
):
    """Verify post_handoff_note logs a CRITICAL alert when append_internal_note fails."""
    client = MockHiverClient(fail_note=True)

    with caplog.at_level(logging.CRITICAL):
        success = post_handoff_note(
            decision=sample_decision,
            retrieval_results=sample_retrieval_results,
            corpus=sample_corpus,
            client=client,
        )

    assert success is False
    assert any(
        record.levelno == logging.CRITICAL
        and "CRITICAL: Failed to append internal handoff note" in record.message
        and "th_12345" in record.message
        for record in caplog.records
    )


def test_post_handoff_note_exception_logs_critical(
    sample_decision: EscalationDecision,
    sample_corpus: List[RetrievalDocument],
    sample_retrieval_results: List[RetrievalResult],
    caplog: pytest.LogCaptureFixture,
):
    """Verify post_handoff_note catches exceptions, logs CRITICAL, and returns False."""
    client = MockHiverClient(note_exception=ConnectionResetError("Socket reset by peer"))

    with caplog.at_level(logging.CRITICAL):
        success = post_handoff_note(
            decision=sample_decision,
            retrieval_results=sample_retrieval_results,
            corpus=sample_corpus,
            client=client,
        )

    assert success is False
    assert any(
        record.levelno == logging.CRITICAL
        and "CRITICAL: Exception occurred while posting internal handoff note" in record.message
        and "Socket reset by peer" in record.message
        for record in caplog.records
    )
