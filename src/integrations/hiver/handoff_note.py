"""Contextual handoff note assembly, formatting, and posting for Hiver (M6.P6.4.F1).

Upon escalation, assembles internal reasoning, retrieved historical resolutions,
and un-sent drafts into a private internal note in the Hiver thread so human
operators have full situational context without re-deriving the investigation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.integrations.hiver.client import HiverClient
from src.integrations.hiver.schema import HandoffNoteContent
from src.models.intent.escalation_router import EscalationDecision
from src.retrieval.schema import RetrievalDocument, RetrievalResult

logger = logging.getLogger(__name__)

# Tunable constants
SNIPPET_MAX_LEN: int = 200
UNAVAILABLE_SNIPPET: str = "[resolution unavailable]"
NO_RESOLUTIONS_FOUND: str = "No relevant historical resolutions found"
SEVERITY_BYPASS_TEXT: str = "Classification skipped — routed directly on severity signal"
AI_DRAFT_HEADER: str = "=== AI DRAFT — NOT SENT — for reference only ==="


def assemble_handoff_content(
    decision: EscalationDecision,
    retrieval_results: List[RetrievalResult],
    corpus: List[RetrievalDocument],
    draft_response: Optional[str] = None,
) -> HandoffNoteContent:
    """Assemble structured handoff note data from pipeline artifacts.

    Looks up each RetrievalResult's doc_id in the provided corpus to extract
    the resolution_text snippet, truncating long snippets to SNIPPET_MAX_LEN
    for scannability. Stale references gracefully fall back to a placeholder.

    Args:
        decision: EscalationDecision from upstream routing (M2.P2.3).
        retrieval_results: List of ranked RetrievalResult items from HybridRetriever (M3.P3.3).
        corpus: List of grounding RetrievalDocument items in the corpus.
        draft_response: Optional un-sent draft response generated before escalation.

    Returns:
        HandoffNoteContent Pydantic model populated with structured context.
    """
    escalation_reason = (
        decision.bypass_reason
        if decision.bypass_reason
        else f"Escalated to {decision.escalation_tier}"
    )

    doc_map = {doc.doc_id: doc for doc in corpus}
    snippets: List[str] = []

    for res in retrieval_results:
        doc = doc_map.get(res.doc_id)
        if doc and doc.resolution_text and doc.resolution_text.strip():
            raw_text = doc.resolution_text.strip()
            if len(raw_text) > SNIPPET_MAX_LEN:
                snippet = raw_text[:SNIPPET_MAX_LEN] + "..."
            else:
                snippet = raw_text
            snippets.append(snippet)
        else:
            snippets.append(UNAVAILABLE_SNIPPET)

    return HandoffNoteContent(
        thread_id=decision.thread_id,
        escalation_reason=escalation_reason,
        escalation_tier=decision.escalation_tier,
        predicted_intent=decision.predicted_intent,
        confidence=decision.confidence,
        retrieved_chunks=snippets,
        draft_response_if_any=draft_response,
    )


def format_note_text(content: HandoffNoteContent) -> str:
    """Format structured handoff content into a scannable, human-readable plain-text note.

    Produces a cleanly structured note suited for Gmail-overlay UI in Hiver:
    1. Summary header (Thread ID, Tier, Reason, Timestamp)
    2. Classification & Intent (with confidence or clear severity-bypass indicator)
    3. Historical Context (top snippets or explicit 'none found' message)
    4. AI Draft (only if present, clearly marked as un-sent for reference only)

    Args:
        content: Populated HandoffNoteContent model.

    Returns:
        Formatted multi-line plain text note.
    """
    sections: List[str] = []

    # Section 1: Summary Header
    formatted_timestamp = content.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_lines = [
        "=== AI HANDOFF SUMMARY ===",
        f"Thread ID: {content.thread_id}",
        f"Escalation Tier: {content.escalation_tier.upper()}",
        f"Reason: {content.escalation_reason}",
        f"Timestamp: {formatted_timestamp}",
    ]
    sections.append("\n".join(summary_lines))

    # Section 2: Classification & Intent
    intent_lines = ["=== CLASSIFICATION & INTENT ==="]
    if content.predicted_intent is None:
        intent_lines.append(f"Classification: {SEVERITY_BYPASS_TEXT}")
    else:
        conf_str = f"{content.confidence:.2f}" if content.confidence is not None else "N/A"
        intent_lines.append(f"Predicted Intent: {content.predicted_intent} (Confidence: {conf_str})")
    sections.append("\n".join(intent_lines))

    # Section 3: Historical Context
    history_lines = ["=== HISTORICAL CONTEXT ==="]
    if not content.retrieved_chunks:
        history_lines.append(NO_RESOLUTIONS_FOUND)
    else:
        for idx, chunk in enumerate(content.retrieved_chunks, 1):
            history_lines.append(f"[{idx}] {chunk}")
    sections.append("\n".join(history_lines))

    # Section 4: AI Draft (omitted entirely if None or empty)
    if content.draft_response_if_any is not None and content.draft_response_if_any.strip():
        draft_lines = [
            AI_DRAFT_HEADER,
            content.draft_response_if_any.strip(),
        ]
        sections.append("\n".join(draft_lines))

    return "\n\n".join(sections)


def post_handoff_note(
    decision: EscalationDecision,
    retrieval_results: List[RetrievalResult],
    corpus: List[RetrievalDocument],
    client: HiverClient,
    draft_response: Optional[str] = None,
) -> bool:
    """Orchestrate handoff note assembly, formatting, and posting to Hiver.

    Catches any exceptions and logs at CRITICAL level if appending fails,
    ensuring failures do not raise or disrupt upstream routing.

    Args:
        decision: EscalationDecision from upstream routing.
        retrieval_results: Ranked retrieval results.
        corpus: Grounding document corpus.
        client: HiverClient instance (real API client or mock).
        draft_response: Optional un-sent AI draft.

    Returns:
        True if note was successfully appended, False otherwise.
    """
    try:
        content = assemble_handoff_content(
            decision=decision,
            retrieval_results=retrieval_results,
            corpus=corpus,
            draft_response=draft_response,
        )
        note_text = format_note_text(content)
        success = client.append_internal_note(decision.thread_id, note_text)

        if not success:
            logger.critical(
                "CRITICAL: Failed to append internal handoff note for thread '%s' (tier '%s'). "
                "Human reviewer will have no AI situational context.",
                decision.thread_id,
                decision.escalation_tier,
            )
            return False

        logger.info(
            "Successfully appended internal handoff note for thread '%s' (tier '%s').",
            decision.thread_id,
            decision.escalation_tier,
        )
        return True
    except Exception as exc:
        logger.critical(
            "CRITICAL: Exception occurred while posting internal handoff note for thread '%s': %s",
            decision.thread_id,
            exc,
            exc_info=True,
        )
        return False
