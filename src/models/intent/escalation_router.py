"""High-Severity Bypass and Escalation Routing Gate.

Phase: M2.P2.3.F1
Evaluates severity signals prior to full inference/retrieval and determines routing:
skip RAG for critical/urgent escalations or route to human vs auto-response.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_CRITICAL_LABELS: set[str] = {
    "Outrage_Escalation",
    "Legal_Threat",
    "Safety_Hazard",
}

DEFAULT_LOG_PATH = Path("data/logs/escalation_log.jsonl")


class EscalationDecision(BaseModel):
    """Routing decision determining whether a message bypasses RAG and routes to human queues."""

    thread_id: str
    tweet_id: str
    bypassed: bool
    bypass_reason: Optional[str] = None
    predicted_intent: Optional[str] = None
    confidence: Optional[float] = None
    escalation_tier: Literal["none", "standard", "urgent", "critical"]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def evaluate_severity_bypass(
    text: str,
    keyword_flag: bool,
    llm_label: Optional[str],
    critical_labels: Optional[Set[str]] = None,
) -> bool:
    """Evaluate whether a message triggers dual-gate high-severity bypass.

    Pure function with no side effects (no DB writes, no HTTP calls).
    Returns True ONLY when text is non-empty, keyword_flag is True, AND llm_label is in critical_labels.

    Args:
        text: Raw text of the incoming message.
        keyword_flag: Boolean flag from severity keyword rules.
        llm_label: Cluster/intent label assigned by upstream LLM classifier.
        critical_labels: Configurable set of critical labels. Defaults to DEFAULT_CRITICAL_LABELS.

    Returns:
        True if dual-gate consensus agrees on critical severity, False otherwise.
    """
    if not text or not text.strip():
        return False

    if critical_labels is None:
        critical_labels = DEFAULT_CRITICAL_LABELS

    # Upstream data quality edge-case: keyword is flagged but LLM label is missing (None)
    if keyword_flag and llm_label is None:
        logger.warning(
            "Severity keyword flagged True, but upstream LLM label is None. "
            "Treating as LLM agreement absent and falling back from critical bypass."
        )
        return False

    return bool(keyword_flag and (llm_label in critical_labels))


def append_escalation_log(
    decision: EscalationDecision,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> None:
    """Stream-append an EscalationDecision with bypassed=True to JSONL log.

    Never loads the whole log into memory. Catches IO exceptions to ensure routing
    pipeline resilience.

    Args:
        decision: EscalationDecision object to record.
        log_path: Target path for the JSONL log.
    """
    if not decision.bypassed:
        return

    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = decision.model_dump()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # A logging failure must not crash the pipeline - escalation must still proceed.
        logger.error(
            "Failed to append escalation decision %s to log %s: %s",
            decision.tweet_id,
            log_path,
            exc,
            exc_info=True,
        )


def route_message(
    context_record: Mapping[str, Any],
    classifier_output: Optional[tuple[str, float]],
    severity_auditor: Optional[Any] = None,
    confidence_threshold: float = 0.65,
    critical_labels: Optional[Set[str]] = None,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> EscalationDecision:
    """Evaluate severity bypass and confidence gate to determine routing path.

    Args:
        context_record: Dictionary containing message metadata:
            - 'text': message content (str)
            - 'keyword_flag': bool
            - 'llm_label': Optional[str]
            - 'thread_id': Optional[str]
            - 'tweet_id': Optional[str]
        classifier_output: (predicted_intent, confidence_score) or None if bypassed upstream.
        severity_auditor: Optional SeverityAuditor to record signal disagreements.
        confidence_threshold: Threshold for confident classification (default 0.65).
        critical_labels: Configurable set of critical labels. Defaults to DEFAULT_CRITICAL_LABELS.
        log_path: Path to write escalation log records (default data/logs/escalation_log.jsonl).

    Returns:
        EscalationDecision object with routing instructions.
    """
    if critical_labels is None:
        critical_labels = DEFAULT_CRITICAL_LABELS

    text = str(context_record.get("text", "") or "")
    keyword_flag = bool(context_record.get("keyword_flag", False))
    llm_label = context_record.get("llm_label")
    thread_id = str(context_record.get("thread_id", ""))
    tweet_id = str(context_record.get("tweet_id", ""))

    # Step 1: Run evaluate_severity_bypass
    is_bypass = evaluate_severity_bypass(
        text=text,
        keyword_flag=keyword_flag,
        llm_label=llm_label,
        critical_labels=critical_labels,
    )

    # Record disagreement telemetry if auditor is supplied
    if severity_auditor is not None and hasattr(severity_auditor, "record_disagreement"):
        if keyword_flag and not is_bypass:
            reason = f"Keyword flagged True but llm_label='{llm_label}' not in critical_labels"
            try:
                severity_auditor.record_disagreement(
                    tweet_id=tweet_id,
                    text=text,
                    keyword_flag=keyword_flag,
                    llm_flag=False,
                    reason=reason,
                )
            except Exception as exc:
                logger.warning("Failed to record disagreement in auditor: %s", exc)

    if is_bypass:
        decision = EscalationDecision(
            thread_id=thread_id,
            tweet_id=tweet_id,
            bypassed=True,
            bypass_reason="severity_keyword+llm_agree",
            predicted_intent=None,
            confidence=None,
            escalation_tier="critical",
        )
        append_escalation_log(decision, log_path=log_path)
        return decision

    # Step 2: Evaluate classifier confidence if not bypassed
    pred_intent: Optional[str] = None
    confidence: Optional[float] = None

    if classifier_output is not None:
        pred_intent, confidence = classifier_output

    # Low confidence handling (< threshold) or missing classifier output
    # Boundary note: confidence >= threshold passes; strictly < fails.
    if confidence is None or float(confidence) < float(confidence_threshold):
        is_critical_adjacent = pred_intent in critical_labels if pred_intent else False
        tier: Literal["urgent", "standard"] = "urgent" if is_critical_adjacent else "standard"

        decision = EscalationDecision(
            thread_id=thread_id,
            tweet_id=tweet_id,
            bypassed=True,
            bypass_reason="low_confidence",
            predicted_intent=pred_intent,
            confidence=confidence,
            escalation_tier=tier,
        )
        append_escalation_log(decision, log_path=log_path)
        return decision

    # Step 3: Confidence >= threshold AND no severity bypass -> proceed to RAG
    decision = EscalationDecision(
        thread_id=thread_id,
        tweet_id=tweet_id,
        bypassed=False,
        bypass_reason=None,
        predicted_intent=pred_intent,
        confidence=confidence,
        escalation_tier="none",
    )
    # bypassed=False is NOT appended to escalation_log (only bypassed=True reaches human queue)
    return decision
