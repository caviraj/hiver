"""Tests for high-severity bypass and escalation routing logic.

Phase: M2.P2.3.F1
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models.intent.escalation_router import (
    DEFAULT_CRITICAL_LABELS,
    EscalationDecision,
    append_escalation_log,
    evaluate_severity_bypass,
    route_message,
)
from src.models.intent.severity_rules import SeverityAuditor


def test_dual_gate_bypass_triggers_on_both_signals(tmp_path: Path) -> None:
    """When keyword_flag is True and llm_label is in critical_labels, bypass must trigger."""
    context = {
        "text": "I will sue your company, this is fraud!",
        "keyword_flag": True,
        "llm_label": "Legal_Threat",
        "thread_id": "th_101",
        "tweet_id": "tw_101",
    }
    log_file = tmp_path / "escalation_log.jsonl"
    decision = route_message(
        context_record=context,
        classifier_output=("Legal_Threat", 0.95),
        log_path=log_file,
    )

    assert decision.bypassed is True
    assert decision.escalation_tier == "critical"
    assert decision.bypass_reason == "severity_keyword+llm_agree"
    assert decision.predicted_intent is None
    assert decision.confidence is None

    # Check that it was appended to log
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["tweet_id"] == "tw_101"
    assert data["escalation_tier"] == "critical"
    assert data["bypassed"] is True


def test_keyword_only_does_not_trigger_bypass(tmp_path: Path) -> None:
    """Single signal (keyword=True, llm_label not in critical) must NOT trigger critical bypass."""
    context = {
        "text": "Where is my money? Urgent!",
        "keyword_flag": True,
        "llm_label": "Refund_Request",  # Not in critical_labels
        "thread_id": "th_102",
        "tweet_id": "tw_102",
    }
    log_file = tmp_path / "escalation_log.jsonl"
    auditor = MagicMock(spec=SeverityAuditor)

    # Confident classification (>= 0.65)
    decision = route_message(
        context_record=context,
        classifier_output=("Refund_Request", 0.88),
        severity_auditor=auditor,
        confidence_threshold=0.65,
        log_path=log_file,
    )

    # Should NOT be bypassed, tier none, proceed to RAG
    assert decision.bypassed is False
    assert decision.escalation_tier == "none"
    assert decision.predicted_intent == "Refund_Request"
    assert decision.confidence == 0.88

    # Auditor should have recorded disagreement
    auditor.record_disagreement.assert_called_once()

    # Should NOT be appended to escalation_log since bypassed is False
    assert not log_file.exists()


def test_llm_label_none_fallback_and_warning(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """keyword_flag=True but llm_label=None must log warning and fall back to confidence gate."""
    context = {
        "text": "Report you to consumer protection agency",
        "keyword_flag": True,
        "llm_label": None,  # LLM labeling failed upstream
        "thread_id": "th_103",
        "tweet_id": "tw_103",
    }
    log_file = tmp_path / "escalation_log.jsonl"

    with caplog.at_level("WARNING"):
        decision = route_message(
            context_record=context,
            classifier_output=("General_Inquiry", 0.72),
            confidence_threshold=0.65,
            log_path=log_file,
        )

    # Should not trigger critical bypass
    assert decision.bypassed is False
    assert decision.escalation_tier == "none"
    assert "Severity keyword flagged True, but upstream LLM label is None" in caplog.text


def test_confidence_boundary_equality_passes_gate(tmp_path: Path) -> None:
    """Confidence exactly equal to threshold (0.65 == 0.65) passes gate (>= threshold)."""
    context = {
        "text": "How do I update my profile?",
        "keyword_flag": False,
        "llm_label": "Profile_Update",
        "thread_id": "th_104",
        "tweet_id": "tw_104",
    }
    decision = route_message(
        context_record=context,
        classifier_output=("Profile_Update", 0.65),
        confidence_threshold=0.65,
        log_path=tmp_path / "escalation_log.jsonl",
    )
    assert decision.bypassed is False
    assert decision.escalation_tier == "none"
    assert decision.predicted_intent == "Profile_Update"
    assert decision.confidence == 0.65


def test_urgent_vs_standard_low_confidence_tier(tmp_path: Path) -> None:
    """Low-confidence on high-severity-adjacent label gives 'urgent'; otherwise 'standard'."""
    log_file = tmp_path / "escalation_log.jsonl"

    # 1. Low confidence on critical-adjacent label (e.g. Outrage_Escalation)
    context_urgent = {
        "text": "Someone help this is ridiculous!",
        "keyword_flag": False,
        "llm_label": "General_Complaint",
        "thread_id": "th_105a",
        "tweet_id": "tw_105a",
    }
    decision_urgent = route_message(
        context_record=context_urgent,
        classifier_output=("Outrage_Escalation", 0.55),
        confidence_threshold=0.65,
        log_path=log_file,
    )
    assert decision_urgent.bypassed is True
    assert decision_urgent.escalation_tier == "urgent"
    assert decision_urgent.bypass_reason == "low_confidence"
    assert decision_urgent.predicted_intent == "Outrage_Escalation"

    # 2. Low confidence on normal topic (e.g. FAQ / Balance_Check)
    context_std = {
        "text": "What is my current balance?",
        "keyword_flag": False,
        "llm_label": "Balance_Check",
        "thread_id": "th_105b",
        "tweet_id": "tw_105b",
    }
    decision_std = route_message(
        context_record=context_std,
        classifier_output=("Balance_Check", 0.40),
        confidence_threshold=0.65,
        log_path=log_file,
    )
    assert decision_std.bypassed is True
    assert decision_std.escalation_tier == "standard"
    assert decision_std.bypass_reason == "low_confidence"
    assert decision_std.predicted_intent == "Balance_Check"

    # Both should be written to log
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    tiers = [json.loads(line)["escalation_tier"] for line in lines]
    assert tiers == ["urgent", "standard"]


def test_empty_text_input_handled_gracefully(tmp_path: Path) -> None:
    """Empty text input must skip severity bypass and route through confidence gate."""
    context = {
        "text": "",
        "keyword_flag": True,  # should be ignored since text is empty
        "llm_label": "Legal_Threat",
        "thread_id": "th_106",
        "tweet_id": "tw_106",
    }
    decision = route_message(
        context_record=context,
        classifier_output=("Greeting", 0.90),
        confidence_threshold=0.65,
        log_path=tmp_path / "escalation_log.jsonl",
    )
    assert decision.bypassed is False
    assert decision.escalation_tier == "none"


def test_log_write_failure_does_not_crash_router(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """If escalation_log.jsonl append raises an exception, the router must not crash."""
    context = {
        "text": "I will file a lawsuit right now!",
        "keyword_flag": True,
        "llm_label": "Legal_Threat",
        "thread_id": "th_107",
        "tweet_id": "tw_107",
    }

    # Point to an invalid path or monkeypatch open to simulate permission/disk error
    def mock_open(*args, **kwargs):
        raise OSError("Disk full or permission denied")

    monkeypatch.setattr(Path, "open", mock_open)

    with caplog.at_level("ERROR"):
        decision = route_message(
            context_record=context,
            classifier_output=("Legal_Threat", 0.99),
            log_path="/forbidden/escalation_log.jsonl",
        )

    # Escalation decision must still be returned successfully!
    assert decision.bypassed is True
    assert decision.escalation_tier == "critical"
    assert "Failed to append escalation decision tw_107" in caplog.text


def test_evaluate_severity_bypass_pure_function() -> None:
    """evaluate_severity_bypass is a pure function respecting custom critical_labels."""
    custom_labels = {"Custom_Emergency", "Explosion_Hazard"}

    # True only when all conditions align
    assert evaluate_severity_bypass("bomb", True, "Explosion_Hazard", custom_labels) is True
    # False if keyword is False
    assert evaluate_severity_bypass("bomb", False, "Explosion_Hazard", custom_labels) is False
    # False if label not in set
    assert evaluate_severity_bypass("bomb", True, "Other_Label", custom_labels) is False
    # False if empty text
    assert evaluate_severity_bypass("", True, "Explosion_Hazard", custom_labels) is False
