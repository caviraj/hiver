"""Unit tests for src.models.intent.severity_rules.

Tests pattern-based severity rules, keyword matching, dual consensus evaluation,
and disagreement auditing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.intent.severity_rules import (
    DEFAULT_SEVERITY_KEYWORDS,
    TARGET_SEVERE_LABEL,
    SeverityAuditor,
    SeverityDisagreement,
    build_keyword_regex,
    evaluate_severity_consensus,
    find_matching_keywords,
    flag_severity_candidates,
)


def test_build_keyword_regex_empty() -> None:
    """Empty keyword list compiles to a pattern matching nothing."""
    pattern = build_keyword_regex([])
    assert pattern.search("lawyer fire hazard sue") is None
    assert pattern.search("") is None


def test_build_keyword_regex_phrase_priority() -> None:
    """Multi-word phrases are prioritized over shorter substring words."""
    keywords = ["fire", "fire hazard", "battery", "battery explosion"]
    pattern = build_keyword_regex(keywords)
    match = pattern.search("Warning: this is a fire hazard!")
    assert match is not None
    assert match.group(1).lower() == "fire hazard"


def test_flag_severity_candidates_defaults() -> None:
    """Test flagging using default keywords."""
    assert flag_severity_candidates("I will contact my lawyer immediately") is True
    assert flag_severity_candidates("The device is a dangerous fire hazard") is True
    assert flag_severity_candidates("The battery exploded in my pocket!") is True
    assert flag_severity_candidates("I was injured and had to go to the hospital") is True
    assert flag_severity_candidates("I am going to sue your company") is True

    # Benign messages
    assert flag_severity_candidates("How do I check my account balance?") is False
    assert flag_severity_candidates("Please help me transfer money to my savings account") is False
    assert flag_severity_candidates("") is False


def test_flag_severity_candidates_word_boundaries_and_case() -> None:
    """Verify case-insensitivity and word boundary preservation."""
    assert flag_severity_candidates("LAWSUIT PENDING") is True
    assert flag_severity_candidates("He had severe Burn wounds") is True

    # Substring matches should not trigger if boundaries don't match
    # 'burn' in 'burnish' or 'burnishing'
    assert flag_severity_candidates("We are burnishing the metal casing") is False
    # 'attorney' in 'attorneyship'
    assert flag_severity_candidates("The attorneyship was brief") is False


def test_flag_severity_candidates_custom_keywords() -> None:
    """Test overriding default keywords with custom keyword lists."""
    custom = ["critical vulnerability", "data breach", "ransomware"]
    assert flag_severity_candidates("We detected ransomware on the server", custom_keywords=custom) is True
    assert flag_severity_candidates("I will call a lawyer", custom_keywords=custom) is False


def test_find_matching_keywords() -> None:
    """Test extraction and deduplication of matched keywords in order."""
    text = "My attorney advised me that this fire hazard is grounds to sue. Yes, we will sue!"
    matches = find_matching_keywords(text)
    assert matches == ["attorney", "fire hazard", "sue"]

    assert find_matching_keywords("") == []
    assert find_matching_keywords("All good here, thank you!") == []


def test_evaluate_severity_consensus_agreement_severe() -> None:
    """Both keyword rule and LLM agree on Outrage_Escalation -> is_severe=True, is_disagreement=False."""
    is_severe, is_disagreement = evaluate_severity_consensus(
        candidate_flag=True,
        llm_label="Outrage_Escalation",
    )
    assert is_severe is True
    assert is_disagreement is False


def test_evaluate_severity_consensus_disagreement_keyword_only() -> None:
    """Keyword matches (e.g. 'update killed my battery') but LLM labels non-severe -> disagreement."""
    is_severe, is_disagreement = evaluate_severity_consensus(
        candidate_flag=True,
        llm_label="Device_Performance_Degradation",
    )
    assert is_severe is False
    assert is_disagreement is True


def test_evaluate_severity_consensus_agreement_benign() -> None:
    """Neither keyword matches nor LLM predicts severe -> agreement benign."""
    is_severe, is_disagreement = evaluate_severity_consensus(
        candidate_flag=False,
        llm_label="Account_Inquiry",
    )
    assert is_severe is False
    assert is_disagreement is False


def test_evaluate_severity_consensus_llm_only() -> None:
    """LLM predicts severe but no keywords matched -> is_severe=False, is_disagreement=False."""
    is_severe, is_disagreement = evaluate_severity_consensus(
        candidate_flag=False,
        llm_label="Outrage_Escalation",
    )
    assert is_severe is False
    assert is_disagreement is False


def test_evaluate_severity_consensus_custom_target() -> None:
    """Support custom target severe label."""
    is_severe, is_disagreement = evaluate_severity_consensus(
        candidate_flag=True,
        llm_label="Safety_Hazard",
        target_severe_label="Safety_Hazard",
    )
    assert is_severe is True
    assert is_disagreement is False


def test_severity_disagreement_frozen() -> None:
    """SeverityDisagreement model should be immutable."""
    disagreement = SeverityDisagreement(
        text="Sample text",
        matched_keywords=["lawyer"],
        llm_label="General_Inquiry",
    )
    with pytest.raises(Exception):
        disagreement.text = "New text"  # type: ignore[misc]

    dump = disagreement.to_dict()
    assert dump["text"] == "Sample text"
    assert dump["matched_keywords"] == ["lawyer"]
    assert dump["llm_label"] == "General_Inquiry"
    assert "timestamp" in dump


def test_severity_auditor_workflow(tmp_path: Path) -> None:
    """Verify recording, retrieval, and JSON export in SeverityAuditor."""
    auditor = SeverityAuditor()
    assert auditor.get_disagreements() == []

    rec1 = auditor.record_disagreement(
        text="The software update killed my battery, I want a lawyer!",
        matched_keywords=["lawyer"],
        llm_label="Device_Performance_Degradation",
        reason="Update issue misclassified as legal threat",
    )
    assert rec1.matched_keywords == ["lawyer"]
    assert len(auditor.get_disagreements()) == 1

    rec2 = auditor.record_disagreement(
        text="My battery got very hot, almost a fire hazard",
        matched_keywords=["fire hazard"],
        llm_label="Hardware_Defect",
    )
    assert len(auditor.get_disagreements()) == 2

    # Export to JSON
    export_file = tmp_path / "disagreements.json"
    auditor.export_disagreements(export_file)
    assert export_file.exists()

    with open(export_file, encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded) == 2
    assert loaded[0]["matched_keywords"] == ["lawyer"]
    assert loaded[1]["matched_keywords"] == ["fire hazard"]
