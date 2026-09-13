"""Pattern-based severity rules and dual consensus evaluation.

Enforces deterministic legal/hazard pattern checks alongside semantic LLM label
consensus to flag high-severity customer issues (e.g. Outrage_Escalation).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

DEFAULT_SEVERITY_KEYWORDS: list[str] = [
    "lawyer",
    "attorney",
    "sue",
    "lawsuit",
    "legal action",
    "fire hazard",
    "caught fire",
    "battery explosion",
    "exploded",
    "injured",
    "burn",
    "hospital",
]

TARGET_SEVERE_LABEL: str = "Outrage_Escalation"


def build_keyword_regex(keywords: list[str]) -> re.Pattern[str]:
    """Compile a word-boundary regex for keywords, prioritizing longer phrases."""
    sorted_keywords = sorted([k.strip() for k in keywords if k.strip()], key=len, reverse=True)
    if not sorted_keywords:
        return re.compile(r"(?=a)b")  # Matches nothing (never matches any string)
    pattern = r"\b(" + "|".join(re.escape(k) for k in sorted_keywords) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def flag_severity_candidates(
    text: str,
    custom_keywords: list[str] | None = None,
) -> bool:
    """Check if the given text contains any legal or hazard severity keywords.

    Args:
        text: Customer message or query text.
        custom_keywords: Optional list of severity keywords to override defaults.

    Returns:
        True if one or more severity keywords match, False otherwise.
    """
    if not text:
        return False
    keywords = custom_keywords if custom_keywords is not None else DEFAULT_SEVERITY_KEYWORDS
    regex = build_keyword_regex(keywords)
    return bool(regex.search(text))


def find_matching_keywords(
    text: str,
    custom_keywords: list[str] | None = None,
) -> list[str]:
    """Find all unique severity keywords present in the text.

    Args:
        text: Customer message or query text.
        custom_keywords: Optional list of severity keywords to override defaults.

    Returns:
        List of matching keywords (lowercase).
    """
    if not text:
        return []
    keywords = custom_keywords if custom_keywords is not None else DEFAULT_SEVERITY_KEYWORDS
    regex = build_keyword_regex(keywords)
    matches = regex.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            result.append(m_lower)
    return result


def evaluate_severity_consensus(
    candidate_flag: bool,
    llm_label: str,
    target_severe_label: str = TARGET_SEVERE_LABEL,
) -> tuple[bool, bool]:
    """Evaluate dual consensus between pattern-based severity and LLM classification.

    Both pattern match AND LLM semantic agreement are required to confirm high-severity.
    Cases where pattern matches but LLM disagrees are flagged as review disagreements.

    Args:
        candidate_flag: Whether pattern-based keyword rules matched.
        llm_label: Predicted or assigned label from the LLM.
        target_severe_label: The severe label name (defaults to Outrage_Escalation).

    Returns:
        tuple of (is_severe, is_disagreement):
            - is_severe: True if BOTH candidate_flag and llm_label matches target.
            - is_disagreement: True if candidate_flag is True but llm_label differs.
    """
    is_severe = candidate_flag and (llm_label == target_severe_label)
    is_disagreement = candidate_flag and (llm_label != target_severe_label)
    return is_severe, is_disagreement


class SeverityDisagreement(BaseModel):
    """Log record for severity rule and LLM consensus disagreements."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="Customer message text")
    matched_keywords: list[str] = Field(default_factory=list, description="Keywords matched by regex")
    llm_label: str = Field(..., description="LLM label assigned to the text/cluster")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of disagreement logging",
    )
    reason: str = Field(
        default="Keyword match without LLM Outrage_Escalation consensus",
        description="Reason for logging disagreement",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SeverityAuditor:
    """In-memory auditor and persistence manager for severity disagreements."""

    def __init__(self) -> None:
        self.disagreements: list[SeverityDisagreement] = []

    def record_disagreement(
        self,
        text: str,
        matched_keywords: list[str],
        llm_label: str,
        reason: str = "Keyword match without LLM Outrage_Escalation consensus",
    ) -> SeverityDisagreement:
        """Record and log a disagreement between keyword rule and LLM label."""
        record = SeverityDisagreement(
            text=text,
            matched_keywords=matched_keywords,
            llm_label=llm_label,
            reason=reason,
        )
        self.disagreements.append(record)
        logger.warning(
            "Severity consensus disagreement: text='%s...' matched=%s but LLM label='%s'",
            text[:50],
            matched_keywords,
            llm_label,
        )
        return record

    def get_disagreements(self) -> list[SeverityDisagreement]:
        """Return all recorded disagreements."""
        return list(self.disagreements)

    def export_disagreements(self, output_path: str | Path) -> None:
        """Export disagreements to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [d.to_dict() for d in self.disagreements]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported %d severity disagreements to %s", len(data), path)
