"""Diagnostic failure-mode candidate tagger for RAG evaluation.

Implements heuristic diagnostic candidate tagging across five documented failure categories:
1. CONVERSATIONAL_TRUNCATION: Short/fragmented thread context causing retrieval drop.
2. SARCASM_SENTIMENT_INVERSION: Tone/empathy failure due to inverted sentiment/sarcasm.
3. MULTI_HOP_REASONING_DEFICIT: Synthesis failure across multiple conditions/entities.
4. STALE_KNOWLEDGE_INDEXING: Policy/version drift causing hallucinated terms despite good context.
5. ADVERSARIAL_EXTRACTION: Inbound prompt-injection or jailbreak extraction attempts.

CRITICAL ARCHITECTURE DISCIPLINE:
Tags produced by this module are CANDIDATE diagnostic signals intended for human triage,
NOT confirmed ground-truth classifications. An inbound message containing 'ignore previous'
(e.g., 'please ignore the previous rep') must yield an unconfirmed candidate flag,
never an automated conviction.
"""

from __future__ import annotations

from enum import Enum
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_LOG_PATH: Path = Path("data/eval/failure_modes.jsonl")


class FailureMode(str, Enum):
    """Documented failure categories for customer support evaluation triage."""

    CONVERSATIONAL_TRUNCATION = "CONVERSATIONAL_TRUNCATION"
    SARCASM_SENTIMENT_INVERSION = "SARCASM_SENTIMENT_INVERSION"
    MULTI_HOP_REASONING_DEFICIT = "MULTI_HOP_REASONING_DEFICIT"
    STALE_KNOWLEDGE_INDEXING = "STALE_KNOWLEDGE_INDEXING"
    ADVERSARIAL_EXTRACTION = "ADVERSARIAL_EXTRACTION"


class FailureModeCandidate(BaseModel):
    """Candidate diagnostic tag produced for human evaluation triage."""

    mode: FailureMode
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in the diagnostic heuristic signal",
    )
    signal_description: str = Field(
        ...,
        description="Detailed description of heuristic triggers and observed symptoms",
    )
    confirmed: bool = Field(
        default=False,
        description="Strictly False on creation; requires human confirmation",
    )


class FailureModeRecord(BaseModel):
    """Record of diagnostic failure-mode candidate tags for an evaluated item."""

    item_id: str = Field(..., description="Unique identifier of evaluated item")
    query: str = Field(..., description="Customer query or prompt text")
    candidates: List[FailureModeCandidate] = Field(
        default_factory=list,
        description="List of candidate failure modes identified",
    )
    is_unclassified: bool = Field(
        default=False,
        description="True if no heuristic signals fired, indicating an unclassified failure",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.candidates:
            self.is_unclassified = True
        else:
            self.is_unclassified = False


# -------------------------------------------------------------------------
# Heuristic Patterns & Dictionaries
# -------------------------------------------------------------------------

ADVERSARIAL_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|directives|prompts|rules)", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system|initial|hidden)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in|unrestricted)", re.I),
    re.compile(r"disregard\s+(all\s+)?(prior|previous)\s+instructions", re.I),
    re.compile(r"system\s*prompt\s*leak", re.I),
    re.compile(r"developer\s+mode\s+output", re.I),
    re.compile(r"ignore\s+previous", re.I),  # Broad pattern produces candidate for human triage
]

POLICY_KEYWORDS = {
    "refund",
    "warranty",
    "window",
    "return",
    "policy",
    "guarantee",
    "coverage",
    "receipt",
    "30-day",
    "14-day",
    "terms",
    "replacement",
}

SARCASTIC_POSITIVE_MARKERS = {
    "great job",
    "love it",
    "fantastic",
    "brilliant",
    "wonderful",
    "so helpful",
    "thanks a lot",
    "super helpful",
    "amazing job",
    "best service ever",
}

NEGATIVE_LEXICON = {
    "broken",
    "broke",
    "break",
    "useless",
    "terrible",
    "worst",
    "fails",
    "failing",
    "fail",
    "ruined",
    "unacceptable",
    "crashed",
    "crash",
    "horrible",
    "sucks",
    "hate",
    "brick",
    "bricked",
    "waste of money",
    "not working",
    "garbage",
}

CONJUNCTION_PATTERN = re.compile(
    r"\b(and|but|as well as|while|whereas|in addition to)\b", re.I
)


# -------------------------------------------------------------------------
# Candidate Tagging Engine
# -------------------------------------------------------------------------


def tag_failure_candidates(
    eval_item: Any,
    ragas_result: Optional[Any] = None,
    geval_result: Optional[Any] = None,
    retrieval_result: Optional[Any] = None,
) -> List[FailureModeCandidate]:
    """Analyze evaluation metrics and input features to produce candidate failure tags.

    Evaluates all five heuristic signals independently and returns all matching
    candidates. Does not force a single label or merge disjoint signals.

    Args:
        eval_item: Evaluation item (GoldenExample, RAGASInput, or dict) containing query,
            generated_response, and optional metadata.
        ragas_result: Optional RAGASResult object or dict with context_precision, context_recall,
            faithfulness.
        geval_result: Optional GEvalResult object, list of GEvalResult, or dict containing
            weighted_score, score, and rubric_name.
        retrieval_result: Optional raw retrieval output for additional context inspection.

    Returns:
        List of FailureModeCandidate objects. Empty list indicates no heuristic fired
        (treated downstream as 'unclassified').
    """
    candidates: List[FailureModeCandidate] = []

    # 1. Normalize Query and Context
    query = ""
    generated_response = ""
    metadata: Dict[str, Any] = {}

    if isinstance(eval_item, dict):
        query = eval_item.get("query") or eval_item.get("tweet_text", "")
        generated_response = eval_item.get("generated_response", "")
        metadata = eval_item.get("metadata", {})
    else:
        query = getattr(eval_item, "query", "") or getattr(eval_item, "tweet_text", "")
        generated_response = getattr(eval_item, "generated_response", "")
        metadata = getattr(eval_item, "metadata", {}) or {}

    query_lower = query.lower()
    response_lower = generated_response.lower()

    # Normalize RAGAS metrics
    def _get_ragas_val(attr: str) -> Optional[float]:
        if ragas_result is None:
            return None
        if isinstance(ragas_result, dict):
            return ragas_result.get(attr)
        return getattr(ragas_result, attr, None)

    context_precision = _get_ragas_val("context_precision")
    context_recall = _get_ragas_val("context_recall")
    faithfulness = _get_ragas_val("faithfulness")

    # Normalize G-Eval scores
    geval_scores: Dict[str, float] = {}
    if isinstance(geval_result, list):
        for item in geval_result:
            r_name = getattr(item, "rubric_name", None) or (
                item.get("rubric_name") if isinstance(item, dict) else None
            )
            w_score = getattr(item, "weighted_score", None) or (
                item.get("weighted_score") if isinstance(item, dict) else None
            )
            if r_name and w_score is not None:
                geval_scores[r_name] = float(w_score)
    elif isinstance(geval_result, dict):
        if "rubric_name" in geval_result and "weighted_score" in geval_result:
            geval_scores[geval_result["rubric_name"]] = float(geval_result["weighted_score"])
        elif "score" in geval_result:
            geval_scores["general"] = float(geval_result["score"])
            geval_scores["tone"] = float(geval_result["score"])
            geval_scores["empathy"] = float(geval_result["score"])
        else:
            for k, v in geval_result.items():
                if isinstance(v, (int, float)):
                    geval_scores[k] = float(v)
    elif geval_result is not None:
        r_name = getattr(geval_result, "rubric_name", "general")
        w_score = getattr(geval_result, "weighted_score", None)
        if w_score is None:
            w_score = getattr(geval_result, "score", None)
        if w_score is not None:
            geval_scores[r_name] = float(w_score)
            geval_scores["general"] = float(w_score)

    # ---------------------------------------------------------------------
    # Signal 1: Conversational Truncation
    # Low context_recall or context_precision + short/fragmented thread history
    # ---------------------------------------------------------------------
    has_low_retrieval = (context_recall is not None and context_recall < 0.5) or (
        context_precision is not None and context_precision < 0.5
    )
    thread_history = metadata.get("thread_history", []) or metadata.get("history", [])
    turn_count = metadata.get(
        "turn_count",
        metadata.get("thread_length", len(thread_history) if thread_history else 1),
    )
    is_fragment = (
        metadata.get("is_thread_fragment", False)
        or turn_count <= 1
        or metadata.get("thread_length", 1) <= 1
        or bool(re.match(r"^(it|that|still|also|why|then|yes)\b", query_lower.strip()))
    )

    if has_low_retrieval and is_fragment:
        conf = 0.75 if is_fragment and (context_recall or 0) < 0.3 else 0.60
        candidates.append(
            FailureModeCandidate(
                mode=FailureMode.CONVERSATIONAL_TRUNCATION,
                confidence=conf,
                signal_description=(
                    f"Low retrieval precision/recall (precision={context_precision}, "
                    f"recall={context_recall}) observed with fragmented or single-turn "
                    f"thread history (turns={turn_count}). Potential loss of upstream context."
                ),
            )
        )

    # ---------------------------------------------------------------------
    # Signal 2: Sarcasm / Sentiment Inversion
    # Decent faithfulness + low G-Eval tone/empathy + negative lexicon vs positive intent
    # ---------------------------------------------------------------------
    has_decent_faithfulness = faithfulness is not None and faithfulness >= 0.6
    tone_or_empathy_score = min(
        [
            geval_scores[k]
            for k in ("empathy", "brand_tone", "tone", "general")
            if k in geval_scores
        ]
        or [5.0]
    )
    has_low_tone_or_empathy = tone_or_empathy_score < 3.0

    has_positive_marker = any(p in query_lower for p in SARCASTIC_POSITIVE_MARKERS)
    has_negative_marker = any(n in query_lower for n in NEGATIVE_LEXICON)
    intent = (metadata.get("intent") or "").lower()
    is_positive_intent = intent in ("praise", "positive", "compliment")

    sentiment_mismatch = (
        (has_positive_marker and has_negative_marker)
        or metadata.get("intent_sentiment_mismatch", False)
        or (metadata.get("sentiment") == "negative" and has_positive_marker)
        or (is_positive_intent and (has_negative_marker or has_positive_marker))
    )

    if has_decent_faithfulness and has_low_tone_or_empathy and sentiment_mismatch:
        candidates.append(
            FailureModeCandidate(
                mode=FailureMode.SARCASM_SENTIMENT_INVERSION,
                confidence=0.70,
                signal_description=(
                    f"Faithfulness is solid ({faithfulness:.2f}) but tone/empathy G-Eval "
                    f"is depressed ({tone_or_empathy_score:.2f}) with sarcastic/inverted lexicon "
                    f"mismatch in query. Agent may have taken sarcasm literally."
                ),
            )
        )

    # ---------------------------------------------------------------------
    # Signal 3: Multi-Hop Reasoning Deficit
    # Query contains multiple distinct conditions/entities + low faithfulness despite
    # decent individual context_precision
    # ---------------------------------------------------------------------
    has_conjunctions = len(CONJUNCTION_PATTERN.findall(query)) >= 1
    has_multiple_entities = (
        metadata.get("num_entities", 0) >= 2
        or len(re.split(r"[,;]|\band\b|\bbut\b", query_lower)) >= 2
    )
    has_low_faithfulness = faithfulness is not None and faithfulness < 0.6
    has_decent_precision = context_precision is not None and context_precision >= 0.5

    if has_conjunctions and has_multiple_entities and has_low_faithfulness and has_decent_precision:
        candidates.append(
            FailureModeCandidate(
                mode=FailureMode.MULTI_HOP_REASONING_DEFICIT,
                confidence=0.68,
                signal_description=(
                    f"Query contains multi-clause conjunctions/entities with high context "
                    f"precision ({context_precision:.2f}) but low faithfulness ({faithfulness:.2f}). "
                    f"Indicates failure to synthesize multi-hop relationships."
                ),
            )
        )

    # ---------------------------------------------------------------------
    # Signal 4: Stale Knowledge Indexing
    # High context_precision/recall but a faithfulness failure specifically on a claim
    # containing a policy-like term (refund, warranty, window, etc.)
    # ---------------------------------------------------------------------
    has_high_retrieval = (
        (context_precision is not None and context_precision >= 0.6)
        or (context_recall is not None and context_recall >= 0.6)
    )
    has_policy_term = any(
        kw in query_lower or kw in response_lower for kw in POLICY_KEYWORDS
    )

    if has_high_retrieval and has_low_faithfulness and has_policy_term:
        candidates.append(
            FailureModeCandidate(
                mode=FailureMode.STALE_KNOWLEDGE_INDEXING,
                confidence=0.72,
                signal_description=(
                    f"High retrieval metrics (precision={context_precision}, recall={context_recall}) "
                    f"yet low faithfulness ({faithfulness}) with policy terms present "
                    f"(e.g. warranty/refund/window). Suggests outdated indexed policy knowledge."
                ),
            )
        )

    # ---------------------------------------------------------------------
    # Signal 5: Adversarial Extraction
    # Query matches prompt-injection patterns -> CANDIDATE ONLY, never auto-confirmed
    # ---------------------------------------------------------------------
    matching_patterns = [p.pattern for p in ADVERSARIAL_PATTERNS if p.search(query)]
    if matching_patterns:
        candidates.append(
            FailureModeCandidate(
                mode=FailureMode.ADVERSARIAL_EXTRACTION,
                confidence=0.65,  # Non-authoritative candidate confidence
                signal_description=(
                    f"Query matches known extraction/override pattern(s): {matching_patterns[:2]}. "
                    f"CANDIDATE ONLY — requires human confirmation; must not be treated as confirmed "
                    f"jailbreak without human verification."
                ),
            )
        )

    return candidates


# -------------------------------------------------------------------------
# Streaming Persistence Logger
# -------------------------------------------------------------------------


def log_failure_mode_record(
    record_or_item_id: Union[FailureModeRecord, str],
    query_or_log_path: Union[str, Path] = DEFAULT_FAILURE_LOG_PATH,
    candidates: Optional[List[FailureModeCandidate]] = None,
    log_path: Union[str, Path] = DEFAULT_FAILURE_LOG_PATH,
) -> None:
    """Stream a failure-mode tagging record to a JSONL log file.

    Supports both object and positional invocations:
        log_failure_mode_record(record: FailureModeRecord, log_path: Union[str, Path])
        log_failure_mode_record(eval_item_id: str, query: str, candidates: List[...], log_path: ...)

    Non-crashing: write failures or IO exceptions are logged as warnings
    and will not abort the evaluation process.
    """
    if isinstance(record_or_item_id, FailureModeRecord):
        record = record_or_item_id
        target_path = Path(query_or_log_path)
    else:
        item_id = str(record_or_item_id)
        query_text = str(query_or_log_path)
        cands = candidates or []
        record = FailureModeRecord(
            item_id=item_id,
            query=query_text,
            candidates=cands,
        )
        target_path = Path(log_path)

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "item_id": record.item_id,
            "example_id": record.item_id,
            "query": record.query,
            "status": "unclassified" if record.is_unclassified else "classified",
            "is_unclassified": record.is_unclassified,
            "candidate_count": len(record.candidates),
            "candidates": [
                {
                    "mode": c.mode.value,
                    "confidence": c.confidence,
                    "signal_description": c.signal_description,
                    "confirmed": c.confirmed,
                }
                for c in record.candidates
            ],
        }
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as exc:
        logger.warning(
            "Failed to append failure mode log to %s: %s",
            target_path,
            exc,
            exc_info=False,
        )
