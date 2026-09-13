"""Judge calibration and deployment gating using Quadratic Weighted Kappa.

Evaluates agreement between continuous/discrete LLM judge ratings (e.g. G-Eval)
and validated human ground truth labels from the golden dataset.

Enforces a hard threshold of kappa > 0.60 for deployment qualification.
Quadratic weighting penalizes extreme discrepancies (human=1 vs LLM=5) far
more severely than adjacent discrepancies (human=3 vs LLM=4), matching the
asymmetric business risk of catastrophic judge hallucination.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

# Hard deployment qualification threshold per PRD specification
JUDGE_DEPLOYMENT_KAPPA_THRESHOLD: float = 0.60


class CalibrationResult(BaseModel):
    """Structured result of LLM judge calibration against human ground truth."""

    kappa: float = Field(..., description="Quadratic Weighted Cohen's Kappa score")
    passed: bool = Field(..., description="True if kappa strictly exceeds the deployment threshold")
    n_examples: int = Field(..., description="Number of paired examples evaluated (excluding missing labels)")
    threshold: float = Field(
        default=JUDGE_DEPLOYMENT_KAPPA_THRESHOLD,
        description="Minimum kappa required to pass the deployment gate (strictly greater than)",
    )
    notes: List[str] = Field(default_factory=list, description="Diagnostic notes or warnings from evaluation")


def discretize_scores(continuous_scores: List[float]) -> List[int]:
    """Discretize continuous scores to integer ordinal ratings [1, 5] using round-half-up.

    Continuous scores from logit-weighted G-Eval must be mapped to discrete categories
    for Cohen's Kappa calculation.

    NOTE: Standard Python `round()` uses round-half-to-even (banker's rounding), which
    rounds 2.5 to 2 and 3.5 to 4. To ensure deterministic, intuitive score boundaries,
    we explicitly use the round-half-up rule (`floor(x + 0.5)`), so 2.5 -> 3, 3.5 -> 4.
    Results are clamped to the valid [1, 5] rubric scale.

    Args:
        continuous_scores: List of floating-point scores.

    Returns:
        List of integer ratings clamped between 1 and 5.
    """
    discretized: List[int] = []
    for score in continuous_scores:
        rounded = int(math.floor(score + 0.5))
        clamped = max(1, min(5, rounded))
        discretized.append(clamped)
    return discretized


def compute_quadratic_weighted_kappa(
    human_scores: List[int],
    llm_scores: List[int],
) -> float:
    """Compute Quadratic Weighted Cohen's Kappa between human and LLM ratings.

    Quadratic weighting penalizes large disagreements quadratically, reflecting
    the asymmetric business risk where high-discordance errors are catastrophic.

    Guards against zero-variance degenerate conditions where chance agreement math
    divides by zero, producing NaN in sklearn.

    Args:
        human_scores: Validated human ground truth ratings [1-5].
        llm_scores: Discretized LLM judge ratings [1-5].

    Returns:
        Quadratic weighted kappa coefficient (-1.0 to 1.0).

    Raises:
        ValueError: If input lengths differ, if lists are empty, or if score variance
            is zero across the entire dataset (causing division by zero / NaN).
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError(
            f"Length mismatch: human_scores has {len(human_scores)} items, "
            f"but llm_scores has {len(llm_scores)} items."
        )

    if not human_scores:
        raise ValueError("Cannot compute Kappa on empty score lists.")

    # Check for zero-variance degenerate case (e.g. all 3s across both lists)
    combined_unique = set(human_scores) | set(llm_scores)
    if len(combined_unique) <= 1:
        raise ValueError(
            "insufficient score variance for Kappa calculation: all scores fall into a single category."
        )

    # Compute Cohen's Kappa with quadratic weights
    kappa = float(
        cohen_kappa_score(
            y1=human_scores,
            y2=llm_scores,
            weights="quadratic",
            labels=[1, 2, 3, 4, 5],
        )
    )

    if math.isnan(kappa):
        raise ValueError(
            "insufficient score variance for Kappa calculation: chance agreement resulted in NaN."
        )

    return kappa


def evaluate_judge_calibration(
    human_scores: List[Optional[int]],
    llm_scores: List[int],
    threshold: float = JUDGE_DEPLOYMENT_KAPPA_THRESHOLD,
) -> CalibrationResult:
    """Evaluate an LLM judge's agreement with human labels against the deployment gate.

    Filters out missing human labels (`None`) without imputation. Requires kappa
    to be STRICTLY GREATER than `threshold` (0.60) to pass; exactly 0.60 fails.

    Args:
        human_scores: List of human ground truth scores, potentially containing None for unlabeled items.
        llm_scores: List of LLM judge scores.
        threshold: Minimum kappa required to qualify for deployment (default: 0.60).

    Returns:
        CalibrationResult object detailing kappa score and pass/fail decision.

    Raises:
        ValueError: If length mismatch occurs, or if valid paired examples are fewer than 2.
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError(
            f"Length mismatch: human_scores ({len(human_scores)}) != llm_scores ({len(llm_scores)})"
        )

    paired_human: List[int] = []
    paired_llm: List[int] = []
    excluded_count = 0

    for h_score, l_score in zip(human_scores, llm_scores):
        if h_score is None:
            excluded_count += 1
            continue
        paired_human.append(int(h_score))
        paired_llm.append(int(l_score))

    notes: List[str] = []
    if excluded_count > 0:
        msg = f"Excluded {excluded_count} unlabeled items from calibration calculation."
        logger.info(msg)
        notes.append(msg)

    n_paired = len(paired_human)
    if n_paired < 2:
        raise ValueError(
            f"Insufficient labeled examples for calibration: found only {n_paired} valid pairs."
        )

    kappa = compute_quadratic_weighted_kappa(paired_human, paired_llm)

    # Hard deployment gate: MUST strictly exceed threshold (> 0.60)
    # Exactly 0.60 fails per specification
    passed = kappa > threshold

    if not passed:
        failure_msg = (
            f"JUDGE CALIBRATION FAILED: Quadratic Weighted Kappa {kappa:.4f} <= threshold {threshold:.2f} "
            f"across {n_paired} examples. The LLM judge DOES NOT meet the reliability bar for production."
        )
        logger.error(failure_msg)
        notes.append(failure_msg)
    else:
        success_msg = (
            f"Judge calibration passed: Quadratic Weighted Kappa {kappa:.4f} > threshold {threshold:.2f} "
            f"across {n_paired} examples."
        )
        logger.info(success_msg)
        notes.append(success_msg)

    return CalibrationResult(
        kappa=kappa,
        passed=passed,
        n_examples=n_paired,
        threshold=threshold,
        notes=notes,
    )
