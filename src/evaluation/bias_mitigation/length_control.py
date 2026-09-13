"""Verbosity bias mitigation via length-controlled regression and stratification."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Try to import tiktoken if available, otherwise use heuristic
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def estimate_or_count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Estimate or count the number of tokens in a string.

    Uses tiktoken if available, otherwise uses a conservative word/character-based
    heuristic (1.3 tokens per word or 1 token per 4 chars).
    """
    if not text:
        return 0

    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:
            pass

    # Heuristic fallback: roughly 1.33 tokens per whitespace-separated word
    words = text.split()
    if not words:
        return 0
    return max(1, int(math.ceil(len(words) * 1.33)))


@dataclass
class EvalItem:
    """Represents an evaluated text item for length-bias analysis.

    Attributes:
        text: The evaluated text content.
        score: Raw score assigned by a judge or metric.
        token_count: Token count of text (calculated if None).
        residualized_score: Score after removing length correlation via regression.
        is_degenerate: True if token_count <= 0 or text is empty.
        metadata: Extra contextual metadata.
    """

    text: str
    score: float
    token_count: Optional[int] = None
    residualized_score: Optional[float] = None
    is_degenerate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count is None:
            self.token_count = estimate_or_count_tokens(self.text)
        if self.token_count <= 0:
            self.is_degenerate = True


class ResidualizedScores(list):
    """List of residualized scores with regression metadata.

    Attributes:
        slope: OLS slope beta (rate of score increase per token).
        intercept: OLS intercept alpha.
        skipped: True if regression was skipped (e.g. < 10 samples or zero variance).
        skip_reason: Description of why regression was skipped, if applicable.
        zero_token_indices: Indices of items with token_count <= 0 excluded from OLS fit.
    """

    def __init__(
        self,
        scores: Sequence[float],
        slope: Optional[float] = None,
        intercept: Optional[float] = None,
        skipped: bool = False,
        skip_reason: Optional[str] = None,
        zero_token_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__(scores)
        self.slope = slope
        self.intercept = intercept
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.zero_token_indices = zero_token_indices or []


def bucket_by_token_count(
    items: List[EvalItem],
    bucket_edges: Sequence[Union[int, float]] = (0, 50, 150, 400, float("inf")),
) -> Dict[str, List[EvalItem]]:
    """Partition evaluation items into length buckets for stratified evaluation.

    Args:
        items: List of EvalItem instances.
        bucket_edges: Monotonically increasing sequence of edge thresholds.
                      Default: [0, 50, 150, 400, inf] generating buckets:
                      "0-50", "50-150", "150-400", "400-inf".

    Returns:
        Dictionary mapping bucket label to list of EvalItems falling in that range.
    """
    # Create empty buckets
    buckets: Dict[str, List[EvalItem]] = {}
    labels: List[tuple[str, float, float]] = []

    for i in range(len(bucket_edges) - 1):
        low = float(bucket_edges[i])
        high = float(bucket_edges[i + 1])
        high_str = "inf" if math.isinf(high) else str(int(high))
        label = f"{int(low)}-{high_str}"
        buckets[label] = []
        labels.append((label, low, high))

    for item in items:
        tc = float(item.token_count if item.token_count is not None else 0)
        placed = False
        for label, low, high in labels:
            if math.isinf(high):
                if tc >= low:
                    buckets[label].append(item)
                    placed = True
                    break
            else:
                # Interval: [low, high)
                if low <= tc < high:
                    buckets[label].append(item)
                    placed = True
                    break
        if not placed and labels:
            # Fallback for any values outside edges (e.g. negative or edge boundaries)
            if tc < labels[0][1]:
                buckets[labels[0][0]].append(item)
            else:
                buckets[labels[-1][0]].append(item)

    return buckets


def apply_length_controlled_regression(
    scores: Sequence[float],
    token_counts: Sequence[int],
) -> ResidualizedScores:
    """Fit OLS regression of score on token_count and return residualized scores.

    Mitigates Verbosity Bias:
    Judges often rate longer responses higher regardless of content quality.
    By fitting:
        score_i = alpha + beta * token_count_i + epsilon_i
    the residual:
        epsilon_i = score_i - (alpha + beta * token_count_i)
    represents the length-adjusted score, with zero linear correlation to length.

    Edge case rules:
    - Exclude items with token_count <= 0 from OLS fit (degenerate). Preserve them
      at their original index with their raw score, and record their index in
      `zero_token_indices`.
    - Minimum 10 valid items required to fit OLS. If < 10, log a warning, skip
      regression, and return raw scores with skipped=True.
    - If variance of token counts in valid items is 0 (all identical), skip
      regression, log a warning, and return raw scores with skipped=True.

    Args:
        scores: Sequence of raw numerical scores.
        token_counts: Sequence of token counts matching scores by index.

    Returns:
        ResidualizedScores (list of float with slope, intercept, skipped metadata).
    """
    if len(scores) != len(token_counts):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(token_counts)} token_counts"
        )

    # 1. Separate valid items from degenerate (token_count <= 0)
    valid_indices: List[int] = []
    zero_indices: List[int] = []
    for idx, (s, tc) in enumerate(zip(scores, token_counts)):
        if tc > 0:
            valid_indices.append(idx)
        else:
            zero_indices.append(idx)

    # 2. Check minimum sample size (< 10 valid items)
    if len(valid_indices) < 10:
        logger.warning(
            "Length-controlled regression requires at least 10 valid items with token_count > 0. "
            "Found %d. Skipping regression and returning raw scores.",
            len(valid_indices),
        )
        return ResidualizedScores(
            scores=list(scores),
            skipped=True,
            skip_reason=f"Insufficient valid items: {len(valid_indices)} < 10",
            zero_token_indices=zero_indices,
        )

    # 3. Extract X and Y for valid items
    x_valid = [float(token_counts[i]) for i in valid_indices]
    y_valid = [float(scores[i]) for i in valid_indices]
    n = len(valid_indices)

    x_mean = sum(x_valid) / n
    y_mean = sum(y_valid) / n

    # Compute variance of X and covariance(X, Y)
    var_x = sum((x - x_mean) ** 2 for x in x_valid) / n
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_valid, y_valid)) / n

    # 4. Check zero variance in token counts
    if var_x < 1e-12:
        logger.warning(
            "Token counts have zero variance across all %d valid items. "
            "Cannot fit OLS slope. Skipping regression and returning raw scores.",
            n,
        )
        return ResidualizedScores(
            scores=list(scores),
            skipped=True,
            skip_reason="Zero variance in token counts",
            zero_token_indices=zero_indices,
        )

    # 5. Compute OLS slope (beta) and intercept (alpha)
    beta = cov_xy / var_x
    alpha = y_mean - (beta * x_mean)

    logger.info(
        "Fitted length regression: alpha=%.4f, beta=%.6f on %d items. Length correlation adjusted.",
        alpha,
        beta,
        n,
    )

    # 6. Compute residuals
    # For valid items: residual = raw_score - (alpha + beta * length)
    # For degenerate items (token_count <= 0): preserve raw score
    res_scores = list(scores)
    for i in valid_indices:
        x_i = token_counts[i]
        predicted = alpha + (beta * x_i)
        residual = scores[i] - predicted
        res_scores[i] = residual

    return ResidualizedScores(
        scores=res_scores,
        slope=beta,
        intercept=alpha,
        skipped=False,
        zero_token_indices=zero_indices,
    )


def residualize_eval_items(items: List[EvalItem]) -> List[EvalItem]:
    """Convenience helper to compute and assign residualized scores to a list of EvalItems."""
    scores = [it.score for it in items]
    token_counts = [it.token_count if it.token_count is not None else 0 for it in items]
    residuals = apply_length_controlled_regression(scores, token_counts)

    for it, res in zip(items, residuals):
        it.residualized_score = res

    return items
