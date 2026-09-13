"""Position bias mitigation via swap-consistency protocol."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class SwapConsistencyResult:
    """Result of an A/B swap consistency evaluation.

    Attributes:
        winner: The winning candidate ("a", "b", or "tie") if consistent or discordant tie.
                None if insufficient data due to judge failures.
        consistent: Whether both forward and reversed comparisons agreed after candidate translation.
        forward_verdict: Raw verdict from the forward judge call ("position_1", "position_2", "tie").
        reversed_verdict: Raw verdict from the reversed judge call ("position_1", "position_2", "tie").
        insufficient_data: True if either judge call failed after retry, preventing unbiased evaluation.
        error_message: Error description if a failure occurred.
    """

    winner: Optional[Literal["a", "b", "tie"]]
    consistent: bool
    forward_verdict: Optional[str]
    reversed_verdict: Optional[str]
    insufficient_data: bool = False
    error_message: Optional[str] = None


def _call_judge_with_retry(
    judge_fn: Callable[[str, str], Any],
    pos_1: str,
    pos_2: str,
    direction_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """Call judge_fn with one retry on failure.

    Returns:
        (raw_verdict, error_message)
    """
    for attempt in range(2):
        try:
            raw_verdict = judge_fn(pos_1, pos_2)
            # Normalize verdict to lowercase string
            normalized = str(raw_verdict).strip().lower()
            if normalized not in ("position_1", "position_2", "tie"):
                raise ValueError(
                    f"Unexpected verdict from judge_fn in {direction_name} call: {raw_verdict}. "
                    "Expected 'position_1', 'position_2', or 'tie'."
                )
            return normalized, None
        except Exception as err:
            logger.warning(
                "Judge call failed in %s direction (attempt %d/2): %s",
                direction_name,
                attempt + 1,
                err,
            )
            if attempt == 1:
                return None, f"{direction_name} call failed after retry: {err}"
    return None, f"{direction_name} call failed after retry"


def run_ab_swap_consistency(
    candidate_a: str,
    candidate_b: str,
    judge_fn: Callable[[str, str], str],
) -> SwapConsistencyResult:
    """Run an A/B swap-consistency evaluation between two candidates.

    Position Bias mitigation:
    LLMs often prefer whatever candidate is shown in position 1. By evaluating
    both (A, B) and (B, A) and mapping position verdicts back to candidate identities,
    we can detect position bias as discordance and eliminate it.

    Args:
        candidate_a: Text of candidate A.
        candidate_b: Text of candidate B.
        judge_fn: Pairwise judge callable taking (position_1_text, position_2_text)
                  and returning "position_1", "position_2", or "tie".

    Returns:
        SwapConsistencyResult with winner, consistency flag, and raw verdicts.
    """
    # 1. Forward Call: Position 1 = Candidate A, Position 2 = Candidate B
    forward_raw, forward_err = _call_judge_with_retry(
        judge_fn, candidate_a, candidate_b, "forward"
    )
    if forward_err is not None or forward_raw is None:
        logger.error(
            "Forward judge call failed after retry. Marking as insufficient_data. Error: %s",
            forward_err,
        )
        return SwapConsistencyResult(
            winner=None,
            consistent=False,
            forward_verdict=None,
            reversed_verdict=None,
            insufficient_data=True,
            error_message=forward_err,
        )

    # 2. Reversed Call: Position 1 = Candidate B, Position 2 = Candidate A
    reversed_raw, reversed_err = _call_judge_with_retry(
        judge_fn, candidate_b, candidate_a, "reversed"
    )
    if reversed_err is not None or reversed_raw is None:
        logger.error(
            "Reversed judge call failed after retry. Marking as insufficient_data to avoid "
            "reintroducing position bias. Error: %s",
            reversed_err,
        )
        return SwapConsistencyResult(
            winner=None,
            consistent=False,
            forward_verdict=forward_raw,
            reversed_verdict=None,
            insufficient_data=True,
            error_message=reversed_err,
        )

    # 3. Translate position verdicts back to candidate identities ("a", "b", "tie")
    # Forward: pos_1 = A, pos_2 = B
    forward_translated = (
        "a" if forward_raw == "position_1"
        else "b" if forward_raw == "position_2"
        else "tie"
    )

    # Reversed: pos_1 = B, pos_2 = A
    reversed_translated = (
        "b" if reversed_raw == "position_1"
        else "a" if reversed_raw == "position_2"
        else "tie"
    )

    # 4. Check for agreement
    if forward_translated == reversed_translated:
        # Both calls agree on candidate A, candidate B, or genuine tie
        return SwapConsistencyResult(
            winner=forward_translated,  # type: ignore[arg-type]
            consistent=True,
            forward_verdict=forward_raw,
            reversed_verdict=reversed_raw,
            insufficient_data=False,
        )
    else:
        # Discordant verdicts (e.g. forward favored pos 1 (A), reversed favored pos 1 (B))
        # Per PRD: Discordant verdicts are discarded or recorded as ties.
        logger.info(
            "Discordant verdicts detected: forward translated='%s', reversed translated='%s'. "
            "Recording as tie with consistent=False.",
            forward_translated,
            reversed_translated,
        )
        return SwapConsistencyResult(
            winner="tie",
            consistent=False,
            forward_verdict=forward_raw,
            reversed_verdict=reversed_raw,
            insufficient_data=False,
        )
