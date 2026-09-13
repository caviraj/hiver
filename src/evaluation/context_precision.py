"""Context Precision metric computation according to the RAGAS formulation."""

import logging
from typing import Any, List
from src.evaluation.relevance_judge import judge_relevance, RelevanceJudgeError

logger = logging.getLogger(__name__)


def compute_context_precision(
    reference_answer: str,
    retrieved_contexts: List[str],
    llm_client: Any,
) -> float:
    """Compute Context Precision (Precision@k weighted average) for retrieved contexts.

    Formula:
        For each retrieved context chunk k (rank k + 1):
            precision_at_k = (count of relevant items in contexts[0..k]) / (k + 1)
        Context Precision = sum(precision_at_k * relevance_indicator_k for all k) / total_relevant_items_found

    Edge cases:
        - If retrieved_contexts is empty: returns 0.0 (nothing retrieved = 0.0 score).
        - If total_relevant_items_found == 0: returns 0.0 (zero relevant retrieved = zero score).
        - If a relevance judgment fails after retry: that specific chunk is excluded from
          the evaluated ranks and counts, logging the failed index.

    Args:
        reference_answer: Ground truth reference answer.
        retrieved_contexts: Ordered list of retrieved context chunks.
        llm_client: LLM client wrapper or callable used for judging relevance.

    Returns:
        float: Context Precision score in [0.0, 1.0].
    """
    if not retrieved_contexts or not reference_answer.strip():
        return 0.0

    # Evaluate relevance indicators for each retrieved context chunk
    relevance_indicators: List[bool] = []
    for idx, ctx in enumerate(retrieved_contexts):
        try:
            is_rel = judge_relevance(
                query_or_reference=reference_answer,
                context_chunk=ctx,
                llm_client=llm_client,
            )
            relevance_indicators.append(is_rel)
        except RelevanceJudgeError as e:
            logger.warning(
                "Excluding context at index %d from Context Precision evaluation due to judge error: %s",
                idx,
                e,
            )

    total_relevant = sum(1 for is_rel in relevance_indicators if is_rel)
    if total_relevant == 0:
        # Zero relevant retrieved = zero score (not a division error)
        return 0.0

    running_relevant_count = 0
    weighted_precision_sum = 0.0

    for rank_idx, is_rel in enumerate(relevance_indicators):
        if is_rel:
            running_relevant_count += 1
            precision_at_k = running_relevant_count / (rank_idx + 1)
            weighted_precision_sum += precision_at_k

    context_precision = weighted_precision_sum / total_relevant
    return float(context_precision)
