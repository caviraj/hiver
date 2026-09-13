"""Reciprocal Rank Fusion (RRF) implementation (M3.P3.3.F1).

Fuses ranked candidate lists from disparate retrieval strategies (e.g. BM25 sparse
lexical retrieval and Dense semantic embedding retrieval) into a unified ranking.

BM25 scores (unbounded positive floats) and dense cosine similarity scores
(bounded [-1, 1]) are mathematically incompatible for direct score normalization or
linear combination without extensive domain-specific calibration. RRF circumvents
this by discarding raw scores and fusing purely on 1-indexed RANK POSITION.

Furthermore, this implementation supports ASYMMETRIC k design:
Each retriever r gets an independent constant k_r. While standard RRF uses a global
default (e.g., k=60), the sparse track's k can be independently tuned downward
(e.g., k_sparse=10, k_dense=60). This sharply boosts rank-1 sparse matches,
ensuring exact alphanumeric matches (such as product codes, order numbers, or
device models like 'SM-T280') dominate fusion over dense semantic approximations
per the PRD design specification.
"""

from collections import defaultdict
import logging
from typing import Dict, List

from src.retrieval.schema import RetrievalResult

logger = logging.getLogger(__name__)

# Standard balanced default RRF smoothing constant (Cormack et al., 2009)
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: Dict[str, List[RetrievalResult]],
    k_per_retriever: Dict[str, int],
) -> List[RetrievalResult]:
    """Fuse multiple ranked RetrievalResult lists into a single ranking using RRF.

    Parameters
    ----------
    result_lists : Dict[str, List[RetrievalResult]]
        Dictionary mapping retriever identifier (e.g. 'sparse', 'dense') to its
        list of rank-ordered RetrievalResult candidates. Each list must already
        be ordered by that retriever's relevance score descending.
    k_per_retriever : Dict[str, int]
        Dictionary mapping retriever identifier to its specific smoothing constant k.
        Each retriever gets its own k:
        - Symmetric baseline: {"sparse": 60, "dense": 60}
        - Asymmetric tuning: k_per_retriever["sparse"] can be tuned downward
          (e.g., to 10) to heavily weight exact product-code / alphanumeric
          matches at rank 1, per PRD design decisions.

    Returns
    -------
    List[RetrievalResult]
        Unified list of RetrievalResult objects ordered by fused RRF score descending,
        with new rank values 1..N assigned. Ties in fused score are broken
        deterministically by doc_id (alphabetical ascending).

    Raises
    ------
    KeyError
        If a retriever name present in `result_lists` with non-empty results is
        missing from `k_per_retriever`.
    ValueError
        If any smoothing parameter k in `k_per_retriever` is negative.
    """
    if not result_lists:
        return []

    # Validate k parameters
    for name, k_val in k_per_retriever.items():
        if k_val < 0:
            raise ValueError(f"k parameter for retriever '{name}' must be non-negative, got {k_val}")

    # Accumulate fused scores: doc_id -> fused_score
    doc_scores: Dict[str, float] = defaultdict(float)

    for retriever_name, results in result_lists.items():
        if not results:
            continue

        if retriever_name not in k_per_retriever:
            raise KeyError(
                f"Retriever '{retriever_name}' is present in result_lists but missing from k_per_retriever. "
                f"Provided k_per_retriever keys: {list(k_per_retriever.keys())}"
            )

        k_r = k_per_retriever[retriever_name]

        for item in results:
            # 1-indexed rank within this retriever's candidate list
            # A retriever contributes ZERO if document is absent from its result list
            rank = item.rank
            doc_scores[item.doc_id] += 1.0 / (k_r + rank)

    if not doc_scores:
        return []

    # Sort documents by:
    # 1. Fused score descending (-score)
    # 2. doc_id ascending (alphabetical tie-breaking for deterministic reproducibility)
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    # Reassign new 1-indexed ranks based on fused order
    fused_results: List[RetrievalResult] = [
        RetrievalResult(
            doc_id=doc_id,
            score=round(score, 8),
            rank=new_rank,
        )
        for new_rank, (doc_id, score) in enumerate(sorted_docs, start=1)
    ]

    logger.debug(
        "Fused %d unique documents from %d retrievers",
        len(fused_results),
        len(result_lists),
    )
    return fused_results
