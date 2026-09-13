"""Hybrid Retriever orchestrator for BM25 and Dense Semantic Retrieval (M3.P3.3.F1).

Coordinates sparse lexical retrieval (BM25Index) and dense semantic retrieval (DenseIndex)
over an identical document corpus, applying Reciprocal Rank Fusion (RRF) with
asymmetric k support to produce a unified, highly grounded set of candidates.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from src.retrieval.bm25_index import BM25Index, query_index as query_bm25_index
from src.retrieval.dense_index import DenseIndex, query_dense_index
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.schema import RetrievalResult

logger = logging.getLogger(__name__)

# Default asymmetric/symmetric fusion constants
DEFAULT_K_PER_RETRIEVER: Dict[str, int] = {
    "sparse": 60,
    "dense": 60,
}


class CorpusMismatchError(ValueError):
    """Raised when the document ID sets of sparse and dense indices diverge."""
    pass


class HybridRetriever:
    """Orchestrates sparse lexical and dense semantic retrieval tracks with RRF fusion.

    Wraps both a BM25Index and a DenseIndex constructed from the exact same corpus.
    On initialization, verifies that both indices share identical doc_ids, preventing
    silent partial fusion and corrupt ranking.

    Parameters
    ----------
    bm25_index : BM25Index
        Fitted BM25 sparse index.
    dense_index : DenseIndex
        Fitted dense semantic index.
    encoder : Optional[Any], default=None
        Optional sentence transformer encoder instance used to compute query embeddings.
        Can alternatively be provided during individual `retrieve()` calls.

    Raises
    ------
    CorpusMismatchError
        If the doc_ids present in bm25_index and dense_index are not strictly identical.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        encoder: Optional[Any] = None,
    ) -> None:
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.encoder = encoder

        # Verify corpus consistency across both tracks
        self._verify_corpus_consistency()

    def _verify_corpus_consistency(self) -> None:
        """Verify both underlying indices index the exact same document IDs.

        Raises
        ------
        CorpusMismatchError
            If there is any discrepancy between sparse and dense document ID sets.
        """
        sparse_doc_ids: Set[str] = set(self.bm25_index.doc_ids)
        dense_doc_ids: Set[str] = set(self.dense_index.doc_ids)

        if sparse_doc_ids != dense_doc_ids:
            missing_in_dense = sparse_doc_ids - dense_doc_ids
            missing_in_sparse = dense_doc_ids - sparse_doc_ids
            err_msg = (
                f"Corpus inconsistency detected between BM25 sparse index and Dense index. "
                f"Sparse index has {len(sparse_doc_ids)} docs, Dense index has {len(dense_doc_ids)} docs. "
                f"Missing in dense: {len(missing_in_dense)} docs ({sorted(list(missing_in_dense))[:5]}...). "
                f"Missing in sparse: {len(missing_in_sparse)} docs ({sorted(list(missing_in_sparse))[:5]}...)."
            )
            logger.error(err_msg)
            raise CorpusMismatchError(err_msg)

        logger.info(
            "Corpus consistency verified: both sparse and dense indices contain %d identical documents",
            len(sparse_doc_ids),
        )

    def retrieve(
        self,
        query_text: str,
        top_k_per_track: int = 20,
        final_top_k: int = 10,
        k_per_retriever: Optional[Dict[str, int]] = None,
        encoder: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Execute hybrid retrieval across sparse and dense indices, fusing outputs via RRF.

        Parameters
        ----------
        query_text : str
            Incoming customer query string.
        top_k_per_track : int, default=20
            Number of top candidates to retrieve from each individual track before fusion.
            Typically wider than final_top_k to capture candidates whose rank improves post-fusion.
        final_top_k : int, default=10
            Maximum number of fused candidates to return. If fewer unique candidates are
            found across both tracks, all available fused results are returned without padding.
        k_per_retriever : Optional[Dict[str, int]], default=None
            Per-retriever smoothing constants for RRF. Defaults to {"sparse": 60, "dense": 60}.
            Can be tuned (e.g. {"sparse": 10, "dense": 60}) to prioritize exact sparse matches.
        encoder : Optional[Any], default=None
            Sentence transformer encoder. If not provided, uses `self.encoder`.

        Returns
        -------
        List[RetrievalResult]
            Rank-ordered list of top fused RetrievalResult objects, with ranks 1..final_top_k.

        Raises
        ------
        ValueError
            If no encoder is available for dense query execution.
        """
        if not query_text or not query_text.strip():
            logger.debug("Received empty query text in HybridRetriever.retrieve; returning empty list.")
            return []

        active_encoder = encoder if encoder is not None else self.encoder
        if active_encoder is None:
            raise ValueError(
                "An encoder instance must be provided either at HybridRetriever initialization "
                "or passed explicitly to retrieve()."
            )

        active_k = k_per_retriever if k_per_retriever is not None else DEFAULT_K_PER_RETRIEVER

        # 1. Query Sparse Track (BM25)
        sparse_results = query_bm25_index(
            self.bm25_index,
            query_text=query_text,
            top_k=top_k_per_track,
        )

        # 2. Query Dense Track (Semantic Cosine)
        dense_results = query_dense_index(
            self.dense_index,
            query_text=query_text,
            encoder=active_encoder,
            top_k=top_k_per_track,
        )

        # 3. Fuse Results via Asymmetric RRF
        result_lists = {
            "sparse": sparse_results,
            "dense": dense_results,
        }

        fused_results = reciprocal_rank_fusion(
            result_lists=result_lists,
            k_per_retriever=active_k,
        )

        # 4. Return top final_top_k (no padding if fewer candidates available)
        return fused_results[:final_top_k]
