"""Unit and integration tests for hybrid_retriever.py (M3.P3.3.F1).

Verifies:
1. Corpus mismatch detection on HybridRetriever initialization (CorpusMismatchError).
2. Diagnostic count of missing doc_ids in CorpusMismatchError message.
3. Successful initialization when sparse and dense doc_ids match perfectly.
4. Empty/whitespace query returns empty list without error.
5. Missing encoder raises ValueError.
6. End-to-end hybrid retrieval with RRF fusion on synthetic corpus.
7. Asymmetric k tuning in retrieve() prioritizing exact product code match.
8. final_top_k exceeding available candidate documents returns all available results.
9. top_k_per_track=1 low-pool scenario.
10. Ranks sequentially re-assigned 1..N post-fusion.
"""

from typing import List
import pytest

from src.retrieval.bm25_index import BM25Index, build_index as build_bm25_index
from src.retrieval.dense_index import DenseIndex, build_dense_index
from src.retrieval.embedding_model import DEFAULT_MODEL_NAME, load_encoder
from src.retrieval.hybrid_retriever import (
    CorpusMismatchError,
    DEFAULT_K_PER_RETRIEVER,
    HybridRetriever,
)
from src.retrieval.schema import RetrievalDocument, RetrievalResult


@pytest.fixture(scope="module")
def shared_encoder():
    """Load the default encoder once for test module efficiency."""
    return load_encoder(DEFAULT_MODEL_NAME)


@pytest.fixture
def synthetic_corpus() -> List[RetrievalDocument]:
    """Small synthetic corpus containing exact product codes and semantic concepts."""
    return [
        RetrievalDocument(
            doc_id="DOC-SM-T280",
            resolution_text="Reset firmware for Samsung tablet model SM-T280.",
            query_text="My tablet model SM-T280 keeps rebooting into recovery mode.",
        ),
        RetrievalDocument(
            doc_id="DOC-BILLING",
            resolution_text="Navigate to Billing -> Invoices to download receipt.",
            query_text="Where can I find my monthly receipt and invoice statement?",
        ),
        RetrievalDocument(
            doc_id="DOC-PASSWORD",
            resolution_text="Use Account Recovery page to reset your master credentials.",
            query_text="I forgot my login password and cannot access my account.",
        ),
        RetrievalDocument(
            doc_id="DOC-HARDWARE",
            resolution_text="Inspect battery connector and clean the charging port.",
            query_text="Device will not hold a charge and shuts off when unplugged.",
        ),
    ]


class TestCorpusConsistencyVerification:
    """Tests covering corpus doc_id parity checks at HybridRetriever initialization."""

    def test_corpus_mismatch_raises_error(self, synthetic_corpus, shared_encoder):
        """Confirm CorpusMismatchError is raised when indices have diverging doc_id sets."""
        bm25_docs = synthetic_corpus[:3]  # DOC-SM-T280, DOC-BILLING, DOC-PASSWORD
        dense_docs = synthetic_corpus[1:]  # DOC-BILLING, DOC-PASSWORD, DOC-HARDWARE

        bm25_idx = build_bm25_index(bm25_docs)
        dense_idx = build_dense_index(dense_docs, shared_encoder)

        with pytest.raises(CorpusMismatchError) as exc_info:
            HybridRetriever(bm25_index=bm25_idx, dense_index=dense_idx, encoder=shared_encoder)

        err_msg = str(exc_info.value)
        assert "Corpus inconsistency detected" in err_msg
        assert "Missing in dense: 1 docs" in err_msg
        assert "DOC-SM-T280" in err_msg
        assert "Missing in sparse: 1 docs" in err_msg
        assert "DOC-HARDWARE" in err_msg

    def test_corpus_match_succeeds(self, synthetic_corpus, shared_encoder):
        """Confirm initialization succeeds when doc_ids are completely identical."""
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)

        retriever = HybridRetriever(
            bm25_index=bm25_idx,
            dense_index=dense_idx,
            encoder=shared_encoder,
        )
        assert retriever.bm25_index is bm25_idx
        assert retriever.dense_index is dense_idx
        assert retriever.encoder is shared_encoder


class TestHybridRetrieveExecution:
    """Tests covering the retrieve() method of HybridRetriever."""

    def test_empty_or_whitespace_query(self, synthetic_corpus, shared_encoder):
        """Confirm empty or whitespace-only query returns empty list without error."""
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)
        retriever = HybridRetriever(bm25_idx, dense_idx, shared_encoder)

        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []
        assert retriever.retrieve("\t\n") == []

    def test_missing_encoder_raises_value_error(self, synthetic_corpus):
        """Confirm ValueError is raised if encoder is neither passed to __init__ nor retrieve()."""
        # Create dummy mock indices with matching doc_ids
        mock_bm25 = BM25Index(bm25=None, corpus=synthetic_corpus, corpus_hash="h1")  # type: ignore
        mock_dense = DenseIndex(documents=synthetic_corpus, embeddings=None, corpus_hash="h1")  # type: ignore

        retriever = HybridRetriever(mock_bm25, mock_dense, encoder=None)

        with pytest.raises(ValueError, match="An encoder instance must be provided"):
            retriever.retrieve("some query")

    def test_end_to_end_retrieve_with_fused_ranks(self, synthetic_corpus, shared_encoder):
        """Confirm end-to-end hybrid retrieval produces fused results with sequential ranks 1..N."""
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)
        retriever = HybridRetriever(bm25_idx, dense_idx, shared_encoder)

        query = "How do I reset my forgotten password?"
        results = retriever.retrieve(query_text=query, top_k_per_track=4, final_top_k=3)

        assert len(results) <= 3
        assert len(results) > 0

        # Ranks must be strictly sequential starting at 1
        for idx, res in enumerate(results, start=1):
            assert isinstance(res, RetrievalResult)
            assert res.rank == idx
            assert res.score > 0.0

        # Most relevant document should be DOC-PASSWORD
        assert results[0].doc_id == "DOC-PASSWORD"

    def test_asymmetric_k_tuning_prioritizes_sparse_code_match(
        self, synthetic_corpus, shared_encoder
    ):
        """Verify asymmetric k tuning (k_sparse=10, k_dense=60) boosts exact product code match.

        The query contains exact product code 'SM-T280'.
        Sparse BM25 will rank DOC-SM-T280 #1 due to exact token match.
        Dense retrieval might rank DOC-HARDWARE high due to semantic terms like 'rebooting', 'power'.
        With asymmetric k (sparse=10, dense=60), DOC-SM-T280 should confidently lead the fusion.
        """
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)
        retriever = HybridRetriever(bm25_idx, dense_idx, shared_encoder)

        query = "Samsung tablet SM-T280 won't turn on"

        # Test with asymmetric k favoring sparse track
        asymmetric_k = {"sparse": 10, "dense": 60}
        results = retriever.retrieve(
            query_text=query,
            top_k_per_track=4,
            final_top_k=2,
            k_per_retriever=asymmetric_k,
        )

        assert len(results) >= 1
        assert results[0].doc_id == "DOC-SM-T280"
        assert results[0].rank == 1

    def test_final_top_k_exceeding_candidates(self, synthetic_corpus, shared_encoder):
        """Confirm if final_top_k exceeds total available candidates, all are returned without padding."""
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)
        retriever = HybridRetriever(bm25_idx, dense_idx, shared_encoder)

        results = retriever.retrieve(
            query_text="billing statement invoice",
            top_k_per_track=4,
            final_top_k=100,  # Far exceeds corpus size (4)
        )

        assert len(results) <= len(synthetic_corpus)
        # Verify no duplicate doc_ids
        doc_ids = [r.doc_id for r in results]
        assert len(doc_ids) == len(set(doc_ids))

    def test_low_top_k_per_track(self, synthetic_corpus, shared_encoder):
        """Confirm retriever handles low top_k_per_track (e.g. 1) without error."""
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)
        retriever = HybridRetriever(bm25_idx, dense_idx, shared_encoder)

        results = retriever.retrieve(
            query_text="invoice payment",
            top_k_per_track=1,
            final_top_k=5,
        )

        # At most 2 unique docs (1 from sparse + 1 from dense)
        assert len(results) in (1, 2)
        assert results[0].rank == 1
        if len(results) == 2:
            assert results[1].rank == 2
            assert results[0].score >= results[1].score

    def test_encoder_provided_at_retrieve_time(self, synthetic_corpus, shared_encoder):
        """Confirm encoder can be omitted from __init__ and passed directly into retrieve()."""
        bm25_idx = build_bm25_index(synthetic_corpus)
        dense_idx = build_dense_index(synthetic_corpus, shared_encoder)

        retriever = HybridRetriever(bm25_idx, dense_idx, encoder=None)
        results = retriever.retrieve(
            query_text="invoice receipt",
            encoder=shared_encoder,
            final_top_k=2,
        )

        assert len(results) > 0
        assert results[0].rank == 1
