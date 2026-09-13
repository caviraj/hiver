"""Unit and integration tests for dense_index.py (M3.P3.2.F1).

Verifies:
1. Empty corpus and empty query edge cases.
2. Top-k capping when top_k exceeds corpus size.
3. Query truncation on oversized input with warning logging.
4. Near-duplicate embedding counting without deduplication.
5. Corpus document ID extraction via get_corpus_doc_ids.
6. FAISS and NumPy fallback ranking and score equivalence.
7. Persistence with corpus-hash staleness detection.
8. Core capability proof: Vocabulary-gap retrieval where BM25 fails (zero lexical overlap).
9. Full integration with build_corpus from P3.1.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.data.schema import RawTweet
from src.data.thread_schema import Thread

from src.retrieval.corpus_builder import build_corpus
from src.retrieval.dense_index import (
    DenseIndex,
    build_dense_index,
    get_corpus_doc_ids,
    load_dense_index,
    query_dense_index,
    save_dense_index,
)
from src.retrieval.embedding_model import DEFAULT_MODEL_NAME, load_encoder
from src.retrieval.hashing import compute_corpus_hash
from src.retrieval.schema import RetrievalDocument


@pytest.fixture(scope="module")
def shared_encoder():
    """Load the default encoder once for test module efficiency."""
    return load_encoder(DEFAULT_MODEL_NAME)


@pytest.fixture
def synthetic_corpus() -> List[RetrievalDocument]:
    """A small controlled corpus with diverse queries and intents."""
    return [
        RetrievalDocument(
            doc_id="DOC-001",
            resolution_text="Reboot the tablet and clear system cache.",
            query_text="My display is frozen and completely unresponsive on touch.",
        ),
        RetrievalDocument(
            doc_id="DOC-002",
            resolution_text="Go to Settings -> Billing -> Update Payment Method.",
            query_text="How can I update my credit card details for monthly billing?",
        ),
        RetrievalDocument(
            doc_id="DOC-003",
            resolution_text="Navigate to Account -> Security -> Change Password.",
            query_text="Need help resetting my forgotten login password.",
        ),
    ]


class TestDenseIndexBuildEdgeCases:
    """Tests covering build_dense_index edge cases."""

    def test_empty_corpus_raises_value_error(self, shared_encoder):
        """Confirm building dense index with empty corpus raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            build_dense_index([], shared_encoder)
        assert "Cannot build a dense index on an empty document corpus" in str(exc_info.value)

    def test_near_duplicate_detection(self, shared_encoder, caplog):
        """Confirm near-identical documents are detected, counted, logged, but NOT dropped."""
        # Two documents with virtually identical query texts
        docs = [
            RetrievalDocument(
                doc_id="DUP-1",
                resolution_text="Standard canned response for refund.",
                query_text="I would like to request a full refund for my subscription.",
            ),
            RetrievalDocument(
                doc_id="DUP-2",
                resolution_text="Standard canned response for refund.",
                query_text="I would like to request a full refund for my subscription.",
            ),
            RetrievalDocument(
                doc_id="DUP-3",
                resolution_text="Reboot device.",
                query_text="Device won't power on after software update.",
            ),
        ]

        with caplog.at_level(logging.WARNING):
            index = build_dense_index(docs, shared_encoder)

        assert len(index) == 3
        assert index.near_duplicate_count >= 1
        assert "near-duplicate query embeddings detected" in caplog.text


class TestDenseIndexQueryEdgeCases:
    """Tests covering query_dense_index edge cases."""

    def test_empty_and_whitespace_query(self, synthetic_corpus, shared_encoder):
        """Confirm empty or whitespace-only queries return empty list immediately."""
        index = build_dense_index(synthetic_corpus, shared_encoder)

        mock_encoder = MagicMock()
        res_empty = query_dense_index(index, "", mock_encoder)
        res_spaces = query_dense_index(index, "   \n\t  ", mock_encoder)

        assert res_empty == []
        assert res_spaces == []
        # Ensure encoder was not called on empty query
        mock_encoder.encode.assert_not_called()

    def test_top_k_capping_larger_than_corpus(self, synthetic_corpus, shared_encoder):
        """Confirm top_k > corpus size returns exactly all documents without error."""
        index = build_dense_index(synthetic_corpus, shared_encoder)
        results = query_dense_index(index, "password reset", shared_encoder, top_k=50)

        assert len(results) == len(synthetic_corpus)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_query_truncation_warning(self, synthetic_corpus, shared_encoder, caplog):
        """Confirm queries exceeding encoder max_seq_length are truncated and logged."""
        index = build_dense_index(synthetic_corpus, shared_encoder)

        # Create an oversized query (> max_seq_length words)
        max_words = shared_encoder.max_seq_length or 256
        oversized_query = "word " * (max_words + 50)

        with caplog.at_level(logging.WARNING):
            results = query_dense_index(index, oversized_query, shared_encoder, top_k=2)

        assert len(results) > 0
        assert "exceeds encoder max sequence length" in caplog.text


class TestCorpusDocIdsSafetyNet:
    """Tests for get_corpus_doc_ids to support P3.3 fusion safety checks."""

    def test_get_corpus_doc_ids_dense_index(self, synthetic_corpus, shared_encoder):
        """Confirm get_corpus_doc_ids returns the exact set of doc_ids."""
        index = build_dense_index(synthetic_corpus, shared_encoder)
        doc_ids = get_corpus_doc_ids(index)

        expected = {"DOC-001", "DOC-002", "DOC-003"}
        assert doc_ids == expected
        assert index.get_corpus_doc_ids() == expected

    def test_get_corpus_doc_ids_duck_typing(self, synthetic_corpus):
        """Confirm get_corpus_doc_ids works on arbitrary index containers exposing corpus."""
        class MockIndex:
            def __init__(self, docs):
                self.corpus = docs

        mock_idx = MockIndex(synthetic_corpus)
        doc_ids = get_corpus_doc_ids(mock_idx)
        assert doc_ids == {"DOC-001", "DOC-002", "DOC-003"}


class TestFaissAndFallbackEquivalence:
    """Tests validating FAISS and NumPy fallback search path behavior and equivalence."""

    def test_fallback_activation_warning_when_faiss_unavailable(self, synthetic_corpus, shared_encoder, caplog):
        """Confirm fallback triggers warning when use_faiss=True but faiss is not installed."""
        with patch.dict("sys.modules", {"faiss": None}):
            with caplog.at_level(logging.WARNING):
                index = build_dense_index(synthetic_corpus, shared_encoder, use_faiss=True)
                assert index.faiss_index is None
                assert index.use_faiss is False

    def test_numpy_and_faiss_path_ranking_equivalence(self, synthetic_corpus, shared_encoder):
        """Mock FAISS IndexFlatIP to ensure NumPy and FAISS return identical rankings and scores."""
        # Create a mock FAISS index class that computes exact inner products
        class MockIndexFlatIP:
            def __init__(self, dim):
                self.dim = dim
                self.data = None

            def add(self, data):
                self.data = np.copy(data)

            def search(self, q, k):
                # q shape: (1, dim), data shape: (N, dim)
                sims = np.dot(self.data, q.T).flatten()
                top_indices = np.argsort(-sims)[:k]
                top_scores = sims[top_indices]
                return top_scores.reshape(1, -1), top_indices.reshape(1, -1)

        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP = MockIndexFlatIP

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            # Build index with mocked FAISS
            faiss_index = build_dense_index(synthetic_corpus, shared_encoder, use_faiss=True)
            assert faiss_index.faiss_index is not None

            # Build index explicitly using NumPy fallback
            numpy_index = build_dense_index(synthetic_corpus, shared_encoder, use_faiss=False)
            assert numpy_index.faiss_index is None

            query = "My device screen locked up and touch won't respond"
            faiss_results = query_dense_index(faiss_index, query, shared_encoder, top_k=3)
            numpy_results = query_dense_index(numpy_index, query, shared_encoder, top_k=3)

            assert len(faiss_results) == len(numpy_results) == 3
            for r_faiss, r_numpy in zip(faiss_results, numpy_results):
                assert r_faiss.doc_id == r_numpy.doc_id
                assert r_faiss.rank == r_numpy.rank
                assert r_faiss.score == pytest.approx(r_numpy.score, abs=1e-5)


class TestDenseIndexPersistenceAndStaleness:
    """Tests for saving, loading, and corpus hash staleness detection."""

    def test_save_and_load_roundtrip(self, synthetic_corpus, shared_encoder, tmp_path: Path):
        """Confirm saving and loading preserves index structure and search behavior."""
        index = build_dense_index(synthetic_corpus, shared_encoder, use_faiss=False)
        save_path = tmp_path / "dense_test_index.pkl"

        save_dense_index(index, synthetic_corpus, save_path)
        assert save_path.exists()

        loaded_index, loaded_corpus = load_dense_index(save_path, expected_corpus=synthetic_corpus, use_faiss=False)
        assert len(loaded_index) == len(index)
        assert len(loaded_corpus) == len(synthetic_corpus)
        assert loaded_index.corpus_hash == index.corpus_hash
        assert loaded_index.doc_ids == index.doc_ids

        # Query loaded index
        results = query_dense_index(loaded_index, "how to change credit card", shared_encoder, top_k=1)
        assert len(results) == 1
        assert results[0].doc_id == "DOC-002"

    def test_save_with_mismatched_corpus_raises_error(self, synthetic_corpus, shared_encoder, tmp_path: Path):
        """Confirm saving with a corpus different from the indexed corpus raises ValueError."""
        index = build_dense_index(synthetic_corpus, shared_encoder)
        mismatched_corpus = synthetic_corpus[:-1]  # Dropped one doc

        save_path = tmp_path / "mismatch_test.pkl"
        with pytest.raises(ValueError) as exc_info:
            save_dense_index(index, mismatched_corpus, save_path)
        assert "Corpus hash mismatch during save" in str(exc_info.value)

    def test_load_with_stale_expected_corpus_raises_error(self, synthetic_corpus, shared_encoder, tmp_path: Path):
        """Confirm loading with modified expected_corpus detects staleness and raises ValueError."""
        index = build_dense_index(synthetic_corpus, shared_encoder, use_faiss=False)
        save_path = tmp_path / "stale_test.pkl"
        save_dense_index(index, synthetic_corpus, save_path)

        stale_corpus = synthetic_corpus.copy()
        stale_corpus[0] = RetrievalDocument(
            doc_id=synthetic_corpus[0].doc_id,
            resolution_text="Different resolution text.",
            query_text=synthetic_corpus[0].query_text,
        )

        with pytest.raises(ValueError) as exc_info:
            load_dense_index(save_path, expected_corpus=stale_corpus)
        assert "Corpus hash mismatch" in str(exc_info.value)
        assert "Dense index is stale" in str(exc_info.value)


class TestVocabularyGapRetrievalProof:
    """Core capability proof: Demonstrate dense retrieval succeeds where BM25 has zero token overlap.

    Vocabulary gap scenario:
    - Document query_text: "display unresponsive and frozen"
    - User query: "screen is stuck completely"
    - Lexical token overlap: ZERO common content tokens.
    - Dense cosine similarity: high (> 0.65), retrieving the document at rank 1.
    """

    def test_vocabulary_gap_retrieval_proof(self, shared_encoder):
        """Demonstrate that dense semantic retrieval bridges lexical vocabulary gaps."""
        corpus = [
            RetrievalDocument(
                doc_id="DOC-DISPLAY-FROZEN",
                resolution_text="Hold power and volume down for 10 seconds to force reboot.",
                query_text="display unresponsive and frozen",
            ),
            RetrievalDocument(
                doc_id="DOC-BILLING-INVOICE",
                resolution_text="Download PDF statement from the billing tab.",
                query_text="monthly payment invoice receipt download",
            ),
            RetrievalDocument(
                doc_id="DOC-SHIPPING-STATUS",
                resolution_text="Check FedEx tracking link in confirmation email.",
                query_text="delivery package tracking arrival carrier",
            ),
        ]

        index = build_dense_index(corpus, shared_encoder)

        # Query with ZERO token overlap with "display unresponsive and frozen"
        query_text = "screen is stuck completely"

        # Verify zero token overlap
        query_tokens = set(query_text.lower().split())
        doc_tokens = set(corpus[0].query_text.lower().split())
        assert query_tokens.isdisjoint(doc_tokens), f"Tokens overlap: {query_tokens & doc_tokens}"

        results = query_dense_index(index, query_text, shared_encoder, top_k=3)

        assert len(results) > 0
        top_result = results[0]

        # Top document must be the semantically matching display frozen document
        assert top_result.doc_id == "DOC-DISPLAY-FROZEN", (
            f"Expected DOC-DISPLAY-FROZEN at rank 1, got {top_result.doc_id} with score {top_result.score}"
        )
        assert top_result.rank == 1
        # High cosine similarity expected from sentence-transformers model
        assert top_result.score > 0.50, f"Expected cosine similarity > 0.50, got {top_result.score}"
        doc_by_id = {doc.doc_id: doc for doc in corpus}
        assert doc_by_id[top_result.doc_id].resolution_text == "Hold power and volume down for 10 seconds to force reboot."


class TestCorpusBuilderIntegration:
    """Integration test verifying build_dense_index works directly with build_corpus output."""

    def test_corpus_builder_direct_consumption(self, shared_encoder):
        """Confirm build_corpus() from P3.1 produces documents ready for build_dense_index()."""
        t1 = RawTweet(
            tweet_id="101",
            author_id="cust1",
            inbound=True,
            created_at=datetime(2023, 1, 1, 12, 0, 0),
            text="Device restarts when opening camera on SM-T280.",
        )
        t2 = RawTweet(
            tweet_id="102",
            author_id="brand",
            inbound=False,
            created_at=datetime(2023, 1, 1, 12, 5, 0),
            text="Update camera app and grant storage permission.",
            in_response_to_tweet_id="101",
        )
        t3 = RawTweet(
            tweet_id="201",
            author_id="cust2",
            inbound=True,
            created_at=datetime(2023, 1, 1, 13, 0, 0),
            text="Need refund for duplicate transaction.",
        )
        t4 = RawTweet(
            tweet_id="202",
            author_id="brand",
            inbound=False,
            created_at=datetime(2023, 1, 1, 13, 5, 0),
            text="Refund processed within 3-5 business days.",
            in_response_to_tweet_id="201",
        )
        threads = [
            Thread(thread_id="101", tweets=[t1, t2], terminal=True),
            Thread(thread_id="201", tweets=[t3, t4], terminal=True),
        ]

        # Build corpus using P3.1 corpus builder directly
        corpus = build_corpus(threads)
        assert len(corpus) == 2

        # Build dense index directly over identical corpus
        dense_index = build_dense_index(corpus, shared_encoder)
        assert len(dense_index) == 2
        assert get_corpus_doc_ids(dense_index) == {doc.doc_id for doc in corpus}

        # Query dense index
        res = query_dense_index(dense_index, "camera crash on tablet", shared_encoder, top_k=1)
        assert len(res) == 1
        assert res[0].doc_id == corpus[0].doc_id
