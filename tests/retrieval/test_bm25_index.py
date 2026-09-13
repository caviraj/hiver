"""Tests for BM25 Sparse Retrieval Index (M3.P3.1.F1)."""

from pathlib import Path
import pytest

from src.retrieval.bm25_index import (
    BM25Index,
    build_index,
    compute_corpus_hash,
    load_index,
    query_index,
    save_index,
)
from src.retrieval.schema import RetrievalDocument, RetrievalResult


@pytest.fixture
def sample_documents():
    return [
        RetrievalDocument(
            doc_id="doc-1",
            query_text="My Samsung galaxy tablet SM-T280 battery draining very fast.",
            resolution_text="Try turning off background refresh on your SM-T280.",
        ),
        RetrievalDocument(
            doc_id="doc-2",
            query_text="My Android tablet battery drains quickly during video playback.",
            resolution_text="Lower your screen brightness and close background apps.",
        ),
        RetrievalDocument(
            doc_id="doc-3",
            query_text="Cannot connect to WiFi on my laptop after update.",
            resolution_text="Reset network adapter in control panel.",
        ),
        RetrievalDocument(
            doc_id="doc-4",
            query_text="How do I update iOS 11.1 on my older phone?",
            resolution_text="Go to Settings > General > Software Update.",
        ),
    ]


def test_build_index_empty_corpus():
    """Test that build_index raises ValueError on empty corpus."""
    with pytest.raises(ValueError, match="Cannot build BM25 index on an empty document corpus"):
        build_index([])


def test_empty_and_whitespace_query(sample_documents):
    """Test that query_index returns empty list on empty or whitespace queries."""
    index = build_index(sample_documents)
    assert query_index(index, "") == []
    assert query_index(index, "   ") == []
    assert query_index(index, "\n\t") == []


def test_top_k_larger_than_corpus(sample_documents):
    """Test top_k > corpus size returns all documents without padding or errors."""
    index = build_index(sample_documents)
    results = query_index(index, "battery", top_k=100)
    # Corpus has 4 documents, so at most 4 are returned
    assert len(results) <= len(sample_documents)
    assert len(results) == 4
    # Ensure ranks are 1-indexed consecutive
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_single_word_query(sample_documents):
    """Test very short query (single word) returns ranked results without error."""
    index = build_index(sample_documents)
    results = query_index(index, "battery", top_k=2)
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    # doc-1 and doc-2 both have 'battery'
    matched_ids = {r.doc_id for r in results}
    assert "doc-1" in matched_ids
    assert "doc-2" in matched_ids


def test_out_of_vocabulary_code_does_not_crash(sample_documents):
    """Test querying with a device code not present in corpus (true OOV term)."""
    index = build_index(sample_documents)
    # SM-X900 is an unseen Galaxy Tab S8 Ultra model code
    results = query_index(index, "SM-X900 screen frozen", top_k=5)
    # Does not crash; documents without the term score based on remaining words or 0
    assert isinstance(results, list)
    assert len(results) > 0


def test_exact_match_beats_semantic_similarity_proof(sample_documents):
    """CRITICAL PRD PROOF TEST:

    Per the PRD, BM25 is 'indispensable for retrieving highly specific alphanumeric
    strings, product codes, and proprietary error messages where semantic approximation
    would fail'.

    Verify that a query containing the exact product code 'SM-T280' ranks 'doc-1'
    (which has 'SM-T280') strictly ABOVE 'doc-2' (topically similar tablet battery issue
    without the exact code).
    """
    index = build_index(sample_documents)
    query = "battery problem SM-T280"
    results = query_index(index, query, top_k=5)

    assert len(results) >= 2
    assert results[0].doc_id == "doc-1", (
        f"Expected doc-1 (containing exact code SM-T280) to rank #1, but got {results[0].doc_id}"
    )
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_save_and_load_index(tmp_path, sample_documents):
    """Test index persistence and retrieval accuracy after load."""
    index = build_index(sample_documents)
    save_path = tmp_path / "test_bm25"
    save_index(index, sample_documents, save_path)

    loaded_index, loaded_corpus = load_index(save_path)
    assert len(loaded_corpus) == len(sample_documents)
    assert loaded_index.corpus_hash == index.corpus_hash

    # Query loaded index and ensure results match original
    original_results = query_index(index, "WiFi connection", top_k=2)
    loaded_results = query_index(loaded_index, "WiFi connection", top_k=2)

    assert len(original_results) == len(loaded_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.doc_id == loaded.doc_id
        assert orig.rank == loaded.rank
        assert pytest.approx(orig.score) == loaded.score


def test_stale_index_detection_on_load(tmp_path, sample_documents):
    """Test that load_index detects stale index if expected_corpus differs."""
    index = build_index(sample_documents)
    save_path = tmp_path / "test_bm25_stale"
    save_index(index, sample_documents, save_path)

    # Valid load when expected_corpus matches
    loaded_index, _ = load_index(save_path, expected_corpus=sample_documents)
    assert loaded_index is not None

    # Modified corpus (e.g. policy change or updated resolution)
    modified_corpus = [
        RetrievalDocument(
            doc_id="doc-1",
            query_text="My Samsung galaxy tablet SM-T280 battery draining very fast.",
            resolution_text="Updated brand policy: please visit service center immediately.",
        ),
        sample_documents[1],
        sample_documents[2],
        sample_documents[3],
    ]

    with pytest.raises(ValueError, match="Stale index detected"):
        load_index(save_path, expected_corpus=modified_corpus)


def test_configurable_k1_and_b(sample_documents):
    """Test custom k1 and b parameters."""
    index = build_index(sample_documents, k1=2.0, b=0.5)
    assert index.k1 == 2.0
    assert index.b == 0.5
