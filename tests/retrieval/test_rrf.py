"""Tests for Reciprocal Rank Fusion (RRF) logic (M3.P3.3.F1).

Validates:
- Basic multi-retriever fusion with full overlap.
- Documents present in only one retriever list (partial overlap / absence handling).
- Asymmetric k proof: sparse rank #1 with k=10 outranks dense rank #1 with k=60.
- Empty result lists and zero-candidate safety.
- Deterministic alphabetical tie-breaking on identical fused scores.
- Input validation (missing retriever keys, negative k).
"""

import pytest
from src.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.retrieval.schema import RetrievalResult


def test_basic_two_retriever_full_overlap():
    """Verify standard RRF fusion when both retrievers return overlapping documents."""
    sparse_results = [
        RetrievalResult(doc_id="doc_A", score=12.5, rank=1),
        RetrievalResult(doc_id="doc_B", score=8.0, rank=2),
        RetrievalResult(doc_id="doc_C", score=5.2, rank=3),
    ]
    dense_results = [
        RetrievalResult(doc_id="doc_B", score=0.92, rank=1),
        RetrievalResult(doc_id="doc_A", score=0.85, rank=2),
        RetrievalResult(doc_id="doc_C", score=0.71, rank=3),
    ]

    k_config = {"sparse": 60, "dense": 60}
    fused = reciprocal_rank_fusion(
        result_lists={"sparse": sparse_results, "dense": dense_results},
        k_per_retriever=k_config,
    )

    assert len(fused) == 3
    # doc_A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.03252248
    # doc_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252248
    # doc_A and doc_B have identical scores! Tie-breaker: doc_A comes before doc_B alphabetically.
    # doc_C: 1/(60+3) + 1/(60+3) = 2/63 = 0.03174603
    assert fused[0].doc_id == "doc_A"
    assert fused[0].rank == 1
    assert fused[1].doc_id == "doc_B"
    assert fused[1].rank == 2
    assert fused[2].doc_id == "doc_C"
    assert fused[2].rank == 3
    assert fused[0].score == fused[1].score
    assert fused[0].score > fused[2].score


def test_document_in_only_one_retriever_list():
    """Verify that document present only in one retriever gets single track contribution, not zeroed out or penalized."""
    sparse_results = [
        RetrievalResult(doc_id="doc_sparse_only", score=15.0, rank=1),
        RetrievalResult(doc_id="doc_shared", score=10.0, rank=2),
    ]
    dense_results = [
        RetrievalResult(doc_id="doc_shared", score=0.95, rank=1),
        RetrievalResult(doc_id="doc_dense_only", score=0.88, rank=2),
    ]

    k_config = {"sparse": 60, "dense": 60}
    fused = reciprocal_rank_fusion(
        result_lists={"sparse": sparse_results, "dense": dense_results},
        k_per_retriever=k_config,
    )

    assert len(fused) == 3
    # doc_shared: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252248
    # doc_sparse_only: 1/(60+1) + 0 = 1/61 = 0.01639344
    # doc_dense_only: 0 + 1/(60+2) = 1/62 = 0.01612903
    assert fused[0].doc_id == "doc_shared"
    assert fused[0].rank == 1
    assert fused[1].doc_id == "doc_sparse_only"
    assert fused[1].rank == 2
    assert fused[2].doc_id == "doc_dense_only"
    assert fused[2].rank == 3

    # Ensure sparse_only score is exactly round(1/61, 8)
    assert fused[1].score == round(1.0 / 61, 8)
    # Ensure dense_only score is exactly round(1/62, 8)
    assert fused[2].score == round(1.0 / 62, 8)


def test_asymmetric_k_proof():
    """Concrete proof of the PRD asymmetric k design rationale.

    With k_sparse=10 and k_dense=60:
    - doc_exact: rank #1 in sparse (exact product-code match), absent from dense
      score = 1 / (10 + 1) = 1/11 ≈ 0.09090909
    - doc_semantic: rank #1 in dense (high semantic similarity), absent from sparse
      score = 1 / (60 + 1) = 1/61 ≈ 0.01639344

    Even though both documents were rank #1 in their respective tracks,
    doc_exact outranks doc_semantic by ~5.5x due to the lower k_sparse.
    """
    sparse_results = [
        RetrievalResult(doc_id="doc_exact_code_match", score=25.0, rank=1),
    ]
    dense_results = [
        RetrievalResult(doc_id="doc_semantic_match", score=0.99, rank=1),
    ]

    k_config = {"sparse": 10, "dense": 60}
    fused = reciprocal_rank_fusion(
        result_lists={"sparse": sparse_results, "dense": dense_results},
        k_per_retriever=k_config,
    )

    assert len(fused) == 2
    assert fused[0].doc_id == "doc_exact_code_match"
    assert fused[0].rank == 1
    assert fused[1].doc_id == "doc_semantic_match"
    assert fused[1].rank == 2

    # Verify scores reflect the asymmetric formula
    expected_sparse_score = round(1.0 / (10 + 1), 8)
    expected_dense_score = round(1.0 / (60 + 1), 8)
    assert fused[0].score == expected_sparse_score
    assert fused[1].score == expected_dense_score
    assert fused[0].score > fused[1].score


def test_empty_result_lists():
    """Verify graceful handling of empty result lists."""
    k_config = {"sparse": 60, "dense": 60}

    # Completely empty result_lists dict
    assert reciprocal_rank_fusion({}, k_config) == []

    # Retrievers present but candidate lists are empty
    empty_tracks = {"sparse": [], "dense": []}
    assert reciprocal_rank_fusion(empty_tracks, k_config) == []

    # One empty, one populated
    one_empty = {
        "sparse": [RetrievalResult(doc_id="doc_1", score=10.0, rank=1)],
        "dense": [],
    }
    res = reciprocal_rank_fusion(one_empty, k_config)
    assert len(res) == 1
    assert res[0].doc_id == "doc_1"
    assert res[0].rank == 1


def test_deterministic_tie_breaking():
    """Verify that documents with identical scores are sorted alphabetically by doc_id."""
    # Construct 3 documents that will each get identical score from sparse retriever
    # (e.g. all rank 1 in separate tracks or symmetric dual tracks)
    sparse_results = [
        RetrievalResult(doc_id="doc_zebra", score=10.0, rank=1),
        RetrievalResult(doc_id="doc_apple", score=8.0, rank=2),
    ]
    dense_results = [
        RetrievalResult(doc_id="doc_apple", score=0.9, rank=1),
        RetrievalResult(doc_id="doc_zebra", score=0.8, rank=2),
    ]

    # Both doc_zebra and doc_apple have rank 1 and rank 2 -> identical fused score:
    # 1/(60+1) + 1/(60+2)
    k_config = {"sparse": 60, "dense": 60}
    fused = reciprocal_rank_fusion(
        result_lists={"sparse": sparse_results, "dense": dense_results},
        k_per_retriever=k_config,
    )

    assert len(fused) == 2
    assert fused[0].score == fused[1].score
    # Alphabetical order: 'doc_apple' before 'doc_zebra'
    assert fused[0].doc_id == "doc_apple"
    assert fused[0].rank == 1
    assert fused[1].doc_id == "doc_zebra"
    assert fused[1].rank == 2


def test_input_validation_errors():
    """Verify error handling for missing retriever keys and negative k values."""
    results = {
        "sparse": [RetrievalResult(doc_id="doc_1", score=1.0, rank=1)],
    }

    # Missing retriever key in k_per_retriever
    with pytest.raises(KeyError, match="missing from k_per_retriever"):
        reciprocal_rank_fusion(results, {"dense": 60})

    # Negative k value
    with pytest.raises(ValueError, match="must be non-negative"):
        reciprocal_rank_fusion(results, {"sparse": -5})
