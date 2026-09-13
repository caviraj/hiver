"""Tests for length control and verbosity bias mitigation."""

import math
import pytest
from src.evaluation.bias_mitigation.length_control import (
    EvalItem,
    ResidualizedScores,
    apply_length_controlled_regression,
    bucket_by_token_count,
    estimate_or_count_tokens,
    residualize_eval_items,
)


def test_bucket_by_token_count_default_edges():
    items = [
        EvalItem(text="a " * 20, score=1.0, metadata={"id": 1}),    # ~20 tokens -> [0, 50) -> "0-50"
        EvalItem(text="b " * 80, score=2.0, metadata={"id": 2}),    # ~80 tokens -> [50, 150) -> "50-150"
        EvalItem(text="c " * 200, score=3.0, metadata={"id": 3}),   # ~200 tokens -> [150, 400) -> "150-400"
        EvalItem(text="d " * 500, score=4.0, metadata={"id": 4}),   # ~500 tokens -> [400, inf) -> "400-inf"
    ]
    buckets = bucket_by_token_count(items)
    assert len(buckets["0-50"]) == 1
    assert buckets["0-50"][0].metadata["id"] == 1
    assert len(buckets["50-150"]) == 1
    assert buckets["50-150"][0].metadata["id"] == 2
    assert len(buckets["150-400"]) == 1
    assert buckets["150-400"][0].metadata["id"] == 3
    assert len(buckets["400-inf"]) == 1
    assert buckets["400-inf"][0].metadata["id"] == 4


def test_bucket_by_token_count_custom_edges_and_eval_items():
    items = [
        EvalItem(text="a b c", score=4.0, token_count=10, metadata={"id": "item-1"}),
        EvalItem(text="a " * 40, score=3.5, token_count=40, metadata={"id": "item-2"}),
        EvalItem(text="a " * 90, score=5.0, token_count=90, metadata={"id": "item-3"}),
    ]
    buckets = bucket_by_token_count(items, bucket_edges=[0, 30, 80, math.inf])
    assert "0-30" in buckets
    assert "30-80" in buckets
    assert "80-inf" in buckets
    assert len(buckets["0-30"]) == 1
    assert buckets["0-30"][0].metadata["id"] == "item-1"
    assert len(buckets["30-80"]) == 1
    assert buckets["30-80"][0].metadata["id"] == "item-2"
    assert len(buckets["80-inf"]) == 1
    assert buckets["80-inf"][0].metadata["id"] == "item-3"


def test_synthetic_verbosity_bias_removal():
    """Verify that OLS residualization removes linear length correlation."""
    # Create 50 items where raw_score = 1.0 + 0.04 * length
    token_counts = [10 + i * 5 for i in range(50)]
    # Perfect linear relationship with small variation
    raw_scores = [1.0 + 0.04 * x for x in token_counts]

    res = apply_length_controlled_regression(raw_scores, token_counts)
    assert isinstance(res, ResidualizedScores)
    assert not res.skipped
    assert abs(res.slope - 0.04) < 1e-4

    # The residualized scores should be ~ 0.0 because residual = raw - (alpha + beta * x) = 0
    for score in res:
        assert score == pytest.approx(0.0, abs=1e-4)

    # Check correlation with token_counts is ~0
    n = len(res)
    mean_x = sum(token_counts) / n
    mean_y = sum(res) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(token_counts, res))
    var_x = sum((x - mean_x) ** 2 for x in token_counts)
    var_y = sum((y - mean_y) ** 2 for y in res)
    if var_y > 1e-9:
        corr = cov / math.sqrt(var_x * var_y)
        assert abs(corr) < 0.05
    else:
        # All residuals are identical (zero variance), meaning correlation is 0
        assert True


def test_fewer_than_ten_items_skips_regression():
    token_counts = [20, 40, 60, 80, 100]
    raw_scores = [2.0, 3.0, 4.0, 4.5, 5.0]

    res = apply_length_controlled_regression(raw_scores, token_counts)
    assert isinstance(res, ResidualizedScores)
    assert res.skipped is True
    assert "Insufficient valid items" in res.skip_reason
    assert list(res) == raw_scores
    assert res.slope is None


def test_zero_variance_token_counts_skips_regression():
    token_counts = [50] * 15
    raw_scores = [1.0 + (i % 5) for i in range(15)]

    res = apply_length_controlled_regression(raw_scores, token_counts)
    assert isinstance(res, ResidualizedScores)
    assert res.skipped is True
    assert "Zero variance" in res.skip_reason
    assert list(res) == raw_scores
    assert res.slope is None


def test_zero_or_negative_token_counts_excluded_from_fit():
    # 12 items: 2 have token_count <= 0, 10 have valid positive counts
    token_counts = [0, -5] + [20 + i * 10 for i in range(10)]
    raw_scores = [3.0, 2.5] + [1.0 + 0.02 * (20 + i * 10) for i in range(10)]

    res = apply_length_controlled_regression(raw_scores, token_counts)
    assert not res.skipped
    assert 0 in res.zero_token_indices
    assert 1 in res.zero_token_indices
    # Preserves length and index alignment
    assert len(res) == len(raw_scores)
    # The degenerate items retain their original scores
    assert res[0] == 3.0
    assert res[1] == 2.5


def test_zero_tokens_bringing_valid_count_below_ten():
    # 10 items total, but 2 have 0 tokens -> only 8 valid -> should skip
    token_counts = [0, 0] + [30 + i * 5 for i in range(8)]
    raw_scores = [2.0] * 10

    res = apply_length_controlled_regression(raw_scores, token_counts)
    assert res.skipped is True
    assert "Insufficient valid items" in res.skip_reason
    assert len(res.zero_token_indices) == 2


def test_mismatched_lengths_raises_value_error():
    with pytest.raises(ValueError, match="Length mismatch"):
        apply_length_controlled_regression([1.0, 2.0], [10])


def test_residualize_eval_items_helper():
    items = [
        EvalItem(text=f"text {i} " * (10 + i * 2), score=2.0 + 0.05 * i, metadata={"id": f"doc_{i}"})
        for i in range(15)
    ]
    res_items = residualize_eval_items(items)
    assert len(res_items) == 15
    for item in res_items:
        assert item.residualized_score is not None
        assert isinstance(item.residualized_score, float)


def test_estimate_or_count_tokens():
    assert estimate_or_count_tokens("") == 0
    # "hello world" is 2 tokens in tiktoken, 3 tokens under 1.33 heuristic
    assert estimate_or_count_tokens("hello world") in (2, 3)
    # 3 words is 3 tokens in tiktoken, 4 tokens under 1.33 heuristic
    assert estimate_or_count_tokens("   one   two   three   ") in (3, 4)
