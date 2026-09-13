"""Tests for LLM judge calibration and deployment gating."""

import math
from unittest.mock import patch
import pytest
from sklearn.metrics import cohen_kappa_score

from src.evaluation.judge_calibration import (
    CalibrationResult,
    compute_quadratic_weighted_kappa,
    discretize_scores,
    evaluate_judge_calibration,
    JUDGE_DEPLOYMENT_KAPPA_THRESHOLD,
)


class TestDiscretizeScores:
    """Test suite for score discretization and boundary rounding rules."""

    def test_round_half_up_exact_boundary(self):
        """Verify round-half-up behavior at exact .5 boundaries.

        Python's native round() uses round-half-to-even (banker's rounding):
        round(2.5) == 2, round(3.5) == 4.
        Our specification mandates deterministic round-half-up:
        2.5 -> 3, 3.5 -> 4.
        """
        assert round(2.5) == 2, "Sanity check: standard Python round(2.5) is 2"
        assert discretize_scores([2.5]) == [3], "discretize_scores must round 2.5 to 3"
        assert discretize_scores([3.5]) == [4], "discretize_scores must round 3.5 to 4"
        assert discretize_scores([1.5]) == [2], "discretize_scores must round 1.5 to 2"
        assert discretize_scores([4.5]) == [5], "discretize_scores must round 4.5 to 5"

    def test_general_rounding(self):
        """Test non-boundary rounding."""
        inputs = [1.1, 1.4, 1.6, 2.9, 4.2, 4.8]
        expected = [1, 1, 2, 3, 4, 5]
        assert discretize_scores(inputs) == expected

    def test_clamping_to_valid_range(self):
        """Test that extreme or out-of-range floats clamp strictly to [1, 5]."""
        extreme_inputs = [-2.0, 0.4, 0.9, 5.1, 7.5, 100.0]
        expected = [1, 1, 1, 5, 5, 5]
        assert discretize_scores(extreme_inputs) == expected

    def test_empty_list(self):
        """Test discretization of empty score list."""
        assert discretize_scores([]) == []


class TestQuadraticWeightedKappa:
    """Test suite for Quadratic Weighted Cohen's Kappa calculation."""

    def test_perfect_agreement(self):
        """Test kappa computation when human and LLM ratings match perfectly."""
        human = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        llm = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        kappa = compute_quadratic_weighted_kappa(human, llm)
        assert math.isclose(kappa, 1.0, rel_tol=1e-5)

    def test_degenerate_zero_variance_raises_informative_error(self):
        """Test degenerate case where zero score variance causes chance agreement division by zero.

        When all ratings across both human and LLM are the exact same integer (e.g. all 3s),
        cohen_kappa_score produces NaN. The function must intercept this and raise ValueError.
        """
        all_threes_human = [3, 3, 3, 3, 3, 3]
        all_threes_llm = [3, 3, 3, 3, 3, 3]

        with pytest.raises(ValueError, match="insufficient score variance for Kappa calculation"):
            compute_quadratic_weighted_kappa(all_threes_human, all_threes_llm)

    def test_input_length_mismatch(self):
        """Test that unequal list lengths raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_quadratic_weighted_kappa([1, 2, 3], [1, 2])

    def test_empty_inputs_raise(self):
        """Test that empty score lists raise ValueError."""
        with pytest.raises(ValueError, match="empty score lists"):
            compute_quadratic_weighted_kappa([], [])

    def test_quadratic_vs_linear_penalty_proof(self):
        """Verify that quadratic weighting penalizes severe discordance more than linear weighting.

        Human ratings are low (1s and 2s) while LLM ratings are high (4s and 5s).
        Because quadratic weighting squares the distances between categories ((5-1)^2 = 16 vs 5-1 = 4),
        the resulting quadratic kappa must be strictly lower (more negative) than linear kappa.
        """
        human = [1, 1, 2, 2, 1, 2, 1, 2]
        llm = [5, 5, 4, 4, 5, 4, 5, 4]

        kappa_quad = compute_quadratic_weighted_kappa(human, llm)
        kappa_linear = float(
            cohen_kappa_score(
                y1=human,
                y2=llm,
                weights="linear",
                labels=[1, 2, 3, 4, 5],
            )
        )

        assert kappa_quad < kappa_linear, (
            f"Expected quadratic kappa ({kappa_quad:.4f}) to be strictly lower than "
            f"linear kappa ({kappa_linear:.4f}) for maximally discordant data."
        )


class TestEvaluateJudgeCalibration:
    """Test suite for the hard deployment qualification gate."""

    def test_boundary_exactly_point_six_zero_fails(self):
        """Test that kappa exactly equal to 0.60 fails (requires strictly greater than 0.60)."""
        human = [1, 2, 3, 4, 5]
        llm = [1, 2, 3, 4, 5]

        with patch(
            "src.evaluation.judge_calibration.compute_quadratic_weighted_kappa",
            return_value=0.60,
        ):
            result = evaluate_judge_calibration(human, llm, threshold=0.60)
            assert result.kappa == 0.60
            assert result.passed is False, "A kappa of exactly 0.60 must fail the deployment gate"
            assert any("FAILED" in note for note in result.notes)

    def test_boundary_exceeding_point_six_zero_passes(self):
        """Test that kappa strictly greater than 0.60 passes."""
        human = [1, 2, 3, 4, 5]
        llm = [1, 2, 3, 4, 5]

        with patch(
            "src.evaluation.judge_calibration.compute_quadratic_weighted_kappa",
            return_value=0.6001,
        ):
            result = evaluate_judge_calibration(human, llm, threshold=0.60)
            assert result.kappa == 0.6001
            assert result.passed is True
            assert any("passed" in note.lower() for note in result.notes)

    def test_missing_human_scores_excluded_without_imputation(self):
        """Verify that unlabeled golden examples (human_score=None) are excluded without imputation."""
        # 10 pairs where 3 human scores are None
        human: list[int | None] = [1, 2, None, 4, 5, None, 2, 3, None, 5]
        llm = [1, 2, 3, 4, 5, 2, 2, 3, 4, 5]

        result = evaluate_judge_calibration(human, llm)
        assert result.n_examples == 7, "Must evaluate exactly the 7 labeled pairs"
        assert any("Excluded 3 unlabeled items" in note for note in result.notes)

    def test_fewer_than_two_labeled_pairs_raises(self):
        """Test that having fewer than 2 valid paired items raises ValueError."""
        human = [None, 3, None]
        llm = [1, 3, 5]

        with pytest.raises(ValueError, match="Insufficient labeled examples"):
            evaluate_judge_calibration(human, llm)

    def test_length_mismatch_raises(self):
        """Test that unequal lists raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_judge_calibration([1, 2], [1, 2, 3])
