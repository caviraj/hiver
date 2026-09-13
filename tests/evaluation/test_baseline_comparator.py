"""Tests for baseline comparator module.

M5.P5.5.F1: Baselines & Failure Mode Tracking.
"""

from unittest.mock import MagicMock
import pytest

from src.evaluation.baseline_comparator import (
    BaselineComparisonReport,
    EvalItem,
    run_baseline_comparison,
)
from src.evaluation.baselines.trivial_baseline import trivial_baseline_response
from src.evaluation.baselines.zeroshot_baseline import (
    is_baseline_error,
    zero_shot_baseline_response,
)


class TestBaselineComparator:
    def test_empty_eval_set_raises_error(self):
        """CRITICAL: Empty eval set must raise a clear ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            run_baseline_comparison(
                eval_set=[],
                primary_pipeline_fn=lambda q: "primary",
                trivial_fn=lambda q: None,
                zeroshot_fn=lambda q: "zero-shot",
            )

    def test_trivial_baseline_faithfulness_is_none_not_zero(self, tmp_path):
        """CRITICAL: Trivial baseline retrieves 0 contexts, so faithfulness is None, never 0.0."""
        eval_set = [
            EvalItem(query="My phone is broken", retrieved_contexts=["ctx1"]),
            EvalItem(query="My screen is frozen", retrieved_contexts=["ctx2"]),
        ]

        # primary returns text
        def mock_primary(q):
            return {"response": f"Primary reply to {q}", "retrieved_contexts": ["ctx"]}

        # trivial returns static responder
        def mock_trivial(q):
            return trivial_baseline_response(q)

        # zeroshot returns basic answer
        def mock_zeroshot(q):
            return f"Raw LLM reply to {q}"

        def mock_ragas(query, response, contexts, reference_answer=None):
            return {
                "faithfulness": 0.95,
                "context_precision": 0.9,
                "context_recall": 0.85,
            }

        def mock_geval(query, response, contexts):
            return 4.5

        out_path = tmp_path / "baseline_report.json"
        report = run_baseline_comparison(
            eval_set=eval_set,
            primary_pipeline_fn=mock_primary,
            trivial_fn=mock_trivial,
            zeroshot_fn=mock_zeroshot,
            ragas_evaluator=mock_ragas,
            geval_scorer=mock_geval,
            output_path=str(out_path),
        )

        assert report.total_items == 2
        assert len(report.item_records) == 2

        for record in report.item_records:
            # Trivial responded (because broken/frozen triggered it)
            assert record.trivial.status == "success"
            # Faithfulness MUST BE None (undefined), not 0.0!
            assert record.trivial.faithfulness is None
            assert record.trivial.faithfulness != 0.0

            # Zeroshot also has None faithfulness
            assert record.zeroshot.faithfulness is None

            # Primary pipeline has computed faithfulness
            assert record.primary.faithfulness == 0.95

        # In summary, trivial and zeroshot faithfulness means must be None
        assert report.trivial_summary.faithfulness.mean is None
        assert report.zeroshot_summary.faithfulness.mean is None
        assert report.primary_summary.faithfulness.mean == 0.95

    def test_trivial_baseline_declined_excluded_from_scoring_and_counts_decline_rate(
        self, tmp_path
    ):
        """None responses from trivial baseline are excluded from scoring and counted in decline_rate."""
        eval_set = [
            EvalItem(query="My phone is broken"),  # matches keyword -> response
            EvalItem(query="I am a happy customer"),  # no keyword -> declines (None)
        ]

        report = run_baseline_comparison(
            eval_set=eval_set,
            primary_pipeline_fn=lambda q: "primary response",
            trivial_fn=trivial_baseline_response,
            zeroshot_fn=lambda q: "zeroshot response",
            output_path=str(tmp_path / "report.json"),
        )

        assert report.total_items == 2
        assert report.trivial_summary.decline_count == 1
        assert report.trivial_summary.decline_rate == 0.5

        item_declined = report.item_records[1]
        assert item_declined.trivial.status == "declined"
        assert item_declined.trivial.response is None
        assert item_declined.trivial.geval_score is None

    def test_zeroshot_baseline_error_tracked_as_baseline_error(self, tmp_path):
        """Zero-shot baseline failures are captured as 'baseline_error' and not scored as regular failures."""
        eval_set = [
            EvalItem(query="Query 1"),
            EvalItem(query="Query 2"),
        ]

        def failing_zeroshot(q):
            if q == "Query 2":
                return "[BASELINE_ERROR]: TimeoutException: LLM timed out"
            return "Valid response"

        report = run_baseline_comparison(
            eval_set=eval_set,
            primary_pipeline_fn=lambda q: "primary response",
            trivial_fn=lambda q: "static response",
            zeroshot_fn=failing_zeroshot,
            output_path=str(tmp_path / "report.json"),
        )

        assert report.zeroshot_summary.error_count == 1
        assert report.zeroshot_summary.error_rate == 0.5
        assert report.item_records[1].zeroshot.status == "baseline_error"
        assert report.item_records[0].zeroshot.status == "success"

    def test_primary_pipeline_uplift_calculated(self, tmp_path):
        """Comparator computes uplift of primary over trivial and zero-shot."""
        eval_set = [EvalItem(query="Query 1")]

        def mock_primary(q):
            return "Primary"

        def mock_geval(query, response, contexts):
            if response == "Primary":
                return 4.8
            if response == "Trivial":
                return 1.5
            if response == "Zeroshot":
                return 2.5
            return 1.0

        report = run_baseline_comparison(
            eval_set=eval_set,
            primary_pipeline_fn=mock_primary,
            trivial_fn=lambda q: "Trivial",
            zeroshot_fn=lambda q: "Zeroshot",
            geval_scorer=mock_geval,
            output_path=str(tmp_path / "report.json"),
        )

        assert report.primary_summary.geval_score.mean == 4.8
        assert report.trivial_summary.geval_score.mean == 1.5
        assert report.zeroshot_summary.geval_score.mean == 2.5

        assert report.uplift_over_trivial.geval_score_uplift == round(4.8 - 1.5, 4)
        assert report.uplift_over_zeroshot.geval_score_uplift == round(4.8 - 2.5, 4)
