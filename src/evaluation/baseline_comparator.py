"""Baseline comparator for comparing primary RAG pipeline against trivial and zero-shot baselines.

M5.P5.5.F1: Baselines & Failure Mode Tracking.
Evaluates the primary pipeline, a trivial keyword-based baseline, and an ungrounded
zero-shot LLM baseline across an evaluation set, computing uplift and tracking
decline/error rates with strict None-vs-0.0 faithfulness semantics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.evaluation.baselines.zeroshot_baseline import is_baseline_error

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_OUTPUT_PATH = "data/eval/baseline_comparison.json"


class EvalItem(BaseModel):
    """An individual item in the evaluation set."""

    query: str
    reference_answer: Optional[str] = None
    retrieved_contexts: List[str] = Field(default_factory=list)
    item_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApproachScores(BaseModel):
    """Metric scores for a single approach on a single item."""

    response: Optional[str] = None
    status: str = "success"  # "success", "declined", "baseline_error", "eval_error"
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    geval_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComparisonItemRecord(BaseModel):
    """Record of all three approaches evaluated on a single item."""

    item_id: Optional[str] = None
    query: str
    primary: ApproachScores
    trivial: ApproachScores
    zeroshot: ApproachScores


class MetricSummary(BaseModel):
    """Summary statistics for an evaluation metric."""

    mean: Optional[float] = None
    count: int = 0


class ApproachSummary(BaseModel):
    """Aggregate performance summary for an approach."""

    total_evaluated: int = 0
    decline_count: int = 0
    decline_rate: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    faithfulness: MetricSummary = Field(default_factory=MetricSummary)
    context_precision: MetricSummary = Field(default_factory=MetricSummary)
    context_recall: MetricSummary = Field(default_factory=MetricSummary)
    geval_score: MetricSummary = Field(default_factory=MetricSummary)


class UpliftSummary(BaseModel):
    """Performance uplift of primary pipeline over a baseline."""

    faithfulness_uplift: Optional[float] = None
    context_precision_uplift: Optional[float] = None
    context_recall_uplift: Optional[float] = None
    geval_score_uplift: Optional[float] = None


class BaselineComparisonReport(BaseModel):
    """Comprehensive baseline comparison report across all items and approaches."""

    total_items: int
    primary_summary: ApproachSummary
    trivial_summary: ApproachSummary
    zeroshot_summary: ApproachSummary
    uplift_over_trivial: UpliftSummary
    uplift_over_zeroshot: UpliftSummary
    item_records: List[ComparisonItemRecord] = Field(default_factory=list)


def _calculate_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _compute_approach_summary(
    scores_list: List[ApproachScores], total_items: int
) -> ApproachSummary:
    decline_count = sum(1 for s in scores_list if s.status == "declined")
    error_count = sum(1 for s in scores_list if s.status in ("baseline_error", "eval_error"))

    faith_vals = [s.faithfulness for s in scores_list if s.faithfulness is not None]
    prec_vals = [s.context_precision for s in scores_list if s.context_precision is not None]
    rec_vals = [s.context_recall for s in scores_list if s.context_recall is not None]
    geval_vals = [s.geval_score for s in scores_list if s.geval_score is not None]

    return ApproachSummary(
        total_evaluated=len(scores_list),
        decline_count=decline_count,
        decline_rate=round(decline_count / total_items, 4) if total_items > 0 else 0.0,
        error_count=error_count,
        error_rate=round(error_count / total_items, 4) if total_items > 0 else 0.0,
        faithfulness=MetricSummary(mean=_calculate_mean(faith_vals), count=len(faith_vals)),
        context_precision=MetricSummary(mean=_calculate_mean(prec_vals), count=len(prec_vals)),
        context_recall=MetricSummary(mean=_calculate_mean(rec_vals), count=len(rec_vals)),
        geval_score=MetricSummary(mean=_calculate_mean(geval_vals), count=len(geval_vals)),
    )


def _compute_uplift(primary: ApproachSummary, baseline: ApproachSummary) -> UpliftSummary:
    def _diff(p_mean: Optional[float], b_mean: Optional[float]) -> Optional[float]:
        if p_mean is not None and b_mean is not None:
            return round(p_mean - b_mean, 4)
        return None

    return UpliftSummary(
        faithfulness_uplift=_diff(primary.faithfulness.mean, baseline.faithfulness.mean),
        context_precision_uplift=_diff(primary.context_precision.mean, baseline.context_precision.mean),
        context_recall_uplift=_diff(primary.context_recall.mean, baseline.context_recall.mean),
        geval_score_uplift=_diff(primary.geval_score.mean, baseline.geval_score.mean),
    )


def persist_baseline_comparison_report(
    report: BaselineComparisonReport,
    output_path: str = DEFAULT_COMPARISON_OUTPUT_PATH,
) -> None:
    """Non-crashing persistence of baseline comparison report to JSON."""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        logger.info(f"Persisted baseline comparison report to {output_path}")
    except Exception as exc:
        logger.warning(
            f"Failed to persist baseline comparison report to {output_path}: {exc}",
            exc_info=True,
        )


def _evaluate_ragas_safe(
    evaluator: Optional[Callable],
    query: str,
    response: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Safely invokes RAGAS evaluator if provided, extracting metrics dict."""
    if not evaluator:
        return {"faithfulness": None, "context_precision": None, "context_recall": None}

    try:
        res = evaluator(
            query=query,
            response=response,
            contexts=contexts,
            reference_answer=reference_answer,
        )
        if isinstance(res, dict):
            return {
                "faithfulness": res.get("faithfulness"),
                "context_precision": res.get("context_precision"),
                "context_recall": res.get("context_recall"),
            }
        # If Pydantic model
        return {
            "faithfulness": getattr(res, "faithfulness", None),
            "context_precision": getattr(res, "context_precision", None),
            "context_recall": getattr(res, "context_recall", None),
        }
    except Exception as exc:
        logger.warning(f"RAGAS evaluation failed safely: {exc}")
        return {"faithfulness": None, "context_precision": None, "context_recall": None}


def _evaluate_geval_safe(
    scorer: Optional[Callable],
    query: str,
    response: str,
    contexts: List[str],
) -> Optional[float]:
    """Safely invokes G-Eval scorer if provided."""
    if not scorer:
        return None

    try:
        res = scorer(query=query, response=response, contexts=contexts)
        if isinstance(res, (int, float)):
            return float(res)
        if isinstance(res, dict):
            return res.get("score") or res.get("geval_score")
        return getattr(res, "score", getattr(res, "geval_score", None))
    except Exception as exc:
        logger.warning(f"G-Eval scoring failed safely: {exc}")
        return None


def run_baseline_comparison(
    eval_set: List[Union[EvalItem, Dict[str, Any]]],
    primary_pipeline_fn: Callable,
    trivial_fn: Callable,
    zeroshot_fn: Callable,
    ragas_evaluator: Optional[Callable] = None,
    geval_scorer: Optional[Callable] = None,
    output_path: str = DEFAULT_COMPARISON_OUTPUT_PATH,
) -> BaselineComparisonReport:
    """Runs primary pipeline, trivial baseline, and zero-shot baseline on eval_set.

    Args:
        eval_set: List of evaluation items (EvalItem or dict). Must not be empty.
        primary_pipeline_fn: Callable returning pipeline output (str, dict, or object).
        trivial_fn: Callable returning trivial baseline response (str or None).
        zeroshot_fn: Callable returning zero-shot baseline response (str).
        ragas_evaluator: Optional callable for RAGAS metrics.
        geval_scorer: Optional callable for G-Eval metric.
        output_path: Destination path for comparison JSON output.

    Returns:
        BaselineComparisonReport containing per-approach metrics, item records, and uplift.

    Raises:
        ValueError: If eval_set is empty.
    """
    if not eval_set:
        raise ValueError("eval_set cannot be empty for baseline comparison")

    # Normalize eval_set items
    items: List[EvalItem] = []
    for raw in eval_set:
        if isinstance(raw, EvalItem):
            items.append(raw)
        elif isinstance(raw, dict):
            items.append(
                EvalItem(
                    query=raw.get("query") or raw.get("tweet_text") or raw.get("text", ""),
                    reference_answer=raw.get("reference_answer"),
                    retrieved_contexts=raw.get("retrieved_contexts", []),
                    item_id=raw.get("item_id") or raw.get("id"),
                    metadata=raw.get("metadata", {}),
                )
            )
        else:
            items.append(
                EvalItem(
                    query=getattr(raw, "query", getattr(raw, "tweet_text", str(raw))),
                    reference_answer=getattr(raw, "reference_answer", None),
                    retrieved_contexts=getattr(raw, "retrieved_contexts", []),
                    item_id=getattr(raw, "item_id", None),
                )
            )

    total_items = len(items)
    primary_scores_list: List[ApproachScores] = []
    trivial_scores_list: List[ApproachScores] = []
    zeroshot_scores_list: List[ApproachScores] = []
    item_records: List[ComparisonItemRecord] = []

    for item in items:
        # --- 1. Primary Pipeline ---
        p_res = primary_pipeline_fn(item.query)
        if isinstance(p_res, dict):
            p_text = p_res.get("response") or p_res.get("answer", "")
            p_contexts = p_res.get("retrieved_contexts") or item.retrieved_contexts
        elif hasattr(p_res, "response"):
            p_text = getattr(p_res, "response")
            p_contexts = getattr(p_res, "retrieved_contexts", item.retrieved_contexts)
        else:
            p_text = str(p_res)
            p_contexts = item.retrieved_contexts

        p_ragas = _evaluate_ragas_safe(
            ragas_evaluator,
            item.query,
            p_text,
            p_contexts,
            item.reference_answer,
        )
        p_geval = _evaluate_geval_safe(geval_scorer, item.query, p_text, p_contexts)
        p_scores = ApproachScores(
            response=p_text,
            status="success",
            faithfulness=p_ragas.get("faithfulness"),
            context_precision=p_ragas.get("context_precision"),
            context_recall=p_ragas.get("context_recall"),
            geval_score=p_geval,
        )
        primary_scores_list.append(p_scores)

        # --- 2. Trivial Baseline ---
        t_res = trivial_fn(item.query)
        if t_res is None:
            # Declined to respond: Do NOT score. Faithfulness and G-Eval are omitted/None.
            t_scores = ApproachScores(
                response=None,
                status="declined",
                faithfulness=None,
                context_precision=None,
                context_recall=None,
                geval_score=None,
            )
        else:
            # Trivial baseline retrieved zero context.
            # Faithfulness MUST BE None (undefined, zero context to verify against), NEVER 0.0.
            t_geval = _evaluate_geval_safe(geval_scorer, item.query, t_res, [])
            t_scores = ApproachScores(
                response=t_res,
                status="success",
                faithfulness=None,  # STRICT: Faithfulness is undefined with 0 retrieved contexts
                context_precision=0.0,
                context_recall=0.0,
                geval_score=t_geval,
            )
        trivial_scores_list.append(t_scores)

        # --- 3. Zero-Shot Baseline ---
        z_res = zeroshot_fn(item.query)
        if is_baseline_error(z_res):
            # Record as baseline error, distinct from low-quality response. Do NOT score.
            z_scores = ApproachScores(
                response=z_res,
                status="baseline_error",
                faithfulness=None,
                context_precision=None,
                context_recall=None,
                geval_score=None,
            )
        else:
            # Zero-shot baseline has no retrieved contexts.
            # Faithfulness is undefined (None).
            z_geval = _evaluate_geval_safe(geval_scorer, item.query, z_res, [])
            z_scores = ApproachScores(
                response=z_res,
                status="success",
                faithfulness=None,  # Zero-shot has no retrieval context
                context_precision=0.0,
                context_recall=0.0,
                geval_score=z_geval,
            )
        zeroshot_scores_list.append(z_scores)

        item_records.append(
            ComparisonItemRecord(
                item_id=item.item_id,
                query=item.query,
                primary=p_scores,
                trivial=t_scores,
                zeroshot=z_scores,
            )
        )

    # Compute Summaries
    primary_summary = _compute_approach_summary(primary_scores_list, total_items)
    trivial_summary = _compute_approach_summary(trivial_scores_list, total_items)
    zeroshot_summary = _compute_approach_summary(zeroshot_scores_list, total_items)

    # Compute Uplifts
    uplift_over_trivial = _compute_uplift(primary_summary, trivial_summary)
    uplift_over_zeroshot = _compute_uplift(primary_summary, zeroshot_summary)

    report = BaselineComparisonReport(
        total_items=total_items,
        primary_summary=primary_summary,
        trivial_summary=trivial_summary,
        zeroshot_summary=zeroshot_summary,
        uplift_over_trivial=uplift_over_trivial,
        uplift_over_zeroshot=uplift_over_zeroshot,
        item_records=item_records,
    )

    persist_baseline_comparison_report(report, output_path)
    return report
