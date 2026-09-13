"""Evaluation module for RAG pipeline component metrics."""

from src.evaluation.baseline_comparator import (
    ApproachScores,
    ApproachSummary,
    BaselineComparisonReport,
    ComparisonItemRecord,
    EvalItem,
    MetricSummary,
    UpliftSummary,
    persist_baseline_comparison_report,
    run_baseline_comparison,
)
from src.evaluation.baselines.trivial_baseline import trivial_baseline_response
from src.evaluation.baselines.zeroshot_baseline import (
    is_baseline_error,
    zero_shot_baseline_response,
)
from src.evaluation.claim_verification import (
    compute_faithfulness,
    extract_claims,
    verify_claims,
)
from src.evaluation.context_precision import compute_context_precision
from src.evaluation.context_recall import (
    compute_context_recall,
    decompose_reference,
)
from src.evaluation.failure_mode_tagger import (
    FailureMode,
    FailureModeCandidate,
    FailureModeRecord,
    log_failure_mode_record,
    tag_failure_candidates,
)
from src.evaluation.geval_prompt import build_geval_prompt
from src.evaluation.geval_rubrics import get_rubric, list_rubrics
from src.evaluation.geval_scorer import (
    LogprobToken,
    LogprobsPosition,
    LogprobsResponse,
    ScoreExtractionError,
    call_with_logprobs,
    compute_weighted_score,
    extract_score_distribution,
    run_geval,
)
from src.evaluation.golden_dataset import (
    GoldenExample,
    load_golden_set,
    stratified_sample,
)
from src.evaluation.judge_calibration import (
    JUDGE_DEPLOYMENT_KAPPA_THRESHOLD,
    CalibrationResult,
    compute_quadratic_weighted_kappa,
    discretize_scores,
    evaluate_judge_calibration,
)
from src.evaluation.krippendorff_agreement import compute_krippendorffs_alpha
from src.evaluation.ragas_evaluator import evaluate_ragas
from src.evaluation.relevance_judge import (
    RelevanceJudgeError,
    judge_relevance,
)
from src.evaluation.schema import GEvalResult, RAGASInput, RAGASResult

__all__ = [
    "RAGASInput",
    "RAGASResult",
    "GEvalResult",
    "RelevanceJudgeError",
    "judge_relevance",
    "compute_context_precision",
    "decompose_reference",
    "compute_context_recall",
    "extract_claims",
    "verify_claims",
    "compute_faithfulness",
    "evaluate_ragas",
    "get_rubric",
    "list_rubrics",
    "build_geval_prompt",
    "ScoreExtractionError",
    "LogprobToken",
    "LogprobsPosition",
    "LogprobsResponse",
    "call_with_logprobs",
    "extract_score_distribution",
    "compute_weighted_score",
    "run_geval",
    "GoldenExample",
    "load_golden_set",
    "stratified_sample",
    "compute_krippendorffs_alpha",
    "discretize_scores",
    "compute_quadratic_weighted_kappa",
    "evaluate_judge_calibration",
    "CalibrationResult",
    "JUDGE_DEPLOYMENT_KAPPA_THRESHOLD",
    "trivial_baseline_response",
    "zero_shot_baseline_response",
    "is_baseline_error",
    "EvalItem",
    "ApproachScores",
    "ComparisonItemRecord",
    "MetricSummary",
    "ApproachSummary",
    "UpliftSummary",
    "BaselineComparisonReport",
    "run_baseline_comparison",
    "persist_baseline_comparison_report",
    "FailureMode",
    "FailureModeCandidate",
    "FailureModeRecord",
    "tag_failure_candidates",
    "log_failure_mode_record",
]


