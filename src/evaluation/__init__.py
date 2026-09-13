"""Evaluation module for RAG pipeline component metrics."""

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
from src.evaluation.ragas_evaluator import evaluate_ragas
from src.evaluation.relevance_judge import (
    judge_relevance,
    RelevanceJudgeError,
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
]

