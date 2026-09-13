"""Orchestrator for RAGAS evaluation components with streaming persistence."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.evaluation.claim_verification import compute_faithfulness
from src.evaluation.context_precision import compute_context_precision
from src.evaluation.context_recall import compute_context_recall
from src.evaluation.schema import RAGASInput, RAGASResult

logger = logging.getLogger(__name__)

DEFAULT_EVAL_LOG_PATH = "data/eval/ragas_results.jsonl"


def _persist_evaluation_log(
    eval_record: dict,
    log_path: str = DEFAULT_EVAL_LOG_PATH,
) -> None:
    """Streamingly append evaluation record to JSONL without crashing on I/O error."""
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(eval_record) + "\n")
    except Exception as e:
        logger.error(
            "Failed to append RAGAS evaluation record to %s: %s. Continuing without crashing.",
            log_path,
            e,
        )


def evaluate_ragas(
    input_data: RAGASInput,
    llm_client: Any,
    hhem_model: Any,
    log_path: Optional[str] = DEFAULT_EVAL_LOG_PATH,
) -> RAGASResult:
    """Evaluate RAG pipeline output using decoupled RAGAS metrics.

    Metrics computed:
        - Context Precision (requires reference_answer, returns None if reference is missing)
        - Context Recall (requires reference_answer, returns None if reference is missing)
        - Faithfulness (does NOT require reference_answer; checks claims against retrieved contexts)

    Edge Cases Handled:
        - Zero retrieved contexts: precision=0.0 and recall=0.0 (if reference present) with explanatory
          notes; faithfulness=None (cannot verify claims against empty context).
        - reference_answer is None: precision and recall remain None; faithfulness is computed.
        - Zero claims in generated_response: faithfulness=None with explanatory note.
        - All claims unsupported: faithfulness=0.0 with claims > 0 (hallucination signal).
        - Streaming JSONL log write failures are swallowed and logged, preventing pipeline crashes.

    Args:
        input_data: RAGASInput containing query, retrieved contexts, response, and optional reference.
        llm_client: LLM client for relevance judging and fact decomposition.
        hhem_model: Model for claim verification (e.g. HHEM-2.1-Open).
        log_path: Path to append JSONL evaluation logs. Pass None to disable file logging.

    Returns:
        RAGASResult: Populated evaluation scores and diagnostics.
    """
    notes = []
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    num_claims: int = 0
    num_supported: int = 0

    has_contexts = bool(input_data.retrieved_contexts and len(input_data.retrieved_contexts) > 0)
    has_reference = bool(input_data.reference_answer and input_data.reference_answer.strip())

    # --- 1. Context Precision & Recall ---
    if not has_reference:
        notes.append("reference_answer is None; context_precision and context_recall skipped (None).")
    elif not has_contexts:
        # Reference is present, but no contexts were retrieved
        context_precision = 0.0
        context_recall = 0.0
        notes.append("Empty retrieved_contexts: context_precision and context_recall set to 0.0.")
    else:
        try:
            context_precision = compute_context_precision(
                reference_answer=input_data.reference_answer,
                retrieved_contexts=input_data.retrieved_contexts,
                llm_client=llm_client,
            )
        except Exception as e:
            logger.error("Error computing context precision: %s", e)
            notes.append(f"Context precision computation error: {e}")

        try:
            context_recall = compute_context_recall(
                reference_answer=input_data.reference_answer,
                retrieved_contexts=input_data.retrieved_contexts,
                llm_client=llm_client,
            )
        except Exception as e:
            logger.error("Error computing context recall: %s", e)
            notes.append(f"Context recall computation error: {e}")

    # --- 2. Faithfulness ---
    if not has_contexts:
        faithfulness = None
        notes.append("Empty retrieved_contexts: faithfulness cannot be computed (None).")
    else:
        try:
            faithfulness, num_claims, num_supported = compute_faithfulness(
                generated_response=input_data.generated_response,
                retrieved_contexts=input_data.retrieved_contexts,
                hhem_model=hhem_model,
            )
            if faithfulness is None and num_claims == 0:
                notes.append("No verifiable claims extracted from generated_response; faithfulness is undefined (None).")
        except Exception as e:
            logger.error("Error computing faithfulness: %s", e)
            notes.append(f"Faithfulness computation error: {e}")

    result = RAGASResult(
        context_precision=context_precision,
        context_recall=context_recall,
        faithfulness=faithfulness,
        num_claims_extracted=num_claims,
        num_claims_supported=num_supported,
        notes=notes,
    )

    # --- 3. Streaming Persistence ---
    if log_path:
        eval_record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": input_data.query,
            "has_reference": has_reference,
            "num_contexts": len(input_data.retrieved_contexts),
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "faithfulness": result.faithfulness,
            "num_claims_extracted": result.num_claims_extracted,
            "num_claims_supported": result.num_claims_supported,
            "notes": result.notes,
            "result": result.model_dump(),
        }
        _persist_evaluation_log(eval_record, log_path=log_path)

    return result
