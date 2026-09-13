"""G-Eval Logit-Weighted Holistic Scorer.

Computes continuous, logit-weighted evaluation scores for subjective dimensions
(brand tone, empathy, structural coherence) using Chain-of-Thought reasoning
and token probability extraction at the score-token position.
"""

import datetime
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.evaluation.geval_prompt import build_geval_prompt
from src.evaluation.geval_rubrics import get_rubric
from src.evaluation.schema import GEvalResult

logger = logging.getLogger(__name__)

DEFAULT_GEVAL_LOG_PATH = "data/eval/geval_results.jsonl"
VALID_SCORE_DIGITS = {"1", "2", "3", "4", "5"}
LOW_CONFIDENCE_THRESHOLD = 0.5


class ScoreExtractionError(Exception):
    """Raised when score token extraction fails or anchor is missing/misaligned."""

    pass


class LogprobToken(BaseModel):
    """Candidate token and its log-probability."""

    token: str
    logprob: float


class LogprobsPosition(BaseModel):
    """Top log-probabilities for a single output token position."""

    token: str
    top_logprobs: List[LogprobToken] = Field(default_factory=list)


class LogprobsResponse(BaseModel):
    """Response containing output text and optional token logprob distributions."""

    text: str
    logprobs: Optional[List[LogprobsPosition]] = None


def _persist_geval_log(
    result: GEvalResult,
    log_path: Optional[str] = DEFAULT_GEVAL_LOG_PATH,
) -> None:
    """Streamingly append G-Eval evaluation record to JSONL without crashing on I/O error."""
    if not log_path:
        return
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = result.model_dump()
        record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(
            "Failed to append G-Eval record to %s: %s. Continuing without crashing.",
            log_path,
            e,
        )


def call_with_logprobs(
    prompt: str,
    llm_client: Any,
    top_logprobs: int = 10,
) -> LogprobsResponse:
    """Request text generation with top-N log-probabilities from LLM client.

    Gracefully detects when logprobs are unsupported by the client/model and returns
    a LogprobsResponse with logprobs=None.

    Args:
        prompt: Evaluation prompt string.
        llm_client: LLM client object, mock, or callable.
        top_logprobs: Number of top token alternatives to request per position.

    Returns:
        LogprobsResponse containing text and optional token logprob sequence.
    """
    if llm_client is None:
        raise ValueError("llm_client must be provided to call_with_logprobs.")

    try:
        if hasattr(llm_client, "call_with_logprobs"):
            raw = llm_client.call_with_logprobs(prompt=prompt, top_logprobs=top_logprobs)
        elif hasattr(llm_client, "generate_with_logprobs"):
            raw = llm_client.generate_with_logprobs(prompt=prompt, top_logprobs=top_logprobs)
        elif hasattr(llm_client, "generate"):
            raw = llm_client.generate(prompt=prompt, logprobs=True, top_logprobs=top_logprobs)
        elif callable(llm_client):
            raw = llm_client(prompt)
        else:
            raise AttributeError(f"Unsupported llm_client type: {type(llm_client)}")

        # Parse raw output into LogprobsResponse
        if isinstance(raw, LogprobsResponse):
            return raw

        if isinstance(raw, dict):
            return LogprobsResponse.model_validate(raw)

        if isinstance(raw, str):
            # Client only returned a raw string; logprobs unavailable
            return LogprobsResponse(text=raw, logprobs=None)

        # Handle object with .text and optional .logprobs
        if hasattr(raw, "text"):
            text = raw.text
            logprobs = getattr(raw, "logprobs", None)
            if logprobs is not None:
                # If logprobs are dicts or objects, normalize to list of LogprobsPosition
                parsed_positions = []
                for p in logprobs:
                    if isinstance(p, LogprobsPosition):
                        parsed_positions.append(p)
                    elif isinstance(p, dict):
                        parsed_positions.append(LogprobsPosition.model_validate(p))
                    elif hasattr(p, "token") and hasattr(p, "top_logprobs"):
                        top_tokens = [
                            LogprobToken(
                                token=getattr(t, "token", str(t)),
                                logprob=getattr(t, "logprob", 0.0),
                            )
                            for t in p.top_logprobs
                        ]
                        parsed_positions.append(
                            LogprobsPosition(token=p.token, top_logprobs=top_tokens)
                        )
                return LogprobsResponse(text=text, logprobs=parsed_positions)
            return LogprobsResponse(text=text, logprobs=None)

        return LogprobsResponse(text=str(raw), logprobs=None)

    except (TypeError, NotImplementedError, AttributeError) as e:
        logger.warning(
            "LLM client does not support logprobs API (%s). Falling back to text output.",
            e,
        )
        # Attempt fallback to simple generation if possible
        if callable(llm_client):
            try:
                plain_text = llm_client(prompt)
                return LogprobsResponse(text=str(plain_text), logprobs=None)
            except Exception:
                pass
        return LogprobsResponse(text="", logprobs=None)


def extract_score_distribution(response: LogprobsResponse) -> Dict[str, float]:
    """Locate score token following 'Final Score:' and extract valid digit probability mass.

    Anchors strictly to the LAST occurrence of 'Final Score:' in the output text to guard
    against prompt echoes or criteria mentioning 'Final Score:'. Handles both bare ('5')
    and space-prefixed (' 5') digit tokens.

    Args:
        response: LogprobsResponse with output text and logprob positions.

    Returns:
        Dict mapping valid score digits ('1' to '5') to unnormalized linear probabilities.

    Raises:
        ScoreExtractionError: If anchor is missing, position cannot be resolved,
            or no valid digit tokens appear in top_logprobs.
    """
    if not response.text:
        raise ScoreExtractionError("Cannot extract score distribution from empty response text.")

    if not response.logprobs:
        raise ScoreExtractionError("Cannot extract score distribution: logprobs are None or empty.")

    # Locate the LAST occurrence of 'Final Score:' in the response text
    matches = list(re.finditer(r"Final\s+Score:", response.text, re.IGNORECASE))
    if not matches:
        raise ScoreExtractionError(
            "Could not locate 'Final Score:' anchor in response text. Model failed strict formatting."
        )

    last_match = matches[-1]
    colon_end_char = last_match.end()

    # Reconstruct character positions across the token sequence
    current_char = 0
    target_pos_idx: Optional[int] = None

    for i, pos in enumerate(response.logprobs):
        tok_len = len(pos.token)
        tok_start = current_char
        tok_end = current_char + tok_len
        current_char = tok_end

        # We look for the first non-whitespace token starting at or after the colon
        if tok_end > last_match.start():
            if tok_start >= colon_end_char and pos.token.strip():
                target_pos_idx = i
                break

    if target_pos_idx is None:
        raise ScoreExtractionError(
            "Could not locate valid score token position following 'Final Score:'."
        )

    score_pos = response.logprobs[target_pos_idx]
    distribution: Dict[str, float] = {}

    for cand in score_pos.top_logprobs:
        cleaned = cand.token.strip()
        if cleaned in VALID_SCORE_DIGITS:
            linear_p = math.exp(cand.logprob)
            distribution[cleaned] = distribution.get(cleaned, 0.0) + linear_p

    if not distribution:
        raise ScoreExtractionError(
            f"No valid score tokens ('1'-'5') found in top_logprobs at token position '{score_pos.token}'."
        )

    return distribution


def compute_weighted_score(distribution: Dict[str, float]) -> Tuple[float, float]:
    """Compute probability-weighted continuous score renormalized over valid token mass.

    Formula:
        weighted_score = sum(int(token) * prob) / sum(prob)
        captured_probability_mass = sum(prob)

    Renormalization note:
        The raw top_logprobs list may contain irrelevant or non-digit tokens that were
        filtered out. Renormalizing strictly over the valid 1-5 captured probability mass
        ensures the expected score reflects the model's conditional belief across valid
        rubric ratings without dilution.

    Args:
        distribution: Dict mapping digit strings ('1'-'5') to linear probabilities.

    Returns:
        Tuple of (weighted_score, captured_probability_mass).

    Raises:
        ScoreExtractionError: If distribution is empty or total captured mass is zero.
    """
    if not distribution:
        raise ScoreExtractionError("Cannot compute weighted score from empty distribution.")

    captured_mass = sum(distribution.values())
    if captured_mass <= 0.0:
        raise ScoreExtractionError("Captured probability mass is non-positive.")

    weighted_sum = sum(int(digit) * prob for digit, prob in distribution.items())
    weighted_score = weighted_sum / captured_mass

    return weighted_score, captured_mass


def run_geval(
    rubric_name: str,
    generated_response: str,
    context: Optional[str] = None,
    llm_client: Any = None,
    log_path: Optional[str] = DEFAULT_GEVAL_LOG_PATH,
) -> GEvalResult:
    """Orchestrate complete G-Eval evaluation workflow.

    Workflow:
        1. Validate rubric name (raises ValueError if unknown).
        2. Detect degenerate inputs (empty response or context) and record diagnostic notes.
        3. Build CoT-enforcing prompt with anchored rubric definitions.
        4. Query LLM client requesting token logprobs.
        5. Extract score distribution at anchor position, or fallback to integer parsing if logprobs unavailable.
        6. Compute continuous logit-weighted score and assess confidence.
        7. Persist result to JSONL log streamingly without crashing.

    Args:
        rubric_name: Name of rubric ('brand_tone', 'empathy', 'structural_coherence').
        generated_response: Generated text to evaluate.
        context: Optional conversational or retrieval context.
        llm_client: LLM client supporting logprobs or fallback string output.
        log_path: Path to append JSONL evaluation records.

    Returns:
        GEvalResult model containing continuous score, reasoning, and metadata.

    Raises:
        ValueError: If rubric_name is unknown.
        ScoreExtractionError: If score extraction fails and fallback cannot parse an integer.
    """
    # 1. Validate rubric name
    get_rubric(rubric_name)

    notes: List[str] = []

    # 2. Check degenerate inputs
    if not generated_response or not generated_response.strip():
        notes.append("Degenerate input: generated_response is empty.")
        logger.warning("Evaluating empty generated_response for rubric '%s'.", rubric_name)

    if context is not None and not context.strip():
        notes.append("Degenerate input: context provided is empty string.")

    # 3. Build prompt
    prompt = build_geval_prompt(
        rubric_name=rubric_name,
        generated_response=generated_response,
        context=context,
    )

    # 4. Query LLM
    response = call_with_logprobs(prompt=prompt, llm_client=llm_client)

    # Extract CoT text (everything prior to final score anchor)
    anchor_matches = list(re.finditer(r"Final\s+Score:", response.text, re.IGNORECASE))
    if anchor_matches:
        cot_reasoning_text = response.text[: anchor_matches[-1].start()].strip()
    else:
        cot_reasoning_text = response.text.strip()

    # 5. Extraction & Scoring
    logprobs_unavailable = False
    low_confidence = False

    if response.logprobs is None:
        # Fallback: parse plain "Final Score: X" integer directly from text
        logprobs_unavailable = True
        notes.append("Logprobs unavailable from provider; fell back to integer parsing.")

        matches = list(re.finditer(r"Final\s+Score:\s*([1-5])", response.text, re.IGNORECASE))
        if not matches:
            raise ScoreExtractionError(
                "Logprobs unavailable and could not extract integer score 'Final Score: X' (1-5) from model output."
            )

        integer_score = int(matches[-1].group(1))
        weighted_score = float(integer_score)
        captured_probability_mass = 1.0
    else:
        # Logprobs available: extract distribution and compute weighted expectation
        distribution = extract_score_distribution(response)
        weighted_score, captured_mass = compute_weighted_score(distribution)
        captured_probability_mass = captured_mass

        if captured_probability_mass < LOW_CONFIDENCE_THRESHOLD:
            low_confidence = True
            logger.warning(
                "Low confidence G-Eval score for rubric '%s': captured probability mass is %.4f (< %.2f).",
                rubric_name,
                captured_probability_mass,
                LOW_CONFIDENCE_THRESHOLD,
            )
            notes.append(
                f"Low confidence: captured probability mass {captured_probability_mass:.4f} is below "
                f"{LOW_CONFIDENCE_THRESHOLD} threshold."
            )

    result = GEvalResult(
        rubric_name=rubric_name,
        weighted_score=round(weighted_score, 4),
        captured_probability_mass=round(captured_probability_mass, 4),
        cot_reasoning_text=cot_reasoning_text,
        low_confidence=low_confidence,
        logprobs_unavailable=logprobs_unavailable,
        notes=notes,
    )

    # 6. Streamingly persist to JSONL
    _persist_geval_log(result, log_path=log_path)

    return result
