"""Unit tests for G-Eval logit-weighted holistic scorer."""

import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.geval_scorer import (
    DEFAULT_GEVAL_LOG_PATH,
    LogprobsPosition,
    LogprobsResponse,
    LogprobToken,
    ScoreExtractionError,
    _persist_geval_log,
    call_with_logprobs,
    compute_weighted_score,
    extract_score_distribution,
    run_geval,
)
from src.evaluation.schema import GEvalResult


# ============================================================================
# 1. Weighted Score Computation Tests
# ============================================================================


def test_compute_weighted_score_exact_math():
    """Verify exact weighted score computation against manually-calculated distribution."""
    # Distribution: {"5": 0.6, "4": 0.3, "3": 0.1}
    # Expected weighted sum = 5*0.6 + 4*0.3 + 3*0.1 = 3.0 + 1.2 + 0.3 = 4.5
    # Captured mass = 0.6 + 0.3 + 0.1 = 1.0
    # Expected score = 4.5 / 1.0 = 4.5
    distribution = {"5": 0.6, "4": 0.3, "3": 0.1}
    weighted_score, captured_mass = compute_weighted_score(distribution)

    assert math.isclose(weighted_score, 4.5, rel_tol=1e-6)
    assert math.isclose(captured_mass, 1.0, rel_tol=1e-6)


def test_compute_weighted_score_renormalization():
    """Verify weighted score is properly renormalized over captured probability mass."""
    # Distribution: {"5": 0.4, "4": 0.2} (captured mass = 0.6 < 1.0)
    # Expected weighted sum = 5*0.4 + 4*0.2 = 2.0 + 0.8 = 2.8
    # Expected renormalized score = 2.8 / 0.6 = 4.6666666667
    distribution = {"5": 0.4, "4": 0.2}
    weighted_score, captured_mass = compute_weighted_score(distribution)

    assert math.isclose(captured_mass, 0.6, rel_tol=1e-6)
    assert math.isclose(weighted_score, 2.8 / 0.6, rel_tol=1e-6)


def test_compute_weighted_score_empty_or_zero_raises_error():
    """Verify ScoreExtractionError is raised on empty distribution or non-positive mass."""
    with pytest.raises(ScoreExtractionError, match="Cannot compute weighted score from empty distribution"):
        compute_weighted_score({})

    with pytest.raises(ScoreExtractionError, match="Captured probability mass is non-positive"):
        compute_weighted_score({"5": 0.0, "4": 0.0})


# ============================================================================
# 2. Extract Score Distribution & Token Handling Tests
# ============================================================================


def _create_mock_logprobs_response(
    cot_text: str = "The response is clear and polite.\n\n",
    anchor: str = "Final Score: ",
    score_token: str = "5",
    top_logprobs: List[tuple] = None,
) -> LogprobsResponse:
    """Helper to assemble a LogprobsResponse with reconstructed token positions."""
    if top_logprobs is None:
        top_logprobs = [("5", math.log(0.7)), ("4", math.log(0.2)), ("3", math.log(0.1))]

    positions: List[LogprobsPosition] = []

    # Tokenize cot_text by words and spaces for realistic token positions
    for word in cot_text.split(" "):
        positions.append(LogprobsPosition(token=word + " ", top_logprobs=[]))

    # Anchor tokens: "Final ", "Score: "
    positions.append(LogprobsPosition(token="Final ", top_logprobs=[]))
    positions.append(LogprobsPosition(token="Score: ", top_logprobs=[]))

    # Target score token position
    cand_tokens = [LogprobToken(token=tok, logprob=lp) for tok, lp in top_logprobs]
    positions.append(LogprobsPosition(token=score_token, top_logprobs=cand_tokens))

    full_text = "".join(p.token for p in positions)
    return LogprobsResponse(text=full_text, logprobs=positions)


def test_extract_score_distribution_bare_digits():
    """Verify extraction filters correctly to 1-5 digits and converts logprobs to linear probabilities."""
    top_candidates = [
        ("5", math.log(0.65)),
        ("4", math.log(0.25)),
        ("3", math.log(0.05)),
        ("Note", math.log(0.03)),  # Non-digit token should be excluded
        ("10", math.log(0.02)),  # Out-of-bounds digit token should be excluded
    ]
    response = _create_mock_logprobs_response(top_logprobs=top_candidates)
    dist = extract_score_distribution(response)

    assert set(dist.keys()) == {"5", "4", "3"}
    assert math.isclose(dist["5"], 0.65, rel_tol=1e-5)
    assert math.isclose(dist["4"], 0.25, rel_tol=1e-5)
    assert math.isclose(dist["3"], 0.05, rel_tol=1e-5)


def test_extract_score_distribution_space_prefixed_tokens():
    """Verify bare and space-prefixed tokens (e.g. '5' and ' 5') are merged properly."""
    top_candidates = [
        (" 5", math.log(0.50)),
        ("5", math.log(0.15)),
        (" 4", math.log(0.20)),
        ("3", math.log(0.10)),
    ]
    response = _create_mock_logprobs_response(top_logprobs=top_candidates)
    dist = extract_score_distribution(response)

    # " 5" and "5" should sum to 0.65
    assert math.isclose(dist["5"], 0.65, rel_tol=1e-5)
    assert math.isclose(dist["4"], 0.20, rel_tol=1e-5)
    assert math.isclose(dist["3"], 0.10, rel_tol=1e-5)


def test_extract_score_distribution_multiple_final_score_anchors():
    """Verify extraction anchors strictly to the LAST occurrence of 'Final Score:'."""
    # Suppose CoT text mentions: "Criteria says Final Score: 1 if toxic, but here Final Score: 4"
    positions = [
        LogprobsPosition(token="Mentioned Final Score: 1 in reasoning. But actual ", top_logprobs=[]),
        LogprobsPosition(token="Final ", top_logprobs=[]),
        LogprobsPosition(token="Score: ", top_logprobs=[]),
        LogprobsPosition(
            token="4",
            top_logprobs=[
                LogprobToken(token="4", logprob=math.log(0.80)),
                LogprobToken(token="5", logprob=math.log(0.15)),
            ],
        ),
    ]
    full_text = "".join(p.token for p in positions)
    response = LogprobsResponse(text=full_text, logprobs=positions)

    dist = extract_score_distribution(response)
    assert "4" in dist
    assert math.isclose(dist["4"], 0.80, rel_tol=1e-5)
    assert math.isclose(dist["5"], 0.15, rel_tol=1e-5)


def test_extract_score_distribution_missing_anchor_raises_score_extraction_error():
    """Verify ScoreExtractionError when 'Final Score:' is missing from output."""
    response = LogprobsResponse(
        text="Reasoning completed. The score is 5.",
        logprobs=[
            LogprobsPosition(token="Reasoning completed. The score is 5.", top_logprobs=[]),
        ],
    )
    with pytest.raises(ScoreExtractionError, match="Could not locate 'Final Score:' anchor"):
        extract_score_distribution(response)


def test_extract_score_distribution_no_valid_digits_raises_score_extraction_error():
    """Verify ScoreExtractionError when top_logprobs contains no tokens from '1'-'5'."""
    top_candidates = [
        ("Excellent", math.log(0.60)),
        ("High", math.log(0.30)),
    ]
    response = _create_mock_logprobs_response(top_logprobs=top_candidates)
    with pytest.raises(ScoreExtractionError, match="No valid score tokens"):
        extract_score_distribution(response)


def test_extract_score_distribution_empty_response_raises_error():
    """Verify ScoreExtractionError when response text or logprobs are empty."""
    with pytest.raises(ScoreExtractionError, match="empty response text"):
        extract_score_distribution(LogprobsResponse(text="", logprobs=[]))

    with pytest.raises(ScoreExtractionError, match="logprobs are None or empty"):
        extract_score_distribution(LogprobsResponse(text="Final Score: 5", logprobs=None))


# ============================================================================
# 3. call_with_logprobs Tests
# ============================================================================


def test_call_with_logprobs_client_with_call_with_logprobs():
    """Verify call_with_logprobs invokes client.call_with_logprobs if available."""
    mock_client = MagicMock()
    mock_resp = LogprobsResponse(text="Output text", logprobs=None)
    mock_client.call_with_logprobs.return_value = mock_resp

    result = call_with_logprobs("Test prompt", mock_client, top_logprobs=5)
    mock_client.call_with_logprobs.assert_called_once_with(prompt="Test prompt", top_logprobs=5)
    assert result == mock_resp


def test_call_with_logprobs_client_fallback_to_string():
    """Verify fallback to plain text LogprobsResponse when client returns a string."""
    mock_client = MagicMock(return_value="Plain string generation")
    del mock_client.call_with_logprobs
    del mock_client.generate_with_logprobs
    del mock_client.generate

    result = call_with_logprobs("Test prompt", mock_client)
    assert result.text == "Plain string generation"
    assert result.logprobs is None


def test_call_with_logprobs_none_client_raises_value_error():
    """Verify ValueError is raised if llm_client is None."""
    with pytest.raises(ValueError, match="llm_client must be provided"):
        call_with_logprobs("Test prompt", None)


# ============================================================================
# 4. run_geval Workflow & Edge Cases Tests
# ============================================================================


def test_run_geval_success_with_logprobs(tmp_path: Path):
    """Verify end-to-end G-Eval run with logprobs produces continuous expectation score."""
    log_file = tmp_path / "geval_results.jsonl"
    mock_resp = _create_mock_logprobs_response(
        cot_text="Tone is warm and professional.\n",
        anchor="Final Score: ",
        score_token="5",
        top_logprobs=[("5", math.log(0.6)), ("4", math.log(0.3)), ("3", math.log(0.1))],
    )

    mock_client = MagicMock()
    mock_client.call_with_logprobs.return_value = mock_resp

    result = run_geval(
        rubric_name="brand_tone",
        generated_response="Thank you for your patience while we investigate.",
        context="Customer inquiry regarding billing.",
        llm_client=mock_client,
        log_path=str(log_file),
    )

    assert isinstance(result, GEvalResult)
    assert result.rubric_name == "brand_tone"
    assert result.weighted_score == 4.5
    assert result.captured_probability_mass == 1.0
    assert result.low_confidence is False
    assert result.logprobs_unavailable is False
    assert "Tone is warm and professional." in result.cot_reasoning_text
    assert log_file.exists()


def test_run_geval_low_confidence_flagging(tmp_path: Path):
    """Verify low_confidence=True is set when captured probability mass is < 0.5."""
    log_file = tmp_path / "geval_results.jsonl"
    # Mass = 0.3 (below 0.5 threshold)
    mock_resp = _create_mock_logprobs_response(
        top_logprobs=[
            ("4", math.log(0.2)),
            ("5", math.log(0.1)),
            ("Unrelated", math.log(0.7)),
        ]
    )

    mock_client = MagicMock()
    mock_client.call_with_logprobs.return_value = mock_resp

    result = run_geval(
        rubric_name="empathy",
        generated_response="I understand this is frustrating.",
        llm_client=mock_client,
        log_path=str(log_file),
    )

    assert result.low_confidence is True
    assert math.isclose(result.captured_probability_mass, 0.3, rel_tol=1e-4)
    # Expected weighted sum = 4*0.2 + 5*0.1 = 0.8 + 0.5 = 1.3
    # Expected weighted score = 1.3 / 0.3 = 4.3333
    assert math.isclose(result.weighted_score, 1.3 / 0.3, rel_tol=1e-4)
    assert any("Low confidence" in note for note in result.notes)


def test_run_geval_logprobs_unavailable_fallback(tmp_path: Path):
    """Verify fallback to direct integer parsing when logprobs are None."""
    log_file = tmp_path / "geval_results.jsonl"
    plain_output = "The response is coherent and structured.\n\nFinal Score: 4"

    mock_client = MagicMock(return_value=plain_output)
    del mock_client.call_with_logprobs
    del mock_client.generate_with_logprobs
    del mock_client.generate

    result = run_geval(
        rubric_name="structural_coherence",
        generated_response="Step 1: check. Step 2: verify.",
        llm_client=mock_client,
        log_path=str(log_file),
    )

    assert result.logprobs_unavailable is True
    assert result.weighted_score == 4.0
    assert result.captured_probability_mass == 1.0
    assert result.low_confidence is False
    assert any("Logprobs unavailable" in note for note in result.notes)


def test_run_geval_logprobs_unavailable_parse_failure_raises_error():
    """Verify ScoreExtractionError if logprobs are unavailable and no integer score is found."""
    mock_client = MagicMock(return_value="The response is okay, but I won't give a score.")
    del mock_client.call_with_logprobs
    del mock_client.generate_with_logprobs
    del mock_client.generate

    with pytest.raises(ScoreExtractionError, match="Logprobs unavailable and could not extract integer score"):
        run_geval(
            rubric_name="structural_coherence",
            generated_response="Step 1: check.",
            llm_client=mock_client,
            log_path=None,
        )


def test_run_geval_degenerate_inputs(tmp_path: Path):
    """Verify empty response or empty context executes without crashing and logs notes."""
    log_file = tmp_path / "geval_results.jsonl"
    mock_resp = _create_mock_logprobs_response(
        cot_text="Empty response lacks empathy and structure.\n",
        score_token="1",
        top_logprobs=[("1", math.log(0.95))],
    )
    mock_client = MagicMock()
    mock_client.call_with_logprobs.return_value = mock_resp

    result = run_geval(
        rubric_name="empathy",
        generated_response="",  # Degenerate empty response
        context="",  # Degenerate empty context
        llm_client=mock_client,
        log_path=str(log_file),
    )

    assert result.weighted_score == 1.0
    assert any("Degenerate input: generated_response is empty" in note for note in result.notes)
    assert any("Degenerate input: context provided is empty string" in note for note in result.notes)


def test_run_geval_unknown_rubric_raises_value_error():
    """Verify ValueError is raised if an unregistered rubric name is passed."""
    with pytest.raises(ValueError, match="Unknown rubric"):
        run_geval(
            rubric_name="speed_and_efficiency",
            generated_response="Some response",
            llm_client=MagicMock(),
        )


def test_persist_geval_log_io_error_non_crashing():
    """Verify logging failure does not crash the evaluation execution."""
    result = GEvalResult(
        rubric_name="brand_tone",
        weighted_score=4.5,
        captured_probability_mass=1.0,
        cot_reasoning_text="Reasoning",
        low_confidence=False,
    )

    # Patch open to simulate an OS permission or disk error
    with patch("builtins.open", side_effect=OSError("Disk write error")):
        # Should catch exception and return cleanly without raising
        _persist_geval_log(result, log_path="unwritable/dir/test.jsonl")
