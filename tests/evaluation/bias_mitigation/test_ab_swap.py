"""Unit tests for position bias mitigation via swap consistency."""

import pytest
from src.evaluation.bias_mitigation.ab_swap import (
    SwapConsistencyResult,
    run_ab_swap_consistency,
)


def test_consistent_winner_a():
    """When judge consistently prefers candidate A regardless of position."""
    # Forward: pos 1 is A -> returns position_1
    # Reversed: pos 1 is B, pos 2 is A -> returns position_2
    def judge(p1: str, p2: str) -> str:
        if "Candidate A" in p1:
            return "position_1"
        return "position_2"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", judge)
    assert result.consistent is True
    assert result.winner == "a"
    assert result.forward_verdict == "position_1"
    assert result.reversed_verdict == "position_2"
    assert result.insufficient_data is False


def test_consistent_winner_b():
    """When judge consistently prefers candidate B regardless of position."""
    # Forward: pos 1 is A, pos 2 is B -> returns position_2
    # Reversed: pos 1 is B, pos 2 is A -> returns position_1
    def judge(p1: str, p2: str) -> str:
        if "Candidate B" in p1:
            return "position_1"
        return "position_2"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", judge)
    assert result.consistent is True
    assert result.winner == "b"
    assert result.forward_verdict == "position_2"
    assert result.reversed_verdict == "position_1"
    assert result.insufficient_data is False


def test_genuine_tie():
    """When judge reports a tie in both forward and reversed order."""
    def judge(p1: str, p2: str) -> str:
        return "tie"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", judge)
    assert result.consistent is True
    assert result.winner == "tie"
    assert result.forward_verdict == "tie"
    assert result.reversed_verdict == "tie"
    assert result.insufficient_data is False


def test_position_bias_always_first_discordant():
    """A position-biased judge that always prefers position_1 is detected as discordant."""
    def biased_judge(p1: str, p2: str) -> str:
        return "position_1"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", biased_judge)
    assert result.consistent is False
    assert result.winner == "tie"
    assert result.forward_verdict == "position_1"
    assert result.reversed_verdict == "position_1"
    assert result.insufficient_data is False


def test_position_bias_always_second_discordant():
    """A judge that always prefers position_2 is detected as discordant."""
    def biased_judge(p1: str, p2: str) -> str:
        return "position_2"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", biased_judge)
    assert result.consistent is False
    assert result.winner == "tie"
    assert result.forward_verdict == "position_2"
    assert result.reversed_verdict == "position_2"
    assert result.insufficient_data is False


def test_judge_retry_success():
    """A judge call that fails once but succeeds on retry should produce a valid result."""
    attempts = {"forward": 0, "reversed": 0}

    def flaky_judge(p1: str, p2: str) -> str:
        if "Candidate A" in p1:
            attempts["forward"] += 1
            if attempts["forward"] == 1:
                raise ConnectionError("Temporary network glitch")
            return "position_1"
        else:
            attempts["reversed"] += 1
            return "position_2"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", flaky_judge)
    assert result.consistent is True
    assert result.winner == "a"
    assert attempts["forward"] == 2
    assert result.insufficient_data is False


def test_forward_failure_after_retry():
    """If forward call fails after retry, result is insufficient_data=True and winner=None."""
    def failing_judge(p1: str, p2: str) -> str:
        raise RuntimeError("API timeout")

    result = run_ab_swap_consistency("Candidate A", "Candidate B", failing_judge)
    assert result.insufficient_data is True
    assert result.consistent is False
    assert result.winner is None
    assert result.forward_verdict is None
    assert "failed after retry" in (result.error_message or "")


def test_reversed_failure_after_retry_does_not_default():
    """If reversed call fails after retry, must NOT default to forward winner."""
    def partial_fail_judge(p1: str, p2: str) -> str:
        if "Candidate A" in p1:
            return "position_1"
        raise RuntimeError("Reversed call error")

    result = run_ab_swap_consistency("Candidate A", "Candidate B", partial_fail_judge)
    assert result.insufficient_data is True
    assert result.consistent is False
    assert result.winner is None
    assert result.forward_verdict == "position_1"
    assert result.reversed_verdict is None
    assert "reversed call failed after retry" in (result.error_message or "").lower()


def test_invalid_verdict_format_retried_and_handled():
    """If judge returns unexpected format string, it counts as a failure."""
    def bad_format_judge(p1: str, p2: str) -> str:
        return "candidate_a_is_the_best"

    result = run_ab_swap_consistency("Candidate A", "Candidate B", bad_format_judge)
    assert result.insufficient_data is True
    assert result.consistent is False
    assert result.winner is None
