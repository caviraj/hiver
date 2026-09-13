"""Tests for zero-shot ungrounded baseline responder.

M5.P5.5.F1: Baselines & Failure Mode Tracking.
"""

from unittest.mock import MagicMock

import pytest

from src.evaluation.baselines.zeroshot_baseline import (
    is_baseline_error,
    zero_shot_baseline_response,
)


class TestZeroShotBaseline:
    def test_successful_call_openai_client_format(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Restart your iPhone by holding down power button."
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        res = zero_shot_baseline_response("How do I restart iPhone?", mock_client)
        assert res == "Restart your iPhone by holding down power button."
        assert not is_baseline_error(res)

        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1]["content"] == "How do I restart iPhone?"

    def test_successful_call_direct_callable_format(self):
        mock_client = MagicMock(return_value="Direct callable answer")
        # Ensure it doesn't have chat attribute
        del mock_client.chat

        res = zero_shot_baseline_response("Hello", mock_client)
        assert res == "Direct callable answer"
        assert not is_baseline_error(res)

    def test_client_exception_returns_structured_baseline_error(self):
        """CRITICAL: On failure, returns structured error marker, not silent retry."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("API connection timeout")

        res = zero_shot_baseline_response("Broken phone", mock_client)
        assert is_baseline_error(res)
        assert "[BASELINE_ERROR]" in res
        assert "ConnectionError" in res
        assert "API connection timeout" in res

    def test_empty_tweet_text_validation(self):
        mock_client = MagicMock()
        res = zero_shot_baseline_response("   ", mock_client)
        assert is_baseline_error(res)
        assert "ValueError" in res

    def test_none_client_validation(self):
        res = zero_shot_baseline_response("Valid tweet", None)
        assert is_baseline_error(res)
        assert "ValueError" in res
