"""Tests for trivial baseline responder.

M5.P5.5.F1: Baselines & Failure Mode Tracking.
"""

import pytest

from src.evaluation.baselines.trivial_baseline import trivial_baseline_response


class TestTrivialBaseline:
    def test_trigger_keyword_matches(self):
        text = "My iPhone screen is broken, what should I do?"
        res = trivial_baseline_response(text)
        assert res is not None
        assert "https://support.apple.com" in res
        assert "automated assistant" in res

    def test_word_boundary_avoids_helpful_matching_help(self):
        """CRITICAL: Word boundary check ensuring 'help' does not trigger on 'helpful'."""
        text = "The previous agent was very helpful, thank you!"
        res = trivial_baseline_response(text)
        assert res is None, "'helpful' must not trigger a match for 'help'"

    def test_word_boundary_avoids_unbroken_matching_broken(self):
        text = "The seal on the package was unbroken."
        res = trivial_baseline_response(text)
        assert res is None, "'unbroken' must not trigger a match for 'broken'"

    def test_word_boundary_avoids_antifreeze_matching_frozen(self):
        text = "Does this car have antifreeze?"
        res = trivial_baseline_response(text)
        assert res is None, "'antifreeze' must not trigger a match for 'frozen'"

    def test_case_insensitivity(self):
        text = "My phone is FROZEN completely!"
        res = trivial_baseline_response(text)
        assert res is not None
        assert "https://support.apple.com" in res

    def test_no_match_returns_none_floor_behavior(self):
        text = "I love the new design of iOS."
        res = trivial_baseline_response(text)
        assert res is None

    def test_custom_keywords_and_url(self):
        text = "I have a billing inquiry."
        custom_url = "https://support.company.com/billing"
        res = trivial_baseline_response(
            text, trigger_keywords=["billing", "invoice"], support_url=custom_url
        )
        assert res is not None
        assert custom_url in res
