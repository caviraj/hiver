"""Comprehensive unit and edge case tests for NemoGuardrailsController (M4.P4.1.F1)."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.guardrails.nemo_controller import (
    NemoGuardrailsController,
    GuardrailsConfigError,
    FAIL_CLOSED_RESPONSE,
    DEFAULT_REDIRECT_RESPONSE,
    DEFAULT_WARRANTY_REDIRECT_RESPONSE,
)
from src.guardrails.schema import GuardrailResult


@pytest.fixture
def controller():
    """Initializes a standard NemoGuardrailsController with repo config."""
    return NemoGuardrailsController(config_dir="config/guardrails")


class TestInputRails:
    """Tests for Tier 1: Input Rails topic boundary enforcement."""

    def test_blocks_genuine_political_discourse(self, controller):
        """Input rail must block genuine political discourse."""
        messages = [
            "Who should I vote for in the upcoming election?",
            "What do you think about the democrat and republican policies?",
            "Tell me about the president and prime minister election.",
        ]
        for msg in messages:
            res = controller.check_input(msg, thread_id="thread_politics")
            assert not res.passed
            assert res.triggered_rail == "input"
            assert "political discourse" in (res.reason or "").lower()
            assert res.redirected_response == DEFAULT_REDIRECT_RESPONSE

    def test_blocks_genuine_competitor_comparison(self, controller):
        """Input rail must block genuine competitor comparison."""
        messages = [
            "Is Samsung Galaxy S24 better than iPhone 16?",
            "How does Google Pixel compare to Apple?",
            "Should I buy OnePlus or an iPhone?",
            "Compare Xiaomi with Apple iPad.",
        ]
        for msg in messages:
            res = controller.check_input(msg, thread_id="thread_comp")
            assert not res.passed
            assert res.triggered_rail == "input"
            assert "competitor comparison" in (res.reason or "").lower()
            assert res.redirected_response == DEFAULT_REDIRECT_RESPONSE

    def test_handles_competitor_plurals_and_possessives(self, controller):
        """Input rail must handle plural and possessive competitor names accurately."""
        # e.g., Samsung's, Samsungs
        res_possessive = controller.check_input("Samsung's camera is great, right?", thread_id="t1")
        assert not res_possessive.passed
        assert res_possessive.triggered_rail == "input"

        res_plural = controller.check_input("Are Samsungs better than iPhones?", thread_id="t2")
        assert not res_plural.passed
        assert res_plural.triggered_rail == "input"

    def test_false_positive_avoidance_apple_vs_apple(self, controller):
        """CRITICAL: Apple-vs-Apple past product comparisons must NOT be blocked.

        'Is my new iPhone better than the old one I had?' compares Apple to Apple's own past product,
        not a competitor. Must pass through.
        """
        valid_queries = [
            "Is my new iPhone better than the old one I had?",
            "How does the iPhone 16 compare to the iPhone 12?",
            "Is the new M3 MacBook Air faster than my 2020 Intel MacBook?",
            "Can I trade in my old Apple Watch for the new series?",
            "My iPad battery drains faster than my previous iPad.",
        ]
        for query in valid_queries:
            res = controller.check_input(query, thread_id="thread_fp")
            assert res.passed, f"False positive triggered for query: {query}"
            assert res.triggered_rail is None
            assert res.redirected_response is None

    def test_empty_user_message_handling(self, controller):
        """Empty or whitespace-only messages must pass gracefully without crashing."""
        for empty_text in ["", "   ", "\n\t  "]:
            res = controller.check_input(empty_text, thread_id="thread_empty")
            assert res.passed
            assert res.triggered_rail is None


class TestOutputRails:
    """Tests for Tier 3: Output Rails content filtering."""

    def test_blocks_competitor_mentions_in_generated_output(self, controller):
        """Output rail must block generated answers mentioning competitor brand names."""
        bad_outputs = [
            "If you want that feature, we recommend looking at Samsung Galaxy.",
            "Google Pixel also has a very similar night sight mode.",
            "You might consider buying OnePlus instead.",
        ]
        for out in bad_outputs:
            res = controller.check_output(out, thread_id="thread_out_comp")
            assert not res.passed
            assert res.triggered_rail == "output"
            assert "competitor mention" in (res.reason or "").lower()
            assert res.redirected_response == DEFAULT_REDIRECT_RESPONSE

    def test_blocks_unqualified_warranty_commitments(self, controller):
        """Output rail must block unqualified legal or warranty commitments."""
        bad_warranty_outputs = [
            "Don't worry, a free replacement guaranteed for your damaged screen!",
            "We offer a lifetime warranty on all cracked screens.",
            "Apple will promise to fix free of charge regardless of accidental damage.",
            "We give you 100% money back unconditionally anytime.",
        ]
        for out in bad_warranty_outputs:
            res = controller.check_output(out, thread_id="thread_out_warr")
            assert not res.passed
            assert res.triggered_rail == "output"
            assert "unauthorized warranty commitment" in (res.reason or "").lower()
            assert res.redirected_response == DEFAULT_WARRANTY_REDIRECT_RESPONSE

    def test_allows_valid_apple_support_output(self, controller):
        """Output rail must allow valid, helpful Apple support responses."""
        valid_outputs = [
            "To reset your iPhone, go to Settings > General > Transfer or Reset iPhone.",
            "Your AppleCare+ plan covers up to two incidents of accidental damage per 12 months.",
            "You can book a Genius Bar appointment at your nearest Apple Store.",
        ]
        for out in valid_outputs:
            res = controller.check_output(out, thread_id="thread_out_valid")
            assert res.passed
            assert res.triggered_rail is None


class TestDialogRailsAndStateIsolation:
    """Tests for Tier 2: Dialog state machine and per-thread state isolation."""

    def test_per_thread_dialog_state_isolation(self, controller):
        """Conversation A's topic state must NOT leak into Conversation B."""
        # Thread A enters battery troubleshooting
        controller.check_dialog("My iPhone battery is dying really fast", thread_id="thread_A")
        state_a = controller._get_or_create_thread_state("thread_A")
        assert state_a["current_topic"] == "battery troubleshooting"

        # Thread B starts fresh
        state_b = controller._get_or_create_thread_state("thread_B")
        assert state_b["current_topic"] is None

        # Thread B can ask about pricing or buying without triggering abrupt topic shift
        res_b = controller.check_dialog("Can I buy a new phone case?", thread_id="thread_B")
        assert res_b.passed

        # In Thread A, abruptly switching to buying or weather triggers dialog rail clarification
        res_a = controller.check_dialog("Can I buy a new phone case?", thread_id="thread_A")
        assert not res_a.passed
        assert res_a.triggered_rail == "dialog"
        assert "abrupt topic shift" in (res_a.reason or "").lower()


class TestFailClosedBehavior:
    """Tests fail-closed semantics across all rail evaluations."""

    def test_input_rail_fail_closed_on_exception(self, controller):
        """If an internal error or regex exception occurs during input checking, fail closed."""
        mock_regex = MagicMock()
        mock_regex.search.side_effect = RuntimeError("Engine timeout")
        with patch.object(controller, "_political_regex", mock_regex):
            res = controller.check_input("Hello there", thread_id="fail_t1")
            assert not res.passed
            assert res.triggered_rail == "input"
            assert "fail-closed" in (res.reason or "")
            assert res.redirected_response == FAIL_CLOSED_RESPONSE

    def test_output_rail_fail_closed_on_exception(self, controller):
        """If an internal error occurs during output checking, fail closed."""
        mock_regex = MagicMock()
        mock_regex.search.side_effect = Exception("Regex crash")
        with patch.object(controller, "_competitor_regex", mock_regex):
            res = controller.check_output("Your screen is fixed.", thread_id="fail_t2")
            assert not res.passed
            assert res.triggered_rail == "output"
            assert "fail-closed" in (res.reason or "")
            assert res.redirected_response == FAIL_CLOSED_RESPONSE

    def test_process_turn_fail_closed_on_generation_exception(self, controller):
        """If generation callback throws an exception, fail closed."""
        def faulty_generator(_):
            raise TimeoutError("LLM upstream service unavailable")

        res = controller.process_turn(
            "How do I restart my iPhone?",
            thread_id="t_gen_fail",
            generate_fn=faulty_generator,
        )
        assert not res.passed
        assert "fail-closed" in (res.reason or "")
        assert res.redirected_response == FAIL_CLOSED_RESPONSE


class TestFullProcessTurnPipeline:
    """Tests end-to-end process_turn execution."""

    def test_successful_turn_with_generation_callback(self, controller):
        """Happy path: input passes, dialog passes, generation runs, output passes."""
        def mock_llm(_):
            return "To force restart your iPhone, press and quickly release Volume Up, then Volume Down, and hold Side button."

        res = controller.process_turn(
            "How do I force restart my iPhone?",
            thread_id="t_happy",
            generate_fn=mock_llm,
        )
        assert res.passed
        assert res.triggered_rail is None
        assert "force restart" in (res.redirected_response or "")

    def test_turn_blocked_at_input_tier(self, controller):
        """When input violates topic boundary, generation callback is never called."""
        llm_called = False

        def mock_llm(_):
            nonlocal llm_called
            llm_called = True
            return "response"

        res = controller.process_turn(
            "Who won the last election?",
            thread_id="t_turn_in",
            generate_fn=mock_llm,
        )
        assert not res.passed
        assert res.triggered_rail == "input"
        assert not llm_called

    def test_turn_blocked_at_output_tier(self, controller):
        """When generation generates a competitor name, output rail catches it."""
        def mock_llm(_):
            return "You could alternatively consider buying a Google Pixel."

        res = controller.process_turn(
            "I want a phone with high battery life.",
            thread_id="t_turn_out",
            generate_fn=mock_llm,
        )
        assert not res.passed
        assert res.triggered_rail == "output"
        assert res.redirected_response == DEFAULT_REDIRECT_RESPONSE


class TestDynamicConfigurationAndFailFast:
    """Tests configuration loading, fail-fast on malformed files, and dynamic updates."""

    def test_fail_fast_on_missing_config_directory(self):
        """Controller initialization must fail fast if config directory is missing."""
        with pytest.raises(GuardrailsConfigError, match="Config directory does not exist"):
            NemoGuardrailsController(config_dir="non_existent_path_xyz")

    def test_fail_fast_on_malformed_yaml(self, tmp_path):
        """Controller initialization must fail fast on malformed YAML."""
        # Create temp config structure
        (tmp_path / "rails").mkdir()
        (tmp_path / "config.yml").write_text("invalid: [yaml: broken", encoding="utf-8")
        (tmp_path / "restricted_topics.yaml").write_text("restricted_categories: []", encoding="utf-8")
        (tmp_path / "rails" / "input_rails.co").write_text("define user ask\n  \"hi\"", encoding="utf-8")
        (tmp_path / "rails" / "dialog_rails.co").write_text("define user ask\n  \"hi\"", encoding="utf-8")
        (tmp_path / "rails" / "output_rails.co").write_text("define bot say\n  \"hi\"", encoding="utf-8")

        with pytest.raises(GuardrailsConfigError, match="Malformed config.yml"):
            NemoGuardrailsController(config_dir=tmp_path)

    def test_fail_fast_on_invalid_colang(self, tmp_path):
        """Controller initialization must fail fast if Colang file has invalid structure."""
        (tmp_path / "rails").mkdir()
        (tmp_path / "config.yml").write_text("version: '2.0'", encoding="utf-8")
        (tmp_path / "restricted_topics.yaml").write_text(
            "restricted_categories:\n  - political_discourse", encoding="utf-8"
        )
        # Empty Colang file
        (tmp_path / "rails" / "input_rails.co").write_text("   \n", encoding="utf-8")
        (tmp_path / "rails" / "dialog_rails.co").write_text("define user ask\n  \"hi\"", encoding="utf-8")
        (tmp_path / "rails" / "output_rails.co").write_text("define bot say\n  \"hi\"", encoding="utf-8")

        with pytest.raises(GuardrailsConfigError, match="Colang file input_rails.co is empty"):
            NemoGuardrailsController(config_dir=tmp_path)

    def test_dynamic_restricted_topic_update_at_runtime(self, tmp_path):
        """Adding a new competitor to restricted_topics.yaml is picked up on controller re-init without touching Colang."""
        rails_dir = tmp_path / "rails"
        rails_dir.mkdir()
        (tmp_path / "config.yml").write_text("version: '2.0'", encoding="utf-8")
        (rails_dir / "input_rails.co").write_text("define flow check topic\n  user ask\n", encoding="utf-8")
        (rails_dir / "dialog_rails.co").write_text("define flow check dialog\n  user ask\n", encoding="utf-8")
        (rails_dir / "output_rails.co").write_text("define flow check output\n  bot say\n", encoding="utf-8")

        # Initial topics without NewBrandX
        topics_file = tmp_path / "restricted_topics.yaml"
        topics_file.write_text(
            yaml.dump({
                "restricted_categories": ["competitor_comparison"],
                "competitor_names": ["Samsung"],
            }),
            encoding="utf-8",
        )

        c1 = NemoGuardrailsController(config_dir=tmp_path)
        # NewBrandX should pass initially
        assert c1.check_input("Should I get NewBrandX phone?", thread_id="dyn1").passed

        # Update restricted_topics.yaml to include NewBrandX
        topics_file.write_text(
            yaml.dump({
                "restricted_categories": ["competitor_comparison"],
                "competitor_names": ["Samsung", "NewBrandX"],
            }),
            encoding="utf-8",
        )

        c2 = NemoGuardrailsController(config_dir=tmp_path)
        # NewBrandX should now be blocked
        res2 = c2.check_input("Should I get NewBrandX phone?", thread_id="dyn2")
        assert not res2.passed
        assert "competitor comparison (NewBrandX)" in (res2.reason or "")
