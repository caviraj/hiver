"""NeMo Guardrails Controller for conversational safety & flow discipline (M4.P4.1.F1).

Provides topic boundary enforcement (political discourse, competitor comparisons),
conversational flow discipline (dialog rails state machine), and output filtering.
Enforces strict fail-closed safety semantics.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

from src.guardrails.schema import GuardrailResult

logger = logging.getLogger(__name__)

# Fallback response for fail-closed safety
FAIL_CLOSED_RESPONSE = (
    "I apologize, but an internal safety evaluation could not be completed. "
    "To ensure your security and support quality, your request has been safely routed to a human specialist."
)
DEFAULT_REDIRECT_RESPONSE = (
    "I can only help with Apple product support and technical inquiries — let's focus on that!"
)
DEFAULT_WARRANTY_REDIRECT_RESPONSE = (
    "I cannot make unqualified warranty or replacement guarantees. "
    "Please check our official Apple Warranty and AppleCare documentation or speak with an authorized specialist."
)


class GuardrailsConfigError(ValueError):
    """Raised when NeMo Guardrails configuration files fail validation."""
    pass


class NemoGuardrailsController:
    """Dialogue-level safety and flow orchestrator wrapping NeMo Guardrails architecture.

    Features:
    - Input rails: deterministic topic boundary check (politics, competitor comparison).
    - Dialog rails: topic continuity state machine per thread_id.
    - Output rails: competitor mentions and unqualified warranty commitments.
    - Fail-closed behavior: errors during rail checks default to passed=False and safe fallback.
    - False-positive prevention: Apple-vs-Apple past product comparisons pass safely.
    - Source-of-truth data configuration in restricted_topics.yaml.
    """

    def __init__(self, config_dir: str | Path = "config/guardrails") -> None:
        self.config_dir = Path(config_dir)
        self.config_data: Dict[str, Any] = {}
        self.restricted_categories: List[str] = []
        self.competitor_names: List[str] = []
        self.allowed_brand_names: List[str] = []
        self.warranty_prohibited_terms: List[str] = []
        self.political_keywords: List[str] = []

        # Per-thread dialog state machine storage: thread_id -> dict of state
        self._thread_states: Dict[str, Dict[str, Any]] = {}

        # Load and validate all configuration files (fail-fast)
        self._load_and_validate_config()

    def _load_and_validate_config(self) -> None:
        """Loads and parses config.yml, restricted_topics.yaml, and *.co files.

        Fails FAST with GuardrailsConfigError if any file is missing, empty, or malformed.
        """
        if not self.config_dir.exists() or not self.config_dir.is_dir():
            raise GuardrailsConfigError(f"Config directory does not exist: {self.config_dir}")

        config_yml_path = self.config_dir / "config.yml"
        if not config_yml_path.exists():
            raise GuardrailsConfigError(f"Missing main config file: {config_yml_path}")

        try:
            with open(config_yml_path, "r", encoding="utf-8") as f:
                self.config_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise GuardrailsConfigError(f"Malformed config.yml: {e}") from e

        # Validate restricted_topics.yaml
        restricted_yaml_path = self.config_dir / "restricted_topics.yaml"
        if not restricted_yaml_path.exists():
            raise GuardrailsConfigError(f"Missing restricted topics file: {restricted_yaml_path}")

        try:
            with open(restricted_yaml_path, "r", encoding="utf-8") as f:
                topics_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise GuardrailsConfigError(f"Malformed restricted_topics.yaml: {e}") from e

        self.restricted_categories = topics_data.get("restricted_categories", [])
        self.competitor_names = topics_data.get("competitor_names", [])
        self.allowed_brand_names = topics_data.get("allowed_brand_names", [
            "Apple", "iPhone", "iPad", "Mac", "MacBook", "iMac", "Apple Watch", "AirPods"
        ])
        self.warranty_prohibited_terms = topics_data.get("warranty_prohibited_terms", [
            "free replacement guaranteed", "lifetime warranty", "promise to fix free of charge",
            "we guarantee free", "100% money back unconditionally"
        ])
        self.political_keywords = topics_data.get("political_keywords", [
            "election", "vote", "democrat", "republican", "parliament", "president",
            "prime minister", "senate", "congress", "ballot", "campaign donation",
            "political party", "geopolitics", "voting"
        ])

        if not self.restricted_categories:
            raise GuardrailsConfigError("restricted_topics.yaml must define non-empty 'restricted_categories'")

        # Validate Colang files in rails/
        rails_dir = self.config_dir / "rails"
        if not rails_dir.exists() or not rails_dir.is_dir():
            raise GuardrailsConfigError(f"Missing Colang rails directory: {rails_dir}")

        required_rails = ["input_rails.co", "dialog_rails.co", "output_rails.co"]
        for rail_file in required_rails:
            rail_path = rails_dir / rail_file
            if not rail_path.exists():
                raise GuardrailsConfigError(f"Missing required Colang rail file: {rail_path}")
            self._validate_colang_syntax(rail_path)

        # Build regex patterns
        self._rebuild_patterns()

    def _validate_colang_syntax(self, file_path: Path) -> None:
        """Basic syntax and structure validation for Colang (.co) files."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise GuardrailsConfigError(f"Cannot read Colang file {file_path}: {e}") from e

        if not content.strip():
            raise GuardrailsConfigError(f"Colang file {file_path.name} is empty")

        # Check for minimum Colang constructs (define user, define bot, or define flow)
        if not re.search(r"define\s+(user|bot|flow)\s+", content):
            raise GuardrailsConfigError(
                f"Colang file {file_path.name} does not contain any valid 'define user/bot/flow' definitions"
            )

    def _rebuild_patterns(self) -> None:
        """Compiles regex matchers based on the loaded restricted topics data."""
        # Competitors pattern: matching word boundaries, plurals ('s or s)
        # e.g., \b(?:Samsung|Google Pixel|Pixel|OnePlus)(?:'s|s)?\b
        if self.competitor_names:
            escaped_competitors = [re.escape(name) for name in self.competitor_names]
            comp_pattern = r"\b(?:" + "|".join(escaped_competitors) + r")(?:'s|s)?\b"
            self._competitor_regex = re.compile(comp_pattern, re.IGNORECASE)
        else:
            self._competitor_regex = re.compile(r"(?!.*)")

        # Political discourse pattern
        if self.political_keywords:
            escaped_politics = [re.escape(w) for w in self.political_keywords]
            pol_pattern = r"\b(?:" + "|".join(escaped_politics) + r")(?:'s|s)?\b"
            self._political_regex = re.compile(pol_pattern, re.IGNORECASE)
        else:
            self._political_regex = re.compile(r"(?!.*)")

        # Warranty commitment pattern
        if self.warranty_prohibited_terms:
            escaped_terms = [re.escape(w) for w in self.warranty_prohibited_terms]
            warranty_pattern = r"\b(?:" + "|".join(escaped_terms) + r")\b"
            self._warranty_regex = re.compile(warranty_pattern, re.IGNORECASE)
        else:
            self._warranty_regex = re.compile(r"(?!.*)")

    def check_input(self, text: str, thread_id: str = "default") -> GuardrailResult:
        """Runs input rails only.

        Validates user message against restricted topic boundaries (politics, competitor comparison).
        Handles empty inputs gracefully (passes through).
        Enforces false-positive avoidance for intra-brand comparisons (e.g., iPhone vs old iPhone).
        Fails closed on internal exceptions.
        """
        try:
            if not text or not text.strip():
                # Empty message passes input rail (nothing to classify)
                return GuardrailResult(
                    passed=True,
                    triggered_rail=None,
                    reason=None,
                    redirected_response=None,
                    thread_id=thread_id,
                )

            cleaned_text = text.strip()

            # 1. Check Political Discourse
            if "political_discourse" in self.restricted_categories:
                if self._political_regex.search(cleaned_text):
                    return GuardrailResult(
                        passed=False,
                        triggered_rail="input",
                        reason="Restricted topic: political discourse detected.",
                        redirected_response=DEFAULT_REDIRECT_RESPONSE,
                        thread_id=thread_id,
                    )

            # 2. Check Competitor Comparison
            if "competitor_comparison" in self.restricted_categories:
                competitor_match = self._competitor_regex.search(cleaned_text)
                if competitor_match:
                    # Competitor explicitly mentioned
                    matched_competitor = competitor_match.group(0)
                    return GuardrailResult(
                        passed=False,
                        triggered_rail="input",
                        reason=f"Restricted topic: competitor comparison ({matched_competitor}) detected.",
                        redirected_response=DEFAULT_REDIRECT_RESPONSE,
                        thread_id=thread_id,
                    )

            # Pass
            return GuardrailResult(
                passed=True,
                triggered_rail=None,
                reason=None,
                redirected_response=None,
                thread_id=thread_id,
            )

        except Exception as e:
            logger.error(f"Input rail failed with exception: {e}", exc_info=True)
            return GuardrailResult(
                passed=False,
                triggered_rail="input",
                reason=f"Input rail evaluation error (fail-closed): {e}",
                redirected_response=FAIL_CLOSED_RESPONSE,
                thread_id=thread_id,
            )

    def check_output(self, text: str, thread_id: str = "default") -> GuardrailResult:
        """Runs output rails only.

        Filters generated LLM response:
        - Blocks competitor name mentions.
        - Blocks unqualified legal/warranty commitments.
        Fails closed on internal exceptions.
        """
        try:
            if not text or not text.strip():
                return GuardrailResult(
                    passed=True,
                    triggered_rail=None,
                    reason=None,
                    redirected_response=None,
                    thread_id=thread_id,
                )

            cleaned_text = text.strip()

            # 1. Competitor mentions check
            comp_match = self._competitor_regex.search(cleaned_text)
            if comp_match:
                matched_name = comp_match.group(0)
                return GuardrailResult(
                    passed=False,
                    triggered_rail="output",
                    reason=f"Output policy violation: competitor mention ({matched_name}) detected in response.",
                    redirected_response=DEFAULT_REDIRECT_RESPONSE,
                    thread_id=thread_id,
                )

            # 2. Unqualified warranty / legal commitments check
            warranty_match = self._warranty_regex.search(cleaned_text)
            if warranty_match:
                matched_term = warranty_match.group(0)
                return GuardrailResult(
                    passed=False,
                    triggered_rail="output",
                    reason=f"Output policy violation: unauthorized warranty commitment ({matched_term}).",
                    redirected_response=DEFAULT_WARRANTY_REDIRECT_RESPONSE,
                    thread_id=thread_id,
                )

            return GuardrailResult(
                passed=True,
                triggered_rail=None,
                reason=None,
                redirected_response=None,
                thread_id=thread_id,
            )

        except Exception as e:
            logger.error(f"Output rail failed with exception: {e}", exc_info=True)
            return GuardrailResult(
                passed=False,
                triggered_rail="output",
                reason=f"Output rail evaluation error (fail-closed): {e}",
                redirected_response=FAIL_CLOSED_RESPONSE,
                thread_id=thread_id,
            )

    def _get_or_create_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Returns dialog state dict isolated to the given thread_id."""
        if thread_id not in self._thread_states:
            self._thread_states[thread_id] = {
                "current_topic": None,
                "history_turns": 0,
                "last_active_task": None,
            }
        return self._thread_states[thread_id]

    def check_dialog(self, text: str, thread_id: str = "default") -> GuardrailResult:
        """Runs dialog rails state machine.

        Enforces conversational flow discipline (e.g. topic continuity without sudden jarring shifts).
        States are strictly isolated per thread_id.
        """
        try:
            state = self._get_or_create_thread_state(thread_id)
            cleaned_text = text.strip() if text else ""

            # Check for abrupt topic shifts if an existing active technical resolution task is underway
            # Simple slot tracking: current_topic
            if state["current_topic"] and cleaned_text:
                # If currently troubleshooting battery, and user abruptly asks about purchasing or unrelated refund
                if "battery" in state["current_topic"].lower():
                    if re.search(r"\b(buy|purchase|pricing|store hours|weather|refund)\b", cleaned_text, re.IGNORECASE):
                        return GuardrailResult(
                            passed=False,
                            triggered_rail="dialog",
                            reason=f"Abrupt topic shift from '{state['current_topic']}' without resolution.",
                            redirected_response=(
                                f"We were currently troubleshooting your {state['current_topic']}. "
                                "Would you like to continue resolving that first, or switch topics?"
                            ),
                            thread_id=thread_id,
                        )

            # Update topic slot if recognized
            if re.search(r"\bbattery\b", cleaned_text, re.IGNORECASE):
                state["current_topic"] = "battery troubleshooting"
            elif re.search(r"\bscreen|display\b", cleaned_text, re.IGNORECASE):
                state["current_topic"] = "screen repair"

            state["history_turns"] += 1

            return GuardrailResult(
                passed=True,
                triggered_rail=None,
                reason=None,
                redirected_response=None,
                thread_id=thread_id,
            )

        except Exception as e:
            logger.error(f"Dialog rail failed with exception: {e}", exc_info=True)
            return GuardrailResult(
                passed=False,
                triggered_rail="dialog",
                reason=f"Dialog rail evaluation error (fail-closed): {e}",
                redirected_response=FAIL_CLOSED_RESPONSE,
                thread_id=thread_id,
            )

    def process_turn(
        self,
        text: str,
        thread_id: str = "default",
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> GuardrailResult:
        """Full 3-tier pipeline execution: input rail -> dialog rail -> generation -> output rail.

        Parameters
        ----------
        text : str
            The incoming user message.
        thread_id : str
            Unique identifier for per-thread dialog state isolation.
        generate_fn : Optional[Callable[[str], str]]
            Callback executing downstream LLM generation. If omitted, returns pass on input/dialog.

        Returns
        -------
        GuardrailResult
            Outcome of the turn processing.
        """
        try:
            # 1. Tier 1: Input Rails
            input_res = self.check_input(text, thread_id=thread_id)
            if not input_res.passed:
                return input_res

            # 2. Tier 2: Dialog Rails
            dialog_res = self.check_dialog(text, thread_id=thread_id)
            if not dialog_res.passed:
                return dialog_res

            # 3. LLM Generation step (if provided)
            generated_text = ""
            if generate_fn:
                try:
                    generated_text = generate_fn(text)
                except Exception as gen_err:
                    logger.error(f"Generation callback failed: {gen_err}", exc_info=True)
                    return GuardrailResult(
                        passed=False,
                        triggered_rail=None,
                        reason=f"Generation callback error (fail-closed): {gen_err}",
                        redirected_response=FAIL_CLOSED_RESPONSE,
                        thread_id=thread_id,
                    )

            # If no generation callback or empty generation, return passed result
            if not generated_text:
                return GuardrailResult(
                    passed=True,
                    triggered_rail=None,
                    reason=None,
                    redirected_response=None,
                    thread_id=thread_id,
                )

            # 4. Tier 3: Output Rails
            output_res = self.check_output(generated_text, thread_id=thread_id)
            if not output_res.passed:
                return output_res

            # Everything passed successfully
            return GuardrailResult(
                passed=True,
                triggered_rail=None,
                reason=None,
                redirected_response=generated_text,
                thread_id=thread_id,
            )

        except Exception as e:
            logger.error(f"Pipeline turn execution failed with exception: {e}", exc_info=True)
            return GuardrailResult(
                passed=False,
                triggered_rail=None,
                reason=f"Pipeline turn error (fail-closed): {e}",
                redirected_response=FAIL_CLOSED_RESPONSE,
                thread_id=thread_id,
            )
