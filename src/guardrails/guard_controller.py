"""Guardrails AI Controller & Schema Validator
===============================================

Orchestrates Guardrails AI output validation on LLM responses.
Enforces:
1. PII detection and masking (RegexMatch - unconditionally first)
2. Structural ID format validation (RegexMatch)
3. Competitor mention check (reusing config/guardrails/restricted_topics.yaml)
4. Toxicity and professional tone check

Implements a bounded re-prompt-and-correct loop before falling back to
a hard escalation to human review (connecting to M2.P2.3 escalation pattern).
Guarantees the critical invariant: invalid or unmasked output is NEVER
returned to the caller.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Union

from src.guardrails.competitor_check import check_competitor_mention, load_competitors
from src.guardrails.id_format import DEFAULT_TICKET_ID_PATTERN, validate_id_format, validate_ticket_tags
from src.guardrails.output_schema import AgentResponse, GuardValidationResult
from src.guardrails.pii_masking import mask_pii
from src.guardrails.toxicity_check import check_toxicity, score_toxicity

logger = logging.getLogger(__name__)

# Optional Guardrails AI Guard import
try:
    import guardrails as gd
    from guardrails import Guard
    HAS_GUARDRAILS_AI = True
except ImportError:
    HAS_GUARDRAILS_AI = False
    Guard = None

DEFAULT_VALIDATION_LOG_PATH = Path("data/logs/guardrail_validation_log.jsonl")


class GuardController:
    """
    Controller orchestrating output schema and semantic validator enforcement
    with a bounded re-prompt-and-correct retry loop and hard-exception fallback.
    """

    def __init__(
        self,
        toxicity_threshold: float = 0.7,
        ticket_id_pattern: str = DEFAULT_TICKET_ID_PATTERN,
        competitor_names: Optional[List[str]] = None,
        config_path: Optional[Union[str, Path]] = None,
        log_file_path: Optional[Union[str, Path]] = None,
        log_path: Union[str, Path] = DEFAULT_VALIDATION_LOG_PATH,
        fail_on_pii: bool = True,
    ):
        """
        Initialize GuardController.

        Args:
            toxicity_threshold: Rejection threshold for toxicity check (default: 0.7).
            ticket_id_pattern: Regex pattern for valid ticket identifiers (default: r"^TCKT-\\d{6}$").
            competitor_names: Explicit list of competitors, or None to load from config.
            config_path: Custom path to restricted_topics.yaml (if None, uses default).
            log_file_path: Alias for log_path if provided.
            log_path: Path for streaming append validation log (default: data/logs/guardrail_validation_log.jsonl).
            fail_on_pii: If True, detecting PII records a validator failure and triggers correction/escalation.
        """
        self.toxicity_threshold = toxicity_threshold
        self.ticket_id_pattern = ticket_id_pattern
        self.config_path = config_path
        self.log_path = Path(log_file_path) if log_file_path is not None else Path(log_path)
        self.fail_on_pii = fail_on_pii

        if competitor_names is not None:
            self.competitor_names = competitor_names
        else:
            loaded = load_competitors(self.config_path)
            # Ensure commonly referenced competitors like Zendesk are included
            self.competitor_names = list(set(loaded + ["Zendesk"]))

        # Initialize Guardrails AI Guard object if library is installed
        self.guard = None
        if HAS_GUARDRAILS_AI:
            try:
                self.guard = Guard.from_pydantic(output_class=AgentResponse)
                logger.info("Guardrails AI Guard initialized with AgentResponse schema.")
            except Exception as exc:
                logger.warning(f"Could not build Guardrails AI Guard: {exc}")

    def _log_failure(
        self,
        failed_validators: Optional[List[str]] = None,
        failure_reasons: Optional[List[str]] = None,
        reason: Optional[str] = None,
        retry_count: int = 0,
        attempt: Optional[int] = None,
        masked_pii_types: Optional[List[str]] = None,
        raw_text: str = "",
        raw_output: Optional[str] = None,
        escalated: bool = False,
        **kwargs,
    ) -> None:
        """
        Log validation failure to jsonl file.
        Non-crashing logging pattern: I/O or permissions failures log a warning
        but NEVER crash the validation process.
        """
        try:
            count = attempt if attempt is not None else retry_count
            text = raw_output if raw_output is not None else raw_text
            validators = failed_validators or []
            reasons_list = failure_reasons or ([reason] if reason else [])
            final_reason = reason or ("; ".join(reasons_list) if reasons_list else "")

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failed_validators": validators,
                "failure_reasons": reasons_list,
                "reason": final_reason,
                "retry_count": count,
                "masked_pii_types": masked_pii_types or [],
                "escalated": escalated,
                "raw_output_preview": text[:200] if text else "",
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as exc:
            logger.warning(
                f"Failed to append to guardrail validation log ({self.log_path}): {exc}. "
                "Validation proceeding without log append."
            )

    def _parse_agent_response(self, text: str) -> Optional[AgentResponse]:
        """
        Extract and parse JSON matching AgentResponse from string, stripping
        markdown code fences if present.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()

        # Strip markdown fences if present (e.g. ```json ... ``` or ``` ... ```)
        if "```" in clean_text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
            if match:
                clean_text = match.group(1).strip()

        try:
            data = json.loads(clean_text)
            if not isinstance(data, dict):
                return None
            return AgentResponse.model_validate(data)
        except Exception as exc:
            logger.debug(f"JSON/Schema parsing error: {exc}")
            return None

    def _build_correction_prompt(
        self,
        failed_validators: List[str],
        reasons: List[str],
        last_output: str,
    ) -> str:
        """
        Construct a targeted correction prompt explaining the specific failure reasons.
        """
        prompt_lines = [
            "Your previous response failed safety and quality validation with the following errors:",
        ]
        for reason in reasons:
            prompt_lines.append(f"- {reason}")

        prompt_lines.append("\nPlease revise the output to satisfy all required constraints:")
        if "pii_masking" in failed_validators:
            prompt_lines.append(
                "- Do NOT include sensitive personal data (e.g., Social Security Numbers or Credit Card numbers)."
            )
        if "schema_validation" in failed_validators:
            prompt_lines.append(
                "- Return STRICT JSON matching: {\"reply_text\": str, \"ticket_tags\": [str], "
                "\"requires_human_review\": bool, \"confidence\": float (0.0 to 1.0)}."
            )
        if "id_format" in failed_validators:
            prompt_lines.append(
                f"- Ensure any ticket/case identifiers strictly match pattern '{self.ticket_id_pattern}' (e.g. TCKT-123456)."
            )
        if "competitor_check" in failed_validators:
            prompt_lines.append(
                "- Do NOT recommend, promote, or disparage competing products or brands."
            )
        if "toxicity_check" in failed_validators:
            prompt_lines.append(
                "- Maintain a professional, courteous, and polite tone."
            )

        prompt_lines.append(f"\nOriginal response:\n{last_output}")
        return "\n".join(prompt_lines)

    def validate_and_correct(
        self,
        raw_llm_output: Optional[str],
        generation_retry_fn: Optional[Callable[[str], str]] = None,
        max_retries: int = 2,
    ) -> GuardValidationResult:
        """
        Validate and optionally correct LLM output via bounded re-prompting.

        Steps:
        1. Mask PII unconditionally first (never allow unmasked PII to escape).
        2. Attempt schema parse and evaluate registered validators.
        3. On failure, if retries remain and generation_retry_fn is provided, re-prompt and retry.
        4. If retries exhausted or input is unparseable/empty, escalate to human review.
           Critical Invariant: NEVER return invalid or unmasked output.

        Args:
            raw_llm_output: Raw text output from LLM.
            generation_retry_fn: Callable taking a correction instruction and returning a new LLM generation.
            max_retries: Maximum number of re-prompt attempts (default: 2).

        Returns:
            GuardValidationResult: Contains validation status, validated AgentResponse (or None),
            escalate flag, failure reasons, and detected PII types.
        """
        # Edge case: Empty or None raw_llm_output
        if raw_llm_output is None or not str(raw_llm_output).strip():
            reason = "Raw LLM output is empty or None."
            self._log_failure(
                failed_validators=["empty_input"],
                failure_reasons=[reason],
                reason=reason,
                retry_count=0,
                masked_pii_types=[],
                raw_text="",
                escalated=True,
            )
            return GuardValidationResult(
                validated=False,
                response=None,
                escalate=True,
                reason=reason,
                failure_reasons=[reason],
                failed_validators=["empty_input"],
                retry_count=0,
                masked_pii_types=[],
            )

        current_text = str(raw_llm_output)
        all_masked_pii: List[str] = []

        for attempt in range(max_retries + 1):
            # Step 1: Mask PII unconditionally and FIRST
            masked_text, pii_types = mask_pii(current_text)
            for pt in pii_types:
                if pt not in all_masked_pii:
                    all_masked_pii.append(pt)

            failed_validators: List[str] = []
            reasons: List[str] = []

            # Step 2: Attempt schema parse
            parsed_response = self._parse_agent_response(masked_text)
            if parsed_response is None:
                failed_validators.append("schema_validation")
                reasons.append("Schema Validation Failed: Response failed to parse into required AgentResponse JSON schema.")
            else:
                # If PII was found and masked, ensure human review flag is set
                if pii_types:
                    parsed_response.requires_human_review = True

                # Evaluate remaining validators on parsed response
                # A. Structural ID Format Validator
                if not validate_ticket_tags(parsed_response.ticket_tags, id_pattern=self.ticket_id_pattern):
                    failed_validators.append("id_format")
                    reasons.append(f"ID Format Validator Failed: Ticket tags contain malformed identifier: {parsed_response.ticket_tags}")

                # Also inspect reply_text for malformed TCKT- identifiers
                candidate_ids = re.findall(r"\bTCKT-[A-Za-z0-9_]+\b", parsed_response.reply_text, re.IGNORECASE)
                for cid in candidate_ids:
                    if not validate_id_format(cid, pattern=self.ticket_id_pattern):
                        if "id_format" not in failed_validators:
                            failed_validators.append("id_format")
                        reasons.append(f"ID Format Validator Failed: Malformed ticket ID '{cid}' found in reply text.")
                        break

                # B. Competitor Check Validator
                competitor = check_competitor_mention(
                    parsed_response.reply_text,
                    competitor_names=self.competitor_names,
                    config_path=self.config_path,
                )
                if competitor:
                    failed_validators.append("competitor_check")
                    reasons.append(f"Competitor Validator Failed: Competitor '{competitor}' mentioned in recommending/disparaging context.")

                # C. Toxicity Check Validator
                is_toxic = check_toxicity(
                    parsed_response.reply_text,
                    threshold=self.toxicity_threshold,
                )
                if is_toxic:
                    failed_validators.append("toxicity_check")
                    tox_score = score_toxicity(parsed_response.reply_text)
                    reasons.append(
                        f"Toxicity Validator Failed: Toxic language detected (score={tox_score:.2f} >= threshold={self.toxicity_threshold})."
                    )

            # Check if all validations passed
            # Note: PII presence does not invalidate the response by itself as long as it was masked,
            # but it flags requires_human_review = True on the response object.
            if not failed_validators and parsed_response is not None:
                return GuardValidationResult(
                    validated=True,
                    response=parsed_response,
                    escalate=False,
                    reason=None,
                    failure_reasons=[],
                    failed_validators=[],
                    retry_count=attempt,
                    masked_pii_types=all_masked_pii,
                )

            # Step 3: Handle validation failure
            combined_reason = "; ".join(reasons)
            is_exhausted = (attempt >= max_retries or generation_retry_fn is None)

            self._log_failure(
                failed_validators=failed_validators,
                failure_reasons=reasons,
                reason=combined_reason,
                retry_count=attempt,
                masked_pii_types=all_masked_pii,
                raw_text=masked_text,
                escalated=is_exhausted,
            )

            # If retries remain and a generation retry function is provided, re-prompt
            if not is_exhausted and generation_retry_fn is not None:
                correction_prompt = self._build_correction_prompt(
                    failed_validators=failed_validators,
                    reasons=reasons,
                    last_output=masked_text,
                )
                try:
                    new_output = generation_retry_fn(correction_prompt)
                    if new_output is None or not str(new_output).strip():
                        # Retry function returned empty output
                        logger.warning("generation_retry_fn returned empty output, exhausting retries.")
                        break
                    current_text = str(new_output)
                    continue
                except Exception as retry_exc:
                    logger.error(f"generation_retry_fn failed on attempt {attempt}: {retry_exc}")
                    break

        # Step 4: Retries exhausted -> Fall back to hard escalation to human review
        # CRITICAL INVARIANT: NEVER return invalid output to caller (response=None)
        final_reason = f"Validation failed after {attempt} retries: {combined_reason}"
        return GuardValidationResult(
            validated=False,
            response=None,
            escalate=True,
            reason=final_reason,
            failure_reasons=reasons,
            failed_validators=failed_validators,
            retry_count=attempt,
            masked_pii_types=all_masked_pii,
        )

