"""Assignment orchestration and state persistence for Hiver (M6.P6.3.F1).

Coordinates strategy selection (round-robin vs skill-based), agent assignment,
remote Hiver API calls, and atomic state persistence for round-robin fairness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from src.integrations.hiver.agent_pools import (
    AgentPool,
    load_intent_to_skill_mapping,
)
from src.integrations.hiver.assignment_strategy import (
    round_robin_next,
    skill_based_match,
)
from src.integrations.hiver.client import HiverClient
from src.integrations.hiver.schema import AssignmentResult, ConfigurationError
from src.models.intent.escalation_router import EscalationDecision

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/state/assignment_state.json")

TECHNICAL_INTENTS: set[str] = {
    "hardware_failure",
    "software_update_failure",
    "technical_support",
    "bug_report",
    "system_outage",
    "network_issue",
}


class AssignmentState(BaseModel):
    """Pydantic model tracking last-assigned agent indices per pool.

    Persisted to ensure round-robin fairness survives process restarts and redeployments.
    """

    last_assigned_indices: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of pool_name to the index of the agent assigned in the previous round-robin cycle.",
    )

    def get_index(self, pool_name: str, default: int = 0) -> int:
        """Get the last assigned index for a pool, defaulting to 0 if untracked."""
        return self.last_assigned_indices.get(pool_name, default)

    def set_index(self, pool_name: str, index: int) -> None:
        """Update the last assigned index for a pool."""
        self.last_assigned_indices[pool_name] = index


def load_assignment_state(state_path: Union[str, Path] = DEFAULT_STATE_PATH) -> AssignmentState:
    """Load the assignment state from JSON disk storage.

    Handles missing files gracefully by returning an initialized state (last_assigned_index=0).
    Handles corrupted JSON gracefully by logging a warning and resetting indices to 0.

    Args:
        state_path: Path to the JSON state file.

    Returns:
        AssignmentState instance.
    """
    path = Path(state_path)
    if not path.exists():
        logger.info("Assignment state file not found at %s. Initializing fresh state with index 0.", path)
        return AssignmentState()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"State file contents at {path} must be a JSON object, got {type(data).__name__}")

        return AssignmentState.model_validate(data)
    except Exception as exc:
        logger.warning(
            "Assignment state file at '%s' is corrupted or invalid (%s). Resetting indices to 0 for all pools.",
            path,
            exc,
        )
        return AssignmentState()


def save_assignment_state(state: AssignmentState, state_path: Union[str, Path] = DEFAULT_STATE_PATH) -> None:
    """Persist assignment state to disk using an atomic replace operation.

    Args:
        state: The AssignmentState to persist.
        state_path: Path to the target state file.
    """
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the same directory before atomic replacement
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)

    try:
        json.dump(state.model_dump(), temp_file, indent=2)
        temp_file.flush()
        temp_file.close()

        # Atomic replacement (atomic rename/replace on POSIX and Windows)
        temp_path.replace(path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        logger.error("Failed to save assignment state to '%s': %s", path, exc)
        raise


def select_assignment_strategy(
    escalation_tier: str,
    predicted_intent: Optional[str],
) -> Tuple[str, Literal["round_robin", "skill_based"]]:
    """Determine the target agent pool and assignment strategy based on tier and intent.

    Routing rules:
    - "critical" tier:
        - If predicted_intent is provided: "escalation_critical" pool via "skill_based".
        - If predicted_intent is None: "escalation_critical" pool via "round_robin" default.
    - "urgent" tier:
        - "technical_specialist" pool via "round_robin".
    - "standard" tier:
        - If predicted_intent indicates a technical issue: "technical_specialist" pool via "round_robin".
        - Otherwise or if predicted_intent is None: "standard_support" pool via "round_robin".
    - "none" or other tiers:
        - "standard_support" pool via "round_robin".

    Args:
        escalation_tier: The tier from EscalationDecision ('critical', 'urgent', 'standard', 'none').
        predicted_intent: The predicted intent label, or None if classification did not run.

    Returns:
        Tuple of (pool_name, strategy_type).
    """
    tier = (escalation_tier or "").strip().lower()

    if tier == "critical":
        if predicted_intent and predicted_intent.strip():
            return "escalation_critical", "skill_based"
        # Documented default behavior when predicted_intent is None (e.g. keyword-only bypass)
        logger.info("Critical tier ticket with predicted_intent=None; routing to 'escalation_critical' via round_robin.")
        return "escalation_critical", "round_robin"

    if tier == "urgent":
        return "technical_specialist", "round_robin"

    if tier == "standard":
        if predicted_intent and predicted_intent.strip().lower() in TECHNICAL_INTENTS:
            return "technical_specialist", "round_robin"
        return "standard_support", "round_robin"

    # Default fallback for tier == "none" or unrecognized tier
    return "standard_support", "round_robin"


def assign_escalated_ticket(
    decision: EscalationDecision,
    client: HiverClient,
    pools: Dict[str, AgentPool],
    assignment_state: AssignmentState,
    state_path: Optional[Union[str, Path]] = DEFAULT_STATE_PATH,
    intent_to_skill: Optional[Dict[str, str]] = None,
) -> AssignmentResult:
    """Orchestrate ticket assignment for an escalated thread.

    Only invoked when decision.bypassed is True. Non-bypassed decisions proceed
    to RAG and bypass assignment, returning bypassed=False and success=True.

    CRITICAL INVARIANT: The local round-robin index is NEVER advanced or persisted
    if client.assign_ticket returns False or fails with an exception.

    Args:
        decision: The EscalationDecision from the escalation router.
        client: HiverClient implementation (API or mock).
        pools: Mapping of pool names to AgentPool definitions.
        assignment_state: Current in-memory AssignmentState.
        state_path: Optional path to save updated assignment state upon successful assignment.
        intent_to_skill: Optional mapping of intent labels to required skill tags.

    Returns:
        AssignmentResult detailing the assignment outcome.
    """
    if not decision.bypassed:
        logger.debug("Ticket %s was not escalated (bypassed=False); skipping auto-assignment.", decision.thread_id)
        return AssignmentResult(
            thread_id=decision.thread_id,
            assignee=None,
            pool_name=None,
            strategy_used=None,
            success=True,
            bypassed=False,
            errors=[],
        )

    pool_name, strategy = select_assignment_strategy(
        escalation_tier=decision.escalation_tier,
        predicted_intent=decision.predicted_intent,
    )

    if pool_name not in pools:
        err_msg = (
            f"Configured agent pools missing target pool '{pool_name}' "
            f"for tier '{decision.escalation_tier}'."
        )
        logger.error(err_msg)
        return AssignmentResult(
            thread_id=decision.thread_id,
            assignee=None,
            pool_name=pool_name,
            strategy_used=strategy,
            success=False,
            bypassed=True,
            errors=[err_msg],
        )

    pool = pools[pool_name]
    assignee: Optional[str] = None
    strategy_used: str = strategy
    new_index: Optional[int] = None

    if strategy == "skill_based":
        skill_map = intent_to_skill if intent_to_skill is not None else load_intent_to_skill_mapping()
        required_skill = skill_map.get(decision.predicted_intent or "", "")
        matched_agent = skill_based_match(pool, required_skill)

        if matched_agent is not None:
            assignee = matched_agent
        else:
            # Skill-based fallback to round-robin within the same pool
            logger.warning(
                "Skill-based match found NO agent in pool '%s' with required skill '%s' "
                "(predicted_intent='%s'). Falling back to round-robin within '%s'. "
                "WARNING: The skill roster for pool '%s' needs review!",
                pool_name,
                required_skill,
                decision.predicted_intent,
                pool_name,
                pool_name,
            )
            strategy_used = "round_robin"
            last_idx = assignment_state.get_index(pool_name, default=0)
            assignee, new_index = round_robin_next(pool, last_idx)
    else:
        # Standard round-robin
        last_idx = assignment_state.get_index(pool_name, default=0)
        assignee, new_index = round_robin_next(pool, last_idx)

    # Perform remote assignment call
    try:
        assigned_ok = client.assign_ticket(decision.thread_id, assignee)
    except Exception as exc:
        logger.error("client.assign_ticket raised an error for thread %s: %s", decision.thread_id, exc)
        assigned_ok = False
        error_detail = str(exc)
    else:
        error_detail = None

    if assigned_ok:
        # Only advance and persist index if remote assignment succeeded
        if new_index is not None:
            assignment_state.set_index(pool_name, new_index)
            if state_path:
                try:
                    save_assignment_state(assignment_state, state_path)
                except Exception as exc:
                    logger.error("Failed to persist assignment state after successful assign: %s", exc)

        logger.info(
            "Successfully assigned ticket %s to %s (pool: %s, strategy: %s)",
            decision.thread_id,
            assignee,
            pool_name,
            strategy_used,
        )
        return AssignmentResult(
            thread_id=decision.thread_id,
            assignee=assignee,
            pool_name=pool_name,
            strategy_used=strategy_used,
            success=True,
            bypassed=True,
            errors=[],
        )

    # Remote assignment failed: DO NOT advance local index or persist state
    failure_msg = (
        f"Remote assign_ticket call failed for thread '{decision.thread_id}' "
        f"with assignee '{assignee}'."
    )
    if error_detail:
        failure_msg += f" Error: {error_detail}"

    logger.error(failure_msg)
    return AssignmentResult(
        thread_id=decision.thread_id,
        assignee=assignee,
        pool_name=pool_name,
        strategy_used=strategy_used,
        success=False,
        bypassed=True,
        errors=[failure_msg],
    )
