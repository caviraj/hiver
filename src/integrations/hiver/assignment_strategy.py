"""Assignment strategies for Hiver auto-assignment routing (M6.P6.3.F1).

Provides round-robin assignment and skill-based matching strategies for agent pools.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.integrations.hiver.agent_pools import AgentPool
from src.integrations.hiver.schema import ConfigurationError

logger = logging.getLogger(__name__)


def round_robin_next(pool: AgentPool, last_assigned_index: int) -> Tuple[str, int]:
    """Select the next agent from an agent pool using round-robin ordering.

    The index must be persisted between calls by the caller. This function does
    not maintain module-level mutable state to ensure concurrency and process safety.

    Args:
        pool: The AgentPool containing the roster of agent_ids.
        last_assigned_index: The index of the agent assigned on the previous call.

    Returns:
        Tuple of (next_agent_id, new_index).

    Raises:
        ConfigurationError: If the pool has zero configured agents.
    """
    if not pool.agent_ids:
        raise ConfigurationError(
            f"Cannot perform round-robin assignment on empty agent pool (tier='{pool.tier}', 0 agents configured)."
        )

    new_index = (last_assigned_index + 1) % len(pool.agent_ids)
    next_agent_id = pool.agent_ids[new_index]
    return next_agent_id, new_index


def skill_based_match(pool: AgentPool, required_skill: str) -> Optional[str]:
    """Match a required skill against agents in an agent pool.

    Scans the pool's roster and checks per-agent skill tags.

    Args:
        pool: The AgentPool to search.
        required_skill: The required skill tag to match.

    Returns:
        The agent_id of an agent with the required skill, or None if no match is found.

    Raises:
        ConfigurationError: If the pool has zero configured agents.
    """
    if not pool.agent_ids:
        raise ConfigurationError(
            f"Cannot perform skill-based assignment on empty agent pool (tier='{pool.tier}', 0 agents configured)."
        )

    if not required_skill:
        return None

    req_normalized = required_skill.strip().lower()

    for agent_id in pool.agent_ids:
        skills = pool.agent_skills.get(agent_id, [])
        if any(s.strip().lower() == req_normalized for s in skills):
            return agent_id

    return None
