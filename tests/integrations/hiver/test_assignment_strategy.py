from __future__ import annotations

from collections import Counter

import pytest

from src.integrations.hiver.agent_pools import AgentPool
from src.integrations.hiver.assignment_strategy import round_robin_next, skill_based_match
from src.integrations.hiver.schema import ConfigurationError


def test_round_robin_next_fairness_across_full_cycle() -> None:
    """Each agent is selected exactly once in a full round-robin cycle."""
    pool = AgentPool(
        tier="standard",
        agent_ids=["agent_a", "agent_b", "agent_c"],
        skill_tags=["general_support"],
        agent_skills={
            "agent_a": ["general_support"],
            "agent_b": ["general_support"],
            "agent_c": ["general_support"],
        },
    )

    selected: list[str] = []
    last_index = 0
    for _ in range(len(pool.agent_ids)):
        agent_id, last_index = round_robin_next(pool, last_index)
        selected.append(agent_id)

    assert Counter(selected) == {"agent_a": 1, "agent_b": 1, "agent_c": 1}
    assert len(selected) == len(pool.agent_ids)


def test_skill_based_match_returns_matching_agent() -> None:
    """Skill-based matching honors per-agent tags in the target pool."""
    pool = AgentPool(
        tier="critical",
        agent_ids=["agent_legal", "agent_safety"],
        skill_tags=["legal_liaison", "safety_lead"],
        agent_skills={
            "agent_legal": ["legal_liaison"],
            "agent_safety": ["safety_lead"],
        },
    )

    assert skill_based_match(pool, "legal_liaison") == "agent_legal"


def test_skill_based_match_raises_configuration_error_for_empty_pool() -> None:
    """Empty pools fail loudly with a clear configuration error."""
    pool = AgentPool(tier="standard", agent_ids=[], skill_tags=[], agent_skills={})

    with pytest.raises(ConfigurationError, match="empty agent pool|zero configured agents"):
        skill_based_match(pool, "billing")

    with pytest.raises(ConfigurationError, match="empty agent pool|zero configured agents"):
        round_robin_next(pool, 0)
