from __future__ import annotations

import json

import pytest

from src.integrations.hiver.agent_pools import AgentPool
from src.integrations.hiver.assignment import (
    AssignmentState,
    assign_escalated_ticket,
    load_assignment_state,
    save_assignment_state,
    select_assignment_strategy,
)
from src.integrations.hiver.client import MockHiverClient
from src.integrations.hiver.schema import ConfigurationError
from src.models.intent.escalation_router import EscalationDecision


def test_select_assignment_strategy_critical_defaults_to_round_robin_when_intent_none() -> None:
    """When model output is missing, the default critical routing still works and is documented."""
    pool_name, strategy = select_assignment_strategy("critical", None)
    assert pool_name == "escalation_critical"
    assert strategy == "round_robin"


def test_select_assignment_strategy_uses_skill_based_for_critical_intents() -> None:
    """Critical escalation uses skill-based assignment when a supported intent is present."""
    pool_name, strategy = select_assignment_strategy("critical", "Legal_Threat")
    assert pool_name == "escalation_critical"
    assert strategy == "skill_based"


def test_assign_escalated_ticket_falls_back_to_round_robin_when_skill_missing() -> None:
    """No matching skill in the target pool should still produce a valid assignment via round robin."""
    pool = AgentPool(
        tier="critical",
        agent_ids=["agent_a", "agent_b"],
        skill_tags=["safety_lead"],
        agent_skills={
            "agent_a": ["safety_lead"],
            "agent_b": ["safety_lead"],
        },
    )
    decision = EscalationDecision(
        thread_id="t_100",
        tweet_id="tweet_100",
        escalation_tier="critical",
        predicted_intent="Legal_Threat",
        bypassed=True,
    )
    state = AssignmentState(last_assigned_indices={"escalation_critical": 0})
    client = MockHiverClient()

    result = assign_escalated_ticket(
        decision=decision,
        client=client,
        pools={"escalation_critical": pool},
        assignment_state=state,
    )

    assert result.success is True
    assert result.strategy_used == "round_robin"
    assert result.assignee in {"agent_a", "agent_b"}
    assert client.assigned_tickets["t_100"] == result.assignee


def test_assign_escalated_ticket_does_not_advance_state_on_remote_failure() -> None:
    """Local round-robin position must stay unchanged when the remote assignment fails."""
    pool = AgentPool(
        tier="urgent",
        agent_ids=["agent_tech_1", "agent_tech_2"],
        skill_tags=["hardware_repair"],
        agent_skills={
            "agent_tech_1": ["hardware_repair"],
            "agent_tech_2": ["hardware_repair"],
        },
    )
    decision = EscalationDecision(
        thread_id="t_200",
        tweet_id="tweet_200",
        escalation_tier="urgent",
        predicted_intent="Hardware_Failure",
        bypassed=True,
    )
    state = AssignmentState(last_assigned_indices={"technical_specialist": 0})
    client = MockHiverClient(fail_assign=True)

    result = assign_escalated_ticket(
        decision=decision,
        client=client,
        pools={"technical_specialist": pool},
        assignment_state=state,
    )

    assert result.success is False
    assert state.get_index("technical_specialist") == 0
    assert "t_200" not in client.assigned_tickets


def test_assignment_state_persists_across_restart_and_uses_round_robin_sequence() -> None:
    """A simulated restart retains the prior round-robin cursor and continues the cycle."""
    pool = AgentPool(
        tier="standard",
        agent_ids=["agent_alpha", "agent_beta", "agent_gamma"],
        skill_tags=["general_support"],
        agent_skills={
            "agent_alpha": ["general_support"],
            "agent_beta": ["general_support"],
            "agent_gamma": ["general_support"],
        },
    )
    state = AssignmentState(last_assigned_indices={"standard_support": 0})
    state.set_index("standard_support", 1)
    saved = state.model_copy()

    payload_path = "data/state/test_assignment_state.json"
    save_assignment_state(saved, payload_path)
    loaded = load_assignment_state(payload_path)

    assert loaded.get_index("standard_support") == 1

    next_agent, next_index = __import__("src.integrations.hiver.assignment_strategy", fromlist=["round_robin_next"]).round_robin_next(pool, loaded.get_index("standard_support"))
    assert next_agent == "agent_gamma"
    assert next_index == 2

    loaded.set_index("standard_support", next_index)
    assert loaded.get_index("standard_support") == 2


def test_missing_and_corrupt_assignment_state_files_are_handled_gracefully() -> None:
    """Missing and malformed state files should reset to a fresh index 0 rather than crashing."""
    missing = load_assignment_state("data/state/absent_assignment_state.json")
    assert missing.model_dump() == {"last_assigned_indices": {}}

    corrupt_path = "data/state/corrupt_assignment_state.json"
    with open(corrupt_path, "w", encoding="utf-8") as handle:
        handle.write('{"not_valid": [')

    corrupted = load_assignment_state(corrupt_path)
    assert corrupted.model_dump() == {"last_assigned_indices": {}}


def test_empty_pool_is_rejected_before_round_robin_or_skill_match() -> None:
    """A new pool with no roster must fail with an explicit ConfigurationError."""
    pool = AgentPool(tier="critical", agent_ids=[], skill_tags=[], agent_skills={})

    with pytest.raises(ConfigurationError):
        __import__("src.integrations.hiver.assignment_strategy", fromlist=["round_robin_next"]).round_robin_next(pool, 0)

    with pytest.raises(ConfigurationError):
        __import__("src.integrations.hiver.assignment_strategy", fromlist=["skill_based_match"]).skill_based_match(pool, "legal_liaison")


def test_assign_escalated_ticket_remembers_round_robin_after_restart() -> None:
    """Persisting the assignment state keeps the round-robin cursor consistent across restarts."""
    pool = AgentPool(
        tier="standard",
        agent_ids=["agent_1", "agent_2", "agent_3"],
        skill_tags=["general_support"],
        agent_skills={
            "agent_1": ["general_support"],
            "agent_2": ["general_support"],
            "agent_3": ["general_support"],
        },
    )
    state = AssignmentState(last_assigned_indices={"standard_support": 0})
    decision = EscalationDecision(
        thread_id="t_300",
        tweet_id="tweet_300",
        escalation_tier="standard",
        predicted_intent="Billing_Issue",
        bypassed=True,
    )
    client = MockHiverClient()

    result = assign_escalated_ticket(
        decision=decision,
        client=client,
        pools={"standard_support": pool},
        assignment_state=state,
        state_path="data/state/test_assignment_state_roundtrip.json",
    )

    assert result.success is True
    assert client.assigned_tickets["t_300"] == result.assignee
    assert state.get_index("standard_support") == 1

    reloaded = load_assignment_state("data/state/test_assignment_state_roundtrip.json")
    assert reloaded.get_index("standard_support") == 1
