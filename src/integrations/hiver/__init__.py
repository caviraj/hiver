"""Hiver Integration package for triage, tagging, SLA timers, and shared client interface.

Phase: M6.P6.1.F1
"""

from src.integrations.hiver.agent_pools import (
    AgentPool,
    load_agent_pools,
    load_intent_to_skill_mapping,
)
from src.integrations.hiver.assignment import (
    AssignmentState,
    assign_escalated_ticket,
    load_assignment_state,
    save_assignment_state,
    select_assignment_strategy,
)
from src.integrations.hiver.assignment_strategy import (
    round_robin_next,
    skill_based_match,
)
from src.integrations.hiver.client import (
    HiverAPIClient,
    HiverClient,
    MockHiverClient,
)
from src.integrations.hiver.schema import (
    AssignmentResult,
    ConfigurationError,
    SLAConfig,
    TagConfig,
    TriageResult,
)
from src.integrations.hiver.sla import (
    derive_sla_duration,
    start_sla_for_thread,
)
from src.integrations.hiver.tagging import (
    apply_triage_tags,
    derive_tags,
)
from src.integrations.hiver.triage import (
    load_hiver_config,
    run_triage,
)

__all__ = [
    "HiverClient",
    "HiverAPIClient",
    "MockHiverClient",
    "ConfigurationError",
    "TagConfig",
    "SLAConfig",
    "TriageResult",
    "AssignmentResult",
    "AgentPool",
    "load_agent_pools",
    "load_intent_to_skill_mapping",
    "round_robin_next",
    "skill_based_match",
    "AssignmentState",
    "load_assignment_state",
    "save_assignment_state",
    "select_assignment_strategy",
    "assign_escalated_ticket",
    "derive_tags",
    "apply_triage_tags",
    "derive_sla_duration",
    "start_sla_for_thread",
    "load_hiver_config",
    "run_triage",
]

