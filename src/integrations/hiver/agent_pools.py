"""Agent pool definitions and loader for Hiver auto-assignment (M6.P6.3.F1).

Defines the AgentPool model and utility functions for loading agent pool rosters
and skill tags from YAML configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from src.integrations.hiver.schema import ConfigurationError

logger = logging.getLogger(__name__)


class AgentPool(BaseModel):
    """Pydantic model representing an agent pool and its routing characteristics.

    Attributes:
        tier: Escalation tier associated with this pool (e.g., standard, urgent, critical).
        agent_ids: List of agent identifiers (e.g. emails or user IDs) in the pool.
        skill_tags: Pool-level skill tags describing the pool's capabilities.
        agent_skills: Mapping of agent_id to specific skill tags possessed by that agent.
    """

    tier: str
    agent_ids: List[str]
    skill_tags: List[str] = Field(default_factory=list)
    agent_skills: Dict[str, List[str]] = Field(default_factory=dict)


def load_agent_pools(config_path: Union[str, Path] = "config/hiver_agent_pools.yaml") -> Dict[str, AgentPool]:
    """Load and validate agent pools from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary mapping pool_name to AgentPool instances.

    Raises:
        ConfigurationError: If the config file cannot be found, cannot be parsed,
            or contains an empty agent pool (zero agents configured).
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Agent pools configuration file not found at '{path.resolve()}'")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse agent pools YAML at '{path}': {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationError(f"Agent pools YAML at '{path}' must contain a mapping.")

    pools_dict = raw_data.get("pools", raw_data)
    if not isinstance(pools_dict, dict):
        raise ConfigurationError(f"Expected 'pools' section in '{path}' to be a dictionary.")

    loaded_pools: Dict[str, AgentPool] = {}
    for pool_name, pool_cfg in pools_dict.items():
        if not isinstance(pool_cfg, dict):
            raise ConfigurationError(f"Configuration for pool '{pool_name}' must be a dictionary.")

        agent_ids = pool_cfg.get("agent_ids", [])
        if not agent_ids:
            raise ConfigurationError(f"Agent pool '{pool_name}' has no configured agents (zero agents).")

        try:
            pool = AgentPool(
                tier=pool_cfg.get("tier", "standard"),
                agent_ids=list(agent_ids),
                skill_tags=list(pool_cfg.get("skill_tags", [])),
                agent_skills=dict(pool_cfg.get("agent_skills", {})),
            )
            loaded_pools[pool_name] = pool
        except Exception as exc:
            raise ConfigurationError(f"Invalid pool definition for '{pool_name}': {exc}") from exc

    return loaded_pools


def load_intent_to_skill_mapping(config_path: Union[str, Path] = "config/hiver_agent_pools.yaml") -> Dict[str, str]:
    """Load the intent-to-skill mapping from configuration if present.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary mapping intent names to required skill tags.
    """
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        if isinstance(raw_data, dict) and "intent_to_skill" in raw_data:
            return dict(raw_data["intent_to_skill"])
    except Exception as exc:
        logger.warning("Could not read intent_to_skill mapping from %s: %s", path, exc)

    return {}
