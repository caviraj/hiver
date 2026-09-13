"""Triage orchestration for Hiver threads.

Phase: M6.P6.1.F1
Coordinates tag derivation/application and SLA timer initiation independently.
Failures in Hiver API sync are recorded in TriageResult and never raise/bubble up
to block or reverse upstream routing decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from src.integrations.hiver.client import HiverClient
from src.integrations.hiver.schema import (
    ConfigurationError,
    SLAConfig,
    TagConfig,
    TriageResult,
)
from src.integrations.hiver.sla import derive_sla_duration, start_sla_for_thread
from src.integrations.hiver.tagging import apply_triage_tags, derive_tags

logger = logging.getLogger(__name__)


def load_hiver_config(
    config_path: Union[str, Path] = "config/hiver_config.yaml"
) -> Tuple[TagConfig, SLAConfig]:
    """Load TagConfig and SLAConfig from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (TagConfig, SLAConfig).

    Raises:
        ConfigurationError: If the configuration file cannot be found or parsed.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigurationError(f"Hiver configuration file not found at '{path.resolve()}'")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data: Optional[Dict[str, Any]] = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse Hiver configuration YAML at '{path}': {e}") from e

    data = raw_data or {}
    tag_mapping = data.get("tag_mapping", {})
    fallback_tag = data.get("fallback_tag", "Uncategorized")
    sla_durations = data.get("sla_durations", {})

    tag_config = TagConfig(mapping=tag_mapping, fallback_tag=fallback_tag)
    sla_config = SLAConfig(durations_minutes=sla_durations)

    return tag_config, sla_config


def run_triage(
    thread_id: str,
    intent_label: str,
    escalation_tier: str,
    client: HiverClient,
    tag_config: TagConfig,
    sla_config: SLAConfig,
) -> TriageResult:
    """Run Hiver triage for a thread, executing tagging and SLA timer start independently.

    Business Invariants:
        1. Tagging and SLA initiation execute INDEPENDENTLY. If tagging succeeds but SLA
           timer fails (or vice versa), each outcome is preserved distinctly.
        2. Hiver sync failures NEVER raise uncaught exceptions to block or reverse upstream
           decisions made in M2 (intent/escalation routing).
        3. Specific error descriptions are captured in TriageResult.errors.

    Args:
        thread_id: Unique identifier for the Hiver thread.
        intent_label: Intent label from M2.P2.2.
        escalation_tier: Escalation tier from M2.P2.3 ("critical", "urgent", "standard", "none").
        client: HiverClient implementation (HiverAPIClient or MockHiverClient).
        tag_config: TagConfig defining mapping and fallback tag.
        sla_config: SLAConfig defining duration per escalation tier.

    Returns:
        TriageResult detailing tags applied, SLA minutes, distinct success flags, and errors.
    """
    errors: list[str] = []
    tags_applied: list[str] = []
    tagging_success = False

    # Sub-operation 1: Tag derivation and application
    try:
        tags_applied = derive_tags(intent_label, escalation_tier, tag_config)
        applied = apply_triage_tags(thread_id, tags_applied, client)
        if applied:
            tagging_success = True
            logger.info("Successfully applied tags %s to thread %s", tags_applied, thread_id)
        else:
            tagging_success = False
            err_msg = f"Client failed to apply tags {tags_applied} to thread {thread_id}."
            errors.append(err_msg)
            logger.warning(err_msg)
    except Exception as exc:
        tagging_success = False
        err_msg = f"Unexpected error during tagging for thread {thread_id}: {exc}"
        errors.append(err_msg)
        logger.exception(err_msg)

    # Sub-operation 2: SLA duration derivation and timer initiation
    sla_minutes: Optional[int] = None
    sla_success = False

    try:
        duration = derive_sla_duration(escalation_tier, sla_config)
        sla_minutes = duration
        started = start_sla_for_thread(thread_id, duration, client)
        if started:
            sla_success = True
            logger.info("Successfully initiated %dm SLA timer for thread %s", duration, thread_id)
        else:
            sla_success = False
            err_msg = f"Client failed to start SLA timer ({duration}m) for thread {thread_id}."
            errors.append(err_msg)
            logger.warning(err_msg)
    except ConfigurationError as ce:
        sla_success = False
        err_msg = f"SLA configuration error for tier '{escalation_tier}': {ce}"
        errors.append(err_msg)
        logger.warning(err_msg)
    except Exception as exc:
        sla_success = False
        err_msg = f"Unexpected error starting SLA timer for thread {thread_id}: {exc}"
        errors.append(err_msg)
        logger.exception(err_msg)

    return TriageResult(
        thread_id=thread_id,
        tags_applied=tags_applied,
        sla_minutes=sla_minutes,
        tagging_success=tagging_success,
        sla_success=sla_success,
        errors=errors,
    )
