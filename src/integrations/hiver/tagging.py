"""Tag derivation and application logic for Hiver threads.

Phase: M6.P6.1.F1
Maps intent labels and escalation tiers to categorical tags,
handles dynamic taxonomy fallback gracefully, and applies tags via HiverClient.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from src.integrations.hiver.client import HiverClient
from src.integrations.hiver.schema import TagConfig

logger = logging.getLogger(__name__)


def derive_tags(intent_label: str, escalation_tier: str, config: TagConfig) -> List[str]:
    """Derive categorical Hiver tags from intent label and escalation tier.

    Lookup Order:
        1. Composite key "{intent_label}:{escalation_tier}" (for tier-specific overrides)
        2. Base key "{intent_label}" (general intent mapping)
        3. Fallback to [config.fallback_tag] if neither is present, logging a warning.

    Special Rule:
        Always appends "Urgent" when escalation_tier is "critical" or "urgent"
        (case-insensitive), preserving existing tags without duplicates.

    Args:
        intent_label: Intent classification from M2.P2.2 (e.g., "Billing_Issue").
        escalation_tier: Escalation tier from M2.P2.3 ("critical", "urgent", "standard", "none").
        config: TagConfig containing tag mapping and fallback definition.

    Returns:
        List of tag strings to be applied.
    """
    mapping = config.mapping or {}
    composite_key = f"{intent_label}:{escalation_tier}"

    if composite_key in mapping:
        tags = list(mapping[composite_key])
    elif intent_label in mapping:
        tags = list(mapping[intent_label])
    else:
        logger.warning(
            "Unrecognized intent_label '%s' not present in Hiver TagConfig; "
            "falling back to tag '%s'. Hiver tag configuration should be updated for this induced taxonomy label.",
            intent_label,
            config.fallback_tag,
        )
        tags = [config.fallback_tag]

    # Always append "Urgent" when escalation tier is critical or urgent
    norm_tier = (escalation_tier or "").strip().lower()
    if norm_tier in ("critical", "urgent"):
        if "Urgent" not in tags:
            tags.append("Urgent")

    return tags


def apply_triage_tags(thread_id: str, tags: Sequence[str], client: HiverClient) -> bool:
    """Apply triage tags to a Hiver thread via the client interface.

    Retry and backoff are handled inside the client implementation.

    Args:
        thread_id: Target Hiver thread identifier.
        tags: Sequence of tags to attach.
        client: HiverClient instance (API or Mock).

    Returns:
        True if tags were applied successfully, False otherwise.
    """
    if not tags:
        logger.debug("No tags to apply for thread %s", thread_id)
        return True
    return client.apply_tags(thread_id, tags)
