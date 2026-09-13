"""Golden dataset schema, loading, and stratified sampling.

Defines the GoldenExample data structure for evaluation ground truth,
enforces statistically sufficient dataset sizing (minimum 150 examples),
and provides stratified sampling utilities across categorical metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GoldenExample(BaseModel):
    """Schema for a human-labeled golden evaluation example."""

    example_id: str = Field(..., description="Unique identifier for the evaluation item")
    query: str = Field(..., description="User query or prompt")
    retrieved_contexts: List[str] = Field(
        default_factory=list, description="Passages retrieved for RAG grounding"
    )
    generated_response: str = Field(..., description="System generated response under evaluation")
    human_score: Optional[int] = Field(
        default=None,
        description="Ground truth human rating on an ordinal 1-5 scale, or None if unannotated",
    )
    annotator_id: str = Field(..., description="Identifier of the annotator who provided the score")
    rubric_name: str = Field(..., description="Evaluation dimension or rubric applied (e.g. brand_tone)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata such as intent category or severity tier"
    )

    @field_validator("human_score")
    @classmethod
    def validate_score_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 5):
            raise ValueError(f"human_score must be an integer between 1 and 5 (inclusive), got {v}")
        return v


def load_golden_set(path: str | Path) -> List[GoldenExample]:
    """Load a golden evaluation dataset from a JSON or JSONL file.

    Enforces a strict statistical minimum of 150 examples per PRD specification.
    Datasets with fewer than 150 examples raise a ValueError to prevent premature
    judge calibration against an under-powered test set. Datasets with more than
    250 examples emit a warning indicating over-provisioning.

    Args:
        path: Filepath to the JSON or JSONL dataset file.

    Returns:
        List of GoldenExample instances.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If file format is unrecognized or contains fewer than 150 examples.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden dataset file not found at: {file_path}")

    raw_items: List[Dict[str, Any]] = []
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    raw_items.append(json.loads(clean_line))
                except json.JSONDecodeError as err:
                    raise ValueError(f"Malformed JSON on line {line_no} of {file_path}: {err}") from err
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    raw_items = loaded
                elif isinstance(loaded, dict) and "examples" in loaded:
                    raw_items = loaded["examples"]
                else:
                    raise ValueError(
                        f"Expected JSON root to be a list or a dict with an 'examples' key in {file_path}"
                    )
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON in {file_path}: {err}") from err
    else:
        raise ValueError(f"Unsupported golden dataset file extension: {suffix}. Expected .json or .jsonl")

    examples = [GoldenExample(**item) for item in raw_items]
    n_examples = len(examples)

    # Hard architectural requirement: >= 150 examples
    if n_examples < 150:
        raise ValueError(
            f"Golden dataset contains {n_examples} examples, which is statistically insufficient. "
            "A minimum of 150 examples is strictly required for reliable judge calibration."
        )

    # Soft warning for over-provisioning: > 250 examples
    if n_examples > 250:
        logger.warning(
            "Golden dataset contains %d examples (exceeding recommended 250 examples). "
            "Over-provisioning is acceptable but increases annotation and evaluation cost.",
            n_examples,
        )

    return examples


def stratified_sample(
    pool: List[T],
    target_size: int,
    stratify_by: str,
    seed: Optional[int] = None,
) -> List[T]:
    """Sample a representative subset across strata defined by a metadata field or attribute.

    Partitions items in the pool by `stratify_by`, computing proportional allocations
    for each non-empty stratum. Warns (does not error) if any stratum has zero examples.

    Args:
        pool: List of candidate items (GoldenExample, EvalItem, or dicts).
        target_size: Total number of items desired in the resulting sample.
        stratify_by: Attribute name or metadata key to stratify across.
        seed: Optional random seed for reproducible sampling.

    Returns:
        List of sampled items preserving proportional representation.
    """
    import random

    if not pool:
        return []

    if target_size <= 0:
        return []

    if target_size >= len(pool):
        return list(pool)

    rng = random.Random(seed)

    # Group items by stratum key
    strata: Dict[str, List[T]] = {}
    for item in pool:
        val: Any = None
        if isinstance(item, dict):
            val = item.get("metadata", {}).get(stratify_by) or item.get(stratify_by)
        elif hasattr(item, "metadata") and isinstance(getattr(item, "metadata"), dict):
            val = getattr(item, "metadata").get(stratify_by) or getattr(item, stratify_by, None)
        elif hasattr(item, stratify_by):
            val = getattr(item, stratify_by)

        stratum_key = str(val) if val is not None else "unknown"
        strata.setdefault(stratum_key, []).append(item)

    total_pool_size = len(pool)
    sampled: List[T] = []

    # Calculate proportional allocation per stratum
    allocated_counts: Dict[str, int] = {}
    remaining_slots = target_size

    for key, items in strata.items():
        if len(items) == 0:
            logger.warning("Stratum '%s' has 0 available examples in the candidate pool.", key)
            allocated_counts[key] = 0
            continue

        prop = len(items) / total_pool_size
        count = int(round(prop * target_size))
        # Ensure at least 1 item per non-empty stratum if possible
        count = max(1, min(count, len(items)))
        allocated_counts[key] = count

    # Adjust counts so total matches target_size
    current_total = sum(allocated_counts.values())
    sorted_strata = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)

    while current_total > target_size:
        for k in sorted_strata:
            if allocated_counts[k] > 1:
                allocated_counts[k] -= 1
                current_total -= 1
                if current_total == target_size:
                    break
        else:
            break

    while current_total < target_size:
        for k in sorted_strata:
            if allocated_counts[k] < len(strata[k]):
                allocated_counts[k] += 1
                current_total += 1
                if current_total == target_size:
                    break
        else:
            break

    # Sample within each stratum
    for key, items in strata.items():
        k = allocated_counts.get(key, 0)
        if k > 0:
            stratum_sample = rng.sample(items, min(k, len(items)))
            sampled.extend(stratum_sample)

    rng.shuffle(sampled)
    return sampled
