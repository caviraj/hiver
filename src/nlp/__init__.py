"""NLP preprocessing and normalization package."""

from typing import Any

from src.nlp.contraction_map import CONTRACTIONS

__all__ = [
    "CONTRACTIONS",
    "expand_contractions",
    "expand_thread",
    "mask_urls",
    "preserve_device_entities",
    "mask_entities",
    "mask_thread",
    "run",
]


def __getattr__(name: str) -> Any:
    if name in {"expand_contractions", "expand_thread"}:
        import src.nlp.decontraction as decontraction

        return getattr(decontraction, name)
    if name in {
        "mask_urls",
        "preserve_device_entities",
        "mask_entities",
        "mask_thread",
    }:
        import src.nlp.entity_masking as entity_masking

        return getattr(entity_masking, name)
    if name == "run":
        import src.nlp.decontraction as decontraction

        return getattr(decontraction, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

