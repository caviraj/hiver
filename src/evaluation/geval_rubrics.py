"""Rubric definitions for G-Eval holistic subjective scoring.

Implements anchored rubrics for subjective dimensions (brand tone, empathy, structural coherence)
following G-Eval methodology with explicit criteria and discriminative 1-3-5 anchors.
"""

from typing import Any, Dict, List

RUBRICS: Dict[str, Dict[str, Any]] = {
    "brand_tone": {
        "name": "brand_tone",
        "description": "Adherence to professional, courteous, authoritative, and brand-aligned communication.",
        "criteria": (
            "Evaluate whether the response adheres to professional, courteous, authoritative, "
            "and brand-aligned communication standards. Consider language appropriateness, respectfulness, "
            "warmth, professionalism, and absence of slang, sarcasm, or dismissiveness."
        ),
        "anchors": {
            1: (
                "Severe brand misalignment. The response uses unprofessional, dismissive, sarcastic, "
                "or overly casual/slang language. It violates brand identity, sounds hostile or careless, "
                "and undermines trust."
            ),
            3: (
                "Moderate brand alignment. The response is generally neutral, polite, and acceptable, "
                "but lacks distinctive brand warmth, polish, or authority. It may sound slightly mechanical, "
                "stiff, or overly transactional."
            ),
            5: (
                "Exemplary brand alignment. The response flawlessly embodies the brand persona: courteous, "
                "articulate, reassuring, and highly professional. Tone is perfectly calibrated, engaging, "
                "and inspires utmost confidence."
            ),
        },
    },
    "empathy": {
        "name": "empathy",
        "description": "Recognition, validation, and compassionate response to user situation, emotion, and frustration.",
        "criteria": (
            "Evaluate how effectively the response recognizes, validates, and responds to the user's "
            "emotional state, concerns, frustration, or distress with genuine empathy, active listening, "
            "and supportive framing."
        ),
        "anchors": {
            1: (
                "Completely devoid of empathy. The response is cold, robotic, dismissive, or blatantly ignores "
                "user distress, urgency, or frustration. It invalidates the user's experience and feels hostile or indifferent."
            ),
            3: (
                "Superficial or formulaic empathy. The response includes generic canned apology phrases "
                "(e.g., 'We apologize for any inconvenience') without personalizing or truly addressing the user's "
                "specific emotional context or pain points."
            ),
            5: (
                "Deep, authentic empathy. The response actively acknowledges and validates the user's specific feelings "
                "and situation with sincere warmth, proactive reassurance, and compassionate, solution-oriented advocacy."
            ),
        },
    },
    "structural_coherence": {
        "name": "structural_coherence",
        "description": "Logical progression, clarity of organization, formatting, and seamless readability.",
        "criteria": (
            "Evaluate the logical organization, progression of ideas, transition clarity, formatting, "
            "and overall readability of the response. The text should flow logically from problem statement "
            "to resolution without confusion or disjointed transitions."
        ),
        "anchors": {
            1: (
                "Disorganized and incoherent. The response contains fragmented thoughts, conflicting statements, "
                "confusing sequencing, or unformatted text walls that obscure the core message and create severe cognitive friction."
            ),
            3: (
                "Passable but rough structure. The main points are understandable, but transitions are abrupt, "
                "the ordering could be significantly improved, or key takeaways are buried within dense paragraphs."
            ),
            5: (
                "Impeccably structured and coherent. Ideas flow effortlessly with logical sequencing, crisp transitions, "
                "and effective formatting (clear paragraphs, bulleting where helpful). Reading is effortless and immediately actionable."
            ),
        },
    },
}


def list_rubrics() -> List[str]:
    """Return a list of all registered rubric names."""
    return sorted(list(RUBRICS.keys()))


def get_rubric(rubric_name: str) -> Dict[str, Any]:
    """Retrieve rubric definition by name.

    Args:
        rubric_name: The identifier of the rubric to retrieve.

    Returns:
        Dict containing name, description, criteria, and anchors.

    Raises:
        ValueError: If rubric_name is not registered, listing available rubrics.
    """
    if rubric_name not in RUBRICS:
        available = ", ".join(f"'{r}'" for r in list_rubrics())
        raise ValueError(
            f"Unknown rubric '{rubric_name}'. Available rubrics are: [{available}]."
        )
    return RUBRICS[rubric_name]
