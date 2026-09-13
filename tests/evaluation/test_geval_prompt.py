"""Unit tests for G-Eval prompt builder."""

import pytest

from src.evaluation.geval_prompt import build_geval_prompt
from src.evaluation.geval_rubrics import RUBRICS, list_rubrics


def test_build_geval_prompt_contains_all_rubrics():
    """Verify prompt builder successfully generates prompts for all registered rubrics."""
    rubric_names = list_rubrics()
    assert set(rubric_names) == {"brand_tone", "empathy", "structural_coherence"}

    for name in rubric_names:
        prompt = build_geval_prompt(
            rubric_name=name,
            generated_response="Thank you for reaching out. We are investigating your issue.",
        )
        assert RUBRICS[name]["criteria"] in prompt
        assert f"Evaluation Dimension: {name}" in prompt
        assert "- Score 1:" in prompt
        assert "- Score 3:" in prompt
        assert "- Score 5:" in prompt


def test_build_geval_prompt_enforces_cot_instructions():
    """Verify that the prompt explicitly requires step-by-step Chain-of-Thought reasoning."""
    prompt = build_geval_prompt(
        rubric_name="brand_tone",
        generated_response="Sample response text",
    )
    assert "Chain-of-Thought reasoning" in prompt
    assert "step-by-step" in prompt.lower()
    assert "strengths and weaknesses" in prompt.lower()


def test_build_geval_prompt_enforces_strict_final_score_format():
    """Verify that the prompt strictly commands the model to end with 'Final Score: X'."""
    prompt = build_geval_prompt(
        rubric_name="empathy",
        generated_response="Sample response text",
    )
    assert "STRICT OUTPUT FORMAT REQUIREMENT" in prompt
    assert "Final Score: X" in prompt
    assert "single digit integer from 1 to 5" in prompt


def test_build_geval_prompt_with_context():
    """Verify context is properly included in the prompt section when provided."""
    prompt = build_geval_prompt(
        rubric_name="structural_coherence",
        generated_response="First, restart the service. Second, check the logs.",
        context="User reported HTTP 500 error when clicking submit.",
    )
    assert "[Context]" in prompt
    assert "User reported HTTP 500 error when clicking submit." in prompt


def test_build_geval_prompt_without_context():
    """Verify context section is omitted when context is None or whitespace."""
    prompt_none = build_geval_prompt(
        rubric_name="brand_tone",
        generated_response="Sample response text",
        context=None,
    )
    assert "[Context]" not in prompt_none

    prompt_empty = build_geval_prompt(
        rubric_name="brand_tone",
        generated_response="Sample response text",
        context="   ",
    )
    assert "[Context]" not in prompt_empty


def test_build_geval_prompt_unknown_rubric_raises_value_error():
    """Verify ValueError is raised when an unknown rubric name is supplied, listing available options."""
    with pytest.raises(ValueError) as exc_info:
        build_geval_prompt(
            rubric_name="non_existent_dimension",
            generated_response="Sample response text",
        )
    err_msg = str(exc_info.value)
    assert "Unknown rubric 'non_existent_dimension'" in err_msg
    assert "brand_tone" in err_msg
    assert "empathy" in err_msg
    assert "structural_coherence" in err_msg
