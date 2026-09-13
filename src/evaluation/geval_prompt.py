"""Prompt builder for G-Eval Chain-of-Thought evaluation.

Constructs structured evaluation prompts incorporating anchored rubrics, explicit
step-by-step reasoning instructions, and strict final-line score formatting.
"""

from typing import Optional

from src.evaluation.geval_rubrics import get_rubric


def build_geval_prompt(
    rubric_name: str,
    generated_response: str,
    context: Optional[str] = None,
) -> str:
    """Build a G-Eval prompt enforcing Chain-of-Thought reasoning and strict final score anchor.

    Args:
        rubric_name: Name of rubric in RUBRICS ('brand_tone', 'empathy', 'structural_coherence').
        generated_response: The generated response text to evaluate.
        context: Optional conversational or retrieval context.

    Returns:
        The full formatted prompt string for LLM evaluation.

    Raises:
        ValueError: If rubric_name is unknown, listing registered rubric options.
    """
    rubric = get_rubric(rubric_name)

    criteria = rubric["criteria"]
    anchors = rubric["anchors"]
    anchors_text = (
        f"- Score 1: {anchors[1]}\n"
        f"- Score 3: {anchors[3]}\n"
        f"- Score 5: {anchors[5]}"
    )

    context_section = ""
    if context and context.strip():
        context_section = f"\n[Context]\n{context.strip()}\n"

    prompt = (
        f"You are an expert evaluator assessing the quality of an AI-generated customer service response.\n\n"
        f"Evaluation Dimension: {rubric['name']}\n"
        f"Description: {rubric['description']}\n\n"
        f"Evaluation Criteria:\n{criteria}\n\n"
        f"Scoring Rubric Anchors:\n{anchors_text}\n"
        f"{context_section}\n"
        f"[Generated Response to Evaluate]\n{generated_response}\n\n"
        f"Instructions:\n"
        f"1. Evaluate the generated response against the criteria and rubric anchors above.\n"
        f"2. Provide your step-by-step Chain-of-Thought reasoning explaining the strengths and weaknesses "
        f"observed in the response.\n"
        f"3. Conclude with a single final integer score from 1 to 5 based on the rubric.\n\n"
        f"STRICT OUTPUT FORMAT REQUIREMENT:\n"
        f"Your response must end on its own final line with EXACTLY:\n"
        f"Final Score: X\n"
        f"where X is a single digit integer from 1 to 5. Do not append any other text, characters, or punctuation "
        f"after the score digit."
    )
    return prompt
