"""LLM-based binary relevance judge for RAGAS evaluation."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RelevanceJudgeError(Exception):
    """Raised when the relevance judge fails to produce a verdict after retries."""

    pass


RELEVANCE_JUDGE_PROMPT = """You are an objective evaluation judge.
Given the following Target (which may be a user query or a reference answer) and a retrieved Context Chunk, determine if the Context Chunk contains information directly relevant or useful to the Target.

Target:
{target}

Context Chunk:
{context}

Respond ONLY with "YES" if the context chunk is relevant, or "NO" if it is irrelevant.
Verdict:"""


def _call_llm(llm_client: Any, prompt: str) -> str:
    """Invoke the LLM client with temperature=0 for evaluation reproducibility."""
    if callable(llm_client):
        try:
            return str(llm_client(prompt, temperature=0))
        except TypeError:
            return str(llm_client(prompt))
    elif hasattr(llm_client, "generate") and callable(llm_client.generate):
        try:
            return str(llm_client.generate(prompt, temperature=0))
        except TypeError:
            return str(llm_client.generate(prompt))
    elif hasattr(llm_client, "chat") and callable(llm_client.chat):
        try:
            return str(llm_client.chat(prompt, temperature=0))
        except TypeError:
            return str(llm_client.chat(prompt))
    elif hasattr(llm_client, "complete") and callable(llm_client.complete):
        try:
            return str(llm_client.complete(prompt, temperature=0))
        except TypeError:
            return str(llm_client.complete(prompt))
    else:
        raise TypeError(f"Unsupported llm_client type: {type(llm_client)}")


def judge_relevance(
    query_or_reference: str,
    context_chunk: str,
    llm_client: Any,
) -> bool:
    """Determine binary relevance of a retrieved context chunk to a query or reference.

    Enforces temperature=0 for strict evaluation score reproducibility.
    Performs exactly one retry on transient failure before raising RelevanceJudgeError.

    Args:
        query_or_reference: Query string or reference ground truth answer.
        context_chunk: Retrieved text chunk to evaluate.
        llm_client: LLM client wrapper or callable.

    Returns:
        bool: True if relevant, False otherwise.

    Raises:
        RelevanceJudgeError: If the call fails on both initial attempt and retry.
    """
    prompt = RELEVANCE_JUDGE_PROMPT.format(
        target=query_or_reference.strip(),
        context=context_chunk.strip(),
    )

    last_error = None
    for attempt in range(2):
        try:
            raw_response = _call_llm(llm_client, prompt).strip().upper()
            if "YES" in raw_response:
                return True
            if "NO" in raw_response:
                return False
            # If neither YES nor NO was clearly returned, consider it a format failure
            raise ValueError(f"Unrecognized judge response: '{raw_response}' (expected YES or NO)")
        except Exception as e:
            last_error = e
            logger.warning(
                "Relevance judge attempt %d failed: %s. %s",
                attempt + 1,
                e,
                "Retrying once..." if attempt == 0 else "No more retries.",
            )

    raise RelevanceJudgeError(
        f"Relevance judge failed after 1 retry: {last_error}"
    ) from last_error
