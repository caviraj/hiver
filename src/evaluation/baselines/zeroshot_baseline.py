"""Zero-shot raw LLM baseline responder.

ARCHITECTURE RESTRICTION & COMPLIANCE NOTICE:
This module is strictly FORBIDDEN from importing or invoking any project
retrieval, guardrail, or classification infrastructure (e.g., HybridRetriever,
DenseIndex, NemoGuardrailsController, intent classifiers, query expanders).
The primary purpose of this baseline is to evaluate ungrounded LLM performance
and quantify the baseline hallucination rate BEFORE RAG and guardrails are introduced.
Do NOT attempt to 'improve' or 'optimize' this baseline with context augmentation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

BASELINE_ERROR_PREFIX: str = "[BASELINE_ERROR]"
DEFAULT_SYSTEM_PROMPT: str = "You are a helpful customer support agent."


def is_baseline_error(response: Optional[str]) -> bool:
    """Check if a response text represents a recorded baseline execution error."""
    if not response:
        return False
    return response.startswith(BASELINE_ERROR_PREFIX)


def zero_shot_baseline_response(
    tweet_text: str,
    llm_client: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate a raw ungrounded zero-shot LLM response.

    Invokes the provided LLM client directly without retrieval, grounding contexts,
    moderation checks, or multi-turn conversational expansion.

    Args:
        tweet_text: Inbound customer tweet or inquiry.
        llm_client: LLM client or callable supporting standard chat or completion APIs.
        system_prompt: Generic system prompt without few-shot examples or domain policies.

    Returns:
        Generated response string, or a structured error marker string
        prefixed with '[BASELINE_ERROR]' if the call fails.
    """
    if llm_client is None:
        return f"{BASELINE_ERROR_PREFIX}: ValueError: llm_client cannot be None."

    if not tweet_text or not tweet_text.strip():
        return f"{BASELINE_ERROR_PREFIX}: ValueError: Empty input query provided."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": tweet_text},
    ]

    try:
        # 1. OpenAI-style client (client.chat.completions.create)
        if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
            response = llm_client.chat.completions.create(
                messages=messages,
                temperature=0.0,
            )
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content or ""
                if hasattr(choice, "text"):
                    return choice.text or ""

        # 2. Callable client (e.g. mock function or wrapper)
        elif callable(llm_client):
            res = llm_client(messages=messages)
            if isinstance(res, str):
                return res
            if hasattr(res, "content"):
                return res.content
            if hasattr(res, "text"):
                return res.text
            return str(res)

        # 3. Client with generate/predict/complete method
        elif hasattr(llm_client, "generate") and callable(getattr(llm_client, "generate")):
            res = llm_client.generate(messages=messages)
            if isinstance(res, str):
                return res
            if hasattr(res, "text"):
                return res.text
            if hasattr(res, "content"):
                return res.content

        raise AttributeError(f"Unsupported llm_client interface: {type(llm_client)}")

    except Exception as exc:
        logger.warning(
            "Zero-shot baseline LLM call failed: %s (%s)",
            type(exc).__name__,
            exc,
            exc_info=False,
        )
        return f"{BASELINE_ERROR_PREFIX}: {type(exc).__name__}: {str(exc)}"
