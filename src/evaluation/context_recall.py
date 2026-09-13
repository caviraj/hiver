"""Context Recall metric computation via reference decomposition and attribution check."""

import json
import logging
import re
from typing import Any, List
from src.evaluation.relevance_judge import _call_llm, judge_relevance, RelevanceJudgeError

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are an expert factual analyst.
Break down the following reference answer into a concise list of atomic, independent factual statements.
Each statement must contain exactly one discrete fact.

Reference Answer:
{reference}

Respond ONLY with a JSON array of strings, where each element is an atomic fact.
Example format:
["Fact 1", "Fact 2"]
Facts:"""

ATTRIBUTION_PROMPT = """You are an objective evaluation judge.
Determine whether the following Atomic Fact is directly supported or stated in the provided Context.

Atomic Fact:
{fact}

Context:
{context}

Respond ONLY with "YES" if the fact is supported by the context, or "NO" if it is not supported.
Verdict:"""


def decompose_reference(reference_answer: str, llm_client: Any) -> List[str]:
    """Decompose a reference ground truth answer into discrete atomic factual statements.

    Uses temperature=0 for reproducible extraction.

    Args:
        reference_answer: Ground truth reference answer.
        llm_client: LLM client wrapper or callable.

    Returns:
        List[str]: List of atomic factual statements.
    """
    if not reference_answer or not reference_answer.strip():
        return []

    prompt = DECOMPOSE_PROMPT.format(reference=reference_answer.strip())

    try:
        raw_response = _call_llm(llm_client, prompt).strip()
        # Attempt JSON parse
        # If response has code fencing like ```json ... ```, strip it
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                facts = [str(item).strip() for item in parsed if str(item).strip()]
                if facts:
                    return facts
        except json.JSONDecodeError:
            pass

        # Fallback: line-by-line parsing (bullets, numbering, or plain lines)
        facts = []
        for line in raw_response.splitlines():
            line = line.strip()
            line = re.sub(r"^[\*\-\d\.\)\s]+", "", line).strip()
            # Strip quotes if present
            line = line.strip('"\'')
            if line:
                facts.append(line)
        return facts if facts else [reference_answer.strip()]

    except Exception as e:
        logger.warning("Failed to decompose reference answer with LLM: %s. Using sentence fallback.", e)
        # Fallback to simple sentence splitting
        sentences = [s.strip() for s in re.split(r"[.!?]+", reference_answer) if s.strip()]
        return sentences if sentences else [reference_answer.strip()]


def _is_fact_supported_by_chunk(fact: str, context_chunk: str, llm_client: Any) -> bool:
    """Check if an atomic fact is supported by a given context chunk."""
    try:
        return judge_relevance(
            query_or_reference=fact,
            context_chunk=context_chunk,
            llm_client=llm_client,
        )
    except RelevanceJudgeError:
        return False


def compute_context_recall(
    reference_answer: str,
    retrieved_contexts: List[str],
    llm_client: Any,
    batch_chunk_size: int = 3,
) -> float:
    """Compute Context Recall by checking what fraction of reference facts are attributed to retrieved contexts.

    A fact is considered recalled if it is supported by ANY retrieved context.
    To prevent context-window overflow when retrieved_contexts is very large,
    contexts are evaluated in batches of `batch_chunk_size` (default 3 chunks),
    short-circuiting as soon as any batch supports the fact.

    Formula:
        Context Recall = count(supported_facts) / total_facts

    Edge cases:
        - If retrieved_contexts is empty: returns 0.0 (nothing retrieved = zero recall).
        - If reference_answer is empty or yields 0 facts: returns 0.0.

    Args:
        reference_answer: Ground truth reference answer.
        retrieved_contexts: List of retrieved context chunks.
        llm_client: LLM client wrapper or callable.
        batch_chunk_size: Number of context chunks grouped per attribution check.

    Returns:
        float: Context Recall score in [0.0, 1.0].
    """
    if not retrieved_contexts or not reference_answer.strip():
        return 0.0

    facts = decompose_reference(reference_answer, llm_client)
    if not facts:
        return 0.0

    # Group contexts into batches to stay well within LLM context window
    context_batches: List[str] = []
    for i in range(0, len(retrieved_contexts), batch_chunk_size):
        chunk_group = retrieved_contexts[i : i + batch_chunk_size]
        combined = "\n\n---\n\n".join(c.strip() for c in chunk_group if c.strip())
        if combined:
            context_batches.append(combined)

    if not context_batches:
        return 0.0

    supported_count = 0

    for fact in facts:
        is_supported = False
        for batch in context_batches:
            if _is_fact_supported_by_chunk(fact, batch, llm_client):
                is_supported = True
                break
        if is_supported:
            supported_count += 1

    return float(supported_count / len(facts))
