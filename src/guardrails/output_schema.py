"""
Guardrails Output Schema & Result Contracts
============================================

Defines the Pydantic models for the LLM output contract (AgentResponse)
and the Guardrails AI validation result (GuardValidationResult).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """
    Strict schema contract that backend ticketing APIs and customer-facing
    interfaces parse from LLM-generated output.
    """

    reply_text: str = Field(
        ...,
        description="The customer-facing reply message drafted by the agent.",
    )
    ticket_tags: List[str] = Field(
        default_factory=list,
        description="Ticket category/routing tags (e.g. ['device:sync', 'priority:normal']).",
    )
    requires_human_review: bool = Field(
        default=False,
        description="Flag indicating if the drafted response requires manual agent approval.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the drafted response (between 0.0 and 1.0).",
    )


class GuardValidationResult(BaseModel):
    """
    Outcome of the Guardrails AI schema and validator pipeline evaluation.
    """

    validated: bool = Field(
        ...,
        description="True if the output passed schema validation and all registered validators.",
    )
    response: Optional[AgentResponse] = Field(
        default=None,
        description="Parsed and validated AgentResponse object, or None if validation failed.",
    )
    escalate: bool = Field(
        default=False,
        description="True if validation failure requires hard escalation to human review.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Diagnostic message or summary of validation failure reasons.",
    )
    failed_validators: List[str] = Field(
        default_factory=list,
        description="List of all validators that failed on the evaluated output.",
    )
    retry_count: int = Field(
        default=0,
        description="Number of re-prompt attempts made before achieving resolution or exhaustion.",
    )
    failure_reasons: List[str] = Field(
        default_factory=list,
        description="Detailed list of specific failure messages for each failed validator.",
    )
    masked_pii_types: List[str] = Field(
        default_factory=list,
        description="Types of PII detected and masked (e.g. ['ssn', 'credit_card']).",
    )

