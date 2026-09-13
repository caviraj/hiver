"""
Guardrails Data Schemas
=======================

Pydantic models representing guardrail evaluation outcomes.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """
    Result of a NeMo Guardrails evaluation gate (input, dialog, or output rail).

    Attributes:
        passed: Whether the message/turn successfully passed all evaluated rails.
        triggered_rail: Which rail stage triggered a block or redirection ("input", "dialog", "output"),
                        or None if all evaluated rails passed.
        reason: Explanatory diagnostic string describing why the rail triggered.
        redirected_response: Canned redirect or fallback response if a rail was triggered, or None.
        thread_id: Unique conversation identifier tracking the dialog state session.
    """

    passed: bool = Field(
        ...,
        description="True if message passed guardrails; False if blocked or error occurred.",
    )
    triggered_rail: Optional[Literal["input", "dialog", "output"]] = Field(
        default=None,
        description="The specific rail stage that blocked or modified execution.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason or rule that caused the guardrail to fire.",
    )
    redirected_response: Optional[str] = Field(
        default=None,
        description="Canned redirect or safe fallback response returned to the user.",
    )
    thread_id: str = Field(
        default="default",
        description="Conversation thread identifier for state tracking.",
    )
