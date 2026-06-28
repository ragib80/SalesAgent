"""
Fail-safe audit writer for AgentQueryAudit records.

All public functions swallow exceptions so a logging failure never
interrupts the main chat request.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Approximate Azure OpenAI cost rates (USD per 1 K tokens).
# Override via env vars OPENAI_COST_PER_1K_INPUT / OPENAI_COST_PER_1K_OUTPUT.
_COST_PER_1K_INPUT: float = 0.0025
_COST_PER_1K_OUTPUT: float = 0.010


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Return a rough USD cost estimate based on GPT-4o pricing."""
    import os

    rate_in = float(os.getenv("OPENAI_COST_PER_1K_INPUT", str(_COST_PER_1K_INPUT)))
    rate_out = float(os.getenv("OPENAI_COST_PER_1K_OUTPUT", str(_COST_PER_1K_OUTPUT)))
    return round((prompt_tokens * rate_in + completion_tokens * rate_out) / 1000, 6)


def write_audit_record(**kwargs: Any) -> None:
    """Persist one AgentQueryAudit row; never raises."""
    try:
        from agent.models import AgentQueryAudit  # late import avoids circular refs

        AgentQueryAudit.objects.create(**kwargs)
    except Exception:
        logger.exception("Failed to write audit record — audit data lost for this request.")
