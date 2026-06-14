"""Structured query planner for Phase 2 of the sales analysis agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.planner.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from agent.planner.schemas import QueryPlan
from agent.rag.retriever import format_rag_context_for_prompt
from agent.utils.conversation_helpers import build_conversation_snapshot_block

logger = logging.getLogger(__name__)


def build_query_plan(
    user_prompt: str,
    *,
    conversation_id: str | None = None,
    llm_client: Any,
    rag_context: dict[str, Any] | None = None,
) -> QueryPlan:
    """Create a structured plan for a sales analysis request.

    The planner is intentionally advisory. Downstream KQL generation still
    enforces schema, date, and access-scope rules.
    """

    conversation_snapshot = build_conversation_snapshot_block(conversation_id)
    prompt = build_planner_prompt(
        user_prompt,
        conversation_snapshot,
        format_rag_context_for_prompt(rag_context, max_documents=4, max_chars=4000),
    )
    response = llm_client.invoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    raw_content = getattr(response, "content", response)
    payload = _extract_json_object(str(raw_content or ""))
    return QueryPlan.from_dict(payload)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    logger.debug("Planner returned non-JSON content: %s", raw_text)
    raise ValueError("Planner response did not contain a JSON object.")
