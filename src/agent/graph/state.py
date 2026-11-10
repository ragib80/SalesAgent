"""State definitions for the SAP sales analysis LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class SalesAgentState(TypedDict, total=False):
    """Shared state passed between sales analysis LangGraph nodes."""

    prompt: str
    conversation_id: str | None
    user: Any
    rag_context: dict[str, Any]
    rag_context_error: str
    query_plan: dict[str, Any]
    query_plan_error: str
    result: Any
    answer: str
    error: str
    events: list[dict[str, Any]]
    on_token: Any
    result_cols: list[str] | None
    result_rows: list[list] | None
    result_total_rows: int | None
    result_kql: str | None
    result_chart_meta: dict | None
