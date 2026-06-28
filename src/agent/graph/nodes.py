"""LangGraph nodes for the SAP sales analysis workflow."""

from __future__ import annotations

import logging
from typing import Any

from agent.agent import handle_user_query, llm
from agent.graph.state import SalesAgentState
from agent.planner.planner import build_query_plan
from agent.rag.retriever import retrieve_sales_knowledge_context

logger = logging.getLogger(__name__)


def _append_event(
    state: SalesAgentState,
    event: str,
    message: str,
    **payload: Any,
) -> list[dict[str, Any]]:
    """Return the current event list with one additional workflow event."""

    events = list(state.get("events", []))
    event_payload: dict[str, Any] = {"event": event, "message": message}
    event_payload.update(payload)
    events.append(event_payload)
    return events


def initialize_request_node(state: SalesAgentState) -> SalesAgentState:
    """Prepare graph state for the existing request pipeline."""

    return {
        "events": _append_event(
            state,
            "status",
            "Analyzing request",
        ),
    }


def retrieve_business_context_node(state: SalesAgentState) -> SalesAgentState:
    """Retrieve advisory SAP business context from Azure AI Search."""

    try:
        rag_context = retrieve_sales_knowledge_context(state["prompt"]).to_dict()
    except Exception as exc:
        logger.warning("RAG business context retrieval failed; continuing without it.")
        return {
            "rag_context_error": str(exc),
            "events": _append_event(
                state,
                "status",
                "Business context retrieval unavailable",
            ),
        }

    documents = rag_context.get("documents", [])
    if documents:
        return {
            "rag_context": rag_context,
            "events": _append_event(
                state,
                "status",
                "Business context retrieved",
                context_count=len(documents),
            ),
        }

    error = str(rag_context.get("error") or "")
    if rag_context.get("source") == "disabled":
        event_message = "Business context retrieval skipped"
    elif error:
        event_message = "Business context retrieval unavailable"
    else:
        event_message = "No external business context found"
    payload: SalesAgentState = {
        "events": _append_event(
            state,
            "status",
            event_message,
        ),
    }
    if error:
        payload["rag_context_error"] = error
    return payload


def create_query_plan_node(state: SalesAgentState) -> SalesAgentState:
    """Create an advisory structured plan before KQL generation."""

    try:
        query_plan = build_query_plan(
            state["prompt"],
            conversation_id=state.get("conversation_id"),
            llm_client=llm,
            rag_context=state.get("rag_context"),
        ).to_dict()
    except Exception as exc:
        logger.warning("Structured query planning failed; continuing without a plan.")
        return {
            "query_plan_error": str(exc),
            "events": _append_event(
                state,
                "status",
                "Structured query planning unavailable",
            ),
        }

    return {
        "query_plan": query_plan,
        "events": _append_event(
            state,
            "status",
            "Structured query plan created",
        ),
    }


def execute_existing_agent_node(state: SalesAgentState) -> SalesAgentState:
    """Run the current agent implementation without changing its behavior."""

    try:
        result = handle_user_query(
            state["prompt"],
            conversation_id=state.get("conversation_id"),
            user=state.get("user"),
            query_plan=state.get("query_plan"),
            rag_context=state.get("rag_context"),
            on_token=state.get("on_token"),
        )
    except Exception as exc:
        logger.exception("Sales agent graph execution failed.")
        return {
            "error": str(exc),
            "events": _append_event(
                state,
                "error",
                "Agent execution failed",
            ),
        }

    return {
        "result": result,
        "events": _append_event(
            state,
            "status",
            "Preparing response",
        ),
    }


def finalize_response_node(state: SalesAgentState) -> SalesAgentState:
    """Extract a best-effort answer for downstream API and SSE callers."""

    if state.get("error"):
        return {
            "answer": "",
            "events": _append_event(
                state,
                "final",
                "Completed with error",
            ),
        }

    result = state.get("result")
    answer = ""
    final_payload: dict[str, Any] = {}

    if isinstance(result, str):
        answer = result.strip()
    elif isinstance(result, dict):
        for key in ("answer", "final_answer", "message", "text", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                answer = value.strip()
                break
        final_payload["data"] = result.get("result")
        final_payload["operation_plan"] = result.get("operation_plan")

    final_payload["answer"] = answer
    return {
        "answer": answer,
        "events": _append_event(
            state,
            "final",
            "Analysis completed",
            **final_payload,
        ),
    }
