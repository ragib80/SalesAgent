"""LangGraph workflow wrapper for the SAP sales analysis agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph

from agent.graph.nodes import (
    create_query_plan_node,
    execute_existing_agent_node,
    finalize_response_node,
    initialize_request_node,
    retrieve_business_context_node,
)
from agent.graph.state import SalesAgentState


@lru_cache(maxsize=1)
def get_sales_analysis_workflow():
    """Compile and cache the sales analysis graph."""

    graph = StateGraph(SalesAgentState)
    graph.add_node("initialize_request", initialize_request_node)
    graph.add_node("retrieve_business_context", retrieve_business_context_node)
    graph.add_node("create_query_plan", create_query_plan_node)
    graph.add_node("execute_existing_agent", execute_existing_agent_node)
    graph.add_node("finalize_response", finalize_response_node)

    graph.add_edge(START, "initialize_request")
    graph.add_edge("initialize_request", "retrieve_business_context")
    graph.add_edge("retrieve_business_context", "create_query_plan")
    graph.add_edge("create_query_plan", "execute_existing_agent")
    graph.add_edge("execute_existing_agent", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph.compile()


def run_sales_analysis_graph(
    prompt: str,
    *,
    conversation_id: str | None = None,
    user: Any = None,
) -> SalesAgentState:
    """Run the LangGraph wrapper and return the final workflow state."""

    initial_state: SalesAgentState = {
        "prompt": prompt,
        "conversation_id": conversation_id,
        "user": user,
        "events": [],
    }
    return get_sales_analysis_workflow().invoke(initial_state)


def stream_sales_analysis_graph(
    prompt: str,
    *,
    conversation_id: str | None = None,
    user: Any = None,
    on_token: Any = None,
) -> Iterator[dict[str, Any]]:
    """Yield workflow events for SSE callers as graph nodes complete."""

    seen_count = 0
    initial_state: SalesAgentState = {
        "prompt": prompt,
        "conversation_id": conversation_id,
        "user": user,
        "events": [],
        "on_token": on_token,
    }

    for update in get_sales_analysis_workflow().stream(initial_state):
        for node_state in update.values():
            events = node_state.get("events", [])
            for event in events[seen_count:]:
                yield event
            seen_count = max(seen_count, len(events))
