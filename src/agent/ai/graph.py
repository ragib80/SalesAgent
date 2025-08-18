from typing import TypedDict, Annotated, Optional, List, Dict
from django.conf import settings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
import json

# Reuse your existing agent pieces
from agent.agent import generate_kql, adx, GSBER_MAPPING
from .frame_types import Frame, FrameDelta

# -------- LLMs (same Azure OpenAI setup you already use) --------
intent_llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0,
)
# delta_llm = intent_llm.with_structured_output(FrameDelta)
delta_llm = intent_llm.with_structured_output(FrameDelta, method="function_calling")


summary_llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0.2,
)

# -------- Graph State --------
class ConvState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    frame: Frame
    kql: Optional[str]
    rows: Optional[List[Dict]]
    meta: Optional[Dict]
    error: Optional[str]
    text: Optional[str]   # final assistant text

INTENT_SYS = SystemMessage(content=
"""You are a sales analytics intent parser for SAP sales on Azure Data Explorer (ADX).
Given the user's latest message and the prior frame, output ONLY a JSON delta with keys you need to change.
If the user says 'same' or 'previous', omit those keys (we’ll reuse them). Never output KQL."""
)

SUMMARY_SYS = SystemMessage(content=
"""You are a precise sales analyst. Summarize ADX results as bullet points showing:
- Date range, scope (depo/zone/region/dealer), metric, grouping, Top N and totals.
- 1–2 crisp business insights at the end. Currency: BDT.
If rows include a dimension (brand/dealer/etc.), list the top items with values."""
)

def deep_merge_frame(base: Frame, delta: FrameDelta) -> Frame:
    data = base.model_dump()
    d = (delta or FrameDelta()).model_dump(exclude_none=True)
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    return Frame(**data)

def frame_defaults_as_hints(frame: Frame) -> str:
    bits = []
    if frame.date.start and frame.date.end:
        bits.append(f"Date range: {frame.date.start} to {frame.date.end}")
    if frame.scope.depo:
        human = next((h for h,c in GSBER_MAPPING.items() if c == frame.scope.depo or h.lower()==str(frame.scope.depo).lower()), None)
        if human and human != frame.scope.depo:
            bits.append(f"Depo/Sales Office (gsber): {human} ({frame.scope.depo})")
        else:
            bits.append(f"Depo/Sales Office (gsber): {frame.scope.depo}")
    if frame.scope.zone:    bits.append(f"Zone (Szone): {frame.scope.zone}")
    if frame.scope.region:  bits.append(f"Region: {frame.scope.region}")
    if frame.scope.dealer:  bits.append(f"Dealer: {frame.scope.dealer}")
    if frame.entities.brand:   bits.append(f"Brand: {frame.entities.brand}")
    if frame.entities.product: bits.append(f"Product: {frame.entities.product}")
    if frame.group_by: bits.append(f"Group by: {frame.group_by}")
    if frame.limit:    bits.append(f"Top N: {frame.limit}")
    return "Assume these defaults if not stated:\n- " + "\n- ".join(bits) if bits else ""

# -------- Nodes --------
def n_extract_delta(state: ConvState) -> ConvState:
    prior = state["frame"]
    user_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    delta = delta_llm.invoke([INTENT_SYS, HumanMessage(content=f"PRIOR_FRAME:\n{prior.model_dump_json()}\n\nUSER:\n{user_text}")])
    state["messages"].append(AIMessage(content=f"[delta]{delta.model_dump_json()}"))
    state["frame"] = deep_merge_frame(prior, delta)
    return state

def n_gen_kql(state: ConvState) -> ConvState:
    if state.get("error"): return state
    frame = state["frame"]
    user_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = frame_defaults_as_hints(frame)
    effective_prompt = user_text if not ctx else f"{user_text}\n\n{ctx}"
    try:
        state["kql"] = generate_kql(effective_prompt)  # your existing function
    except Exception as ex:
        state["error"] = f"generate_kql failed: {ex}"
    return state

def n_run_adx(state: ConvState) -> ConvState:
    if state.get("error"): return state
    try:
        cols, rows = adx().run(state["kql"])
        state["rows"] = [dict(zip(cols, r)) for r in rows]
        state["meta"] = {"rowcount": len(rows), "cols": cols}
    except Exception as ex:
        state["error"] = f"ADX error: {ex}"
    return state

def n_summarize(state: ConvState) -> ConvState:
    frame = state["frame"]; rows = state.get("rows") or []; meta = state.get("meta") or {}
    preview = rows[:50]
    msg = summary_llm.invoke([
        SUMMARY_SYS,
        HumanMessage(content=f"Frame:\n{frame.model_dump_json()}\n\nKQL:\n{state.get('kql')}\n\nMeta:\n{json.dumps(meta)}\n\nRows (sample):\n{json.dumps(preview)}")
    ])
    state["messages"].append(AIMessage(content=msg.content))
    state["text"] = msg.content
    return state

def build_graph():
    g = StateGraph(ConvState)
    g.add_node("extract_delta", n_extract_delta)
    g.add_node("gen_kql", n_gen_kql)
    g.add_node("run_adx", n_run_adx)
    g.add_node("summarize", n_summarize)

    g.set_entry_point("extract_delta")
    g.add_edge("extract_delta", "gen_kql")
    g.add_edge("gen_kql", "run_adx")
    g.add_edge("run_adx", "summarize")
    g.add_edge("summarize", END)
    return g
