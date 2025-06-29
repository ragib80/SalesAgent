
from __future__ import annotations

import json
import math
import collections
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
from django.conf import settings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent, AgentType
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from django.core.cache import cache
from langgraph.checkpoint.base import BaseCheckpointSaver


import threading
# ────────────────────────────────────────────────────────────────────────────────
# 1. COLUMN MAP (add / rename here only)
# ────────────────────────────────────────────────────────────────────────────────
COLUMN_MAP: Dict[str, str] = {
    "id": "Id",
    "created_time": "CreatedTime",
    "modified_time": "ModifiedTime",
    "company_code": "bukrs",
    "division": "spart",
    "division_name": "spart_text",
    "material_group": "matkl",
    "brand": "wgbez",
    "material_number": "matnr",
    "sales_org": "vkorg",
    "customer_number": "kunrg",
    "ship_to_customer_number": "kunnr_sh",
    "payer_number": "Payer_DL",
    "sales_document_number": "vbeln",
    "sales_office_code": "vkbur_c",
    "sales_group_code": "vkgrp_c",
    "customer_classification": "kukla",
    "billing_date": "fkdat",
    "item_position": "posnr",
    "product_description": "arktx",
    "unit_of_measure": "meins",
    "volume_unit": "voleh",
    "territory": "Territory",
    "sales_zone": "Szone",
    "customer_name": "cname",
    "revenue": "Revenue",
    "business_area": "gsber",
    "billing_quantity": "fkimg",
    "volume": "volum",
    "account_group": "ktokd",
    "distribution_channel": "vtweg",
    "entry_time": "erzet_T",
    "credit_control_area": "kkber",
    "temp_billing_date": "FKDAT_TEMP",
    "group_key": "GK",
}

# ────────────────────────────────────────────────────────────────────────────────
# 2. BUILD PURE-$filter ODATA QUERY (no $apply, skips Azure limits)
# ────────────────────────────────────────────────────────────────────────────────


def build_odata_filter(params: Dict[str, Any]) -> str:
    """Return a $filter string (no $apply) honoring date_range & explicit filters."""
    filters: List[str] = []

    # Date range
    dr = params.get("date_range", {}) or {}
    today = datetime.utcnow().date()
    start = dr.get("start", "2024-01-01")
    end = dr.get("end", str(today))

    if len(start) == 10:
        start += "T00:00:00Z"
    if len(end) == 10:
        end += "T23:59:59Z"

    filters.append(f"fkdat ge {start} and fkdat le {end}")

    # Other filters
    for k, v in (params.get("filters") or {}).items():
        if v in (None, ""):
            continue
        col = COLUMN_MAP.get(k.lower(), k)
        clause = f"{col} eq {v}" if isinstance(
            v, (int, float)) else f"{col} eq '{v}'"
        filters.append(clause)

    return "$filter=" + " and ".join(filters)


def extract_filter_and_orderby(odata_query: str) -> tuple[Optional[str], Optional[str]]:
    """Utility to grab $filter and $orderby parts from a query string."""
    f, o = None, None
    for part in odata_query.split("&"):
        if part.startswith("$filter="):
            f = part[8:]
        elif part.startswith("$orderby="):
            o = part[9:]
    return f, o


# ────────────────────────────────────────────────────────────────────────────────
# 3. LLM FUNCTION-CALL SCHEMA
# ────────────────────────────────────────────────────────────────────────────────
openai_function_schema = {
    "name": "extract_sap_sales_query_params",
    "description": "Extracts analytic query structure from a SAP sales analysis question.",
    "parameters": {
        "type": "object",
        "properties": {
            "metric":   {"type": "string"},
            "group_by": {"type": "string"},
            "group_by_2": {"type": "string"},
            "filters":  {"type": "object"},
            "date_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end":   {"type": "string"},
                },
            },
            "aggregation": {"type": "string"},
            "top_n":   {"type": "integer"},
            "order_by": {"type": "string"},
            "order":    {"type": "string"},
            "page":     {"type": "integer"},
            "page_size": {"type": "integer"},
        },
    },
}

# ────────────────────────────────────────────────────────────────────────────────
# 4. OPENAI CALL – PARAMETER EXTRACTOR
# ────────────────────────────────────────────────────────────────────────────────


def extract_sap_sales_query_params(prompt: str) -> Dict[str, Any]:
    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_KEY,
        api_version="2025-01-01-preview",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a SAP sales analytics assistant. Extract metric, group_by, "
                "filters, date_range, aggregation, top_n, etc. Columns: "
                f"{list(COLUMN_MAP.keys())}. Output ONLY valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    rsp = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        tools=[{"type": "function", "function": openai_function_schema}],
        tool_choice={"type": "function", "function": {
            "name": "extract_sap_sales_query_params"}},
        max_tokens=16300,
    )

    payload = rsp.choices[0].message.tool_calls[0].function.arguments
    params = json.loads(payload)
    params.setdefault("page_size", 200)
    params.setdefault("page", 1)
    return params


# ────────────────────────────────────────────────────────────────────────────────
# 5.  AZURE SEARCH CLIENT
# ────────────────────────────────────────────────────────────────────────────────
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY),
)

# ────────────────────────────────────────────────────────────────────────────────
# 6. STREAM-AND-AGGREGATE (with date grouping, Top N, etc.)
# ────────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 1_000  # Azure hard-limit per page
AGG_SUPPORTED = {"sum", "average", "min", "max", "count"}


def _get_date_group(doc: dict, date_field: str, mode: str) -> str:
    raw = doc.get(date_field)
    if not raw:
        return "unknown"
    # Accept both ISO string and datetime
    if isinstance(raw, datetime):
        dt = raw
    else:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return "unknown"
    if mode == "month":
        return dt.strftime("%Y-%m")
    elif mode == "quarter":
        return f"{dt.year}-Q{((dt.month-1)//3)+1}"
    elif mode == "year":
        return str(dt.year)
    else:
        return str(dt.date())


def stream_and_aggregate(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Streams every document that matches the $filter and computes the chosen
    aggregation in constant memory. Adds group by month/quarter support, top N.
    Returns a JSON-serialisable dict: {group_key: {agg_name: value, count: n, …}}.
    """
    agg_type = params.get("aggregation", "sum").lower()
    if agg_type not in AGG_SUPPORTED:
        raise ValueError(
            f"Unsupported aggregation '{agg_type}'. Must be one of {AGG_SUPPORTED}")

    metric_col = COLUMN_MAP.get(params.get(
        "metric", "revenue").lower(), "Revenue")
    group_by_raw = params.get("group_by", "")
    group_mode = None

    # Handle group_by = month(fkdat), quarter(fkdat), year(fkdat)
    if group_by_raw and group_by_raw.lower().startswith("month("):
        group_mode = "month"
        group_field = group_by_raw[6:-1]  # e.g., fkdat
    elif group_by_raw and group_by_raw.lower().startswith("quarter("):
        group_mode = "quarter"
        group_field = group_by_raw[8:-1]
    elif group_by_raw and group_by_raw.lower().startswith("year("):
        group_mode = "year"
        group_field = group_by_raw[5:-1]
    else:
        group_mode = None
        group_field = COLUMN_MAP.get(
            group_by_raw.lower(), group_by_raw) if group_by_raw else None

    odata_filter = build_odata_filter(params)
    filter_expr, _ = extract_filter_and_orderby(odata_filter)

    bucket: Dict[Any, Dict[str, float | int]] = collections.defaultdict(
        lambda: {"sum": 0.0, "min": math.inf, "max": -math.inf, "count": 0}
    )

    select_fields = metric_col
    if group_field and group_field != metric_col:
        select_fields += "," + group_field
    if group_mode:  # always need the date field for group-by
        select_fields += "," + group_field

    results = search_client.search(
        search_text="*",
        filter=filter_expr,
        select=select_fields,
        top=BATCH_SIZE,
    )

    for page in results.by_page():
        for doc in page:
            if group_mode:  # group by month/quarter/year
                key = _get_date_group(doc, group_field, group_mode)
            elif group_field:
                key = doc.get(group_field, "unknown")
            else:
                key = "_all"
            val = doc.get(metric_col, 0) or 0
            b = bucket[key]
            b["sum"] += val
            b["count"] += 1
            b["min"] = min(b["min"], val)
            b["max"] = max(b["max"], val)

    # Finalise average if needed
    if agg_type == "average":
        for b in bucket.values():
            b["average"] = b["sum"] / b["count"] if b["count"] else 0

    # Shape result list for readability
    result_rows = []
    for k, v in bucket.items():
        row = {group_by_raw or "all": k}
        if agg_type == "average":
            row["average"] = v["average"]
        elif agg_type == "count":
            row["count"] = v["count"]
        else:
            row[agg_type] = v[agg_type]
        result_rows.append(row)

    # --- Top N ---
    top_n = params.get("top_n")
    order_field = agg_type if agg_type in (
        "sum", "average", "count", "min", "max") else "sum"
    if top_n:
        result_rows = sorted(
            result_rows,
            key=lambda r: r.get(order_field, 0),
            reverse=True
        )[:top_n]

    return {
        "filter": filter_expr,
        "metric": metric_col,
        "aggregation": agg_type,
        "group_by": group_by_raw,
        "result": result_rows,
    }

# ────────────────────────────────────────────────────────────────────────────────
# 7. RAW ROW FETCHER (pagination, no aggregation)
# ────────────────────────────────────────────────────────────────────────────────


def azure_sales_row_search(params: Dict[str, Any]) -> Dict[str, Any]:
    odata_filter = build_odata_filter(params)
    filter_expr, _ = extract_filter_and_orderby(odata_filter)

    page = int(params["page"])
    page_size = int(params["page_size"])
    skip = (page - 1) * page_size

    resp = search_client.search(
        search_text="*",
        filter=filter_expr,
        select="*",
        top=page_size,
        skip=skip,
    )
    rows = [dict(r) for r in resp]
    return {
        "filter": filter_expr,
        "pagination": {"page": page, "page_size": page_size},
        "rows_returned": len(rows),
        "result": rows,
    }

extractor_tool = Tool(
    name="SAPSalesQueryExtractor",
    func=extract_sap_sales_query_params,
    description="Extracts SAP-sales query meta-data from user text.",
)

aggregator_tool = Tool(
    name="SAPSalesAggregator",
    func=stream_and_aggregate,
    description=("Streams rows matching the filter and computes "
                 "sum/average/min/max/count on the fly. Supports grouping by month/quarter/year and Top N."),
)

row_fetch_tool = Tool(
    name="AzureSAPSalesRowFetcher",
    func=azure_sales_row_search,
    description="Fetches raw SAP-sales rows page-by-page (no aggregation).",
)

tools = [extractor_tool, aggregator_tool, row_fetch_tool]


# ────────────────────────────────────────────────────────────────────────────────
# 8. LANGCHAIN – tools & agent
# ────────────────────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview",
)

# memory = ConversationBufferMemory(memory_key="chat_history")

# extractor_tool = Tool(
#     name="SAPSalesQueryExtractor",
#     func=extract_sap_sales_query_params,
#     description="Extracts SAP-sales query meta-data from user text.",
# )

# aggregator_tool = Tool(
#     name="SAPSalesAggregator",
#     func=stream_and_aggregate,
#     description=("Streams rows matching the filter and computes "
#                  "sum/average/min/max/count on the fly. Supports grouping by month/quarter/year and Top N."),
# )

# row_fetch_tool = Tool(
#     name="AzureSAPSalesRowFetcher",
#     func=azure_sales_row_search,
#     description="Fetches raw SAP-sales rows page-by-page (no aggregation).",
# )
# 4. Per-conversation memory & agent factory (threadsafe)
# ------------------------------------------------------------------------------

# These are quick in-memory dicts for demo/dev.
# In production, use Django cache/Redis for real persistence and scale.


# --------------------------------------------------------------------------
# Django cache-backed per-conversation memory
# --------------------------------------------------------------------------


#latest :class DjangoCacheSaver(BaseCheckpointSaver):
# Django-backed persistent memory saver for LangGraph


class DjangoCacheSaver(BaseCheckpointSaver):
    def __init__(self, timeout=86400):
        self.timeout = timeout

    def _key_to_str(self, key_tuple):
        # Use thread_id as string if possible
        if isinstance(key_tuple, (tuple, list)):
            thread_id = key_tuple[0] if key_tuple and isinstance(key_tuple[0], str) else None
            if thread_id:
                return f"thread:{thread_id}"
        # fallback
        import hashlib
        return "thread:" + hashlib.md5(str(key_tuple).encode()).hexdigest()

    def get_tuple(self, key_tuple):
        return cache.get(self._key_to_str(key_tuple))

    def set_tuple(self, key_tuple, value):
        cache.set(self._key_to_str(key_tuple), value, timeout=self.timeout)

    def put(self, *args, **kwargs):
        # put(config, key, value, ..., **kwargs)
        if len(args) >= 3:
            key = args[1]
            value = args[2]
        elif len(args) == 2:
            key, value = args
        else:
            raise Exception("Not enough arguments to put()")
        self.set_tuple(key, value)

    def put_writes(self, *args, **kwargs):
        """
        Support both (writes) and (config, writes, namespace, ...)
        - writes is always a list of (config, key, value, ...) tuples
        """
        if len(args) == 1:
            writes = args[0]
        elif len(args) >= 2:
            writes = args[1]
        else:
            writes = []

        for w in writes:
            # Usually: (config, key, value, ...)
            if len(w) >= 3:
                _, key, value = w[:3]
                self.set_tuple(key, value)


    def list(self):
        return []

    def get(self, key):
        return self.get_tuple(key)

checkpointer = DjangoCacheSaver()

print ("checkpointer ",checkpointer)

# LangGraph Agent
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer
)

def get_summary_buffer_memory(conversation_id: str, timeout=60*60*24):
    """Loads or creates ConversationSummaryBufferMemory for this conversation using Django cache."""
    print("calling  get_summary_buffer_memory")
    key_summary = f"chat:summary:{conversation_id}"
    key_buffer = f"chat:buffer:{conversation_id}"
    summary = cache.get(key_summary)
    buffer = cache.get(key_buffer)
    print("mesummary ",summary)
    memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1500,
    memory_key="chat_history",
    return_messages=True,
    )


    # Restore persisted state if exists
    if summary:
        memory.moving_summary_buffer = summary
    if buffer:
        memory.chat_memory.messages = buffer
    return memory
def get_conversation_memory(conversation_id: str, timeout=60*60*24) -> ConversationBufferMemory:
    key = f"chat:memory:{conversation_id}"
    buffer = cache.get(key)
    if buffer is None:
        buffer = []
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    memory.chat_memory.messages = buffer
    return memory

def save_summary_buffer_memory(conversation_id: str, memory: ConversationSummaryBufferMemory, timeout=60*60*24):
    print("Saving summary buffer for conversation:", conversation_id)
    print("moving_summary_buffer:", memory.moving_summary_buffer)
    print("chat_memory.messages:", memory.chat_memory.messages)
    cache.set(f"chat:summary:{conversation_id}", memory.moving_summary_buffer, timeout=timeout)
    cache.set(f"chat:buffer:{conversation_id}", memory.chat_memory.messages, timeout=timeout)

# agent = initialize_agent(
#     tools=[extractor_tool, aggregator_tool, row_fetch_tool],
#     llm=llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     memory=memory,
#     verbose=False,
# )

# ────────────────────────────────────────────────────────────────────────────────
# 9. PUBLIC API – what your Django view will call
# ────────────────────────────────────────────────────────────────────────────────
def build_summary_prompt_from_history(memory, user_prompt, data, params, is_agg):
    """
    Build a summary prompt for the LLM, including relevant chat history for context-aware answers.
    """
    history_lines = []
    # Use last N Q/A pairs
    if hasattr(memory, "buffer_as_messages"):
        msgs = memory.buffer_as_messages[-8:]  # last 4 Q/A pairs (8 messages)
        last_q = None
        for msg in msgs:
            if msg.type == "human":
                last_q = msg.content
            elif msg.type == "ai" and last_q:
                history_lines.append(f"Q: {last_q}\nA: {msg.content}")
                last_q = None
    # Now add the current question/data
    if is_agg:
        history_lines.append(
            f"Q: {user_prompt}\nA: Aggregated result: {json.dumps(data['result'], indent=2)}"
        )
    else:
        history_lines.append(
            f"Q: {user_prompt}\nA: Result snippet: {json.dumps(data['result'][:50], indent=2)}"
        )
    chat_history_text = "\n\n".join(history_lines)
    return (
        f"Below is the recent conversation history between a business user and an SAP analytics assistant.\n"
        f"{chat_history_text}\n\n"
        "Based on the conversation above, provide a clear, business-friendly answer to the last user query. "
        "If comparisons are requested, analyze trends, differences, or patterns. "
        "If result is paginated, mention page & size. "
        "Never refer to rows, tokens, or internal processing."
    )


def handle_user_query(prompt: str, conversation_id: str = "default") -> Dict[str, Any]:
    # --- Use LangGraph agent to pick/run the correct tool, track memory, etc. ---
    thread_config = {"configurable": {"thread_id": conversation_id}}
    messages = [{"role": "user", "content": prompt}]

    # The agent will call your tools, and their outputs will be in intermediate_steps
    result = agent_executor.invoke({"messages": messages}, thread_config)
    print("result agent_executor  ",result)

    # Find the tool output
    tool_data = None
    params = None
    is_agg = None
    # Try to parse from the agent's intermediate steps (depends on tool setup)
    for step in (result.get("intermediate_steps") or []):
        tool_call = step.get("tool_calls", [{}])[0]
        if tool_call.get("name") == "SAPSalesAggregator":
            tool_data = tool_call.get("output")
            params = tool_call.get("input")
            is_agg = True
        elif tool_call.get("name") == "AzureSAPSalesRowFetcher":
            tool_data = tool_call.get("output")
            params = tool_call.get("input")
            is_agg = False
        elif tool_call.get("name") == "SAPSalesQueryExtractor":
            params = tool_call.get("output")

    # Fallback for param extraction if not found
    if params is None:
        params = extract_sap_sales_query_params(prompt)
    if is_agg is None:
        is_agg = bool(params.get("aggregation"))

    # --- Compose your custom summary prompt using the actual data ---
    if tool_data and isinstance(tool_data, dict):
        data = tool_data
    else:
        # fallback to direct call if agent/tools misfire
        data = stream_and_aggregate(params) if is_agg else azure_sales_row_search(params)

    if is_agg:
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"Processed parameters: {json.dumps(params, indent=2)}\n"
            f"Aggregated result: {json.dumps(data['result'], indent=2)}\n"
            "Give a clear, business-friendly answer. "
            "Highlight which group is uptrending if possible, and avoid any pagination language. "
            "Do not mention rows, pages, or limits—summarize based on the full dataset."
        )
    else:
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"Processed parameters: {json.dumps(params, indent=2)}\n"
            f"Result snippet : {json.dumps(data['result'][:50], indent=2)}\n"
            "Give a clear, business-friendly answer. "
            "Do not mention rows, pages, or limits—summarize based on the full dataset"
        )

    answer = llm.invoke(summary_prompt).content

    return {
        "answer": answer,
        "data": data,
        "operation_plan": params,
        "steps": result.get("intermediate_steps"),
        "chat_history": result.get("messages"),
    }
