
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
from urllib.parse import parse_qs, unquote_plus
import re

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
    "product": "arktx",
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


BUSINESS_AREA_MAP: Dict[str, str] = {
    "depo":    "1000",  # if you want a generic 'depo' name
    "dhaka factory":   "1000",
    "chittagong factory": "1100",
    "mirsarai factory":   "1200",
    "dhaka sales":       "4000",
    "chittagong sales":  "4010",
    "sylhet sales":      "4020",
    "comilla sales":     "4030",
    "rajshahi sales":    "4040",
    "bogra sales":       "4050",
    "khulna sales":      "4060",
    "mymensing sales":   "4070",
    "barishal sales":    "4080",
    "rangpur sales":     "4090",
    "feni sales":        "4100",
    "dhaka south":       "4110",
    "brahmanbaria sales":"4120",
    "dhaka north":       "4130",
    "test business area":"4500",
    "pphd":              "5000",
    "berger design studio":"5010",
    "berger training institute":"5020",
    "berger tech consulting lt":"5100",
    "jenson & nicholson bd ltd":"6000",
    "jnbl 2nd unit dhaka":"6100",
    "berger becker bangladesh":"7000",
    "berger fosroc limited":"8000",
    "corporate":"9000",
}
# ────────────────────────────────────────────────────────────────────────────────
# 2. BUILD PURE-$filter ODATA QUERY (no $apply, skips Azure limits)
# ────────────────────────────────────────────────────────────────────────────────


# def build_odata_filter(params: Dict[str, Any]) -> str:
#     """Return a $filter string (no $apply) honoring date_range & explicit filters."""
#     filters: List[str] = []

#     # Date range
#     dr = params.get("date_range", {}) or {}
#     today = datetime.utcnow().date()
#     start = dr.get("start", "2024-01-01")
#     end = dr.get("end", str(today))

#     if len(start) == 10:
#         start += "T00:00:00Z"
#     if len(end) == 10:
#         end += "T23:59:59Z"

#     filters.append(f"fkdat ge {start} and fkdat le {end}")

#     # Other filters
#     for k, v in (params.get("filters") or {}).items():
#         if v in (None, ""):
#             continue
#         col = COLUMN_MAP.get(k.lower(), k)
#         clause = f"{col} eq {v}" if isinstance(
#             v, (int, float)) else f"{col} eq '{v}'"
#         filters.append(clause)

#     return "$filter=" + " and ".join(filters)
# ────────────────────────────────────────────────────────────────────────────────
# 2. BUILD PURE-$filter ODATA QUERY
# ────────────────────────────────────────────────────────────────────────────────
def build_odata_filter(params: Dict[str, Any]) -> str:
    """
    Return a $filter string (no $apply), enforcing that params["filters"]
    is always a dict—even if originally supplied as a string—and honoring
    date_range & explicit filters.
    """
    # — Normalize filters: turn a string like "A eq 'X' and B eq 'Y'" into a dict
    f = params.get("filters")
    if isinstance(f, str):
        filters_dict: Dict[str, Any] = {}
        # split on " and " to get individual clauses
        for clause in f.split(" and "):
            if " eq " in clause:
                col, val = clause.split(" eq ", 1)
                col = col.strip()
                val = val.strip()
                # strip surrounding single quotes, if present
                if val.startswith("'") and val.endswith("'"):
                    val = val[1:-1]
                filters_dict[col] = val
        params["filters"] = filters_dict

    filters: List[str] = []

    # — Date range —
    dr = params.get("date_range", {}) or {}
    today = datetime.utcnow().date()
    start = dr.get("start", "2024-01-01")
    end = dr.get("end", str(today))
    if len(start) == 10:
        start += "T00:00:00Z"
    if len(end) == 10:
        end += "T23:59:59Z"
    filters.append(f"fkdat ge {start} and fkdat le {end}")

    # — Other filters —
    for k, v in (params.get("filters") or {}).items():
        if v in (None, ""):
            continue
        col = COLUMN_MAP.get(k.lower(), k)
        if isinstance(v, (int, float)):
            clause = f"{col} eq {v}"
        else:
            # assume string
            clause = f"{col} eq '{v}'"
        filters.append(clause)

    # combine
    return "$filter=" + " and ".join(filters)


# ────────────────────────────────────────────────────────────────────────────────
# 3. AZURE METRIC AGGREGATION (FACET) HELPER
# ────────────────────────────────────────────────────────────────────────────────
def azure_metric_aggregate(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses Azure AI Search facets to compute SUM on Revenue,
    nested by one or more group_by fields.
    """
    # 1) Resolve metric & group_by list
    metric_col = COLUMN_MAP.get(params.get("metric", "revenue").lower(), "Revenue")
    gb = params.get("group_by")
    # accept string or list
    if isinstance(gb, str) and gb:
        group_fields = [gb]
    elif isinstance(gb, list):
        group_fields = gb
    else:
        group_fields = []

    # map to actual index field names
    group_fields = [
        COLUMN_MAP.get(f.lower(), f)
        for f in group_fields
        if f
    ]

    # 2) Build a single nested facet expression:
    #    e.g. ["Field1 > Field2 > (Revenue, metric: sum)"]
    if group_fields:
        nested = " > ".join(group_fields) + f" > ({metric_col}, metric: sum)"
        facets = [nested]
    else:
        # no grouping → just a sum of the metric
        facets = [f"({metric_col}, metric: sum)"]

    # 3) Build $filter clause
    odata_filter = build_odata_filter(params)[len("$filter="):]  # strip prefix

    # 4) Call Azure Search
    results = search_client.search(
        search_text="*",
        filter=odata_filter,
        facets=facets,
        top=0,            # we only want facets
        include_total_count=False,
    )

    # 5) Parse nested facets into a flat list of dicts
    # Azure returns a nested dict under results.facets, matching the first group name
    raw_facets = results.facets or {}
    # drill into the first (and only) facet key
    top_key = next(iter(raw_facets), None)
    buckets = raw_facets.get(top_key, [])

    def recurse(buckets, depth=0, prefix=None):
        out = []
        for b in buckets:
            val = b["value"]
            sum_metric = b.get("@search.facets", {}).get(metric_col, [{}])[0].get("sum", 0)
            key = (prefix or []) + [val]
            # if there are deeper sub-facets:
            sub = b.get("value", None)
            nested_key = group_fields[depth + 1] if depth + 1 < len(group_fields) else None
            if nested_key and b.get("facets", {}).get(nested_key):
                # recurse into that list
                sub_buckets = b["facets"][nested_key]
                out.extend(recurse(sub_buckets, depth + 1, key))
            else:
                # leaf node
                row = {group_fields[i]: key[i] for i in range(len(key))}
                row["sum"] = sum_metric
                out.append(row)
        return out

    result_rows = recurse(buckets)

    return {
        "filter": odata_filter,
        "metric": metric_col,
        "aggregation": "sum",
        "group_by": group_fields,
        "result": result_rows,
    }



def _normalize_params(arg):
    # If it’s already a dict, return it
    if isinstance(arg, dict):
        return arg

    # If it’s JSON text, load it
    if isinstance(arg, str):
        try:
            return json.loads(arg)
        except json.JSONDecodeError:
            # Maybe it’s a URL-encoded query string
            qs = parse_qs(arg)
            out = {}
            for k, v in qs.items():
                val = v[0]
                if val.startswith("{") and val.endswith("}"):
                    try:
                        out[k] = json.loads(val)
                    except json.JSONDecodeError:
                        out[k] = val
                else:
                    out[k] = unquote_plus(val)
            return out

    raise ValueError(f"Cannot normalize params from type={type(arg)}")


def sap_sales_aggregator(arg1: Any) -> Dict[str, Any]:
    # 1) Normalize incoming params
    params = _normalize_params(arg1)

    # 2) Normalize aggregation verb into one of the supported set
    agg = str(params.get("aggregation", "")).lower()
    if agg not in AGG_SUPPORTED:
        # If they passed the metric name, treat as sum; else default to sum
        agg = "sum"
    params["aggregation"] = agg

    # 3) Ensure paging defaults (for any fallbacks)
    params.setdefault("page", 1)
    params.setdefault("page_size", 50)

    # 4) Dispatch:
    #    - If it's a simple SUM on one level of grouping → use Azure facets
    #    - Otherwise → stream & aggregate in Python
    use_azure = (agg == "sum" and isinstance(params.get("group_by"), (str, list)))
    if use_azure:
        result = azure_metric_aggregate(params)
    else:
        result = stream_and_aggregate(params)

    # 5) Apply Top-N slicing & ordering client-side on the aggregation field
    top_n = params.get("top_n")
    if top_n and isinstance(result.get("result"), list):
        # aggregation key in each row ('sum','average','min','max','count')
        key = agg
        desc = params.get("order", "desc").lower() == "desc"
        result["result"] = sorted(
            result["result"],
            key=lambda r: r.get(key, 0),
            reverse=desc
        )[:top_n]

    return result


def sap_sales_rowfetcher(arg1: Any) -> Dict[str, Any]:
    # 1) Normalize incoming parameters (JSON string, query-string, or dict)
    params = _normalize_params(arg1)

    # 2) Ensure pagination defaults so we never KeyError
    params.setdefault("page", 1)
    params.setdefault("page_size", 50)

    # 3) Delegate to the raw row fetcher
    return azure_sales_row_search(params)



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
    params.setdefault("page_size", 50)
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

# ────────────────────────────────────────────────────────────────────────────────
# 6. TOOLS
# ────────────────────────────────────────────────────────────────────────────────
extractor_tool = Tool(
    name="SAPSalesQueryExtractor",
    func=extract_sap_sales_query_params,
    description="Extracts SAP-sales query meta-data from user text.",
)

# aggregator_tool = Tool(
#     name="SAPSalesAggregator",
#     func=lambda params: azure_metric_aggregate(params) if params.get("aggregation") == "sum" else stream_and_aggregate(params),
#     description="Aggregates SAP sales: uses Azure facet for sum, streams otherwise.",
# )

aggregator_tool = Tool(
    name="SAPSalesAggregator",           # exactly this name
    func=sap_sales_aggregator,           # your wrapper
    description="Aggregates SAP sales: uses Azure facet for sum, streams otherwise.",
)

# row_fetch_tool = Tool(
#     name="AzureSAPSalesRowFetcher",
#     func=azure_sales_row_search,
#     description="Fetches raw SAP-sales rows page-by-page (no aggregation).",
# )

row_fetch_tool = Tool(
    name="AzureSAPSalesRowFetcher",      # exactly this name
    func=sap_sales_rowfetcher,           # your wrapper
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
#langgraph agent
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
    # 1) Invoke the agent as before to get params/tool outputs
    thread_config = {"configurable": {"thread_id": conversation_id}}
    messages = [{"role": "user", "content": prompt}]
    result = agent_executor.invoke({"messages": messages}, thread_config)

    # 2) Extract tool outputs & params
    tool_data = None
    params = None
    is_agg = None
    for step in result.get("intermediate_steps", []):
        call = step.get("tool_calls", [{}])[0]
        name = call.get("name")
        if name == "SAPSalesAggregator":
            tool_data, params, is_agg = call.get("output"), call.get("input"), True
        elif name == "AzureSAPSalesRowFetcher":
            tool_data, params, is_agg = call.get("output"), call.get("input"), False
        elif name == "SAPSalesQueryExtractor":
            params = call.get("output")

    # fallback
    if params is None:
        params = extract_sap_sales_query_params(prompt)
    if is_agg is None:
        is_agg = bool(params.get("aggregation"))

    # 3) If this is an aggregation AND it looks like a compare/ vs  question:
    if is_agg and re.search(r"\bcompare\b", prompt, re.I) and re.search(r"\bvs?\.?\b", prompt, re.I):
        # a) figure out the field to split on
        #    (we assume the extractor put both values into params["filters"])
        raw_filters = params.get("filters")
        # normalize to dict
        if isinstance(raw_filters, str):
            # e.g. "division_name eq 'Decorative' and division_name eq 'Marine Paints'"
            filters_dict = {}
            for clause in raw_filters.split(" and "):
                if " eq " in clause:
                    k, v = clause.split(" eq ", 1)
                    filters_dict.setdefault(k.strip(), []).append(v.strip().strip("'"))
            params["filters"] = filters_dict
        # b) pull out the two compare-keys
        field, values = next(iter(params["filters"].items()))
        if len(values) >= 2:
            left_val, right_val = values[0], values[1]
        else:
            # fallback: try parsing from prompt directly
            parts = re.split(r"\bvs?\.?\b", prompt, flags=re.I)
            left_val = parts[0].split()[-1]
            right_val = parts[1].split()[-1]

        # c) build two param sets
        def single_filter_params(value):
            p = dict(params)
            p["filters"] = {field: value}
            return p

        p1, p2 = single_filter_params(left_val), single_filter_params(right_val)
        r1, r2 = sap_sales_aggregator(p1), sap_sales_aggregator(p2)

        # d) now hand off to a mini-prompt that gives both totals
        left_sum  = r1["result"][0].get("sum", 0) if r1["result"] else 0
        right_sum = r2["result"][0].get("sum", 0) if r2["result"] else 0

        summary_prompt = f"""
User asked: {prompt}

- {left_val}: {left_sum:,}
- {right_val}: {right_sum:,}

Please write a concise business-friendly comparison highlighting which is stronger in total revenue, any key drivers, and overall takeaway.
"""
        answer = llm.invoke(summary_prompt).content
        return {
            "answer": answer,
            "data": {"left": r1, "right": r2},
            "operation_plan": params,
            "steps": result.get("intermediate_steps"),
            "chat_history": result.get("messages"),
        }

    # 4) Otherwise fall back to the existing single-aggregation flow
    if tool_data and isinstance(tool_data, dict):
        data = tool_data
    else:
        data = sap_sales_aggregator(params) if is_agg else sap_sales_rowfetcher(params)

    # 5) Build your summary prompt as before
    if is_agg:
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"Processed parameters: {json.dumps(params, indent=2)}\n"
            f"Aggregated result: {json.dumps(data['result'], indent=2)}\n"
            "Give a clear, business-friendly answer. "
            "Highlight trends, and avoid pagination language."
        )
    else:
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"Processed parameters: {json.dumps(params, indent=2)}\n"
            f"Result snippet: {json.dumps(data['result'][:10], indent=2)}\n"
            "Give a clear, business-friendly answer."
        )

    answer = llm.invoke(summary_prompt).content

    return {
        "answer": answer,
        "data": data,
        "operation_plan": params,
        "steps": result.get("intermediate_steps"),
        "chat_history": result.get("messages"),
    }