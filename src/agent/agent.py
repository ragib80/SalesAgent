
from __future__ import annotations

import json
import math
import collections
from datetime import datetime
from typing import Dict, Any, List, Optional

from django.conf import settings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

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
# 8. LANGCHAIN – tools & agent
# ────────────────────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview",
)

memory = ConversationBufferMemory(memory_key="chat_history")

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

agent = initialize_agent(
    tools=[extractor_tool, aggregator_tool, row_fetch_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=False,
)

# ────────────────────────────────────────────────────────────────────────────────
# 9. PUBLIC API – what your Django view will call
# ────────────────────────────────────────────────────────────────────────────────


def handle_user_query(prompt: str) -> Dict[str, Any]:
    params = extract_sap_sales_query_params(prompt)

    # Explicitly choose tool
    is_agg = bool(params.get("aggregation"))
    if is_agg:
        data = stream_and_aggregate(params)
    else:
        data = azure_sales_row_search(params)

    # Summary prompt: different for agg vs. paginated
    if is_agg:
        print("dfsd")
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"Processed parameters: {json.dumps(params, indent=2)}\n"
            f"Aggregated result: {json.dumps(data['result'], indent=2)}\n"
            "Give a clear, business-friendly answer . "
            "Highlight which group is uptrending if possible, and avoid any pagination language. "
            "Do not mention rows, pages, or limits—summarize based on the full dataset."
        )
    else:
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"Processed parameters: {json.dumps(params, indent=2)}\n"
            f"Result snippet (first 20 rows/items): {json.dumps(data['result'][:20], indent=2)}\n"
            "Give a clear, business-friendly answer . "
            "If result is paginated, mention page & size."
        )

    answer = llm.invoke(summary_prompt).content

    return {
        "answer": answer,
        "data": data,
        "operation_plan": params,
    }
