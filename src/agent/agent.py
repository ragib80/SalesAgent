from datetime import datetime
import json
from django.conf import settings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from openai import AzureOpenAI

# ---- 1. COLUMN MAP (ALL COLUMNS) ----
COLUMN_MAP = {
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
    "group_key": "GK"
}

# ---- 2. ODATA QUERY BUILDER (PAGINATION SUPPORTED) ----


def build_odata_query(params):
    """
    Build an OData query string for Azure Cognitive Search based on extracted parameters.
    Defaults to full date range if not specified.
    """
    filters = []
    # ----- DATE RANGE HANDLING -----
    dr = params.get("date_range", {})
    # Use defaults if not found
    today = datetime.utcnow().date()
    start = dr.get("start", "2024-01-01")
    end = dr.get("end", str(today))
    # Ensure proper ISO8601 with time and 'Z'
    if len(start) == 10:
        start += "T00:00:00Z"
    if len(end) == 10:
        end += "T23:59:59Z"
    filters.append(f"fkdat ge {start} and fkdat le {end}")

    # ----- OTHER FILTERS -----
    for k, v in (params.get("filters") or {}).items():
        if v is not None and v != "":
            # If field is in COLUMN_MAP, use it; else use as-is
            col = COLUMN_MAP.get(k.lower(), k)
            # For numbers/dates, don't wrap in quotes
            if isinstance(v, (int, float)):
                filters.append(f"{col} eq {v}")
            else:
                filters.append(f"{col} eq '{v}'")

    # ----- FILTER STRING -----
    filter_str = "$filter=" + " and ".join(filters) if filters else ""

    # ----- AGGREGATION & GROUP BY -----
    metric_col = COLUMN_MAP.get(params.get(
        "metric", "revenue").lower(), params.get("metric", "Revenue"))
    group_by = params.get("group_by")
    group_by_col = COLUMN_MAP.get(
        group_by.lower(), group_by) if group_by else None
    agg_type = params.get("aggregation", "")
    apply_str = ""

    if group_by_col:
        apply_str = f"$apply=groupby(({group_by_col}"
        # Optional: handle group_by_2
        if params.get("group_by_2"):
            gb2 = COLUMN_MAP.get(
                params["group_by_2"].lower(), params["group_by_2"])
            apply_str += f", {gb2}"
        apply_str += f"), aggregate({metric_col} with sum as Total{metric_col.capitalize()}))"
    else:
        apply_str = f"$apply=aggregate({metric_col} with sum as Total{metric_col.capitalize()})"

    # Handle trend (e.g. group by month)
    if agg_type == "trend" and group_by_col:
        apply_str = (f"$apply=groupby(({group_by_col}, month(fkdat)), "
                     f"aggregate({metric_col} with sum as Monthly{metric_col.capitalize()}))")

    # ----- ORDER BY, TOP N -----
    odata_parts = [filter_str, apply_str]
    if agg_type and "top" in agg_type and "top_n" in params:
        odata_parts.append(
            f"$orderby=Total{metric_col.capitalize()} desc&$top={params['top_n']}")
    elif params.get("order_by"):
        order_col = COLUMN_MAP.get(
            params["order_by"].lower(), params["order_by"])
        order = params.get("order", "desc")
        odata_parts.append(f"$orderby={order_col} {order}")

    odata_query = "&".join(part for part in odata_parts if part)
    print("\n==== Built OData query ====")
    print(odata_query)
    return odata_query


# ---- 3. FUNCTION SCHEMA FOR LLM (OpenAI) ----
openai_function_schema = {
    "name": "extract_sap_sales_query_params",
    "description": "Extracts analytic query structure from a SAP sales analysis question.",
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "Metric, e.g. 'revenue', 'volume', etc."},
            "group_by": {"type": "string", "description": "Group by column."},
            "group_by_2": {"type": "string", "description": "Second group by column (optional)."},
            "filters": {"type": "object", "description": "Dict of filters, e.g. {'division_name':'Printing Ink'}"},
            "date_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"}
                }
            },
            "aggregation": {"type": "string", "description": "e.g. 'trend', 'top', 'compare', 'average', etc."},
            "top_n": {"type": "integer"},
            "order_by": {"type": "string"},
            "order": {"type": "string"},
            "page": {"type": "integer"},
            "page_size": {"type": "integer"}
        }
    }
}

# ---- 4. LLM EXTRACTION WITH OPENAI FUNCTION CALLING ----


def extract_sap_sales_query_params(prompt: str) -> dict:
    print("\n=== extract_sap_sales_query_params CALLED ===")
    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_KEY,
        api_version="2025-01-01-preview"
    )

    print("\n=== client ===", client)
    messages = [
        {"role": "system", "content":
            "You are a SAP sales analytics assistant. Given a natural language prompt, extract as much as possible for: "
            "metric, group_by, group_by_2, filters, date_range, aggregation, top_n, order_by, order, page, page_size. "
            f"Columns are: {list(COLUMN_MAP.keys())}. "
            "Output only valid JSON per schema."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=[{"type": "function", "function": openai_function_schema}],
            tool_choice={"type": "function", "function": {
                "name": "extract_sap_sales_query_params"}},
            max_tokens=16300
        )
        print("Azure OpenAI raw response:", response)
    except Exception as ex:
        print("AzureOpenAI LLM extraction error:", ex)
        raise
    print("Azure OpenAI raw response:", response)
    tool_call = response.choices[0].message.tool_calls[0]
    params = json.loads(tool_call.function.arguments)
    print("=============extract_sap_sales_query_params=========")
    print(response)
    # Set sensible defaults for pagination
    if not params.get("page_size"):
        params["page_size"] = 50
    if not params.get("page"):
        params["page"] = 1
    return params


# ---- 5. AZURE SEARCH TOOL WITH PAGINATION AND FULL ACCURACY ----
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)


def azure_sales_odata_search(params):
    odata = build_odata_query(params)
    page = int(params.get("page", 1))
    page_size = int(params.get("page_size", 50))
    skip = (page - 1) * page_size

    # NEW: Parse $filter and $orderby from odata query
    filter_expr, order_by_expr = extract_filter_and_orderby_from_odata(odata)
    print(
        f"\n==== Running Azure Search with filter: {filter_expr} | order_by: {order_by_expr} ====")
    results = []
    try:
        resp = search_client.search(
            search_text="*",
            filter=filter_expr,
            select="*",
            top=page_size,
            skip=skip,
            order_by=order_by_expr
        )
        for item in resp:
            results.append(dict(item))
        print(f"Azure Search returned {len(results)} rows")
    except Exception as ex:
        print("Azure Search error:", ex)
        results.append({"error": str(ex)})
    return {
        "odata_query": odata,
        "result": results,
        "pagination": {"page": page, "page_size": page_size}
    }


# ---- 6. LANGCHAIN TOOLS AND AGENT ----
llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview"
)

memory = ConversationBufferMemory(memory_key="chat_history")

extractor_tool = Tool(
    name="SAPSalesQueryExtractor",
    func=extract_sap_sales_query_params,
    description="Extracts SAP sales analytics query structure from user's prompt."
)

azure_search_tool = Tool(
    name="AzureSAPSalesSearch",
    func=azure_sales_odata_search,
    description="Run SAP sales analysis queries using OData."
)

agent = initialize_agent(
    tools=[extractor_tool, azure_search_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=False
)


def extract_filter_and_orderby_from_odata(odata_query):
    """
    Extracts the $filter and $orderby clause from an OData query string.
    Returns (filter_str, order_by).
    """
    filter_str = None
    order_by = None
    for part in odata_query.split('&'):
        if part.startswith("$filter="):
            filter_str = part[len("$filter="):]
        elif part.startswith("$orderby="):
            order_by = part[len("$orderby="):]
    return filter_str, order_by

# ---- 7. MAIN HANDLER FOR DJANGO API ----


def handle_user_query(prompt: str):
    """
    Main entry point to process user queries via the agent.
    """
    print("\n=== handle_user_query CALLED ===")
    try:

        # Step 1: Extract analytic intent
        params = extract_sap_sales_query_params(prompt)
        print("\n=== handle_user_query CALLED ===")
        print("Prompt received:", prompt)
        # Step 2: Query Azure Search via OData
        data = azure_sales_odata_search(params)
        print("\n=== azure_sales_odata_search  CALLED ===")
        print("Prompt received:", prompt)
        # Step 3: Summarize with LLM (send answer + result to LLM for summary)
        summary_prompt = (
            f"User asked: '{prompt}'.\n"
            f"OData query: {data['odata_query']}.\n"
            f"Result data: {json.dumps(data['result'][:10], indent=2)}.\n"
            "Generate a clear business summary for the user. If the user asked for paginated results, explain which page and how many records are displayed."
        )
        llm_response = llm.invoke(summary_prompt)
        # answer = llm(summary_prompt).content
        answer = getattr(llm_response, "content", llm_response)
        return {
            "answer": answer,
            "result": data['result'],
            "pagination": data["pagination"],
            "operation_plan": {
                "odata_query": data['odata_query'],
                "params": params
            }
        }
    except Exception as ex:
        return {
            "answer": (
                "Sorry, I couldn't process your request. "
                "Please try rephrasing your question with more specifics."
            ),
            "result": None,
            "pagination": None,
            "operation_plan": None
        }
