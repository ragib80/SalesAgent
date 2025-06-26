import os
import json
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from django.conf import settings

search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)

llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

FIELD_SYNONYMS = {
    "revenue":              "Revenue",
    "quantity":             "fkimg",
    "volume":               "volum",
    "customer":             "cname",
    "brand":                "wgbez",
    "product name":         "arktx",
    "product":              "arktx",
    "category":             "matkl",
    "division":             "spart_text",
    "company code":         "bukrs",
    "sales org":            "vkorg",
    "dist channel":         "vtweg",
    "distribution channel": "vtweg",
    "business area":        "gsber",
    "credit control area":  "kkber",
    "customer group":       "kukla",
    "account group":        "ktokd",
    "sales group":          "vkgrp_c",
    "sales office":         "vkbur_c",
    "payer id":             "Payer_DL",
    "product code":         "matnr",
    "unit":                 "meins",
    "volume unit":          "voleh",
    "business group":       "GK",
    "territory":            "Territory",
    "sales zone":           "Szone",
    "date":                 "fkdat",
    "fkdat":                "fkdat",
    "profit":               "Profit"
}

# -----
# Only allow facetable fields here for group_by/facet
FACETABLE_FIELDS = {"cname", "wgbez", "arktx", "Territory", "vkorg", "bukrs", "kukla", "ktokd", "Szone"}  # Add others as needed

ANALYSIS_LOGIC = {
    "deterioration_rate": {
        "columns": ["Revenue", "fkdat"],
        "operation": "deterioration_rate",
        "group_by": None,
    },
    "profit_loss": {
        "columns": ["Revenue", "Cost"],
        "operation": "profit_loss",
        "group_by": None,
    },
    "trend": {
        "columns": ["Revenue", "fkdat"],
        "operation": "trend_analysis",
        "group_by": ["product"],  # This gets mapped to arktx, see below
    },
    "comparison": {
        "columns": ["vkorg", "Revenue", "fkdat"],
        "operation": "comparison",
        "group_by": ["vkorg"],
    },
    "profitability": {
        "columns": ["Revenue", "Cost"],
        "operation": "profitability",
        "group_by": ["product"],
    },
    "low_revenue": {
        "columns": ["Product", "Revenue"],
        "operation": "bottom_n",
        "group_by": ["product"],
    },
}

def parse_prompt_to_plan(prompt):
    instructions = (
        "You are an expert SAP sales analytics assistant. "
        "Given a user request, return ONLY valid JSON (no explanations, no markdown, no extra text) in this format:\n"
        "{\n"
        '  "metric": "default",\n'
        '  "field": "revenue",\n'
        '  "group_by": [],\n'
        '  "compare_values": [],\n'
        '  "start_date": "YYYY-MM-DD",\n'
        '  "end_date": "YYYY-MM-DD"\n'
        "}\n"
        "- metric: one of [profit_loss, deterioration_rate, trend, comparison, profitability, low_revenue, default].\n"
        "- field: (e.g. revenue, profit, cost, dealer, brand, product, tr, etc.).\n"
        "- group_by: list of fields to group by (if needed). Only use one: product→arktx, dealer→cname, brand→wgbez, territory→Territory.\n"
        "- compare_values: list of values for comparison (if any).\n"
        "- start_date: YYYY-MM-DD.\n"
        "- end_date: YYYY-MM-DD.\n"
        "If the user simply asks about 'Revenue' or any single value, use 'default' as metric."
    )
    llm_prompt = f"{instructions}\nUser: {prompt}\nJSON:"
    print("==== LLM PROMPT ====")
    print(llm_prompt)
    try:
        llm_response = llm.invoke(llm_prompt)
        llm_content = getattr(llm_response, "content", llm_response)
        print("==== LLM RAW RESPONSE ====")
        print(llm_content)
        try:
            plan = json.loads(llm_content.split('```json')[-1].split('```')[0] if '```json' in llm_content else llm_content)
        except Exception:
            plan = json.loads(llm_content)
        print("Raw LLM plan:", plan)

        # === FIX: Map field names from LLM to actual schema ===
        field = plan.get("field", "revenue").lower()
        field_mapped = FIELD_SYNONYMS.get(field, field)
        # Special handling for "dealer" → "cname"
        if field in ["dealer", "dealers"]:
            field_mapped = "cname"
        plan["field"] = field_mapped

        # Group by mapping (also fix dealer)
        group_by = []
        for g in plan.get("group_by", []):
            g_syn = FIELD_SYNONYMS.get(g.lower(), g)
            if g.lower() == "dealer":
                g_syn = "cname"
            group_by.append(g_syn)
        plan["group_by"] = group_by

        # === FIX: Fallback to valid dates if LLM outputs "YYYY-MM-DD" ===
        start = plan.get("start_date", "")
        end = plan.get("end_date", "")
        if not start or "YYYY" in start:
            start = f"{datetime.now().year}-01-01"
        if not end or "YYYY" in end:
            end = f"{datetime.now().year}-12-31"
        plan["start_date"] = start
        plan["end_date"] = end

        print("Mapped and filled plan:", plan)
        return plan
    except Exception as ex:
        print("parse_prompt_to_plan ERROR:", ex)
        # Safe default
        return {
            "metric": "default",
            "field": "Revenue",
            "group_by": [],
            "compare_values": [],
            "start_date": f"{datetime.now().year}-01-01",
            "end_date": f"{datetime.now().year}-12-31",
        }

def logic_from_plan(plan):
    metric = plan.get('metric', 'default')
    mapping = ANALYSIS_LOGIC.get(metric, {})
    columns = mapping.get('columns', [])
    group_by = plan.get("group_by") or mapping.get("group_by")
    operation = mapping.get("operation", metric)
    return columns, group_by, operation

def build_odata_query(plan, group_by):
    filters = []
    # Azure needs DateTimeOffset to be like 2025-05-01T00:00:00Z
    def _datefmt(d, end=False):
        if not d: return None
        if 'T' in d: return d  # Already iso
        return f"{d}T23:59:59Z" if end else f"{d}T00:00:00Z"
    if plan.get('start_date') and plan.get('end_date'):
        start = _datefmt(plan['start_date'])
        end = _datefmt(plan['end_date'], end=True)
        filters.append(f"fkdat ge {start} and fkdat le {end}")
    if plan.get('compare_values') and group_by:
        group_field = group_by[0]
        compare = [f"{group_field} eq '{v}'" for v in plan['compare_values']]
        filters.append(f"({' or '.join(compare)})")
    filter_str = " and ".join(filters)
    facets = ",".join([g for g in group_by if g in FACETABLE_FIELDS]) if group_by else None
    return filter_str, facets

def fetch_data_from_search(filter_str, facets=None):
    try:
        kwargs = dict(
            search_text='*',
            filter=filter_str if filter_str else None,
            top=1000
        )
        if facets:
            kwargs['facets'] = [facets]
        print("fetch_data_from_search: kwargs:", kwargs)
        results = list(search_client.search(**kwargs))
        return results
    except Exception as ex:
        print("fetch_data_from_search ERROR:", ex)
        return []

def aggregate_sum(docs, field):
    total = 0.0
    for doc in docs:
        try:
            value = float(doc.get(field, 0) or 0)
            total += value
        except Exception:
            continue
    return total

def handle_deterioration_rate(plan, docs):
    field = plan.get('field', 'Revenue')
    current = aggregate_sum(docs, field)
    s = datetime.strptime(plan['start_date'], '%Y-%m-%d')
    e = datetime.strptime(plan['end_date'], '%Y-%m-%d')
    days = (e - s).days + 1
    prev_end = s - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    prev_plan = plan.copy()
    prev_plan['start_date'] = prev_start.strftime('%Y-%m-%d')
    prev_plan['end_date'] = prev_end.strftime('%Y-%m-%d')
    prev_filter, _ = build_odata_query(prev_plan, plan.get('group_by') or [])
    prev_docs = fetch_data_from_search(prev_filter)
    prev_total = aggregate_sum(prev_docs, field)
    if prev_total == 0:
        rate = None
    else:
        rate = ((prev_total - current) / prev_total) * 100
    return {'deterioration_rate_percent': rate, 'current': current, 'previous': prev_total}

def handle_profit_loss(plan, docs):
    revenue = aggregate_sum(docs, 'Revenue')
    cost = aggregate_sum(docs, 'Cost')
    profit = revenue - cost
    return {'revenue': revenue, 'cost': cost, 'profit': profit}

def handle_trend_analysis(plan, docs):
    group_by = plan.get('group_by', [])
    field = plan.get('field', 'Revenue')
    if not group_by:
        return {"error": "No group_by specified. Please specify a field to group by, like product, dealer, brand, etc."}
    group_field = group_by[0]
    current_agg = {}
    for doc in docs:
        k = doc.get(group_field)
        if not k:
            continue
        try:
            value = float(doc.get(field, 0) or 0)
        except Exception:
            value = 0.0
        current_agg.setdefault(k, 0.0)
        current_agg[k] += value
    # Previous period
    s = datetime.strptime(plan['start_date'], '%Y-%m-%d')
    e = datetime.strptime(plan['end_date'], '%Y-%m-%d')
    days = (e - s).days + 1
    prev_end = s - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    prev_plan = plan.copy()
    prev_plan['start_date'] = prev_start.strftime('%Y-%m-%d')
    prev_plan['end_date'] = prev_end.strftime('%Y-%m-%d')
    prev_filter, _ = build_odata_query(prev_plan, group_by)
    prev_docs = fetch_data_from_search(prev_filter)
    prev_agg = {}
    for doc in prev_docs:
        k = doc.get(group_field)
        if not k:
            continue
        try:
            value = float(doc.get(field, 0) or 0)
        except Exception:
            value = 0.0
        prev_agg.setdefault(k, 0.0)
        prev_agg[k] += value
    # Compare
    trend = {}
    for k in set(current_agg) | set(prev_agg):
        cur = current_agg.get(k, 0.0)
        pre = prev_agg.get(k, 0.0)
        direction = 'up' if cur > pre else 'down' if cur < pre else 'flat'
        trend[k] = {'current': cur, 'previous': pre, 'trend': direction}
    return trend

def handle_comparison(plan, docs):
    field = plan.get('field', 'Revenue')
    group_by = plan.get('group_by', ['vkorg'])
    group_field = group_by[0]
    compare_results = {}
    for val in plan.get('compare_values', []):
        filtered = [doc for doc in docs if doc.get(group_field) == val]
        compare_results[val] = aggregate_sum(filtered, field)
    return compare_results

def handle_profitability(plan, docs):
    group_by = plan.get('group_by', ['product'])
    group_field = group_by[0]
    agg = {}
    for doc in docs:
        k = doc.get(group_field)
        if not k:
            continue
        try:
            rev = float(doc.get('Revenue', 0) or 0)
            cost = float(doc.get('Cost', 0) or 0)
            profit = rev - cost
        except Exception:
            profit = 0.0
        agg.setdefault(k, 0.0)
        agg[k] += profit
    if not agg:
        return {}
    most_prof = max(agg.items(), key=lambda x: x[1])
    return {'most_profitable': most_prof[0], 'profit': most_prof[1]}

def handle_bottom_n(plan, docs, n=10):
    group_by = plan.get('group_by', ['product'])
    group_field = group_by[0]
    field = plan.get('field', 'Revenue')
    agg = {}
    for doc in docs:
        k = doc.get(group_field)
        if not k:
            continue
        try:
            value = float(doc.get(field, 0) or 0)
        except Exception:
            value = 0.0
        agg.setdefault(k, 0.0)
        agg[k] += value
    if not agg:
        return {}
    bottom = sorted(agg.items(), key=lambda x: x[1])[:n]
    return {'bottom_n': bottom}

def handle_default(plan, docs):
    field = plan.get('field', 'Revenue')
    total = aggregate_sum(docs, field)
    if not docs or total == 0:
        return {
            "found": False,
            "field": field,
            "total": 0,
            "message": (
                f"No {field} data was found in the period "
                f"{plan.get('start_date', '')} to {plan.get('end_date', '')}."
            )
        }
    else:
        return {
            "found": True,
            "field": field,
            "total": total,
            "message": (
                f"Total {field} from {plan.get('start_date', '')} to {plan.get('end_date', '')} is {total}."
            )
        }

def generate_llm_response(plan, result):
    try:
        context = f"User's sales analytics request: {json.dumps(plan, indent=2)}\nCalculated result: {json.dumps(result, indent=2)}"
        prompt = (
            f"You are a business data analyst. Given the SAP sales analysis request and the calculated result below, "
            "write a short, clear business summary for a business leader. Be specific about the trend, comparison, profit, or insight."
            "\n\n" + context
        )
        llm_summary = llm.invoke(prompt)
        return getattr(llm_summary, "content", str(llm_summary))
    except Exception:
        return f"Result: {json.dumps(result, indent=2)}"

def handle_user_query(prompt):
    try:
        plan = parse_prompt_to_plan(prompt)
        print("handle_user_query: PARSED PLAN =", plan)
        columns, group_by, operation = logic_from_plan(plan)
        filter_str, facets = build_odata_query(plan, group_by or [])
        docs = fetch_data_from_search(filter_str, facets)
        # Operation routing
        if operation == "deterioration_rate":
            result = handle_deterioration_rate(plan, docs)
        elif operation == "profit_loss":
            result = handle_profit_loss(plan, docs)
        elif operation == "trend_analysis":
            result = handle_trend_analysis(plan, docs)
        elif operation == "comparison":
            result = handle_comparison(plan, docs)
        elif operation == "profitability":
            result = handle_profitability(plan, docs)
        elif operation == "bottom_n":
            result = handle_bottom_n(plan, docs, n=10)
        else:
            result = handle_default(plan, docs)
        return generate_llm_response(plan, result)
    except Exception as ex:
        print("handle_user_query ERROR:", ex)
        return (
            "Sorry, I couldn't process your request. Please try a more specific question, "
            "such as including a date, metric, or sales entity (dealer, product, etc)."
        )
