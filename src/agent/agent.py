import os
import json
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
# from langchain_community.llms.openai import AzureOpenAI
from langchain_openai import AzureOpenAI

from langchain.memory import ConversationBufferMemory
from django.conf import settings
from langchain_openai import AzureChatOpenAI
# -------------------------------
# 1. CONFIGURATION
# -------------------------------

search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)

# llm = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     api_version="2024-02-15-preview",
# )

llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT,   # <- REQUIRED!
    api_version="2025-01-01-preview"
)
print(llm.invoke("Say hello in JSON."))
deployment = settings.AZURE_OPENAI_DEPLOYMENT

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
    'profit':               'Profit'
}


# Logic mapping: maps metric/intent to required fields, operation, and grouping
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
        "group_by": ["dealer", "month"],  # or override with user plan
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

# -------------------------------
# 2. LLM PROMPT PARSING
# -------------------------------

def parse_prompt_to_plan(prompt):
    instructions = (
        "You are an expert SAP sales analytics assistant. "
        "Given a user request, extract and return as strict JSON: "
        "- metric: one of [profit_loss, deterioration_rate, trend, comparison, profitability, low_revenue, default]. "
        "- field: (e.g. revenue, profit, cost, dealer, brand, product, tr, etc.) "
        "- group_by: list of fields to group by (if needed). "
        "- compare_values: list of values for comparison (if any). "
        "- start_date: YYYY-MM-DD. "
        "- end_date: YYYY-MM-DD. "
        "If the user simply asks about 'Revenue' or any single value, use 'default' as metric."
    )
    llm_prompt = f"{instructions}\nUser: {prompt}\nJSON:"
    print("==== LLM PROMT ====")
    print(llm_prompt)
    try:
        llm_response = llm.invoke(llm_prompt).strip()
        print("==== LLM RAW RESPONSE ====")
        print(llm_response)  # <--- ADD THIS LINE
        # If it's a message object, extract content:
        if hasattr(llm_response, "content"):
            llm_content = llm_response.content
        else:
            llm_content = str(llm_response)
        print("==== LLM EXTRACTED CONTENT ====")
        print(llm_content)
        try:
            plan = json.loads(llm_response.split('```json')[-1].split('```')[0] if '```json' in llm_response else llm_response)
        except Exception:
            plan = json.loads(llm_response)
        # >>> THIS IS THE FIX <<<
        if "metric" not in plan or not plan.get("metric"):
            plan["metric"] = "default"
        return plan
    except Exception:
        return {}

# -------------------------------
# 3. LOGIC MAPPING & ODATA BUILDER
# -------------------------------

def logic_from_plan(plan):
    metric = plan.get('metric', 'default')
    mapping = ANALYSIS_LOGIC.get(metric, {})
    columns = mapping.get('columns', [])
    group_by = plan.get("group_by") or mapping.get("group_by")
    operation = mapping.get("operation", metric)
    return columns, group_by, operation

def build_odata_query(plan, group_by):
    filters = []
    if plan.get('start_date') and plan.get('end_date'):
        filters.append(f"fkdat ge '{plan['start_date']}' and fkdat le '{plan['end_date']}'")
    if plan.get('compare_values') and group_by:
        # Only use the first group_by for compare_values
        group_field = FIELD_SYNONYMS.get(group_by[0].lower(), group_by[0])
        compare = [f"{group_field} eq '{v}'" for v in plan['compare_values']]
        filters.append(f"({' or '.join(compare)})")
    filter_str = " and ".join(filters)
    facets = ",".join([FIELD_SYNONYMS.get(g.lower(), g) for g in group_by]) if group_by else None
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
        results = list(search_client.search(**kwargs))
        return results
    except Exception:
        return []

def aggregate_sum(docs, field):
    try:
        return sum(doc.get(field, 0.0) for doc in docs if field in doc)
    except Exception:
        return 0

# -------------------------------
# 4. ANALYSIS TOOL HANDLERS
# -------------------------------

def handle_deterioration_rate(plan, docs):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
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
    # Trend by dealer/month/brand/etc. Compute change per group
    group_by = plan.get('group_by', ['dealer'])
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    group_field = FIELD_SYNONYMS.get(group_by[0].lower(), group_by[0])
    # Current
    current_agg = {}
    for doc in docs:
        key = doc.get(group_field)
        if not key:
            continue
        current_agg.setdefault(key, 0)
        current_agg[key] += doc.get(field, 0.0)
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
        key = doc.get(group_field)
        if not key:
            continue
        prev_agg.setdefault(key, 0)
        prev_agg[key] += doc.get(field, 0.0)
    # Compare
    trend = {}
    for k in set(current_agg) | set(prev_agg):
        cur = current_agg.get(k, 0)
        pre = prev_agg.get(k, 0)
        direction = 'up' if cur > pre else 'down' if cur < pre else 'flat'
        trend[k] = {'current': cur, 'previous': pre, 'trend': direction}
    return trend

def handle_comparison(plan, docs):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    group_by = plan.get('group_by', ['vkorg'])
    group_field = FIELD_SYNONYMS.get(group_by[0].lower(), group_by[0])
    compare_results = {}
    for val in plan.get('compare_values', []):
        filtered = [doc for doc in docs if doc.get(group_field) == val]
        compare_results[val] = aggregate_sum(filtered, field)
    return compare_results

def handle_profitability(plan, docs):
    field = FIELD_SYNONYMS.get('profit', 'Profit')
    group_by = plan.get('group_by', ['product'])
    group_field = FIELD_SYNONYMS.get(group_by[0].lower(), group_by[0])
    agg = {}
    for doc in docs:
        k = doc.get(group_field)
        if not k:
            continue
        rev = doc.get('Revenue', 0.0)
        cost = doc.get('Cost', 0.0)
        profit = rev - cost
        agg.setdefault(k, 0)
        agg[k] += profit
    if not agg:
        return {}
    most_prof = max(agg.items(), key=lambda x: x[1])
    return {'most_profitable': most_prof[0], 'profit': most_prof[1]}

def handle_bottom_n(plan, docs, n=10):
    # Bottom N by revenue
    group_by = plan.get('group_by', ['product'])
    group_field = FIELD_SYNONYMS.get(group_by[0].lower(), group_by[0])
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    agg = {}
    for doc in docs:
        k = doc.get(group_field)
        if not k:
            continue
        agg.setdefault(k, 0)
        agg[k] += doc.get(field, 0.0)
    if not agg:
        return {}
    bottom = sorted(agg.items(), key=lambda x: x[1])[:n]
    return {'bottom_n': bottom}

def handle_default(plan, docs):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    total = aggregate_sum(docs, field)
    # Check if there is any data for the query
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

# -------------------------------
# 5. LLM BUSINESS SUMMARY
# -------------------------------

def generate_llm_response(plan, result):
    # Compose a prompt for LLM to generate a business summary
    try:
        context = f"User's sales analytics request: {json.dumps(plan, indent=2)}\nCalculated result: {json.dumps(result, indent=2)}"
        prompt = (
            f"You are a business data analyst. Given the SAP sales analysis request and the calculated result below, "
            "write a short, clear business summary for a business leader. Be specific about the trend, comparison, profit, or insight."
            "\n\n" + context
        )
        llm_summary = llm(prompt)
        return llm_summary
    except Exception:
        # If LLM fails, fall back to basic answer
        return f"Result: {json.dumps(result, indent=2)}"

# -------------------------------
# 6. MAIN AGENT FUNCTION
# -------------------------------

def handle_user_query(prompt):
    try:
        plan = parse_prompt_to_plan(prompt)
        # >>> ANOTHER SAFETY FIX <<<
        # print( "handle_use_query ",parse_prompt_to_plan)
        print("handle_user_query: PARSED PLAN =", plan)
        if not plan:
            return (
                "Sorry, I couldn't understand your request. "
                "Please try asking about a specific metric, period, or sales entity."
            )
        if "metric" not in plan or not plan.get("metric"):
            plan["metric"] = "default"
        columns, group_by, operation = logic_from_plan(plan)
        filter_str, facets = build_odata_query(plan, group_by or [])
        docs = fetch_data_from_search(filter_str, facets)
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
    except Exception:
        return (
            "Sorry, I couldn't process your request. Please try a more specific question, "
            "such as including a date, metric, or sales entity (dealer, product, etc)."
        )
# -------------------------------
# 7. TEST EXAMPLES
# -------------------------------

# if __name__ == "__main__":
#     user_prompts = [
#         "What is the deterioration rate of Revenue in March 2025?",
#         "Show profit vs loss for April 2025.",
#         "Which dealer is uptrending in Q2 2025?",
#         "Compare Sales Org 1000 vs 2000 in 2024.",
#         "Which products are left and not generate much revenue?",
#         "Who is the exclusive dealer in 2025?",
#         "Which brand is in declining stage from March 2025 to April 2025?"
#     ]
#     for user_prompt in user_prompts:
#         print(f"\nUser Prompt: {user_prompt}")
#         print(handle_user_query(user_prompt))
