import os
import json
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from django.conf import settings

from agent.utils.helpers import extract_date_range_from_prompt, ensure_azure_datetime

FIELD_SYNONYMS = {
    'revenue':      'Revenue',
    'quantity':     'fkimg',
    'volume':       'volum',
    'customer':     'cname',
    'brand':        'wgbez',
    'product':      'arktx',
    'category':     'matkl',
    'division':     'spart_text',
    'company code': 'bukrs',
    'sales org':    'vkorg',
    'dist channel': 'vtweg',
    'business area':'gsber',
    'customer group':'kukla',
    'account group':'ktokd',
    'territory':    'Territory',
    'sales zone':   'Szone',
    'date':         'fkdat',
    'fkdat':        'fkdat'
}

search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2025-01-01-preview"
)
deployment = settings.AZURE_OPENAI_DEPLOYMENT

FN_DEF = [{
    "name": "parse_sales_query",
    "description": "Extract a sales metric operation plan (with OData $filter syntax) from a user prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": ["deterioration_rate", "profit_loss", "trend", "comparison", "profitability", "default"]},
            "field": {"type": "string", "description": "Field to aggregate or analyze"},
            "group_by": {"type": "array", "items": {"type": "string"}, "description": "Group by these fields"},
            "filter": {"type": "string", "description": "OData $filter"},
            "compare_values": {"type": "array", "items": {"type": "string"}, "description": "For comparison metric"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"}
        },
        "required": ["metric"]
    }
}]

def extract_operation_plan(prompt: str) -> dict:
    resp = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {'role': 'system', 'content': 'Parse the user’s request into a structured operation plan for sales analytics using OData $filter syntax.'},
            {'role': 'user', 'content': prompt}
        ],
        functions=FN_DEF,
        function_call={'name':'parse_sales_query'}
    )
    fc = resp.choices[0].message.function_call
    return json.loads(fc.arguments)

def compute_previous_period(start: str, end: str) -> tuple[str, str]:
    s = datetime.fromisoformat(start[:10])  # Only use date part
    e = datetime.fromisoformat(end[:10])
    days = (e - s).days + 1
    prev_end = s - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    return prev_start.isoformat(), prev_end.isoformat()

def aggregate_sum(filter_str: str, field: str = 'Revenue'):
    res = search_client.search(
        search_text='*',
        filter=filter_str,
        top=1000  # For big queries, you may need to handle pagination!
    )
    docs = list(res)
    return sum(doc.get(field, 0.0) for doc in docs)

# ===================== Metric Handlers =====================

def handle_deterioration_rate(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"
    curr = aggregate_sum(filter_str, field)
    # Previous period
    prev_start, prev_end = compute_previous_period(plan['start_date'], plan['end_date'])
    prev_start = ensure_azure_datetime(prev_start)
    prev_end = ensure_azure_datetime(prev_end)
    prev_filter_str = f"{filter_base} and fkdat ge {prev_start} and fkdat le {prev_end}" if filter_base else f"fkdat ge {prev_start} and fkdat le {prev_end}"
    prev = aggregate_sum(prev_filter_str, field)
    rate = ((prev - curr) / prev * 100) if prev else None
    trend = 'down' if rate and rate > 0 else 'up'
    return {'deterioration_rate': rate, 'trend': trend, 'current': curr, 'previous': prev}

def handle_profit_loss(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"
    curr = aggregate_sum(filter_str, field)
    return {'profit_loss': curr}

def handle_trend(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    group_by = FIELD_SYNONYMS.get(plan.get('group_by', ['cname'])[0].lower(), 'cname')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"
    prev_start, prev_end = compute_previous_period(plan['start_date'], plan['end_date'])
    prev_start = ensure_azure_datetime(prev_start)
    prev_end = ensure_azure_datetime(prev_end)
    prev_filter_str = f"{filter_base} and fkdat ge {prev_start} and fkdat le {prev_end}" if filter_base else f"fkdat ge {prev_start} and fkdat le {prev_end}"

    from collections import defaultdict
    curr_docs = search_client.search(search_text='*', filter=filter_str, top=1000)
    prev_docs = search_client.search(search_text='*', filter=prev_filter_str, top=1000)
    curr_agg = defaultdict(float)
    prev_agg = defaultdict(float)
    for doc in curr_docs:
        entity = doc.get(group_by)
        curr_agg[entity] += doc.get(field, 0.0)
    for doc in prev_docs:
        entity = doc.get(group_by)
        prev_agg[entity] += doc.get(field, 0.0)
    trend_by_entity = {}
    for entity in set(curr_agg) | set(prev_agg):
        c = curr_agg.get(entity, 0)
        p = prev_agg.get(entity, 0)
        if p == 0:
            trend = 'up' if c > 0 else 'flat'
        else:
            trend = 'up' if c > p else 'down'
        trend_by_entity[entity] = {'current': c, 'previous': p, 'trend': trend}
    return {'trend_by_entity': trend_by_entity}

def handle_comparison(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', '').lower(), plan.get('field', ''))
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    results = {}
    for val in plan.get('compare_values', []):
        filter_str = f"{field} eq '{val}' and fkdat ge {start} and fkdat le {end}"
        if filter_base:
            filter_str = f"{filter_base} and {filter_str}"
        results[val] = aggregate_sum(filter_str)
    return {'comparison': results}

def handle_profitability(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    group_by = FIELD_SYNONYMS.get('product', 'arktx')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"
    docs = search_client.search(search_text='*', filter=filter_str, top=1000)
    from collections import defaultdict
    agg = defaultdict(float)
    for doc in docs:
        prod = doc.get(group_by)
        agg[prod] += doc.get(field, 0.0)
    if not agg:
        return {}
    most_prof = max(agg, key=lambda k: agg[k])
    return {'most_profitable_product': most_prof, 'revenue': agg[most_prof]}

def handle_default(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"
    curr = aggregate_sum(filter_str, field)
    return {'total': curr}

# ===================== Central dispatcher =====================

def sales_metrics_engine(prompt: str):
    plan = extract_operation_plan(prompt)

    # Date range fallback
    if not plan.get('start_date') or not plan.get('end_date'):
        start, end = extract_date_range_from_prompt(prompt)
        plan['start_date'] = plan.get('start_date', start)
        plan['end_date'] = plan.get('end_date', end)

    # Always construct filter for date range (overrides if present)
    # Azure OData filters require ISO datetime!
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    if not plan.get('filter') or 'fkdat' not in plan['filter']:
        plan['filter'] = f"fkdat ge {start} and fkdat le {end}"

    metric = plan['metric']
    if metric == 'deterioration_rate':
        result = handle_deterioration_rate(plan)
    elif metric == 'profit_loss':
        result = handle_profit_loss(plan)
    elif metric == 'trend':
        result = handle_trend(plan)
    elif metric == 'comparison':
        result = handle_comparison(plan)
    elif metric == 'profitability':
        result = handle_profitability(plan)
    else:
        result = handle_default(plan)
    return {'operation_plan': plan, 'result': result}

def generate_llm_answer(prompt: str, result: dict) -> str:
    resp = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {'role': 'system', 'content': 'You are a helpful sales analyst.'},
            {'role': 'user', 'content': f"User asked: '{prompt}'"},
            {'role': 'system', 'content': f"Result: {json.dumps(result)}"}
        ]
    )
    return resp.choices[0].message.content
