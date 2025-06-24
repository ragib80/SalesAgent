import os
import json
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AzureOpenAI
from django.conf import settings
from azure.search.documents.models import VectorizedQuery

# Field synonyms
FIELD_SYNONYMS = {
    'revenue':      'Revenue',
    'quantity':     'fkimg',
    'volume':       'volum',
    'customer':     'cname',
    'brand':        'wgbez',
    'product':      'arktx',
    'category':     'matkl',
    'division':     'spart_text',
    'company code':'bukrs',
    'sales org':    'vkorg',
    'dist channel':'vtweg',
    'business area':'gsber',
    'customer group':'kukla',
    'account group':'ktokd',
    'territory':    'Territory',
    'sales zone':   'Szone',
    'date':         'fkdat',
    'fkdat':        'fkdat'
}

# Initialize Azure Search client
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)
# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2025-01-01-preview"
)
# Name of your Chat/Embedding deployment in Azure OpenAI Studio
deployment = settings.AZURE_OPENAI_DEPLOYMENT


def embed_query(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model='text-embedding-ada-002',
        input=[text]
    )
    return resp.data[0].embedding


def parse_prompt(prompt: str) -> dict:
    """Use function-calling to extract metric, date ranges, field, compare_values"""
    fn_def = [{
        'name': 'parse_query',
        'description': 'Extract metric, date range, field and compare values',
        'parameters': {
            'type': 'object',
            'properties': {
                'metric': {'type': 'string','enum':[
                    'deterioration_rate','profit_loss','trend',
                    'comparison','profitability','default'
                ]},
                'start_date':   {'type': 'string'},
                'end_date':     {'type': 'string'},
                'field':        {'type': 'string'},
                'compare_values': {'type': 'array','items': {'type': 'string'}}
            },
            'required': ['metric','start_date','end_date']
        }
    }]

    resp = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {'role': 'system', 'content': 'Parse user query to JSON'},
            {'role': 'user', 'content': prompt}
        ],
        functions=fn_def,
        function_call={'name': 'parse_query'}
    )
    fc = resp.choices[0].message.function_call
    return json.loads(fc.arguments)


def aggregate_sum(date_filter: str) -> float:
    res = search_client.search(
        search_text='*',
        filter=date_filter,
        top=0,
        facets=['Revenue,sum']
    )
    facets = res.get_facets() or {}
    return facets.get('Revenue', [{'sum': 0.0}])[0]['sum']


def compute_previous_period(start: str, end: str) -> tuple[str, str]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    days = (e - s).days + 1
    prev_end = s - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    return prev_start.date().isoformat(), prev_end.date().isoformat()


def query_sales(prompt: str, page: int = 1, page_size: int = 20) -> dict:
    qd = parse_prompt(prompt)
    metric, start, end = qd['metric'], qd['start_date'], qd['end_date']
    date_filter = f"fkdat ge {start}Z and fkdat le {end}Z"

    if metric == 'profit_loss':
        curr = aggregate_sum(date_filter)
        ps, pe = compute_previous_period(start, end)
        prev = aggregate_sum(f"fkdat ge {ps}Z and fkdat le {pe}Z")
        return {'profit_loss': curr - prev, 'current': curr, 'previous': prev}

    if metric in ['deterioration_rate', 'trend']:
        curr = aggregate_sum(date_filter)
        ps, pe = compute_previous_period(start, end)
        prev = aggregate_sum(f"fkdat ge {ps}Z and fkdat le {pe}Z")
        rate = ((prev - curr) / prev * 100) if prev else None
        tr = 'down' if rate and rate > 0 else 'up'
        return {'deterioration_rate': rate, 'trend': tr, 'current': curr, 'previous': prev}

    if metric == 'comparison':
        field = FIELD_SYNONYMS.get(qd.get('field', ''), qd.get('field', ''))
        comp = {}
        for v in qd.get('compare_values', []):
            fstr = f"{date_filter} and {field} eq '{v}'"
            comp[v] = aggregate_sum(fstr)
        return {'comparison': comp}

    if metric == 'profitability':
        rs = search_client.search(
            search_text='*',
            filter=date_filter,
            top=0,
            facets=['arktx,count:0,sum(Revenue)']
        )
        prods = rs.get_facets().get('arktx', [])
        best = max(prods, key=lambda x: x.get('sum', 0)) if prods else {}
        return {'most_profitable_product': best.get('value'), 'revenue': best.get('sum')}

    # Default: vector search + pagination
    vec = embed_query(prompt)
    vq = VectorizedQuery(
        vector=vec,
        k_nearest_neighbors=page_size,
        fields="embedding"
    )
    try:
        res = search_client.search(
            vector_queries=[vq],
            vector_filter_mode="preFilter",
            filter=date_filter,
            skip=(page-1)*page_size,
            top=page_size,
            include_total_count=True
        )
    except TypeError:
        # Fallback if vector_queries unsupported
        res = search_client.search(
            search_text='*',
            filter=date_filter,
            skip=(page-1)*page_size,
            top=page_size,
            include_total_count=True
        )
    docs = [
        {'fkdat': r.get('fkdat'), 'wgbez': r.get('wgbez'), 'Revenue': r.get('Revenue')}
        for r in res
    ]
    return {'documents': docs, 'total_count': res.get_count(), 'page': page, 'page_size': page_size}


def generate_answer(prompt: str, data: dict) -> str:
    resp = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {'role': 'system', 'content': 'You are a helpful sales analyst.'},
            {'role': 'user', 'content': f"User asked: '{prompt}'"},
            {'role': 'system', 'content': f"Data provided: {json.dumps(data)}"}
        ]
    )
    return resp.choices[0].message.content