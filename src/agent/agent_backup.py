import os
import json
from datetime import datetime, timedelta
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery
from azure.ai.openai import (
    OpenAIClient,
    ChatCompletionOptions,
    ChatRole
)

# Field synonyms map
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

# Initialize clients
search_client = SearchClient(
    endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
    credential=os.getenv('AZURE_SEARCH_KEY')
)
openai_client = OpenAIClient(
    endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    credential=os.getenv('AZURE_OPENAI_KEY')
)
deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')


def embed_query(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model_name='text-embedding-ada-002',
        input=[text]
    )
    return resp.data[0].embedding


def parse_prompt(prompt: str) -> dict:
    """
    Uses function-calling to extract:
      - metric: one of [deterioration_rate, profit_loss, trend, comparison, profitability, default]
      - start_date, end_date (ISO strings)
      - optional field & compare_values for comparison
    """
    fn_def = [{
        'name': 'parse_query',
        'description': 'Extract metric, date range, field and compare values',
        'parameters': {
            'type': 'object',
            'properties': {
                'metric': {
                    'type': 'string',
                    'enum': [
                        'deterioration_rate', 'profit_loss',
                        'trend', 'comparison',
                        'profitability', 'default'
                    ]
                },
                'start_date':   {'type': 'string'},
                'end_date':     {'type': 'string'},
                'field':        {'type': 'string'},
                'compare_values': {
                    'type': 'array',
                    'items': {'type': 'string'}
                }
            },
            'required': ['metric', 'start_date', 'end_date']
        }
    }]

    opts = ChatCompletionOptions(
        model=deployment,
        messages=[
            ChatRole(system="Parse user query to JSON request for search metrics"),
            ChatRole(user=prompt)
        ],
        functions=fn_def,
        function_call={'name': 'parse_query'}
    )
    resp = openai_client.chat_completions.create(options=opts)
    return json.loads(resp.choices[0].message.function_call.arguments)


def aggregate_sum(date_filter: str) -> float:
    # numeric facet sum of Revenue
    results = search_client.search(
        search_text="*",
        filter=date_filter,
        top=0,
        facets=["Revenue,sum"]
    )
    facets = results.get_facets() or {}
    revenue_facet = facets.get('Revenue', [])
    return revenue_facet[0].get('sum', 0.0) if revenue_facet else 0.0


def compute_previous_period(start: str, end: str) -> tuple[str,str]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    days = (e - s).days + 1
    prev_end = s - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days-1)
    return prev_start.date().isoformat(), prev_end.date().isoformat()


def query_sales(prompt: str, page: int = 1, page_size: int = 20) -> dict:
    qd = parse_prompt(prompt)
    metric = qd['metric']
    start = qd['start_date']
    end   = qd['end_date']
    date_filter = f"fkdat ge {start}Z and fkdat le {end}Z"

    if metric == 'profit_loss':
        curr = aggregate_sum(date_filter)
        ps, pe = compute_previous_period(start, end)
        prev = aggregate_sum(f"fkdat ge {ps}Z and fkdat le {pe}Z")
        return { 'profit_loss': curr - prev, 'current': curr, 'previous': prev }

    if metric in ['deterioration_rate', 'trend']:
        curr = aggregate_sum(date_filter)
        ps, pe = compute_previous_period(start, end)
        prev = aggregate_sum(f"fkdat ge {ps}Z and fkdat le {pe}Z")
        rate = ((prev - curr) / prev * 100) if prev else None
        tr = 'down' if rate and rate > 0 else 'up'
        return { 'deterioration_rate': rate, 'trend': tr, 'current': curr, 'previous': prev }

    if metric == 'comparison':
        field = FIELD_SYNONYMS.get(qd.get('field',''), qd.get('field',''))
        comp = {}
        for v in qd.get('compare_values', []):
            f = f"{date_filter} and {field} eq '{v}'"
            comp[v] = aggregate_sum(f)
        return { 'comparison': comp }

    if metric == 'profitability':
        # facet by product name for sum of revenue
        rs = search_client.search(
            search_text="*",
            filter=date_filter,
            top=0,
            facets=["arktx,count:0,sum(Revenue)"]
        )
        facets = rs.get_facets() or {}
        prods = facets.get('arktx', [])
        best = max(prods, key=lambda x: x.get('sum',0)) if prods else {}
        return { 'most_profitable_product': best.get('value'), 'revenue': best.get('sum') }

    # --- default retrieval + pagination ---
    vec = embed_query(prompt)
    vq  = VectorQuery(value=vec, k=page_size, fields=['embedding'])
    results = search_client.search(
        vector=vq,
        filter=date_filter,
        skip=(page-1)*page_size,
        top=page_size,
        include_total_count=True
    )
    docs = []
    for r in results:
        docs.append({
            'fkdat':     r.get('fkdat'),
            'wgbez':     r.get('wgbez'),
            'Revenue':   r.get('Revenue'),
            # add any other fields you wish to show
        })
    return {
        'documents': docs,
        'total_count': results.get_count(),
        'page': page,
        'page_size': page_size
    }


def generate_answer(prompt: str, data: dict) -> str:
    # Send to OpenAI to craft a natural-language answer
    msgs = [
        ChatRole(system="You are a helpful sales analyst."),
        ChatRole(user=f"User asked: '{prompt}'"),
        ChatRole(system=f"Here is the data or metrics: {json.dumps(data)}")
    ]
    opts = ChatCompletionOptions(
        model=deployment,
        messages=msgs
    )
    resp = openai_client.chat_completions.create(options=opts)
    return resp.choices[0].message.content