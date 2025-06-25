import os
import json
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from django.conf import settings
from conversation.models.message import Message

from agent.utils.helpers import extract_date_range_from_prompt, ensure_azure_datetime,extract_dates_from_past_messages

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
    'business area': 'gsber',
    'customer group': 'kukla',
    'account group': 'ktokd',
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
        function_call={'name': 'parse_sales_query'}
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


def handle_deterioration_rate(plan, conversation=None):
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

    # Get previous revenue, check if it exists
    prev = aggregate_sum(prev_filter_str, field)

    if prev == 0 or prev is None:
        # If there's no previous data (or it's zero), we can't calculate deterioration
        # Fallback to LLM for assistance
        return handle_fallback_to_llm(plan, curr, prev, "deterioration_rate", conversation)

    # Calculate deterioration rate if previous data exists
    rate = ((prev - curr) / prev * 100) if prev else None
    trend = 'down' if rate and rate > 0 else 'up'
    
    return {
        'deterioration_rate': rate,
        'trend': trend,
        'current': curr,
        'previous': prev,
        'message': None  # No additional message if calculation is successful
    }

def handle_fallback_to_llm(plan, curr, prev, metric, conversation):
    # If the deterioration rate calculation fails, fallback to LLM
    message = f"Could not calculate the {metric} due to missing or zero previous data for the period {plan['start_date']} to {plan['end_date']}."

    # Send the prompt to LLM to ask for clarification or provide suggestions
    prompt = f"{message} Based on the provided revenue data, how should we proceed with calculating the {metric}? Please provide a specific period or clarify the data."

    # Optionally include past conversation context to LLM
    response = generate_llm_answer(prompt, plan, conversation)

    return {
        "answer": response,
        "suggestions": [
            "Please provide the full period's revenue data.",
            "Try asking with a specific quarter or year to get the revenue calculation."
        ],
        "data": None
    }




def handle_profit_loss(plan):
    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"
    curr = aggregate_sum(filter_str, field)
    return {'profit_loss': curr}


def handle_trend(plan):
    """
    Handles trend analysis, comparing sales data between two periods.
    The trend is determined based on whether sales are increasing, decreasing, or remaining the same.
    """
    group_by_fields = plan.get('group_by', [])
    if not group_by_fields:
        # Fallback: group by 'cname'
        group_by_fields = ['cname']
    group_by_field = FIELD_SYNONYMS.get(group_by_fields[0].lower(), group_by_fields[0])

    field = FIELD_SYNONYMS.get(plan.get('field', 'revenue').lower(), 'Revenue')
    start = ensure_azure_datetime(plan['start_date'])
    end = ensure_azure_datetime(plan['end_date'])
    filter_base = plan.get('filter', '')
    filter_str = f"{filter_base} and fkdat ge {start} and fkdat le {end}" if filter_base else f"fkdat ge {start} and fkdat le {end}"

    prev_start, prev_end = compute_previous_period(plan['start_date'], plan['end_date'])
    prev_start = ensure_azure_datetime(prev_start)
    prev_end = ensure_azure_datetime(prev_end)
    prev_filter_str = f"{filter_base} and fkdat ge {prev_start} and fkdat le {prev_end}" if filter_base else f"fkdat ge {prev_start} and fkdat le {prev_end}"

    # Fetch current and previous period sales data
    curr_docs = search_client.search(search_text='*', filter=filter_str, top=1000)
    prev_docs = search_client.search(search_text='*', filter=prev_filter_str, top=1000)

    # Aggregate sales for current and previous periods
    from collections import defaultdict
    curr_agg = defaultdict(float)
    prev_agg = defaultdict(float)

    for doc in curr_docs:
        entity = doc.get(group_by_field)
        curr_agg[entity] += doc.get(field, 0.0)

    for doc in prev_docs:
        entity = doc.get(group_by_field)
        prev_agg[entity] += doc.get(field, 0.0)

    # Compare sales and determine the trend
    trend_by_entity = {}
    for entity in set(curr_agg) | set(prev_agg):
        current_sales = curr_agg.get(entity, 0)
        previous_sales = prev_agg.get(entity, 0)

        if previous_sales == 0:
            # If previous sales are 0, consider it as an uptrend if current sales > 0
            trend = 'up' if current_sales > 0 else 'flat'
        else:
            # Compare current vs. previous sales
            trend = 'up' if current_sales > previous_sales else 'down'

        trend_by_entity[entity] = {
            'current': current_sales,
            'previous': previous_sales,
            'trend': trend
        }

    return {'trend_by_entity': trend_by_entity}


def handle_comparison(plan):
    field = FIELD_SYNONYMS.get(
        plan.get('field', '').lower(), plan.get('field', ''))
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
def extract_periods_from_prompt(prompt):
    """
    Extracts periods from the user’s prompt. This function handles time-based queries like Q1, Q2, or specific months.
    """
    periods = []
    
    # Match for quarters (e.g., Q1 2025, Q2 2025)
    if "Q1" in prompt:
        periods.append({"name": "Q1 2025", "start_date": "2025-01-01", "end_date": "2025-03-31"})
    if "Q2" in prompt:
        periods.append({"name": "Q2 2025", "start_date": "2025-04-01", "end_date": "2025-06-30"})
    if "Q3" in prompt:
        periods.append({"name": "Q3 2025", "start_date": "2025-07-01", "end_date": "2025-09-30"})
    if "Q4" in prompt:
        periods.append({"name": "Q4 2025", "start_date": "2025-10-01", "end_date": "2025-12-31"})
    
    # Match for specific months (e.g., "March 2025", "April 2025")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for month in months:
        if month in prompt:
            periods.append({"name": f"{month} 2025", "start_date": f"2025-{months.index(month)+1:02d}-01", "end_date": f"2025-{months.index(month)+1:02d}-28"})
    
    # Add custom date ranges if mentioned
    if "from" in prompt and "to" in prompt:
        start_date = extract_date_range_from_prompt(prompt)[0]
        end_date = extract_date_range_from_prompt(prompt)[1]
        periods.append({"name": f"{start_date} to {end_date}", "start_date": start_date, "end_date": end_date})
    
    return periods


def sales_metrics_engine(prompt: str, conversation=None):
    try:
        # 1. Extract operation plan from the user's prompt
        plan = extract_operation_plan(prompt)

        # 2. Handle missing date ranges dynamically
        if not plan.get('start_date') or not plan.get('end_date'):
            # Try to extract the date range from past messages in the conversation
            date_ranges = extract_dates_from_past_messages(conversation)

            if date_ranges:
                plan['start_date'], plan['end_date'] = date_ranges[-2]  # Use the last two periods from conversation
            else:
                # Handle the case where no date ranges can be found
                return {
                    "answer": "I couldn't find any date ranges in your request or previous messages. Can you please provide a specific date range?",
                    "suggestions": [
                        "Try asking for a specific date range: 'Revenue for March 2025'.",
                        "Or use a quarter or year: 'Revenue for Q1 2025' or 'Revenue for 2025'."
                    ],
                    "data": None
                }

        # Ensure the filter includes the date range (using OData filter syntax)
        start = ensure_azure_datetime(plan['start_date'])
        end = ensure_azure_datetime(plan['end_date'])
        if not plan.get('filter') or 'fkdat' not in plan['filter']:
            plan['filter'] = f"fkdat ge {start} and fkdat le {end}"

        # 3. Handle vague prompts such as "compare"
        vague_prompts = ["compare", "compare that", "what about that", "more", "show more"]
        if any(phrase in prompt.lower() for phrase in vague_prompts):
            periods_to_compare = extract_periods_from_prompt(prompt)
            print(periods_to_compare)

            if not periods_to_compare:
                # If periods are not provided, use the last two periods from past messages
                # periods_to_compare = [{"name": "March 2025", "start_date": "2025-03-01", "end_date": "2025-03-31"},
                #                       {"name": "May 2025", "start_date": "2025-05-01", "end_date": "2025-05-31"}]
                return handle_fallback_to_llm(prompt, "date range", conversation)
            comparison_results = {}
            
            for period in periods_to_compare:
                filter_str = f"fkdat ge {period['start_date']} and fkdat le {period['end_date']}"
                try:
                    sales_data = aggregate_sum(filter_str, 'Revenue')
                    comparison_results[period['name']] = sales_data
                except Exception as e:
                    # Fallback to LLM if there's an issue with comparison
                    return handle_fallback_to_llm(prompt, "comparison", conversation)

            # Construct the comparison message
            comparison_message = "Here is the comparison of the sales data:\n"
            for period, sales in comparison_results.items():
                comparison_message += f"{period}: {sales}\n"

            return {
                "answer": comparison_message,
                "result": comparison_results,
                "operation_plan": None
            }

        # Regular handling for other metrics (e.g., deterioration_rate, profit_loss)
        metric = plan['metric']
        try:
            if metric == 'deterioration_rate':
                result = handle_deterioration_rate(plan)
            elif metric == 'profit_loss':
                result = handle_profit_loss(plan)
            elif metric == 'trend':
                result = handle_trend(plan)
            elif metric == 'comparison':
                result = handle_comparison(plan, conversation)
            elif metric == 'profitability':
                result = handle_profitability(plan)
            else:
                result = handle_default(plan)
        except Exception as e:
            # Fallback to LLM model for ambiguous or missing data
            return handle_fallback_to_llm(prompt, "general error", conversation)
        return {'operation_plan': plan, 'result': result}

    except Exception as e:
        # Catch unhandled exceptions and ask LLM for clarification
         return handle_fallback_to_llm(prompt, "general error", conversation)

def handle_fallback_to_llm(prompt, issue_type, conversation=None):
    """
    If the sales_metrics_engine cannot process the request, it will call the LLM for clarification.
    """
    # Build a prompt for the LLM model based on the issue type
    if issue_type == "date range":
        message = "I couldn't find a date range in your request. Could you please specify a date range, from the previous history'?"
    elif issue_type == "comparison":
        message = "I couldn't fetch sales data for the specified periods. Could you please provide specific periods like 'Revenue for March 2025 and May 2025'?"
    elif issue_type == "general error":
        message = "There seems to be an issue processing your request. Could you please provide more specific details about what you're looking for?"
    
    # Send the message to the LLM model for further processing
    prompt = f"{message} Here's what I know so far: {prompt}"
    response = generate_llm_answer(prompt, None, conversation)

    return {
        "answer": response,
        "suggestions": [
            "You can ask for revenue by specific month or year: 'Revenue for March 2025'.",
            "Or you can compare periods like: 'Compare revenue for Q1 2025 and Q2 2025'."
        ],
        "data": None
    }


# def generate_llm_answer(prompt: str, result: dict) -> str:
#     resp = openai_client.chat.completions.create(
#         model=deployment,
#         messages=[
#             {'role': 'system', 'content': 'You are a helpful sales analyst.'},
#             {'role': 'user', 'content': f"User asked: '{prompt}'"},
#             {'role': 'system', 'content': f"Result: {json.dumps(result)}"}
#         ]
#     )
#     return resp.choices[0].message.content


def generate_llm_answer(prompt: str, result: dict, conversation=None, suggestions=None) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a helpful SAP sales analyst.'}
    ]
    
    # Include previous conversation history, clearly marked
    if conversation:
        history_messages = Message.objects.filter(
            conversation=conversation, is_deleted=False
        ).order_by('created_at')[:10]  # oldest to newest
        
        # Add each message as part of the conversation history
        for msg in history_messages:
            role = 'user' if msg.sender == 'user' else 'system'
            messages.append({'role': role, 'content': f"[Previous message] {msg.text}"})
    
    # Add the new user prompt (clearly marked as new prompt)
    messages.append({'role': 'user', 'content': f"[New user prompt] {prompt}"})
    
    # If any result data exists (from RAG), include that as well
    if result:
        messages.append({'role': 'system', 'content': f"Result: {json.dumps(result)}"})
    
    # Optionally, include suggestions if available
    if suggestions:
        messages.append({'role': 'system', 'content': f"Suggestions: {json.dumps(suggestions)}"})
    print ("messages before call llm ",messages)
    # Send the conversation context + user prompt to LLM
    resp = openai_client.chat.completions.create(
        model=deployment,
        messages=messages
    )

    # Return the LLM's generated response
    return resp.choices[0].message.content

