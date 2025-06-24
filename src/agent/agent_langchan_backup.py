import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from django.conf import settings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import AzureChatOpenAI
import re
import logging

logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────
# 1) Your FIELD_SYNONYMS: map business terms to index fields
# ─────────────────────────────────────────────────────────────
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
    "cost":                 "Cost"
}

def map_business_terms_to_fields(text: str) -> str:
    """
    Replaces business-friendly terms with actual index fields.
    This should handle terms related to revenue, sales, date, etc.
    """
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

        "declining": "Revenue"
    }

    # Replace terms in text using synonyms
    def replace_term(match):
        term = match.group(0).lower()
        return FIELD_SYNONYMS.get(term, match.group(0))
    
    pattern = r'\b(' + '|'.join(re.escape(k) for k in FIELD_SYNONYMS.keys()) + r')\b'
    return re.sub(pattern, replace_term, text, flags=re.IGNORECASE)

# 1) Cognitive Search client (unchanged)
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)

# 2) Search tool: fetch all matching records (unchanged)
def azure_search_all(query: str = "", filter_exp: str = None) -> str:
    docs = []
    results = search_client.search(
        search_text=query,
        filter=filter_exp,
        top=1000
    )
    for page in results.by_page():
        docs.extend([dict(d) for d in page])
    return json.dumps(docs, default=str)

# 3) Existing aggregation tools (unchanged)
def sum_revenue_tool(input_str: str) -> str:
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        data = json.loads(azure_search_all(filter_exp=input_str))
    total = sum((item.get('Revenue') or 0) for item in data)
    return f"{total:.2f}"

def monthly_revenue_tool(input_str: str) -> str:
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        data = json.loads(azure_search_all(filter_exp=input_str))
    month_map = {}
    for item in data:
        date_str = item.get('fkdat', '')
        if not date_str:
            continue
        month = date_str[:7]
        month_map.setdefault(month, 0)
        month_map[month] += (item.get('Revenue') or 0)
    return json.dumps(month_map)

# ─────────────────────────────────────────────────────────────
# 4) NEW ANALYTIC TOOLS
# ─────────────────────────────────────────────────────────────

def profit_loss_tool(input_str: str) -> str:
    """
    Calculates total profit = sum(Revenue - Cost), plus totals of gains and losses separately.
    Accepts either JSON list or OData filter_exp.
    """
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        data = json.loads(azure_search_all(filter_exp=input_str))
    total_profit = 0.0
    gains = 0.0
    losses = 0.0
    for item in data:
        rev = item.get('Revenue') or 0
        cost = item.get('Cost') or 0
        profit = rev - cost
        total_profit += profit
        if profit >= 0:
            gains += profit
        else:
            losses += profit
    return json.dumps({
        "total_profit": round(total_profit, 2),
        "total_gains": round(gains, 2),
        "total_losses": round(losses, 2)
    })

def deterioration_rate_tool(input_str: str) -> str:
    """
    Computes percent decline in Revenue from the previous month to the current month.
    """
    # reuse monthly_revenue_tool logic
    month_map = json.loads(monthly_revenue_tool(input_str))
    if len(month_map) < 2:
        return "Not enough periods to calculate deterioration rate."
    # sort months chronologically
    months = sorted(month_map.keys())
    prev, curr = months[-2], months[-1]
    prev_val, curr_val = month_map[prev], month_map[curr]
    if prev_val == 0:
        return "Cannot compute deterioration rate (previous period is zero)."
    rate = (curr_val - prev_val) / prev_val * 100
    return f"{rate:.2f}% decline from {prev} ({prev_val:.2f}) to {curr} ({curr_val:.2f})"

def trend_tool(input_str: str) -> str:
    """
    Indicates whether Revenue is uptrending or downtrending based on last two months.
    """
    month_map = json.loads(monthly_revenue_tool(input_str))
    if len(month_map) < 2:
        return "Not enough data to determine trend."
    months = sorted(month_map.keys())
    prev_val, curr_val = month_map[months[-2]], month_map[months[-1]]
    return "uptrending 📈" if curr_val > prev_val else "downfall 📉"

def comparison_tool(input_str: str) -> str:
    """
    Compare the sum of a given field for two filter expressions.
    Input JSON must contain: {"filter1":"...", "filter2":"...", "field":"Revenue"}.
    """
    try:
        params = json.loads(input_str)
        
        # Ensure all necessary fields are present
        if not all(key in params for key in ["filter1", "filter2", "field"]):
            return "Missing required keys: 'filter1', 'filter2', and 'field' are required."
        
        f1, f2, field = params["filter1"], params["filter2"], params["field"]
    except (ValueError, KeyError):
        return (
            "Invalid input. Please supply JSON with keys "
            "`filter1`, `filter2`, and `field`."
        )
    
    # sum for first filter
    data1 = json.loads(azure_search_all(filter_exp=f1))
    sum1 = sum(item.get(field, 0) or 0 for item in data1)
    
    # sum for second filter
    data2 = json.loads(azure_search_all(filter_exp=f2))
    sum2 = sum(item.get(field, 0) or 0 for item in data2)
    
    ratio = (sum1 / sum2) if sum2 != 0 else None
    return json.dumps({
        "sum1": round(sum1, 2),
        "sum2": round(sum2, 2),
        "ratio": round(ratio, 2) if ratio is not None else None
    })


def profitability_tool(input_str: str) -> str:
    """
    Finds the single most-profitable product (by Revenue - Cost).
    """
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        data = json.loads(azure_search_all(filter_exp=input_str))
    if not data:
        return "No data available."
    # determine profit per item
    best = max(
        data,
        key=lambda itm: (itm.get("Revenue") or 0) - (itm.get("Cost") or 0)
    )
    name = best.get("arktx") or best.get("matnr") or "Unknown Product"
    profit = (best.get("Revenue") or 0) - (best.get("Cost") or 0)
    return json.dumps({
        "product": name,
        "profit": round(profit, 2),
        "details": best
    })

def sum_revenue_tool(input_str: str) -> str:
    """
    Sums the 'Revenue' field across all filtered documents.
    """
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        data = json.loads(azure_search_all(filter_exp=input_str))
    
    # Sum the 'Revenue' field
    total_revenue = sum(item.get('Revenue', 0) for item in data)
    
    return f"Total Revenue: {total_revenue:.2f}"

def generate_search_filter_and_aggregate(query: str) -> str:
    """
    Generate a filter expression and aggregation query based on user input.
    Handles operators like LIKE, IN, BETWEEN, GROUP BY, etc.
    """
    filter_expression = ""
    aggregation_expression = ""
    
    # 1. Handle Year or Date Filtering (like 2025 or a date range)
    if re.search(r'\b\d{4}\b', query):
        year_match = re.search(r'\b(\d{4})\b', query)
        if year_match:
            start_date = f"{year_match.group(1)}-01-01T00:00:00Z"
            end_date = f"{year_match.group(1)}-12-31T23:59:59Z"
            filter_expression += f"fkdat ge {start_date} and fkdat le {end_date}"
    
    # 2. Handle BETWEEN operator for numerical values (Revenue, Quantity, etc.)
    if 'between' in query.lower():
        match = re.search(r'between (\d+) and (\d+)', query, re.IGNORECASE)
        if match:
            filter_expression += f"Revenue ge {match.group(1)} and Revenue le {match.group(2)}"
    
    # 3. Handle LIKE operator for text-based filters (Product Name, Brand, etc.)
    if 'like' in query.lower():
        match = re.search(r'like "(.*?)"', query, re.IGNORECASE)
        if match:
            filter_expression += f"ProductName eq '{match.group(1)}'"
    
    # 4. Handle IN operator (multiple values, e.g., for Sales Org, Region)
    if 'in' in query.lower():
        match = re.search(r'in \((.*?)\)', query, re.IGNORECASE)
        if match:
            values = match.group(1).split(',')
            values = [f"'{value.strip()}'" for value in values]
            filter_expression += f"SalesOrg in ({','.join(values)})"
    
    # 5. Aggregations and Grouping (SUM, COUNT, GROUP BY)
    if 'group by' in query.lower():
        group_by_field = re.search(r'group by (.*?)$', query, re.IGNORECASE)
        if group_by_field:
            aggregation_expression += f"GROUP BY {group_by_field.group(1)}"
    
    # 6. Handle SUM or COUNT (aggregate functions)
    if 'sum' in query.lower():
        aggregation_expression += "SUM(Revenue)"
    
    return filter_expression, aggregation_expression


def sum_revenue_by_month(input_str: str) -> str:
    """
    Sum the 'Revenue' by month for the given filter expression.
    """
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        data = json.loads(azure_search_all(filter_exp=input_str))
    
    # Create a dictionary to hold revenue per month
    month_map = {}
    
    for item in data:
        date_str = item.get('fkdat', '')
        if not date_str:
            continue
        month = date_str[:7]  # Extract the month (e.g., '2025-01')
        month_map.setdefault(month, 0)
        month_map[month] += (item.get('Revenue') or 0)
    
    # Return the month-wise breakdown in a structured format
    return json.dumps(month_map)


# 5) Wrap all tools for LangChain
tools = [
    Tool(
        name="azure_search_all",
        func=azure_search_all,
        description="Fetch all sales records matching a query/filter; returns JSON list."
    ),
    Tool(
        name="sum_revenue",
        func=sum_revenue_tool,
        description="Sum the Revenue field over a JSON array or filter expression."
    ),
    Tool(
        name="monthly_revenue",
        func=monthly_revenue_tool,
        description="Compute Revenue per month from a JSON array or filter expression."
    ),
    Tool(
        name="profit_loss",
        func=profit_loss_tool,
        description="Compute total profit, total gains, and total losses for a set of records."
    ),
    Tool(
        name="deterioration_rate",
        func=deterioration_rate_tool,
        description="Compute % decline in Revenue from the previous period to the current."
    ),
    Tool(
        name="trend",
        func=trend_tool,
        description="Indicate whether Revenue is uptrending or downtrending based on recent data."
    ),
    Tool(
        name="comparison",
        func=comparison_tool,
        description=(
            "Compare sums of any numeric field for two different filter expressions. "
            "Input JSON: {\"filter1\":\"...\",\"filter2\":\"...\",\"field\":\"Revenue\"}."
        )
    ),
    Tool(
        name="profitability",
        func=profitability_tool,
        description="Find the single most-profitable product (by Revenue − Cost)."
    ),
]


# 6) Instantiate Azure OpenAI LLM (unchanged)
llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

system_message = """
You are the SAP Sales Analytics Agent. You are connected to an Azure Cognitive Search index 'ysales-index' populated with SAP Sales data.

**FIELD SYNONYMS**  
Use this map to translate any business‐friendly term in the user’s request into its actual index field name before constructing filters:

""" + "\n".join(f"- {k!r} → `{v}`" for k, v in FIELD_SYNONYMS.items()) + """

**Your job**:
1. **Parse the user’s request**: identify measures (Revenue, Quantity, Margin), dimensions/filters (Product Category, Sales Org, Region) and dates.
2. **Always use the synonyms map** to convert user terms → index fields (e.g. “revenue” → `Revenue`, “sales org” → `vkorg`).
3. **Build an OData‐style filter expression** (or SQL‐like WHERE clause) from the mapped prompt.
4. **Fetch** data with `azure_search_all(query, filter_exp)`.
5. **For aggregates**, call `sum_revenue` or the other analytic tools.
6. **Compute** KPIs, trends, deterioration rates, comparisons, profitability as requested.
7. **Return** raw data (JSON or CSV) **plus**:
   - 📊 **Summary** (3–5 bullets)
   - 🔍 **Key Insights** (2–4 bullets)
   - 💡 **Recommendations** (3–5 bullets)

**Error Handling**:
- If a term is ambiguous, ask a clarifying question.
- If no data matches, say “No records match—please adjust your filters or date range.”
- If user’s filters can’t be parsed, ask them to rephrase.

Always treat “NLP → SQL/OData” mapping as critical. 
"""

# 7) Build the conversational agent (include your original detailed system_message)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    system_message=system_message
)

# 8) Expose run function
def run_sales_agent(raw_prompt: str) -> str:
    """
    1. Parse the user prompt.
    2. Convert the parsed prompt into filter expressions and aggregations.
    3. Perform the aggregation and return results.
    """
    try:
        # Step 1: Map business terms to actual fields
        mapped_prompt = map_business_terms_to_fields(raw_prompt)
        
        # Step 2: Generate the search filter and aggregation logic
        filter_expression, aggregation_expression = generate_search_filter_and_aggregate(mapped_prompt)

        # Step 3: Perform the aggregation or breakdown
        if aggregation_expression:
            # Here we assume that the aggregation logic is handled based on the user's query
            result = sum_revenue_by_month(filter_expression)  # This can be extended to other aggregations
        else:
            result = "No aggregation logic applied."

        return result
    except Exception as e:
        logger.error(f"Error while running agent: {e}")
        return str(e)
