import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from django.conf import settings
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_community.llms.openai import AzureOpenAI
# Import Azure OpenAI
from langchain_openai import AzureOpenAI
# from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from langchain_openai import AzureChatOpenAI


# 1) Cognitive Search client
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)

# 2) Search tool: fetch all matching records (paged)
def azure_search_all(query: str = "", filter_exp: str = None) -> str:
    """
    Fetch *all* matching records from the SAP sales index.
    Input: natural-language query or OData filter string (via filter_exp).
    Output: JSON list of documents.
    """
    docs = []
    results = search_client.search(
        search_text=query,
        filter=filter_exp,
        top=1000
    )
    for page in results.by_page():
        docs.extend([dict(d) for d in page])
    return json.dumps(docs, default=str)

# 3) Aggregation tools with fallback logic
def sum_revenue_tool(input_str: str) -> str:
    """
    Sum the 'Revenue' field across all docs.
    If input_str is JSON array, parse directly; otherwise treat it as filter_exp.
    """
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        raw = azure_search_all(filter_exp=input_str)
        data = json.loads(raw)
    total = sum((item.get('Revenue') or 0) for item in data)
    return f"{total:.2f}"

def monthly_revenue_tool(input_str: str) -> str:
    """
    Group docs by YYYY-MM of 'fkdat' and sum Revenue per month.
    If input_str is JSON, use directly; otherwise treat as filter_exp.
    """
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        raw = azure_search_all(filter_exp=input_str)
        data = json.loads(raw)
    month_map = {}
    for item in data:
        date_str = item.get('fkdat', '')
        if not date_str:
            continue
        month = date_str[:7]
        month_map.setdefault(month, 0)
        month_map[month] += (item.get('Revenue') or 0)
    return json.dumps(month_map)

# 4) Wrap tools for LangChain
tools = [
    Tool(
        name='azure_search_all',
        func=azure_search_all,
        description=(
            'Fetch all sales records matching a query/filter from the SAP index; returns JSON list.'
        ),
    ),
    Tool(
        name='sum_revenue',
        func=sum_revenue_tool,
        description='Sum the Revenue field over a JSON array or filter expression.',
    ),
    Tool(
        name='monthly_revenue',
        func=monthly_revenue_tool,
        description='Compute Revenue per month from a JSON array or filter expression.',
    ),
]

# 5) Instantiate Azure OpenAI LLM

llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,  # or your deployment
    api_version="2025-01-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# 5) Build the conversational agent with a detailed system prompt


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    system_message="""
You are the SAP Sales Analytics Agent. You are connected to an Azure Cognitive Search index 'ysales-index' populated with SAP Sales data. Your job is to:
1. Take a user’s natural-language request.
2. Map business-friendly terms (or raw SAP field names) to the correct index fields.
Parse the user’s request: identify measures (Revenue, Quantity, Margin), dimensions/filters (Product Category, Sales Org, Region), and date range (explicit or default).
 If prompt specifies dates use them; else default start=2024-01-01 to end=today or cap year-end.. If prompt specifies dates use them; else default start=2024-01-01 to end=today or cap year-end.
 Generate appropriate filter and call `azure_search_all(query, filter)` to retrieve full dataset.

4. Aggregate, analyze, and interpret the results—including trending/declining stages, profit/loss, margins, and deterioration rates.
 4. For aggregate requests, call `sum_revenue`; for month-by-month requests, call `monthly_revenue`.
 5. Compute KPIs: Gross Profit, Profit Margin %, up/down trends, deterioration rates.
5. Return both raw data (JSON, CSV tables, or charts) AND a concise business summary with actionable recommendations.

Data Schema & Synonyms (users may refer to either):
Revenue→Revenue, Quantity→fkimg, Volume→volum, Customer→cname, Brand→wgbez,
Product Name→arktx, Category→matkl, Division→spart_text, Company Code→bukrs,
Sales Org→vkorg, Distribution Channel→vtweg, Business Area→gsber,
Credit Control Area→kkber, Customer Group→kukla, Account Group→ktokd,
Sales Group→vkgrp_c, Sales Office→vkbur_c, Payer ID→Payer_DL,
Product Code→matnr, Unit→meins, Volume Unit→voleh, Business Group→GK,
Territory→Territory, Sales Zone→Szone, Date→fkdat (e.g. '2024-07-16 00:00:00.000').

Processing Flow:
- **Intent & Entity Extraction**: pick out measures (Revenue, Quantity, Margin), dimensions/filters (Product Category, Sales Org, Region), and time windows (explicit or relative).
- **Date Range Logic**:
    • If the prompt specifies “from X to Y,” extract those dates and filter `fkdat` accordingly.
    • Otherwise default to start=01-01-2024 and end=today.
    • If the user says “2025,” interpret as 2025-01-01 through 2025-12-31, but if 2025 is still in progress, use today’s date as the end.
- **SQL-Style Query Generation**: build SELECT, GROUP BY, and WHERE clauses,Aggregate functions.
- **Execute** on the `ysales-index` via your `azure_search` tool.
- **Compute KPIs**:
    • Gross Profit = Revenue − Cost (when available).
    • Profit Margin % = Gross Profit ÷ Revenue × 100.
    • Identify up-trending vs. down-trending lines, margin erosion, exceptions.
- **Summarize**:
    • Top performers & laggards.
    • Trend analysis (MoM, YoY).
    • Alerts for negative margins or high-risk discounts.
- **Recommendations**:
    • Pricing strategy.
    • Promotional focus.
    • Inventory optimization.
    • Credit risk management.

Output Requirements:
Always include:
- **Raw data** (table or CSV download link; charts if asked).
- **📊 Summary** (3–5 bullets of high-level findings).
- **🔍 Key Insights** (2–4 concise bullets).
- **💡 Recommendations** (3–5 actionable items).

Error Handling:
- If a term is ambiguous, ask: “Did you mean Territory or Sales Zone?”
- If no data matches, say: “No records match—please adjust your filters or date range.”
- If an unknown field is requested, suggest the closest match.
If ambiguous, ask clarifying questions. Always handle SAP Sales–specific terms as per schema or Data Schema & Synonyms.
Ensure you handle **all** SAP Sales–related queries as above.
"""
)


# 6) Expose run function
def run_sales_agent(prompt: str) -> str:
    """
    Feed the user prompt to our RAG agent and return its answer.
    """
    return agent.run(prompt)