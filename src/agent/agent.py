import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from django.conf import settings
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_community.llms.openai import AzureOpenAI
# Import Azure OpenAI
from langchain_openai import AzureOpenAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from langchain_openai import AzureChatOpenAI


# 1) Cognitive Search client
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)

# 2) Define search tool
def azure_search_tool(query: str) -> str:
    """
    Query Azure Cognitive Search and return top 5 matching sales records.
    """
    results = search_client.search(search_text=query, top=5)
    docs = [dict(doc) for doc in results]
    # Simplify to key fields
    # simple_docs = [
    #     {
    #         'product': d.get('arktx'),
    #         'customer': d.get('cname'),
    #         'revenue': d.get('Revenue'),
    #         'date': d.get('fkdat')
    #     }
    #     for d in docs
    # ]
    simple_docs = [
    {
        'revenue':                d.get('Revenue'),
        'quantity':               d.get('fkimg'),
        'volume':                 d.get('volum'),
        'customer':               d.get('cname'),
        'brand':                  d.get('wgbez'),
        'product':                d.get('arktx'),
        'category':               d.get('matkl'),
        'division':               d.get('spart_text'),
        'company_code':           d.get('bukrs'),
        'sales_org':              d.get('vkorg'),
        'dist_channel':           d.get('vtweg'),
        'business_area':          d.get('gsber'),
        'credit_control_area':    d.get('kkber'),
        'customer_group':         d.get('kukla'),
        'account_group':          d.get('ktokd'),
        'sales_group':            d.get('vkgrp_c'),
        'sales_office':           d.get('vkbur_c'),
        'payer_id':               d.get('Payer_DL'),
        'product_code':           d.get('matnr'),
        'unit':                   d.get('meins'),
        'volume_unit':            d.get('voleh'),
        'business_group':         d.get('GK'),
        'territory':              d.get('Territory'),
        'zone':                   d.get('Szone'),
        'date':                   d.get('fkdat'),
    }
    for d in docs
]

    return json.dumps(simple_docs, default=str)

# 3) Instantiate LLM with correct AzureOpenAI args
# 
# Make sure you set these environment variables or in .env and settings:
# OPENAI_API_TYPE=azure
# OPENAI_API_BASE=<your-Azure-OpenAI-endpoint>
# OPENAI_API_VERSION=2023-05-15
# OPENAI_API_KEY=<your-Azure-OpenAI-key>
# AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>

llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,  # or your deployment
    api_version="2025-01-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# llm = AzureAIChatCompletionsModel(
#     model_name=settings.AZURE_OPENAI_DEPLOYMENT,
#     api_version="2025-01-01-preview",
# )
# 4) Tool wrapper
tools = [
    Tool(
        name="azure_search",
        func=azure_search_tool,
        description=(
            "Use this tool to search SAP sales data with a natural-language filter. "
            "Returns JSON of up to 5 records (product, customer, revenue, date)."
        )
    )
]

# 5) Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 6) Expose run function
def run_sales_agent(user_prompt: str) -> str:
    """
    Runs the RAG agent: it will call azure_search as needed and generate an answer.
    """
    return agent.run(user_prompt)