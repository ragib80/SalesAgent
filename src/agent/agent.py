import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from django.conf import settings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import AzureChatOpenAI
import re
import logging
import openai
import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

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




# OpenAI API Key and Model Configuration
# openai.api_key = settings.AZURE_OPENAI_KEY

# Cognitive Search client
search_client = SearchClient(
    endpoint=settings.AZURE_SEARCH_ENDPOINT,
    index_name=settings.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
)



# openai_client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     api_version="2025-01-01-preview"
# )

# openai_client = OpenAIClient(
#     endpoint   = AZURE_OPENAI_ENDPOINT,
#     credential = AzureKeyCredential(AZURE_OPENAI_API_KEY)
# )
openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2025-01-01-preview"
        )
# Example mapping of user-friendly terms to SAP sales data fields
keyword_to_field_map = {
    "revenue": "Revenue",
    "sales": "Revenue",
    "profit": "Profit",
    "loss": "Loss",
    "deterioration_rate": "DeteriorationRate",
    "deterioration rate": "Deterioration Rate",
    "trend": "Trend",
    "comparison": "Comparison",
    "profitability": "Profitability",
    "product": "arktx",
    "quantity": "fkimg",
    "volume": "volum",
    "month": "fkdat",  # Assuming 'fkdat' is the field for the date
    "year": "fkdat",
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
    # Add more mappings as needed
}

def run_sales_agent(raw_prompt: str) -> str:
    """
    1. Parse the user prompt.
    2. Convert the parsed prompt into filter expressions and aggregations.
    3. Perform the aggregation and return results.
    """
    try:
        # Step 1: Use OpenAI's text-embedding-ada-002 to interpret the user's query semantically
        prompt_embedding = get_prompt_embedding(raw_prompt)
        
        # Step 2: Parse the prompt and get semantic understanding
        filter_expression = parse_prompt_with_embedding(raw_prompt, prompt_embedding)
        
        # Step 3: Generate the aggregation logic based on the parsed prompt
        aggregation_expression = generate_aggregation_expression(raw_prompt)

        # Step 4: Execute the query or aggregation
        if aggregation_expression:
            result = perform_aggregation(filter_expression, aggregation_expression, raw_prompt)
        else:
            result = "No aggregation logic applied."

        return result
    except Exception as e:
        print(f"Error while running agent: {e}")
        return str(e)

def get_prompt_embedding(prompt: str) -> list:
    """
    Generate the semantic embedding for the user prompt using OpenAI's text-embedding-ada-002.
    """
    try:
        # Use the Azure OpenAI client to generate embeddings
        # response = openai_client.completions.create(
        #     model="text-embedding-ada-002",  # Use the appropriate model for embeddings
        #     prompt=prompt
        # )
        res = openai_client.embeddings.create(
            model="text-embedding-ada-002",  # Use the appropriate model for embeddings
            input=[prompt])
        
        raw_embedding = res.data[0].embedding
        print(" print(raw_embedding) ",raw_embedding)
        
        # res = openai_client.embeddings.create(model="text-embedding-ada-002", input=[prompt])
        # raw = res.data[0].embedding
        
        return raw_embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def parse_prompt_with_embedding(raw_prompt: str, prompt_embedding: list) -> dict:
    """
    Use semantic understanding to generate filter expressions based on the user prompt.
    """
    filters = {}

    # Utilize the embedding to enhance keyword matching (this could be extended to use the embedding for more advanced logic)
    for keyword, field in keyword_to_field_map.items():
        if keyword.lower() in raw_prompt.lower():
            filters[field] = True  # Flag the field as present in the query

    # Here you can integrate more advanced logic to dynamically adjust filters based on the semantic meaning of the query using `prompt_embedding`
    # For example, use the embedding to map more nuanced terms into specific fields
    return filters

def generate_aggregation_expression(raw_prompt: str) -> str:
    """
    Generate the aggregation expression based on the raw prompt.
    """
    if re.search(r'\b(revenue|sales)\b', raw_prompt, re.IGNORECASE):
        if re.search(r'\bmonth\b', raw_prompt, re.IGNORECASE):
            return "SUM(Revenue) by month"
        else:
            return "SUM(Revenue)"
    elif re.search(r'\bprofit|loss\b', raw_prompt, re.IGNORECASE):
        return "SUM(Profit), SUM(Loss)"
    elif re.search(r'\bdeterioration_rate\b', raw_prompt, re.IGNORECASE):
        return "Deterioration Rate"
    elif re.search(r'\btrend\b', raw_prompt, re.IGNORECASE):
        return "Trend"
    elif re.search(r'\bcomparison\b', raw_prompt, re.IGNORECASE):
        return "Comparison"
    elif re.search(r'\bprofitability\b', raw_prompt, re.IGNORECASE):
        return "Most Profitable Product"
    return ""

def perform_aggregation(filter_expression, aggregation_expression, raw_prompt) -> str:
    """
    Perform the aggregation based on the filter expression and the generated aggregation expression.
    """
    if aggregation_expression == "SUM(Revenue)":
        return sum_revenue(filter_expression)
    elif aggregation_expression == "SUM(Revenue) by month":
        return sum_revenue_by_month(filter_expression)
    elif aggregation_expression == "SUM(Profit), SUM(Loss)":
        return calculate_profit_loss(filter_expression)
    elif aggregation_expression == "Deterioration Rate":
        return calculate_deterioration_rate(filter_expression)
    elif aggregation_expression == "Trend":
        return detect_trend(filter_expression)
    elif aggregation_expression == "Comparison":
        return compare_fields(filter_expression, raw_prompt)
    elif aggregation_expression == "Most Profitable Product":
        return find_most_profitable_product(filter_expression)
    else:
        return "No aggregation logic found."

# Tool Implementations with Azure Search Integration

def sum_revenue(filter_expression) -> str:
    """
    Sums the revenue based on the filter expression from Azure AI Search.
    """
    search_results = search_client.search(search_text="*", filter="Revenue ge 0")  # Example for all records with revenue greater than or equal to 0
    revenue_data = [doc["Revenue"] for doc in search_results]
    total_revenue = sum(revenue_data)
    return f"Total Revenue: {total_revenue}"

def sum_revenue_by_month(filter_expression) -> str:
    """
    Sums the revenue by month based on the filter expression from Azure AI Search.
    """
    search_results = search_client.search(search_text="*", filter="fkdat ge '2025-01-01' and fkdat le '2025-12-31'")  # Example for year 2025
    monthly_revenue = {}
    for doc in search_results:
        month = doc["fkdat"].split('-')[1]  # Assuming the date format is 'YYYY-MM-DD'
        monthly_revenue[month] = monthly_revenue.get(month, 0) + doc["Revenue"]
    return f"Monthly Revenue: {monthly_revenue}"

def calculate_profit_loss(filter_expression) -> str:
    """
    Calculate profit and loss based on SAP sales data from Azure AI Search.
    """
    search_results = search_client.search(search_text="*", filter="Revenue ge 0 and Cost ge 0")  # Example filter
    profit_data = [doc["Revenue"] - doc["Cost"] for doc in search_results]
    total_profit = sum([value for value in profit_data if value > 0])
    total_loss = sum([value for value in profit_data if value < 0])
    return f"Total Profit: {total_profit}, Total Loss: {total_loss}"

def calculate_deterioration_rate(filter_expression) -> str:
    """
    Calculate the deterioration rate (percent decline from prior period) from Azure AI Search data.
    """
    # Example: comparing current period to prior period
    current_period_value = 10000  # Example current period value from Azure Search
    prior_period_value = 12000  # Example prior period value from Azure Search
    deterioration_rate = ((prior_period_value - current_period_value) / prior_period_value) * 100
    return f"Deterioration Rate: {deterioration_rate:.2f}%"

def detect_trend(filter_expression) -> str:
    """
    Detect the trend (uptrending or downfall) based on the comparison of periods.
    """
    current_period_value = 15000  # Example current period value from Azure Search
    prior_period_value = 12000  # Example prior period value from Azure Search
    if current_period_value > prior_period_value:
        return "Uptrending"
    else:
        return "Downfall"

def compare_fields(filter_expression, raw_prompt) -> str:
    """
    Compare values between two fields based on the user's request.
    """
    # Example: comparing 'Revenue' for different months or years
    match = re.search(r'\b(Revenue|profit|loss)\b.*\b(Revenue|profit|loss)\b', raw_prompt, re.IGNORECASE)
    if match:
        # Perform comparison logic based on the matched fields
        return "Comparison Result: Revenue in 2025 vs 2024"
    return "Unable to compare the fields."

def find_most_profitable_product(filter_expression) -> str:
    """
    Identify the most profitable product based on revenue from Azure AI Search.
    """
    search_results = search_client.search(search_text="*", filter="Revenue ge 0")
    products = {}
    for doc in search_results:
        product_name = doc["arktx"]
        revenue = doc["Revenue"]
        products[product_name] = products.get(product_name, 0) + revenue
    most_profitable = max(products, key=products.get)
    return f"The most profitable product is {most_profitable} with a revenue of {products[most_profitable]}."
