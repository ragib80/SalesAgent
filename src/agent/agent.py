# agent.py ─ Simplified SAP Sales bot for Azure ADX (SAPSalesInfos)
import os, re, json
from functools import lru_cache
import datetime
from django.conf import settings
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoApiError

from langchain_openai import AzureChatOpenAI

# ───────────────────────── 1.  ADX helper ──────────────────────────
class ADXTool:
    def __init__(self, cluster: str, database: str):
        kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)
        self.client = KustoClient(kcsb)
        self.database = database
    def run(self, kql: str):
        tbl = self.client.execute(self.database, kql).primary_results[0]
        cols = [c.column_name for c in tbl.columns]
        rows = [list(r) for r in tbl]
        return cols, rows

@lru_cache(maxsize=1)
def adx() -> ADXTool:
    return ADXTool(
        getattr(settings, "ADX_CLUSTER",  os.getenv("ADX_CLUSTER")),
        getattr(settings, "ADX_DATABASE", os.getenv("ADX_DATABASE")),
    )

# ───────────────────────── 2.  Prompt assets ───────────────────────
TABLE_NAME = "SAPSalesInfos"

FIELD_MAPPINGS = {
    "revenue":"Revenue","quantity":"fkimg","volume":"volum","Dealer":"cname",
    "brand":"wgbez","product name":"arktx","product":"arktx","category":"matkl",
    "division":"spart_text","company code":"bukrs","sales org":"vkorg",
    "dist channel":"vtweg","distribution channel":"vtweg","business area":"gsber","depo":"gsber",
    "credit control area":"kkber","Dealer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL",
    "product code":"matnr","unit":"meins","volume unit":"voleh","business group":"GK",
    "territory":"Territory","sales zone":"Szone","date":"fkdat",
    "fkdat":"fkdat"
}
MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in FIELD_MAPPINGS.items())

KUSTO_SCHEMA = """
.create table SAPSalesInfos (
    Id: long, CreatedTime: datetime, ModifiedTime: datetime, bukrs: string,
    spart: string, matkl: string, wgbez: string, matnr: string, vkorg: string,
    kunrg: string, kunnr_sh: string, Payer_DL: string, vbeln: string, vkbur_c: string,
    vkgrp_c: string, kukla: string, fkdat: datetime, posnr: string, arktx: string,
    meins: string, voleh: string, Territory: string, Szone: string, cname: string,
    spart_text: string, Revenue: real, gsber: string, fkimg: real, volum: real,
    ktokd: string, vtweg: string, erzet_T: string, kkber: string, FKDAT_TEMP: string,
    GK: string
)
"""

GSBER_MAPPING = {
    "Dhaka Factory": "1000",
    "Chittagong Factory": "1100",
    "Mirsarai Factory": "1200",
    "Dhaka Sales": "4000",
    "Chittagong Sales": "4010",
    "Sylhet Sales": "4020",
    "Comilla Sales": "4030",
    "Rajshahi Sales": "4040",
    "Bogra Sales": "4050",
    "Khulna Sales": "4060",
    "Mymensing Sales": "4070",
    "Barishal Sales": "4080",
    "Rangpur Sales": "4090",
    "Feni Sales": "4100",
    "Dhaka South": "4110",  # Mapping "Dhaka South" to gsber == '4110'
    "Brahmanbaria Sales": "4120",
    "Dhaka North": "4130",
    "Test Business Area": "4500",
    "PPHD": "5000",
    "Berger Design Studio": "5010",
    "Berger Training Institute": "5020",
    "Berger Tech Consulting Ltd": "5100",
    "Jenson & Nicholson BD Ltd": "6000",
    "JNBL 2nd Unit Dhaka": "6100",
    "Berger Becker Bangladesh": "7000",
    "Berger Fosroc Limited": "8000",
    "Corporate": "9000"
}
GSBER_MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in GSBER_MAPPING.items())

SYSTEM_PROMPT_KQL = (
    "You are an expert Kusto (ADX) analyst for SAP sales data.\n"
    "Output **only raw KQL**, no markdown or commentary.\n"
    "Rules:\n"
    "• Use the table SAPSalesInfos and columns below.\n"
    "• If a date range is required, declare:\n"
    "      let StartDate = datetime(YYYY-MM-DD);\n"
    "      let EndDate   = datetime(YYYY-MM-DD);\n"
 
    "• End every statement with a semicolon.\n"
    "• Provide real line-breaks (no \\n literals).\n\n"
    "Business → column mapping:\n" + MAPPING_STR +
    "\n\nDepo/Business Area (gsber) → column Value mapping:\n" + GSBER_MAPPING_STR +
    "\n\nTable schema:\n" + KUSTO_SCHEMA
)

# ───────────────────────── 3.  LLM instance ────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0,
)

# ───────────────────────── 4.  Helpers ────────────────────────────
def _extract_kql(raw: str) -> str:
    """Remove ``` fences/backticks and unescape \\n / \\r / \\t."""
    fenced = re.search(r"```(?:kql|kusto)?\s*([\s\S]*?)```", raw, re.I)
    raw = fenced.group(1) if fenced else raw
    raw = raw.strip("`").replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
    return raw.replace("SAPSalesInfos", TABLE_NAME).strip()

def generate_kql(user_req: str, strict=False) -> str:
    prompt = SYSTEM_PROMPT_KQL
    if strict:
        prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."
    prompt += f"\n\nUser request: {user_req}"
    print("_extract_kql-------------",prompt)
    response = llm.invoke([{"role":"user","content":prompt}]).content

    return _extract_kql(response)



# ───────────────────────── 5.  Main entry ─────────────────────────


def format_dates(kql_query: str) -> str:
    """Ensure all date-like strings are properly formatted as datetime literals."""
    return re.sub(r'(\d{4}-\d{2}-\d{2})', r'datetime(\1)', kql_query)

# Detect trend from user prompt (e.g., increasing, declining, etc.)
def detect_trend(user_prompt: str) -> str:
    if any(word in user_prompt.lower() for word in ["declining", "downtrending", "negative growth", "falling", "decrease"]):
        return "declining"
    elif any(word in user_prompt.lower() for word in ["increasing", "uptrending", "positive growth", "rising", "growth"]):
        return "increasing"
    else:
        return "stable"

# Handle user queries dynamically and generate the corresponding KQL query



import datetime
import re

def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    """
    Dynamically handle SAP Sales prompts, ensuring correct KQL generation,
    and map business area/territory to the correct 'gsber' code.
    """
    # Check for "last n months" pattern in the user prompt
    last_n_months_match = re.search(r'last\s+(\d+)\s+month[s]?', user_prompt, re.IGNORECASE)
    
    # If "last n months" is detected in the user prompt
    if last_n_months_match:
        n_months = int(last_n_months_match.group(1))
        
        # Calculate the current date (today)
        end_date = datetime.datetime.now()
        
        # Calculate the start date as n months ago from today
        start_date = end_date - datetime.timedelta(days=n_months * 30)  # Approximate 30 days per month
        
        # Format the start and end dates as datetime strings for KQL
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Update the user prompt with the dynamic date range
        user_prompt += f" from {start_date_str} to {end_date_str}"
    
    # If no date range or "last n months" is not found, ask the user for one
    else:
        user_prompt += " Please specify a date range for the data (e.g., from 2025-01-01 to 2025-12-31)."
    
    # Generate raw KQL from the user prompt using LLM
    kql = generate_kql(user_prompt)

    # Print the generated query for debugging
    print(f"Generated KQL Query: {kql}")

    # Format the dates dynamically
    kql = format_dates(kql)

    # Handle known issues like '3mo' to '90d' for date ranges
    kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)

    # Fix unsupported functions like `startofquarter`, replacing with `startofmonth`
    kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)

    # Dynamically map the business area/territory name to the corresponding gsber code
    for territory, gsber_value in GSBER_MAPPING.items():
        if territory.lower() in user_prompt.lower():  # If user mentions a territory/business area
            # Replace the filter on Territory with gsber for the matching business area
            kql = re.sub(r"where Territory == .+?", f"where gsber == '{gsber_value}'", kql)
            break  # Once mapped, no need to continue

    # Detect trend direction (increase or decline) dynamically from the user's prompt
    trend = detect_trend(user_prompt)

    if trend == "declining":
        kql = kql.replace("RevenueChange < 0", "RevenueChange < 0")  # Declining trend
    elif trend == "increasing":
        kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")  # Increasing trend
    else:
        # For stable or other trends, you can just leave it as it is or do any specific handling
        kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")  # Stable trend (no change)

    # Execute the query and handle retries
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError as err:
            if attempt == 1:
                kql = generate_kql(user_prompt, strict=True)
                continue
            # Log the error and return user-friendly feedback.
            print(f"Error: {err}")
            return "Please refine your query for better results. I’m learning day by day and will help you improve your query."

    # If no data found, provide feedback
    if not rows:
        return "No data found matching your criteria. Please refine your query for more specific results."

    # Prepare the data for LLM to process
    result_data = [dict(zip(cols, r)) for r in rows[:20]]  # Get top 5 rows or adjust as needed
    result_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Sample Data:\n{json.dumps(result_data, indent=2)}\n\n"
        "Based on the query results, format the output in bulleted format. "
        "If the result is numerical or comparative, bullet points for proper indication. If it's categorical or simple, use bullet points. "
        "After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Amount is in BDT."
        "If Needed,Based on the Sample  context data  give meaningful business-related suggestions such as increasing sales, revenue."
    )

    # Let LLM decide on how to format the result: tabular or bulleted
    formatted_result = llm.invoke([{"role": "user", "content": result_prompt}]).content

    return formatted_result




# def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
#     """
#     Dynamically handle SAP Sales prompts, ensuring correct KQL generation,
#     and map business area/territory to the correct 'gsber' code.
#     """
#     # Check if the user's prompt contains a valid date or range
#     date_pattern = r'(\d{4}-\d{2}-\d{2})|(\d{4})|(from\s+\w+\s+\d{4})|(to\s+\w+\s+\d{4})|(\bago\b\s*\(\d+[a-zA-Z]*\))'
#     date_matches = re.findall(date_pattern, user_prompt)

#     # If no date range or date references found, ask the user for one
#     if not date_matches:
#         user_prompt += " Please specify a date range for the data (e.g., from 2025-01-01 to 2025-12-31)."

#     # Generate raw KQL from the user prompt using LLM
#     kql = generate_kql(user_prompt)

#     # Print the generated query for debugging
#     print(f"Generated KQL Query: {kql}")

#     # Format the dates dynamically
#     kql = format_dates(kql)

#     # Handle known issues like '3mo' to '90d' for date ranges
#     kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)

#     # Fix unsupported functions like `startofquarter`, replacing with `startofmonth`
#     kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)

#     # Dynamically map the business area/territory name to the corresponding gsber code
#     for territory, gsber_value in GSBER_MAPPING.items():
#         if territory.lower() in user_prompt.lower():  # If user mentions a territory/business area
#             # Replace the filter on Territory with gsber for the matching business area
#             kql = re.sub(r"where Territory == .+?", f"where gsber == '{gsber_value}'", kql)
#             break  # Once mapped, no need to continue

#     # Detect trend direction (increase or decline) dynamically from the user's prompt
#     trend = detect_trend(user_prompt)

#     if trend == "declining":
#         kql = kql.replace("RevenueChange < 0", "RevenueChange < 0")  # Declining trend
#     elif trend == "increasing":
#         kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")  # Increasing trend
#     else:
#         # For stable or other trends, you can just leave it as it is or do any specific handling
#         kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")  # Stable trend (no change)

#     # Execute the query and handle retries
#     for attempt in (1, 2):
#         try:
#             cols, rows = adx().run(kql)
#             break
#         except KustoApiError as err:
#             if attempt == 1:
#                 kql = generate_kql(user_prompt, strict=True)
#                 continue
#             # Log the error and return user-friendly feedback.
#             print(f"Error: {err}")
#             return "Please refine your query for better results. I’m learning day by day and will help you improve your query."

#     # If no data found, provide feedback
#     if not rows:
#         return "No data found matching your criteria. Please refine your query for more specific results."

#     # Prepare the data for LLM to process
#     result_data = [dict(zip(cols, r)) for r in rows[:20]]  # Get top 5 rows or adjust as needed
#     result_prompt = (
#         f"User asked: {user_prompt}\n\n"
#         f"Sample Data:\n{json.dumps(result_data, indent=2)}\n\n"
#         "Based on the query results, format the output in bulleted format. "
#         "If the result is numerical or comparative, bullet points for proper indication . If it's categorical or simple, use bullet points. "
#         "After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Give the full amount.Amount is in BDT"
#         "If possible give businness related suggestion such as incresing sales, revinue  related to the data. "
#     )

#     # Let LLM decide on how to format the result: tabular or bulleted
#     formatted_result = llm.invoke([{"role": "user", "content": result_prompt}]).content

#     return formatted_result
