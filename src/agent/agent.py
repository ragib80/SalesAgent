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
    "dist channel":"vtweg","distribution channel":"vtweg","business area":"gsber",
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
    GK: string, Cost: real
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
    "  Or use ago(…) if relative.\n"
    "• End every statement with a semicolon.\n"
    "• Provide real line-breaks (no \\n literals).\n\n"
    "Business → column mapping:\n" + MAPPING_STR +
    "\n\nDepo/Business Area → column Value mapping:\n" + GSBER_MAPPING_STR +
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
def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    """
    Dynamically handle SAP Sales prompts, ensuring correct KQL generation,
    and map business area/territory to the correct 'gsber' code.
    """
    
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
            return f"❌ ADX error even after retry\n---KQL---\n{kql}\n\n{err}"

    # If no data found, provide feedback
    if not rows:
        return "No data found matching your criteria. Please refine your query."

    # Sample rows for summarization
    sample = [dict(zip(cols, r)) for r in rows[:20]]
    summary_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Sample (20 rows):\n{json.dumps(sample, indent=2)}\n\n"
        "Provide a concise business insight, mentioning Depots/Sales Offices clearly. "
        "Include all monetary values in BDT."
    )

    # Get the summarized result from LLM
    return llm.invoke([{"role": "user", "content": summary_prompt}]).content

# ───────────────────────── 5.  Main entry ─────────────────────────
# def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
#     """
#     Analyse a natural-language prompt and return a business summary.
#     `conversation_id` is accepted for future multi-turn support but
#     is not used in the current implementation.
#     """
#     kql = generate_kql(user_prompt)
#     for attempt in (1, 2):
#         try:
#             cols, rows = adx().run(kql)
#             break
#         except KustoApiError as err:
#             if attempt == 1:
#                 kql = generate_kql(user_prompt, strict=True)
#                 continue
#             return f"❌ ADX error even after retry\n---KQL---\n{kql}\n\n{err}"

#     if not rows:
#         return "No data found."

#     sample = [dict(zip(cols, r)) for r in rows[:20]]
#     print('sample ',sample)
#     summary_prompt = (
#         f"User asked: {user_prompt}\n\n"
#         f"Sample (20 rows):\n{json.dumps(sample, indent=2)}\n\n"
#         "Provide a concise business insight. Give the full amount.Amount is in BDT"
#     )
#     return llm.invoke([{"role": "user", "content": summary_prompt}]).content



