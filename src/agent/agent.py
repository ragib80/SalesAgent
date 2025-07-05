# agent.py ─ Simplified SAP Sales bot for Azure ADX (YSales)
import os, re, json
from functools import lru_cache

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
    "revenue":"Revenue","quantity":"fkimg","volume":"volum","customer":"cname",
    "brand":"wgbez","product name":"arktx","product":"arktx","category":"matkl",
    "division":"spart_text","company code":"bukrs","sales org":"vkorg",
    "dist channel":"vtweg","distribution channel":"vtweg","business area":"gsber",
    "credit control area":"kkber","customer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL",
    "product code":"matnr","unit":"meins","volume unit":"voleh","business group":"GK",
    "territory":"Territory","sales zone":"Szone","date":"FKDAT_TEMP",
    "fkdat":"FKDAT_TEMP","cost":"Cost"
}
MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in FIELD_MAPPINGS.items())

KUSTO_SCHEMA = """
.create table YSales (
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

SYSTEM_PROMPT_KQL = (
    "You are an expert Kusto (ADX) analyst for SAP sales data.\n"
    "Output **only raw KQL**, no markdown or commentary.\n"
    "Rules:\n"
    "• Use the table YSales and columns below.\n"
    "• If a date range is required, declare:\n"
    "      let StartDate = datetime(YYYY-MM-DD);\n"
    "      let EndDate   = datetime(YYYY-MM-DD);\n"
    "  Or use ago(…) if relative.\n"
    "• End every statement with a semicolon.\n"
    "• Provide real line-breaks (no \\n literals).\n\n"
    "Business → column mapping:\n" + MAPPING_STR +
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
    response = llm.invoke([{"role":"user","content":prompt}]).content
    return _extract_kql(response)



# ───────────────────────── 5.  Main entry ─────────────────────────
# ───────────────────────── 5.  Main entry ─────────────────────────
def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    """
    Analyse a natural-language prompt and return a business summary.
    `conversation_id` is accepted for future multi-turn support but
    is not used in the current implementation.
    """
    kql = generate_kql(user_prompt)
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError as err:
            if attempt == 1:
                kql = generate_kql(user_prompt, strict=True)
                continue
            return f"❌ ADX error even after retry\n---KQL---\n{kql}\n\n{err}"

    if not rows:
        return "No data found."

    sample = [dict(zip(cols, r)) for r in rows[:20]]
    summary_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Sample (20 rows):\n{json.dumps(sample, indent=2)}\n\n"
        "Provide a concise business insight. Give the full amount.Amount is in BDT"
    )
    return llm.invoke([{"role": "user", "content": summary_prompt}]).content



# def handle_user_query(user_prompt: str) -> str:
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
#     summary_prompt = (
#         f"User asked: {user_prompt}\n\n"
#         f"Sample (20 rows):\n{json.dumps(sample, indent=2)}\n\n"
#         "Provide a concise business insight.Give the full amount."
#     )
#     return llm.invoke([{"role":"user","content":summary_prompt}]).content


