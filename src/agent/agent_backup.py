# agent.py
import os, re, json
from functools import lru_cache
from django.conf import settings
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoApiError
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool

# ────────────────────────── 1.  Kusto helper ──────────────────────────
class ADXTool:
    def __init__(self, cluster_url: str, database: str):
        kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster_url)
        self.client = KustoClient(kcsb)
        self.database = database

    def run(self, kql: str):
        """Return (columns, rows) and raise KustoApiError if syntax is bad."""
        result = self.client.execute(self.database, kql).primary_results[0]
        cols = [c.column_name for c in result.columns]
        rows = [list(r) for r in result]
        return cols, rows

@lru_cache(maxsize=1)
def get_adx_tool() -> ADXTool:
    cluster  = getattr(settings, "ADX_CLUSTER",  os.getenv("ADX_CLUSTER"))
    database = getattr(settings, "ADX_DATABASE", os.getenv("ADX_DATABASE"))
    if not (cluster and database):
        raise RuntimeError("ADX_CLUSTER or ADX_DATABASE missing in environment / settings.py")
    return ADXTool(cluster, database)

# Tool wrapper for LangChain
query_adx_tool = Tool(
    name="query_adx",
    func=lambda q: "\n".join(json.dumps(row) for row in get_adx_tool().run(q)[1][:20]),
    description="Run a KQL query on Azure Data Explorer and return first 20 rows."
)

# ────────────────────────── 2.  Prompt scaffolding ────────────────────
FIELD_MAPPINGS = {
    "revenue":"Revenue","quantity":"fkimg","volume":"volum","customer":"cname","brand":"wgbez",
    "product name":"arktx","product":"arktx","category":"matkl","division":"spart_text",
    "company code":"bukrs","sales org":"vkorg","dist channel":"vtweg","distribution channel":"vtweg",
    "business area":"gsber","credit control area":"kkber","customer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL","product code":"matnr",
    "unit":"meins","volume unit":"voleh","business group":"GK","territory":"Territory",
    "sales zone":"Szone","date":"FKDAT_TEMP","fkdat":"FKDAT_TEMP","cost":"Cost"
}
MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in FIELD_MAPPINGS.items())

KUSTO_SCHEMA = """
.create table SAPSalesInfos (
    Id:long, CreatedTime:datetime, ModifiedTime:datetime, bukrs:string, spart:string, matkl:string,
    wgbez:string, matnr:string, vkorg:string, kunrg:string, kunnr_sh:string, Payer_DL:string,
    vbeln:string, vkbur_c:string, vkgrp_c:string, kukla:string, fkdat:datetime, posnr:string,
    arktx:string, meins:string, voleh:string, Territory:string, Szone:string, cname:string,
    spart_text:string, Revenue:real, gsber:string, fkimg:real, volum:real, ktokd:string,
    vtweg:string, erzet_T:string, kkber:string, FKDAT_TEMP:string, GK:string
)
"""

SYSTEM_PROMPT = (
    "You are an expert SAP Sales Data Analyst.\n"
    "You must output **only** a valid KQL query (no markdown) that answers the user's request.\n\n"
    "Business-to-column mappings:\n"
    f"{MAPPING_STR}\n\n"
    "Table schema:\n{KUSTO_SCHEMA}\n\n"
    "Use only these columns. Translate business terms to the correct column names."
)

# ────────────────────────── 3.  LLM & agent ───────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    api_version="2025-01-01-preview",
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    temperature=0,
)

react_agent = create_react_agent(model=llm, tools=[query_adx_tool])

# ────────────────────────── 4.  Helper to clean LLM output ────────────
def _extract_kql(raw: str) -> str:
    """Strip ``` fences/backticks & unescape \\n so ADX sees real new-lines."""
    fenced = re.search(r"```(?:kql|kusto)?\s*([\s\S]*?)```", raw, re.I)
    raw = fenced.group(1) if fenced else raw
    raw = raw.strip("`").strip()
    raw = raw.replace("\\n", "\n").replace("\\r", "")
    lines = [ln for ln in raw.splitlines()
             if ln.strip() and not ln.lstrip().startswith(("--", "#"))]
    return "\n".join(lines)


# ────────────────────────── 5.  Public entry: NLP→KQL→Data→Summary ────
def handle_user_query(user_prompt: str) -> str:
    # ---- 1) ask LLM for KQL -----------------------------------------------------------------
    sys_and_user = f"{SYSTEM_PROMPT}\n\nUser request: {user_prompt}"
    kql_raw = react_agent.invoke({"messages":[{"role":"user","content":sys_and_user}]})
    if isinstance(kql_raw, dict):
        kql_raw = " ".join(str(v) for v in kql_raw.values())
    kql = _extract_kql(kql_raw)

    # ---- 2) query ADX -----------------------------------------------------------------------
    try:
        cols, rows = get_adx_tool().run(kql)
    except KustoApiError as e:
        return f"❌ ADX returned an error.\nKQL:\n{kql}\n\n{e}"

    if not rows:
        return "No data found for that request."

    preview = [dict(zip(cols, r)) for r in rows[:20]]

    # ---- 3) narrative summary ---------------------------------------------------------------
    analysis_prompt = (
        f"User asked: {user_prompt}\n"
        f"KQL used:\n{kql}\n"
        f"Sample data (first 20 rows):\n{json.dumps(preview, indent=2)}\n\n"
        "Write a concise business-friendly summary of what this data shows."
    )
    return llm.invoke([{"role":"user","content":analysis_prompt}]).content

# # ────────────────────────── 6.  CLI test (optional) ───────────────────
# if __name__ == "__main__":
#     os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")  # adjust if project name differs
#     import django; django.setup()
#     print(handle_user_query("Show total revenue by division in 2025"))
