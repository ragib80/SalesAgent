# agent.py  –  end-to-end SAP-ADX conversational agent
import os, re, json
from functools import lru_cache
from django.conf import settings
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoApiError
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool

# ─────────────────── 1. ADX Connector ────────────────────
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

query_adx_tool = Tool(
    name="query_adx",
    func=lambda q: "\n".join(json.dumps(r) for r in adx().run(q)[1][:20]),
    description="Run a KQL query against {TABLE_NAME} and return first 20 rows.",
)

# ─────────────────── 2. Prompt assets ────────────────────
FIELD_MAPPINGS = {       # ←-- trimmed for brevity – keep your full mapping
    "revenue": "Revenue", "brand": "wgbez", "division": "spart_text",
    "customer": "cname",  "quantity": "fkimg", "date": "FKDAT_TEMP",
}
MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in FIELD_MAPPINGS.items())
# agent.py   – put this near the top
TABLE_NAME = "YSales"          # <-- your actual ADX table

KUSTO_SCHEMA = """
.create table {TABLE_NAME}  (
    Id:long, FKDAT_TEMP:datetime, Revenue:real, wgbez:string, spart_text:string,
    fkimg:real, cname:string, ...
)
"""

SYSTEM_PROMPT_BASE = (
    "You are an expert Kusto (ADX) analyst for SAP sales data.\n"
    "Translate the business request into **valid KQL only** – no markdown, comments or explanations.\n"
    "Rules:\n"
    "• Use the column names from the mapping/schema.\n"
    "• If you need a date range use:\n"
    "    let StartDate = ago(90d);\n"
    "    let EndDate   = now();\n"
    "  (two scalar let statements, each ending with a semicolon.)\n"
    "• End every statement with a semicolon.\n"
    "• Return plain text (real new-lines, no \\n literals).\n\n"
    "Business-to-column mapping:\n" + MAPPING_STR +
    "\n\nTable schema:\n" + KUSTO_SCHEMA
)

# ─────────────────── 3. LLM & agent ──────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    api_version="2025-01-01-preview",
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    temperature=0,
)
react_agent = create_react_agent(model=llm, tools=[query_adx_tool])

# ─────────────────── 4. Helpers ──────────────────────────
# def _extract_kql(raw: str) -> str:
#     """Strip ``` fences/backticks + turn \\n into real new-lines."""
#     fenced = re.search(r"```(?:kql|kusto)?\s*([\s\S]*?)```", raw, re.I)
#     raw = fenced.group(1) if fenced else raw
#     raw = raw.strip("`").strip().replace("\\n", "\n").replace("\\r", "")
#     lines = [ln for ln in raw.splitlines()
#              if ln.strip() and not ln.lstrip().startswith(("--", "#"))]
#     return "\n".join(lines)

def _extract_kql(raw: str) -> str:
    """Clean LLM KQL: strip fences, unescape \\n/\\r/\\t, normalise quotes/dashes."""
    fenced = re.search(r"```(?:kql|kusto)?\s*([\s\S]*?)```", raw, re.I)
    raw = fenced.group(1) if fenced else raw
    raw = (
        raw.strip("` …")
        .replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
        .replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
    )
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith(("--", "#"))]
    return "\n".join(lines)


def _ask_for_kql(user_request: str, strict: bool = False) -> str:
    """Single LLM call that returns cleaned KQL."""
    prompt = SYSTEM_PROMPT_BASE + (
        "\n\n>>> STRICT MODE – previous KQL had syntax errors, fix them now and return only KQL." 
        if strict else ""
    ) + f"\n\nUser request: {user_request}"
    raw = react_agent.invoke({"messages":[{"role":"user","content":prompt}]})
    if isinstance(raw, dict):
        raw = " ".join(str(v) for v in raw.values())
    return _extract_kql(raw)

# ─────────────────── 5. Public API ───────────────────────
def handle_user_query(user_prompt: str) -> str:
    # 1. Generate KQL (first attempt)
    kql = _ask_for_kql(user_prompt)

    # 2. Query ADX – try once; on syntax error, reprompt in STRICT mode
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError as e:
            if attempt == 1:
                kql = _ask_for_kql(user_prompt, strict=True)
                continue
            return f"❌ ADX syntax error even after retry.\nKQL:\n{kql}\n\n{e}"
    if not rows:
        return "No data found for that request."

    sample = [dict(zip(cols, r)) for r in rows[:20]]
    summary_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Data sample (20 rows):\n{json.dumps(sample, indent=2)}\n\n"
        "Write a concise, business-friendly insight."
    )
    return llm.invoke([{"role":"user","content":summary_prompt}]).content

# ─────────────────── 6. CLI test (optional) ──────────────
# if __name__ == "__main__":
#     os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
#     import django; django.setup()
#     print(handle_user_query("Which brands had declining sales in the last 3 months?"))
