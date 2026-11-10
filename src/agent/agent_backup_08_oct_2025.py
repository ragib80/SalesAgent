# agent.py
from __future__ import annotations
import os, re, json
from functools import lru_cache
import datetime
from django.conf import settings
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoApiError
from typing import Optional
from django.db.models import Q
from langchain_openai import AzureChatOpenAI
from openai import BadRequestError 
import logging
import dateutil.parser
import calendar
from user_auth.models import UserDepoMap, UserZoneMap, UserTerritoryMap
from typing import List, Dict
from dataclasses import dataclass
from agent.utils.conversation_history import fetch_history,pack_history_by_chars,build_history_prompt_block,_build_carryover_block,build_applied_context_block,get_last_n_history, build_context_decision_rules,get_latest_meta,save_meta,get_latest_message_id
import logging
from agent.utils.conversation_helpers import (
    get_conversation_id_from_uuid,
    get_last_20_messages,
    get_last_20_message_metas,
    serialize_context_for_llm,
    build_conversation_snapshot_block,
    build_context_memory_contract,

)
logger = logging.getLogger(__name__)


# ───────────────────────── 1.  ADX helper ──────────────────────────
# class ADXTool:
#     def __init__(self, cluster: str, database: str):
#         kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)
#         self.client = KustoClient(kcsb)
#         self.database = database
#     def run(self, kql: str):
#         tbl = self.client.execute(self.database, kql).primary_results[0]
#         cols = [c.column_name for c in tbl.columns]
#         rows = [list(r) for r in tbl]
#         return cols, rows

# @lru_cache(maxsize=1)
# def adx() -> ADXTool:
#     return ADXTool(
#         getattr(settings, "ADX_CLUSTER",  os.getenv("ADX_CLUSTER")),
#         getattr(settings, "ADX_DATABASE", os.getenv("ADX_DATABASE")),
#     )

class ADXTool:
    def __init__(self, cluster: str, database: str):
        # use the cached az CLI token instead of device code
        kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
        self.client = KustoClient(kcsb)
        self.database = database

    def run(self, kql: str):
        response = self.client.execute(self.database, kql)
        table = response.primary_results[0]
        cols = [c.column_name for c in table.columns]
        rows = [list(r) for r in table]
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
    "revenue":"Revenue","sale":"Revenue","quantity":"fkimg","volume":"volum","Dealer":"cname",
    "brand":"wgbez","product name":"arktx","product":"arktx","category":"matkl","Material Group":"matkl",
    "division":"spart_text","division code":"spart","company code":"bukrs","sales org":"vkorg",
    "dist channel":"vtweg","distribution channel":"vtweg","business area":"gsber","depo":"gsber",
    "credit control area":"kkber","Dealer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL",
    "product code":"matnr","material code":"matnr","material code":"meins","volume unit":"voleh","business group":"GK",
    "territory":"Territory","sales zone":"Szone","date":"fkdat","Dealer Code":"kunrg","dealer code":"kunrg",
    "fkdat":"fkdat","invoice number":"vbeln", "sales org":"vkorg","sales organization":"vkorg","credit control area":"kkber"
}
MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in FIELD_MAPPINGS.items())

KUSTO_SCHEMA = """
.create table SAPSalesInfos (
    Id: long, CreatedTime: datetime, ModifiedTime: datetime, bukrs: long,
    spart: long, matkl: string, wgbez: string, matnr: string, vkorg: long,
    kunrg: long, kunnr_sh: long, Payer_DL: long, vbeln: long, vkbur_c: long,
    vkgrp_c: string, kukla: long, fkdat: datetime, posnr: long, arktx: string,
    meins: string, voleh: string, Territory: string, Szone: string, cname: string,
    spart_text: string, Revenue: real, gsber: long, fkimg: long, volum: real,
    ktokd: string, vtweg: string, erzet_T: string, kkber: long, FKDAT_TEMP: string,
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

# SYSTEM_PROMPT_KQL = (
#     "You are an expert Kusto (ADX) analyst for SAP sales data.\n"
#     "Output **only raw KQL**, no markdown or commentary.\n"
#     "Rules:\n"
#     "• Use the table SAPSalesInfos and columns below.\n"
#     "• If a date range is required, declare:\n"
#     "      let StartDate = datetime(YYYY-MM-DD);\n"
#     "      let EndDate   = datetime(YYYY-MM-DD);\n"
 
#     "• End every statement with a semicolon.\n"
#     "• Provide real line-breaks (no \\n literals).\n\n"
#     "Business → column mapping:\n" + MAPPING_STR +
#     "\n\nDepo/Business Area (gsber) → column Value mapping:\n" + GSBER_MAPPING_STR +
#     "\n\nTable schema:\n" + KUSTO_SCHEMA
# )

#last working
# SYSTEM_PROMPT_KQL = (
#     "You are an expert Kusto (ADX) analyst for SAP sales data.\n"
#     "Output **only raw KQL**, no markdown or commentary.\n"
#     "Rules:\n"
#     "• Use the table SAPSalesInfos and columns below.\n"
#     "• If a date range is optional  , declare:\n"
#     "      let StartDate = datetime(YYYY-MM-DD);\n"
#     "      let EndDate   = datetime(YYYY-MM-DD);\n"
 
#     "• End every statement with a semicolon.\n"
#     "• Provide real line-breaks (no \\n literals).\n\n"
#     "Business → column mapping:\n" + MAPPING_STR +
#     "\n\nDepo/Business Area (gsber) → column Value mapping:\n" + GSBER_MAPPING_STR +
#     "\n\nTable schema:\n" + KUSTO_SCHEMA
# )

SYSTEM_PROMPT_KQL =( f"""
You are an expert Azure Data Explorer (Kusto, ADX) analyst specialized in SAP sales data analysis.

### MISSION:
Convert ANY natural language query about SAP sales data into syntactically correct, optimized KQL.
Handle ALL types of business questions: trends, comparisons, rankings, filtering, aggregations, calculations, and complex analytics.

### OUTPUT RULES:
- Generate ONLY raw KQL code (no markdown, no commentary, no explanations)
- End every statement with semicolon (;)
- Use real line breaks (not \\n)
- Always include performance optimization (limits, efficient filters)

### DATA UNDERSTANDING:
**Table**: {TABLE_NAME}
**Available Data**: Complete SAP sales transactions with customer, product, financial, geographic, and temporal dimensions

**Core Metrics Available**:
- Revenue (real): Sales value in currency
- fkimg (long): Sold quantity 
- volum (real): Sales volume
- Transaction counts, customer counts, product counts

**All Dimensions Available**:
- **Time**: fkdat (transaction date) - supports ANY time-based analysis
- **Customers**: cname (names), kunrg (codes), kukla (groups), ktokd (account types)
- **Products**: arktx (names), matnr (codes), wgbez (brands), matkl (categories)
- **Geography**: Territory, Szone (zones), gsber (business areas/depots)
- **Organization**: spart_text (divisions),spart (division code), bukrs (company), vkorg (sales org), vkgrp_c (sales groups)
- **Documents**: vbeln (invoice numbers), posnr (line items)

### BUSINESS TERM TRANSLATION:
{MAPPING_STR}

### BUSINESS AREA/DEPOT CODES:
{GSBER_MAPPING_STR}

### INTELLIGENT QUERY PROCESSING:
**Handle ANY query type**:
1. **Time Analysis**: "last month", "Q1", "2024", "past 6 months", "year over year", trends, growth
2. **Customer Analysis**: rankings, segmentation, behavior, performance, territory analysis
3. **Product Analysis**: performance, brand analysis, category trends, comparisons, lifecycle
4. **Geographic Analysis**: regional performance, territory comparisons, depot analysis
5. **Organizational Analysis**: division performance, sales team analysis, office comparisons
6. **Financial Analysis**: revenue analysis, profitability, financial trends, ratios
7. **Operational Analysis**: transaction patterns, invoice analysis, volume vs value
8. **Complex Analytics**: statistical analysis, multi-dimensional breakdowns, advanced calculations

### SMART DATE HANDLING:
**Relative Periods** (from now()):
- "yesterday" → let StartDate = ago(1d); let EndDate = now();
- "last week" → let StartDate = ago(7d); let EndDate = now();
- "last month" → let StartDate = ago(30d); let EndDate = now();
- "last 3 months" → let StartDate = ago(90d); let EndDate = now();
- "last 6 months" → let StartDate = ago(180d); let EndDate = now();
- "last year" → let StartDate = ago(365d); let EndDate = now();
- "year to date" → let StartDate = startofyear(now()); let EndDate = now();
- "month to date" → let StartDate = startofmonth(now()); let EndDate = now();

**Specific Periods**:
- "2024" → let StartDate = datetime(2024-01-01); let EndDate = datetime(2024-12-31);
- "January 2025" → let StartDate = datetime(2025-01-01); let EndDate = datetime(2025-01-31);
- "Q1 2024" → let StartDate = datetime(2024-04-01); let EndDate = datetime(2024-06-30);
- "July 2025" → let StartDate = datetime(2025-07-01); let EndDate = datetime(2025-07-31);

**Always filter with**: | where fkdat between (StartDate .. EndDate)

### SMART STRING MATCHING:
**Product/Customer Names**: Use contains for partial match, =~ for exact match
- Single item: arktx contains "ProductName" or cname contains "CustomerName"
- Multiple items: arktx has_any("Product1", "Product2") or cname has_any("Customer1", "Customer2")
- Brand filtering: wgbez contains "BrandName"
- Exact codes: matnr =~ "CODE123" or kunrg == 12345

**Geographic Terms**: Normalize variations automatically
- "Depo 4000", "Depot 4000", "DSC 4000", "Dhaka Sales", "Dhaka" → gsber == 4000
- "Div 1100", "Industrial", "Industrial Division" → spart_text =~ "Industrial"

### PERFORMANCE OPTIMIZATION (MANDATORY):
**Always include result limits**:
- Summary/aggregation queries: | take 500
- Ranking queries: | top 100 by [metric] desc  
- Detail queries: | take 1000
- Top-N queries: | top [N] by [metric] desc

**Efficient query structure**:
1. Date variable declarations (if needed)
2. Base table with most selective filters first
3. Extended calculations
4. Aggregations and grouping
5. Sorting and limiting

### TIME GROUPING INTELLIGENCE:
**Never use bin() - Always use proper time functions**:
- Monthly trends: extend TimePeriod = startofmonth(fkdat)
- Quarterly analysis: extend TimePeriod = startofquarter(fkdat)  
- Yearly analysis: extend TimePeriod = startofyear(fkdat)
- Daily analysis: extend TimePeriod = startofday(fkdat)
- Weekly analysis: extend TimePeriod = startofweek(fkdat)

### ADVANCED ANALYTICS SUPPORT:
**Statistical Functions**: percentile(), avg(), stdev(), dcount(), count()
**Business Calculations**: 
- Unit price: Revenue/fkimg (handle division by zero)
- Growth rate: (Current - Previous) / Previous * 100
- Market share: Customer revenue / Total revenue * 100
- Running totals: Use prev(), next(), row_cumsum()

**Complex Grouping**: Support multi-dimensional analysis, hierarchical breakdowns, pivot operations

### UNIVERSAL QUERY PATTERNS:

**Simple Aggregation**:
```kql
{TABLE_NAME}
| where fkdat >= ago(365d)
| summarize TotalRevenue = sum(Revenue), TotalQuantity = sum(fkimg)
| take 1;
```

**Ranking Analysis**:
```kql
let StartDate = ago(180d);
{TABLE_NAME}
| where fkdat >= StartDate
| summarize TotalRevenue = sum(Revenue), TotalVolume = sum(volum) by cname
| top 10 by TotalRevenue desc;
```

**Trend Analysis**:
```kql
let StartDate = ago(365d);
{TABLE_NAME}
| where fkdat >= StartDate
| extend TimePeriod = startofmonth(fkdat)
| summarize Revenue = sum(Revenue), Volume = sum(volum) by TimePeriod
| sort by TimePeriod asc;
```

**Comparison Analysis**:
```kql
let CurrentYear = getyear(now());
let PreviousYear = CurrentYear - 1;
{TABLE_NAME}
| where getyear(fkdat) in (CurrentYear, PreviousYear)
| extend Year = getyear(fkdat)
| summarize Revenue = sum(Revenue) by Year, cname
| evaluate pivot(Year, sum(Revenue))
| top 20 by [tostring(CurrentYear)] desc;
```

**Multi-dimensional Analysis**:
```kql
let StartDate = ago(365d);
{TABLE_NAME}
| where fkdat >= StartDate
| extend TimePeriod = startofmonth(fkdat)
| summarize Revenue = sum(Revenue), Volume = sum(volum), Customers = dcount(cname) 
  by TimePeriod, Territory, wgbez
| extend UnitPrice = iff(Volume > 0, Revenue / Volume, 0.0)
| sort by TimePeriod asc, Revenue desc
| take 200;
```

### DATA TYPE HANDLING:
**Critical - Use correct data types**:
- gsber comparisons: gsber == 4000 (numeric, NO quotes)
- bukrs comparisons: bukrs == 1000 (numeric, NO quotes)
- String comparisons: cname =~ "CustomerName" (with quotes)
- Date comparisons: fkdat >= datetime(2024-01-01)
- Long comparisons: kunrg == 12345 (numeric, NO quotes)

### ERROR PREVENTION:
**Never use**: bin(fkdat, 1mo) → **Always use**: startofmonth(fkdat)
**Never use**: summarize by , → **Always specify**: summarize ... by TimePeriod  
**Never forget**: Result limiting with take or top
**Never forget**: Semicolon at end of query
**Always use**: Proper data types (numeric vs string)

### OUTPUT FORMAT:
```
// META {{"query_type":"trend|ranking|comparison|summary|analysis","complexity":"simple|medium|complex","filters":{{"applied_filters"}}}}
[Raw KQL Query Here];
```

### COMPLETE TABLE SCHEMA:
{KUSTO_SCHEMA}

### FINAL INSTRUCTION:
For ANY user query about SAP sales data:
1. Understand what they want to analyze (time, customers, products, geography, etc.)
2. Determine required metrics (revenue, quantity, volume, counts, ratios, etc.)
3. Apply appropriate filters, groupings, and calculations
4. Generate optimized KQL with proper performance limits
5. Handle edge cases (division by zero, null values, empty results)

Generate KQL that completely answers their business question using all available data dimensions and analytical capabilities.
"""
)

# ───────────────────────── 3.  LLM instance ────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0,
)

analysis_llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_ANALYSIS,  # e.g., deployment of gpt-5-mini
    temperature      = 1,  # set 0 if you want fully deterministic phrasing
)

# ───────────────────────── 4.  Helpers ────────────────────────────
def _extract_kql(raw: str) -> str:
    """Remove ``` fences/backticks and unescape \\n / \\r / \\t."""
    fenced = re.search(r"```(?:kql|kusto)?\s*([\s\S]*?)```", raw, re.I)
    raw = fenced.group(1) if fenced else raw
    raw = raw.strip("`").replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
    return raw.replace("SAPSalesInfos", TABLE_NAME).strip()


# def build_trend_kql(start: str, end: str, dim_col: str, top_n: int = 5) -> str:
#     return f"""
# // 1) input dates
# let StartDate         = datetime({start});
# let EndDate           = datetime({end});
# // if only one month… previous month window
# let PreviousStartDate = startofmonth(StartDate - 1d);
# let PreviousEndDate   = endofmonth(PreviousStartDate);

# // 2) roll up by month & dimension
# let Monthly = {TABLE_NAME}
# | where fkdat between (PreviousStartDate .. EndDate)
# | summarize Revenue = sum(Revenue)
#     by Period = startofmonth(fkdat), {dim_col};

# // 3) compute growth
# let Growth = Monthly
# | summarize
#     PrevRev = anyif(Revenue, Period == PreviousStartDate),
#     CurrRev = anyif(Revenue, Period == StartDate)
#   by {dim_col}
# | extend GrowthPct = iff(PrevRev == 0, real(null), (CurrRev - PrevRev)*100.0/PrevRev)
# | order by GrowthPct desc
# | take {top_n};

# // 4) output
# Growth
# """.strip()

# def generate_kql(user_req: str, strict=False) -> str:
#     prompt = SYSTEM_PROMPT_KQL
#     if strict:
#         prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."
#     prompt += f"\n\nUser request: {user_req}"
#     print("_extract_kql-------------",prompt)
#     response = llm.invoke([{"role":"user","content":prompt}]).content

#     return _extract_kql(response)
def find_gsber_code(user_input, mapping):
    """
    Robust, case-insensitive, favoring *exact* match (ignoring 'depo', 'sales', etc.)
    """
    cleaned = user_input.lower()
    cleaned = re.sub(r'\b(depo|sales office|sales|office|area|unit)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Build mapping: key without ignored words -> code
    normalized_map = {}
    for k, v in mapping.items():
        nk = k.lower()
        nk = re.sub(r'\b(depo|sales office|sales|office|area|unit)\b', '', nk, flags=re.IGNORECASE)
        nk = nk.strip()
        nk = re.sub(r'\s+', ' ', nk)
        normalized_map[nk] = (k, v)  # original_key, code

    # 1. Exact match after normalization
    if cleaned in normalized_map:
        return normalized_map[cleaned]
    # 2. Try startswith
    for nk, (orig_k, code) in normalized_map.items():
        if nk.startswith(cleaned):
            return orig_k, code
    # 3. Try contains as whole word
    for nk, (orig_k, code) in normalized_map.items():
        if f' {cleaned} ' in f' {nk} ':
            return orig_k, code
    # 4. Fallback: substring anywhere
    for nk, (orig_k, code) in normalized_map.items():
        if cleaned in nk:
            return orig_k, code
    return None, None

#role based access
# def _is_admin(user) -> bool:
#     if not user or not user.is_authenticated:
#         return False
#     if getattr(user, "is_superuser", False):
#         return True
#     try:
#         return user.groups.filter(name__in=["Admin", "Super Admin"]).exists()
#     except Exception:
#         return False
# def _is_admin(user) -> bool:
#     if not getattr(user, "is_authenticated", False):
#         return False
#     if getattr(user, "is_superuser", False) or getattr(user, "is_staff", False):
#         return True
#     # Optional: support app-specific role fields, if you have them
#     for attr in ("role", "designation", "user_role"):
#         val = getattr(user, attr, None)
#         if isinstance(val, str) and val.lower() in ("admin", "super admin", "superadmin"):
#             return True
#     try:
#         return user.groups.filter(name__icontains="admin").exists()
#     except Exception:
#         return False

def _is_admin(user) -> bool:
    if not getattr(user, "is_authenticated", False):
        return False

    if getattr(user, "is_superuser", False) or getattr(user, "is_staff", False):
        return True

    # Optional: support app-specific role fields, if you have them
    for attr in ("role", "designation", "user_role"):
        val = getattr(user, attr, None)
        if isinstance(val, str) and val.lower() in ("admin", "super admin", "superadmin"):
            return True

    try:
        return user.groups.filter(
            Q(name__icontains="admin") | Q(name__iexact="BetaUser")
        ).exists()
    except Exception:
        return False

#column data type
def _parse_schema_types_from_string(schema: str) -> dict[str, str]:
    m = re.search(r'\.create\s+table\s+\w+\s*\(\s*([\s\S]*?)\s*\)\s*', schema, flags=re.I)
    if not m:
        return {}
    body = m.group(1)
    pairs = re.findall(r'([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)', body)
    return {col: typ.lower() for col, typ in pairs}

@lru_cache(maxsize=1)
def get_schema_types_from_static() -> dict[str, str]:
    return _parse_schema_types_from_string(KUSTO_SCHEMA)

def split_types(types: dict[str, str]):
    string_cols   = {c for c, t in types.items() if t == "string"}
    numeric_cols  = {c for c, t in types.items() if t in {"int","long","real","float","double","decimal","bool"}}
    datetime_cols = {c for c, t in types.items() if t in {"datetime","date"}}
    return string_cols, numeric_cols, datetime_cols

def build_schema_prompt_block() -> str:
    types = get_schema_types_from_static()          # parsed from KUSTO_SCHEMA
    string_cols, numeric_cols, datetime_cols = split_types(types)

    schema_lines = "\n".join(f"- {c}: {t}" for c, t in types.items())
    string_cols_csv  = ", ".join(sorted(string_cols)) or "(none)"
    numeric_cols_csv = ", ".join(sorted(numeric_cols)) or "(none)"
    datetime_cols_csv= ", ".join(sorted(datetime_cols)) or "(none)"

    return (
        "\n\nTABLE SCHEMA (from code):\n"
        f"{schema_lines}\n"
        "\nFILTER RULES:\n"
        "- For STRING columns, use case-insensitive operators: `=~` for equality and `in~` for lists.\n"
        "- For NUMERIC columns, use `==` / `in` (no quotes for numbers).\n"
        "- Do NOT use tolower()/toupper(); prefer =~ / in~ for strings.\n"
        "- Do NOT use `bin(fkdat, 1mo)`; use `startofmonth(fkdat)` etc. when grouping.\n"
        "\nSPECIAL COLUMN RULES:\n"
        "- `spart_text` must be matched by case-insensitive substring: use `spart_text contains \"<text>\"`.\n"
        "  If multiple values are provided, expand to `(spart_text contains \"A\" or spart_text contains \"B\" ...)`.\n"
        "\nDERIVED TYPE GROUPS:\n"
        f"- STRING columns: {string_cols_csv}\n"
        f"- NUMERIC columns: {numeric_cols_csv}\n"
        f"- DATETIME columns: {datetime_cols_csv}\n"
    )

def join_system_blocks(blocks: List[str]) -> str:
    return "\n\n".join([b for b in blocks if b and b.strip()])


# Holds the META from the most recent KQL generation (use wherever you need)
LAST_KQL_META: dict = {}

# // META {"dates":{...},"filters":[...]}
_META_LINE_RE = re.compile(r'^\s*//\s*META\s+(\{.*?\})\s*$', re.M)

def _extract_meta_line_and_strip(text: str) -> tuple[dict, str]:
    """
    Looks for a first-line comment:  // META {...}
    Returns (meta_dict, text_without_that_line).
    If not found or invalid JSON → ({}, original_text).
    """
    if not text:
        return {}, text
    m = _META_LINE_RE.search(text)
    if not m:
        return {}, text
    try:
        meta = json.loads(m.group(1))
    except Exception:
        meta = {}
    # remove exactly that line
    stripped = text[:m.start()] + text[m.end():]
    return meta, stripped.strip()

#end column data type

@dataclass
class UserAreaScope:
    depots: List[str]
    zones: List[str]
    territories: List[str]
    restricted: bool

def get_user_area_scope(user) -> UserAreaScope:
    print(">>> get_user_area_scope user:", user, "| is_authenticated:", getattr(user, "is_authenticated", None))
    if _is_admin(user):
        print(">>> user is admin/superadmin; unrestricted scope")
        return UserAreaScope([], [], [], restricted=False)

    depots = list(UserDepoMap.objects.filter(user=user).values_list("depo__code", flat=True))
    zones = list(UserZoneMap.objects.filter(user=user).values_list("zone__code", flat=True))
    territories = list(UserTerritoryMap.objects.filter(user=user).values_list("territory__code", flat=True))
    print(f">>> resolved scope depots={depots} zones={zones} territories={territories}")

    return UserAreaScope(depots=depots, zones=zones, territories=territories, restricted=True)
#end role based access

# compile once
MTD_RE = re.compile(r'\b(?:mtd|month[- ]to[- ]date)\b', re.IGNORECASE)
YTD_RE = re.compile(r'\b(?:ytd|year[- ]to[- ]date)\b', re.IGNORECASE)

CONTRIBUTION_RE = re.compile(r'\b(contribution of|contribution from|contribution by)\b', re.IGNORECASE)
AVG_SALES_RE = re.compile(
    r'\b(?:average|avg|mean)[ -]?(?:sales|revenue|amount|quantity|volume)?\b',
    re.IGNORECASE
)

# lower-case keys for matching
FIELD_MAP_LOWER = {k.lower(): v for k, v in FIELD_MAPPINGS.items()}

TREND_RE     = re.compile(r'\b(?:up[- ]?trending|trending)\b', re.IGNORECASE)
DOWN_TREND_RE = re.compile(r'\b(?:down[- ]?trending|downtrend|negative trend|falling|declining|decreasing)\b', re.IGNORECASE) 
EXCLUDE_KEYS = {"revenue", "quantity", "volume", "date", "fkdat"}

# 3. Cleanup function for LLM-generated KQL
def cleanup_kql(kql: str) -> str:
    # Remove any 'extend' lines and any reference to 'TimePeriod'
    kql = re.sub(r'\s*\|\s*extend[^\n]*\n', '\n', kql)
    kql = re.sub(r'TimePeriod\s*=\s*startofmonth\(fkdat\)', '', kql)
    kql = re.sub(r'TimePeriod', '', kql)
    return kql

# agent.py

# Helper method to build the LLM prompt with metadata

#new
def build_llm_prompt(user_req: str, conversation_id: str, result_json: str, from_agent_meta: str):
    """
    Build the LLM prompt dynamically, including metadata from previous conversation turns.
    """
    prompt = SYSTEM_PROMPT_KQL
    print("----------------------meta-----------------------", from_agent_meta)
    
    # Fetch the latest 20 meta data
    meta_data = get_latest_meta(conversation_id)
    meta_block = ""
    for meta in meta_data:
        if meta.meta_json:
            meta_block += f"\n\n{json.dumps(meta.meta_json)}"

    if meta_block:
        prompt += f"\n\nPrevious Context: {meta_block}"

    # Build the result prompt
    result_prompt = f"User asked: {user_req}\n\n"
    result_prompt += f"Context Data:\n{result_json}\n\n"
    
    # Add meta information if available
    if from_agent_meta:
        result_prompt += f"Meta Information: {from_agent_meta}\n\n"
    
    # Add formatting instructions
    result_prompt += (
         "- If available, show Meta Information naturally in the response. For example: 'from April 2025 to March 2026 in Dhaka North' instead of using structured Meta Information format.\n"
        "- Based on the query results, format the output in bulleted format.\n"
        "- If you found 'gsber', then it's human readable name is Depo/Sales Office. So if you find gsber use Depo/Sales Office.\n"
        "- If the result is numerical or comparative, use bullet points for proper indication. If it's categorical or simple, use bullet points.\n"
        "- After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Amount is in BDT.\n"
        "- If needed, based on the Context Data give meaningful business-related suggestions such as increasing sales, revenue."
    )

    print("****************************************final result_prompt", result_prompt)
    return result_prompt

#old
# def build_llm_prompt(user_req: str, conversation_id: str,result_json:str,from_agent_meta:str):
#     """
#     Build the LLM prompt dynamically, including metadata from previous conversation turns.
#     """
#     prompt = SYSTEM_PROMPT_KQL
#     print("----------------------meta-----------------------",from_agent_meta)
#     # Fetch the latest 20 meta data
#     meta_data = get_latest_meta(conversation_id)
#     meta_block = ""
#     for meta in meta_data:
#         if meta.meta_json:
#             meta_block += f"\n\n{json.dumps(meta.meta_json)}"

#     if meta_block:
#         prompt += f"\n\nPrevious Context: {meta_block}"

#     if from_agent_meta:
#         result_prompt += f"Meta Information: {from_agent_meta}\n\n"
#     # Add the user request to the prompt
#     # prompt += f"\n\nUser request: {user_req}"
#     result_prompt = (
#         f"User asked: {user_req}\n\n"
#         f"Context Data:\n{result_json}\n\n"
#         +"- if available show Meta Information. For example : Period: April 2025 ,Division: Decorative etc. "
#         +"Based on the query results, format the output in bulleted format. "
#         +"Based on the query results, format the output in bulleted format. "
#         + "if you found gsber, then it's human readable name is Depo/Sales Office.so if you find gsber use Depo/Sales Office"
#         +"If the result is numerical or comparative, bullet points for proper indication. If it's categorical or simple, use bullet points. "
#         +"After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Amount is in BDT."
#         +"If Needed, Based on the Context Data give meaningful business-related suggestions such as increasing sales, revenue."
#     )

#     # Add more logic here if necessary for specific user query types
#     print("****************************************final result_prompt",result_prompt)
#     return result_prompt


# Helper method to save the metadata for the current conversation turn
def save_metadata_for_current_turn(conversation_id, message_id, new_meta):
    """
    Save the metadata for the current conversation turn.
    """
    save_meta(conversation_id, message_id, new_meta)

def is_sales_analysis_query(user_req: str, *, conversation_id: str | None = None) -> bool:
    """
    Uses the LLM model to determine if the user's request is related to SAP sales data analysis and KQL generation.
    If a conversation_id is provided, include the last 20 messages as additional context.
    """

    # Base instruction
    prompt = (
        "You are an expert SAP Sales Analysis Assistant.\n"
        f'The user has sent the following request: "{user_req}".\n\n'
        "Please determine if the request is related to SAP sales analysis, such as sales reports, revenue analysis, "
        "growth calculations, or KQL generation.\n"
        'If the request is about SAP sales data analysis, return "yes". If the request is a general question, '
        'unrelated to sales analysis, return "no".'
    )

    # Append last 20 messages if conversation_id is provided
    if conversation_id:
        try:
            conv_id = get_conversation_id_from_uuid(conversation_id)
            last_msgs = get_last_20_messages(conv_id) or []
            if last_msgs:
                prompt += "\n\nRecent conversation (last 20 messages):\n"
                for m in last_msgs[-20:]:
                    role = "USER" if m.sender == "user" else "ASSISTANT"
                    text = (m.text or "").strip()
                    # keep it compact to avoid hitting context limits
                    if len(text) > 400:
                        text = text[:400] + "..."
                    prompt += f"{role}: {text}\n"
        except Exception:
            # fail-open: just proceed without context if anything goes wrong
            pass

    response = llm.invoke([{"role": "user", "content": prompt}]).content.strip()
    return response.lower() == "yes"




def generate_kql(user_req: str, conversation_uuid: Optional[str] = None, strict=False) -> str:
    global LAST_KQL_META
    
    prompt = SYSTEM_PROMPT_KQL
    
    prompt += build_schema_prompt_block()
    # prompt += """
    # OUTPUT FORMAT (must follow exactly):
    # - First line MUST be a one-line comment with compact JSON, then raw KQL only:
    # // META {"dates":{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"},"filters":{"<column>":["<v1>","<v2>"]}}
    # - `dates` should reflect the actual StartDate/EndDate you set (or null if not used).
    # - `filters` must list only the columns and values you actually apply in WHERE, example:
    # {"filters":{"spart_text":["Industrial Paints"], "cname":["Delwar Paint"], "gsber":["4110"]}}
    # """
    prompt += "\n\n" + build_context_memory_contract() + "\n\n"
    prompt += "### SNAPSHOT (use to infer current context)\n"
    prompt += build_conversation_snapshot_block(conversation_uuid)
    prompt += "\n\n### NEW USER MESSAGE\n" + user_req + "\n"

    # Handle conversation history for multi-turn conversation
    # if conversation_uuid:
    #     try:
    #         # Convert UUID to conversation ID
    #         conversation_id = get_conversation_id_from_uuid(conversation_uuid)
            
    #         # Get last 20 messages and message metas
    #         last_20_messages = get_last_20_messages(conversation_id)
    #         last_20_message_metas = get_last_20_message_metas(conversation_id)
            
    #         # Build conversation history prompt
    #         if last_20_messages or last_20_message_metas:
    #             prompt += """
                
    #             MULTI-TURN CONVERSATION CONTEXT:
    #             You are continuing a conversation about KQL generation. Use the previous messages and metadata below to understand the context and generate better KQL queries.
                
    #             CONVERSATION HISTORY (oldest to newest, latest at bottom):
    #             """
                
    #             # Add message history
    #             if last_20_messages:
    #                 prompt += "\n--- PREVIOUS MESSAGES ---"
    #                 for i, message in enumerate(last_20_messages, 1):
    #                     sender_label = "USER" if message.sender == "user" else "ASSISTANT"
    #                     prompt += f"\n{i}. {sender_label}: {message.text}"
    #                     if message.image_url:
    #                         prompt += f" [IMAGE: {message.image_url}]"
    #                 prompt += "\n--- END MESSAGES ---"
                
    #             # Add message meta history
    #             if last_20_message_metas:
    #                 prompt += "\n\n--- PREVIOUS MESSAGE METADATA (KQL Generation Context) ---"
    #                 for i, meta in enumerate(last_20_message_metas, 1):
    #                     if hasattr(meta, 'meta_json') and meta.meta_json:
    #                         prompt += f"\n{i}. META: {json.dumps(meta.meta_json)}"
    #                     elif hasattr(meta, 'content'):  # Adjust field name based on your MessageMeta model
    #                         prompt += f"\n{i}. META: {meta.content}"
    #                 prompt += "\n--- END MESSAGE METADATA ---"
                
    #             prompt += """
                
    #             CONTEXT USAGE INSTRUCTIONS FOR MULTI-TURN CONVERSATION:
    #             1. ANALYZE the conversation history to understand:
    #                - What data the user has been exploring
    #                - Previous filters, date ranges, and query patterns
    #                - User's analytical goals and preferences
                   
    #             2. INHERIT CONTEXT intelligently:
    #                - If current request doesn't specify dates, use the most recent date range from message metadata
    #                - If current request doesn't specify filters (gsber, spart_text, cname, etc.), carry forward relevant filters from previous queries
    #                - Maintain consistency with previous query patterns unless explicitly asked to change
                   
    #             3. ADAPT TO NEW REQUESTS:
    #                - When user specifies new filter values, use them instead of previous ones
    #                - When user asks to "change" or "switch" something, override previous context
    #                - When user asks comparative questions ("compare with", "show difference"), use both old and new parameters
                   
    #             4. CONVERSATIONAL AWARENESS:
    #                - If user says "same period" or "same filters", refer to the most recent metadata
    #                - If user says "this time" or "now try", they're likely modifying the previous query
    #                - If user references "previous results" or "last query", incorporate that context
                   
    #             5. METADATA PRIORITY:
    #                - Latest message metadata (bottom of the list) has highest priority for context inheritance
    #                - Use metadata JSON to understand applied filters, dates, and query structure
    #                - Apply context information as comments in the generated KQL for transparency
                
    #             Current user request: "{user_req}"
                
    #             Generate KQL considering the full conversation context above.
    #             """.format(user_req=user_req)
                
    #     except Exception as e:
    #         # If conversation history fails, continue without it
    #         print(f"Failed to load conversation history: {e}")
    #         pass
    
    # # Keep existing meta_data logic for backwards compatibility
    # meta_data = get_latest_meta(conversation_uuid)
    # meta_block = ""
    # for meta in meta_data:
    #     if meta.meta_json:
    #         meta_block += f"\n\n{json.dumps(meta.meta_json)}"
    
    # if meta_block:
    #     prompt += f"\n\nLegacy Context (for backwards compatibility): {meta_block}"
    #     prompt += """
    #         LEGACY CONTEXT HANDLING:
    #         When generating KQL, intelligently reuse filters and parameters from the legacy context:
    #         - If the current request specifies new filter values, use the new values
    #         - For unspecified filters in the current request, inherit the most recent non-null/non-empty values
    #         - If dates are not mentioned in the current request, reuse the most recent date range
    #         - If other filters (gsber, spart_text, etc.) are not mentioned, carry forward the last specified values
    #         - Only override previous context when explicitly requested or when new values are provided
    #         - Treat null or empty filter values as placeholders that should inherit from previous non-null context
    #         - Add the applied context in comment on the generated KQL
    #         """
    
    # if strict:
    #     prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."
    

    # print("-------------------------------dasdsasadsadsadsad------", prompt)
    
    # types = get_schema_types_from_static()
    # STRING_COLUMNS, NUMERIC_COLUMNS, DATETIME_COLUMNS = split_types(types)

    # prompt += (
    #     "\n\nTABLE SCHEMA (dynamic from code):\n" +
    #     "\n".join(f"- {c}: {t}" for c, t in types.items()) +
    #     "\n\nFILTER RULES:\n"
    #     "- For STRING columns, use case-insensitive operators: `=~` for equality and `in~` for lists.\n"
    #     "- For NUMERIC columns, use `==` / `in` (no quotes for numbers).\n"
    #     "- Do NOT use tolower()/toupper(); prefer =~ / in~.\n"
    # )
    
    #user access 
    _scope = None
    try:
        from core.middleware.current_user import get_current_chat_user 
        # from salesbot.utils.access_scope import get_user_area_scope
        _user = get_current_chat_user()
        print(">>> agent current_user:", _user, "| id:", getattr(_user, "id", None))
        print(_user)
        _scope = get_user_area_scope(_user) if _user else None
        print(">>> generate_kql scope:", _scope)
    except Exception:
        _scope = None

    try:
        # Column map for ADX area columns; override via settings.ADX_AREA_COLUMNS if needed
        _colmap = getattr(settings, "ADX_AREA_COLUMNS", {
            "depo":      {"col": "gsber",     "type": "long"},
            "zone":      {"col": "Szone",     "type": "string"},
            "territory": {"col": "Territory", "type": "string"},
        })

        # Attach JSON block the LLM will use to add filters (case-insensitive)
        if _scope and getattr(_scope, "restricted", False):
            depots_raw = list(getattr(_scope, "depots", []) or [])
            depots_num = []
            for v in depots_raw:
                try:
                    depots_num.append(int(str(v).strip()))
                except Exception:
                    pass  # silently drop non-numeric
            scope_payload = {
                "restricted": True,
                "depots": depots_num,                                # numeric list
                "zones": list(getattr(_scope, "zones", []) or []),
                "territories": list(getattr(_scope, "territories", []) or []),
                "column_map": _colmap,
            }
            # prompt += (
            #     "\n\nUSER_AREA_SCOPE (JSON):\n"
            #     + json.dumps(scope_payload, ensure_ascii=False) + "\n"
            #     "Rules for area scoping:\n"
            #     "- If restricted=true, RESTRICT results to this scope right after the table.\n"
            #     "- Column types:\n"
            #     f"    • depo → {_colmap['depo']['col']} ({_colmap['depo']['type']})\n"
            #     f"    • zone → {_colmap['zone']['col']} ({_colmap['zone']['type']})\n"
            #     f"    • territory → {_colmap['territory']['col']} ({_colmap['territory']['type']})\n"
            #     "- Build filters by type:\n"
            #     "    • long:    <col> in (4000, 4010)  OR  <col> == 4000  (NO quotes, NO in~)\n"
            #     "    • string:  <col> in~ (\"A\",\"B\")  OR  <col> =~ \"A\" (case-insensitive)\n"
            #     "- If a scope array is empty, DO NOT add a filter for that dimension.\n"
            #     "- If the user already asked for area filters, INTERSECT them with this scope using AND.\n"
            #     "- Do not use joins/subqueries just to enforce scope; keep simple where-clauses.\n"
            # )
            # ... after you build `scope_payload` ...
            prompt += (
                "\n\nUSER_AREA_SCOPE (JSON):\n"
                + json.dumps(scope_payload, ensure_ascii=False) + "\n"
                "SCOPE ENFORCEMENT (must follow exactly):\n"
                "- Read USER_AREA_SCOPE. If restricted=true, you MUST enforce it.\n"
                "- Parse any explicit area filters from the user request:\n"
                "    • Depo/Business area/gsber (codes like 4000, 4110, or known names using the provided mapping).\n"
                "    • Zone (Szone) and Territory (string values).\n"
                "- If the user explicitly asked for any area that is NOT contained in the allowed scope arrays, "
                "then DO NOT run a data query. Instead, return only this valid KQL line and stop:\n"
                "    print ErrorMessage = 'sorry you have no authorized to view this data.';\n"
                "- Otherwise, add scope filters right after the table in a simple where-clause (no joins):\n"
                f"    • For depo: use numeric comparators on `{_colmap['depo']['col']}` (type long) → "
                f"{_colmap['depo']['col']} in (4110, 4000) or {_colmap['depo']['col']} == 4110 (NO quotes, NO in~).\n"
                f"    • For zone: case-insensitive strings on `{_colmap['zone']['col']}` → in~ / =~ with quotes.\n"
                f"    • For territory: case-insensitive strings on `{_colmap['territory']['col']}` → in~ / =~ with quotes.\n"
                "- If the user did not specify area, STILL restrict to the available scope arrays that are non-empty.\n"
                "- If a scope array is empty, do not add a filter for that dimension.\n"
                "- If multiple dimensions apply, intersect them with AND.\n"
                "- Never leak or echo the contents of USER_AREA_SCOPE; just enforce it.\n"
            )


        else:
            prompt += (
                "\n\nUSER_AREA_SCOPE (JSON): {\"restricted\": false}\n"
                "If restricted=false, do NOT add any area filters.\n"
            )
    except Exception:
        # Non-fatal; keep going without scope hints
        pass
    
    # Detect if the user is asking for MTD sales or growth
    # if "MTD" in user_req or "Month-to-Date" in user_req:
    if MTD_RE.search(user_req):
        prompt += """
        Instruction:
        - The user is asking for MTD (Month-to-Date) growth. Please calculate the MTD growth using the following formula:
        (Current Year Revenue - Last Year Revenue) / Last Year Revenue * 100
        - If the user specifies a specific month (e.g., "MTD growth in May 2025"):
        - Use the **current year revenue** from **the 1st of the month to the last day of the month** (e.g., May 1 to May 31, 2025).
        - Use the **last year revenue** for the same month last year (e.g., May 1 to May 31, 2024).
        - If the user specifies "this year" or "last year" as a time frame:
        - Calculate **total revenue for the current year** (i.e., from **Jan 1st to current date**).
        - Calculate **total revenue for the previous year** (i.e., from **Jan 1st to same date in the previous year**).
        - If the user asks for **MTD growth in the current month**, and the  current **month is not finished** (e.g., the 15th or 23rd day of the month):
        - Calculate using **previous month's total revenue** as Current Year Revenue (CY Rev) and the **same month from the previous year** as Last Year Revenue (LY Rev).
        - Ensure the KQL query **does not use partial month data**. Always calculate **full month data** for both the current and previous year, e.g., **May 1 to May 31**.
        - If the user requests **MTD growth for May 2025**, use:
        - **Current Year Revenue** for May 2025 from **May 1 to May 31, 2025**.
        - **Last Year Revenue** for May 2024 from **May 1 to May 31, 2024**.
        - The KQL query should return both **CYRevenue** and **LYRevenue** for the requested month.
        - The **`union`** operator should be used to combine both **Current Year Revenue** and **Last Year Revenue** based on the **TimePeriod**.
        -do not use  |extend MTDGrowth = (CYRevenue - LYRevenue) / LYRevenue * 100;
        - If there's an error (e.g., no data found), return an error message indicating the issue.
        """


    # elif "YTD" in user_req or "Year-to-Date" in user_req:
    elif YTD_RE.search(user_req):
        prompt += """
        Instruction:
        - Fiscal year runs April 1 → March 31.
        - Compute YTD through the **last day of the previous month**, **not** through today:
            let FiscalYearStart = datetime(YYYY-04-01);
            // AsOfDate must be end of *prior* month:
            let AsOfDate        = startofmonth(now()) - 1d;  
            // e.g. if today is 2025-07-17, AsOfDate = 2025-06-30
        - Pull two scalars with `toscalar(...)`:
            let CYRevenue = toscalar(
            SAPSalesInfos
            | where fkdat between (FiscalYearStart .. AsOfDate)
            | summarize sum(Revenue)
            );
            let LYRevenue = toscalar(
            SAPSalesInfos
            | where fkdat between (
                datetime_add('year', -1, FiscalYearStart)
                .. datetime_add('year', -1, AsOfDate)
                )
            | summarize sum(Revenue)
            );
        - Emit them with `print`, naming each:
            print 
            YTDGrowth = (CYRevenue - LYRevenue) / LYRevenue * 100,
            CYRevenue = CYRevenue,
            LYRevenue = LYRevenue
        | extend 
            ErrorMessage = iff(isnull(YTDGrowth), "Error: missing data", ""),
            GrowthType   = iff(isnull(YTDGrowth), "N/A", iff(YTDGrowth > 0, "positive growth", "negative growth"))
        - **Do not** use `now()` in any `where` clauses—only use `AsOfDate` as defined above.
        - If either scalar is null, return an appropriate error via `ErrorMessage`.
        """
        prompt += f"\n\nUser request: {user_req}"

    

    # TREND detection logic based on user query working for up trend
    elif TREND_RE.search(user_req):
        # 1) Extract explicit dates or default to last full month
        m = re.search(r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', user_req)
        if m:
            start_date, end_date = m.groups()
        else:
            now = datetime.datetime.now()
            last_month_end = now.replace(day=1) - datetime.timedelta(days=1)
            start_date = f"{last_month_end.year}-{last_month_end.month:02d}-01"
            end_date   = f"{last_month_end.year}-{last_month_end.month:02d}-{last_month_end.day:02d}"

        # 2) Pick the dimension dynamically
        lowered = user_req.lower()
        dim_key = next(
            (k for k in FIELD_MAP_LOWER if k in lowered and k not in EXCLUDE_KEYS),
            "product"
        )
        dim_col = FIELD_MAP_LOWER[dim_key]

        # 3) Count how many months are in the range
        sd = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        ed = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
        month_count = (ed.year - sd.year) * 12 + (ed.month - sd.month) + 1

        if month_count == 2:
            prompt += f"""
            Instruction:
            - The user requested a trend analysis for exactly two months: {start_date} to {end_date}.
            - Identify the top 50 {dim_col} by percentage revenue growth between these two months.
            - For each {dim_col}, calculate the revenue for each month (previous and current).
            - Calculate GrowthPct = (CurrentMonthRevenue - PreviousMonthRevenue) / PreviousMonthRevenue * 100.
            - Add a column TrendType: if CurrentMonthRevenue > PreviousMonthRevenue then "up trend", else "down trend".
            - Output a table:
                | {dim_col} | PreviousMonth | CurrentMonth | PrevRev | CurrRev | GrowthPct | TrendType |
            - Use only `startofmonth(fkdat)` for extracting month, never `bin(fkdat, 1mo)`.
            - Return only raw KQL, no markdown, no commentary.
            """
            prompt += f"\n\nUser request: {user_req}"

        else:
            prompt += f"""
            Instruction:
            - The user requested a trend analysis for more than two months.
            - Identify the top 10 {dim_col} values (e.g., product, brand, dealer, etc.) by total revenue in the period {start_date} to {end_date}.
            - For each of these top 10, return the month-wise revenue for every month in the range, with columns: `{dim_col}`, Period (first of month), Revenue.
            - Use: group by `startofmonth(fkdat)` for each `{dim_col}`.
            - The result should be a table like:
                | {dim_col} | Period      | Revenue   |
                |-----------|-------------|-----------|
                | Example1  | 2025-04-01  | 1200.50   |
                | Example1  | 2025-05-01  | 1350.90   |
                | Example2  | 2025-04-01  | 900.75    |
                | ...       | ...         | ...       |
            - Do not use `bin(fkdat, 1mo)`, only use `startofmonth(fkdat)`.
            - Return only raw KQL, no markdown, no commentary.
            """
            prompt += f"\n\nUser request: {user_req}"

    elif DOWN_TREND_RE.search(user_req):
   
        m = re.search(r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', user_req)
        if m:
            start_date, end_date = m.groups()
        else:
            now = datetime.datetime.now()
            last_month_end = now.replace(day=1) - datetime.timedelta(days=1)
            start_date = f"{last_month_end.year}-{last_month_end.month:02d}-01"
            end_date   = f"{last_month_end.year}-{last_month_end.month:02d}-{last_month_end.day:02d}"

        lowered = user_req.lower()
        dim_key = next(
            (k for k in FIELD_MAP_LOWER if k in lowered and k not in EXCLUDE_KEYS),
            "product"
        )
        dim_col = FIELD_MAP_LOWER[dim_key]

        sd = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        ed = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
        month_count = (ed.year - sd.year) * 12 + (ed.month - sd.month) + 1

        if month_count == 2:
            prompt += f"""
            Instruction:
            - The user requested a **downward trend** analysis for exactly two months: {start_date} to {end_date}.
            - Identify the top 50 {dim_col} by **lowest** percentage revenue growth between these two months (negative growth or least positive).
            - For each {dim_col}, calculate the revenue for each month (previous and current).
            - Calculate GrowthPct = (CurrentMonthRevenue - PreviousMonthRevenue) / PreviousMonthRevenue * 100.
            - Add a column TrendType: if CurrentMonthRevenue < PreviousMonthRevenue then "down trend", else "up trend".
            - Output a table:
                | {dim_col} | PreviousMonth | CurrentMonth | PrevRev | CurrRev | GrowthPct | TrendType |
            - Use only `startofmonth(fkdat)` for extracting month, never `bin(fkdat, 1mo)`.
            - Sort by GrowthPct **ascending** (lowest/most negative growth on top).
            - Return only raw KQL, no markdown, no commentary.
            """
            prompt += f"\n\nUser request: {user_req}"

        else:
            prompt += f"""
            Instruction:
            - The user requested a **downward trend** analysis for more than two months.
            - Identify the top 10 {dim_col} values (e.g., product, brand, dealer, etc.) by **lowest** total revenue growth trend over the period {start_date} to {end_date}.
            - For each of these top 10, return the month-wise revenue for every month in the range, with columns: `{dim_col}`, Period (first of month), Revenue.
            - Use: group by `startofmonth(fkdat)` for each `{dim_col}`.
            - The result should be a table like:
                | {dim_col} | Period      | Revenue   |
                |-----------|-------------|-----------|
                | Example1  | 2025-04-01  | 1200.50   |
                | Example1  | 2025-05-01  | 1350.90   |
                | Example2  | 2025-04-01  | 900.75    |
                | ...       | ...         | ...       |
            - Do not use `bin(fkdat, 1mo)`, only use `startofmonth(fkdat)`.
            - Return only raw KQL, no markdown, no commentary.
            """
            prompt += f"\n\nUser request: {user_req}"


    elif CONTRIBUTION_RE.search(user_req):
        # 1. Parse date range
        m = re.search(r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', user_req)
        if m:
            start_date, end_date = m.groups()
        else:
            # Try "from April 2025 to June 2025"
            month_range = re.search(r'from ([a-zA-Z]+ \d{4}) to ([a-zA-Z]+ \d{4})', user_req, re.IGNORECASE)
            if month_range:
                try:
                    start_dt = dateutil.parser.parse("1 " + month_range.group(1))
                    end_month_dt = dateutil.parser.parse("1 " + month_range.group(2))
                    last_day = calendar.monthrange(end_month_dt.year, end_month_dt.month)[1]
                    end_dt = end_month_dt.replace(day=last_day)
                    start_date = start_dt.strftime("%Y-%m-%d")
                    end_date = end_dt.strftime("%Y-%m-%d")
                except Exception:
                    return "// Could not parse month range. Use format like 'from April 2025 to June 2025'."
            else:
                now = datetime.datetime.now()
                last_month_end = now.replace(day=1) - datetime.timedelta(days=1)
                start_date = f"{last_month_end.year}-{last_month_end.month:02d}-01"
                end_date = f"{last_month_end.year}-{last_month_end.month:02d}-{last_month_end.day:02d}"

        # 2. Detect dimension (brand, division, etc)
        lowered = user_req.lower()
        dim_key = next((k for k in FIELD_MAP_LOWER if k in lowered and k not in EXCLUDE_KEYS), None)
        if not dim_key:
            return "// Could not detect which dimension to use for contribution."
        dim_col = FIELD_MAP_LOWER[dim_key]

        # 3. Extract segment value robustly (remove dimension, remove date/month phrases)
        seg_m = re.search(
            r'contribution (?:of|from)\s+(.*?)(?:\s+in|\s+for|\s+from|\s+by|\s+on|$)', user_req, re.IGNORECASE)
        if not seg_m:
            return "// Could not parse the segment name."
        raw_segment = seg_m.group(1).strip()
        # Remove dimension keyword if present
        strip_dim = re.compile(rf'\b{re.escape(dim_key)}\b', re.IGNORECASE)
        segment = strip_dim.sub('', raw_segment).strip()
        # Remove trailing month/year phrases
        segment = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}$', '', segment, flags=re.IGNORECASE).strip()

        # 4. Build filter clause (handles gsber/depo mapping if needed)
        segment_display = segment
        gsber_key, code = None, None
        if dim_col == "gsber":
            gsber_key, code = find_gsber_code(segment, GSBER_MAPPING)
            print("gsber_key ",gsber_key)
            print("code ",code)
            if gsber_key and code:
                filter_clause = f'{dim_col} == "{code}"'
                segment_display = gsber_key
            else:
                filter_clause = f'{dim_col} == "{segment}"'
                segment_display = segment
        else:
            filter_clause = f'{dim_col} == "{segment}"'
            segment_display = segment
        
        # 5. KQL Template
        return f"""
       
        let StartDate = datetime({start_date});
        let EndDate   = datetime({end_date});
        let TotalRevenue = toscalar(
            {TABLE_NAME}
            | where fkdat >= StartDate and fkdat <= EndDate
            | summarize TotalRevenue = sum(Revenue)
        );
        let SegmentRevenue = toscalar(
            {TABLE_NAME}
            | where fkdat >= StartDate and fkdat <= EndDate and {filter_clause}
            | summarize SegmentRevenue = sum(Revenue)
        );
        print
            Dimension       = "{dim_col}",
            Segment         = "{segment_display}",
            TotalRevenue    = TotalRevenue,
            SegmentRevenue  = SegmentRevenue,
            ContributionPct = iff(isnull(TotalRevenue) or TotalRevenue == 0, real(null), SegmentRevenue * 100.0 / TotalRevenue)
        | extend
            Insight = strcat("The contribution of {segment_display} under {dim_col} is ", round(ContributionPct, 2), "%.")
        """.strip()


    elif AVG_SALES_RE.search(user_req):
        print("-*-------------------------------avg sales--------------------")
        # 1. Extract date range
        m = re.search(r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', user_req)
        if m:
            start_date, end_date = m.groups()
        else:
            now = datetime.datetime.now()
            start_date = now.replace(day=1).strftime("%Y-%m-%d")
            last_day = calendar.monthrange(now.year, now.month)[1]
            end_date = now.replace(day=last_day).strftime("%Y-%m-%d")

        # 2. Detect dimension (brand, dealer, etc.)
        lowered = user_req.lower()
        dim_key = next(
            (k for k in FIELD_MAP_LOWER if k in lowered and k not in EXCLUDE_KEYS),
            None
        )
        dim_col = FIELD_MAP_LOWER[dim_key] if dim_key else None

        # 3. Detect granularity
        period = "monthly"
        if "weekly" in lowered: period = "weekly"
        elif "yearly" in lowered or "annual" in lowered: period = "yearly"
        elif "daily" in lowered: period = "daily"

        period_func = {
            "monthly": "Month = startofmonth(fkdat)",
            "weekly": "Week = startofweek(fkdat)",
            "yearly": "Year = startofyear(fkdat)",
            "daily": "Day = startofday(fkdat)"
        }[period]
        avg_col = {
            "monthly": "AvgMonthlySales",
            "weekly": "AvgWeeklySales",
            "yearly": "AvgYearlySales",
            "daily": "AvgDailySales"
        }[period]

        # 4. Build prompt for LLM
        prompt += f"""
            Hard rule: Never use the `extend` operator anywhere in this query. Never use `extend` for period extraction. All period extractions (Month, Week, etc.) must be done only inside the `summarize by` clause. Do not use or create a `TimePeriod` field.

            Instruction:
            - The user requested an average {period} sales analysis{f' by {dim_col}' if dim_col else ''} for the period {start_date} to {end_date}.
            - Filter data between {start_date} and {end_date}{f' and by {dim_col}' if dim_col else ''}.
            - Step 1: Summarize total revenue per {period} using `{period_func}` inside the `summarize by` clause.{f' Also include {dim_col} in the by clause if specified.' if dim_col else ''}
            - Step 2: Calculate the average of these totals using `summarize {avg_col} = avg(TotalRevenue)`{f' by {dim_col}' if dim_col else ''}.
            - After the first summarize, you may only use columns you have grouped by or calculated.
            - Output columns: |{f' {dim_col} |' if dim_col else ''}{avg_col} |
            - Example KQL:

            let StartDate = datetime({start_date});
            let EndDate = datetime({end_date});
            SAPSalesInfos
            | where fkdat >= StartDate and fkdat <= EndDate{f' and {dim_col} == "<value>"' if dim_col else ''}
            | summarize TotalRevenue = sum(Revenue) by{f' {dim_col},' if dim_col else ''} {period_func}
            | summarize {avg_col} = avg(TotalRevenue){f' by {dim_col}' if dim_col else ''}
            """

        prompt += f"\n\nUser request: {user_req}"

        # 5. Call the LLM to generate KQL
        # kql_generated = llm.invoke([{"role": "user", "content": prompt}]).content

        # # 6. Clean up any forbidden 'extend' or 'TimePeriod'
        # kql_generated = cleanup_kql(kql_generated)

        # print("response from generate kql ", kql_generated)
        # return _extract_kql(kql_generated)

        kql_generated = llm.invoke([{"role": "user", "content": prompt}]).content
        kql_generated = cleanup_kql(kql_generated)

        # NEW — capture META if present
        # global LAST_KQL_META
        meta, kql_body = _extract_meta_line_and_strip(kql_generated)
        LAST_KQL_META = meta
  
        return _extract_kql(kql_body)






    
    # Explicitly instruct LLM to avoid using `bin(fkdat, 1mo)` and instead use `startofmonth(fkdat)`
    else:
        prompt += """
        Instruction: 
        - Do not use the `bin(fkdat, 1mo)` operator for time-based grouping.
        -If not data limit is given on the prompt take top 500 row.
        - Instead, use `startofmonth(fkdat)` for monthly grouping (or other appropriate time functions based on the query).
        - Ensure the query does not use `bin` and directly uses time-based functions for grouping.
        - Group by the result of the time-based function using an `extend` statement, for example: `extend TimePeriod = startofmonth(fkdat)`
        - Always use the named columns in the `summarize` statement.
        """
    
    # prompt += f"\n\nUser request: {user_req}"
    prompt += (
        "\n\nOUTPUT FORMAT (strict):\n"
        "- First line: // META {compact-json-of-actually-applied dates, filters}\n"
        "- Then: RAW KQL ONLY (no markdown, no commentary)."
    )

    if strict:
        prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."

    # Print the full prompt for debugging
    print("_extract_kql-------------", prompt)
    
    # Send the request to the LLM
    response = llm.invoke([{"role": "user", "content": prompt}]).content

    # global LAST_KQL_META
    meta, kql_body = _extract_meta_line_and_strip(response)
    LAST_KQL_META = meta
    #  do cleanups on kql_body 
    print("-------------------------------  LAST_KQL_META    ------------------------",LAST_KQL_META)
    try:
        # Save META to DB for this turn
        message_id = get_latest_message_id(conversation_uuid)
        save_meta(conversation_uuid, message_id, meta)
    except Exception as e:
        print("META save failed:", e)

    kql_clean = kql_body.replace("bin(fkdat, 1mo)", "startofmonth(fkdat)")
    

    if "summarize" in kql_clean and "by ," in kql_clean:
        kql_clean = kql_clean.replace("by ,", "by TimePeriod")

    if "summarize" in kql_clean and ", )" in kql_clean:
        kql_clean = kql_clean.replace(", )", ", TimePeriod)")

    if "summarize" in kql_clean and "by TimePeriod" not in kql_clean:
        kql_clean = kql_clean.replace("summarize", "extend TimePeriod = startofmonth(fkdat)\n| summarize")

    print("response from generate kql ", kql_clean)
    return _extract_kql(kql_clean)

    # Explicitly replace `bin(fkdat, 1mo)` with `startofmonth(fkdat)` or appropriate time function if found
    # response = response.replace("bin(fkdat, 1mo)", "startofmonth(fkdat)")  # Replace bin with startofmonth

    # # Ensure the query has the correct `extend` and `summarize` structure
    # if "summarize" in response and "by ," in response:  # Check if summarize doesn't have a valid grouping field
    #     response = response.replace("by ,", "by TimePeriod")  # Insert a valid field for grouping

    # # If `summarize` is missing the grouping field, add a default grouping by `TimePeriod`
    # if "summarize" in response and ", )" in response:
    #     response = response.replace(", )", ", TimePeriod)")  # Correct the empty `summarize`

    # # If the query doesn't contain a `TimePeriod` column, we add it dynamically (for time-based queries)
    # if "summarize" in response and "by TimePeriod" not in response:
    #     response = response.replace("summarize", "extend TimePeriod = startofmonth(fkdat)\n| summarize")  # Ensure TimePeriod is used
    
    # print("response from generate kql ", response)
    # return _extract_kql(response)



# ───────────────────────── 5.  Main entry ─────────────────────────
# Enhance the handle_user_query to dynamically detect date range via LLM
def detect_date_filter_using_llm(user_prompt: str) -> tuple:
    """
    Call LLM to intelligently detect the date range from the user's prompt.
    """
    today_date = datetime.datetime.now()
    # Construct a prompt to pass to the LLM asking it to extract a date range from the user query
    prompt = f"""
    You are an expert in parsing dates. Based on the following user request, identify and return the start and end date in the format "StartDate = datetime(YYYY-MM-DD); EndDate = datetime(YYYY-MM-DD);" (if applicable):

    Please note that the current date is {today_date.strftime('%B %d, %Y')}.

    Our fiscal year starts from April to March. So, instead of calculating the year from January to December, calculate it from April to March.
    
    User query: {user_prompt}
    
    Please extract the date range from the query. If no date range is detected, return 'None'.
    """
    
    print("from detect_date_filter_using_llm prompt",user_prompt)
    # Send the request to the LLM
    response = llm.invoke([{"role": "user", "content": prompt}]).content
    print("from detect_date_filter_using_llm response",response)
    
    # Parse the LLM's response
    if "None" in response:
        return None, None
    else:
        # Extract start and end date from the response
        date_range_match = re.findall(r'datetime\((\d{4}-\d{2}-\d{2})\)', response)
        if len(date_range_match) == 2:
            start_date = datetime.datetime.strptime(date_range_match[0], "%Y-%m-%d")
            end_date = datetime.datetime.strptime(date_range_match[1], "%Y-%m-%d")
            print("from date_range_match  start_date",start_date)
            print("from date_range_match  end_date",end_date)
            return start_date, end_date
        else:
            return None, None

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
    Dynamically handle SAP Sales prompts with multi-turn conversation support,
    ensuring correct KQL generation, and mapping business area/territory to the correct 'gsber' code.
    Continuity is driven by conversation history only (no META reuse).
    """

    # -----------------------------
    # 1) Non-sales queries → general assistant
    # -----------------------------
    if not is_sales_analysis_query(user_prompt,conversation_id=conversation_id):
        general_prompt = """
        You are a SAP Sales Analysis Assistant. The user has asked a general question not related to sales data analysis or KQL.

        Please respond as a friendly and helpful SAP Sales Analysis Assistant. Let the user know:
        - You are specialized in SAP sales data analysis
        - You can help with sales reports, revenue analysis, growth calculations, trends, etc.
        - Invite them to ask sales-related queries

        Do not generate KQL for general questions.
        """.strip()

        if conversation_id:
            try:
                conv_id = get_conversation_id_from_uuid(conversation_id)
                last_msgs = get_last_20_messages(conv_id)
                if last_msgs:
                    general_prompt += "\n\nCONVERSATION CONTEXT:\n"
                    for m in last_msgs[-20:]:
                        sender = "USER" if m.sender == "user" else "ASSISTANT"
                        general_prompt += f"{sender}: {m.text}\n"
                    general_prompt += "\nBase your answer on the above context."
            except Exception:
                pass

        general_prompt += f"\n\nCurrent user message: {user_prompt}"
        return llm.invoke([{"role": "user", "content": general_prompt}]).content

    # -----------------------------
    # 2) Sales queries → generate KQL
    # -----------------------------
    start_date, end_date = detect_date_filter_using_llm(user_prompt)
    if start_date and end_date:
        user_prompt += f" from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"

    kql = generate_kql(user_prompt, conversation_id)
    kql = format_dates(kql)
    kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
    kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)

    # -----------------------------
    # 3) Territory → gsber mapping
    # -----------------------------
    for territory, gsber_value in GSBER_MAPPING.items():
        if territory.lower() in user_prompt.lower():
            kql = re.sub(
                r"where\s+Territory\s*==\s*['\"]?.+?['\"]?",
                f"where gsber == {gsber_value}",
                kql
            )
            break

    # -----------------------------
    # 4) Trend detection
    # -----------------------------
    trend = detect_trend(user_prompt)
    if trend == "increasing":
        kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")
    elif trend == "steady":
        kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")

    # -----------------------------
    # 5) Execute query (with retry)
    # -----------------------------
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError:
            if attempt == 1:
                kql = generate_kql(user_prompt, conversation_id, strict=True)
                continue
            return "Please refine your query. I couldn't generate a valid KQL this time."

    if not rows:
        return "No data found matching your criteria."

    # -----------------------------
    # 6) Process results
    # -----------------------------
    rows_to_show = rows[:30]
    result_data = []
    for row in rows_to_show:
        row_dict = dict(zip(cols, row))
        for col_name, val in row_dict.items():
            if isinstance(val, datetime.datetime):
                row_dict[col_name] = val.strftime("%Y-%m-%d")
        result_data.append(row_dict)

    # Sort if time-like column exists
    date_cols = [c for c in cols if c.lower() in ("timeperiod", "week", "month", "date")]
    if date_cols:
        result_data.sort(key=lambda x: x[date_cols[0]])

    result_json = json.dumps(result_data, default=str, indent=2)

    # -----------------------------
    # 7) Build narrative prompt
    # -----------------------------
    result_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Context Data:\n{result_json}\n\n"
        "Format the output in bulleted format.\n"
        "- Amount is in BDT and Volume is in gallons.\n"
        "- Replace 'gsber' with 'Depo/Sales Office'.\n"
        "- Use bullet points for both numerical and categorical results.\n\n"
        "Then generate two sections:\n"
        "1. Insights on [context] → trends, patterns, anomalies, risks, opportunities.\n"
        "2. Strategic Recommendations for [context] → actionable suggestions.\n"
        "Section titles should adapt dynamically (e.g. 'Insights on Customer Sales Distribution').\n"
        "Provide meaningful, business-related recommendations if possible."
    )

    if conversation_id:
        try:
            conv_id = get_conversation_id_from_uuid(conversation_id)
            last_msgs = get_last_20_messages(conv_id)
            if last_msgs:
                result_prompt += "\n\nRecent conversation for context:\n"
                for m in last_msgs[-20:]:
                    sender = "user" if m.sender == "user" else "assistant"
                    # msg_text = m.text[:300] + "..." if len(m.text) > 300 else m.text
                    msg_text = m.text or ""
                    result_prompt += f"{sender}: {msg_text}\n"
        except Exception:
            pass

    # -----------------------------
    # 8) Generate narrative output
    # -----------------------------
    # print("dsadsadasd   final result promt           ",result_prompt)
    return llm.invoke([{"role": "user", "content": result_prompt}]).content
    # try:
    #     return analysis_llm.invoke([{"role": "user", "content": result_prompt}]).content
    # except BadRequestError as e:
    #     # Azure returns a JSON body with details (e.g., context length, invalid param, etc.)
    #     try:
    #         err_json = e.response.json()
    #     except Exception:
    #         err_json = {"message": str(e)}
    #     return f"Azure OpenAI 400 Bad Request.\nDetails: {err_json}"

# Enhance handle_user_query to use dynamic date range detection
# def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
#     """
#     Dynamically handle SAP Sales prompts with multi-turn conversation support,
#     ensuring correct KQL generation, and mapping business area/territory to the correct 'gsber' code.
#     """
#     if not is_sales_analysis_query(user_prompt):
#         # If it's a general query, return the response from LLM with conversation context
#         general_prompt = """
#         You are a SAP Sales Analysis Assistant. The user has asked a general question that is not related to sales data analysis or KQL generation.
        
#         Please respond as a friendly and helpful SAP Sales Analysis Assistant. Let the user know:
#         - You are specialized in SAP sales data analysis
#         - You can help with sales reports, revenue analysis, growth calculations, trends, etc.
#         - Invite them to ask about sales-related queries
        
#         Keep the response conversational, helpful, and focused on your role as a sales analysis assistant.
#         Do not generate any KQL code for general conversation.
#         """
        
#         # Add conversation context for general queries if available
#         if conversation_id:
#             try:
#                 conv_id = get_conversation_id_from_uuid(conversation_id)
#                 last_20_messages = get_last_20_messages(conv_id)
                
#                 if last_20_messages:
#                     general_prompt += "\n\nCONVERSATION CONTEXT:\n"
#                     for message in last_20_messages[-5:]:  # Show last 5 messages for context
#                         sender_label = "USER" if message.sender == "user" else "ASSISTANT"
#                         general_prompt += f"{sender_label}: {message.text}\n"
#                     general_prompt += "\nBased on our conversation history above, provide a contextual response."
#             except Exception:
#                 pass
        
#         general_prompt += f"\n\nCurrent user message: {user_prompt}"
        
#         # Get response from LLM for general conversation
#         response = llm.invoke([{"role": "user", "content": general_prompt}]).content
#         return response

#     # -- [unchanged] detect or ask for dates
#     print("user_prompt:", user_prompt)
#     logger.debug("This is a debug message")
#     start_date, end_date = detect_date_filter_using_llm(user_prompt)
  
#     if start_date and end_date:
#         start_date_str = start_date.strftime("%Y-%m-%d")
#         end_date_str   = end_date.strftime("%Y-%m-%d")
#         user_prompt   += f" from {start_date_str} to {end_date_str}"

#     # -- UPDATED: KQL generation with multi-turn conversation support
#     kql = generate_kql(user_prompt, conversation_id)
#     kql = format_dates(kql)
#     kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
#     kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)

#     # -- [unchanged] territory → gsber mapping
#     for territory, gsber_value in GSBER_MAPPING.items():
#         if territory.lower() in user_prompt.lower():
#             kql = re.sub(r"where Territory == .+?", f"where gsber == '{gsber_value}'", kql)
#             break

#     # -- [unchanged] trend detection
#     trend = detect_trend(user_prompt)
#     if trend == "declining":
#         kql = kql.replace("RevenueChange < 0", "RevenueChange < 0")
#     elif trend == "increasing":
#         kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")
#     else:
#         kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")

#     # -- [unchanged] execute with retry
#     for attempt in (1, 2):
#         try:
#             cols, rows = adx().run(kql)
#             break
#         except KustoApiError:
#             if attempt == 1:
#                 kql = generate_kql(user_prompt, conversation_id, strict=True)
#                 print("*********************** kql******************", kql)
#                 continue

#             # If KQL still fails after retry, check if it's a sales-related query at all
#             if not is_sales_analysis_query(user_prompt):
#                 # Not a sales query → return general assistant response with context
#                 general_prompt = """
#                 You are a SAP Sales Analysis Assistant. The user has asked a general question that is not related to sales data analysis or KQL generation.

#                 Please respond as a friendly and helpful SAP Sales Analysis Assistant. Let the user know:
#                 - You are specialized in SAP sales data analysis
#                 - You can help with sales reports, revenue analysis, growth calculations, trends, etc.
#                 - Invite them to ask about sales-related queries

#                 Keep the response conversational, helpful, and focused on your role as a sales analysis assistant.
#                 Do not generate any KQL code for general conversation.
#                 """
                
#                 # Add conversation context for failed queries
#                 if conversation_id:
#                     try:
#                         conv_id = get_conversation_id_from_uuid(conversation_id)
#                         last_20_messages = get_last_20_messages(conv_id)
                        
#                         if last_20_messages:
#                             general_prompt += "\n\nCONVERSATION CONTEXT:\n"
#                             for message in last_20_messages[-3:]:  # Show last 3 messages for context
#                                 sender_label = "USER" if message.sender == "user" else "ASSISTANT"
#                                 general_prompt += f"{sender_label}: {message.text[:200]}...\n"  # Truncate long messages
#                             general_prompt += "\nI see we've been having a conversation, but I had trouble processing your latest request."
#                     except Exception:
#                         pass

#                 general_prompt += f"\n\nUser message: {user_prompt}"

#                 response = llm.invoke([{"role": "user", "content": general_prompt}]).content
#                 return response

#             return "Please refine your query for better results. I'm learning day by day and will help you improve your query."

#     if not rows:
#         return "No data found matching your criteria. Please refine your query for more specific results."

#     # —————————————————————————
#     # ↓ [unchanged] fully dynamic datetime formatting ↓
#     # —————————————————————————

#     # 1) Limit to top N rows
#     rows_to_show = rows[:30]
#     print("rows_to_show = rows[:30]:", rows_to_show)
#     # 2) Build result_data, converting any datetime to "YYYY-MM-DD"
#     result_data = []
#     for row in rows_to_show:
#         row_dict = dict(zip(cols, row))
#         for col_name, value in row_dict.items():
#             if isinstance(value, datetime.datetime):
#                 row_dict[col_name] = value.strftime("%Y-%m-%d")
#         result_data.append(row_dict)

#     # 3) Optionally sort by detected date-like column
#     date_cols = [c for c in cols if c.lower() in ("timeperiod", "week", "month", "date")]
#     if date_cols:
#         key = date_cols[0]
#         result_data.sort(key=lambda x: x[key])

#     # 4) Safe JSON serialization
#     result_json = json.dumps(result_data, default=str, indent=2)
#     print("json.dumps result_data:", result_json)
#     # from_agent_meta = ""
#     # if isinstance(LAST_KQL_META, (list, dict)):
#     #     from_agent_meta = build_applied_context_block(LAST_KQL_META)

#     # meta_data_block = ""
#     # if from_agent_meta:
#     #     meta_data_block = from_agent_meta
#     if conversation_id:
#         # If current meta data is null, try to get previous meta data from conversation
#         try:
#             conv_id = get_conversation_id_from_uuid(conversation_id)
#             last_20_message_metas = get_last_20_message_metas(conv_id)
            
#             if last_20_message_metas:
#                 # Get the most recent non-empty meta data
#                 for meta in reversed(last_20_message_metas):  # Start from latest
#                     if hasattr(meta, 'meta_json') and meta.meta_json:
#                         previous_meta = json.dumps(meta.meta_json)
                        
#                         # Add intelligence prompt to decide if previous meta should be used
#                         # Determine context relevance for decision guidance
#                         decision_guidance = "USED if any context seems relevant else IGNORED"
                        
#                         meta_decision_prompt = f"""
#                         PREVIOUS META DATA ANALYSIS:
#                         Previous metadata: {previous_meta}
#                         Current user prompt: "{user_prompt}"
                        
#                         Analyze if the previous metadata context should be applied to the current query:
                        
#                         DECISION CRITERIA:
#                         - If user mentions "same", "similar", "like before", "previous", "last time" → USE previous meta
#                         - If user asks about same business area/territory/product without specifying new filters → USE previous meta
#                         - If user specifies completely new filters/dates/criteria → DO NOT use previous meta
#                         - If user asks comparative questions ("compare with", "vs last") → USE previous meta for context
#                         - If user asks for different time period without other specifications → USE previous meta filters but update dates
                        
#                         Based on the analysis above, the previous metadata should be {decision_guidance}.
#                         """
                        
#                         meta_data_block = f"PREVIOUS_META_DATA: {previous_meta}\n\nMETA_DECISION_CONTEXT: {meta_decision_prompt}"
#                         break
#                     elif hasattr(meta, 'content') and meta.content:
#                         previous_meta = meta.content
                        
#                         meta_decision_prompt = f"""
#                         PREVIOUS META DATA ANALYSIS:
#                         Previous metadata: {previous_meta}
#                         Current user prompt: "{user_prompt}"
                        
#                         Analyze if the previous metadata context should be applied to the current query based on the decision criteria above.
#                         The previous metadata should be used intelligently based on user intent and query similarity.
#                         """
                        
#                         meta_data_block = f"PREVIOUS_META_DATA: {previous_meta}\n\nMETA_DECISION_CONTEXT: {meta_decision_prompt}"
#                         break
#         except Exception as e:
#             print(f"Failed to retrieve previous meta data: {e}")
    
#     result_prompt = (
#         f"User asked: {user_prompt}\n\n"
#         f"Context Data:\n{result_json}\n\n"
#         + "META_DATA :" + (meta_data_block + "\n\n" if meta_data_block else "") + 
#         "RENDERING RULES for META_DATA:\n"
#         + "- Use META_DATA to understand which dates/filters were applied.\n"
#         + "- If PREVIOUS_META_DATA is provided, intelligently decide whether to reference it based on the META_DECISION_CONTEXT.\n"
#         + "- SHOW ONLY the subset of META_DATA that the user explicitly asked for in this prompt, "
#         + "or that is clearly implied by the prompt (e.g., 'same division' implies the division in META). "
#         + "Hide unrelated filters.\n"
#         + "- When using PREVIOUS_META_DATA, clearly indicate in your response what context from previous queries is being applied.\n"
#     )
    
#     # Add conversation context to the result prompt for better multi-turn responses
#     if conversation_id:
#         try:
#             conv_id = get_conversation_id_from_uuid(conversation_id)
#             last_20_messages = get_last_20_messages(conv_id)
            
#             if last_20_messages:
#                 result_prompt += "\nCONVERSATION CONTEXT (for reference):\n"
#                 # Show only the last few relevant messages to avoid prompt bloat
#                 for message in last_20_messages[-3:]:
#                     sender_label = "USER" if message.sender == "user" else "ASSISTANT"
#                     # Truncate very long messages but keep important context
#                     message_text = message.text[:300] + "..." if len(message.text) > 300 else message.text
#                     result_prompt += f"{sender_label}: {message_text}\n"
                
#                 result_prompt += (
#                     "\nCONTEXT USAGE INSTRUCTIONS:\n"
#                     "- Consider the conversation flow and user's previous questions\n"
#                     "- If the user references 'previous results', 'last query', 'same as before', etc., acknowledge this context\n"
#                     "- Provide comparative insights if the user is building on previous analysis\n"
#                     "- Maintain consistency in terminology and analysis approach\n\n"
#                 )
#         except Exception as e:
#             print(f"Failed to add conversation context to result prompt: {e}")
    
#     # result_prompt += (
#     #     "Based on the query results, format the output in bulleted format.Amount is in BDT and Volume is in gallon. "
#     #     + "if you found gsber, then it's human readable name is Depo/Sales Office.so if you find gsber use Depo/Sales Office"
#     #     +"If the result is numerical or comparative, bullet points for proper indication. If it's categorical or simple, use bullet points. "
#     #     +"After formatting, Insights on [context] → Highlight key trends, patterns, anomalies, risks, and opportunities based on the dataset.Strategic Recommendations for [context] → Provide actionable business suggestions tailored to the insights.Ensure section titles adapt dynamically (e.g., if data is about customers → Insights on Customer Sales Distribution, if about products → Insights on Product Sales Mix)."
#     #     +"If Needed, Based on the Context Data give meaningful business-related suggestions such as increasing sales, revenue."
#     # )
#     result_prompt += (
#         "Based on the query results, format the output in bulleted format. "
#         "Amount is in BDT and Volume is in gallons. "
#         "If you find 'gsber', replace it with the human-readable name 'Depo/Sales Office'. "
#         "If the result is numerical or comparative, use bullet points for clarity. "
#         "If it's categorical or descriptive, also use bullet points for consistency. "
#         "After formatting, generate dynamic analysis with context-aware section titles: "
#         "1. Insights on [context] → Highlight key trends, patterns, anomalies, risks, and opportunities in the dataset. "
#         "2. Strategic Recommendations for [context] → Provide actionable business suggestions tailored to the insights. "
#         "Ensure section titles adapt dynamically (e.g., if data is about customers → 'Insights on Customer Sales Distribution', "
#         "if about products → 'Insights on Product Sales Mix'). "
#         "If relevant, provide meaningful business-related suggestions such as improving sales, increasing revenue, "
#         "enhancing customer retention, or reducing risks etc."
#     )
    
#     print("****************************************final prompt", result_prompt)
#     formatted_result = llm.invoke([{"role": "user", "content": result_prompt}]).content
#     return formatted_result

