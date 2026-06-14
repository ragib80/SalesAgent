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
from user_auth.models import UserDepoMap, UserZoneMap, UserTerritoryMap, UserDivisionMap
from typing import List, Dict
from dataclasses import dataclass, field
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

from core.middleware.current_user import get_current_chat_user ,set_current_chat_user

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
        # kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
        kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
            cluster,
            aad_app_id=getattr(settings, "AZURE_CLIENT_ID", os.getenv("AZURE_CLIENT_ID")),
            app_key=getattr(settings, "AZURE_CLIENT_SECRET", os.getenv("AZURE_CLIENT_SECRET")),
            authority_id=getattr(settings, "AZURE_TENANT_ID", os.getenv("AZURE_TENANT_ID")),
        )
        
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
    "division":"spart_text","division code":"spart","company code":"bukrs","company":"bukrs","sales org":"vkorg",
    "dist channel":"vtweg","distribution channel":"vtweg","channel":"vtweg","business area":"gsber","depo":"gsber",
    "credit control area":"kkber","Dealer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL",
    "product code":"matnr","material code":"matnr","material code":"meins","volume unit":"voleh","business group":"GK",
    "territory":"Territory","sales zone":"Szone","date":"fkdat","Dealer Code":"kunrg","dealer code":"kunrg",
    "fkdat":"fkdat","invoice number":"vbeln", "sales org":"vkorg","sales organization":"vkorg","credit control area":"kkber"
}
MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in FIELD_MAPPINGS.items())

KUSTO_SCHEMA = """
.create table SAPSalesInfos (
    Id: string, CreatedTime: string, ModifiedTime: string, bukrs: long,
    spart: long, matkl: string, wgbez: string, matnr: string, vkorg: long,
    kunrg: long, kunnr_sh: long, Payer_DL: long, vbeln: long, vkbur_c: long,
    vkgrp_c: string, kukla: long, fkdat: datetime, posnr: long, arktx: string,
    meins: string, voleh: string, Territory: string, Szone: string, cname: string,
    spart_text: string, Revenue: real, gsber: long, fkimg: long, volum: real,
    ktokd: string, vtweg: long, erzet_T: timespan, kkber: long, FKDAT_TEMP: datetime,
    GK: string, temp_col: string
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

DIVISION_MAPPING = {
    "Decorative": "10",
    "Industrial Paints": "20",
    "Marine Paints": "30",
    "Powder Coating": "40",
    "Adhesive & Chemicals": "50",
    "Trading": "60",
    "Wood Coating": "80",
    "Construction Chemica": "90",
}
DIVISION_MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in DIVISION_MAPPING.items())

VTWEG_MAPPING = {
    "Dealer": 10,
    "Customer": 20,
    "Project Customer": 30
}

# vtweg is long (numeric) — KQL must use: vtweg == 10, NOT vtweg == "10"
VTWEG_MAPPING_STR = "\n".join(f'"{k}": {v}' for k, v in VTWEG_MAPPING.items())

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
- **MANDATORY GLOBAL FILTER — NO EXCEPTIONS**: Every subquery or table scan on {TABLE_NAME} MUST include `| where bukrs == 1000` as the FIRST filter after the table name. This applies to every `let` block, every inline scan, every toscalar() call — every single access to {TABLE_NAME}. Never omit this filter regardless of what the user asks.

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

### DISTRIBUTION CHANNEL (vtweg) CODES:
{VTWEG_MAPPING_STR}

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
**CRITICAL — ALWAYS use CURRENT DATE CONTEXT (injected below) for relative periods. Do NOT guess.**

ALL relative period phrases listed below MUST use the EXACT datetime values from the CURRENT DATE CONTEXT table:
- "this year" / "current year" / "this fiscal year"
- "last year" / "previous year" / "last fiscal year"
- "this month" / "current month"
- "last month" / "previous month"
- "MTD" / "month-to-date"
- "YTD" / "year-to-date"
- "this quarter" / "current quarter"
- "last quarter" / "previous quarter"
- "last week" / "this week"
- "today" / "yesterday"
- "last 3 months" / "last 6 months"

**FORBIDDEN approximations** — NEVER use these for the above phrases:
- ❌ `ago(365d)` for "last year" (fiscal year is April-March, NOT rolling 365 days)
- ❌ `startofyear(now())` for "this year" / "YTD" (ignores fiscal year)
- ❌ `startofmonth(now())` for "MTD" (current month is incomplete — use last complete month)
- ❌ `getyear(now())` / `getyear(fkdat)` for fiscal year comparisons (calendar year ≠ fiscal year)

**Exception** — ONLY use `ago()` / `now()` when user explicitly requests a rolling window:
- "past 7 days" → `ago(7d)`
- "past 30 days" → `ago(30d)`

**Specific Explicit Periods** (when user names a calendar month/year/quarter):
- "2024" → let StartDate = datetime(2024-01-01); let EndDate = datetime(2024-12-31);
- "January 2025" → let StartDate = datetime(2025-01-01); let EndDate = datetime(2025-01-31);
- "Q1 FY2024" (Fiscal Q1 = Apr-Jun) → let StartDate = datetime(2024-04-01); let EndDate = datetime(2024-06-30);
- "Q2 FY2024" (Fiscal Q2 = Jul-Sep) → let StartDate = datetime(2024-07-01); let EndDate = datetime(2024-09-30);
- "Q3 FY2024" (Fiscal Q3 = Oct-Dec) → let StartDate = datetime(2024-10-01); let EndDate = datetime(2024-12-31);
- "Q4 FY2024" (Fiscal Q4 = Jan-Mar) → let StartDate = datetime(2025-01-01); let EndDate = datetime(2025-03-31);
- "July 2025" → let StartDate = datetime(2025-07-01); let EndDate = datetime(2025-07-31);

**Always filter with**: | where fkdat between (StartDate .. EndDate)

### SMART STRING MATCHING:
**Product/Customer Names**: Use contains for partial match, =~ for exact match, always use contains for cname.
-cname with code suffix: When a cname is shown like "<Name> (<digits>)" (e.g., Delwar Paint (24)), treat the (<digits>) as the dealer/customer code kunrg.For name filtering, ignore the trailing (<digits>) and match only the name with contains (e.g., cname contains "Delwar Paint").you may also filter exactly by kunrg (e.g., kunrg == 24)
- Single item: arktx contains "ProductName" or cname contains "CustomerName"
- Multiple items: arktx has_any("Product1", "Product2") or cname has_any("Customer1", "Customer2")
- Brand filtering: wgbez contains "BrandName"
- Exact codes: matnr =~ "CODE123" or kunrg == 12345

**Geographic Terms**: Normalize variations automatically
- "Depo 4000", "Depot 4000", "DSC 4000", "Dhaka Sales", "Dhaka" → gsber == 4000
- "Div 1100", "Industrial", "Industrial Division" → spart_text contains "Industrial"

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
6. if user ask about lifting ,then lifting means the total Volume and total revenue of product 

### TIME GROUPING INTELLIGENCE:
**Never use bin() - Always use proper time functions**:
- Monthly trends: extend TimePeriod = startofmonth(fkdat)
- Quarterly analysis: extend TimePeriod = startofquarter(fkdat)  
- Yearly analysis: extend TimePeriod = startofyear(fkdat)
- Daily analysis: extend TimePeriod = startofday(fkdat)
- Weekly analysis: extend TimePeriod = startofweek(fkdat)



### MULTI-YEAR / MULTI-PERIOD COMPARISONS ("INDIVIDUALLY")
When the user asks about multiple periods "individually" or "separately", for example:
- "in 2025 and 2024 individually"
- "compare 2023 and 2024 separately"
- "show each year separately"
- "month wise individually" or similar wording,

you MUST follow this pattern:

1. Create ONE subquery per period using let.
2. In each subquery:
   - Filter the base table by that period (for example using fkdat between datetime(YYYY-01-01) .. datetime(YYYY-12-31) for whole years).
   - Apply all other requested filters (brand, division, depot, etc.).
   - Summarize the required metrics by the requested dimensions (e.g. dealer, area, region).
   - AFTER summarize, add a CONSTANT label column for that period, such as:
       | extend Year = "2025"
     or for months:
       | extend Period = "2025-01"
3. Use union to combine all the period subqueries:
   union Period2024, Period2025
4. Project the period label (Year / Period) along with the grouped dimensions and metrics.
5. Optionally order by the period label and the main metric.

IMPORTANT RULES:
-if "company code" or "company" is 1000 it means it is for 'Berger Paints Bangladesh Limited'
- DO NOT try to infer the year or period AFTER union using fkdat. fkdat usually does not exist after summarize.
- NEVER write:
    | union Sales2025, Sales2024
    | extend Year = iff(fkdat between (...), "2025", "2024")
  because fkdat is not guaranteed to be present at that stage.
- For multi-year comparisons where explicit periods are given (like 2024 and 2025), PREFER the "one let per period + union + constant label" pattern instead of getyear(fkdat).

EXAMPLE PATTERN (DEALER LIST FOR TWO YEARS):
User: "show me the dealer list who bought Brand APE CLASSIC in Division Decorative in 2025 and 2024 individually."

Generated KQL should follow this structure:

let Sales2025 = {TABLE_NAME}
| where fkdat between (datetime(2025-01-01) .. datetime(2025-12-31))
| where wgbez contains "APE CLASSIC"
  and spart_text contains "Decorative"
| summarize
    TotalRevenue  = sum(Revenue),
    TotalVolume   = sum(volum),
    TotalQuantity = sum(fkimg)
  by kunrg, cname, gsber, vtweg
| extend Year = "2025";

let Sales2024 = {TABLE_NAME}
| where fkdat between (datetime(2024-01-01) .. datetime(2024-12-31))
| where wgbez contains "APE CLASSIC"
  and spart_text contains "Decorative"
| summarize
    TotalRevenue  = sum(Revenue),
    TotalVolume   = sum(volum),
    TotalQuantity = sum(fkimg)
  by kunrg, cname, gsber, vtweg
| extend Year = "2024";

union Sales2025, Sales2024
| project Year, cname, kunrg, gsber, vtweg,
          TotalRevenue, TotalVolume, TotalQuantity
| order by Year desc, TotalRevenue desc
| take 500;


### SPECIAL SCENARIO: CUSTOMER DROP-OFF (BOUGHT BEFORE, NOT AFTER)
If the user asks:
- "Which customers/dealers bought [a product] in one time period but did not buy it in another?"
- "Show me who purchased in April 2025 but not in May 2025"
- "Which customers stopped buying this product next month?"
- or similar questions about customers missing in later periods —

Then generate KQL that:

1. **Identifies the first time window (Period A)** → customers who bought the specific product(s) in that period.
2. **Identifies the second time window (Period B)** → customers who bought the same product(s) in that later period.
3. **Compares both sets** using `join kind=leftanti` on customer code (`kunrg`) to find those who are **present in Period A but missing in Period B**.
4. **Summarizes key metrics** such as Revenue, Quantity (`fkimg`), and Volume (`volum`).
5. **Projects** customer name (`cname`), customer code (`kunrg`), business area (`gsber`), and distribution channel (`vtweg`).
6. **Orders** the result by highest Revenue or Volume to highlight major missing customers.

**Example Pattern:**
```kql
let buyers_A = {TABLE_NAME}
| where fkdat between (start_period .. end_period)
| where arktx contains "product_name"
| summarize Revenue_A = sum(Revenue) by kunrg, cname, vtweg, gsber;

let buyers_B = {TABLE_NAME}
| where fkdat between (next_start .. next_end)
| where arktx contains "product_name"
| summarize Revenue_B = sum(Revenue) by kunrg, cname, vtweg, gsber;

buyers_A
| join kind=leftanti buyers_B on kunrg
| project cname, kunrg, vtweg, gsber, Revenue_A
| order by Revenue_A desc;
```

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

**Comparison Analysis** (fiscal year — use CURRENT DATE CONTEXT for dates):
```kql
// ⚠ Use exact dates from CURRENT DATE CONTEXT — do NOT use getyear(now()) for fiscal year
let CY_Start = datetime(YYYY-04-01);   // current fiscal year start (from CURRENT DATE CONTEXT)
let CY_End   = datetime(YYYY-MM-DD);   // today (from CURRENT DATE CONTEXT)
let PY_Start = datetime(YYYY-04-01);   // previous fiscal year start (from CURRENT DATE CONTEXT)
let PY_End   = datetime(YYYY-03-31);   // previous fiscal year end (from CURRENT DATE CONTEXT)

let CY = {TABLE_NAME}
| where fkdat between (CY_Start .. CY_End)
| summarize CY_Revenue = sum(Revenue) by cname, kunrg;

let PY = {TABLE_NAME}
| where fkdat between (PY_Start .. PY_End)
| summarize PY_Revenue = sum(Revenue) by kunrg;

CY
| join kind=leftouter PY on kunrg
| extend GrowthPct = iff(PY_Revenue == 0 or isnull(PY_Revenue), real(null),
                         (CY_Revenue - PY_Revenue) / PY_Revenue * 100.0)
| project cname, kunrg, CY_Revenue, PY_Revenue, GrowthPct
| order by CY_Revenue desc
| take 40;
```

**NOTE — calendar year explicit comparison** (ONLY when user explicitly says "in 2024 vs 2025" with calendar years):
```kql
// Only for explicit calendar year comparisons — NOT for "this year vs last year"
let CY_Start = datetime(2025-01-01);
let CY_End   = datetime(2025-12-31);
let PY_Start = datetime(2024-01-01);
let PY_End   = datetime(2024-12-31);
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
- **bukrs (company code) — GLOBAL MANDATORY FILTER**: ALWAYS add `| where bukrs == 1000` as the first filter in EVERY subquery. bukrs is long, use numeric (no quotes). This is company-scoped data — other bukrs values belong to different companies and must never appear in results.
- gsber comparisons: gsber == 4000 (numeric, NO quotes)
- vtweg comparisons: vtweg == 10 (numeric, NO quotes) — Dealer=10, Customer=20, Project Customer=30
- String comparisons: field =~ "Value" (with quotes)
 -Exception – cname: use cname contains "CustomerName" instead of =~

- Date comparisons: fkdat >= datetime(2024-01-01)
- Long comparisons: kunrg == 12345 (numeric, NO quotes)
 -matkl normalization (critical): When the user provides matkl like f010 (RSE) or F010(ABC), extract only the leading F + digits (F\\d+) and ignore everything after (spaces/parentheses).

### DATA TYPE ENFORCEMENT (STRICT)
- Before using any column in WHERE, determine its type from schema:
  - long / real → numeric equality (== 12345) without quotes
  - string → use contains(), =~, has_any(), inside quotes "ABC"
  - datetime → use datetime() wrappers

- NEVER produce string comparison on numeric columns.

### SPECIAL SCENARIO: DEALERS WHO BOUGHT IN ONE PERIOD BUT NOT IN THE NEXT (RISK / DROP-OFF LIST)
When the user asks things like:
- "Dealers who bought in May but did not buy in June"
- "Show dealers who purchased in [Period A] but not in [Period B]"
- "Risk / drop-off dealers between two months/years"
- "Which dealers bought the product last period but not this period"
you MUST treat this as a **risk / drop-off dealer list**, NOT a list of all active dealers.

#### INTENT
- Return the **list of dealers** who:
  - **Did buy** the selected product/brand in an earlier period (**Period A**), and
  - **Did NOT buy** the same product/brand in a later period (**Period B**).
- Use the **same product/criteria filters** in both periods.
- This is explicitly a **“drop-off / at-risk dealers”** list.

#### TIME PERIOD HANDLING
1. Identify two explicit periods from the user request:
   - Period A = base period (e.g., "May 2025").
   - Period B = comparison period (e.g., "June 2025").
2. Convert them to date ranges using SMART DATE HANDLING rules, for example:
   - "May 2025"  → PeriodA_Start = datetime(2025-05-01), PeriodA_End = datetime(2025-05-31)
   - "June 2025" → PeriodB_Start = datetime(2025-06-01), PeriodB_End = datetime(2025-06-30)

#### PRODUCT / CRITERIA FILTERS
- Apply the SAME product/brand/division filters in both periods, for example:
  - Specific brand: `wgbez contains "APE CLASSIC"`
  - Specific product: `matnr =~ "12345"` or `arktx contains "ProductName"`
  - Specific division: `spart_text contains "Decorative"`


#### KQL PATTERN TO FOLLOW
Generate KQL that:
1. Builds the set of dealers who bought in Period A.
2. Builds the set of dealers who bought in Period B.
3. Uses `join kind=leftanti` to keep only dealers who are in Period A but NOT in Period B.
4. Returns a dealer-level table with May (Period A) Sales/Volume/Quantity only.

Example template (adapt dates and filters based on the user’s requested periods and criteria):

```kql
// Period A: May 2025
let PeriodA_Start = datetime(2025-05-01);
let PeriodA_End   = datetime(2025-05-31);

// Period B: June 2025
let PeriodB_Start = datetime(2025-06-01);
let PeriodB_End   = datetime(2025-06-30);

// Dealers who bought selected product/brand in Period A (base period)
let DealersA = {TABLE_NAME}
| where fkdat between (PeriodA_Start .. PeriodA_End)
| where <PRODUCT_AND_CRITERIA_FILTERS>          // e.g. wgbez contains "APE CLASSIC" and spart_text contains "Decorative"

| summarize
    TotalRevenue  = sum(Revenue),
    TotalVolume   = sum(volum),
    TotalQuantity = sum(fkimg)
  by kunrg, cname, gsber, vtweg;

// Dealers who bought selected product/brand in Period B (later period)
let DealersB = {TABLE_NAME}
| where fkdat between (PeriodB_Start .. PeriodB_End)
| where <PRODUCT_AND_CRITERIA_FILTERS>
| summarize
    TotalRevenueB = sum(Revenue)
  by kunrg;

// Risk / drop-off dealers: bought in Period A but NOT in Period B
DealersA
| join kind=leftanti DealersB on kunrg
| project kunrg, cname, gsber, vtweg, TotalRevenue, TotalVolume, TotalQuantity
| order by TotalRevenue desc
| take 500;
```

### ERROR PREVENTION:
**Never use**: bin(fkdat, 1mo) → **Always use**: startofmonth(fkdat)
**Never use**: summarize by , → **Always specify**: summarize ... by TimePeriod  
**Never forget**: Result limiting with take or top
**Never forget**: Semicolon at end of query
**Always use**: Proper data types (numeric vs string)

### COMPLETE TABLE SCHEMA:
{KUSTO_SCHEMA}

### DECLINING / NEGATIVE GROWTH ANALYSIS (ENTITY RANKING — CRITICAL):
**TRIGGER WORDS**: "declining", "negative growth", "most negative growth", "sales declining", "revenue declining", "decreasing sales/revenue", "which dealers/brands/products made negative growth"

When the user asks **which/top N [dealers/brands/products/divisions/etc.] have declining or negative growth** compared to another period:

**THIS IS NOT A TREND CHART** — this is a **period-over-period comparison** to rank entities by negative change.

**REQUIRED PATTERN** (adapt entity columns per request):
```kql
let CY_Start = datetime(YYYY-MM-DD);   // comparison period start (the NEWER / LATER period, e.g. Feb 2026)
let CY_End   = datetime(YYYY-MM-DD);   // comparison period end
let PY_Start = datetime(YYYY-MM-DD);   // base period start (the OLDER / EARLIER period, e.g. Jan 2026)
let PY_End   = datetime(YYYY-MM-DD);   // base period end

// CY: only the entity key — no name column here
let CY = {TABLE_NAME}
| where fkdat between (CY_Start .. CY_End)
| summarize CY_Revenue = sum(Revenue) by [entity_key];

// PY: entity key + name + any extra project cols (name lives here so it is always available after leftouter join)
let PY = {TABLE_NAME}
| where fkdat between (PY_Start .. PY_End)
| summarize PY_Revenue = sum(Revenue) by [entity_key], [entity_name_col], [extra_project_cols];

PY
| join kind=leftouter CY on [entity_key]                              // leftouter: keep base-period entities even with ZERO comparison-period sales
| extend CY_Revenue = iif(isnull(CY_Revenue), 0.0, CY_Revenue)       // no transactions in comparison period → treat as 0 revenue
| extend GrowthPct = iff(PY_Revenue == 0, real(null), (CY_Revenue - PY_Revenue) / PY_Revenue * 100.0)
| where PY_Revenue > 0            // CRITICAL: exclude SAP credit-memo artifacts in the base period
| where CY_Revenue >= 0           // CRITICAL: exclude net-negative comparison revenue (returns > sales is an SAP artifact, NOT a real decline — these pollute the top results)
| where CY_Revenue < PY_Revenue   // only genuinely declining entities
| project [entity_name_col], [entity_key], [extra_project_cols], CY_Revenue, PY_Revenue, GrowthPct
| top 10 by GrowthPct asc;        // most negative first; adjust N per user request
```

**WHY THIS PATTERN (do not deviate)**:
- `leftouter` from PY: a dealer who had sales in Jan but ZERO sales in Feb is the biggest decline (100%) — `innerunique` would silently drop them.
- `CY_Revenue >= 0` filter: SAP posts credit memos / returns as negative revenue rows. A dealer with Feb net-revenue = -50,000 BDT is NOT a "declining dealer" — it is a data artifact. Without this filter the entire top-10 list fills up with return-heavy dealers instead of genuine decliners.
- `PY_Revenue > 0` filter: same reason — exclude credit-memo artifacts from the base period.
- Entity name is summarized in PY (not CY) so it is always present after a leftouter join.

**QUARTER COMPARISON RULE (CRITICAL)**:
For "last quarter" declining/growth → always use **year-over-year** (same quarter vs same quarter last year):
- CY = last complete fiscal quarter (e.g., Oct-Dec 2025)
- PY = same quarter one year earlier (e.g., Oct-Dec 2024)
Use the exact dates from CURRENT DATE CONTEXT row "last quarter".
**NEVER** compare last quarter vs the quarter before it (quarter-over-quarter).

**ENTITY COLUMN MAP** — choose based on what user asks about:
| User mentions | entity_key | summarize by (CY) | extra project cols |
|---|---|---|---|
| dealer / dealers | kunrg | kunrg, cname | gsber, vtweg |
| brand / brands | wgbez | wgbez | — |
| product / products | matnr | matnr, arktx | — |
| category / categories | matkl | matkl | wgbez |
| division / divisions | spart | spart, spart_text | — |
| territory | Territory | Territory | — |
| depo / business area | gsber | gsber | — |

**FISCAL YEAR DATES** — always use these for "this year" / "last year":
- "this year" → current fiscal year start (April 1) to today (from CURRENT DATE CONTEXT)
- "last year" → previous fiscal year (April 1 → March 31) (from CURRENT DATE CONTEXT)
- Default (no period mentioned) → same as "this year vs last year"

**IMPORTANT**: "sales" = Revenue. "quantity" = fkimg. "volume" = volum. Apply all other filters (brand, division, depo) in both CY and PY subqueries.

**POSITIVE GROWTH VARIANT** — use this when user asks for top performers / highest growth (NOT declining):
- Use `join kind=innerunique` (both periods must have sales for a meaningful growth figure).
- Entity name col goes in CY summarize (both are present so either side works, but keep it in CY for positive variant).
- Keep `| where CY_Revenue >= 100` and `| where PY_Revenue >= 100` to exclude SAP credit-memo artifacts AND near-zero phantom revenue in both periods. Using `> 0` is insufficient — a PY_Revenue of 0.001 BDT passes the filter but causes astronomically large growth percentages (e.g., 2,565,969,868,159,451,000%). The 100 BDT floor eliminates floating-point residuals and trivial one-off transactions that produce meaningless ratios.
- Remove `| where CY_Revenue < PY_Revenue` (we want growers, not decliners).
- Sort `| top 10 by GrowthPct desc` (most positive first).
- Example triggers: "top 10 dealers by growth", "best performing brands", "highest revenue increase", "top performers this year vs last year"

---

### MTD (MONTH-TO-DATE) GROWTH CALCULATION:
**TRIGGER**: User mentions "MTD", "month-to-date", "MTD growth"

**DATE RULES**:
- Use the last **complete** month as CY (avoid partial current month).
- If user specifies a month (e.g., "MTD May 2025"): use full May 2025 vs full May 2024.
- Always use `endofmonth()` for month-end to handle leap years automatically.
- MTD growth = (CY_Revenue - LY_Revenue) / LY_Revenue * 100

**SCENARIO A — Overall MTD (no entity) or single specific entity**:
Apply the entity filter in BOTH CY and LY subqueries.
```kql
let CY_Start = datetime(YYYY-MM-01);
let CY_End   = endofmonth(CY_Start);                      // handles leap years
let LY_Start = datetime_add('year', -1, CY_Start);
let LY_End   = endofmonth(LY_Start);

// For a specific entity, add: | where cname contains "X"  (or wgbez, gsber, spart_text, etc.)
let CYRevenue = toscalar({TABLE_NAME}
    | where fkdat between (CY_Start .. CY_End)
    [| where <entity_filter_if_specified>]
    | summarize sum(Revenue));
let LYRevenue = toscalar({TABLE_NAME}
    | where fkdat between (LY_Start .. LY_End)
    [| where <entity_filter_if_specified>]
    | summarize sum(Revenue));

print
    CYRevenue    = CYRevenue,
    LYRevenue    = LYRevenue,
    MTDGrowthPct = iff(LYRevenue == 0, real(null), (CYRevenue - LYRevenue) / LYRevenue * 100),
    GrowthType   = iff(LYRevenue == 0, "N/A", iff(CYRevenue > LYRevenue, "Positive Growth", "Negative Growth"));
```

**SCENARIO B — MTD growth ranked BY entity** (e.g., "MTD growth by dealer", "top brands by MTD growth"):
Use `join kind=leftouter` so entities with CY data but no LY data are still included.
```kql
let CY_Start = datetime(YYYY-MM-01);
let CY_End   = endofmonth(CY_Start);
let LY_Start = datetime_add('year', -1, CY_Start);
let LY_End   = endofmonth(LY_Start);

let CY = {TABLE_NAME}
| where fkdat between (CY_Start .. CY_End)
| summarize CY_Revenue = sum(Revenue), CY_Qty = sum(fkimg) by [entity_key], [entity_name_col];

let LY = {TABLE_NAME}
| where fkdat between (LY_Start .. LY_End)
| summarize LY_Revenue = sum(Revenue) by [entity_key];

CY
| join kind=leftouter LY on [entity_key]
| extend MTDGrowthPct = iff(LY_Revenue == 0 or isnull(LY_Revenue), real(null),
                            (CY_Revenue - LY_Revenue) / LY_Revenue * 100)
| project [entity_name_col], [entity_key], CY_Revenue, LY_Revenue, MTDGrowthPct
| order by MTDGrowthPct desc
| top 50 by MTDGrowthPct desc;
```

**ENTITY KEY MAP** (same as DECLINING section): dealers→kunrg/cname, brands→wgbez, products→matnr/arktx, depo→gsber, division→spart/spart_text, territory→Territory

---

### YTD (YEAR-TO-DATE) GROWTH CALCULATION — FISCAL YEAR APRIL → MARCH:
**TRIGGER**: User mentions "YTD", "year-to-date"

**DATE RULES**:
- Fiscal year: April 1 → March 31
- YTD window: FiscalYearStart → end of last **complete** month (NOT today)
- Do NOT use `now()` in WHERE clauses — use `AsOfDate`

**SCENARIO A — Overall YTD or single specific entity**:
Apply entity filter in BOTH CY and LY subqueries.
```kql
let FY_Start    = datetime(YYYY-04-01);                    // from CURRENT DATE CONTEXT
let AsOfDate    = startofmonth(now()) - 1d;                // end of last complete month
let PY_Start    = datetime_add('year', -1, FY_Start);
let PY_AsOfDate = datetime_add('year', -1, AsOfDate);

// For a specific entity, add: | where cname contains "X"  (or wgbez, gsber, spart_text, etc.)
let CYRevenue = toscalar({TABLE_NAME}
    | where fkdat between (FY_Start .. AsOfDate)
    [| where <entity_filter_if_specified>]
    | summarize sum(Revenue));
let LYRevenue = toscalar({TABLE_NAME}
    | where fkdat between (PY_Start .. PY_AsOfDate)
    [| where <entity_filter_if_specified>]
    | summarize sum(Revenue));

print
    YTDGrowth    = iff(LYRevenue == 0, real(null), (CYRevenue - LYRevenue) / LYRevenue * 100),
    CYRevenue    = CYRevenue,
    LYRevenue    = LYRevenue,
    ErrorMessage = iff(isnull(CYRevenue) or isnull(LYRevenue), "Error: missing data", ""),
    GrowthType   = iff(isnull(YTDGrowth), "N/A", iff(YTDGrowth > 0, "positive growth", "negative growth"));
```

**SCENARIO B — YTD growth ranked BY entity** (e.g., "YTD growth by dealer", "top brands by YTD growth"):
```kql
let FY_Start    = datetime(YYYY-04-01);
let AsOfDate    = startofmonth(now()) - 1d;
let PY_Start    = datetime_add('year', -1, FY_Start);
let PY_AsOfDate = datetime_add('year', -1, AsOfDate);

let CY = {TABLE_NAME}
| where fkdat between (FY_Start .. AsOfDate)
| summarize CY_Revenue = sum(Revenue), CY_Qty = sum(fkimg) by [entity_key], [entity_name_col];

let LY = {TABLE_NAME}
| where fkdat between (PY_Start .. PY_AsOfDate)
| summarize LY_Revenue = sum(Revenue) by [entity_key];

CY
| join kind=leftouter LY on [entity_key]
| extend YTDGrowthPct = iff(LY_Revenue == 0 or isnull(LY_Revenue), real(null),
                             (CY_Revenue - LY_Revenue) / LY_Revenue * 100)
| project [entity_name_col], [entity_key], CY_Revenue, LY_Revenue, YTDGrowthPct
| order by YTDGrowthPct desc
| top 50 by YTDGrowthPct desc;
```

**ENTITY KEY MAP**: dealers→kunrg/cname, brands→wgbez, products→matnr/arktx, depo→gsber, division→spart/spart_text, territory→Territory, sales zone→Szone

---

### SALES CONTRIBUTION ANALYSIS:
**TRIGGER**: "contribution of X", "contribution from X", "contribution by X"

**PATTERN**:
```kql
let StartDate = datetime(YYYY-MM-DD);
let EndDate   = datetime(YYYY-MM-DD);

let TotalRevenue = toscalar(
    {TABLE_NAME}
    | where fkdat between (StartDate .. EndDate)
    | summarize sum(Revenue)
);
let SegmentRevenue = toscalar(
    {TABLE_NAME}
    | where fkdat between (StartDate .. EndDate)
    | where [dimension_column] [operator] "[segment_value]"
    | summarize sum(Revenue)
);

print
    Segment         = "[segment_name]",
    TotalRevenue    = TotalRevenue,
    SegmentRevenue  = SegmentRevenue,
    ContributionPct = iff(TotalRevenue == 0, real(null), SegmentRevenue * 100.0 / TotalRevenue)
| extend Insight = strcat("Contribution: ", round(ContributionPct, 2), "%");
```

Dimension operators: brand → `wgbez contains "X"`, division → `spart_text contains "X"`, depo → `gsber == [numeric]`, dealer → `cname contains "X"`.

---

### AVERAGE SALES ANALYSIS:
**TRIGGER**: "average sales", "avg sales", "average revenue", "mean sales"

Two-step summarize (period first, then average). **Never use `extend` for period extraction** — do it inside `summarize by`:

```kql
let StartDate = datetime(YYYY-MM-DD);
let EndDate   = datetime(YYYY-MM-DD);

{TABLE_NAME}
| where fkdat between (StartDate .. EndDate)
| summarize TotalRevenue = sum(Revenue) by [entity_col,] startofmonth(fkdat)
| summarize AvgMonthlySales = avg(TotalRevenue) [by entity_col];
```

Granularity: monthly → `startofmonth(fkdat)`, weekly → `startofweek(fkdat)`, daily → `startofday(fkdat)`, yearly → `startofyear(fkdat)`.

---

### TREND ANALYSIS (PERIOD-OVER-PERIOD CHANGES):
**TRIGGER**: "trending", "uptrending", "downtrending", "falling trend", "rising trend", "monthly trend", "trend over time"

**DISAMBIGUATION — TREND vs DECLINING/GROWTH RANKING**:
- **TREND ANALYSIS** = time-series chart showing how revenue/quantity CHANGED OVER TIME (month-by-month or period-by-period pattern). Use when user wants to visualize direction over multiple periods.
- **DECLINING/GROWTH RANKING** = a ranked list showing WHICH entities declined or grew between two specific periods (e.g., this year vs last year). Use the DECLINING/NEGATIVE GROWTH section for those (trigger: "which dealers are declining", "top dealers by growth").
- When in doubt: if user asks "show me declining dealers" → DECLINING section. If user asks "show monthly revenue trend" → TREND ANALYSIS.

For **two periods** (e.g., from month A to month B):
- Compute revenue per entity per period
- GrowthPct = (Current - Previous) / Previous * 100
- Add TrendType: "up trend" or "down trend"
- **Down trend**: sort ascending by GrowthPct (most negative first)
- **Up trend**: sort descending by GrowthPct (most positive first)

For **multiple periods** (3+ months):
- Find top N entities by total revenue
- Return month-wise revenue: group by entity + startofmonth(fkdat)
- **NEVER use** `bin(fkdat, 1mo)` — **always use** `startofmonth(fkdat)`

---

### FINAL INSTRUCTION:
For ANY user query about SAP sales data:
1. Identify the query type:
   - Declining/growth ranking → use DECLINING / NEGATIVE GROWTH ANALYSIS section
   - MTD growth → use MTD section
   - YTD growth → use YTD section
   - Trend over time → use TREND ANALYSIS section
   - Contribution → use SALES CONTRIBUTION ANALYSIS section
   - Average → use AVERAGE SALES ANALYSIS section
   - Drop-off / at-risk dealers → use CUSTOMER DROP-OFF section
   - Multi-period individual → use MULTI-YEAR section
   - Otherwise → standard ranking/filtering/aggregation
2. **MANDATORY**: Use CURRENT DATE CONTEXT table (injected below this prompt) for ALL date resolution.
   - "this year" / "last year" / "MTD" / "YTD" / "last quarter" → ALWAYS from CURRENT DATE CONTEXT table.
   - NEVER use ago(), getyear(now()), startofyear(now()), startofmonth(now()) for these phrases.
3. Apply appropriate KQL pattern from the sections above
4. Enforce FIELD_MAPPINGS, GSBER_MAPPING, and DATA TYPE ENFORCEMENT
5. Handle edge cases (division by zero → iff(..., real(null), ...), null values, empty results)
6. Apply PERFORMANCE OPTIMIZATION (take / top limits)
7. For year-over-year quarter comparisons, always use same quarter one year earlier (NOT quarter-over-quarter).

Generate KQL that completely answers their business question using all available data dimensions and analytical capabilities.
"""
)

def _fiscal_quarter_dates(fq_idx: int, fy: int):
    """
    Return (start, end) for a fiscal quarter.
    fq_idx: 0=Q1(Apr-Jun), 1=Q2(Jul-Sep), 2=Q3(Oct-Dec), 3=Q4(Jan-Mar)
    fy: fiscal year start year (e.g. 2025 means FY Apr-2025 → Mar-2026)
    """
    fq_start_months = [4, 7, 10, 1]
    start_month = fq_start_months[fq_idx]
    start_year  = (fy + 1) if start_month < 4 else fy   # Q4 starts in Jan of fy+1
    q_start = datetime.date(start_year, start_month, 1)
    # End = start of next quarter - 1 day
    next_idx = (fq_idx + 1) % 4
    next_month = fq_start_months[next_idx]
    next_year  = (fy + 1) if next_month <= start_month else start_year
    if next_idx == 0:          # next is Q1 → Apr of fy+1
        next_year = fy + 1
    q_end = datetime.date(next_year, next_month, 1) - datetime.timedelta(days=1)
    return q_start, q_end


def _build_date_context() -> str:
    """Return a prompt block with today's date and resolved fiscal-year period strings."""
    now = datetime.datetime.now()
    today = now.date()
    today_str = today.strftime("%Y-%m-%d")

    # Fiscal year: April 1 → March 31
    fy_year = now.year if now.month >= 4 else now.year - 1
    fy_start      = datetime.date(fy_year, 4, 1)
    fy_end        = datetime.date(fy_year + 1, 3, 31)
    prev_fy_start = datetime.date(fy_year - 1, 4, 1)
    prev_fy_end   = datetime.date(fy_year, 3, 31)

    # Month helpers
    curr_month_start = today.replace(day=1)
    prev_month_last  = curr_month_start - datetime.timedelta(days=1)
    prev_month_first = prev_month_last.replace(day=1)
    ytd_end = prev_month_last

    # Simple helpers
    last_week_start = today - datetime.timedelta(days=7)
    yesterday       = today - datetime.timedelta(days=1)
    last_3m         = today - datetime.timedelta(days=90)
    last_6m         = today - datetime.timedelta(days=180)

    # ── Fiscal quarter logic ──────────────────────────────────────────
    # Which quarter is today in? (0=Q1 Apr-Jun, 1=Q2 Jul-Sep, 2=Q3 Oct-Dec, 3=Q4 Jan-Mar)
    fy_month_idx   = (now.month - 4) % 12         # 0=April … 11=March
    curr_fq_idx    = fy_month_idx // 3             # 0-3
    last_fq_idx    = (curr_fq_idx - 1) % 4        # previous quarter index

    # FY that the last complete quarter belongs to
    # If we wrapped back (curr Q1 → last Q4), the last quarter is from prev FY
    last_fq_fy = fy_year if last_fq_idx < curr_fq_idx else fy_year - 1

    curr_q_start, _ = _fiscal_quarter_dates(curr_fq_idx, fy_year)
    last_q_start, last_q_end         = _fiscal_quarter_dates(last_fq_idx, last_fq_fy)

    # Same quarter one year earlier (for YoY comparison)
    ly_last_q_start, ly_last_q_end   = _fiscal_quarter_dates(last_fq_idx, last_fq_fy - 1)
    # ─────────────────────────────────────────────────────────────────

    return f"""
### CURRENT DATE & FISCAL YEAR CONTEXT (MANDATORY — resolve ALL time references using these exact dates):
- **Today**: {today_str}
- **Fiscal Year Rule**: April 1 → March 31

**Period Translations** (use EXACTLY these datetime values in KQL):
| Natural language | KQL StartDate | KQL EndDate |
|---|---|---|
| "this year" / "current year" | datetime({fy_start}) | datetime({today_str}) |
| "last year" / "previous year" | datetime({prev_fy_start}) | datetime({prev_fy_end}) |
| "this month" | datetime({curr_month_start}) | datetime({today_str}) |
| "last month" | datetime({prev_month_first}) | datetime({prev_month_last}) |
| "last week" | datetime({last_week_start}) | datetime({today_str}) |
| "today" | datetime({today_str}) | datetime({today_str}) |
| "yesterday" | datetime({yesterday}) | datetime({yesterday}) |
| "YTD" / "year to date" | datetime({fy_start}) | datetime({ytd_end}) |
| "MTD" / "month to date" | datetime({prev_month_first}) | datetime({prev_month_last}) |
| "last 3 months" | datetime({last_3m}) | datetime({today_str}) |
| "last 6 months" | datetime({last_6m}) | datetime({today_str}) |
| "this quarter" | datetime({curr_q_start}) | datetime({today_str}) |
| "last quarter" (current period) | datetime({last_q_start}) | datetime({last_q_end}) |
| "last quarter" (previous year — for YoY) | datetime({ly_last_q_start}) | datetime({ly_last_q_end}) |

**QUARTER COMPARISON RULE (CRITICAL)**:
When comparing "last quarter" for declining/growth analysis, ALWAYS use year-over-year:
- CY period = datetime({last_q_start}) to datetime({last_q_end})  ← last complete quarter
- PY period = datetime({ly_last_q_start}) to datetime({ly_last_q_end})  ← same quarter, 1 year ago
Do NOT compare last quarter vs the quarter before that (quarter-over-quarter) — use the same quarter last year.

**Default (no period mentioned)**: CY_Start=datetime({fy_start}), CY_End=datetime({today_str}), PY_Start=datetime({prev_fy_start}), PY_End=datetime({prev_fy_end})

"""

# ───────────────────────── 3.  LLM instance ────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    # api_version      = "2025-01-01-preview",
    api_version      = "2024-12-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    # temperature      = 0,
    temperature      = 1,
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


def _enforce_bukrs_filter(kql: str) -> str:
    """
    Safety net: inject | where bukrs == 1000 immediately after every table scan
    that does not already have it as the first filter.

    Handles the common pattern:
        SAPSalesInfos          ← table reference (end of line)
        | where fkdat ...      ← first filter (not bukrs) → inject bukrs before this

    Leaves alone if bukrs == 1000 is already the first filter.
    """
    pattern = re.compile(
        r'(' + re.escape(TABLE_NAME) + r'\b)'   # table name
        r'((?:[ \t]*\n)+)'                       # newline(s) after table name
        r'([ \t]*\|[ \t]*where[ \t]+)'           # next pipe-where
        r'(?!bukrs\b)',                           # only if NOT already bukrs
        re.MULTILINE,
    )
    fixed = pattern.sub(r'\1\2| where bukrs == 1000\n\3', kql)
    if fixed != kql:
        print("[bukrs-guard] injected missing bukrs == 1000 filter")
    return fixed


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
    divisions: List[str] = field(default_factory=list)

def get_user_area_scope(user) -> UserAreaScope:
    print(">>> get_user_area_scope user:", user, "| is_authenticated:", getattr(user, "is_authenticated", None))
    if _is_admin(user):
        print(">>> user is admin/superadmin; unrestricted scope")
        return UserAreaScope([], [], [], restricted=False)

    depots = list(UserDepoMap.objects.filter(user=user).values_list("depo__code", flat=True))
    zones = list(UserZoneMap.objects.filter(user=user).values_list("zone__code", flat=True))
    territories = list(UserTerritoryMap.objects.filter(user=user).values_list("territory__code", flat=True))
    divisions = list(UserDivisionMap.objects.filter(user=user).values_list("division__code", flat=True))
    print(f">>> resolved scope depots={depots} zones={zones} territories={territories} divisions={divisions}")

    return UserAreaScope(depots=depots, zones=zones, territories=territories, restricted=True, divisions=divisions)
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

# def is_sales_analysis_query(user_req: str, *, conversation_id: str | None = None) -> bool:
#     print("dsadsad")
#     """
#     Uses the LLM model to determine if the user's request is related to SAP sales data analysis and KQL generation.
#     If a conversation_id is provided, include the last 20 messages as additional context.
#     """

#     # Base instruction
#     prompt = (
#         "You are an expert SAP Sales Analysis Assistant.\n"
#         f'The user has sent the following request: "{user_req}".\n\n'
#         "Please determine if the request is related to SAP sales analysis, such as sales reports, revenue analysis, "
#         "growth calculations, or KQL generation.\n"
#         'If the request is about SAP sales data analysis, return "yes". If the request is a general question, '
#         'unrelated to sales analysis, return "no".'
#     )
    

#     # Append last 20 messages if conversation_id is provided
#     if conversation_id:
#         try:
#             conv_id = get_conversation_id_from_uuid(conversation_id)
#             last_msgs = get_last_20_messages(conv_id) or []
#             if last_msgs:
#                 prompt += "\n\nRecent conversation (last 20 messages):\n"
#                 for m in last_msgs[-20:]:
#                     role = "USER" if m.sender == "user" else "ASSISTANT"
#                     text = (m.text or "").strip()
#                     # keep it compact to avoid hitting context limits
#                     if len(text) > 400:
#                         text = text[:400] + "..."
#                     prompt += f"{role}: {text}\n"
#             else:
#             # ✅ Minimal addition: explicitly mark that this is the first message (no prior context)
#                 prompt += "\n\nNote: This is the first message in this conversation (no prior context)."
#         except Exception:
#             # fail-open: just proceed without context if anything goes wrong
#             pass
#     else:
#         #  Also handle when no conversation_id is passed at all (brand-new chat)
#         prompt += "\n\nNote: This is the first message in this conversation (no prior context).return yes"
    
#     print("from check ",prompt)
#     response = llm.invoke([{"role": "user", "content": prompt}]).content.strip()
#     print("from check ",response)
#     return response.lower() == "yes"

def is_sales_analysis_query(user_req: str, *, conversation_id: str | None = None) -> bool:
    print("dsadsad")
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
                    if len(text) > 400:
                        text = text[:400] + "..."
                    prompt += f"{role}: {text}\n"
            else:
                # First message within this conversation
                prompt += "\n\nNote: This is the first message in this conversation (no prior context)."
        except Exception:
            pass
    else:
        # Brand-new chat (no conversation_id at all)
        prompt += "\n\nNote: This is the first message in this conversation (no prior context)."

    print("from check ", prompt)

    # --- Only change: add a SYSTEM message + deterministic params ---
    messages = [
        {"role": "system",
         "content": (
             "You are a STRICT binary classifier for SAP Sales queries. "
             "Answer with exactly 'yes' or 'no' in lowercase, no punctuation. "
             "If the request is ambiguous or could plausibly refer to SAP sales, answer 'yes'."
         )},
        {"role": "user", "content": prompt},
    ]
    try:
        resp_obj = llm.invoke(messages, temperature=0, max_output_tokens=3)
    except TypeError:
        resp_obj = llm.invoke(messages)

    response = getattr(resp_obj, "content", resp_obj)
    response = (response or "").strip().lower()
    print("from check ", response)

    # Minimal normalization (no heuristics): accept "yes", "yes." etc.
    return response.startswith("yes")

#helper for retricted acces

def _kql_error(meta: dict, message: str) -> str:
    """Return strict-format output with a single KQL error line."""
    return "// META " + json.dumps(meta, separators=(",", ":"), ensure_ascii=False) + \
           f"\nprint ErrorMessage = '{message}';"

def _normalize_depots(raw):
    depots_num = []
    for v in list(raw or []):
        try:
            depots_num.append(int(str(v).strip()))
        except Exception:
            pass  # silently ignore non-numeric
    return depots_num

def _normalize_divisions(raw):
    divs_num = []
    for v in list(raw or []):
        try:
            divs_num.append(int(str(v).strip()))
        except Exception:
            pass
    return divs_num

def _parse_explicit_division_filters(user_req: str, division_mapping: dict) -> set:
    """Extract explicitly requested division codes from the user text."""
    divs_req = set()

    # numeric division code (2-digit, avoid years)
    for m in re.finditer(r'\b(?:division|div|spart)\s*(?:is|=|:)?\s*(\d{1,2})\b', user_req, flags=re.I):
        try:
            divs_req.add(int(m.group(1)))
        except Exception:
            pass

    # textual division name via mapping (e.g., "Decorative", "Industrial Paints")
    low = user_req.lower()
    for name, code in division_mapping.items():
        if re.search(rf'\b{re.escape(name.lower())}\b', low):
            try:
                divs_req.add(int(code))
            except Exception:
                pass

    return divs_req

def _parse_explicit_area_filters(user_req: str, gsber_mapping: dict[str, str]):
    """Extract explicitly requested depots/zones/territories from the user text."""
    depots_req, zones_req, terr_req = set(), set(), set()

    # numeric depo when mentioned as depo/business area/gsber  (avoid years like 2025)
    for m in re.finditer(r'\b(?:depo|depot|business\s*area|gsber)\s*(?:is|=|:)?\s*(\d{4})\b', user_req, flags=re.I):
        try:
            depots_req.add(int(m.group(1)))
        except Exception:
            pass

    # textual depo via mapping (e.g., "Dhaka South")
    low = user_req.lower()
    for name, code in gsber_mapping.items():
        if re.search(rf'\b{re.escape(name.lower())}\b', low):
            try:
                depots_req.add(int(code))
            except Exception:
                pass

    # territory like "territory I04"
    for m in re.finditer(r'\b(?:territory|terr|ter)\s*(?:is|=|:)?\s*([A-Za-z0-9._-]+)\b', user_req, flags=re.I):
        terr_req.add(m.group(1))

    # zone like "zone Z010" or "szone Z003"
    for m in re.finditer(r'\b(?:s?zone)\s*(?:is|=|:)?\s*([A-Za-z0-9._-]+)\b', user_req, flags=re.I):
        zones_req.add(m.group(1))

    return depots_req, zones_req, terr_req

def _build_mandatory_where(_colmap, depots_num, terr_list, zones_list, divs_list=None) -> str:
    """Build the exact where-clause to inject after the table, supporting multiple values."""
    parts = []
    depots_num = sorted(set(int(x) for x in depots_num))
    terr_list  = sorted({str(t) for t in (terr_list or [])}, key=str.lower)
    zones_list = sorted({str(z) for z in (zones_list or [])}, key=str.lower)
    divs_list  = sorted(set(int(x) for x in (divs_list or [])))

    if depots_num:
        if len(depots_num) == 1:
            parts.append(f"{_colmap['depo']['col']} == {depots_num[0]}")
        else:
            parts.append(f"{_colmap['depo']['col']} in ({', '.join(map(str, depots_num))})")

    if terr_list:
        parts.append(f"{_colmap['territory']['col']} in~ ({', '.join(json.dumps(t) for t in terr_list)})")

    if zones_list:
        parts.append(f"{_colmap['zone']['col']} in~ ({', '.join(json.dumps(z) for z in zones_list)})")

    if divs_list and "division" in _colmap:
        if len(divs_list) == 1:
            parts.append(f"{_colmap['division']['col']} == {divs_list[0]}")
        else:
            parts.append(f"{_colmap['division']['col']} in ({', '.join(map(str, divs_list))})")

    return " | where " + " and ".join(parts) if parts else ""


#END 
def _parse_dates_for_meta(user_req: str):
    m = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', user_req)
    return {"start": m.group(1), "end": m.group(2)} if m else None

def _ci_set(values):
    """case-insensitive set of strings"""
    return {str(v).strip().lower() for v in (values or [])}


def generate_kql(user_req: str, conversation_uuid: Optional[str] = None, strict=False) -> str:
    global LAST_KQL_META

    # ── 1. Static system knowledge + dynamic date context ──
    prompt = SYSTEM_PROMPT_KQL
    prompt += _build_date_context()
    prompt += build_schema_prompt_block()

    # ── 2. Multi-turn conversation context ──
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
    # user access 
    # user access 
    # ====================== user access (REPLACE THIS BLOCK) ======================
    _scope = None
    try:
         
        _user = get_current_chat_user()
        print(">>> agent current_user:", _user, "| id:", getattr(_user, "id", None))
        print(_user)
        _scope = get_user_area_scope(_user) if _user else None
        print(">>> generate_kql scope:", _scope)
    except Exception:
        _scope = None

    try:
        _colmap = getattr(settings, "ADX_AREA_COLUMNS", {
            "depo":      {"col": "gsber",     "type": "long"},
            "zone":      {"col": "Szone",     "type": "string"},
            "territory": {"col": "Territory", "type": "string"},
            "division":  {"col": "spart",     "type": "long"},
        })

        # Unrestricted (admin/is_staff/Admin-group/BetaUser) → no scoping
        if not _scope or not getattr(_scope, "restricted", False):
            prompt += (
                '\n\nUSER_AREA_SCOPE (JSON): {"restricted": false}\n'
                "If restricted=false, do NOT add any area filters.\n"
            )

        else:
            # Restricted: depo is mandatory; support MULTIPLE depots/territories/zones/divisions
            depots_num = _normalize_depots(getattr(_scope, "depots", []))
            zones_list = list(getattr(_scope, "zones", []) or [])
            terr_list  = list(getattr(_scope, "territories", []) or [])
            divs_list  = _normalize_divisions(getattr(_scope, "divisions", []))

            if not depots_num:
                meta = {"restricted": True,
                        "filters": {"gsber": [], "Szone": zones_list, "Territory": terr_list, "spart": divs_list},
                        "dates": None}
                return _kql_error(meta, "no depo is assigned.")

            # Block ONLY when user explicitly asks for out-of-scope area(s)
            req_depos, req_zones, req_terr = _parse_explicit_area_filters(user_req, GSBER_MAPPING)
            req_divs = _parse_explicit_division_filters(user_req, DIVISION_MAPPING)
            dates_meta = _parse_dates_for_meta(user_req)

            if req_depos and not set(req_depos).issubset(set(_normalize_depots(depots_num))):
                meta = {"restricted": True, "filters": {}, "dates": dates_meta}
                return _kql_error(meta, "sorry you have no authorized to view this data.")

            if req_terr and terr_list and not _ci_set(req_terr).issubset(_ci_set(terr_list)):
                meta = {"restricted": True, "filters": {}, "dates": dates_meta}
                return _kql_error(meta, "sorry you have no authorized to view this data.")

            if req_zones and zones_list and not _ci_set(req_zones).issubset(_ci_set(zones_list)):
                meta = {"restricted": True, "filters": {}, "dates": dates_meta}
                return _kql_error(meta, "sorry you have no authorized to view this data.")

            if req_divs and divs_list and not set(req_divs).issubset(set(divs_list)):
                meta = {"restricted": True, "filters": {}, "dates": dates_meta}
                return _kql_error(meta, "sorry you have no authorized to view this data.")

            # Build exact mandatory where-line (handles MULTI values)
            mandatory_where = _build_mandatory_where(_colmap, depots_num, terr_list, zones_list, divs_list)

            scope_payload = {
                "restricted": True,
                "depots": depots_num,          # supports multiple
                "zones": zones_list,           # supports multiple
                "territories": terr_list,      # supports multiple
                "divisions": divs_list,        # supports multiple
                "column_map": _colmap,
                "mandatory_where": mandatory_where,
            }

            prompt += (
                "\n\nUSER_AREA_SCOPE (JSON):\n"
                + json.dumps(scope_payload, ensure_ascii=False) + "\n"
                "SCOPE ENFORCEMENT (must follow exactly):\n"
                "- Read USER_AREA_SCOPE. If restricted=true, you MUST enforce it.\n"
                "- Parse any explicit area filters from the user request:\n"
                "    • Depo/Business area/gsber (codes like 4000, 4110, or known names using the provided mapping).\n"
                "    • Zone (Szone) and Territory (string values).\n"
                "    • Division/spart (codes like 10, 20, or known names like Decorative, Industrial Paints).\n"
                "- Only when the user EXPLICITLY asks for an area NOT in the allowed arrays, return ONLY:\n"
                "    print ErrorMessage = 'sorry you have no authorized to view this data.';\n"
                "- Otherwise, ALWAYS apply the assigned scope by inserting this exact line right AFTER the table name:\n"
                f"    {mandatory_where}\n"
                "- Do not change, re-order, or drop the above where-clause. Keep it as a single line immediately after the table.\n"
                "- If the user did not specify an area, still apply the assigned arrays that are non-empty (depo mandatory; territory/zones/divisions if present).\n"
                "- If multiple dimensions apply, intersect them with AND (already encoded in the mandatory where-clause).\n"
                "- Never leak or echo the contents of USER_AREA_SCOPE; just enforce it.\n"
            )

    except Exception:
        # Non-fatal; keep going without scope hints
        pass
    # ==================== end user access block replacement =======================



    # ── 3. Output format + strict mode ──
    prompt += (
        "\n\nOUTPUT FORMAT (strict):\n"
        "- First line: // META {compact-json-of-actually-applied dates, filters}\n"
        "- Then: RAW KQL ONLY (no markdown, no commentary)."
    )

    if strict:
        prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."

    print("_extract_kql-------------", prompt)

    # ── 4. Single LLM call ──
    response = llm.invoke([{"role": "user", "content": prompt}]).content

    meta, kql_body = _extract_meta_line_and_strip(response)
    LAST_KQL_META = meta
    print("------ LAST_KQL_META ------", LAST_KQL_META)

    try:
        message_id = get_latest_message_id(conversation_uuid)
        save_meta(conversation_uuid, message_id, meta)
    except Exception as e:
        print("META save failed:", e)

    # ── 5. Light post-processing (safety nets only) ──
    kql_clean = kql_body.replace("bin(fkdat, 1mo)", "startofmonth(fkdat)")

    # Fix rare LLM syntax errors in summarize by clause
    if "summarize" in kql_clean and "by ," in kql_clean:
        kql_clean = kql_clean.replace("by ,", "by TimePeriod")
    if "summarize" in kql_clean and ", )" in kql_clean:
        kql_clean = kql_clean.replace(", )", ", TimePeriod)")

    kql_clean = _enforce_bukrs_filter(kql_clean)
    print("response from generate kql", kql_clean)
    return _extract_kql(kql_clean)


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
    """Wrap bare YYYY-MM-DD strings in datetime() — skip ones already inside datetime()."""
    # Negative lookbehind: only wrap dates NOT already preceded by 'datetime('
    return re.sub(r'(?<!datetime\()(\d{4}-\d{2}-\d{2})', r'datetime(\1)', kql_query)


def _fmt_date(d: str) -> str:
    """Format YYYY-MM-DD as '14 Mar 2026'."""
    try:
        return datetime.datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    except Exception:
        return d


def _extract_kql_period_context(kql: str) -> str:
    """
    Return a human-readable period string extracted from the KQL.

    Priority 1 — let-variable pattern (most explicit):
        let CY_Start = datetime(2026-03-01); ...
    Priority 2 — inline between (datetime(X) .. datetime(Y))
    Priority 3 — fallback: min/max of all datetime() values
    """
    # Priority 1: let CY_Start / CY_End / PY_Start / PY_End
    let_pat = re.compile(
        r'let\s+(CY_Start|CY_End|PY_Start|PY_End)\s*=\s*datetime\((\d{4}-\d{2}-\d{2})\)',
        re.IGNORECASE,
    )
    let_vars = {m.group(1).upper(): m.group(2) for m in let_pat.finditer(kql)}

    if {"CY_START", "CY_END", "PY_START", "PY_END"}.issubset(let_vars):
        return (
            f"Current Period : {_fmt_date(let_vars['CY_START'])} to {_fmt_date(let_vars['CY_END'])}\n"
            f"Previous Period: {_fmt_date(let_vars['PY_START'])} to {_fmt_date(let_vars['PY_END'])}"
        )
    if {"CY_START", "CY_END"}.issubset(let_vars):
        return f"Period: {_fmt_date(let_vars['CY_START'])} to {_fmt_date(let_vars['CY_END'])}"

    # Priority 2: between (datetime(X) .. datetime(Y))
    between_pat = re.compile(
        r'between\s*\(\s*datetime\((\d{4}-\d{2}-\d{2})\)\s*\.\.\s*datetime\((\d{4}-\d{2}-\d{2})\)\s*\)',
        re.IGNORECASE,
    )
    pairs = list(dict.fromkeys(between_pat.findall(kql)))

    if len(pairs) == 1:
        s, e = pairs[0]
        return f"Period: {_fmt_date(s)} to {_fmt_date(e)}"
    if len(pairs) == 2:
        sorted_pairs = sorted(pairs, key=lambda p: p[0])
        py_s, py_e = sorted_pairs[0]
        cy_s, cy_e = sorted_pairs[1]
        return (
            f"Current Period : {_fmt_date(cy_s)} to {_fmt_date(cy_e)}\n"
            f"Previous Period: {_fmt_date(py_s)} to {_fmt_date(py_e)}"
        )
    if len(pairs) > 2:
        return "\n".join(
            f"Period {i+1}: {_fmt_date(s)} to {_fmt_date(e)}" for i, (s, e) in enumerate(pairs)
        )

    # Priority 3: all datetime() values → min/max
    all_dates = sorted(set(re.findall(r'datetime\((\d{4}-\d{2}-\d{2})\)', kql)))
    if len(all_dates) >= 2:
        return f"Period: {_fmt_date(all_dates[0])} to {_fmt_date(all_dates[-1])}"
    if all_dates:
        return f"Date: {_fmt_date(all_dates[0])}"
    return ""


# Detect trend from user prompt (e.g., increasing, declining, etc.)
def detect_trend(user_prompt: str) -> str:
    if any(word in user_prompt.lower() for word in ["declining", "downtrending", "negative growth", "falling", "decrease"]):
        return "declining"
    elif any(word in user_prompt.lower() for word in ["increasing", "uptrending", "positive growth", "rising", "growth"]):
        return "increasing"
    else:
        return "stable"


#for insight
def _reverse_gsber_name(code: int) -> str:
    """Return human name for a gsber code using GSBER_MAPPING; empty string if not found."""
    try:
        for name, val in GSBER_MAPPING.items():
            if str(val) == str(code):
                return name
    except Exception:
        pass
    return ""

def _build_scope_title_and_insight(user) -> tuple[str, str]:
    """
    Returns (title_suffix, insight_note).
    - title_suffix: text to append to the title if scope is small enough.
    - insight_note: bullet sentence to include at the end of insights.
    If user is unrestricted or no depo assigned, returns ("","").
    """
    try:
        scope = get_user_area_scope(user)
    except Exception:
        scope = None

    if not scope or not getattr(scope, "restricted", False):
        return "", ""

    # Normalize depots to ints
    depots = []
    for v in (scope.depots or []):
        try:
            depots.append(int(str(v).strip()))
        except Exception:
            pass

    if not depots:
        # Restricted but no depo — let upstream logic handle the error path.
        return "", ""

    # Build human-friendly pieces
    depots_parts = []
    for d in sorted(set(depots)):
        nm = _reverse_gsber_name(d)
        depots_parts.append(f"{d}" + (f" ({nm})" if nm else ""))

    terr_parts  = [str(t) for t in sorted(set(scope.territories or []), key=str.lower)]
    zones_parts = [str(z) for z in sorted(set(scope.zones or []), key=str.lower)]

    pieces = []
    if depots_parts:
        pieces.append("Depo/Sales Office: " + ", ".join(depots_parts))
    if terr_parts:
        pieces.append("Territory: " + ", ".join(terr_parts))
    if zones_parts:
        pieces.append("Zone: " + ", ".join(zones_parts))

    scope_text = "; ".join(pieces)
    if not scope_text:
        return "", ""

    # Heuristic: if total scoped items is small (<= 3), show in title too
    item_count = len(depots_parts) + len(terr_parts) + len(zones_parts)
    show_in_title = item_count <= 3

    title_suffix = scope_text if show_in_title else ""
    insight_note = f"Note: Results are limited to your access scope — {scope_text}."

    return title_suffix, insight_note


#end 

# Handle user queries dynamically and generate the corresponding KQL query

def handle_user_query(user_prompt: str, *, conversation_id: str | None = None, user: Optional["User"] = None,) -> str:
    """
    Dynamically handle SAP Sales prompts with multi-turn conversation support,
    ensuring correct KQL generation, and mapping business area/territory to the correct 'gsber' code.
    Continuity is driven by conversation history only (no META reuse).
    """

    # -----------------------------
    # 1) Non-sales queries → general assistant
    # -----------------------------
    print("calll llmmm")
    if not is_sales_analysis_query(user_prompt, conversation_id=conversation_id):
        general_prompt = """
        You are a SAP Sales Analysis Assistant. The user has asked a general question not related to sales data analysis or KQL.

        Please respond as a friendly and helpful SAP Sales Analysis Assistant. Let the user know:
        - You are specialized in SAP sales data analysis
        - You can help with sales reports, revenue analysis, growth calculations, trends, etc.
        - Invite them to ask sales-related queries

        Do not generate KQL for general questions.
        """.strip()

        # Show previous data in USER/ASSISTANT mode, then the current ask
        if conversation_id:
            try:
                conv_id = get_conversation_id_from_uuid(conversation_id)
                last_msgs = get_last_20_messages(conv_id)
                if last_msgs:
                    history_block = "Previous Conversation (for context only):\n"
                    for m in last_msgs[-20:]:
                        role = "USER" if m.sender == "user" else "ASSISTANT"
                        history_block += f"{role}: {m.text or ''}\n"
                    general_prompt += "\n\n" + history_block
            except Exception:
                pass

        general_prompt += f"\n\nNow, CURRENT USER MESSAGE:\nUSER: {user_prompt}"

        general_messages = [
            {
                "role": "system",
                "content": "Do not reuse numbers or conclusions from Previous Conversation; answer the current question directly. Do not generate KQL for general questions."
            },
            {"role": "user", "content": general_prompt},
        ]
        return llm.invoke(general_messages).content

    # -----------------------------
    # 2) Sales queries → generate KQL
    # (date/period resolution handled by _build_date_context() inside generate_kql)
    # -----------------------------
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
    # 4) Execute query (with retry)
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
    # Build a history block in USER/ASSISTANT mode (previous data)
    history_block = ""
    if conversation_id:
        try:
            conv_id = get_conversation_id_from_uuid(conversation_id)
            last_msgs = get_last_20_messages(conv_id)
            if last_msgs:
                history_block = "Previous Conversation (for context only):\n"
                for m in last_msgs[-20:]:
                    role = "USER" if m.sender == "user" else "ASSISTANT"
                    history_block += f"{role}: {m.text or ''}\n"
        except Exception:
            pass
    #scope
    # --- scope-aware title & insight additions (only if user is RESTRICTED) ---
    try:
        _user_for_scope = get_current_chat_user()
    except Exception:
        _user_for_scope = None

    # Check unrestricted first (is_superuser / is_staff / admin / BetaUser handled in get_user_area_scope)
    try:
        _scope_for_prompt = get_user_area_scope(_user_for_scope) if _user_for_scope else None
        _is_unrestricted = bool(_scope_for_prompt and getattr(_scope_for_prompt, "restricted", False) is False)
    except Exception:
        _scope_for_prompt = None
        _is_unrestricted = False

    if _is_unrestricted:
        # Admins & BetaUser: do NOT inject scope into title/insights
        title_suffix = ""
        insight_note = ""
    else:
        # Restricted users: build human-readable scope text
        title_suffix, insight_note = _build_scope_title_and_insight(_user_for_scope)
    #end scope

    period_context = _extract_kql_period_context(kql)

    result_prompt = (
        (history_block + "\n" if history_block else "")
        + "Now, CURRENT USER MESSAGE:\n"
        + f"USER: {user_prompt}\n\n"
        + "Context Data (use ONLY this JSON for any numbers):\n"
        + f"{result_json}\n\n"
        + (f"Analysis Period (MUST mention this clearly in the response):\n{period_context}\n\n" if period_context else "")
        + "Format the output in bulleted format.\n"
        + "After decimal take upto two places. Example: 1253.89"
        + "- Begin with a concise Title for the result.\n"
        + (f"- If helpful (scope is small), append this to the Title: \"{title_suffix}\".\n" if title_suffix else "")
        + "- Amount is in BDT and Volume is in gallons.\n"
        + "- Replace 'gsber' with 'Depo/Sales Office'.\n"
        + "- If 'vtweg' is found, show:\n"
        + "    - '10' → Dealer (10)\n"
        + "    - '20' → Customer (20)\n"
        + "    - '30' → Project Customer (30)\n"
        + "- Use bullet points for both numerical and categorical results.Highlight the names where it needed.\n\n"
        + (f"- Add a final bullet in Insights: \"{insight_note}\"\n" if insight_note else "")
        + "Then generate short Insights on [context]. \n"
    )

    # -----------------------------
    # 8) Generate narrative output
    # -----------------------------
    messages = [
        {
            "role": "system",
            "content": (
                "You are a SAP Sales Data Analyst. For ALL numeric facts, use ONLY the JSON under 'Context Data'. "
                "If any part of Previous Conversation conflicts with 'Context Data', ignore it. "
                "Do not reuse headings or numbers from earlier assistant messages."
            ),
        },
        {"role": "user", "content": result_prompt},
    ]
    return llm.invoke(messages).content

    # try:
    #     return analysis_llm.invoke([{"role": "user", "content": result_prompt}]).content
    # except BadRequestError as e:
    #     # Azure returns a JSON body with details (e.g., context length, invalid param, etc.)
    #     try:
    #         err_json = e.response.json()
    #     except Exception:
    #         err_json = {"message": str(e)}
    #     return f"Azure OpenAI 400 Bad Request.\nDetails: {err_json}"

