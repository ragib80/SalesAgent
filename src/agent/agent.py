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
    "division":"spart_text","division code":"spart","company code":"bukrs","sales org":"vkorg",
    "dist channel":"vtweg","distribution channel":"vtweg","channel":"vtweg","business area":"gsber","depo":"gsber",
    "credit control area":"kkber","Dealer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL",
    "product code":"matnr","material code":"matnr","material code":"meins","volume unit":"voleh","business group":"GK",
    "territory":"Territory","sales zone":"Szone","zone":"Szone","date":"fkdat","Dealer Code":"kunrg","dealer code":"kunrg",
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

VTWEG_MAPPING = {
    "Dealer": "10",
    "Customer": "20",
    "Project Customer": "30"
}

VTWEG_MAPPING_STR = "\n".join(f'"{k}": "{v}"' for k, v in VTWEG_MAPPING.items())

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

**Comparison Analysis**:
```kql
let CurrentYear = getyear(now());
let PreviousYear = CurrentYear - 1;
{TABLE_NAME}
| where getyear(fkdat) in (CurrentYear, PreviousYear)
| extend Year = getyear(fkdat)
| summarize Revenue = sum(Revenue) by Year, cname
| order by Year desc, Revenue desc
| take 40;
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

CONTRIBUTION_RE = re.compile(
    r'\b(contribution (?:of|from|by)|% contribution|percentage contribution|what (?:percent\w*|%)|did\b.*\bcontribute)\b',
    re.IGNORECASE
)
AVG_SALES_RE = re.compile(
    r'\b(?:average|avg|mean)[ -]?(?:sales|revenue|amount|quantity|volume)?\b',
    re.IGNORECASE
)

# lower-case keys for matching
FIELD_MAP_LOWER = {k.lower(): v for k, v in FIELD_MAPPINGS.items()}

TREND_RE     = re.compile(r'\b(?:up[- ]?trending|trending)\b', re.IGNORECASE)
DOWN_TREND_RE = re.compile(r'\b(?:down[- ]?trending|downtrend|negative trend|falling|declining|decreasing)\b', re.IGNORECASE) 
EXCLUDE_KEYS = {"revenue", "sale", "quantity", "volume", "date", "fkdat"}

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

def _build_mandatory_where(_colmap, depots_num, terr_list, zones_list) -> str:
    """Build the exact where-clause to inject after the table, supporting multiple values."""
    parts = []
    depots_num = sorted(set(int(x) for x in depots_num))
    terr_list  = sorted({str(t) for t in (terr_list or [])}, key=str.lower)
    zones_list = sorted({str(z) for z in (zones_list or [])}, key=str.lower)

    if depots_num:
        if len(depots_num) == 1:
            parts.append(f"{_colmap['depo']['col']} == {depots_num[0]}")
        else:
            parts.append(f"{_colmap['depo']['col']} in ({', '.join(map(str, depots_num))})")

    if terr_list:
        parts.append(f"{_colmap['territory']['col']} in~ ({', '.join(json.dumps(t) for t in terr_list)})")

    if zones_list:
        parts.append(f"{_colmap['zone']['col']} in~ ({', '.join(json.dumps(z) for z in zones_list)})")

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
    # user access 
    # user access
    # ====================== user access (REPLACE THIS BLOCK) ======================
    _scope = None
    _mandatory_where_clause = ""
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
        })

        # Unrestricted (admin/is_staff/Admin-group/BetaUser) → no scoping
        if not _scope or not getattr(_scope, "restricted", False):
            prompt += (
                '\n\nUSER_AREA_SCOPE (JSON): {"restricted": false}\n'
                "If restricted=false, do NOT add any area filters.\n"
            )

        else:
            # Restricted: depo is mandatory; support MULTIPLE depots/territories/zones
            depots_num = _normalize_depots(getattr(_scope, "depots", []))
            zones_list = list(getattr(_scope, "zones", []) or [])
            terr_list  = list(getattr(_scope, "territories", []) or [])

            if not depots_num:
                meta = {"restricted": True,
                        "filters": {"gsber": [], "Szone": zones_list, "Territory": terr_list},
                        "dates": None}
                return _kql_error(meta, "no depo is assigned.")

            # Block ONLY when user explicitly asks for out-of-scope area(s)
            req_depos, req_zones, req_terr = _parse_explicit_area_filters(user_req, GSBER_MAPPING)
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

            # Build exact mandatory where-line (handles MULTI values)
            mandatory_where = _build_mandatory_where(_colmap, depots_num, terr_list, zones_list)
            _mandatory_where_clause = mandatory_where

            scope_payload = {
                "restricted": True,
                "depots": depots_num,          # supports multiple
                "zones": zones_list,           # supports multiple
                "territories": terr_list,      # supports multiple
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
                "- Only when the user EXPLICITLY asks for an area NOT in the allowed arrays, return ONLY:\n"
                "    print ErrorMessage = 'sorry you have no authorized to view this data.';\n"
                "- Otherwise, ALWAYS apply the assigned scope by inserting this exact line right AFTER the table name:\n"
                f"    {mandatory_where}\n"
                "- Do not change, re-order, or drop the above where-clause. Keep it as a single line immediately after the table.\n"
                "- If the user did not specify an area, still apply the assigned arrays that are non-empty (depo mandatory; territory/zones if present).\n"
                "- If multiple dimensions apply, intersect them with AND (already encoded in the mandatory where-clause).\n"
                "- Never leak or echo the contents of USER_AREA_SCOPE; just enforce it.\n"
            )

    except Exception:
        # Non-fatal; keep going without scope hints
        pass
    # ==================== end user access block replacement =======================



    # ── Pre-compute concrete dates so the LLM never has to calculate them ──
    _now = datetime.datetime.now()
    # AsOfDate = last day of the most recently completed month
    _asof = _now.replace(day=1) - datetime.timedelta(days=1)
    _asof_str = _asof.strftime("%Y-%m-%d")
    # LY AsOfDate = same month/day last year (handle Feb-28/29 edge via calendar)
    _ly_asof_month_days = calendar.monthrange(_asof.year - 1, _asof.month)[1]
    _ly_asof_day = min(_asof.day, _ly_asof_month_days)
    _ly_asof = _asof.replace(year=_asof.year - 1, day=_ly_asof_day)
    _ly_asof_str = _ly_asof.strftime("%Y-%m-%d")
    # Default MTD month = last fully completed calendar month
    _mtd_cy_start = _asof.replace(day=1).strftime("%Y-%m-%d")
    _mtd_cy_end   = _asof_str
    _mtd_ly_start = _ly_asof.replace(day=1).strftime("%Y-%m-%d")
    _mtd_ly_end   = _ly_asof_str
    # Current fiscal year: April 1 → March 31
    _fy_start_year = _now.year - 1 if _now.month < 4 else _now.year
    _fy_start_str  = f"{_fy_start_year}-04-01"
    # LY fiscal year start
    _ly_fy_start_str = f"{_fy_start_year - 1}-04-01"

    # Detect if the user is asking for MTD sales or growth
    # if "MTD" in user_req or "Month-to-Date" in user_req:
    if MTD_RE.search(user_req):
        prompt += f"""
        Instruction — MTD (Month-to-Date):
        - Identify the target month from the user request.
        - If no specific month is given, default to the last fully completed month:
            CYStart = {_mtd_cy_start}, CYEnd = {_mtd_cy_end}
        - If the user specifies a past month (e.g. "May 2025"), use that month's exact first and last day.
        - Use only datetime literals — NEVER use now(), startofmonth(), or endofmonth() for date boundaries.

        CASE A — Simple total (e.g. "MTD sales of May 2025", "What is MTD sales?"):
        - Use toscalar and print:
            let CYStart = datetime(YYYY-MM-01);
            let CYEnd   = datetime(YYYY-MM-LD);   // LD = last day of that month
            let LYStart = datetime(YYYY-1-MM-01);
            let LYEnd   = datetime(YYYY-1-MM-LD);
            let CYRevenue = toscalar(SAPSalesInfos | where fkdat between (CYStart .. CYEnd) | summarize sum(Revenue));
            let LYRevenue = toscalar(SAPSalesInfos | where fkdat between (LYStart .. LYEnd) | summarize sum(Revenue));
            print
                CYRevenue = CYRevenue,
                LYRevenue = LYRevenue,
                MTDGrowth = iff(LYRevenue == 0, real(null), (CYRevenue - LYRevenue) / LYRevenue * 100)
            | extend GrowthType = iff(isnull(MTDGrowth), "N/A", iff(MTDGrowth > 0, "positive growth", "negative growth"));

        CASE B — Grouped by dimension (e.g. "MTD sales by division", "MTD by brand", "MTD by depot"):
        - Do NOT use toscalar or print. Use summarize ... by [dimension_column]:
            let CYStart = datetime(YYYY-MM-01);
            let CYEnd   = datetime(YYYY-MM-LD);   // LD = last day of that month
            SAPSalesInfos
            | where fkdat between (CYStart .. CYEnd)
            | summarize CYRevenue = sum(Revenue), TotalQuantity = sum(fkimg) by [dimension_column]
            | order by CYRevenue desc
            | take 100;

        Choose the correct case based on the user request.
        IMPORTANT: Use only concrete datetime literals in the generated KQL. Do NOT use now(), startofmonth(), or endofmonth().
        """


    # elif "YTD" in user_req or "Year-to-Date" in user_req:
    elif YTD_RE.search(user_req):
        prompt += f"""
        Instruction — YTD (Year-to-Date):
        - Fiscal year runs April 1 → March 31.
        - Current fiscal year start  : {_fy_start_str}
        - YTD cut-off (AsOfDate)     : {_asof_str}  ← use this EXACT date, do NOT recompute it
        - Last fiscal year start     : {_ly_fy_start_str}

        RULES:
        - Use the concrete dates provided above — do NOT call now(), startofmonth(), or endofmonth() in the query.
        - Do NOT use now() in any where clauses — only use the literal datetime values given above.
        - If the user asks for a PAST fiscal year (e.g. "ytd sales of 2023"), adjust FiscalYearStart accordingly
          (e.g. 2023-04-01) and set AsOfDate to 2024-03-31 (end of that year). Otherwise use the values above.

        CASE A — Simple total (e.g. "What is YTD sales?", "YTD growth", "YTD sales this year"):
        - Pull two scalars and emit with print:
            let FiscalYearStart = datetime({_fy_start_str});
            let AsOfDate        = datetime({_asof_str});
            let LYFiscalStart   = datetime({_ly_fy_start_str});
            let LYAsOfDate      = datetime({_ly_asof_str});
            let CYRevenue = toscalar(
                SAPSalesInfos
                | where fkdat between (FiscalYearStart .. AsOfDate)
                | summarize sum(Revenue)
            );
            let LYRevenue = toscalar(
                SAPSalesInfos
                | where fkdat between (LYFiscalStart .. LYAsOfDate)
                | summarize sum(Revenue)
            );
            print
                YTDGrowth = (CYRevenue - LYRevenue) / LYRevenue * 100,
                CYRevenue = CYRevenue,
                LYRevenue = LYRevenue
            | extend
                ErrorMessage = iff(isnull(YTDGrowth), "Error: missing data", ""),
                GrowthType = iff(isnull(YTDGrowth), "N/A", iff(YTDGrowth > 0, "positive growth", "negative growth"));

        CASE B — Grouped by dimension (e.g. "YTD sales by division", "YTD by brand", "YTD by zone", "YTD by depo"):
        - Do NOT use toscalar or print. Use summarize ... by [dimension_column]:
            let FiscalYearStart = datetime({_fy_start_str});
            let AsOfDate        = datetime({_asof_str});
            SAPSalesInfos
            | where fkdat between (FiscalYearStart .. AsOfDate)
            | summarize CYRevenue = sum(Revenue), TotalQuantity = sum(fkimg) by [dimension_column]
            | order by CYRevenue desc
            | take 100;

        Choose the correct case based on the user request.
        IMPORTANT: Copy the datetime literals exactly as shown above. Do NOT recompute or guess any dates.
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

        # 3. Extract segment value — try multiple patterns
        seg_m = re.search(
            r'contribution (?:of|from|by)\s+(.*?)(?:\s+in|\s+for|\s+from|\s+by|\s+on|$)', user_req, re.IGNORECASE)
        if not seg_m:
            # "What % of sales did Brand A contribute?"
            seg_m = re.search(r'did\s+(.*?)\s+contribute', user_req, re.IGNORECASE)
        if not seg_m:
            return "// Could not parse the segment name."
        raw_segment = seg_m.group(1).strip()
        # Remove dimension keyword if present
        strip_dim = re.compile(rf'\b{re.escape(dim_key)}\b', re.IGNORECASE)
        segment = strip_dim.sub('', raw_segment).strip()
        # Remove trailing month/year phrases
        segment = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}$', '', segment, flags=re.IGNORECASE).strip()
        # Strip trailing/leading punctuation users may type (e.g. "APE CLASSIC,")
        segment = segment.strip('.,;:"\' ').strip()

        # 4. Build filter clause with correct types
        _contains_cols = {"cname", "arktx", "wgbez", "spart_text", "matkl"}
        segment_display = segment
        gsber_key, code = None, None
        if dim_col == "gsber":
            gsber_key, code = find_gsber_code(segment, GSBER_MAPPING)
            print("gsber_key ", gsber_key)
            print("code ", code)
            if gsber_key and code:
                filter_clause = f'{dim_col} == {int(code)}'   # numeric — no quotes
                segment_display = gsber_key
            else:
                try:
                    filter_clause = f'{dim_col} == {int(segment)}'
                except ValueError:
                    filter_clause = f'{dim_col} == {segment}'
                segment_display = segment
        elif dim_col in _contains_cols:
            filter_clause = f'{dim_col} contains "{segment}"'
            segment_display = segment
        else:
            filter_clause = f'{dim_col} =~ "{segment}"'
            segment_display = segment

        # 5. KQL Template — inject scope filter so restricted users can't bypass
        _scope_line = f"\n            {_mandatory_where_clause.strip()}" if _mandatory_where_clause else ""

        return f"""
        let StartDate = datetime({start_date});
        let EndDate   = datetime({end_date});
        let TotalRevenue = toscalar(
            {TABLE_NAME}{_scope_line}
            | where fkdat >= StartDate and fkdat <= EndDate
            | summarize TotalRevenue = sum(Revenue)
        );
        let SegmentRevenue = toscalar(
            {TABLE_NAME}{_scope_line}
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
    """Wrap bare YYYY-MM-DD dates with datetime(), skipping ones already wrapped."""
    def _wrap(m):
        # If already inside datetime(...), return unchanged
        start = m.start()
        if start >= 9 and kql_query[start - 9:start] == 'datetime(':
            return m.group(0)
        return f'datetime({m.group(0)})'
    return re.sub(r'\d{4}-\d{2}-\d{2}', _wrap, kql_query)

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
    # -----------------------------
    # Skip date detection for YTD/MTD — they manage their own date logic internally.
    # Appending an explicit date range to these queries causes the LLM to hardcode
    # potentially incorrect dates instead of using startofmonth(now())-1d etc.
    if not MTD_RE.search(user_prompt) and not YTD_RE.search(user_prompt):
        start_date, end_date = detect_date_filter_using_llm(user_prompt)
        if start_date and end_date:
            if not re.search(r'from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}', user_prompt):
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

    result_prompt = (
        (history_block + "\n" if history_block else "")
        + "Now, CURRENT USER MESSAGE:\n"
        + f"USER: {user_prompt}\n\n"
        + "Context Data (use ONLY this JSON for any numbers):\n"
        + f"{result_json}\n\n"
        + "Format the output in bulleted format.\n"
        + "- Begin with a concise Title for the result.\n"
        + (f"- If helpful (scope is small), append this to the Title: \"{title_suffix}\".\n" if title_suffix else "")
        + "- Amount is in BDT and Volume is in gallons.\n"
        + "- When you show a money value for Sales:\n"
        + "    - First, show the FULL exact value from the JSON with thousand separators (e.g. 2902212694.11 → 2,902,212,694.11 BDT). upto 2 decimal places\n"
        + "    - Immediately after that, in brackets, show an approximate value in million or billion BDT, for example:\n"
        + "        - 2,902,212,694.1105475 BDT (~2.90 billion BDT)\n"
        + "        - 75,123,456.78 BDT (~75.12 million BDT)\n"
        + "    - Do NOT replace the full number with only 'about X billion'; always show the full value first, then the rounded value in brackets.\n"
        + "- Replace 'gsber' with 'Depo/Sales Office', and translate gsber numeric codes to names using this mapping: "
        + ", ".join(f"{v}={k}" for k, v in GSBER_MAPPING.items())
        + ". If a code is not in the mapping, show it as-is.\n"
        + "- If 'vtweg' is found, show:\n"
        + "    - '10' → Dealer (10)\n"
        + "    - '20' → Customer (20)\n"
        + "    - '30' → Project Customer (30)\n"
        + "- Use bullet points for both numerical and categorical results.\n\n"
        +"-Intead of using Revenue Use Sales Value"
        + (f"- Add a final bullet in Insights: \"{insight_note}\"\n" if insight_note else "")
        + "Then generate short Insights on [context]. \n"
    )

#     result_prompt = (
#     (history_block + "\n" if history_block else "")
#     + "Now, CURRENT USER MESSAGE:\n"
#     + f"USER: {user_prompt}\n\n"
#     + "Context Data (use ONLY this JSON for any numbers):\n"
#     + f"{result_json}\n\n"
#     + "Format the output in bulleted format.\n"
#     + "- Begin with a concise Title for the result.\n"
#     + "- The Title should mention the time scope if it is obvious from the data (e.g. 'July 2024 vs July 2025 Sales').\n"
#     + (f"- If helpful (scope is small), append this to the Title: \"{title_suffix}\".\n" if title_suffix else "")
#     + "- Amount is in BDT and Volume is in gallons.\n"
#     + "- When describing the metric named 'Revenue' in the data, always refer to it as 'Sales' in the narrative. Do NOT use the word 'Revenue'.\n"
#     + "- Replace 'gsber' with 'Depo/Sales Office'.\n"
#     + "- If 'vtweg' is found, show:\n"
#     + "    - '10' → Dealer (10)\n"
#     + "    - '20' → Customer (20)\n"
#     + "    - '30' → Project Customer (30)\n"
#     + "- Use bullet points for both numerical and categorical results.\n"
#     + "- Always base ALL numeric values ONLY on the Context Data JSON. Do NOT invent or guess numbers.\n"
#     + "- When you show a money value for Sales:\n"
#     + "    - First, show the FULL exact value from the JSON with thousand separators (e.g. 2902212694.1105475 → 2,902,212,694.1105475 BDT).\n"
#     + "    - Immediately after that, in brackets, show an approximate value in million or billion BDT, for example:\n"
#     + "        - 2,902,212,694.1105475 BDT (~2.90 billion BDT)\n"
#     + "        - 75,123,456.78 BDT (~75.12 million BDT)\n"
#     + "    - Use 'billion' if the value is ≥ 1,000,000,000 BDT, otherwise use 'million' if the value is ≥ 1,000,000 BDT.\n"
#     + "    - Do NOT replace the full number with only 'about X billion'; always show the full value first, then the rounded value in brackets.\n"
#     + "- If the data contains a time-like field such as 'TimePeriod', 'Year', 'Month', 'fkdat', etc.:\n"
#     + "    - First, list EACH time period present in the data with its Sales value (full amount + brackets), one bullet per period.\n"
#     + "      Example: 'Sales in July 2024: 2,887,503,249.590391 BDT (~2.89 billion BDT)'.\n"
#     + "    - After listing all periods, add a separate bullet section for comparison:\n"
#     + "        - Compare the latest period to the immediately previous period.\n"
#     + "        - Compute the absolute change in Sales (full value + brackets in million/billion).\n"
#     + "        - Compute the percentage change in Sales (e.g. +0.51%).\n"
#     + "        - State clearly whether it is an increase or decrease (e.g. 'slight increase', 'moderate decline').\n"
#     + "- If only a single time period is present, just describe the level of Sales for that period using the full amount + brackets.\n"
#     + "- When there are multiple categories (e.g. multiple depots, brands, customers), highlight the top ones by Sales or Volume when useful.\n"
#     + "- Do NOT talk about columns or JSON structure; talk in business terms (Sales, time period, dealers, depots, etc.).\n\n"
#     + (f"- Add a final bullet in Insights: \"{insight_note}\"\n" if insight_note else "")
#     + "Then generate short Insights on this context, focusing on what changed, which periods or categories are higher or lower, and any obvious patterns.\n"
# )



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
    return analysis_llm.invoke(messages).content

    # try:
    #     return analysis_llm.invoke([{"role": "user", "content": result_prompt}]).content
    # except BadRequestError as e:
    #     # Azure returns a JSON body with details (e.g., context length, invalid param, etc.)
    #     try:
    #         err_json = e.response.json()
    #     except Exception:
    #         err_json = {"message": str(e)}
    #     return f"Azure OpenAI 400 Bad Request.\nDetails: {err_json}"

