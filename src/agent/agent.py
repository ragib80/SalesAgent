# agent.py ─ Simplified SAP Sales bot for Azure ADX (SAPSalesInfos)
import os, re, json
from functools import lru_cache
import datetime
from django.conf import settings
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoApiError

from langchain_openai import AzureChatOpenAI
import logging
import dateutil.parser
import calendar

import logging

logger = logging.getLogger(__name__)

# Replace print with logging
logger.debug("This is a debug message")
logger.info("This is an info message")
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

# compile once
MTD_RE = re.compile(r'\b(?:mtd|month[- ]to[- ]date)\b', re.IGNORECASE)
YTD_RE = re.compile(r'\b(?:ytd|year[- ]to[- ]date)\b', re.IGNORECASE)

CONTRIBUTION_RE = re.compile(r'\b(contribution of|contribution from|contribution by)\b', re.IGNORECASE)


# lower-case keys for matching
FIELD_MAP_LOWER = {k.lower(): v for k, v in FIELD_MAPPINGS.items()}

TREND_RE     = re.compile(r'\b(?:up[- ]?trending|trending)\b', re.IGNORECASE)
DOWN_TREND_RE = re.compile(r'\b(?:down[- ]?trending|downtrend|negative trend|falling|declining|decreasing)\b', re.IGNORECASE) 
EXCLUDE_KEYS = {"revenue", "quantity", "volume", "date", "fkdat"}

def generate_kql(user_req: str, strict=False) -> str:
    # Start with the base prompt for LLM
    prompt = SYSTEM_PROMPT_KQL
    if strict:
        prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."
    
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



    # ————— Contribution branch ——————————————————

    # working
    # elif CONTRIBUTION_RE.search(user_req):
    #     # 1. Parse date range
    #     m = re.search(r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', user_req)
    #     if m:
    #         start_date, end_date = m.groups()
    #     else:
    #         # Natural language month: from April 2025 to June 2025
    #         month_range = re.search(r'from ([a-zA-Z]+ \d{4}) to ([a-zA-Z]+ \d{4})', user_req, re.IGNORECASE)
    #         if month_range:
    #             try:
    #                 start_dt = dateutil.parser.parse("1 " + month_range.group(1))
    #                 end_month_dt = dateutil.parser.parse("1 " + month_range.group(2))
    #                 last_day = calendar.monthrange(end_month_dt.year, end_month_dt.month)[1]
    #                 end_dt = end_month_dt.replace(day=last_day)
    #                 start_date = start_dt.strftime("%Y-%m-%d")
    #                 end_date = end_dt.strftime("%Y-%m-%d")
    #             except Exception:
    #                 return "// Could not parse month range. Use format like 'from April 2025 to June 2025'."
    #         else:
    #             # Default to last full month
    #             now = datetime.datetime.now()
    #             last_month_end = now.replace(day=1) - datetime.timedelta(days=1)
    #             start_date = f"{last_month_end.year}-{last_month_end.month:02d}-01"
    #             end_date = f"{last_month_end.year}-{last_month_end.month:02d}-{last_month_end.day:02d}"

    #     # 2. Detect dimension (brand, division, etc)
    #     lowered = user_req.lower()
    #     dim_key = next((k for k in FIELD_MAP_LOWER if k in lowered and k not in EXCLUDE_KEYS), None)
    #     if not dim_key:
    #         return "// Could not detect which dimension to use for contribution."
    #     dim_col = FIELD_MAP_LOWER[dim_key]

    #     # 3. Extract segment value robustly (remove dimension, remove date/month phrases)
    #     seg_m = re.search(
    #         r'contribution (?:of|from)\s+(.*?)(?:\s+in|\s+for|\s+from|\s+by|\s+on|$)', user_req, re.IGNORECASE)
    #     if not seg_m:
    #         return "// Could not parse the segment name."
    #     raw_segment = seg_m.group(1).strip()
    #     # Remove dimension keyword if present
    #     strip_dim = re.compile(rf'\b{re.escape(dim_key)}\b', re.IGNORECASE)
    #     segment = strip_dim.sub('', raw_segment).strip()
    #     # Remove trailing month/year phrases
    #     segment = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}$', '', segment, flags=re.IGNORECASE).strip()

    #     # 4. Build filter clause (handles gsber/depo mapping if needed)
    #     if dim_col == "gsber":
    #         code = GSBER_MAPPING.get(segment)
    #         if not code:
    #             return f"// Unknown business area '{segment}'."
    #         filter_clause = f'{dim_col} == "{code}"'
    #     else:
    #         # Use exact match first, fallback to contains_cs if needed (case-sensitive)
    #         filter_clause = f'{dim_col} == "{segment}"'

    #     # 5. KQL Template
    #     return f"""
    #     let StartDate = datetime({start_date});
    #     let EndDate   = datetime({end_date});
    #     let TotalRevenue = toscalar(
    #         {TABLE_NAME}
    #         | where fkdat >= StartDate and fkdat <= EndDate
    #         | summarize TotalRevenue = sum(Revenue)
    #     );
    #     let SegmentRevenue = toscalar(
    #         {TABLE_NAME}
    #         | where fkdat >= StartDate and fkdat <= EndDate and {filter_clause}
    #         | summarize SegmentRevenue = sum(Revenue)
    #     );
    #     print
    #         Dimension       = "{dim_col}",
    #         Segment         = "{segment}",
    #         TotalRevenue    = TotalRevenue,
    #         SegmentRevenue  = SegmentRevenue,
    #         ContributionPct = iff(isnull(TotalRevenue) or TotalRevenue == 0, real(null), SegmentRevenue * 100.0 / TotalRevenue)
    #     | extend
    #         Insight = strcat("The contribution of {segment} under {dim_col} is ", round(ContributionPct, 2), "%.")
    #     """.strip()

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
                # Default to last full month
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
        if dim_col == "gsber":
            norm_segment = segment.lower()
            norm_segment = re.sub(r'\b(sales\s*office|sales|office|area)\b', '', norm_segment, flags=re.IGNORECASE).strip()
            # 4.1 Try mapping dictionary first (case-insensitive, loose match)
            gsber_key = next((k for k in GSBER_MAPPING if k.lower() == norm_segment), None)
            if not gsber_key:
                # Try partial match as fallback
                gsber_key = next((k for k in GSBER_MAPPING if norm_segment in k.lower()), None)
            if gsber_key:
                code = GSBER_MAPPING[gsber_key]
                filter_clause = f'{dim_col} == "{code}"'
                segment_display = gsber_key
            else:
                # 4.2 If mapping fails, allow direct match on gsber code or value in ADX
                filter_clause = f'{dim_col} == "{segment}" or {dim_col} == toscalar(SAPSalesInfos | where {dim_col} == "{segment}" | summarize take_any({dim_col}))'
                # Optionally add an LLM instruction for future: "If mapping is missing, try direct match with gsber field"
        else:
            filter_clause = f'{dim_col} == "{segment}"'
            segment_display = segment

        # 5. KQL Template + Optional LLM instruction (if mapping was not found)
        extra_instruction = ""
        if dim_col == "gsber" and not gsber_key:
            extra_instruction = "// Could not map sales office/depo from predefined dictionary, attempting direct ADX field match.\n"

        return f"""
        {extra_instruction}
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


    
    # Explicitly instruct LLM to avoid using `bin(fkdat, 1mo)` and instead use `startofmonth(fkdat)`
    else:
        prompt += """
        Instruction: 
        - Do not use the `bin(fkdat, 1mo)` operator for time-based grouping.
        - Instead, use `startofmonth(fkdat)` for monthly grouping (or other appropriate time functions based on the query).
        - Ensure the query does not use `bin` and directly uses time-based functions for grouping.
        - Group by the result of the time-based function using an `extend` statement, for example: `extend TimePeriod = startofmonth(fkdat)`
        - Always use the named columns in the `summarize` statement.
        """
    
    prompt += f"\n\nUser request: {user_req}"

    # Print the full prompt for debugging
    print("_extract_kql-------------", prompt)
    
    # Send the request to the LLM
    response = llm.invoke([{"role": "user", "content": prompt}]).content

    # Explicitly replace `bin(fkdat, 1mo)` with `startofmonth(fkdat)` or appropriate time function if found
    response = response.replace("bin(fkdat, 1mo)", "startofmonth(fkdat)")  # Replace bin with startofmonth

    # Ensure the query has the correct `extend` and `summarize` structure
    if "summarize" in response and "by ," in response:  # Check if summarize doesn't have a valid grouping field
        response = response.replace("by ,", "by TimePeriod")  # Insert a valid field for grouping

    # If `summarize` is missing the grouping field, add a default grouping by `TimePeriod`
    if "summarize" in response and ", )" in response:
        response = response.replace(", )", ", TimePeriod)")  # Correct the empty `summarize`

    # If the query doesn't contain a `TimePeriod` column, we add it dynamically (for time-based queries)
    if "summarize" in response and "by TimePeriod" not in response:
        response = response.replace("summarize", "extend TimePeriod = startofmonth(fkdat)\n| summarize")  # Ensure TimePeriod is used
    
    print("response from generate kql ", response)
    return _extract_kql(response)



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


# Enhance handle_user_query to use dynamic date range detection

def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    """
    Dynamically handle SAP Sales prompts, ensuring correct KQL generation,
    and map business area/territory to the correct 'gsber' code.
    """
    # -- [unchanged] detect or ask for dates
    print("user_prompt:", user_prompt)
    logger.debug("This is a debug message")
    start_date, end_date = detect_date_filter_using_llm(user_prompt)
    if start_date and end_date:
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str   = end_date.strftime("%Y-%m-%d")
        user_prompt   += f" from {start_date_str} to {end_date_str}"
  
    else:
        user_prompt   += " Please specify a date range for the data (e.g., from 2025-01-01 to 2025-12-31)."
    print("date range :", start_date)
    print("date range :", end_date_str)
    # -- [unchanged] raw KQL generation + fixes
    kql = generate_kql(user_prompt)
    kql = format_dates(kql)
    kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
    kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)

    # -- [unchanged] territory → gsber mapping
    for territory, gsber_value in GSBER_MAPPING.items():
        if territory.lower() in user_prompt.lower():
            kql = re.sub(r"where Territory == .+?", f"where gsber == '{gsber_value}'", kql)
            break

    # -- [unchanged] trend detection
    trend = detect_trend(user_prompt)
    if trend == "declining":
        kql = kql.replace("RevenueChange < 0", "RevenueChange < 0")
    elif trend == "increasing":
        kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")
    else:
        kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")

    # -- [unchanged] execute with retry
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError:
            if attempt == 1:
                kql = generate_kql(user_prompt, strict=True)
                continue
            return "Please refine your query for better results. I’m learning day by day and will help you improve your query."

    if not rows:
        return "No data found matching your criteria. Please refine your query for more specific results."

    # —————————————————————————
    # ↓ NEW: fully dynamic datetime formatting ↓
    # —————————————————————————

    # 1) Limit to top N rows
    rows_to_show = rows[:30]
    print("rows_to_show = rows[:30]:", rows_to_show)
    # 2) Build result_data, converting any datetime to "YYYY-MM-DD"
    result_data = []
    for row in rows_to_show:
        row_dict = dict(zip(cols, row))
        for col_name, value in row_dict.items():
            if isinstance(value, datetime.datetime):
                row_dict[col_name] = value.strftime("%Y-%m-%d")
        result_data.append(row_dict)

    # 3) Optionally sort by detected date-like column
    date_cols = [c for c in cols if c.lower() in ("timeperiod", "week", "month", "date")]
    if date_cols:
        key = date_cols[0]
        result_data.sort(key=lambda x: x[key])

    # 4) Safe JSON serialization
    result_json = json.dumps(result_data, default=str, indent=2)
    print("json.dumps result_data:", result_json)

    # —————————————————————————
    # Resume your original LLM-prompting logic
    # —————————————————————————
    result_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Context Data:\n{result_json}\n\n"
        "Based on the query results, format the output in bulleted format. "
        "If the result is numerical or comparative, bullet points for proper indication. If it's categorical or simple, use bullet points. "
        "After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Amount is in BDT."
        "If Needed, Based on the Context Data give meaningful business-related suggestions such as increasing sales, revenue."
    )
    formatted_result = llm.invoke([{"role": "user", "content": result_prompt}]).content
    return formatted_result

