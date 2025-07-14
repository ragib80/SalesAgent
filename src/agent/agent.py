# agent.py ─ SAP Sales bot for Azure ADX (SAPSalesInfos)
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
    "• For monthly aggregation, group by month: | summarize sum(Revenue) by Month = bin(fkdat, 30d)\n"
    "• For weekly aggregation, use bin(fkdat, 7d); for daily, bin(fkdat, 1d)\n"
    "• For yearly aggregation, use bin(fkdat, 1y).\n"
    "• To calculate previous period's revenue, use prev(TotalRevenue, 1).\n"
    "• To calculate revenue growth, use the difference between current period and previous period.\n"
    "• To calculate percentage growth, use: RevenueGrowthPercent = (RevenueGrowth / PreviousPeriodRevenue) * 100\n"
    "• End every statement with a semicolon.\n"
    "• Provide real line-breaks (no \\n literals).\n\n"
    
    "Business → column mapping:\n" + MAPPING_STR +
    "\n\nDepo/Business Area (gsber) → column Value mapping:\n" + GSBER_MAPPING_STR +
    "\n\nTable schema:\n" + KUSTO_SCHEMA +

    "\nExample KQLs for reference:\n"
    "1. **Monthly Revenue Growth (April 2025):**\n"
    "SAPSalesInfos\n"
    "| where fkdat >= datetime(2025-04-01) and fkdat <= datetime(2025-04-30)\n"
    "| summarize TotalRevenue = sum(Revenue) by Month = bin(fkdat, 30d)\n"
    "| serialize\n"
    "| extend PreviousMonthRevenue = prev(TotalRevenue, 1)\n"
    "| extend RevenueGrowth = TotalRevenue - PreviousMonthRevenue\n"
    "| extend RevenueGrowthPercent = iif(PreviousMonthRevenue != 0, todouble(RevenueGrowth) / todouble(PreviousMonthRevenue) * 100, 0.0)\n"

    "2. **Weekly Revenue Growth (4/7/2025 - 4/14/2025):**\n"
    "SAPSalesInfos\n"
    "| where fkdat >= datetime(2025-04-01) and fkdat <= datetime(2025-04-30)\n"
    "| summarize TotalRevenue = sum(Revenue) by Week = bin(fkdat, 7d)\n"
    "| serialize\n"
    "| extend PreviousWeekRevenue = prev(TotalRevenue, 1)\n"
    "| extend RevenueGrowth = TotalRevenue - PreviousWeekRevenue\n"
    "| extend RevenueGrowthPercent = iif(PreviousWeekRevenue != 0, todouble(RevenueGrowth) / todouble(PreviousWeekRevenue) * 100, 0.0)\n"

    "3. **Yearly Revenue Growth (2025):**\n"
    "SAPSalesInfos\n"
    "| where fkdat >= datetime(2025-01-01) and fkdat <= datetime(2025-12-31)\n"
    "| summarize TotalRevenue = sum(Revenue) by Year = bin(fkdat, 1y)\n"
    "| serialize\n"
    "| extend PreviousYearRevenue = prev(TotalRevenue, 1)\n"
    "| extend RevenueGrowth = TotalRevenue - PreviousYearRevenue\n"
    "| extend RevenueGrowthPercent = iif(PreviousYearRevenue != 0, todouble(RevenueGrowth) / todouble(PreviousYearRevenue) * 100, 0.0)\n"

    "\nPlease note that the query should be based on the user’s input for **periodicity** (daily, monthly, or yearly) and the corresponding **date range**.\n"
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

def format_dates(kql_query: str) -> str:
    """Ensure all date-like strings are properly formatted as datetime literals."""
    return re.sub(r'(\d{4}-\d{2}-\d{2})', r'datetime(\1)', kql_query)

def detect_period_aggregation(prompt: str):
    prompt = prompt.lower()
    if 'daily' in prompt or 'per day' in prompt:
        return '1d'
    elif 'weekly' in prompt or 'per week' in prompt:
        return '1w'
    elif 'monthly' in prompt or 'per month' in prompt or 'by month' in prompt:
        return '1mo'
    elif 'quarterly' in prompt or 'per quarter' in prompt:
        return '3mo'
    elif 'yearly' in prompt or 'per year' in prompt:
        return '1y'
    else:
        return None

def detect_trend_request(prompt: str):
    keywords = ['trend', 'uptrend', 'downtrend', 'growth', 'decline', 'increase', 'decrease', 'compare', 'comparison', 'difference', 'change']
    for k in keywords:
        if k in prompt.lower():
            return True
    return False

def extract_month_year_pairs(prompt: str):
    # Extracts specific months/years like "January 2025", "Feb 2025", etc.
    # Returns a list of tuples: [('2025-01-01', '2025-01-31'), ...]
    import calendar
    month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    month_names.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
    results = []
    for match in re.finditer(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]+(\d{4})', prompt, re.I):
        month = month_names[match.group(1).lower()]
        year = int(match.group(2))
        start = datetime.date(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end = datetime.date(year, month, last_day)
        results.append((str(start), str(end)))
    return results

# Detect trend from user prompt (e.g., increasing, declining, etc.)
def detect_trend(user_prompt: str) -> str:
    if any(word in user_prompt.lower() for word in ["declining", "downtrending", "negative growth", "falling", "decrease"]):
        return "declining"
    elif any(word in user_prompt.lower() for word in ["increasing", "uptrending", "positive growth", "rising", "growth"]):
        return "increasing"
    else:
        return "stable"

# ───────────────────────── 5.  Main entry ─────────────────────────

def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    """
    Dynamically handle SAP Sales prompts, ensuring correct KQL generation,
    and map business area/territory to the correct 'gsber' code.
    """

    # 1. Handle "last n months"
    last_n_months_match = re.search(r'last\s+(\d+)\s+month[s]?', user_prompt, re.IGNORECASE)
    if last_n_months_match:
        n_months = int(last_n_months_match.group(1))
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=n_months * 30)  # Approximate
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        user_prompt += f" from {start_date_str} to {end_date_str}"

    # 2. Check for periodic aggregation request
    period = detect_period_aggregation(user_prompt)
    if period:
        user_prompt += f" Please group the result by period using bin(fkdat, {period}). Show the sum of Revenue for each period."

    # 3. Check for trend/comparison requests
    if detect_trend_request(user_prompt):
        user_prompt += (
            " For trend or comparison analysis, group by period (e.g., bin(fkdat, 1mo)) or the appropriate period, "
            "calculate sum(Revenue) per period, and show the difference or percent change between periods if relevant."
        )

    # 4. Handle explicit month comparisons (e.g. "Compare January 2025 and February 2025")
    month_years = extract_month_year_pairs(user_prompt)
    if len(month_years) >= 2:
        # Only filter for the earliest to latest month to keep the data
        start, _ = month_years[0]
        _, end = month_years[-1]
        user_prompt += f" from {start} to {end}. Please group by month and show sum(Revenue) for each month."

    # 5. If no date range or "last n months" is not found, ask user
    date_in_prompt = re.search(r'(\d{4}-\d{2}-\d{2})|(\d{4})|(from\s+\w+\s+\d{4})|(to\s+\w+\s+\d{4})|(\bago\b\s*\(\d+[a-zA-Z]*\))', user_prompt)
    if not (last_n_months_match or date_in_prompt or month_years):
        user_prompt += " Please specify a date range for the data (e.g., from 2025-01-01 to 2025-12-31)."

    # 6. Generate raw KQL from the user prompt using LLM
    kql = generate_kql(user_prompt)
    print(f"Generated KQL Query: {kql}")

    # 7. Post-process KQL for formatting and fixes
    kql = format_dates(kql)
    kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
    kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)

    # 8. Map business area/territory name to gsber code
    for territory, gsber_value in GSBER_MAPPING.items():
        if territory.lower() in user_prompt.lower():
            kql = re.sub(r"where Territory == .+?", f"where gsber == '{gsber_value}'", kql)
            break

    # 9. Trend direction (increase/decline)
    trend = detect_trend(user_prompt)
    if trend == "declining":
        kql = kql.replace("RevenueChange < 0", "RevenueChange < 0")
    elif trend == "increasing":
        kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")
    else:
        kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")

    # 10. Execute the query and handle retries
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError as err:
            if attempt == 1:
                kql = generate_kql(user_prompt, strict=True)
                continue
            print(f"Error: {err}")
            return "Please refine your query for better results. I’m learning day by day and will help you improve your query."

    # 11. No data found case
    if not rows:
        return "No data found matching your criteria. Please refine your query for more specific results."

    # 12. Prepare sample data for summary LLM
    result_data = [dict(zip(cols, r)) for r in rows[:20]]
    result_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Sample Data:\n{json.dumps(result_data, indent=2)}\n\n"
        "Based on the query results, format the output in bulleted format. "
        "If the result is numerical or comparative, bullet points for proper indication. If it's categorical or simple, use bullet points. "
        "After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Amount is in BDT."
        "If Needed,Based on the Sample  context data  give meaningful business-related suggestions such as increasing sales, revenue."
    )
    formatted_result = llm.invoke([{"role": "user", "content": result_prompt}]).content
    return formatted_result

# END OF FILE
