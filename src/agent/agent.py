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

# def generate_kql(user_req: str, strict=False) -> str:
#     prompt = SYSTEM_PROMPT_KQL
#     if strict:
#         prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."
#     prompt += f"\n\nUser request: {user_req}"
#     print("_extract_kql-------------",prompt)
#     response = llm.invoke([{"role":"user","content":prompt}]).content

#     return _extract_kql(response)

def generate_kql(user_req: str, strict=False) -> str:
    # Start with the base prompt for LLM
    prompt = SYSTEM_PROMPT_KQL
    if strict:
        prompt += "\n\nSTRICT MODE: previous query failed. Return corrected KQL only."
    
    # Detect if the user is asking for MTD sales or growth
    if "MTD" in user_req or "Month-to-Date" in user_req:
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


    # Detect if the user is asking for YTD sales or growth
    # elif "YTD" in user_req or "Year-to-Date" in user_req:
    #     prompt += """
    #     Instruction:
    #     - Fiscal year runs April 1 → March 31.
    #     - Compute YTD through the **last day of the previous month**:
    #         let FiscalYearStart = datetime(YYYY-04-01);
    #         let AsOfDate        = startofmonth(now()) - 1d;
    #     - Pull two scalars with `toscalar(...)`:
    #         let CYRevenue = toscalar(
    #         SAPSalesInfos
    #         | where fkdat between (FiscalYearStart .. AsOfDate)
    #         | summarize sum(Revenue)
    #         );
    #         let LYRevenue = toscalar(
    #         SAPSalesInfos
    #         | where fkdat between (datetime_add('year', -1, FiscalYearStart) .. datetime_add('year', -1, AsOfDate))
    #         | summarize sum(Revenue)
    #         );
    #     - Emit three real‐typed scalars with `print`—using `real(null)` for any missing data:
    #         print YTDGrowth = iff(isnull(CYRevenue) or isnull(LYRevenue), real(null), (CYRevenue - LYRevenue) / LYRevenue * 100),
    #             CYRevenue   = iff(isnull(CYRevenue), real(null), CYRevenue),
    #             LYRevenue   = iff(isnull(LYRevenue), real(null), LYRevenue)
    #     - Then immediately `extend` two new string columns:
    #         | extend 
    #             ErrorMessage = iff(isnull(YTDGrowth), "Error: missing CY or LY revenue", ""),
    #             GrowthType   = iff(isnull(YTDGrowth), "N/A", iff(YTDGrowth > 0, "positive growth", "negative growth"))
    #     - Do **not** rely on default names like `print_1` or `print_2`.
    #     """
    #     prompt += f"\n\nUser request: {user_req}"

    elif "YTD" in user_req or "Year-to-Date" in user_req:
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



# import datetime
# import re

# Enhance handle_user_query to use dynamic date range detection
def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    """
    Dynamically handle SAP Sales prompts, ensuring correct KQL generation,
    and map business area/territory to the correct 'gsber' code.
    """
    # Check for date-related filters in the user prompt using LLM
    start_date, end_date = detect_date_filter_using_llm(user_prompt)
    print("start_date from handle",start_date)
    if start_date and end_date:
        # Format the date range based on the detected filter
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Add the date range to the prompt
        user_prompt += f" from {start_date_str} to {end_date_str}"

    # If no date range is detected, ask the user for one
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
    
    print("before attempt  : {kql}")
    # Execute the query and handle retries
    for attempt in (1, 2):
        try:
            cols, rows = adx().run(kql)
            break
        except KustoApiError as err:
            if attempt == 1:
                kql = generate_kql(user_prompt, strict=True)
                print(f" attempt  kql : {kql}")
                continue
            # Log the error and return user-friendly feedback.
            print(f"Error: {err}")
            return "Please refine your query for better results. I’m learning day by day and will help you improve your query."
    print(f" final kql : {kql}")
    # If no data found, provide feedback
    if not rows:
        return "No data found matching your criteria. Please refine your query for more specific results."
    print ("final cols",cols)
    print ("final rows",rows)
    
    # Format datetime columns to strings in result_data
    result_data = []
    for row in rows[:20]:  # Adjust as needed
        row_dict = dict(zip(cols, row))
        # Format datetime fields (e.g., TimePeriod) into string format
        tp = row_dict.get('TimePeriod')
        if isinstance(tp, datetime.datetime):
            row_dict['TimePeriod'] = tp.strftime("%Y-%m-%d")
        # else leave it alone (it’s already a string label)
        result_data.append(row_dict)
    
    print("formatted result_data:", result_data)

    print(f"json.dumps result_data {json.dumps(result_data, indent=2)}")
    result_prompt = (
        f"User asked: {user_prompt}\n\n"
        f"Context Data:\n{json.dumps(result_data, indent=2)}\n\n"
        "Based on the query results, format the output in bulleted format. "
        "If the result is numerical or comparative, bullet points for proper indication. If it's categorical or simple, use bullet points. "
        "After formatting, provide a concise business insight related to the data, such as trends, patterns, or key takeaways. Amount is in BDT."
        "If Needed, Based on the Context Data give meaningful business-related suggestions such as increasing sales, revenue."
    )

    print("final result result_prompt ", result_prompt)

    # Let LLM decide on how to format the result: tabular or bulleted
    formatted_result = llm.invoke([{"role": "user", "content": result_prompt}]).content
    print("formatted_result   ", formatted_result)
    return formatted_result
