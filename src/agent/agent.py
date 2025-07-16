# agent.py ─ Simplified SAP Sales bot for Azure ADX (SAPSalesInfos)

import os
import re
import json
import datetime
from functools import lru_cache
from django.conf import settings

import dateparser
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoApiError
# CORRECTED IMPORTS
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from typing import ClassVar, List
from langchain.chains import SequentialChain
from langchain_core.runnables import RunnableSequence
from langchain_openai import AzureChatOpenAI



# ──────────────────────── 1. ADX helper ────────────────────────────
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

# ─────────────────────── 2. Prompt assets ──────────────────────────
TABLE_NAME = "SAPSalesInfos"

FIELD_MAPPINGS = {
    "revenue":"Revenue","quantity":"fkimg","volume":"volum","Dealer":"cname",
    "brand":"wgbez","product name":"arktx","product":"arktx","category":"matkl",
    "division":"spart_text","company code":"bukrs","sales org":"vkorg",
    "dist channel":"vtweg","distribution channel":"vtweg","business area":"gsber","depo":"gsber",
    "credit control area":"kkber","Dealer group":"kukla","account group":"ktokd",
    "sales group":"vkgrp_c","sales office":"vkbur_c","payer id":"Payer_DL",
    "product code":"matnr","unit":"meins","volume unit":"voleh","business group":"GK",
    "territory":"Territory","sales zone":"Szone","date":"fkdat","fkdat":"fkdat"
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
    "Dhaka South": "4110",
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
    "You are an expert Kusto (ADX) analyst for SAPSalesInfos.\n"
    "Generate only raw KQL (no markdown, no code fences, no backticks).\n"
    "Use table SAPSalesInfos and the date range below.\n\n"
    "Business → column mapping:\n"
    + MAPPING_STR
    + "\n\nDepo/Business Area (gsber) → column Value mapping:\n"
    + GSBER_MAPPING_STR
    + "\n\nTable schema:\n"
    + KUSTO_SCHEMA
)


# ───────────────────────── 3. LLM instance ───────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0,
)
# ─────────────────────── 1. Define your KQL prompt ──────────────────────
kql_prompt = PromptTemplate(
    input_variables=["system_prompt", "user_prompt", "start_date", "end_date"],
    template="""
{system_prompt}

let StartDate = datetime({start_date});
let EndDate   = datetime({end_date});

User request:
{user_prompt}

Output only raw KQL.
"""
)

# ──────────────────── 2. Build the RunnableSequence ───────────────────
# This “|” operator wires the PromptTemplate into your Azure LLM
kql_runnable = RunnableSequence(first=kql_prompt, last=llm)

# 3. Inside your KQLRunnableChain._call():
class KQLRunnableChain(Chain):
    input_keys  = ["user_prompt", "start_date", "end_date"]
    output_keys = ["raw_kql"]

    def _call(self, inputs):
        payload = {
            "system_prompt": SYSTEM_PROMPT_KQL,    # ← here
            "user_prompt":   inputs["user_prompt"],
            "start_date":    inputs["start_date"],
            "end_date":      inputs["end_date"],
        }
        raw_kql = kql_runnable.invoke(payload)
        return {"raw_kql": raw_kql}
# ───────────────────────── 4. Extract & Generate KQL ─────────────────


def _extract_kql(raw: str) -> str:
    """
    Strip Markdown fences/backticks and unescape any literal \n or \t.
    Returns a plain KQL string.
    """
    # 1) Remove fenced code blocks ```…```
    raw = re.sub(r'```(?:kql|kusto)?\s*([\s\S]*?)```', r'\1', raw, flags=re.I)
    # 2) Remove any remaining backticks
    raw = raw.replace("`", "")
    # 3) Unescape JSON-style literals
    raw = raw.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "")
    return raw.strip()




def generate_kql(user_req: str, strict=False) -> str:
    # HERE is where you assemble the prompt:
    prompt = SYSTEM_PROMPT_KQL + "\n\nUser request: " + user_req

    if strict:
        prompt += "\n\nSTRICT MODE: previous query failed—return corrected KQL only."

    raw = llm.invoke([{"role":"user", "content": prompt}]).content
    return _extract_kql(raw)




# ──────────────────────── 5. Formatting Helpers ───────────────────────
# def format_dates(kql_query: str) -> str:
#     """Wrap any YYYY-MM-DD literals in datetime()."""
#     return re.sub(r'(\d{4}-\d{2}-\d{2})', r'datetime(\1)', kql_query)
def format_dates(kql_query: str) -> str:
    """
    Wrap raw YYYY-MM-DD tokens in datetime(...) only if they
    aren’t already inside a datetime() call.
    """
    # Step 1: clean up any accidental nested datetime()
    kql_query = re.sub(
        r'datetime\s*\(\s*datetime\s*\(\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\)\s*\)',
        r'datetime(\1)',
        kql_query,
        flags=re.IGNORECASE,
    )
    # Step 2: wrap any bare dates
    return re.sub(
        r'(?<!datetime\()(\d{4}-\d{2}-\d{2})(?!\))',
        r'datetime(\1)',
        kql_query,
    )


def detect_trend(user_prompt: str) -> str:
    low = user_prompt.lower()
    if any(w in low for w in ["declining","downtrending","negative growth","falling","decrease"]):
        return "declining"
    if any(w in low for w in ["increasing","uptrending","positive growth","rising","growth"]):
        return "increasing"
    return "stable"

# ──────────────────────── 6. Date-Parsing Chain ───────────────────────

class DateRangeChain(Chain):
    input_keys: ClassVar[List[str]]  = ["user_prompt"]
    output_keys: ClassVar[List[str]] = ["start_date", "end_date"]

    def _call(self, inputs):
        print("▶️ DateRangeChain._call inputs:", inputs)
        text = inputs["user_prompt"].strip().lower()
        now = datetime.datetime.now()

        # 1) Explicit “from YYYY-MM-DD to YYYY-MM-DD”
        m = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', text)
        if m:
            out = {
                "start_date": m.group(1),
                "end_date":   m.group(2),
            }
            print("✅ Parsed explicit range:", out)
            return out

        # 2) Relative “last N days/weeks/months/years”
        m = re.search(r'last\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)', text)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if 'day' in unit:
                delta = datetime.timedelta(days=n)
            elif 'week' in unit:
                delta = datetime.timedelta(weeks=n)
            elif 'month' in unit:
                delta = datetime.timedelta(days=30 * n)
            else:
                delta = datetime.timedelta(days=365 * n)
            start = now - delta
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   now.strftime("%Y-%m-%d"),
            }
            print(f"✅ Parsed 'last {n} {unit}':", out)
            return out

        # 3) “today” / “yesterday”
        if re.search(r'\btoday\b', text):
            today = now.strftime("%Y-%m-%d")
            out = {"start_date": today, "end_date": today}
            print("✅ Parsed 'today':", out)
            return out

        if re.search(r'\byesterday\b', text):
            y = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            out = {"start_date": y, "end_date": y}
            print("✅ Parsed 'yesterday':", out)
            return out

        # 4) “this week” / “last week”
        if re.search(r'\bthis\s+week\b', text):
            start = now - datetime.timedelta(days=now.weekday())
            end   = start + datetime.timedelta(days=6)
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   end.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed 'this week':", out)
            return out

        if re.search(r'\blast\s+week\b', text):
            end   = now - datetime.timedelta(days=now.weekday() + 1)
            start = end - datetime.timedelta(days=6)
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   end.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed 'last week':", out)
            return out

        # 5) “this month” / “last month”
        if re.search(r'\bthis\s+month\b', text):
            start = now.replace(day=1)
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   now.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed 'this month':", out)
            return out

        if re.search(r'\blast\s+month\b', text):
            first_of_this = now.replace(day=1)
            last_of_last  = first_of_this - datetime.timedelta(days=1)
            start = last_of_last.replace(day=1)
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   last_of_last.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed 'last month':", out)
            return out

        # 6) “this year” / “last year”
        if re.search(r'\bthis\s+year\b', text):
            start = now.replace(month=1, day=1)
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   now.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed 'this year':", out)
            return out

        if re.search(r'\blast\s+year\b', text):
            start = now.replace(year=now.year - 1, month=1, day=1)
            end   = now.replace(year=now.year - 1, month=12, day=31)
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   end.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed 'last year':", out)
            return out

        # 7) “in YYYY” as full-year
        m = re.search(r'\bin\s+(\d{4})\b', text)
        if m:
            y = int(m.group(1))
            out = {
                "start_date": f"{y}-01-01",
                "end_date":   f"{y}-12-31",
            }
            print("✅ Parsed 'in YYYY':", out)
            return out

        # 8) Fallback: dateparser
        settings_dp = {"RELATIVE_BASE": now}
        found = dateparser.search.search_dates(text, settings=settings_dp) or []
        dates = sorted([d for _, d in found])
        if dates:
            start, end = dates[0], dates[-1]
            out = {
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   end.strftime("%Y-%m-%d"),
            }
            print("✅ Parsed fallback dates via dateparser:", out)
            return out

        # 9) Nothing matched
        msg = (
            "Could not parse a date range from the prompt. "
            "Please specify 'last 3 months', 'from YYYY-MM-DD to YYYY-MM-DD', "
            "or 'this year'."
        )
        print("❌ DateRangeChain error:", msg)
        raise ValueError(msg)


# ──────────────────── 7. KQL-Generation Chain ────────────────────────
class KQLGeneratorChain(Chain):
    print("-----------------KQLGeneratorChain method called --------------------")
    input_keys: ClassVar[List[str]]  = ["user_prompt", "start_date", "end_date"]
    output_keys: ClassVar[List[str]] = ["raw_kql"]

    def _call(self, inputs):
        print("▶️ KQLGeneratorChain._call inputs:", inputs)
        u = inputs["user_prompt"]
        # inject explicit range for KQL
        prompt_req = f"{u} from {inputs['start_date']} to {inputs['end_date']}"
        kql = generate_kql(prompt_req)
        # apply your existing quick fixes
        kql = format_dates(kql)
        
        # remove any remaining nested datetime()
        kql = re.sub(
            r'datetime\s*\(\s*datetime\s*\(\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\)\s*\)',
            r'datetime(\1)',
            kql,
            flags=re.IGNORECASE,
        )

        # your existing quick‐fixes
        kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
        kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)
        # map GSBER if territory mentioned
        for terr, code in GSBER_MAPPING.items():
            if terr.lower() in u.lower():
                kql = re.sub(r"where Territory == .+?", f"where gsber == '{code}'", kql)
                break
        # adjust for trend
        trend = detect_trend(u)
        if trend == "increasing":
            kql = kql.replace("RevenueChange < 0", "RevenueChange > 0")
        elif trend == "declining":
            kql = kql.replace("RevenueChange < 0", "RevenueChange < 0")
        else:
            kql = kql.replace("RevenueChange < 0", "RevenueChange == 0")
        print("▶️ KQLGeneratorChain raw_kql:", kql)
        return {"raw_kql": kql}

# ──────────────────────── 8. ADX-Execution Chain ─────────────────────
class ADXChain(Chain):
    print("-----------------ADXChain method called --------------------")
    input_keys: ClassVar[List[str]]  = ["raw_kql"]
    output_keys: ClassVar[List[str]] = ["cols", "rows"]

    def _call(self, inputs):
        print("▶️ ADXChain._call inputs:", inputs)
        kql = inputs["raw_kql"]
        for attempt in (1, 2):
            try:
                cols, rows = adx().run(kql)
                return {"cols": cols, "rows": rows}
            except KustoApiError:
                if attempt == 1:
                    # regenerate in strict mode
                    kql = generate_kql(inputs["raw_kql"], strict=True)
                    continue
                return {"cols": [], "rows": []}

# ──────────────────────── 9. Summarization Chain ─────────────────────
summary_prompt = PromptTemplate(
    input_variables=["user_prompt", "cols", "rows"],
    template="""
User asked: {user_prompt}

Columns: {cols}
Rows: {rows}

Format in bullets + provide concise business insight. Amounts in BDT.
"""
)

# instantiate your summarizer
summary_runnable = RunnableSequence(first=summary_prompt, last=llm)

# ─────────────────────── KQL‐RunnableChain Wrapper ──────────────────────

class KQLRunnableChain(Chain):
    input_keys:  ClassVar[List[str]] = ["user_prompt", "start_date", "end_date"]
    output_keys: ClassVar[List[str]] = ["raw_kql"]

    def _call(self, inputs):
        payload = {
            "system_prompt": SYSTEM_PROMPT_KQL,
            "user_prompt":   inputs["user_prompt"],
            "start_date":    inputs["start_date"],
            "end_date":      inputs["end_date"],
        }
        raw = kql_runnable.invoke(payload)
        print("▶️ KQLRunnableChain →", raw)
        return {"raw_kql": raw}

# ──────────────────── Summary‐RunnableChain Wrapper ────────────────────

class SummaryRunnableChain(Chain):
    input_keys:  ClassVar[List[str]] = ["user_prompt", "cols", "rows"]
    output_keys: ClassVar[List[str]] = ["summary"]

    def _call(self, inputs):
        payload = {
            "user_prompt": inputs["user_prompt"],
            "cols":        inputs["cols"],
            "rows":        inputs["rows"],
        }
        summ = summary_runnable.invoke(payload)
        print("▶️ SummaryRunnableChain →", summ)
        return {"summary": summ}
# ────────────────── 10. Combine into SequentialChain ────────────────
agent_chain = SequentialChain(
    chains=[
        DateRangeChain(),      # your date parsing
        KQLRunnableChain(),    # wraps PromptTemplate | llm for KQL
        ADXChain(),            # executes the KQL
        SummaryRunnableChain() # wraps PromptTemplate | llm for summary
    ],
    input_variables  = ["user_prompt"],
    output_variables = ["summary"],
    verbose=False
)
# ────────────────────────── 11. Entry Function ───────────────────────
def handle_user_query(user_prompt: str, *, conversation_id: str | None = None) -> str:
    try:

        print("---------------user_prompt--------------",user_prompt)
        # 1. Run your core chain via invoke(), not __call__
        out = agent_chain.invoke({"user_prompt": user_prompt})

        print("---------------out --------------",out)
        cols, rows = out["cols"], out["rows"]

        if not rows:
            return "No data found matching your criteria. Please refine your query for more specific results."

        # 2. Summarize via the summary_runnable as before
        formatted = summary_runnable.invoke({
            "user_prompt": user_prompt,
            "cols": cols,
            "rows": rows,
        })

        return formatted

    except ValueError as e:
        return str(e)

    except Exception:
        return (
            "I’m sorry, something went wrong while processing your request. "
            "Please try rephrasing or specifying a clearer date range."
        )


# Example:
# resp = run_sap_sales_agent("Show average sales by month for last year for Dhaka Sales")
# print(resp)
