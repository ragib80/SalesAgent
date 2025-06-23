# sales/agent.py

import json
from datetime import datetime
from django.conf import settings
from .azure_clients import search_client, openai_client

# 1) Synonyms map: business terms → index fields
FIELD_SYNONYMS = {
    "revenue":              "Revenue",
    "quantity":             "fkimg",
    "volume":               "volum",
    "customer":             "cname",
    "brand":                "wgbez",
    "product name":         "arktx",
    "product":              "arktx",
    "category":             "matkl",
    "division":             "spart_text",
    "company code":         "bukrs",
    "sales org":            "vkorg",
    "dist channel":         "vtweg",
    "distribution channel": "vtweg",
    "business area":        "gsber",
    "credit control area":  "kkber",
    "customer group":       "kukla",
    "account group":        "ktokd",
    "sales group":          "vkgrp_c",
    "sales office":         "vkbur_c",
    "payer id":             "Payer_DL",
    "product code":         "matnr",
    "unit":                 "meins",
    "volume unit":          "voleh",
    "business group":       "GK",
    "territory":            "Territory",
    "sales zone":           "Szone",
    "date":                 "fkdat",
    "fkdat":                "fkdat",
    "cost":                 "Cost"
}

def map_filters(raw: dict) -> dict:
    """
    Only map keys present in FIELD_SYNONYMS; ignore others.
    Returns dict of index_field -> value.
    """
    mapped = {}
    for k, v in (raw or {}).items():
        lk = k.strip().lower()
        if lk in FIELD_SYNONYMS:
            mapped[FIELD_SYNONYMS[lk]] = v
    return mapped

def parse_user_prompt(prompt: str) -> dict:
    """
    Use Azure OpenAI to extract:
      - measures, filters, analysis_type, compare, year, month, date_from, date_to
    Default date_from=2024-01-01, date_to=today if absent.
    """
    system = f"""
You are a JSON extractor for SAP Sales analytics.
Use this mapping:
{json.dumps(FIELD_SYNONYMS, indent=2)}

Output only a JSON object with keys:
- measures: list of strings
- filters: dictionary of business-term→value
- analysis_type: one of total|monthly|trend|declining|profitability|profit_loss|deterioration_rate|comparison
- compare: optional "Field:ValueA vs ValueB"
- year: integer or null
- month: integer (1–12) or null
- date_from: "YYYY-MM-DD" or null
- date_to:   "YYYY-MM-DD" or null
"""
    resp = openai_client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":prompt}]
    )
    intent = json.loads(resp.choices[0].message.content)
    intent["date_from"] = intent.get("date_from") or "2024-01-01"
    intent["date_to"]   = intent.get("date_to")   or datetime.utcnow().strftime("%Y-%m-%d")
    return intent

def build_filter(intent: dict) -> str:
    """
    Build OData $filter:
     - If year+month: fkdat >= first of month AND < first of next month
     - Else: fkdat between date_from and date_to
     - Then other filters mapped via map_filters()
    """
    clauses = []
    y, m = intent.get("year"), intent.get("month")
    if y and m:
        start = f"{y}-{m:02d}-01T00:00:00Z"
        if m == 12:
            next_start = f"{y+1}-01-01T00:00:00Z"
        else:
            next_start = f"{y}-{m+1:02d}-01T00:00:00Z"
        clauses.append(f"fkdat ge {start}")
        clauses.append(f"fkdat lt {next_start}")
    else:
        df = intent["date_from"]
        dt = intent["date_to"]
        clauses.append(f"fkdat ge {df}T00:00:00Z")
        clauses.append(f"fkdat le {dt}T23:59:59Z")

    # Map only valid business-term filters
    raw_filters = intent.get("filters", {}) or {}
    # Remove any date/year/month keys
    for drop in ("date", "fkdat", "year", "month"):
        raw_filters.pop(drop, None)

    mapped = map_filters(raw_filters)
    for fld, val in mapped.items():
        if isinstance(val, (int, float)):
            clauses.append(f"{fld} eq {val}")
        else:
            clauses.append(f"{fld} eq '{val}'")

    return " and ".join(clauses)

def azure_search_all(intent: dict) -> list:
    """
    Fetch all matching documents from ysales-index,
    paging through in batches of 1000.
    """
    filt = build_filter(intent)
    docs = []
    results = search_client.search(search_text="*", filter=filt, top=1000)
    for page in results.by_page():
        docs.extend(dict(d) for d in page)
    return docs

def sum_revenue(docs: list) -> float:
    return sum(d.get("Revenue", 0) for d in docs)

def monthly_revenue(docs: list) -> dict:
    out = {}
    for d in docs:
        key = d.get("fkdat", "")[:7]
        out[key] = out.get(key, 0) + d.get("Revenue", 0)
    return out

def trend(docs: list, intent: dict) -> dict:
    y,m = intent.get("year"), intent.get("month")
    if y and m:
        curr = sum_revenue(docs)
        prev_m = m - 1 or 12
        prev_y = y if m > 1 else y - 1
        prev_int = {**intent, "year": prev_y, "month": prev_m}
        prev_sum = sum_revenue(azure_search_all(prev_int))
        return {"current": curr, "previous": prev_sum, "trend": "up" if curr>prev_sum else "down"}
    return {}

def profitability_tool(docs: list, top_n: int = 5) -> dict:
    profs = {}
    for d in docs:
        prod = d.get("arktx", "Unknown")
        rev, cost = (d.get("Revenue",0) or 0), (d.get("Cost",0) or 0)
        p = rev - cost
        if prod not in profs:
            profs[prod] = {"product": prod, "profit": 0.0, "revenue": 0.0}
        profs[prod]["profit"]  += p
        profs[prod]["revenue"] += rev

    items = []
    for v in profs.values():
        rev = v["revenue"] or 1
        v["margin_pct"] = v["profit"] / rev * 100
        items.append(v)
    items.sort(key=lambda x: x["profit"], reverse=True)
    return {"table": items[:top_n]}

def profit_loss(docs: list) -> dict:
    total = gain = loss = 0.0
    for d in docs:
        p = (d.get("Revenue",0) or 0) - (d.get("Cost",0) or 0)
        total += p
        if p >= 0: gain += p
        else: loss += p
    return {"total_profit": total, "total_gain": gain, "total_loss": loss}

def deterioration_rate(docs: list, intent: dict) -> dict:
    y, m = intent.get("year"), intent.get("month")
    if y and m:
        curr = sum_revenue(docs)
        prev_m = m - 1 or 12
        prev_y = y if m > 1 else y - 1
        prev_int = {**intent, "year": prev_y, "month": prev_m}
        prev_sum = sum_revenue(azure_search_all(prev_int))
        if prev_sum:
            rate = (prev_sum - curr) / prev_sum * 100
            return {"previous": prev_sum, "current": curr, "deterioration_rate_pct": rate}
    return {}

def comparison(docs: list, intent: dict) -> dict:
    raw = intent.get("compare", "")
    if ":" not in raw or "vs" not in raw:
        return {}
    fld, rest = raw.split(":", 1)
    a, b = [x.strip() for x in rest.split("vs", 1)]
    def sum_for(val):
        new_int = {**intent, "filters": {**intent.get("filters", {}), fld: val}}
        return sum_revenue(azure_search_all(new_int))
    return {a: sum_for(a), b: sum_for(b)}

def summarize(results, prompt) -> str:
    system = (
        f"You are a sales analytics assistant. User asked: \"{prompt}\". "
        f"Results: {json.dumps(results, default=str)}. "
        "Return JSON with: table, summary, insights, recommendations."
    )
    resp = openai_client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role":"system","content":system},
            {"role":"assistant","content":""}
        ],
    )
    return resp.choices[0].message.content

def run_sales_agent(prompt: str) -> str:
    intent = parse_user_prompt(prompt)
    docs   = azure_search_all(intent)

    low = prompt.lower()
    if "profitable" in low or "more profit" in low:
        atype = "profitability"
    else:
        atype = intent.get("analysis_type", "total")

    if atype == "monthly":
        data = monthly_revenue(docs)
    elif atype in ("trend","declining","uptrending","downfall"):
        data = trend(docs, intent)
    elif atype == "profitability":
        data = profitability_tool(docs)
    elif atype == "profit_loss":
        data = profit_loss(docs)
    elif atype == "deterioration_rate":
        data = deterioration_rate(docs, intent)
    elif atype == "comparison":
        data = comparison(docs, intent)
    else:
        data = sum_revenue(docs)

    return summarize(data, prompt)
