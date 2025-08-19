# src/agent/ai/graph.py
from typing import TypedDict, Annotated, Optional, List, Dict
from django.conf import settings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
import json
import re
import logging

from agent.agent import generate_kql, adx, GSBER_MAPPING
from .frame_types import Frame, FrameDelta

logger = logging.getLogger(__name__)

# ---------- LLMs ----------
intent_llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0,
)
# Use function_calling for robust structured output
delta_llm = intent_llm.with_structured_output(FrameDelta, method="function_calling")

summary_llm = AzureChatOpenAI(
    azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
    api_key          = settings.AZURE_OPENAI_KEY,
    api_version      = "2025-01-01-preview",
    azure_deployment = settings.AZURE_OPENAI_DEPLOYMENT,
    temperature      = 0.2,
)

# ---------- Graph State ----------
class ConvState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    frame: Frame
    kql: Optional[str]
    rows: Optional[List[Dict]]
    meta: Optional[Dict]
    error: Optional[str]
    text: Optional[str]

INTENT_SYS = SystemMessage(content=
"""You are a sales analytics intent parser for SAP sales on ADX.

Given the user's latest message and the prior frame, output ONLY a JSON delta with fields that changed.
- Do not include fields you intend to keep from prior context.
- Never output KQL.

Canonical metrics (use these exact values if you set metric):
- Revenue  (synonyms: sales, sale, amount, turnover, net sales, gross sales)
- Volume   (synonyms: quantity, qty, units, unit, volum)
- InvoiceCount (synonyms: invoices, invoice count, bills, orders, order count)
- AvgSellingPrice (synonyms: asp, avg price, average selling price, average unit price)

Canonical group_by values:
- dealer (synonyms: customer, client, cname, dealer)
- brand (wgbez, brand)
- product (product name, product, material, sku, item, arktx, matnr)
- depo (business area, gsber, sales office)
- zone (sales zone, szone, zone)
- date (date, day, month, period, timeperiod, fkdat, time)
"""
)

# USER-FACING SUMMARY (no headers / no debug labels)
SUMMARY_SYS = SystemMessage(content=
"""Write a concise, user-facing summary of sales results:

- Do NOT include any title/header.
- Do NOT include the labels "Scope", "Metric", "Grouping", "Top N", or "Total Rows".
- Currency is BDT. Use short bullet points.
- If show_scope=false, DO NOT mention any scope (division/depo/zone/territory/brand/product) even if present in the frame.
- If show_scope=true, you may naturally mention the active filters (e.g., "Marine Paints" or "Dhaka Sales") in the text, but don't print them as separate labeled lines.
- Always include 1–2 crisp business insights at the end.
- Do not echo KQL or internal JSON. Only produce the final text the user should see.
"""
)

# ---------- Normalization helpers ----------
def _canon(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return " ".join(str(s).lower().split())

_METRIC_MAP = {
    # Revenue
    "revenue": "Revenue", "sales": "Revenue", "sale": "Revenue", "amount": "Revenue",
    "turnover": "Revenue", "net sales": "Revenue", "gross sales": "Revenue",
    # Volume
    "volume": "Volume", "volum": "Volume", "quantity": "Volume", "qty": "Volume",
    "units": "Volume", "unit": "Volume",
    # InvoiceCount
    "invoicecount": "InvoiceCount", "invoice count": "InvoiceCount",
    "invoices": "InvoiceCount", "bills": "InvoiceCount",
    "orders": "InvoiceCount", "order count": "InvoiceCount",
    # AvgSellingPrice
    "avgsellingprice": "AvgSellingPrice", "avg selling price": "AvgSellingPrice",
    "average selling price": "AvgSellingPrice", "avg price": "AvgSellingPrice",
    "average unit price": "AvgSellingPrice", "asp": "AvgSellingPrice",
}
_GROUPBY_MAP = {
    "dealer": "dealer", "customer": "dealer", "client": "dealer", "cname": "dealer",
    "brand": "brand", "wgbez": "brand",
    "product": "product", "product name": "product", "material": "product",
    "sku": "product", "item": "product", "arktx": "product", "matnr": "product",
    "depo": "depo", "business area": "depo", "gsber": "depo", "sales office": "depo",
    "zone": "zone", "sales zone": "zone", "szone": "zone",
    "date": "date", "day": "date", "month": "date", "period": "date",
    "timeperiod": "date", "fkdat": "date", "time": "date",
}

def normalize_metric(maybe_metric: Optional[str]) -> Optional[str]:
    key = _canon(maybe_metric);  return _METRIC_MAP.get(key) if key else None

def normalize_group_by(maybe_group: Optional[str]) -> Optional[str]:
    key = _canon(maybe_group);   return _GROUPBY_MAP.get(key) if key else None

def deep_merge_frame(base: Frame, delta: FrameDelta) -> Frame:
    data = base.model_dump()
    d = (delta or FrameDelta()).model_dump(exclude_none=True)

    # Normalize metric & group_by BEFORE merging into strict Frame
    if "metric" in d:
        nm = normalize_metric(d.get("metric"))
        if nm: d["metric"] = nm
        else:  d.pop("metric", None)  # keep previous

    if "group_by" in d:
        ng = normalize_group_by(d.get("group_by"))
        if ng: d["group_by"] = ng
        else:  d.pop("group_by", None)  # keep previous

    # Shallow merge dicts
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v

    return Frame(**data)

# ---------- Column types from your ADX schema ----------
STRING_COLS = {
    "matkl","wgbez","matnr","vkgrp_c","arktx","meins","voleh",
    "Territory","Szone","cname","spart_text","ktokd","GK"
}
LONG_COLS = {
    "bukrs","spart","vkorg","kunrg","kunnr_sh","Payer_DL","vbeln",
    "vkbur_c","kukla","posnr","gsber","fkimg","vtweg","kkber"
}
REAL_COLS = {"Revenue","volum"}
DATETIME_COLS = {"fkdat","FKDAT_TEMP"}
TIMESPAN_COLS = {"erzet_T"}

# ---------- Allowed WHERE columns ----------
def _mentions_any(s: str, words: List[str]) -> bool:
    s = (s or "").lower()
    return any(w in s for w in words)

def allowed_where_columns(frame: Frame, user_text: str) -> List[str]:
    """
    WHERE may only reference columns:
      - Always: fkdat (date window)
      - Area/entity only if present in prior frame OR explicitly asked in user text.
    """
    s = (user_text or "").lower()
    allow = set(["fkdat"])

    # Area filters
    if frame.scope.depo or _mentions_any(s, ["depo","business area","gsber","sales office"]):
        allow.add("gsber")
    if getattr(frame.scope, "zone", None) or _mentions_any(s, ["zone","sales zone","szone"]):
        allow.add("Szone")
    if getattr(frame.scope, "territory", None) or _mentions_any(s, ["territory"]):
        allow.add("Territory")

    # Division (spart_text) only if asked
    if _mentions_any(s, ["division","spart_text","decorative","industrial","protective","marine paints"]):
        allow.add("spart_text")

    # Brand / Product
    if frame.entities.brand or _mentions_any(s, ["brand","wgbez"]):
        allow.add("wgbez")
    if frame.entities.product or _mentions_any(s, ["product","product name","material","sku","arktx","matnr"]):
        allow.update(["arktx","matnr"])

    # Dealer (rarely filtered; usually grouped)
    if _mentions_any(s, ["dealer ","customer ","client ","cname "]):
        allow.add("cname")

    # Other enterprise columns only if explicitly asked
    if _mentions_any(s, ["company code","bukrs"]): allow.add("bukrs")
    if _mentions_any(s, ["sales org","vkorg"]):    allow.add("vkorg")
    if _mentions_any(s, ["dist channel","distribution channel","vtweg"]): allow.add("vtweg")
    if _mentions_any(s, ["credit control area","kkber"]): allow.add("kkber")
    if _mentions_any(s, ["dealer group","kukla"]): allow.add("kukla")
    if _mentions_any(s, ["account group","ktokd"]): allow.add("ktokd")
    if _mentions_any(s, ["sales group","vkgrp_c"]): allow.add("vkgrp_c")
    if _mentions_any(s, ["sales office","vkbur_c"]): allow.add("vkbur_c")
    if _mentions_any(s, ["payer id","payer","payer_dl"]): allow.add("Payer_DL")
    if _mentions_any(s, ["category","matkl"]): allow.add("matkl")
    if _mentions_any(s, ["business group","gk"]): allow.add("GK")
    if _mentions_any(s, ["unit","meins"]): allow.add("meins")
    if _mentions_any(s, ["volume unit","voleh"]): allow.add("voleh")

    return list(allow)

# ---------- Sanitize WHERE lines to only allowed columns ----------
_AND_SPLIT = re.compile(r"\s+and\s+", re.IGNORECASE)

def enforce_allowed_filters(kql: str, allowed_cols: List[str]) -> str:
    """Keep only predicates whose leftmost column is in allowed_cols or refers to Start/EndDate/fkdat."""
    if not kql:
        return kql
    lines = kql.splitlines()
    out_lines = []
    for line in lines:
        if "| where" not in line:
            out_lines.append(line)
            continue

        head, tail = line.split("| where", 1)
        preds_raw = tail.strip()
        parts = _AND_SPLIT.split(preds_raw)

        kept = []
        for p in parts:
            p_stripped = p.strip()

            # always allow date conditions & Start/EndDate refs
            if p_stripped.lower().startswith("fkdat "):
                kept.append(p_stripped);  continue
            if "StartDate" in p_stripped or "EndDate" in p_stripped:
                kept.append(p_stripped);  continue

            # Extract leftmost identifier (column name)
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*[=~<>i]*", p_stripped)
            col = m.group(1) if m else None

            if col and col in allowed_cols:
                kept.append(p_stripped)
            # else drop silently

        if kept:
            out_lines.append(f"{head}| where " + " and ".join(kept))
        # if nothing kept, drop the where line entirely
    return "\n".join(out_lines)

# ---------- Type-aware operator & quoting fixes ----------
def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _is_integer(s: str) -> bool:
    s = s.strip().strip('"').strip("'")
    return bool(re.fullmatch(r"[0-9]+", s))

def _quote(s: str) -> str:
    s = _strip_quotes(s)
    return f'"{s}"'

def _unquote_number_if_needed(s: str) -> str:
    inner = _strip_quotes(s)
    return inner if _is_integer(inner) else s

def _fix_in_list_for_string(content: str) -> str:
    parts = [p.strip() for p in content.split(",") if p.strip()]
    parts = [_quote(_strip_quotes(p)) for p in parts]
    return ", ".join(parts)

def _fix_in_list_for_long(content: str) -> str:
    parts = [p.strip() for p in content.split(",") if p.strip()]
    parts = [_unquote_number_if_needed(p) for p in parts]
    return ", ".join(parts)

def apply_dtype_fixes(kql: str) -> str:
    """Make operators & literals match the ADX column data types."""
    if not kql:
        return kql

    # --- STRING columns: force =~ and in~ with quoted values ---
    for col in STRING_COLS:
        pattern_eq = re.compile(rf'(\b{re.escape(col)}\b)\s*==\s*(".*?"|\S+)', re.IGNORECASE)
        kql = pattern_eq.sub(lambda m: f'{m.group(1)} =~ {_quote(m.group(2))}', kql)

        pattern_in = re.compile(rf'(\b{re.escape(col)}\b)\s+in\s*\(([^)]*)\)', re.IGNORECASE)
        kql = pattern_in.sub(lambda m: f'{m.group(1)} in~ ({_fix_in_list_for_string(m.group(2))})', kql)

        pattern_in_tilde = re.compile(rf'(\b{re.escape(col)}\b)\s+in~\s*\(([^)]*)\)', re.IGNORECASE)
        kql = pattern_in_tilde.sub(lambda m: f'{m.group(1)} in~ ({_fix_in_list_for_string(m.group(2))})', kql)

    # --- LONG columns: force == and in; remove quotes for numeric literals ---
    for col in LONG_COLS:
        pattern_eq_tilde = re.compile(rf'(\b{re.escape(col)}\b)\s*=~\s*(".*?"|\S+)', re.IGNORECASE)
        kql = pattern_eq_tilde.sub(lambda m: f'{m.group(1)} == {_unquote_number_if_needed(m.group(2))}', kql)

        pattern_eq_qnum = re.compile(rf'(\b{re.escape(col)}\b)\s*==\s*"([0-9]+)"', re.IGNORECASE)
        kql = pattern_eq_qnum.sub(lambda m: f'{m.group(1)} == {m.group(2)}', kql)

        pattern_in_tilde = re.compile(rf'(\b{re.escape(col)}\b)\s+in~\s*\(([^)]*)\)', re.IGNORECASE)
        kql = pattern_in_tilde.sub(lambda m: f'{m.group(1)} in ({_fix_in_list_for_long(m.group(2))})', kql)

        pattern_in_plain = re.compile(rf'(\b{re.escape(col)}\b)\s+in\s*\(([^)]*)\)', re.IGNORECASE)
        kql = pattern_in_plain.sub(lambda m: f'{m.group(1)} in ({_fix_in_list_for_long(m.group(2))})', kql)

    return kql

# ---------- Helper: humanize used scope from KQL for logging ----------
_SCOPE_LABELS = {
    "gsber": "Depo/Sales Office",
    "Szone": "Sales Zone",
    "Territory": "Territory",
    "spart_text": "Division",
    "wgbez": "Brand",
    "arktx": "Product",
    "matnr": "Product Code",
    "cname": "Dealer",
    "bukrs": "Company Code",
    "vkorg": "Sales Org",
    "vtweg": "Dist Channel",
    "kkber": "Credit Ctrl Area",
    "kukla": "Dealer Group",
    "ktokd": "Account Group",
    "vkgrp_c": "Sales Group",
    "vkbur_c": "Sales Office",
    "Payer_DL": "Payer ID",
    "matkl": "Category",
    "GK": "Business Group",
    "meins": "Unit",
    "voleh": "Volume Unit",
}

SCOPE_COLS_SET = set(_SCOPE_LABELS.keys())

def _parse_start_end(kql: str) -> Dict[str, Optional[str]]:
    start = None; end = None
    m = re.search(r'let\s+StartDate\s*=\s*datetime\((\d{4}-\d{2}-\d{2})\)\s*;', kql, re.IGNORECASE)
    if m: start = m.group(1)
    m = re.search(r'let\s+EndDate\s*=\s*datetime\((\d{4}-\d{2}-\d{2})\)\s*;', kql, re.IGNORECASE)
    if m: end = m.group(1)
    return {"start": start, "end": end}

def _extract_used_filters(kql: str) -> Dict[str, List[str]]:
    """
    Return dict col -> list of values used in WHERE (eq or in).
    Only for columns we consider as "scope-ish" (_SCOPE_LABELS).
    """
    used: Dict[str, List[str]] = {}
    if not kql:
        return used

    for col in SCOPE_COLS_SET:
        # equality == or =~
        p_eq = re.compile(rf'\b{re.escape(col)}\b\s*[=~]{{1,2}}\s*(".*?"|\S+)', re.IGNORECASE)
        for m in p_eq.finditer(kql):
            val = _strip_quotes(m.group(1))
            used.setdefault(col, []).append(val)

        # membership in(...) or in~(...)
        p_in = re.compile(rf'\b{re.escape(col)}\b\s+in~?\s*\(([^)]*)\)', re.IGNORECASE)
        for m in p_in.finditer(kql):
            content = m.group(1)
            parts = [p.strip() for p in content.split(",") if p.strip()]
            parts = [_strip_quotes(p) for p in parts]
            if parts:
                used.setdefault(col, []).extend(parts)

    # Deduplicate values per column
    for k in list(used.keys()):
        seen = []
        for v in used[k]:
            if v not in seen:
                seen.append(v)
        used[k] = seen
    return used

def _humanize_scope(used: Dict[str, List[str]]) -> List[str]:
    """Turn used scope dict into readable parts for logging."""
    out = []
    for col, vals in used.items():
        label = _SCOPE_LABELS.get(col, col)
        if col == "gsber":
            friendly = []
            for v in vals:
                # try to map code to name
                name = next((k for k, code in GSBER_MAPPING.items() if str(code) == str(v)), None)
                if name:
                    friendly.append(f"{name} ({v})")
                else:
                    friendly.append(str(v))
            out.append(f"{label}: {', '.join(friendly)}")
        else:
            out.append(f"{label}: {', '.join(vals)}")
    return out

# ---------- Helpers ----------
def frame_defaults_as_hints(frame: Frame) -> str:
    bits = []
    if frame.date.start and frame.date.end:
        bits.append(f"Date range: {frame.date.start} to {frame.date.end}")
    if frame.group_by:
        bits.append(f"Group by: {frame.group_by}")
    if frame.limit:
        bits.append(f"Top N: {frame.limit}")
    return "Assume these defaults if not stated:\n- " + "\n- ".join(bits) if bits else ""

# ---------- Nodes ----------
def n_extract_delta(state: ConvState) -> ConvState:
    prior = state["frame"]
    user_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    delta = delta_llm.invoke([
        INTENT_SYS,
        HumanMessage(content=f"PRIOR_FRAME:\n{prior.model_dump_json()}\n\nUSER:\n{user_text}")
    ])
    state["messages"].append(AIMessage(content=f"[delta]{delta.model_dump_json()}"))
    state["frame"] = deep_merge_frame(prior, delta)
    return state

def n_gen_kql(state: ConvState) -> ConvState:
    if state.get("error"):
        return state

    frame = state["frame"]
    user_text = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx = frame_defaults_as_hints(frame)

    # Allowed WHERE columns + HARD CONSTRAINTS
    allow = allowed_where_columns(frame, user_text)

    type_hints = (
        "Column types:\n"
        f"- LONG columns: {', '.join(sorted(LONG_COLS))}\n"
        f"- STRING columns: {', '.join(sorted(STRING_COLS))}\n"
        f"- DATETIME columns: {', '.join(sorted(DATETIME_COLS))}\n"
        "- Rules: For LONG use == / in with numeric literals (no quotes). "
        "For STRING use =~ / in~ with quoted values (case-insensitive). "
        "For DATETIME use >=, <= with datetime(...) literals.\n"
    )

    hard_constraints = (
        "HARD CONSTRAINTS (must follow exactly):\n"
        f"- WHERE may only reference these columns: {', '.join(allow)}\n"
        "- Do NOT add any other filters. If a filter is not explicitly requested by the user "
        "or present in the prior frame, omit it.\n"
        "- Use the operator rules from the Column types guidance above.\n"
    )

    effective_prompt = user_text
    if ctx:
        effective_prompt += f"\n\n{ctx}"
    effective_prompt += f"\n\n{type_hints}{hard_constraints}"

    try:
        raw_kql = generate_kql(effective_prompt)
        # 1) Strip predicates for disallowed columns
        kql = enforce_allowed_filters(raw_kql, allow)
        # 2) Fix operators & quoting by ADX data types
        kql = apply_dtype_fixes(kql)
        state["kql"] = kql
    except Exception as ex:
        state["error"] = f"generate_kql failed: {ex}"
    return state

def n_run_adx(state: ConvState) -> ConvState:
    if state.get("error"):
        return state
    try:
        cols, rows = adx().run(state["kql"])
        state["rows"] = [dict(zip(cols, r)) for r in rows]
        state["meta"] = {"rowcount": len(rows), "cols": cols}
    except Exception as ex:
        state["error"] = f"ADX error: {ex}"
    return state

def n_summarize(state: ConvState) -> ConvState:
    frame = state["frame"]
    rows = state.get("rows") or []
    meta = state.get("meta") or {}
    kql = state.get("kql") or ""

    # Derive which scope filters were ACTUALLY USED in this turn's KQL
    used_filters = _extract_used_filters(kql)            # dict of col -> values
    show_scope_to_user = bool(used_filters)              # only if current KQL has any scope-ish filters
    date_bounds = _parse_start_end(kql)                  # StartDate/EndDate if present

    # Prepare a redacted frame for the summarizer so it doesn't leak old scope
    # frame_for_summary = frame.model_dump(deep=True)
    frame_for_summary = frame.model_dump()

    if not show_scope_to_user:
        # wipe scope-ish fields so LLM doesn't mention them
        if "scope" in frame_for_summary:
            for fld in ("depo", "zone", "territory", "region", "dealer"):
                if fld in frame_for_summary["scope"]:
                    frame_for_summary["scope"][fld] = None
        # also clear entities (brand/product) to be safe
        if "entities" in frame_for_summary:
            for fld in ("brand", "product"):
                if fld in frame_for_summary["entities"]:
                    frame_for_summary["entities"][fld] = None

    # Log debug info to console (NOT shown to user)
    human_scope = _humanize_scope(used_filters)
    logger.info("KQL Date Range: %s -> %s", date_bounds.get("start"), date_bounds.get("end"))
    logger.info("Metric (frame): %s", frame.metric)
    if human_scope:
        logger.info("Active Scope: %s", " | ".join(human_scope))
    logger.info("Grouping: %s | Top N: %s | Total Rows: %s",
                frame.group_by, frame.limit, meta.get("rowcount"))

    # Build concise, user-facing summary
    preview = rows[:50]
    msg = summary_llm.invoke([
        SUMMARY_SYS,
        HumanMessage(content=json.dumps({
            "show_scope": show_scope_to_user,     # governs whether scope can be mentioned
            "date_range": date_bounds,            # used to mention dates cleanly
            "frame": frame_for_summary,           # redacted when needed
            "meta": meta,
            "rows_sample": preview
        }))
    ])
    state["messages"].append(AIMessage(content=msg.content))
    state["text"] = msg.content
    return state

def build_graph():
    g = StateGraph(ConvState)
    g.add_node("extract_delta", n_extract_delta)
    g.add_node("gen_kql", n_gen_kql)
    g.add_node("run_adx", n_run_adx)
    g.add_node("summarize", n_summarize)

    g.set_entry_point("extract_delta")
    g.add_edge("extract_delta", "gen_kql")
    g.add_edge("gen_kql", "run_adx")
    g.add_edge("run_adx", "summarize")
    g.add_edge("summarize", END)
    return g
