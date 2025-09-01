# --- History (ORM + Redis cache) + char-budget packer -----------------
from typing import List, Dict, Optional,Any
from django.core.cache import cache
import re
from django.conf import settings
# Import your models
from conversation.models.conversation import Conversation
from conversation.models.message import Message
from collections import defaultdict
import re, json
import calendar
# from agent.agent import MAPPING_STR
# Tunables (override in Django settings if you want)
HISTORY_CACHE_TTL         = getattr(settings, "HISTORY_CACHE_TTL", 300)       # seconds
HISTORY_CACHE_MAX_MSGS    = getattr(settings, "HISTORY_CACHE_MAX_MSGS", 600)  # max msgs cached
HISTORY_PACK_MAX_CHARS    = getattr(settings, "HISTORY_PACK_MAX_CHARS", 10_000)
HISTORY_PACK_MAX_MESSAGES = getattr(settings, "HISTORY_PACK_MAX_MESSAGES", 80)

def _hist_cache_key(conv_uuid: str) -> str:
    return f"conv_hist:{conv_uuid}"

def invalidate_history_cache(conv_uuid: str) -> None:
    cache.delete(_hist_cache_key(conv_uuid))

_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_HEAVY_JSON = re.compile(r"\{[\s\S]{800,}\}")
_HEAVY_ARR  = re.compile(r"\[[\s\S]{800,}\]")

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

def _strip_heavy(text: str, max_len: int = 1200) -> str:
    if not text:
        return ""
    text = _CODE_FENCE.sub("[[block omitted]]", text)
    text = _HEAVY_JSON.sub("{[[json omitted]]}", text)
    text = _HEAVY_ARR.sub("[[array omitted]]", text)
    text = text.strip()
    return (text[:max_len] + "…") if len(text) > max_len else text

def fetch_history_from_db(conv_uuid: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    # Ensure conversation exists and is active
    Conversation.active.get(uuid=conv_uuid)

    qs = (
        Message.active
        .filter(conversation__uuid=conv_uuid, is_deleted=False)
        .order_by("created_at")  # oldest → newest
        .values("sender", "text", "ai_model_response")
    )
    if limit:
        qs = qs[:limit]

    history: List[Dict[str, str]] = []
    for r in qs:
        sender = (r["sender"] or "").lower()
        if sender == "user":
            content = (r["text"] or "").strip()
            # (optional) also trim if a pasted reply contained BI:
            content = _trim_after_business_insight(content)
            role = "user"
        else:
            # prefer ai_model_response; then trim after Business Insight(s)
            content = (r["ai_model_response"] or r["text"] or "").strip()
            content = _trim_after_business_insight(content)
            role = "assistant"

        if content:
            history.append({"role": role, "content": content})

    return history


def fetch_history(conv_uuid: Optional[str], use_cache: bool = True) -> List[Dict[str, str]]:
    if not conv_uuid:
        return []
    key = _hist_cache_key(conv_uuid)
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached
    data = fetch_history_from_db(conv_uuid, limit=HISTORY_CACHE_MAX_MSGS)
    cache.set(key, data, HISTORY_CACHE_TTL)
    return data

def pack_history_by_chars(
    history: List[Dict[str, str]],
    max_chars: int = HISTORY_PACK_MAX_CHARS,
    max_messages: int = HISTORY_PACK_MAX_MESSAGES,
) -> List[Dict[str, str]]:
    """
    Returns a slice of history (oldest→newest) that fits within max_chars,
    trimming heavy blocks. No token counting.
    """
    if not history:
        return []
    total = 0
    picked: List[Dict[str, str]] = []
    # walk newest→oldest, then reverse to keep chronological order
    for m in reversed(history):
        content = _strip_heavy(m.get("content", ""))
        if not content:
            continue
        if picked and (total + len(content) > max_chars):
            break
        picked.append({"role": m["role"], "content": content})
        total += len(content)
        if len(picked) >= max_messages:
            break
    picked.reverse()
    return picked


def build_history_prompt_block(history_msgs: List[Dict[str, str]]) -> str:
    if not history_msgs:
        return ""
    lines = []
    for m in history_msgs:
        prefix = "U" if m["role"] == "user" else "A"
        lines.append(f"{prefix}: {m['content']}")
    return (
        "CONVERSATION HISTORY (oldest→newest):\n" +
        "\n".join(lines) +
        "\n\nHISTORY USAGE POLICY:\n"
        "- Use history only to resolve missing constraints (dates, areas, dealers, brands, products) "
        "when the current request does not specify them.\n"
        "- If the current request specifies a value, it OVERRIDES history.\n"
        "- If no specific date range present in user prompt,try to get it from history.\n"
        "- Do not invent values; if information is still insufficient, prefer your existing defaults "
        "(e.g., last full month) rather than relying on partial history.\n"
        "- Never echo or summarize the history; use it silently to build the KQL.\n"
    )


_BI_MARK = re.compile(r'(^|\n)\s*#{0,6}\s*Business\s+Insights\s*:?', flags=re.I | re.M)

def _trim_before_business_insights(text: str) -> str:
    if not text:
        return ""
    m = _BI_MARK.search(text)
    return text[:m.start()].rstrip() if m else text

_BI_LINE = re.compile(
    r'(?im)^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*Business[ \t]+Insights?(?:\*\*|__)?[ \t]*:?[ \t]*$'
)

def _trim_after_business_insight(text: str) -> str:
    """
    Keep everything BEFORE the first 'Business Insight'/'Business Insights' heading (any markdown style).
    If not present, return the original text.
    """
    if not text:
        return ""
    m = _BI_LINE.search(text)
    return text[:m.start()].rstrip() if m else text


# --- 2) Build synonyms from your FIELD_MAPPINGS (no hard-coded columns)
def _synonyms_by_col() -> Dict[str, List[str]]:
    syns = defaultdict(set)
    # Use the DICT, not MAPPING_STR:
    for k, v in FIELD_MAPPINGS.items():
        syns[v].add(k.lower())
    # add the column names themselves as synonyms
    for col in list(syns.keys()):
        syns[col].add(col.lower())
    return {c: sorted(list(names)) for c, names in syns.items()}


# --- 3) Extract carry-forward values from trimmed history
_DATE_RE = re.compile(r'\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b', re.I)

def _extract_carryover_values(history_msgs: List[Dict[str, str]]) -> Dict:
    """
    Scan newest→oldest; take the most recent mention of date range + any field:value hints.
    Fields are detected as lines like 'Dealer: Delwar Paint' or 'gsber: 4110', using synonyms.
    """
    syns = _synonyms_by_col()
    filters: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    date_start = date_end = None

    for m in reversed(history_msgs):  # newest first
        text = m.get("content", "") or ""
        # date range
        if date_start is None:
            dm = _DATE_RE.search(text)
            if dm:
                date_start, date_end = dm.group(1), dm.group(2)
                sources["date_range"] = m.get("role", "assistant")

        # fields
        for col, names in syns.items():
            if col in filters:
                continue
            pat = re.compile(r'\b(?:' + "|".join(map(re.escape, names)) + r')\s*[:=]\s*([^\n,;]+)', re.I)
            mm = pat.search(text)
            if mm:
                val = mm.group(1).strip().strip('"').strip("'")
                filters[col] = val
                sources[col] = m.get("role", "assistant")

        if date_start and len(filters) >= 12:
            break

    return {"date_start": date_start, "date_end": date_end, "filters": filters, "sources": sources}

# --- 4) Decide what to actually reuse for THIS request, and produce a prompt block
def _build_carryover_block(history_msgs: List[Dict[str, str]], current_user_req: str) -> str:
    if not history_msgs:
        return ""
    carry = _extract_carryover_values(history_msgs)
    syns = _synonyms_by_col()
    req_l = (current_user_req or "").lower()

    # If the user already provided a date range, don't reuse past one
    has_dates_now = bool(_DATE_RE.search(req_l))
    reuse_date = None if has_dates_now else (
        {"start": carry["date_start"], "end": carry["date_end"]}
        if carry["date_start"] and carry["date_end"] else None
    )

    # For fields: reuse only if current request doesn't mention any synonym for that column
    reused_filters: Dict[str, str] = {}
    for col, val in carry["filters"].items():
        names = syns.get(col, [])
        if not any(name in req_l for name in names):
            reused_filters[col] = val

    if not reuse_date and not reused_filters:
        return ""

    payload = {
        "reuse_if_missing_in_current_request": True,
        "date_range": reuse_date,
        "filters": reused_filters,
    }

    # Human note listing what we're reusing
    notes = []
    if reuse_date:
        notes.append(f"- date_range: {reuse_date['start']} .. {reuse_date['end']}")
    for k, v in reused_filters.items():
        notes.append(f"- {k}: {v}")

    return (
        "CARRIED_FORWARD_HINTS (JSON):\n"
        + json.dumps(payload, ensure_ascii=False)
        + ("\n\nREUSED VALUES (for transparency):\n" + "\n".join(notes) if notes else "")
    )


#new




# Reverse map for gsber code -> name
_GSBER_REV: Dict[str, str] = {v: k for k, v in GSBER_MAPPING.items()}

def _friendly_col_name(col: str) -> str:
    """Use FIELD_MAPPINGS to produce a nice display label for a column."""
    aliases = [k for k, v in FIELD_MAPPINGS.items() if v == col]
    if not aliases:
        return col
    name = min(aliases, key=len).strip()
    return " ".join(w.capitalize() for w in name.split())

def _format_month_range(start: Optional[str], end: Optional[str]) -> Optional[str]:
    """Return 'June 2025' or '2025-06-01 to 2025-06-30' if not same month."""
    if not start or not end:
        return None
    try:
        y1, m1, d1 = map(int, start.split("-"))
        y2, m2, d2 = map(int, end.split("-"))
        if y1 == y2 and m1 == m2:
            month_name = calendar.month_name[m1]
            return f"{month_name} {y1}"
        return f"{start} to {end}"
    except Exception:
        return f"{start} to {end}"

def _resolve_gsber_values(vals: List[str]) -> List[str]:
    """Map gsber codes to human names when possible, keep originals otherwise."""
    out = []
    for v in vals or []:
        v_str = str(v).strip()
        human = _GSBER_REV.get(v_str)
        out.append(human if human else v_str)
    return out

def build_applied_context_block(meta: Dict[str, Any]) -> str:
    """
    Build a small human-readable block from LAST_KQL_META:
      - Dates (pretty)
      - Filters: friendly names; gsber -> Depo/Sales Office human name(s)
    """
    if not meta or not isinstance(meta, dict):
        return ""

    dates = meta.get("dates") or {}
    # be robust to accidental key split like 'en d'
    end_val = (dates.get("end") or dates.get("en d") or dates.get("to"))
    start_val = (dates.get("start") or dates.get("from"))
    period_text = _format_month_range(start_val, end_val)

    filters: Dict[str, List[str]] = {}
    for col, values in (meta.get("filters") or {}).items():
        vals = list(values or [])
        if col == "gsber":
            vals = _resolve_gsber_values(vals)
            label = "Depo/Sales Office"
        else:
            label = _friendly_col_name(col)
        filters[label] = vals

    lines = []
    if period_text:
        lines.append(f"- Period: {period_text}")
    for label, vals in filters.items():
        if vals:
            joined = ", ".join(vals)
            lines.append(f"- {label}: {joined}")

    return ("APPLIED CONTEXT (from KQL):\n" + "\n".join(lines)) if lines else ""

def get_messages_before_business_insights(conv_uuid: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Returns [{role, content}, ...] for a conversation, trimming everything
    after the first 'Business Insight'/'Business Insights' heading in each message.
    Oldest → newest. Respects `limit`.
    """
    return fetch_history_from_db(conv_uuid, limit=limit)

# conversation_history.py

def get_last_n_history(conv_uuid: Optional[str], n: int = 20) -> List[Dict[str, str]]:
    """
    Returns the last N messages for a conversation (already trimmed of heavy blocks
    and assistant 'Business Insights' tails via fetch_history_from_db).
    Oldest → newest.
    """
    if not conv_uuid:
        return []
    # Use cached full history, then slice safely
    full = fetch_history(conv_uuid, use_cache=True)  # oldest → newest
    if not full:
        return []
    return full[-n:]

def build_context_decision_rules() -> str:
    """
    A small, consistent instruction block that tells the LLM how to treat history:
    reuse when missing, override when present, or start fresh when clearly new.
    """
    return (
        "CONTEXT DECISION RULES:\n"
        "- Treat this like ChatGPT follow-ups.\n"
        "- If the current request omits filters (division, brand, dealer, area, date), "
        "you MAY reuse the latest values from history when they make sense.\n"
        "- If the current request specifies any value, it OVERRIDES history.\n"
        "- If the current request clearly starts a new topic (e.g., different metric/dimension), "
        "ignore unrelated historic filters.\n"
        "- Always reflect what you actually applied in the first-line META JSON.\n"
    )
