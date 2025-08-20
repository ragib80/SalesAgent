# --- History (ORM + Redis cache) + char-budget packer -----------------
from typing import List, Dict, Optional
from django.core.cache import cache
import re
from django.conf import settings
# Import your models
from conversation.models.conversation import Conversation
from conversation.models.message import Message

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

def _strip_heavy(text: str, max_len: int = 1200) -> str:
    if not text:
        return ""
    text = _CODE_FENCE.sub("[[block omitted]]", text)
    text = _HEAVY_JSON.sub("{[[json omitted]]}", text)
    text = _HEAVY_ARR.sub("[[array omitted]]", text)
    text = text.strip()
    return (text[:max_len] + "…") if len(text) > max_len else text

def fetch_history_from_db(conv_uuid: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    # Ensure conversation exists and is active (uses your ActiveManager)
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
            role = "user"
        else:
            content = (r["ai_model_response"] or r["text"] or "").strip()
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
        "- Do not invent values; if information is still insufficient, prefer your existing defaults "
        "(e.g., last full month) rather than relying on partial history.\n"
        "- Never echo or summarize the history; use it silently to build the KQL.\n"
    )