import uuid
from django.shortcuts import get_object_or_404
from conversation.models import Conversation, Message, MessageMeta
from django.db.models import Q
from typing import List, Dict, Any
import json, re

# Function to fetch Conversation ID from UUID
def get_conversation_id_from_uuid(conversation_uuid: uuid.UUID) -> int:
    # Fetch the conversation using the UUID
    conversation = get_object_or_404(Conversation, uuid=conversation_uuid)
    return conversation.id


# Function to fetch the last 20 messages of a conversation, ordered by latest message at the bottom

def get_last_20_messages(conversation_id: int):
    qs = (Message.objects
          .filter(conversation_id=conversation_id, is_deleted=False)
          .order_by('-created_at')[:20])
    # turn into a list (so we can reverse safely)
    messages = list(qs)
    messages.reverse()  # now oldest→newest
    return messages


# Function to format messages for LLM (multi-turn conversation)
def format_messages_for_llm(conversation_id: int, new_user_message: str = None, 
                           new_user_image_url: str = None) -> List[Dict[str, Any]]:
    """
    Format conversation messages for LLM API (OpenAI/Anthropic format)
    Returns list of messages in the format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # Get the last 20 messages (or adjust number as needed)
    messages = get_last_20_messages(conversation_id)
    
    formatted_messages = []
    
    # Convert Django messages to LLM format
    for message in messages:
        role = "user" if message.sender == "user" else "assistant"
        
        # Handle text and image content
        if message.image_url and message.text:
            # For messages with both text and image
            content = [
                {"type": "text", "text": message.text},
                {"type": "image_url", "image_url": {"url": message.image_url}}
            ]
        elif message.image_url:
            # For image-only messages
            content = [{"type": "image_url", "image_url": {"url": message.image_url}}]
        else:
            # For text-only messages
            content = message.text
            
        formatted_messages.append({
            "role": role,
            "content": content
        })
    
    # Add the new user message if provided
    if new_user_message:
        if new_user_image_url and new_user_message:
            # New message with both text and image
            content = [
                {"type": "text", "text": new_user_message},
                {"type": "image_url", "image_url": {"url": new_user_image_url}}
            ]
        elif new_user_image_url:
            # New message with only image
            content = [{"type": "image_url", "image_url": {"url": new_user_image_url}}]
        else:
            # New message with only text
            content = new_user_message
            
        formatted_messages.append({
            "role": "user",
            "content": content
        })
    
    return formatted_messages


# Function to fetch the last 20 message metas of a conversation, ordered by latest message_meta at the bottom
def get_last_20_message_metas(conversation_id: int):
    qs = (MessageMeta.objects
          .filter(conversation_id=conversation_id)
          .order_by('-created_at')[:20])
    metas = list(qs)
    metas.reverse()
    return metas

# ADD at top if missing


# --- Helper: strip everything from 'Business Insights' onward (case-insensitive) ---
def strip_business_insights(text: str) -> str:
    """
    Returns content ABOVE the first 'Business Insights' marker.
    Matches both a heading like '### Business Insights' and any occurrence
    of the phrase 'Business Insights' (case-insensitive).
    """
    if not text:
        return text or ""

    # 1) Prefer matching a heading-line like "### Business Insights"
    heading_re = re.compile(r'(?im)^\s*###\s*Business\s*Insights\b.*$', re.MULTILINE)
    m = heading_re.search(text)
    if m:
        return text[:m.start()].rstrip()

    # 2) Otherwise, match the phrase anywhere
    phrase_re = re.compile(r'(?i)\bBusiness\s*Insights\b')
    m = phrase_re.search(text)
    if m:
        return text[:m.start()].rstrip()

    # No marker → return full text
    return text


def serialize_context_for_llm(conversation_id: int) -> str:
    """
    Returns a compact, oldest→newest JSONL-like block of the last 20 messages
    and the last 20 metas. For messages, we include ONLY the content ABOVE
    'Business Insights' (if present).
    """
    msgs = get_last_20_messages(conversation_id)
    metas = get_last_20_message_metas(conversation_id)

    lines = []
    # Messages block
    lines.append("### MESSAGES_JSONL (oldest→newest)")
    for m in msgs:
        text_above_bi = strip_business_insights(m.text or "")
        lines.append(
            json.dumps({
                "role": "user" if m.sender == "user" else "assistant",
                "text": text_above_bi,
                "image_url": m.image_url or None,
                "created_at": m.created_at.isoformat()
            }, ensure_ascii=False)
        )

    # Metas block (oldest→newest)
    lines.append("### METAS_JSONL (oldest→newest)")
    for meta in metas:
        payload = meta.meta_json if getattr(meta, "meta_json", None) else None
        if payload is None and hasattr(meta, "content"):
            payload = {"content": meta.content}
        lines.append(
            json.dumps({
                "meta": payload,
                "created_at": meta.created_at.isoformat()
            }, ensure_ascii=False)
        )

    return "\n".join(lines)


# ---- LLM context memory blocks (ADD) ----
def build_context_memory_contract() -> str:
    """
    Instruct the LLM to act like ChatGPT-style memory without hard-coded logic.
    The LLM will *infer* which dates/filters to carry, how to resolve conflicts,
    and how to output META + raw KQL.
    """
    return (
        "CONTEXT MEMORY CONTRACT:\n"
        "- You are continuing a multi-turn conversation that generates **KQL** for SAPSalesInfos.\n"
        "- You will receive:\n"
        "  • A chronological snapshot of the last 20 messages and last 20 message METAs.\n"
        "  • The user's new message.\n\n"
        "Your tasks:\n"
        "1) INFER ACTIVE CONTEXT (no hard-coded rules):\n"
        "   - From the conversation snapshot, *infer* currently active dates, filters, dimensions, and scope.\n"
        "   - Resolve conflicts by using this priority:\n"
        "     latest explicit user instruction > latest explicit assistant META > earlier context.\n"
        "   - If the new user message contradicts previous context, the new one overrides.\n"
        "   - If something is unspecified in the new message but is consistent/stable in recent turns, reuse it.\n"
        "2) PRODUCE OUTPUT IN TWO PARTS **ONLY**:\n"
        "   a) First line: `// META {json}`\n"
        "      - Compact JSON describing what you actually applied: dates, filters, dims, and any notes.\n"
        "      - Example: // META {\"dates\":{\"start\":\"YYYY-MM-DD\",\"end\":\"YYYY-MM-DD\"},\"filters\":{\"gsber\":[4110],\"spart_text\":[\"Decorative\"]}}\n"
        "   b) Then **raw KQL only**, no markdown, no commentary.\n"
        "3) TECHNICAL RULES FOR KQL:\n"
        "   - Use table SAPSalesInfos and columns from the provided schema block.\n"
        "   - Prefer `startofmonth(fkdat)` (avoid `bin(fkdat, 1mo)`).\n"
        "   - For strings use =~ / in~ (case-insensitive). For numeric use == / in (no quotes).\n"
        "   - End every KQL statement with a semicolon.\n"
        "   - If you define StartDate/EndDate, declare them explicitly as:\n"
        "       let StartDate = datetime(YYYY-MM-DD);\n"
        "       let EndDate   = datetime(YYYY-MM-DD);\n"
        "   - If user has *no* authorization for an explicitly requested area (see USER_AREA_SCOPE), output only:\n"
        "       print ErrorMessage = 'sorry you have no authorized to view this data.';\n"
        "     (and still include a META line explaining the authorization miss.)\n"
        "4) BE CONSISTENT:\n"
        "   - Keep column names and filters consistent across turns unless told otherwise.\n"
        "   - If prior turns established a dimension or date style and the user didn't change it, reuse it.\n"
        "5) AMBIGUITY:\n"
        "   - When ambiguous, pick the *most recent coherent* context from the snapshot, not older ones.\n"
        "   - Avoid inventing unsupported filters; when in doubt, keep filters minimal and reflect only what you inferred in META.\n"
    )

def build_conversation_snapshot_block(conversation_uuid: str | None) -> str:
    if not conversation_uuid:
        return "### MESSAGES_JSONL (none)\n### METAS_JSONL (none)"
    try:
        conv_id = get_conversation_id_from_uuid(conversation_uuid)
        return serialize_context_for_llm(conv_id)
    except Exception as e:
        return f"### MESSAGES_JSONL (error: {e})\n### METAS_JSONL (none)"





# import uuid
# from django.shortcuts import get_object_or_404
# from conversation.models import Conversation, Message, MessageMeta
# from django.db.models import Q

# # Function to fetch Conversation ID from UUID
# def get_conversation_id_from_uuid(conversation_uuid: uuid.UUID) -> int:
#     # Fetch the conversation using the UUID
#     conversation = get_object_or_404(Conversation, uuid=conversation_uuid)
#     return conversation.id


# # Function to fetch the last 20 messages of a conversation, ordered by latest message at the bottom
# def get_last_20_messages(conversation_id: int):
#     # Fetch the last 20 messages for the given conversation id
#     messages = Message.objects.filter(conversation_id=conversation_id, is_deleted=False) \
#                                .order_by('-created_at')[:20] \
#                                .order_by('created_at')
#     return messages


# # Function to fetch the last 20 message metas of a conversation, ordered by latest message_meta at the bottom
# def get_last_20_message_metas(conversation_id: int):
#     # Fetch the last 20 message metas for the given conversation id
#     message_metas = MessageMeta.objects.filter(conversation_id=conversation_id) \
#                                        .order_by('-created_at')[:20] \
#                                        .order_by('created_at')
#     return message_metas