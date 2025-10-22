# conversation_helpers.py

from typing import Optional
import uuid
from django.shortcuts import get_object_or_404
from conversation.models import Conversation, Message, MessageMeta
from django.db.models import Q
from typing import List, Dict, Any
import json, re
<<<<<<< HEAD

=======
import random
# (you already have these)
# from django.shortcuts import get_object_or_404
# from conversation.models import Conversation, Message, MessageMeta
# from .your_existing_imports import ...
>>>>>>> dev_adx_prompt_help_copilot
def get_conversation_id_from_uuid(conversation_uuid: uuid.UUID) -> int:
    # Fetch the conversation using the UUID
    conversation = get_object_or_404(Conversation, uuid=conversation_uuid)
    return conversation.id


# Function to fetch the last 20 messages of a conversation, ordered by latest message at the bottom

def get_last_20_messages(conversation_id: int):
    qs = (Message.objects
          .filter(conversation_id=conversation_id, is_deleted=False)
          .order_by('-created_at')[:20])
    # turn into a list 
    messages = list(qs)
    messages.reverse()  #  oldest→newest
    return messages

def get_last_n_messages(conversation_id: int, limit: int = 20):
    """
    Fetch the last N messages of a conversation (default is 20).
    Returns messages ordered from oldest → newest.
    """
    qs = (
        Message.objects
        .filter(conversation_id=conversation_id, is_deleted=False)
        .order_by('-created_at')[:limit]
    )
    
    messages = list(qs)
    messages.reverse()  # Convert to chronological order (oldest first)
    return messages

def get_random_messages(conversation_id: int, limit: int = 20):
    """
    Faster: Fetch random messages directly via DB without loading all IDs.
    Trims assistant messages to 150 chars.
    """
    #  Fast DB-side random selection
    random_messages = (
        Message.objects
        .filter( is_deleted=False)
        .only('id', 'text', 'sender')  # Load only needed fields
        .order_by("?")[:limit]         # DB random
    )

    messages = list(random_messages)

    # Trim assistant messages
    for msg in messages:
        if msg.sender == "assistant" and msg.text:
            if len(msg.text) > 150:
                msg.text = msg.text[:150] + "..."

    return messages

# def get_random_messages(conversation_id: int, limit: int = 20):
#     """
#     Fetch `limit` number of random messages from a conversation.
#     Does NOT guarantee chronological order.
#     """
#     # Step 1: Get all message IDs from this conversation
#     message_ids = list(
#         Message.objects
#         .filter(is_deleted=False)
#         .values_list('id', flat=True)
#     )

#     if not message_ids:
#         return []

#     # Step 2: Randomly pick message IDs (no duplicates)
#     selected_ids = random.sample(message_ids, min(len(message_ids), limit))

#     # Step 3: Get full message objects
#     random_messages = list(
#         Message.objects.filter(id__in=selected_ids)
#     )

#     return random_messages


# Function to format messages for LLM (multi-turn conversation)
def format_messages_for_llm(conversation_id: int, new_user_message: str = None, 
                           new_user_image_url: str = None) -> List[Dict[str, Any]]:
    """
    Format conversation messages for LLM API (OpenAI/Anthropic format)
    Returns list of messages in the format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # Get the last 20 messages 
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
    
    # Add the new user message 
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

# 1) Strip everything from “Business Insights” downward (case-insensitive)

def strip_business_insights(text: str) -> str:
    """
    Return only the content ABOVE the first 'Business Insights' marker.
    Matches both '### Business Insights' heading and any occurrence of the phrase.
    No truncation anywhere else.
    """
    if not text:
        return text or ""

    # Prefer heading match like: "### Business Insights"
    heading_re = re.compile(r'(?im)^\s*###\s*Business\s*Insights\b.*$', re.MULTILINE)
    m = heading_re.search(text)
    if m:
        return text[:m.start()].rstrip()

    # Otherwise match the phrase anywhere
    phrase_re = re.compile(r'(?i)\bBusiness\s*Insights\b')
    m = phrase_re.search(text)
    if m:
        return text[:m.start()].rstrip()

    return text



# 2) Build a compact, oldest→newest snapshot for the LLM (JSONL-like)

def serialize_context_for_llm(conversation_id: int) -> str:
    """
    Returns a JSONL-like snapshot:
    - ### MESSAGES_JSONL (oldest→newest): each line is a JSON object
      {role, text (above Business Insights only), image_url, created_at}
    - ### METAS_JSONL (oldest→newest): each line is a JSON object
      {meta, created_at}
    Absolutely NO text-length truncation.
    """
    msgs = get_last_20_messages(conversation_id)
    metas = get_last_20_message_metas(conversation_id)

    lines = []

    # Messages
    lines.append("### MESSAGES_JSONL (oldest→newest)")
    for m in msgs:
        msg_text = strip_business_insights(m.text or "")
        lines.append(json.dumps({
            "role": "user" if m.sender == "user" else "assistant",
            "text": msg_text,
            "image_url": m.image_url or None,
            "created_at": m.created_at.isoformat()
        }, ensure_ascii=False))

    # Metas
    # lines.append("### METAS_JSONL (oldest→newest)")
    # for meta in metas:
    #     payload = meta.meta_json if getattr(meta, "meta_json", None) else None
    #     # Some older rows might have 'content' instead of meta_json
    #     if payload is None and hasattr(meta, "content"):
    #         payload = {"content": meta.content}
    #     lines.append(json.dumps({
    #         "meta": payload,
    #         "created_at": meta.created_at.isoformat()
    #     }, ensure_ascii=False))

    return "\n".join(lines)



# 3) Tiny helper used by agent.py to embed a snapshot in the prompt

def build_conversation_snapshot_block(conversation_uuid: Optional[str]) -> str:
    if not conversation_uuid:
        return "### MESSAGES_JSONL (none)"
    try:
        conv_id = get_conversation_id_from_uuid(conversation_uuid)
        return serialize_context_for_llm(conv_id)
    except Exception as e:
        return f"### MESSAGES_JSONL (error: {e})"


# 4) “Context Memory Contract” — tells the LLM how to behave like ChatGPT memory

def build_context_memory_contract() -> str:
    return (
        "CONTEXT MEMORY CONTRACT:\n"
        "- You are continuing a multi-turn conversation that generates **KQL** for SAPSalesInfos.\n"
        "- You will receive:\n"
        "  • A chronological snapshot of the last 20 messages.\n"
        "  • The user's new message.\n\n"
        "Your tasks:\n"
        "1) INFER ACTIVE CONTEXT (no hard-coded rules):\n"
        "   - From the message history, infer currently active dates, filters, dimensions, and scope.\n"
        "   - Resolve conflicts by priority:\n"
        "     latest explicit user instruction > earlier assistant reply > earlier user instruction.\n"
        "   - If the new message contradicts previous context, the new one overrides.\n"
        "   - If unspecified but stable recently, reuse it.\n"
        "   - Ignore any prior `// META {}` lines in the conversation — they are annotations only.\n"
        "2) OUTPUT FORMAT (strict):\n"
        "   a) First line: `// META {json}` — compact JSON of what you actually applied\n"
        "      e.g. // META {\"dates\":{\"start\":\"YYYY-MM-DD\",\"end\":\"YYYY-MM-DD\"},\"filters\":{\"gsber\":[4110],\"spart_text\":[\"Decorative\"]}}\n"
        "   b) Then **RAW KQL ONLY**, no markdown, no commentary.\n"
        "3) TECHNICAL KQL RULES:\n"
        "   - Use SAPSalesInfos and the provided schema.\n"
        "   - Prefer `startofmonth(fkdat)` over `bin(fkdat, 1mo)`.\n"
        "   - Strings: =~ / in~ (case-insensitive). Numeric: == / in (no quotes).\n"
        "   - End every KQL statement with a semicolon.\n"
        "   - If defining dates, use:\n"
        "       let StartDate = datetime(YYYY-MM-DD);\n"
        "       let EndDate   = datetime(YYYY-MM-DD);\n"
        "   - If the user requests an area outside allowed scope (see USER_AREA_SCOPE), output only:\n"
        "       print ErrorMessage = 'sorry you have no authorized to view this data.';\n"
        "     (Still produce a META line describing the authorization miss.)\n"
        "4) CONSISTENCY:\n"
        "   - Keep columns/filters consistent across turns unless told otherwise.\n"
        "5) AMBIGUITY:\n"
        "   - Prefer the most recent coherent context; avoid inventing filters.\n"
    )

