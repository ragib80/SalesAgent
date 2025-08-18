from langchain.schema import HumanMessage
from .graph import build_graph
from .frame_types import Frame
from conversation.models.conversation import Conversation

graph = build_graph().compile()  # no checkpointer

DEFAULT_FRAME = Frame()

def _load_frame(conversation_id: str) -> Frame:
    conv = Conversation.objects.filter(uuid=conversation_id, is_deleted=False).first()
    if conv and conv.frame_json:
        try:
            return Frame(**conv.frame_json)
        except Exception:
            return DEFAULT_FRAME
    return DEFAULT_FRAME

def _save_frame(conversation_id: str, frame: Frame, kql: str | None, meta: dict | None):
    Conversation.objects.filter(uuid=conversation_id, is_deleted=False).update(
        frame_json=frame.model_dump(),
        last_kql=kql or "",
        last_result_meta=meta or {}
    )

def chat_turn(conversation_id: str, user_text: str):
    initial_frame = _load_frame(conversation_id)

    inputs = {
        "messages": [HumanMessage(content=user_text)],
        "frame": initial_frame,
        "kql": None, "rows": None, "meta": None, "error": None, "text": None
    }
    out = graph.invoke(inputs)

    # Persist the updated frame and meta to MS SQL
    new_frame = out.get("frame") or initial_frame
    _save_frame(conversation_id, new_frame, out.get("kql"), out.get("meta"))

    return {
        "answer": out.get("text") or "",
        "kql": out.get("kql"),
        "meta": out.get("meta"),
        "result": out.get("rows"),
        "frame": new_frame.model_dump()
    }
