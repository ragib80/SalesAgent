from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sales_analyzer.serializers import QueryRequestSerializer, QueryResponseSerializer, ChatRequestSerializer, ChatResponseSerializer,FirstChatResponseSerializer
from agent.azure_clients import search_client, openai_client
from django.conf import settings
from django.views.generic import TemplateView
from agent.agent import handle_user_query
from conversation.models.conversation import Conversation
from conversation.models.message import Message
from datetime import datetime
import traceback
from core.middleware.current_user import set_current_chat_user, clear_current_chat_user
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

class ChatView(TemplateView):
    template_name = 'sales/chat_index.html'


# class ChatAPIView(APIView):
#     def post(self, request):
#         ser = ChatRequestSerializer(data=request.data)
#         ser.is_valid(raise_exception=True)
#         prompt = ser.validated_data['prompt']

#         result = sales_metrics_engine(prompt)
#         answer = generate_llm_answer(prompt, result)

#         out = {
#             'answer': answer,
#             'data': result.get('result'),
#             'operation_plan': result.get('operation_plan'),
#         }
#         response_ser = ChatResponseSerializer(out)
#         return Response(response_ser.data, status=status.HTTP_200_OK)



# imports: Conversation, Message, serializers, handle_user_query,
# set_current_chat_user, clear_current_chat_user

def _extract_answer(result):
    """
    Handle different shapes the agent might return on the first turn.
    """
    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        # common keys across first/next turns
        for k in ("answer", "final_answer", "message", "text", "content"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # last-ditch: stringify something meaningful
        if "result" in result and isinstance(result["result"], (str, int, float)):
            return str(result["result"])
        if "operation_plan" in result:
            return "I've prepared an operation plan; please see details."

    # really nothing
    return ""

class ChatAPIView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        token = None
        conversation = None
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data["prompt"]

            # conversation
            conversation_id = request.data.get("conversation_id")
            conversation = (self._get_or_create_conversation(request.user, conversation_id)
                            if conversation_id else self._create_new_conversation(request.user))

            # bind user context ONCE
            token = set_current_chat_user(request.user)

            # run agent
            result = handle_user_query(prompt, conversation_id=str(conversation.uuid), user=request.user)

            # ---- extract / synthesize answer ----
            answer = self._extract_answer(result)
            if not answer.strip():
                # Build a minimal first-turn answer so a bubble is saved
                answer = self._fallback_answer_from_result(result, prompt) or \
                         "I prepared the results for you — see details below."

            # persist messages
            self._create_message(conversation, "user", prompt)
            self._create_message(conversation, "assistant", answer)

            out = {
                "answer": answer,
                "data": result.get("result") if isinstance(result, dict) else None,
                "operation_plan": result.get("operation_plan") if isinstance(result, dict) else None,
                "uuid": str(conversation.uuid),
            }
            return Response(FirstChatResponseSerializer(out).data, status=status.HTTP_200_OK)

        except Exception as e:
            print("[ChatAPIView] ERROR:", repr(e))
            out = {
                "answer": "Sorry, I couldn't process that. Try adding a date/metric/SAP entity.",
                "data": None,
                "operation_plan": None,
                "uuid": str(conversation.uuid) if conversation else None,
            }
            return Response(FirstChatResponseSerializer(out).data, status=status.HTTP_200_OK)
        finally:
            if token:
                clear_current_chat_user(token)

    # ---------- helpers ----------
    def _extract_answer(self, result):
        if isinstance(result, str):
            return (result or "").strip()
        if isinstance(result, dict):
            for k in ("answer", "final_answer", "message", "text", "content"):
                v = result.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    def _fallback_answer_from_result(self, result, prompt):
        """Build a small, readable first-turn bubble from structured data."""
        if not isinstance(result, dict):
            return ""

        data = result.get("result") or {}
        # Try common shapes your agent returns
        title = (data.get("title") or data.get("heading") or "Result").strip() if isinstance(data, dict) else "Result"

        # Try to detect period
        period = ""
        if isinstance(data, dict):
            period_info = data.get("period") or data.get("date_range") or {}
            start = period_info.get("start") or period_info.get("from")
            end = period_info.get("end") or period_info.get("to")
            if start and end:
                period = f" ({start} - {end})"

        # Try totals
        bullets = []
        totals = data.get("totals") if isinstance(data, dict) else None
        if isinstance(totals, dict):
            # Common keys: revenue / total_revenue / value / amount
            rev = totals.get("revenue") or totals.get("total_revenue") or totals.get("value") or totals.get("amount")
            cur = totals.get("currency") or "BDT"
            if rev is not None:
                bullets.append(f"**Total Revenue:** {rev} {cur}")

        # As a last resort, if no totals, but there is any numeric key
        if not bullets and isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)) and k.lower() in ("revenue", "sales", "total", "sum"):
                    bullets.append(f"**{k.capitalize()}:** {v}")
                    break

        # Compose markdown similar to your screenshot
        md = f"### {title}{period}\n"
        if bullets:
            md += "\n" + "\n".join([f"- {b}" for b in bullets])
        else:
            md += "\n_I've computed the results; expand the details in the panel below._"

        return md

    def _create_new_conversation(self, user):
        return Conversation.objects.create(
            user=user, title=f"Chat - {datetime.now():%Y-%m-%d %H:%M:%S}", is_deleted=False
        )

    def _get_or_create_conversation(self, user, conversation_uuid):
        existing = Conversation.objects.filter(user=user, uuid=conversation_uuid, is_deleted=False).first()
        return existing or self._create_new_conversation(user)

    def _create_message(self, conversation, sender_role, text):
        Message.objects.create(conversation=conversation, sender=sender_role, text=text, is_deleted=False)

class ExistingConversationAPIView(APIView):
    token = None
    def post(self, request, conversation_uuid):
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data['prompt']

            conversation = self.get_or_create_conversation(request.user, conversation_uuid)

            token = set_current_chat_user(request.user)  
            # > Run agent 
            # result = handle_user_query(prompt)
            result = handle_user_query(prompt, conversation_id=str(conversation.uuid),user=request.user)

            answer = result if isinstance(result, str) else result.get("answer", "")

            self.create_message(conversation, 'user', prompt)
            self.create_message(conversation, 'assistant', answer)

            out = {
                'answer': answer,
                'data': result.get('result') if isinstance(result, dict) else None,
                'operation_plan': result.get('operation_plan') if isinstance(result, dict) else None,
            }
            response_ser = ChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            print("[ExistingConversationAPIView] ERROR:", repr(e))
            traceback.print_exc()
            out = {
                'answer': (
                    "Sorry, I couldn't process your request. "
                    "Please try a more specific question, such as including a date, metric, or SAP entity."
                ),
                'data': None,
                'operation_plan': None
            }
            response_ser = ChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)
        
        finally:
                clear_current_chat_user(token)

    def get_or_create_conversation(self, user, conversation_uuid):
        existing_conversation = Conversation.objects.filter(
            user=user, uuid=conversation_uuid, is_deleted=False
        ).first()
        return existing_conversation if existing_conversation else self.create_new_conversation(user)

    def create_new_conversation(self, user):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conversation = Conversation.objects.create(
            user=user, title=f"Chat - {current_time}", is_deleted=False
        )
        return conversation

    def create_message(self, conversation, sender_role, message_content):
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )