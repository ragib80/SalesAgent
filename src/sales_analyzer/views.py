from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.utils import timezone
from django.views.generic import TemplateView
from conversation.models.conversation import Conversation
from conversation.models.message import Message

# keep your serializers/utilities
from .serializers import ChatRequestSerializer, FirstChatResponseSerializer
from core.middleware.current_user import set_current_chat_user, clear_current_chat_user

from agent.ai.runtime import chat_turn  # uses MS SQL to persist frame

from datetime import datetime
import traceback


class ChatAPIView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        token = None
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data['prompt']

            # Create or get conversation
            conversation_id = request.data.get("conversation_id")
            if conversation_id:
                conversation = self.get_or_create_conversation(request.user, conversation_uuid=conversation_id)
            else:
                conversation = self.create_new_conversation(request.user)

            token = set_current_chat_user(request.user)

            # Multi-turn via LangGraph; frame persisted in Conversation.frame_json
            result = chat_turn(conversation_id=str(conversation.uuid), user_text=prompt)
            answer = result.get("answer", "")

            # Persist messages
            self.create_message(conversation, 'user', prompt)
            self.create_message(conversation, 'assistant', answer)

            Conversation.objects.filter(pk=conversation.pk).update(last_message_at=timezone.now())

            out = {
                'answer': answer,
                'data': result.get('result'),
                'operation_plan': None,
                'uuid': str(conversation.uuid),
            }
            response_ser = FirstChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)

        except Exception:
            out = {
                'answer': ("Sorry, I couldn't process your request. "
                           "Please try rephrasing your question, e.g., add a date, sales metric, or SAP entity."),
                'data': None,
                'operation_plan': None,
                'uuid': None,
            }
            traceback.print_exc()
            response_ser = FirstChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)
        finally:
            clear_current_chat_user(token)

    # Conversation/message helpers
    def create_new_conversation(self, user):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conversation = Conversation.objects.create(
            user=user, title=f"Chat - {current_time}", is_deleted=False
        )
        return conversation

    def get_or_create_conversation(self, user, conversation_uuid=None):
        if conversation_uuid:
            existing = Conversation.objects.filter(user=user, uuid=conversation_uuid, is_deleted=False).first()
            return existing if existing else self.create_new_conversation(user)
        else:
            existing = Conversation.objects.filter(user=user, is_deleted=False).order_by('-created_at').first()
            return existing if existing else self.create_new_conversation(user)

    def create_message(self, conversation, sender_role, message_content):
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )


class ExistingConversationAPIView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request, conversation_uuid):
        token = None
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data['prompt']

            conversation = Conversation.objects.filter(user=request.user, uuid=conversation_uuid, is_deleted=False).first()
            if not conversation:
                conversation = Conversation.objects.create(
                    user=request.user, title=f"Chat - {datetime.now():%Y-%m-%d %H:%M:%S}", is_deleted=False
                )

            token = set_current_chat_user(request.user)

            result = chat_turn(conversation_id=str(conversation.uuid), user_text=prompt)
            answer = result.get("answer", "")

            self.create_message(conversation, 'user', prompt)
            self.create_message(conversation, 'assistant', answer)
            Conversation.objects.filter(pk=conversation.pk).update(last_message_at=timezone.now())

            out = {
                'answer': answer,
                'data': result.get('result'),
                'operation_plan': None,
                'uuid': str(conversation.uuid),
            }
            response_ser = FirstChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)

        except Exception:
            out = {
                'answer': ("Sorry, I couldn't process your request. "
                           "Please rephrase with a date, sales metric, or SAP entity."),
                'data': None,
                'operation_plan': None,
                'uuid': str(conversation_uuid),
            }
            traceback.print_exc()
            response_ser = FirstChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)
        finally:
            clear_current_chat_user(token)

    def create_message(self, conversation, sender_role, message_content):
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )



class ChatView(TemplateView):
    template_name = 'sales/chat_index.html'