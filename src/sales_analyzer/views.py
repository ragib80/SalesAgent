from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sales_analyzer.serializers import QueryRequestSerializer, QueryResponseSerializer, ChatRequestSerializer, ChatResponseSerializer
from agent.azure_clients import search_client, openai_client
from django.conf import settings
from django.views.generic import TemplateView
from agent.agent import handle_user_query
from conversation.models.conversation import Conversation
from conversation.models.message import Message
from datetime import datetime


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


class ChatAPIView(APIView):
    def post(self, request):
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

            # >>>> Run your intelligent agent <<<<
            result = handle_user_query(prompt)
            # result is already a friendly LLM summary (answer)
            answer = result if isinstance(result, str) else result.get("answer", "")

            # Store user & assistant messages for conversation context
            self.create_message(conversation, 'user', prompt)
            self.create_message(conversation, 'assistant', answer)

            out = {
                'answer': answer,
                # Optionally attach structured results if your handle_user_query returns them
                'data': result.get('result') if isinstance(result, dict) else None,
                'operation_plan': result.get('operation_plan') if isinstance(result, dict) else None,
            }
            response_ser = ChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)

        except Exception as e:
            # Always return a user-friendly error, never a stack trace.
            out = {
                'answer': (
                    "Sorry, I couldn't process your request. "
                    "Please try rephrasing your question, for example by adding a date, sales metric, or SAP entity."
                ),
                'data': None,
                'operation_plan': None
            }
            response_ser = ChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)

    # Conversation/message helpers
    def create_new_conversation(self, user):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conversation = Conversation.objects.create(
            user=user, title=f"Chat - {current_time}", is_deleted=False
        )
        return conversation

    def get_or_create_conversation(self, user, conversation_uuid=None):
        if conversation_uuid:
            existing_conversation = Conversation.objects.filter(
                user=user, uuid=conversation_uuid, is_deleted=False
            ).first()
            return existing_conversation if existing_conversation else self.create_new_conversation(user)
        else:
            existing_conversations = Conversation.objects.filter(
                user=user, is_deleted=False
            )
            if not existing_conversations:
                return self.create_new_conversation(user)
            else:
                return existing_conversations.order_by('-created_at').first()

    def create_message(self, conversation, sender_role, message_content):
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )

class ExistingConversationAPIView(APIView):
    def post(self, request, conversation_uuid):
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data['prompt']

            conversation = self.get_or_create_conversation(request.user, conversation_uuid)

            # >>>> Run your intelligent agent <<<<
            result = handle_user_query(prompt)
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