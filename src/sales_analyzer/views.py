from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sales_analyzer.serializers import QueryRequestSerializer, QueryResponseSerializer, ChatRequestSerializer, ChatResponseSerializer
from agent.azure_clients import search_client, openai_client
from django.conf import settings
from django.views.generic import TemplateView
from agent.agent import sales_metrics_engine, generate_llm_answer
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
        print("sad  fdsf", request.user)

        # Deserialize and validate the input data
        ser = ChatRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        prompt = ser.validated_data['prompt']

        # If no existing conversation ID, create a new one
        if not request.data.get("conversation_id"):
            conversation = self.create_new_conversation(request.user)
        else:
            conversation = self.get_or_create_conversation(request.user, conversation_uuid=request.data.get("conversation_id"))

        # Generate result using the sales_metrics_engine
        result = sales_metrics_engine(prompt)
        answer = generate_llm_answer(
            prompt, result, conversation=conversation
        )

        # Save user message and bot's response
        self.create_message(conversation, 'user', prompt)
        self.create_message(conversation, 'assistant', answer)

        out = {
            'answer': answer,
            'data': result.get('result'),
            'operation_plan': result.get('operation_plan'),
        }

        # Prepare the response
        response_ser = ChatResponseSerializer(out)
        return Response(response_ser.data, status=status.HTTP_200_OK)

    def create_new_conversation(self, user):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conversation = Conversation.objects.create(
            user=user, title=f"Chat - {current_time}", is_deleted=False
        )
        return conversation

    def get_or_create_conversation(self, user, conversation_uuid=None):
        """
        This method checks if there's an existing conversation.
        If not, it creates a new one.
        """
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
        """
        This method will create a new message under the conversation.
        """
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )


class ExistingConversationAPIView(APIView):
    def post(self, request, conversation_uuid):
        print(f"Existing Conversation - {conversation_uuid}:", request.user)

        # Deserialize and validate the input data
        ser = ChatRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        prompt = ser.validated_data['prompt']

        # Get the existing conversation
        conversation = self.get_or_create_conversation(request.user, conversation_uuid)

        # Generate result using the sales_metrics_engine
        result = sales_metrics_engine(prompt)
        answer = generate_llm_answer(
            prompt, result, conversation=conversation
        )

        # Save user message and bot's response
        self.create_message(conversation, 'user', prompt)
        self.create_message(conversation, 'assistant', answer)

        out = {
            'answer': answer,
            'data': result.get('result'),
            'operation_plan': result.get('operation_plan'),
        }

        # Prepare the response
        response_ser = ChatResponseSerializer(out)
        return Response(response_ser.data, status=status.HTTP_200_OK)

    def get_or_create_conversation(self, user, conversation_uuid):
        """
        This method checks if there's an existing conversation by UUID.
        """
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
        """
        This method will create a new message under the conversation.
        """
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )
