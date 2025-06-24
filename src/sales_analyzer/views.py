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

        # Generate result using the sales_metrics_engine
        result = sales_metrics_engine(prompt)
        answer = generate_llm_answer(
            prompt, result, conversation=self.get_or_create_conversation(request.user))

        out = {
            'answer': answer,
            'data': result.get('result'),
            'operation_plan': result.get('operation_plan'),
        }

        # Create user message and agent response
        conversation = self.get_or_create_conversation(request.user)
        self.create_message(conversation, 'user', prompt)
        self.create_message(conversation, 'assistant', answer)

        # Prepare the response
        response_ser = ChatResponseSerializer(out)
        return Response(response_ser.data, status=status.HTTP_200_OK)

    def get_or_create_conversation(self, user):
        current_time = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')  # Format as 'YYYY-MM-DD HH:MM:SS'
        """
        This method checks if there's an existing conversation.
        If not, it creates a new one.
        """
        existing_conversations = Conversation.active.filter(
            user=user, is_deleted=False)

        if not existing_conversations:
            conversation = Conversation.objects.create(
                user=user,   title=f"Chat - {current_time}",  is_deleted=False)
        else:
            conversation = existing_conversations.order_by(
                '-created_at').first()

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


# class SalesQueryAPIView(APIView):
    """
    POST /api/sales/query/
    Body: { "prompt": "Your natural-language question here" }
    
    The agent will:
      1. Turn the NL prompt into a search over Azure Cognitive Search.
      2. Retrieve the top matching records.
      3. Synthesize a concise answer using Azure OpenAI.
    """
    # def post(self, request):
    #     serializer = QueryRequestSerializer(data=request.data)
    #     serializer.is_valid(raise_exception=True)
    #     answer = run_sales_agent(serializer.validated_data["prompt"])
    #     print("answer, ",answer)
    #     return Response({"answer": answer}, status=status.HTTP_200_OK)

    #     return Response({'answer': answer}, status=status.HTTP_200_OK)

    # # 1) NL → SQL-like query generation
    # completion = openai_client.chat.completions.create(
    #     model=settings.AZURE_OPENAI_DEPLOYMENT,
    #     messages=[
    #         {"role": "system", "content": "Convert natural language into SQL-like filters for your SAP fields."},
    #         {"role": "user",   "content": prompt}
    #     ]
    # )
    # sql_query = completion.choices[0].message.content

    # # 2) Query Azure Cognitive Search
    # results = search_client.search(
    #     search_text="*",   # replace with your filter logic
    #     top=100
    # )
    # records = list(results)  # list() the paged results (dicts)
    # sample_records = records[:10]
    # # 3) Summarize with OpenAI
    # analysis_completion = openai_client.chat.completions.create(
    #     model=settings.AZURE_OPENAI_DEPLOYMENT,
    #     messages=[
    #         {"role": "system",    "content": "Summarize these records into insights: top performers, trends, margins."},
    #         {"role": "assistant", "content": str(sample_records)},
    #         {"role": "user",      "content": "Provide concise insights based on these records."}
    #     ]
    # )
    # analysis = analysis_completion.choices[0].message.content

    # return Response({
    #     "sql_query": sql_query,
    #     "results":   records,
    #     "analysis":  analysis
    # }, status=status.HTTP_200_OK)
