from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sales_analyzer.serializers import QueryRequestSerializer, QueryResponseSerializer
from agent.azure_clients import search_client, openai_client
from django.conf import settings
from django.views.generic import TemplateView
from agent.agent import run_sales_agent


class ChatView(TemplateView):
    template_name = 'sales/chat.html'

class SalesQueryAPIView(APIView):
    """
    POST /api/sales/query/
    Body: { "prompt": "Your natural-language question here" }
    
    The agent will:
      1. Turn the NL prompt into a search over Azure Cognitive Search.
      2. Retrieve the top matching records.
      3. Synthesize a concise answer using Azure OpenAI.
    """
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        prompt = serializer.validated_data['prompt']

        # Run the RAG agent end-to-end
        answer = run_sales_agent(prompt)

        return Response({'answer': answer}, status=status.HTTP_200_OK)

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