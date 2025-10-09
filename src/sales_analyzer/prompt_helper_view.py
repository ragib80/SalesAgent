# src/sales_analyzer/prompt_helper_view.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
# src/sales_analyzer/prompt_helper_view.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from agent.agent import adx, TABLE_NAME, FIELD_MAPPINGS, KUSTO_SCHEMA
from langchain_openai import AzureChatOpenAI
from django.conf import settings

# Mock data for now (later: replace with ADX)
MOCK_FILTER_VALUES = {
    "matkl": [f"Category {i}" for i in range(1, 501)],     # 500 fake categories
    "spart_text": [f"Division {i}" for i in range(1, 301)], # 300 fake divisions
    "vkorg": [f"SalesOrg {i}" for i in range(1, 101)],      # 100 fake orgs
    "vkbur_c": [f"Office {i}" for i in range(1, 201)],      # 200 fake offices
}


class GetFilterValuesAPIView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get(self, request, field):
        try:
            page = int(request.GET.get("page", 1))
            per_page = 50
            q = request.GET.get("q", "").lower()

            # Validate field exists
            if field not in KUSTO_SCHEMA:
                return Response(
                    {"error": f"Invalid field: {field}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Build KQL (cast to string for uniformity)
            kql = f"""
            {TABLE_NAME}
            | where isnotempty({field})
            | summarize by tostring({field})
            | project value = tostring({field})
            """

            if q:
                kql += f'| where tolower(value) contains "{q}"'

            kql += "| order by value asc"

            cols, rows = adx().run(kql)
            values = [r[0] for r in rows]

            # Pagination
            total = len(values)
            start = (page - 1) * per_page
            end = start + per_page
            paginated = values[start:end]

            return Response({
                "results": [{"id": v, "text": v} for v in paginated],
                "pagination": {"more": end < total}
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
class ApplyFiltersAPIView(APIView):
    """Accepts applied filters + optional metric, and returns LLM-refined prompts."""
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            filters = request.data.get("filters", {})
            metric = request.data.get("metric")
            generatedPrompt = request.data.get("generatedPrompt")

            print("🎯 Received filters:", filters)
            print("📊 Selected metric:", metric)
            print("📊 Selected generatedPrompt:", generatedPrompt)

            if not filters and not metric:
                return Response(
                    {"status": "error", "message": "No filters or metric provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # ───────────────────────────────
            #  1️⃣ Convert filters to human-readable format
            # ───────────────────────────────
            readable_parts = []
            for col, values in filters.items():
                label = next((k for k, v in FIELD_MAPPINGS.items() if v == col), col)
                formatted_values = ", ".join(str(v) for v in values)
                readable_parts.append(f"{label}: {formatted_values}")

            filter_text = ", ".join(readable_parts) if readable_parts else ""

            # ───────────────────────────────
            #  2️⃣ Build structured base prompt
            # ───────────────────────────────
            if metric:
                base_prompt = f"Show me the {metric} where {filter_text}" if filter_text else f"Show me the {metric}"
            else:
                base_prompt = generatedPrompt  # user may edit manually later

            # ───────────────────────────────
            #  3️⃣ Call Azure OpenAI to refine the natural language prompt
            # ───────────────────────────────
            llm = AzureChatOpenAI(
                azure_endpoint   = settings.AZURE_OPENAI_ENDPOINT,
                api_key          = settings.AZURE_OPENAI_KEY,
                api_version      = "2025-01-01-preview",
                azure_deployment = settings.AZURE_OPENAI_ANALYSIS,  # e.g., deployment of gpt-5-mini
                temperature      = 0,  # set 0 if you want fully deterministic phrasing
            )

            system_prompt = (
                "You are an SAP Sales Analysis Assistant. "
                "The user’s filters {filter_text} describe SAP sales data (Dealer, Brand, Product, etc.). "
                "Refine the given prompt into a natural, concise English query. "
                "Ensure it still includes all key filters and the selected metric. "
                "Do not add extra explanations — only return the final query text."
            )

            user_prompt = f"The base query is: '{base_prompt}'"

            print(" Sending to LLM:", user_prompt)

            refined_prompt = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            ).content.strip()

            print("✨ Refined Prompt:", refined_prompt)

            # ───────────────────────────────
            #  4️⃣ Return response
            # ───────────────────────────────
            return Response(
                {
                    "status": "success",
                    "filters": filters,
                    "metric": metric,
                    "refined_prompt": refined_prompt,
                    "prompts": [base_prompt, refined_prompt],
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            print(" Exception in ApplyFiltersAPIView:", e)
            return Response(
                {"status": "error", "message": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        


#working code 
# class ApplyFiltersAPIView(APIView):
#     """Accepts applied filters and returns one or more generated prompts."""
#     authentication_classes = (JWTAuthentication,)
#     permission_classes = (IsAuthenticated,)

#     def post(self, request):
#         try:
#             filters = request.data.get("filters", {})
#             print("🎯 Received filters:", filters)

#             if not filters:
#                 return Response(
#                     {"status": "error", "message": "No filters provided"},
#                     status=status.HTTP_400_BAD_REQUEST,
#                 )

#             # Define metric fields (dynamic mapping)
#             METRIC_FIELDS = {
#                 "Revenue": "Revenue",
#                 "fkimg": "Quantity",
#                 "volum": "Volume"
#             }

#             # Convert to human-readable parts
#             readable_parts = []
#             for col, values in filters.items():
#                 label = next((k for k, v in FIELD_MAPPINGS.items() if v == col), col)
#                 formatted_values = ", ".join(str(v) for v in values)
#                 readable_parts.append(f"{label}: {formatted_values}")

#             filter_text = ", ".join(readable_parts)

#             # Detect selected metrics
#             selected_metrics = [METRIC_FIELDS[f] for f in filters.keys() if f in METRIC_FIELDS]

#             # If no metric → fallback to Sales Data
#             if not selected_metrics:
#                 selected_metrics = ["Sales Data"]

#             # Build prompts
#             prompts = [f"Show me the {metric} where {filter_text}" for metric in selected_metrics]

#             return Response(
#                 {
#                     "status": "success",
#                     "filters": filters,
#                     "prompts": prompts,  # ✅ return array of prompts
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         except Exception as e:
#             return Response(
#                 {"status": "error", "message": str(e)},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )


# class ApplyFiltersAPIView(APIView):
#     """Accepts applied filters and returns them back (later: pass to ADX)."""
#     authentication_classes = (JWTAuthentication,)
#     permission_classes = (IsAuthenticated,)

#     def post(self, request):
#         try:
#             filters = request.data.get("filters", {})
#             print("🎯 Received filters:", filters)

#             # TODO: integrate with ADX query builder
#             return Response({
#                 "status": "success",
#                 "filters": filters
#             }, status=status.HTTP_200_OK)
#         except Exception as e:
#             return Response({
#                 "status": "error",
#                 "message": str(e)
#             }, status=status.HTTP_400_BAD_REQUEST)
