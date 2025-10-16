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
from agent.utils.conversation_helpers import (
    get_conversation_id_from_uuid,
    get_last_20_messages,
)

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


SAP_FIELD_MAPPINGS = {
    "Dealer": ("cname", "kunrg"),
    "Brand": ("wgbez", None),
    "Product Name": ("arktx", None),
    "Material Group": ("matkl", None),
    "Division": ("spart_text", None),
    "Company Code": ("bukrs", None),
    "Sales Org": ("vkorg", None),
    "Distribution Channel": ("vtweg", None),
    "Business Area": ("gsber", None),
    "Credit Control Area": ("kkber", None),
    "Dealer Group": ("kukla", None),
    "Account Group": ("ktokd", None),
    "Sales Group": ("vkgrp_c", None),
    "Sales Office": ("vkbur_c", None),
    "Payer ID": ("Payer_DL", None),
    "Product Code": ("matnr", None),
    "Volume Unit": ("voleh", None),
    "Business Group": ("GK", None),
    "Territory": ("Territory", None),
    "Sales Zone": ("Szone", None),
    "Date": ("fkdat", None),
    "Dealer Code": ("kunrg", None),
    "Invoice Number": ("vbeln", None),
}

class DynamicFieldAutocompleteAPIView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        field_name = request.GET.get("field")
        query_text = request.GET.get("q", "").strip().lower()
        page = int(request.GET.get("page", 1))
        per_page = 50

        if field_name not in SAP_FIELD_MAPPINGS:
            return Response({"error": f"Invalid field: {field_name}"}, status=400)

        # Handle tuple or string mappings
        mapping = SAP_FIELD_MAPPINGS[field_name]
        if isinstance(mapping, (list, tuple)):
            name_field = mapping[0]
            code_field = mapping[1] if len(mapping) > 1 else None
        else:
            name_field = mapping
            code_field = None

        # Build KQL
        if code_field:
            kql = f"""
            {TABLE_NAME}
            | where isnotempty({name_field}) and isnotempty({code_field})
            | summarize by {name_field}, {code_field}
            | project display = strcat(tostring({name_field}), " (", tostring({code_field}), ")")
            """
        else:
            kql = f"""
            {TABLE_NAME}
            | where isnotempty({name_field})
            | summarize by {name_field}
            | project display = tostring({name_field})
            """

        if query_text:
            kql += f'| where tolower(display) contains "{query_text}"'

        kql += "| order by display asc"

        try:
            client = adx()
            # Always get tuple (cols, rows) from adx().run()
            cols, rows = client.run(kql)
            values = [r[0] for r in rows]

            # Pagination
            total = len(values)
            start = (page - 1) * per_page
            end = start + per_page
            paginated = values[start:end]

            results = [{"id": v, "text": v} for v in paginated]

            return Response({
                "results": results,
                "pagination": {"more": end < total}
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)



class PromptSuggestionAPIView(APIView):
    """Suggest AI prompts based on user's partial input and past chat history."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            input_text = request.data.get("input_text", "").strip()
            conversation_id = request.data.get("conversation_id")

            if not input_text:
                return Response(
                    {"status": "error", "message": "No input text provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # ───────────────────────────────
            #  1️⃣ Build conversation history block
            # ───────────────────────────────
            history_block = ""
            if conversation_id:
                try:
                    conv_id = get_conversation_id_from_uuid(conversation_id)
                    last_msgs = get_last_20_messages(conv_id)
                    if last_msgs:
                        history_block = "Previous Conversation Context:\n"
                        for m in last_msgs[-20:]:
                            role = "USER" if m.sender == "user" else "ASSISTANT"
                            history_block += f"{role}: {m.text or ''}\n"
                except Exception as e:
                    print(" History fetch error:", e)

            # ───────────────────────────────
            #  2️⃣ Construct system and user prompts
            # ───────────────────────────────
            system_prompt = (
                "You are an AI assistant that helps users form natural business or sales analysis prompts. "
                "Use the user's current input and recent chat history to predict what they might want to ask next. "
                "Suggest 3 natural, helpful prompts — either completions of their current text or related questions. "
                "Be concise. Do not add explanations or commentary. Return only a list of short sentences."
            )

            user_prompt = f"""
            Current input: "{input_text}"

            {history_block}

            Suggest 3 next prompts the user may want to type.
            """

            # ───────────────────────────────
            #  3️⃣ Call Azure OpenAI
            # ───────────────────────────────
            llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_KEY,
                api_version="2025-01-01-preview",
                azure_deployment=settings.AZURE_OPENAI_ANALYSIS,
                temperature=0.7,
            )

            resp = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            ).content.strip()

            # Parse output into clean list
            suggestions = [
                s.strip("-• \n\r") for s in resp.split("\n") if s.strip()
            ][:3]

            return Response(
                {"status": "success", "suggestions": suggestions},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            print(" Exception in PromptSuggestionAPIView:", e)
            return Response(
                {"status": "error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )