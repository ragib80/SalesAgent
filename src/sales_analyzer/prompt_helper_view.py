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
    """Accepts applied filters and returns one or more generated prompts."""
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            filters = request.data.get("filters", {})
            print("🎯 Received filters:", filters)

            if not filters:
                return Response(
                    {"status": "error", "message": "No filters provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Define metric fields (dynamic mapping)
            METRIC_FIELDS = {
                "Revenue": "Revenue",
                "fkimg": "Quantity",
                "volum": "Volume"
            }

            # Convert to human-readable parts
            readable_parts = []
            for col, values in filters.items():
                label = next((k for k, v in FIELD_MAPPINGS.items() if v == col), col)
                formatted_values = ", ".join(str(v) for v in values)
                readable_parts.append(f"{label}: {formatted_values}")

            filter_text = ", ".join(readable_parts)

            # Detect selected metrics
            selected_metrics = [METRIC_FIELDS[f] for f in filters.keys() if f in METRIC_FIELDS]

            # If no metric → fallback to Sales Data
            if not selected_metrics:
                selected_metrics = ["Sales Data"]

            # Build prompts
            prompts = [f"Show me the {metric} where {filter_text}" for metric in selected_metrics]

            return Response(
                {
                    "status": "success",
                    "filters": filters,
                    "prompts": prompts,  # ✅ return array of prompts
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"status": "error", "message": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )


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
