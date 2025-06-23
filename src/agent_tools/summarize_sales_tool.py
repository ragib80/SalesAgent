import os
import math
from collections import defaultdict
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

class SalesSummarizerTool:
    def __init__(self, openai_client: AzureOpenAI, search_client: SearchClient):
        self.openai_client = openai_client
        self.search_client = search_client

    def summarize_sales_data(self, docs: list):
        """Summarizes sales data from a list of document dictionaries."""
        if not docs:
            return "No sales data provided for summarization."

        total_revenue = 0.0
        brand_revenues = defaultdict(float)

        for doc in docs:
            revenue = float(doc.get("Revenue", 0.0))
            total_revenue += revenue
            brand = doc.get("wgbez", "Unknown Brand") # Assuming \\\"wgbez\\\" is the brand field
            brand_revenues[brand] += revenue

        # Sort brands by revenue in descending order and get top 5
        top_brands = sorted(brand_revenues.items(), key=lambda item: item[1], reverse=True)[:5]

        summary_parts = [
            f"Total Revenue: ৳{total_revenue:,.2f}"
        ]

        if top_brands:
            summary_parts.append("Top Brands:")
            for brand, revenue in top_brands:
                summary_parts.append(f"- {brand}: ৳{revenue:,.2f}")
        else:
            summary_parts.append("No brand data available for summarization.")

        return "\n".join(summary_parts)

    # If you later want to add AI-powered summarization, you can add a method here
    # def ai_summarize(self, text: str):
    #     try:
    #         response = self.openai_client.chat.completions.create(
    #             model="your-deployment-name", # Use your actual deployment name
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant that summarizes sales data."},
    #                 {"role": "user", "content": f"Summarize the following sales data: {text}"}
    #             ]
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         return f"Error during AI summarization: {e}"


