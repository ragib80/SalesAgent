import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

class VectorSearchTool:
    def __init__(self, openai_client: AzureOpenAI, search_client: SearchClient, embedding_model_name: str):
        self.openai_client = openai_client
        self.search_client = search_client
        self.embedding_model_name = embedding_model_name

    def _generate_embeddings(self, text: str) -> list[float]:
        """Generates embeddings for the given text using Azure OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {e}")

    def perform_vector_search(self, query_text: str, top_n: int = 5) -> list:
        """Performs a vector search on the Azure AI Search index."""
        try:
            # Generate embedding for the query
            query_vector = self._generate_embeddings(query_text)

            # Perform vector search
            results = self.search_client.search(
                search_text=None, # No keyword search
                vector_queries=[{
                    "kind": "vector",
                    "vector": query_vector,
                    "k": top_n,
                    "fields": "contentVector" # Assuming \\\"contentVector\\\" is your vector field name
                }],
                select=["id", "title", "content", "Revenue", "wgbez"] # Select relevant fields
            )

            extracted_data = []
            for result in results:
                extracted_data.append(result)
            return extracted_data

        except Exception as e:
            return f"Error during vector search: {e}"


