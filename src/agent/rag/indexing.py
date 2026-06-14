"""Azure AI Search indexing helpers for SAP sales knowledge documents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


def build_embedding_text(document: dict[str, Any]) -> str:
    """Build the text payload embedded into contentVector."""

    parts: list[str] = []
    for field_name in ("title", "summary", "content", "kql_pattern"):
        value = document.get(field_name)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    for field_name in ("aliases", "keywords", "sap_columns", "kpi_names", "intent_tags"):
        value = document.get(field_name)
        if isinstance(value, list):
            joined = " ".join(str(item).strip() for item in value if str(item).strip())
            if joined:
                parts.append(joined)

    return "\n".join(parts)


def add_content_vectors(
    documents: Iterable[dict[str, Any]],
    *,
    openai_client: Any,
    embedding_deployment: str,
) -> list[dict[str, Any]]:
    """Return copies of documents enriched with Azure OpenAI embeddings."""

    if not embedding_deployment:
        raise ValueError("AZURE_OPENAI_EMBED_DEPLOYMENT is required to generate contentVector.")

    enriched_documents: list[dict[str, Any]] = []
    for document in documents:
        document_copy = deepcopy(document)
        response = openai_client.embeddings.create(
            model=embedding_deployment,
            input=build_embedding_text(document_copy),
        )
        document_copy["contentVector"] = response.data[0].embedding
        enriched_documents.append(document_copy)
    return enriched_documents


def merge_or_upload_in_batches(
    search_client: Any,
    documents: list[dict[str, Any]],
    *,
    batch_size: int = 100,
) -> list[Any]:
    """Upload documents with Azure Search merge_or_upload_documents in batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    results: list[Any] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        results.extend(search_client.merge_or_upload_documents(documents=batch))
    return results

