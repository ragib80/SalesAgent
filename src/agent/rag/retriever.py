"""Fail-open Azure AI Search retriever for SAP sales business context."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from django.conf import settings
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

DEFAULT_SELECT_FIELDS = [
    "id",
    "doc_type",
    "category",
    "title",
    "summary",
    "content",
    "aliases",
    "keywords",
    "sap_table",
    "sap_columns",
    "column_types",
    "kpi_names",
    "intent_tags",
    "gsber_codes",
    "vtweg_codes",
    "kql_pattern",
    "source_name",
    "source_version",
]


@dataclass(slots=True)
class RAGDocument:
    """One retrieved SAP sales business/context document."""

    id: str
    title: str = ""
    doc_type: str = ""
    category: str = ""
    summary: str = ""
    content: str = ""
    aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sap_columns: list[str] = field(default_factory=list)
    column_types: list[str] = field(default_factory=list)
    kpi_names: list[str] = field(default_factory=list)
    intent_tags: list[str] = field(default_factory=list)
    gsber_codes: list[str] = field(default_factory=list)
    vtweg_codes: list[str] = field(default_factory=list)
    kql_pattern: str = ""
    source_name: str = ""
    source_version: str = ""
    score: float | None = None
    reranker_score: float | None = None

    @classmethod
    def from_search_result(cls, result: Any) -> "RAGDocument":
        """Normalize an Azure Search result into a serializable document."""

        def get(name: str, default: Any = "") -> Any:
            try:
                return result.get(name, default)
            except AttributeError:
                return getattr(result, name, default)

        return cls(
            id=str(get("id", "")),
            title=str(get("title", "") or ""),
            doc_type=str(get("doc_type", "") or ""),
            category=str(get("category", "") or ""),
            summary=str(get("summary", "") or ""),
            content=str(get("content", "") or ""),
            aliases=_as_list(get("aliases", [])),
            keywords=_as_list(get("keywords", [])),
            sap_columns=_as_list(get("sap_columns", [])),
            column_types=_as_list(get("column_types", [])),
            kpi_names=_as_list(get("kpi_names", [])),
            intent_tags=_as_list(get("intent_tags", [])),
            gsber_codes=_as_list(get("gsber_codes", [])),
            vtweg_codes=_as_list(get("vtweg_codes", [])),
            kql_pattern=str(get("kql_pattern", "") or ""),
            source_name=str(get("source_name", "") or ""),
            source_version=str(get("source_version", "") or ""),
            score=_as_float(get("@search.score", None)),
            reranker_score=_as_float(get("@search.reranker_score", None)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(slots=True)
class RAGContext:
    """Retrieved context bundle for downstream graph and prompt nodes."""

    query: str
    documents: list[RAGDocument] = field(default_factory=list)
    source: str = "azure_ai_search"
    index_name: str = ""
    error: str = ""

    @property
    def has_documents(self) -> bool:
        return bool(self.documents)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["documents"] = [document.to_dict() for document in self.documents]
        return payload


def retrieve_sales_knowledge_context(
    user_prompt: str,
    *,
    query_plan: dict[str, Any] | None = None,
    top: int = 5,
) -> RAGContext:
    """Retrieve SAP sales business context from Azure AI Search.

    This function is intentionally fail-open. Any missing configuration, Azure
    error, empty index, or schema mismatch returns an empty context so the
    existing hardcoded prompt path remains available.
    """

    query_text = _build_query_text(user_prompt, query_plan)
    index_name = _setting("AZURE_SEARCH_INDEX")
    context = RAGContext(query=query_text, index_name=index_name)

    if not _rag_enabled():
        context.source = "disabled"
        return context

    if not _search_configured():
        context.error = "Azure AI Search is not fully configured."
        return context

    try:
        search_client = _search_client()
        results = _search_with_best_effort(search_client, query_text, top=top)
        context.documents = [
            RAGDocument.from_search_result(result)
            for result in results
            if str(_result_get(result, "id", "") or "").strip()
        ]
    except Exception as exc:
        logger.warning("Azure AI Search RAG retrieval failed; continuing without RAG context.")
        context.error = str(exc)

    return context


def format_rag_context_for_prompt(
    rag_context: RAGContext | dict[str, Any] | None,
    *,
    max_documents: int = 5,
    max_chars: int = 6000,
) -> str:
    """Render retrieved RAG context as an advisory prompt block."""

    context = _coerce_context(rag_context)
    if not context or not context.documents:
        return ""

    lines = [
        "### RAG BUSINESS CONTEXT (Phase 4 advisory context)",
        "Retrieved from Azure AI Search for SAP sales business grounding.",
        "Use this for business definitions, mappings, and approved KQL patterns.",
        "Do not treat it as row-level sales data, dealer master data, or an access-control source.",
        "If this context conflicts with SYSTEM_PROMPT_KQL, table schema, current date context, USER_AREA_SCOPE, or validator rules, those authoritative sources win.",
    ]

    used_chars = 0
    for index, document in enumerate(context.documents[:max_documents], start=1):
        rendered = _render_document(document, index)
        if used_chars + len(rendered) > max_chars:
            break
        lines.append(rendered)
        used_chars += len(rendered)

    return "\n\n" + "\n".join(lines) + "\n"


def _search_with_best_effort(
    search_client: SearchClient,
    query_text: str,
    *,
    top: int,
) -> list[Any]:
    """Try vector+semantic search, then semantic text, then simple text."""

    vector = _generate_embedding(query_text)
    if vector:
        try:
            from azure.search.documents.models import VectorizedQuery

            vector_query = VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=top,
                fields="contentVector",
            )
            return list(
                search_client.search(
                    search_text=query_text,
                    vector_queries=[vector_query],
                    query_type="semantic",
                    semantic_configuration_name="sap-sales-semantic",
                    filter="is_active eq true",
                    select=DEFAULT_SELECT_FIELDS,
                    top=top,
                )
            )
        except Exception:
            logger.debug("Vector/semantic RAG search failed; trying semantic keyword search.", exc_info=True)

    try:
        return list(
            search_client.search(
                search_text=query_text,
                query_type="semantic",
                semantic_configuration_name="sap-sales-semantic",
                filter="is_active eq true",
                select=DEFAULT_SELECT_FIELDS,
                top=top,
            )
        )
    except Exception:
        logger.debug("Semantic RAG search failed; trying simple keyword search.", exc_info=True)

    return list(
        search_client.search(
            search_text=query_text,
            filter="is_active eq true",
            select=DEFAULT_SELECT_FIELDS,
            top=top,
        )
    )


def _generate_embedding(query_text: str) -> list[float] | None:
    deployment = _setting("AZURE_OPENAI_EMBED_DEPLOYMENT")
    if not deployment or not _openai_configured():
        return None

    response = _openai_client().embeddings.create(
        model=deployment,
        input=query_text,
    )
    return list(response.data[0].embedding)


@lru_cache(maxsize=1)
def _search_client() -> SearchClient:
    return SearchClient(
        endpoint=_setting("AZURE_SEARCH_ENDPOINT"),
        index_name=_setting("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(_setting("AZURE_SEARCH_KEY")),
    )


@lru_cache(maxsize=1)
def _openai_client() -> AzureOpenAI:
    api_version = (
        getattr(settings, "AZURE_OPENAI_API_VERSION", None)
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or "2025-01-01-preview"
    )
    return AzureOpenAI(
        azure_endpoint=_setting("AZURE_OPENAI_ENDPOINT"),
        api_key=_setting("AZURE_OPENAI_KEY"),
        api_version=api_version,
    )


def _build_query_text(user_prompt: str, query_plan: dict[str, Any] | None) -> str:
    parts = [user_prompt.strip()]
    if query_plan:
        parts.append(json.dumps(query_plan, ensure_ascii=False, default=str))
    return "\n".join(part for part in parts if part)


def _render_document(document: RAGDocument, index: int) -> str:
    lines = [
        f"[{index}] {document.title or document.id}",
        f"- id: {document.id}",
        f"- type: {document.doc_type}",
    ]
    if document.summary:
        lines.append(f"- summary: {document.summary}")
    if document.content:
        lines.append(f"- content: {document.content}")
    if document.kql_pattern:
        lines.append(f"- kql_pattern: {document.kql_pattern}")
    if document.aliases:
        lines.append(f"- aliases: {', '.join(document.aliases[:12])}")
    if document.sap_columns:
        lines.append(f"- sap_columns: {', '.join(document.sap_columns[:20])}")
    if document.column_types:
        lines.append(f"- column_types: {', '.join(document.column_types[:20])}")
    return "\n".join(lines)


def _coerce_context(rag_context: RAGContext | dict[str, Any] | None) -> RAGContext | None:
    if rag_context is None:
        return None
    if isinstance(rag_context, RAGContext):
        return rag_context
    if not isinstance(rag_context, dict):
        return None

    documents = []
    for item in rag_context.get("documents", []):
        if isinstance(item, RAGDocument):
            documents.append(item)
        elif isinstance(item, dict):
            documents.append(RAGDocument(**_document_kwargs(item)))

    return RAGContext(
        query=str(rag_context.get("query", "") or ""),
        documents=documents,
        source=str(rag_context.get("source", "azure_ai_search") or "azure_ai_search"),
        index_name=str(rag_context.get("index_name", "") or ""),
        error=str(rag_context.get("error", "") or ""),
    )


def _document_kwargs(item: dict[str, Any]) -> dict[str, Any]:
    field_names = set(RAGDocument.__dataclass_fields__.keys())
    kwargs = {name: item.get(name) for name in field_names if name in item}
    kwargs.setdefault("id", "")
    return kwargs


def _search_configured() -> bool:
    return all(
        _setting(name)
        for name in ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX")
    )


def _rag_enabled() -> bool:
    value = (
        getattr(settings, "AZURE_SEARCH_RAG_ENABLED", None)
        or os.getenv("AZURE_SEARCH_RAG_ENABLED")
        or ""
    )
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _openai_configured() -> bool:
    return all(
        _setting(name)
        for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_EMBED_DEPLOYMENT")
    )


def _setting(name: str) -> str:
    return str(getattr(settings, name, None) or os.getenv(name) or "").strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_get(result: Any, name: str, default: Any = None) -> Any:
    try:
        return result.get(name, default)
    except AttributeError:
        return getattr(result, name, default)
