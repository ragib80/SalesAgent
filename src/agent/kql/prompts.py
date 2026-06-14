"""Prompt builders for Phase 3 KQL repair."""

from __future__ import annotations

import json
from typing import Any

from agent.kql.schemas import KQLValidationResult

REPAIR_SYSTEM_PROMPT = (
    "You are a strict Azure Data Explorer KQL repair assistant for SAP sales analytics. "
    "Return only raw executable KQL. Do not include markdown, comments, prose, or explanations."
)


def build_repair_prompt(
    *,
    user_prompt: str,
    kql: str,
    table_name: str,
    schema_types: dict[str, str],
    validation_result: KQLValidationResult,
    query_plan: dict[str, Any] | None = None,
    rag_context_block: str = "",
    execution_error: str | None = None,
) -> str:
    """Build a repair prompt with schema, validation, and optional ADX feedback."""

    schema_lines = "\n".join(f"- {column}: {kind}" for column, kind in schema_types.items())
    plan_block = json.dumps(query_plan or {}, ensure_ascii=False, indent=2, default=str)
    error_block = execution_error or "No ADX execution error was provided."

    return f"""
Repair the KQL below so it safely answers the user's SAP sales analysis request.

Rules:
- Preserve the user's business intent.
- Use only table `{table_name}`.
- Use only the approved columns listed in the schema.
- Every `{table_name}` scan must start with `| where bukrs == 1000`.
- Do not use cross-cluster, cross-database, external table, management, ingestion, mutation, or export commands.
- Keep filters, dimensions, metrics, date ranges, grouping, sorting, and limits consistent with the user request and query plan.
- Return only raw KQL ending with a semicolon.

User request:
{user_prompt}

Structured query plan:
{plan_block}

Retrieved business context:
{rag_context_block or "No RAG business context was provided."}

Approved schema:
{schema_lines}

Validation issues:
{validation_result.as_repair_context()}

ADX execution error:
{error_block}

KQL to repair:
{kql}
""".strip()
