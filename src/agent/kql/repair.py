"""KQL cleanup and repair helpers for Phase 3."""

from __future__ import annotations

import re
from typing import Any

from agent.kql.prompts import REPAIR_SYSTEM_PROMPT, build_repair_prompt
from agent.kql.schemas import KQLValidationResult
from agent.kql.validator import validate_kql
from agent.rag.retriever import format_rag_context_for_prompt


def normalize_kql_for_execution(kql: str, *, table_name: str) -> str:
    """Apply deterministic cleanup before validation or execution."""

    cleaned = _extract_raw_kql(kql)
    cleaned = cleaned.replace("SAPSalesInfos", table_name)
    cleaned = cleaned.replace("bin(fkdat, 1mo)", "startofmonth(fkdat)")
    cleaned = _remove_meta_line(cleaned)
    cleaned = _enforce_bukrs_filter(cleaned, table_name=table_name)
    cleaned = _ensure_trailing_semicolon(cleaned)
    return cleaned.strip()


def repair_kql(
    kql: str,
    *,
    user_prompt: str,
    table_name: str,
    schema_types: dict[str, str],
    validation_result: KQLValidationResult,
    llm_client: Any | None = None,
    query_plan: dict[str, Any] | None = None,
    rag_context: dict[str, Any] | None = None,
    execution_error: str | None = None,
) -> str:
    """Repair generated KQL using deterministic cleanup and optional LLM repair."""

    deterministic = normalize_kql_for_execution(kql, table_name=table_name)
    deterministic_result = validate_kql(
        deterministic,
        table_name=table_name,
        schema_types=schema_types,
    )
    if deterministic_result.is_valid and not execution_error:
        return deterministic

    if llm_client is None:
        return deterministic

    prompt = build_repair_prompt(
        user_prompt=user_prompt,
        kql=deterministic,
        table_name=table_name,
        schema_types=schema_types,
        validation_result=validation_result,
        query_plan=query_plan,
        rag_context_block=format_rag_context_for_prompt(
            rag_context,
            max_documents=4,
            max_chars=4000,
        ),
        execution_error=execution_error,
    )
    response = llm_client.invoke(
        [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    raw_content = getattr(response, "content", response)
    return normalize_kql_for_execution(str(raw_content or ""), table_name=table_name)


def _extract_raw_kql(raw: str) -> str:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:kql|kusto)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    text = text.strip("`")
    return text.replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t").strip()


def _remove_meta_line(kql: str) -> str:
    return re.sub(
        r"^\s*//\s*META\s+\{.*?\}\s*$",
        "",
        kql,
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()


def _ensure_trailing_semicolon(kql: str) -> str:
    if not kql:
        return kql
    return kql if kql.rstrip().endswith(";") else f"{kql.rstrip()};"


def _enforce_bukrs_filter(kql: str, *, table_name: str) -> str:
    pattern = re.compile(
        rf"({re.escape(table_name)}\b)(\s*)(\|)",
        flags=re.IGNORECASE,
    )

    def inject(match: re.Match[str]) -> str:
        after_table = kql[match.end() - 1 :]
        first_pipeline = re.split(r"\|", after_table, maxsplit=2)
        first_clause = first_pipeline[1] if len(first_pipeline) > 1 else ""
        if re.match(r"\s*where\b[\s\S]*?\bbukrs\s*(?:==|=~)\s*['\"]?1000['\"]?", first_clause, flags=re.IGNORECASE):
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}| where bukrs == 1000\n|"

    return pattern.sub(inject, kql)
