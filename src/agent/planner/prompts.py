"""Prompt builders for structured sales query planning."""

from __future__ import annotations


PLANNER_SYSTEM_PROMPT = (
    "You are a strict SAP sales analytics query planner. "
    "Return only one valid JSON object. Do not include markdown, KQL, prose, or comments."
)


def build_planner_prompt(
    user_prompt: str,
    conversation_snapshot: str = "",
    rag_context_block: str = "",
) -> str:
    """Build the LLM prompt used to create a structured analytical plan."""

    snapshot = conversation_snapshot or "### MESSAGES_JSONL (none)"
    rag_block = rag_context_block or "### RAG BUSINESS CONTEXT (none)"
    return f"""
Create a structured analytical plan for the user's SAP sales analysis request.

Use the plan only to describe the intended analysis. Do not generate KQL.
Use known SAPSalesInfos column names when a filter or dimension clearly maps to one:
- Revenue, fkimg, volum
- cname, kunrg, wgbez, arktx, matkl
- gsber, Territory, Szone
- spart_text, spart, vkorg, vtweg, vkgrp_c, vkbur_c
- fkdat

Intent examples:
- summary
- ranking
- trend
- comparison
- detail
- dropoff
- anomaly
- general
- unknown

Return this exact JSON shape:
{{
  "intent": "summary|ranking|trend|comparison|detail|dropoff|anomaly|general|unknown",
  "metrics": ["Revenue"],
  "dimensions": ["cname"],
  "filters": {{"gsber": [4130], "wgbez": ["Brand name"]}},
  "time_range": {{
    "label": "last quarter",
    "start": null,
    "end": null,
    "calendar_type": "fiscal|calendar|rolling|unknown",
    "granularity": "daily|weekly|monthly|quarterly|yearly|none"
  }},
  "aggregation": "sum|avg|count|max|min|none",
  "sorting": {{"field": "Revenue", "direction": "desc"}},
  "limit": 10,
  "comparison": {{}},
  "visualization": {{"type": "table|bar|line|pie|none"}},
  "confidence": 0.0,
  "assumptions": []
}}

Rules:
- Preserve the user's requested business meaning.
- Do not invent filters, dates, brands, dealers, depots, territories, or zones.
- If the request depends on prior context, infer only stable context from the snapshot.
- For relative periods, keep the label and calendar_type; exact dates are resolved later.
- Use null for unknown scalar values, [] for unknown lists, and {{}} for unknown objects.
- If this is not a sales-analysis request, set intent to "general".
- Use RAG business context only for business definitions, mappings, KPI meaning, and approved analysis patterns.
- Do not invent dealer/customer names, row-level sales values, or access-scope filters from RAG context.

Conversation snapshot:
{snapshot}

Retrieved business context:
{rag_block}

Current user request:
{user_prompt}
""".strip()
