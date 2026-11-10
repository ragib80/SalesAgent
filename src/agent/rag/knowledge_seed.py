"""Build seed documents for the SAP sales Azure AI Search knowledge index."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from agent.agent import (
    FIELD_MAPPINGS,
    GSBER_MAPPING,
    KUSTO_SCHEMA,
    TABLE_NAME,
    VTWEG_MAPPING,
    get_schema_types_from_static,
)

SOURCE_VERSION = "1.0"


def build_sales_knowledge_documents(
    *,
    synced_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build repeatable seed documents for the SAP sales RAG index.

    The index is intentionally limited to business/context knowledge. It does
    not include dealer aliases, customer master data, or transaction facts.
    """

    synced_at_value = _format_datetime(synced_at or datetime.now(timezone.utc))
    documents: list[dict[str, Any]] = []
    documents.append(_build_field_mapping_document(synced_at_value))
    documents.append(_build_schema_document(synced_at_value))
    documents.extend(_build_column_documents(synced_at_value))
    documents.extend(_build_gsber_documents(synced_at_value))
    documents.extend(_build_vtweg_documents(synced_at_value))
    documents.extend(_build_curated_prompt_documents(synced_at_value))
    return documents


def _base_document(
    *,
    doc_id: str,
    doc_type: str,
    category: str,
    title: str,
    summary: str,
    content: str,
    synced_at: str,
    aliases: list[str] | None = None,
    keywords: list[str] | None = None,
    sap_columns: list[str] | None = None,
    column_types: list[str] | None = None,
    kpi_names: list[str] | None = None,
    intent_tags: list[str] | None = None,
    gsber_codes: list[str] | None = None,
    vtweg_codes: list[str] | None = None,
    kql_pattern: str = "",
    source_name: str = "agent.py:SYSTEM_PROMPT_KQL",
) -> dict[str, Any]:
    """Return a document matching the sap-sales-knowledge-v1 schema."""

    return {
        "id": doc_id,
        "parent_id": "",
        "chunk_id": doc_id,
        "chunk_ordinal": 0,
        "doc_type": doc_type,
        "category": category,
        "title": title,
        "summary": summary,
        "content": content,
        "aliases": _unique(aliases or []),
        "keywords": _unique(keywords or []),
        "sap_table": TABLE_NAME,
        "sap_columns": _unique(sap_columns or []),
        "column_types": _unique(column_types or []),
        "kpi_names": _unique(kpi_names or []),
        "intent_tags": _unique(intent_tags or []),
        "gsber_codes": _unique(gsber_codes or []),
        "vtweg_codes": _unique(vtweg_codes or []),
        "kql_pattern": kql_pattern,
        "source_name": source_name,
        "source_version": SOURCE_VERSION,
        "last_synced_at": synced_at,
        "is_active": True,
    }


def _build_field_mapping_document(synced_at: str) -> dict[str, Any]:
    schema_types = get_schema_types_from_static()
    mapped_columns = _unique(FIELD_MAPPINGS.values())
    mapping_lines = [
        f"{business_term} maps to ADX column {column_name}."
        for business_term, column_name in sorted(FIELD_MAPPINGS.items(), key=lambda item: item[0].lower())
    ]

    return _base_document(
        doc_id="field-mappings-v1",
        doc_type="field_mapping",
        category="schema_mapping",
        title="SAP sales natural language field mappings",
        summary="Maps user-friendly SAP sales terms to SAPSalesInfos ADX columns.",
        content=" ".join(mapping_lines),
        synced_at=synced_at,
        aliases=list(FIELD_MAPPINGS.keys()) + ["business term translation", "column mapping"],
        keywords=mapped_columns,
        sap_columns=mapped_columns,
        column_types=_column_type_labels(mapped_columns, schema_types),
        intent_tags=["planning", "kql_generation", "schema_grounding"],
        source_name="agent.py:FIELD_MAPPINGS",
    )


def _build_schema_document(synced_at: str) -> dict[str, Any]:
    schema_types = get_schema_types_from_static()
    columns = list(schema_types.keys())

    return _base_document(
        doc_id="adx-schema-sapsalesinfos-v1",
        doc_type="adx_schema",
        category="table_schema",
        title="SAPSalesInfos ADX schema",
        summary="Complete ADX schema for the SAPSalesInfos table used by the sales analysis agent.",
        content=(
            "SAPSalesInfos contains SAP sales transactions with customer, product, "
            "financial, geographic, organization, document, and time dimensions. "
            f"Raw ADX schema: {KUSTO_SCHEMA}"
        ),
        synced_at=synced_at,
        aliases=["SAPSalesInfos", "sales table", "ADX schema", "Kusto schema"],
        keywords=columns,
        sap_columns=columns,
        column_types=_column_type_labels(columns, schema_types),
        intent_tags=["schema_grounding", "kql_generation", "validation"],
        source_name="agent.py:KUSTO_SCHEMA",
    )


def _build_column_documents(synced_at: str) -> list[dict[str, Any]]:
    schema_types = get_schema_types_from_static()
    term_aliases_by_column = _aliases_by_column()

    documents = []
    for column_name, column_type in sorted(schema_types.items(), key=lambda item: item[0].lower()):
        aliases = term_aliases_by_column.get(column_name, [])
        usage = _column_usage_note(column_name, column_type)
        documents.append(
            _base_document(
                doc_id=f"column-{_slug(column_name)}",
                doc_type="column_definition",
                category="table_schema",
                title=f"SAPSalesInfos column {column_name}",
                summary=f"{column_name} is an ADX {column_type} column on SAPSalesInfos.",
                content=(
                    f"Column {column_name} belongs to {TABLE_NAME}. "
                    f"Its ADX type is {column_type}. {usage}"
                ),
                synced_at=synced_at,
                aliases=aliases + [column_name],
                keywords=[column_name, column_type],
                sap_columns=[column_name],
                column_types=[f"{column_name}: {column_type}"],
                intent_tags=["schema_grounding", "kql_generation", "data_type_rule"],
                source_name="agent.py:KUSTO_SCHEMA",
            )
        )
    return documents


def _build_gsber_documents(synced_at: str) -> list[dict[str, Any]]:
    documents = []
    for name, code in sorted(GSBER_MAPPING.items(), key=lambda item: item[1]):
        documents.append(
            _base_document(
                doc_id=f"gsber-{_slug(name)}-{code}",
                doc_type="gsber_mapping",
                category="business_area_mapping",
                title=f"{name} business area mapping",
                summary=f"{name} maps to gsber {code}.",
                content=(
                    f"When the user says {name}, use {TABLE_NAME}.gsber with "
                    f"numeric comparison gsber == {code}. Do not quote gsber values."
                ),
                synced_at=synced_at,
                aliases=[name, f"depot {code}", f"business area {code}", f"gsber {code}"],
                keywords=["gsber", "depot", "business area", name, str(code)],
                sap_columns=["gsber"],
                column_types=["gsber: long"],
                intent_tags=["area_mapping", "kql_generation"],
                gsber_codes=[str(code)],
                source_name="agent.py:GSBER_MAPPING",
            )
        )
    return documents


def _build_vtweg_documents(synced_at: str) -> list[dict[str, Any]]:
    documents = []
    for name, code in sorted(VTWEG_MAPPING.items(), key=lambda item: item[1]):
        documents.append(
            _base_document(
                doc_id=f"vtweg-{_slug(name)}-{code}",
                doc_type="vtweg_mapping",
                category="distribution_channel_mapping",
                title=f"{name} distribution channel mapping",
                summary=f"{name} maps to vtweg {code}.",
                content=(
                    f"When the user asks for {name}, use {TABLE_NAME}.vtweg with "
                    f"numeric comparison vtweg == {code}. Do not quote vtweg values."
                ),
                synced_at=synced_at,
                aliases=[name, f"channel {code}", f"distribution channel {code}", f"vtweg {code}"],
                keywords=["vtweg", "distribution channel", name, str(code)],
                sap_columns=["vtweg"],
                column_types=["vtweg: long"],
                intent_tags=["channel_mapping", "kql_generation"],
                vtweg_codes=[str(code)],
                source_name="agent.py:VTWEG_MAPPING",
            )
        )
    return documents


def _build_curated_prompt_documents(synced_at: str) -> list[dict[str, Any]]:
    return [
        _business_rule_global_bukrs(synced_at),
        _date_rule_fiscal_year(synced_at),
        _date_rule_relative_periods(synced_at),
        _data_type_rule_enforcement(synced_at),
        _business_rule_string_matching(synced_at),
        _business_rule_performance_limits(synced_at),
        _business_rule_time_grouping(synced_at),
        _business_rule_matkl_normalization(synced_at),
        _kpi_rule_sales_quantity_volume(synced_at),
        _kpi_rule_lifting(synced_at),
        _pattern_multi_period_individual(synced_at),
        _pattern_dropoff_leftanti(synced_at),
        _pattern_active_then_inactive_show_zero(synced_at),
        _pattern_dealer_product_brand_inactive_show_zero(synced_at),
        _pattern_declining_negative_growth(synced_at),
        _pattern_positive_growth(synced_at),
        _pattern_mtd_growth(synced_at),
        _pattern_mtd_specific_month(synced_at),
        _pattern_ytd_growth(synced_at),
        _pattern_ytd_specific_year(synced_at),
        _pattern_contribution(synced_at),
        _pattern_average_sales(synced_at),
        _pattern_trend_analysis(synced_at),
    ]


def _business_rule_global_bukrs(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-global-bukrs-1000",
        doc_type="business_rule",
        category="security_scope",
        title="Mandatory company filter for Berger Paints Bangladesh",
        summary="Every SAPSalesInfos table scan must filter bukrs == 1000 first.",
        content=(
            f"For every KQL query against {TABLE_NAME}, add | where bukrs == 1000 "
            "as the first filter after the table name. bukrs is long, so use numeric "
            "comparison without quotes. Company code 1000 means Berger Paints Bangladesh Limited."
        ),
        synced_at=synced_at,
        aliases=["company filter", "bukrs 1000", "Berger company code"],
        keywords=["bukrs", "company", "mandatory filter", "security"],
        sap_columns=["bukrs"],
        column_types=["bukrs: long"],
        intent_tags=["security", "kql_generation", "validation"],
    )


def _date_rule_fiscal_year(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-fiscal-year-april-march",
        doc_type="date_rule",
        category="date_handling",
        title="Fiscal year rule for SAP sales analysis",
        summary="The fiscal year runs from April 1 through March 31.",
        content=(
            "For fiscal-year analysis, use April 1 as the fiscal year start and "
            "March 31 as the fiscal year end. Do not treat calendar year as fiscal "
            "year unless the user explicitly asks for calendar years."
        ),
        synced_at=synced_at,
        aliases=["fiscal year", "FY", "financial year"],
        keywords=["April 1", "March 31", "fkdat", "date range"],
        sap_columns=["fkdat"],
        column_types=["fkdat: datetime"],
        intent_tags=["date_resolution", "planning", "kql_generation"],
    )


def _date_rule_relative_periods(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-relative-date-resolution",
        doc_type="date_rule",
        category="date_handling",
        title="Relative date resolution must use current date context",
        summary="Relative periods must use the injected current date context instead of guessing.",
        content=(
            "Resolve this year, last year, MTD, YTD, this quarter, last quarter, "
            "last month, yesterday, and similar phrases from the CURRENT DATE CONTEXT "
            "block. Do not use ago(365d), getyear(now()), startofyear(now()), or "
            "startofmonth(now()) for fiscal periods."
        ),
        synced_at=synced_at,
        aliases=["this year", "last year", "MTD", "YTD", "last quarter", "last month"],
        keywords=["CURRENT DATE CONTEXT", "fkdat", "relative period"],
        sap_columns=["fkdat"],
        column_types=["fkdat: datetime"],
        intent_tags=["date_resolution", "kql_generation"],
    )


def _data_type_rule_enforcement(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-data-type-enforcement",
        doc_type="data_type_rule",
        category="kql_safety",
        title="Strict ADX data type enforcement",
        summary="KQL filters must match each SAPSalesInfos column type.",
        content=(
            "Before using a column in WHERE, determine its ADX type from the schema. "
            "Use numeric comparisons without quotes for long and real fields. Use "
            "quoted string operations for string fields. Use datetime() values for fkdat. "
            "Never compare numeric columns to quoted strings."
        ),
        synced_at=synced_at,
        aliases=["type enforcement", "numeric columns", "string columns", "datetime columns"],
        keywords=["long", "real", "string", "datetime", "where"],
        sap_columns=["bukrs", "gsber", "vtweg", "kunrg", "Revenue", "fkimg", "volum", "fkdat"],
        column_types=[
            "bukrs: long",
            "gsber: long",
            "vtweg: long",
            "kunrg: long",
            "Revenue: real",
            "fkimg: long",
            "volum: real",
            "fkdat: datetime",
        ],
        intent_tags=["validation", "kql_generation"],
    )


def _business_rule_string_matching(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-smart-string-matching",
        doc_type="business_rule",
        category="filtering",
        title="Smart string matching rules",
        summary="Use contains for partial product/customer matching and correct handling of dealer code suffixes.",
        content=(
            "Use arktx contains for product names, cname contains for dealer/customer names, "
            "wgbez contains for brand names, and spart_text contains for division names. "
            "When a dealer name appears as Name (123), treat 123 as kunrg and match the "
            "name part with cname contains if using text."
        ),
        synced_at=synced_at,
        aliases=["string matching", "dealer name matching", "brand filtering", "product filtering"],
        keywords=["contains", "cname", "arktx", "wgbez", "spart_text", "kunrg"],
        sap_columns=["cname", "kunrg", "arktx", "wgbez", "spart_text", "matnr"],
        column_types=[
            "cname: string",
            "kunrg: long",
            "arktx: string",
            "wgbez: string",
            "spart_text: string",
            "matnr: string",
        ],
        intent_tags=["filtering", "kql_generation"],
    )


def _business_rule_performance_limits(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-performance-limits",
        doc_type="business_rule",
        category="performance",
        title="Mandatory KQL result limits and efficient filters",
        summary="Generated KQL should include limits and selective filters.",
        content=(
            "Always include result limits. Summary and aggregation queries should use take 500. "
            "Detail queries should use take 1000. Ranking and top-N queries should use top N by metric desc. "
            "Place selective date and scope filters early in the pipeline."
        ),
        synced_at=synced_at,
        aliases=["performance", "limits", "take", "top"],
        keywords=["take 500", "take 1000", "top", "where", "performance"],
        sap_columns=["fkdat", "bukrs"],
        intent_tags=["performance", "kql_generation"],
    )


def _business_rule_time_grouping(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-time-grouping",
        doc_type="data_type_rule",
        category="date_handling",
        title="Time grouping rules for fkdat",
        summary="Use ADX startof functions for time grouping and never bin(fkdat, 1mo).",
        content=(
            "For monthly trends use startofmonth(fkdat). For quarterly analysis use "
            "startofquarter(fkdat). For yearly analysis use startofyear(fkdat). For daily "
            "analysis use startofday(fkdat). For weekly analysis use startofweek(fkdat). "
            "Do not use bin(fkdat, 1mo)."
        ),
        synced_at=synced_at,
        aliases=["monthly trend", "time grouping", "period grouping"],
        keywords=["startofmonth", "startofquarter", "startofyear", "startofday", "startofweek"],
        sap_columns=["fkdat"],
        column_types=["fkdat: datetime"],
        intent_tags=["trend", "date_resolution", "kql_generation"],
    )


def _business_rule_matkl_normalization(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="rule-matkl-normalization",
        doc_type="business_rule",
        category="filtering",
        title="Material group normalization for matkl",
        summary="Extract leading F plus digits from material group labels before filtering matkl.",
        content=(
            "When the user provides matkl like f010 (RSE) or F010(ABC), extract only "
            "the leading F plus digits, such as F010, and ignore the remaining label text."
        ),
        synced_at=synced_at,
        aliases=["matkl normalization", "material group code"],
        keywords=["matkl", "material group", "F010"],
        sap_columns=["matkl"],
        column_types=["matkl: string"],
        intent_tags=["filtering", "kql_generation"],
    )


def _kpi_rule_sales_quantity_volume(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="kpi-sales-quantity-volume",
        doc_type="kpi_definition",
        category="metrics",
        title="Core sales metrics",
        summary="Sales maps to Revenue, quantity maps to fkimg, and volume maps to volum.",
        content=(
            "In SAPSalesInfos, sales and revenue mean Revenue. Quantity means fkimg. "
            "Volume means volum. Amounts are BDT and volume is gallons in response formatting."
        ),
        synced_at=synced_at,
        aliases=["sales", "revenue", "quantity", "volume"],
        keywords=["Revenue", "fkimg", "volum", "BDT", "gallons"],
        sap_columns=["Revenue", "fkimg", "volum"],
        column_types=["Revenue: real", "fkimg: long", "volum: real"],
        kpi_names=["sales", "revenue", "quantity", "volume"],
        intent_tags=["metrics", "planning", "kql_generation"],
    )


def _kpi_rule_lifting(synced_at: str) -> dict[str, Any]:
    return _base_document(
        doc_id="kpi-lifting",
        doc_type="kpi_definition",
        category="metrics",
        title="Lifting definition",
        summary="Lifting means total volume and total revenue of product sales.",
        content=(
            "When the user asks about lifting, calculate total volume using sum(volum) "
            "and total revenue using sum(Revenue), usually grouped by the requested product, "
            "brand, dealer, depot, or time dimension."
        ),
        synced_at=synced_at,
        aliases=["lifting", "product lifting"],
        keywords=["Revenue", "volum", "sum"],
        sap_columns=["Revenue", "volum", "arktx", "wgbez", "matnr"],
        column_types=["Revenue: real", "volum: real", "arktx: string", "wgbez: string", "matnr: string"],
        kpi_names=["lifting"],
        intent_tags=["metrics", "kql_generation"],
    )


def _pattern_multi_period_individual(synced_at: str) -> dict[str, Any]:
    pattern = (
        "Create one let subquery per requested period. Filter each subquery by that period "
        "and all requested criteria. Summarize required metrics by requested dimensions. "
        "After summarize, add a constant label such as extend Year = \"2025\". Use union "
        "to combine period subqueries and project the period label with dimensions and metrics."
    )
    return _pattern_doc(
        doc_id="pattern-multi-period-individual",
        title="Multi-period individual comparison pattern",
        summary="Use one let block per period and union constant-labeled results.",
        content=pattern,
        synced_at=synced_at,
        aliases=["individually", "separately", "compare years separately"],
        keywords=["union", "let", "extend Year", "multi-period"],
        sap_columns=["fkdat", "Revenue", "fkimg", "volum"],
        intent_tags=["comparison", "multi_period", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_dropoff_leftanti(synced_at: str) -> dict[str, Any]:
    pattern = (
        "Build Period A buyers and Period B buyers using the same product, brand, division, "
        "and scope filters. Summarize Period A by kunrg, cname, gsber, vtweg with Revenue, "
        "volum, and fkimg. Summarize Period B by kunrg. Use join kind=leftanti on kunrg "
        "to return dealers/customers who bought in Period A but not Period B."
    )
    return _pattern_doc(
        doc_id="pattern-dropoff-leftanti",
        title="Drop-off dealer/customer leftanti pattern",
        summary="Find entities that bought in an earlier period but not a later period.",
        content=pattern,
        synced_at=synced_at,
        aliases=["drop-off dealers", "risk dealers", "bought before not after", "leftanti"],
        keywords=["leftanti", "kunrg", "Period A", "Period B"],
        sap_columns=["kunrg", "cname", "gsber", "vtweg", "Revenue", "volum", "fkimg", "fkdat"],
        kpi_names=["drop-off", "inactive dealer"],
        intent_tags=["dropoff", "risk", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_active_then_inactive_show_zero(synced_at: str) -> dict[str, Any]:
    pattern = (
        "To find dealers/customers who had sales in Period A but no sales in Period B, "
        "and explicitly show Period B values as zero: "
        "Build Period A buyers — summarize PeriodA_Revenue = sum(Revenue), PeriodA_Qty = sum(fkimg) "
        "by kunrg, cname, gsber, vtweg (and any other requested dimensions). "
        "Build Period B buyers — summarize PeriodB_Revenue = sum(Revenue) by kunrg only. "
        "Join using join kind=leftouter Period B on kunrg (start from Period A). "
        "After the join, extend PeriodB_Revenue = iif(isnull(PeriodB_Revenue), 0.0, PeriodB_Revenue). "
        "Filter where PeriodB_Revenue == 0 to keep only dealers with no Period B sales. "
        "Project cname, kunrg, gsber, PeriodA_Revenue, PeriodB_Revenue (showing 0 clearly). "
        "Sort by PeriodA_Revenue desc and take 500. "
        "This pattern differs from leftanti: leftanti hides Period B columns entirely; "
        "leftouter + fill zero shows Period B as 0 so the user clearly sees both periods."
    )
    return _pattern_doc(
        doc_id="pattern-active-then-inactive-show-zero",
        title="Active in Period A, inactive in Period B — show Period B as zero",
        summary=(
            "Find dealers who had sales in one period but no sales in the next, "
            "explicitly displaying the inactive period revenue as zero."
        ),
        content=pattern,
        synced_at=synced_at,
        aliases=[
            "active in April inactive in May",
            "had sales in April but no sales in May",
            "bought in April not in May",
            "dealers list who had sales in April 2026 but no sales in May 2026",
            "active dealers who became inactive",
            "no sales in next month",
            "dropped off showing zero",
            "inactive dealers with zero sales",
            "dealers with zero May sales",
            "churned dealers show zero",
            "sales in month A no sales in month B",
        ],
        keywords=[
            "leftouter", "isnull", "PeriodA_Revenue", "PeriodB_Revenue",
            "show zero", "inactive", "active then inactive", "kunrg", "cname",
        ],
        sap_columns=["kunrg", "cname", "gsber", "vtweg", "Revenue", "fkimg", "fkdat"],
        kpi_names=["drop-off", "inactive dealer", "zero sales"],
        intent_tags=["dropoff", "inactive", "risk", "show_zero", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_dealer_product_brand_inactive_show_zero(synced_at: str) -> dict[str, Any]:
    pattern = (
        "To find which specific product or brand a dealer bought in Period A but did NOT buy "
        "in Period B — even if that dealer bought other products in Period B — use a compound "
        "join key of kunrg + arktx (for product) or kunrg + wgbez (for brand). "
        "Build Period A: summarize PeriodA_Revenue = sum(Revenue), PeriodA_Qty = sum(fkimg) "
        "by kunrg, cname, arktx, wgbez, gsber, vtweg. "
        "Build Period B: summarize PeriodB_Revenue = sum(Revenue) by kunrg, arktx (for product) "
        "or by kunrg, wgbez (for brand) — use only the join key columns in Period B summarize. "
        "Join using join kind=leftouter Period B on kunrg, arktx (or kunrg, wgbez for brand). "
        "After the join, extend PeriodB_Revenue = iif(isnull(PeriodB_Revenue), 0.0, PeriodB_Revenue). "
        "Filter where PeriodB_Revenue == 0 to keep only dealer-product or dealer-brand combinations "
        "with no Period B sales. "
        "Project cname, kunrg, arktx (product name), wgbez (brand), gsber, "
        "PeriodA_Revenue, PeriodB_Revenue (showing 0 explicitly). "
        "Sort by PeriodA_Revenue desc and take 500. "
        "Critical rule: the compound join key must include both kunrg AND the product/brand column — "
        "joining on kunrg alone would exclude dealers who bought any product in Period B, "
        "which is incorrect for this pattern. "
        "If the user asks for both product and brand granularity, include arktx and wgbez "
        "in Period A summarize but use the more specific one (arktx for product, wgbez for brand) "
        "as part of the join key."
    )
    return _pattern_doc(
        doc_id="pattern-dealer-product-brand-inactive-show-zero",
        title="Dealer-product or dealer-brand inactive in Period B — show Period B as zero",
        summary=(
            "Find dealer-product or dealer-brand combinations active in Period A but absent "
            "in Period B, showing Period B revenue as zero. Uses compound join key "
            "(kunrg + arktx or kunrg + wgbez) so dealers who bought other products still appear "
            "for the specific missing product/brand."
        ),
        content=pattern,
        synced_at=synced_at,
        aliases=[
            "dealer product inactive",
            "dealer brand inactive",
            "bought product in April not in May",
            "bought brand in April not in May",
            "dealer stopped buying product",
            "dealer stopped buying brand",
            "product not sold in May",
            "brand not sold in May",
            "specific product drop off",
            "specific brand drop off",
            "dealer product month comparison",
            "dealer brand month comparison",
            "which product dealer did not buy in May",
            "which brand dealer did not buy this month",
            "product level inactive dealer",
            "brand level inactive dealer",
            "dealer inactive for specific product",
            "dealer inactive for specific brand",
        ],
        keywords=[
            "leftouter", "compound join key", "kunrg arktx", "kunrg wgbez",
            "PeriodA_Revenue", "PeriodB_Revenue", "show zero",
            "arktx", "wgbez", "product", "brand", "isnull",
        ],
        sap_columns=["kunrg", "cname", "arktx", "wgbez", "gsber", "vtweg", "Revenue", "fkimg", "fkdat"],
        kpi_names=["product drop-off", "brand drop-off", "inactive product", "inactive brand"],
        intent_tags=["dropoff", "inactive", "product", "brand", "show_zero", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_declining_negative_growth(synced_at: str) -> dict[str, Any]:
    pattern = (
        "For declining or negative growth ranking, compare a newer CY period to an older PY "
        "period. Summarize CY by entity key only. Summarize PY by entity key, entity name, "
        "and extra project columns. Start from PY and use join kind=leftouter CY on entity key. "
        "Set missing CY_Revenue to 0. Filter PY_Revenue > 0, CY_Revenue >= 0, and CY_Revenue < PY_Revenue. "
        "Sort GrowthPct ascending."
    )
    return _pattern_doc(
        doc_id="pattern-declining-negative-growth",
        title="Declining and negative growth ranking pattern",
        summary="Rank entities by genuine negative period-over-period growth.",
        content=pattern,
        synced_at=synced_at,
        aliases=["declining", "negative growth", "decreasing sales", "downtrending"],
        keywords=["leftouter", "GrowthPct", "PY_Revenue", "CY_Revenue"],
        sap_columns=["Revenue", "fkdat", "kunrg", "cname", "wgbez", "matnr", "matkl", "spart", "gsber"],
        kpi_names=["negative growth", "decline"],
        intent_tags=["declining", "growth", "comparison", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_positive_growth(synced_at: str) -> dict[str, Any]:
    pattern = (
        "For top performers and positive growth, compare CY to PY with join kind=innerunique. "
        "Require CY_Revenue >= 100 and PY_Revenue >= 100 to avoid credit memo artifacts and near-zero denominators. "
        "Calculate GrowthPct as (CY_Revenue - PY_Revenue) / PY_Revenue * 100 and sort descending."
    )
    return _pattern_doc(
        doc_id="pattern-positive-growth",
        title="Positive growth ranking pattern",
        summary="Rank entities by meaningful positive period-over-period growth.",
        content=pattern,
        synced_at=synced_at,
        aliases=["top growth", "highest growth", "top performers", "best performing"],
        keywords=["innerunique", "GrowthPct", "top performers"],
        sap_columns=["Revenue", "fkdat"],
        kpi_names=["growth"],
        intent_tags=["growth", "ranking", "comparison", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_mtd_growth(synced_at: str) -> dict[str, Any]:
    pattern = (
        "For MTD growth, use the last complete month as CY unless the user specifies a month. "
        "Compare CY full month to the same month last year using datetime_add('year', -1, CY_Start). "
        "For ranked-by-entity MTD growth, summarize CY by entity key/name and LY by entity key, "
        "join leftouter, and compute MTDGrowthPct."
    )
    return _pattern_doc(
        doc_id="pattern-mtd-growth",
        title="MTD growth calculation pattern",
        summary="Calculate month-to-date growth using complete-month CY versus same month LY.",
        content=pattern,
        synced_at=synced_at,
        aliases=["MTD", "month-to-date", "MTD growth"],
        keywords=["MTDGrowthPct", "endofmonth", "datetime_add"],
        sap_columns=["fkdat", "Revenue", "fkimg"],
        kpi_names=["MTD growth"],
        intent_tags=["mtd", "growth", "date_resolution", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_mtd_specific_month(synced_at: str) -> dict[str, Any]:
    pattern = (
        "When the user names a specific month and year (e.g. 'MTD sales of April 2026' or "
        "'sales in March 2025'), treat the full calendar month as the Analysis Period. "
        "Set CY_Start = datetime(YYYY-MM-01) using the named month and year. "
        "Set CY_End = endofmonth(CY_Start). "
        "Set PY_Start = datetime_add('year', -1, CY_Start) and PY_End = endofmonth(PY_Start). "
        "This gives CY = 1 Apr 2026 to 30 Apr 2026 and PY = 1 Apr 2025 to 30 Apr 2025 for 'April 2026'. "
        "Do NOT use ago() or the CURRENT DATE CONTEXT for the analysis period — use the explicit month/year from the user query. "
        "Build CY and PY subqueries filtered to their respective date windows, then join or compare as requested."
    )
    return _pattern_doc(
        doc_id="pattern-mtd-specific-month",
        title="MTD for a specific named month and year",
        summary="When user names a month+year for MTD, use that full month as CY and same month prior year as PY.",
        content=pattern,
        synced_at=synced_at,
        aliases=[
            "MTD of April 2026", "MTD sales of April", "sales in April 2026",
            "April 2026 sales", "sales of march 2025", "specific month MTD",
            "named month sales", "month year sales comparison",
        ],
        keywords=[
            "specific month", "named month", "datetime_add year -1",
            "endofmonth", "CY_Start", "PY_Start", "MTD",
        ],
        sap_columns=["fkdat", "Revenue", "fkimg", "volum"],
        kpi_names=["MTD", "monthly sales"],
        intent_tags=["mtd", "specific_month", "date_resolution", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_ytd_growth(synced_at: str) -> dict[str, Any]:
    pattern = (
        "For YTD growth, use fiscal year April to March. YTD ends at the last complete month, "
        "not today. Compare current fiscal YTD to the same prior-year YTD window. For ranked "
        "entities, summarize CY and LY by entity, join leftouter, and compute YTDGrowthPct."
    )
    return _pattern_doc(
        doc_id="pattern-ytd-growth",
        title="YTD growth calculation pattern",
        summary="Calculate fiscal YTD growth using April-March fiscal windows.",
        content=pattern,
        synced_at=synced_at,
        aliases=["YTD", "year-to-date", "YTD growth"],
        keywords=["YTDGrowthPct", "fiscal year", "April", "March"],
        sap_columns=["fkdat", "Revenue", "fkimg"],
        kpi_names=["YTD growth"],
        intent_tags=["ytd", "growth", "date_resolution", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_ytd_specific_year(synced_at: str) -> dict[str, Any]:
    pattern = (
        "When the user names a specific fiscal year for YTD (e.g. 'YTD sales of FY2025', "
        "'sales in fiscal year 2025', or 'FY2025 performance'), resolve CY and PY from the named year. "
        "For fiscal year N (April N to March N+1): "
        "CY_Start = datetime(N-04-01), CY_End = datetime(N+1-03-31). "
        "PY_Start = datetime(N-1-04-01), PY_End = datetime(N-04-01) minus 1 day. "
        "Example for FY2025: CY = datetime(2025-04-01) to datetime(2026-03-31), "
        "PY = datetime(2024-04-01) to datetime(2025-03-31). "
        "If the user says a calendar year (e.g. 'YTD of 2026'), use fiscal interpretation "
        "unless they explicitly say calendar year. "
        "Do NOT use ago() or CURRENT DATE CONTEXT for the analysis period — "
        "use the explicit year from the user query."
    )
    return _pattern_doc(
        doc_id="pattern-ytd-specific-year",
        title="YTD for a specific named fiscal year",
        summary="When user names a fiscal year for YTD, use that full fiscal year as CY and prior fiscal year as PY.",
        content=pattern,
        synced_at=synced_at,
        aliases=[
            "YTD of FY2025", "YTD sales of fiscal 2025", "FY2025 YTD", "fiscal year 2025 sales",
            "specific year YTD", "named year sales", "FY sales comparison",
            "year 2025 performance", "full year 2024 sales",
        ],
        keywords=[
            "FY", "fiscal year", "specific year", "CY_Start", "PY_Start",
            "YTD", "April", "March", "named year",
        ],
        sap_columns=["fkdat", "Revenue", "fkimg", "volum"],
        kpi_names=["YTD", "fiscal year sales"],
        intent_tags=["ytd", "specific_year", "date_resolution", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_contribution(synced_at: str) -> dict[str, Any]:
    pattern = (
        "For contribution analysis, compute TotalRevenue over the requested date and scope, "
        "then compute SegmentRevenue with the requested dimension filter. ContributionPct is "
        "SegmentRevenue * 100.0 / TotalRevenue, guarded for zero TotalRevenue."
    )
    return _pattern_doc(
        doc_id="pattern-contribution-analysis",
        title="Sales contribution analysis pattern",
        summary="Calculate contribution percentage for a segment against total revenue.",
        content=pattern,
        synced_at=synced_at,
        aliases=["contribution", "contribution of", "contribution by"],
        keywords=["ContributionPct", "TotalRevenue", "SegmentRevenue"],
        sap_columns=["Revenue", "fkdat", "wgbez", "spart_text", "gsber", "cname"],
        kpi_names=["contribution"],
        intent_tags=["contribution", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_average_sales(synced_at: str) -> dict[str, Any]:
    pattern = (
        "For average sales, first summarize revenue by period and optional entity, then average "
        "the period totals. For monthly average use summarize TotalRevenue = sum(Revenue) by "
        "entity and startofmonth(fkdat), then summarize AvgMonthlySales = avg(TotalRevenue)."
    )
    return _pattern_doc(
        doc_id="pattern-average-sales",
        title="Average sales analysis pattern",
        summary="Average sales should average period totals, not raw transaction rows.",
        content=pattern,
        synced_at=synced_at,
        aliases=["average sales", "avg sales", "average revenue", "mean sales"],
        keywords=["AvgMonthlySales", "avg", "startofmonth"],
        sap_columns=["Revenue", "fkdat"],
        kpi_names=["average sales"],
        intent_tags=["average", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_trend_analysis(synced_at: str) -> dict[str, Any]:
    pattern = (
        "Trend analysis means a time-series view over multiple periods. Use startofmonth(fkdat), "
        "startofweek(fkdat), or another startof function by granularity. For multi-period trends, "
        "group by entity plus the time period and sort by the time period ascending."
    )
    return _pattern_doc(
        doc_id="pattern-trend-analysis",
        title="Trend analysis pattern",
        summary="Use time-series grouping for trend questions, not decline ranking patterns.",
        content=pattern,
        synced_at=synced_at,
        aliases=["trend", "trending", "monthly trend", "trend over time"],
        keywords=["startofmonth", "time series", "trend"],
        sap_columns=["fkdat", "Revenue", "fkimg", "volum"],
        kpi_names=["trend"],
        intent_tags=["trend", "kql_generation"],
        kql_pattern=pattern,
    )


def _pattern_doc(
    *,
    doc_id: str,
    title: str,
    summary: str,
    content: str,
    synced_at: str,
    aliases: list[str],
    keywords: list[str],
    sap_columns: list[str],
    intent_tags: list[str],
    kql_pattern: str,
    kpi_names: list[str] | None = None,
) -> dict[str, Any]:
    return _base_document(
        doc_id=doc_id,
        doc_type="kql_pattern",
        category="kql_generation_pattern",
        title=title,
        summary=summary,
        content=content,
        synced_at=synced_at,
        aliases=aliases,
        keywords=keywords,
        sap_columns=sap_columns,
        column_types=_column_type_labels(sap_columns, get_schema_types_from_static()),
        kpi_names=kpi_names or [],
        intent_tags=intent_tags,
        kql_pattern=kql_pattern,
    )


def _column_usage_note(column_name: str, column_type: str) -> str:
    if column_name == "bukrs":
        return "Always filter this company code with bukrs == 1000 as the first table filter."
    if column_name == "gsber":
        return "This is the numeric business area or depot code. Use numeric comparisons without quotes."
    if column_name == "vtweg":
        return "This is the numeric distribution channel code. Dealer is 10, Customer is 20, Project Customer is 30."
    if column_name == "fkdat":
        return "This is the transaction date. Use datetime() values and fiscal date rules."
    if column_name == "cname":
        return "This stores dealer/customer names. Do not index dealer aliases in Azure Search; query ADX directly."
    if column_name == "kunrg":
        return "This stores dealer/customer codes. It is high-cardinality and must be queried in ADX."
    if column_name in {"Revenue", "fkimg", "volum"}:
        return "This is a core metric column used in aggregation."
    if column_type == "string":
        return "Use quoted string operators such as contains, =~, has_any, or in~."
    if column_type in {"long", "real", "int", "float", "double", "decimal"}:
        return "Use numeric comparisons without quotes."
    if column_type == "datetime":
        return "Use datetime() wrappers and date context."
    return "Use the column according to its ADX type."


def _aliases_by_column() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for business_term, column_name in FIELD_MAPPINGS.items():
        aliases.setdefault(column_name, []).append(business_term)
    return aliases


def _column_type_labels(columns: list[str], schema_types: dict[str, str]) -> list[str]:
    return [f"{column}: {schema_types[column]}" for column in columns if column in schema_types]


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "item"


def _unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

