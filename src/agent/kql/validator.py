"""Static KQL validator for generated SAP sales analysis queries."""

from __future__ import annotations

import re

from agent.kql.schemas import KQLValidationResult

BLOCKED_COMMAND_PATTERNS = (
    r"^\s*\.(?:alter|append|clear|create|delete|drop|export|ingest|move|rename|set|show)\b",
    r"\b(?:alter|append|clear|create|delete|drop|ingest|truncate|update)\s+(?:table|tables|database)\b",
    r"\.set-or-(?:append|replace)\b",
)

BLOCKED_REMOTE_PATTERNS = (
    r"\bcluster\s*\(",
    r"\bdatabase\s*\(",
    r"\bexternal_table\s*\(",
)

MARKDOWN_OR_PROSE_PATTERNS = (
    r"```",
    r"^\s*(?:here\s+is|sure[,:\s]|the\s+kql|kql\s*:)",
)

KNOWN_FUNCTIONS = {
    "ago",
    "any",
    "anyif",
    "avg",
    "bin",
    "case",
    "coalesce",
    "count",
    "countif",
    "datetime",
    "dcount",
    "endofmonth",
    "endofquarter",
    "endofweek",
    "endofyear",
    "extend",
    "iff",
    "isnotnull",
    "isnull",
    "max",
    "min",
    "now",
    "pack",
    "round",
    "startofday",
    "startofmonth",
    "startofquarter",
    "startofweek",
    "startofyear",
    "strcat",
    "sum",
    "tolower",
    "tolong",
    "toreal",
    "tostring",
    "toupper",
}

QUERY_KEYWORDS = {
    "and",
    "asc",
    "between",
    "by",
    "contains",
    "desc",
    "false",
    "has",
    "has_any",
    "in",
    "in~",
    "join",
    "kind",
    "leftanti",
    "let",
    "not",
    "null",
    "on",
    "or",
    "order",
    "print",
    "project",
    "project-away",
    "summarize",
    "take",
    "top",
    "true",
    "union",
    "where",
}


def validate_kql(
    kql: str,
    *,
    table_name: str,
    schema_types: dict[str, str],
    require_bukrs_filter: bool = True,
) -> KQLValidationResult:
    """Validate KQL before it is sent to ADX.

    The validator intentionally focuses on high-signal static checks. It blocks
    clearly unsafe or hallucinated KQL, while leaving deeper syntax validation
    to ADX and the repair loop.
    """

    result = KQLValidationResult()
    query = (kql or "").strip()

    if not query:
        result.add_error("empty_query", "Generated KQL is empty.")
        return result

    _validate_markdown_or_prose(query, result)
    _validate_blocked_operations(query, result)
    _validate_parentheses(query, result)
    _validate_statement_ending(query, result)
    _validate_remote_references(query, result)
    _validate_table_references(query, table_name, result)
    _validate_schema_references(query, table_name, schema_types, result)

    if require_bukrs_filter:
        _validate_bukrs_filter(query, table_name, result)

    _validate_result_limit(query, table_name, result)
    return result


def _validate_markdown_or_prose(
    query: str,
    result: KQLValidationResult,
) -> None:
    for pattern in MARKDOWN_OR_PROSE_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE | re.MULTILINE):
            result.add_error(
                "non_kql_output",
                "Generated output contains markdown or prose instead of raw KQL.",
            )
            return


def _validate_blocked_operations(
    query: str,
    result: KQLValidationResult,
) -> None:
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE | re.MULTILINE):
            result.add_error(
                "blocked_operation",
                "KQL contains a blocked management or mutation operation.",
            )
            return


def _validate_parentheses(query: str, result: KQLValidationResult) -> None:
    balance = 0
    in_single_quote = False
    in_double_quote = False

    for char in query:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if in_single_quote or in_double_quote:
            continue
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
            if balance < 0:
                result.add_error(
                    "unbalanced_parentheses",
                    "KQL has a closing parenthesis without a matching opening parenthesis.",
                )
                return

    if balance:
        result.add_error(
            "unbalanced_parentheses",
            "KQL has unbalanced parentheses.",
        )


def _validate_statement_ending(query: str, result: KQLValidationResult) -> None:
    if not query.rstrip().endswith(";"):
        result.add_error(
            "missing_semicolon",
            "KQL must end with a semicolon.",
        )


def _validate_remote_references(query: str, result: KQLValidationResult) -> None:
    for pattern in BLOCKED_REMOTE_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            result.add_error(
                "remote_reference",
                "KQL contains a cross-cluster, cross-database, or external table reference.",
            )
            return


def _validate_table_references(
    query: str,
    table_name: str,
    result: KQLValidationResult,
) -> None:
    if _is_print_only_query(query):
        return

    if not re.search(rf"\b{re.escape(table_name)}\b", query):
        result.add_error(
            "missing_table",
            f"KQL must query the approved table {table_name}.",
        )

    let_names = _extract_let_names(query)
    allowed_roots = {table_name, *let_names}
    for root in _extract_pipeline_roots(query):
        if root in allowed_roots or root.lower() in QUERY_KEYWORDS:
            continue
        result.add_error(
            "unknown_table_or_alias",
            f"Unknown table or tabular alias '{root}'.",
            location=root,
        )


def _validate_schema_references(
    query: str,
    table_name: str,
    schema_types: dict[str, str],
    result: KQLValidationResult,
) -> None:
    schema_columns = set(schema_types)
    known_names = schema_columns | _extract_aliases(query) | _extract_let_names(query)
    known_names.add(table_name)

    for column in sorted(_extract_likely_column_references(query)):
        if column in known_names:
            continue
        if column.lower() in KNOWN_FUNCTIONS or column.lower() in QUERY_KEYWORDS:
            continue
        if re.fullmatch(r"\d+", column):
            continue
        result.add_error(
            "unknown_column",
            f"KQL references unknown column '{column}'.",
            location=column,
        )


def _validate_bukrs_filter(
    query: str,
    table_name: str,
    result: KQLValidationResult,
) -> None:
    if _is_print_only_query(query):
        return

    for index in _table_occurrences(query, table_name):
        if _table_scan_has_bukrs_filter(query[index + len(table_name) :]):
            continue
        result.add_error(
            "missing_bukrs_filter",
            f"Every {table_name} scan must begin filtering with bukrs == 1000.",
            location=table_name,
        )


def _validate_result_limit(
    query: str,
    table_name: str,
    result: KQLValidationResult,
) -> None:
    if _is_print_only_query(query):
        return
    if not re.search(rf"\b{re.escape(table_name)}\b", query):
        return
    if re.search(r"\b(?:take|top|limit|count|summarize)\b", query, flags=re.IGNORECASE):
        return
    result.add_warning(
        "missing_result_limit",
        "KQL does not include take, top, limit, count, or summarize.",
    )


def _is_print_only_query(query: str) -> bool:
    return bool(re.fullmatch(r"\s*print\b[\s\S]*;\s*", query, flags=re.IGNORECASE))


def _table_occurrences(query: str, table_name: str) -> list[int]:
    return [match.start() for match in re.finditer(rf"\b{re.escape(table_name)}\b", query)]


def _table_scan_has_bukrs_filter(after_table: str) -> bool:
    first_pipe = re.search(r"\|", after_table)
    if not first_pipe:
        return False

    after_pipe = after_table[first_pipe.end() :]
    if not re.match(r"\s*where\b", after_pipe, flags=re.IGNORECASE):
        return False

    first_filter = re.split(r"\|", after_pipe, maxsplit=1)[0]
    return bool(
        re.search(
            r"\bbukrs\s*(?:==|=~)\s*['\"]?1000['\"]?",
            first_filter,
            flags=re.IGNORECASE,
        )
    )


def _extract_let_names(query: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(
            r"\blet\s+([A-Za-z_]\w*)\s*=",
            query,
            flags=re.IGNORECASE,
        )
    }


def _extract_pipeline_roots(query: str) -> set[str]:
    roots: set[str] = set()
    patterns = (
        r"(?:^|;)\s*([A-Za-z_]\w*)\s*(?=\||;|$)",
        r"\blet\s+[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\s*(?=\||\n|;)",
    )
    for pattern in patterns:
        roots.update(
            match.group(1)
            for match in re.finditer(
                pattern,
                query,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
    return roots


def _extract_aliases(query: str) -> set[str]:
    aliases = set()
    assignment_patterns = (
        r"\|\s*extend\s+([^|;]+)",
        r"\|\s*project\s+([^|;]+)",
        r"\|\s*summarize\s+([^|;]+)",
        r"\bby\s+([^|;]+)",
    )
    for pattern in assignment_patterns:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            segment = match.group(1)
            for alias in re.findall(r"\b([A-Za-z_]\w*)\s*=", segment):
                aliases.add(alias)

    for match in re.finditer(r"\blet\s+([A-Za-z_]\w*)\s*=", query, flags=re.IGNORECASE):
        aliases.add(match.group(1))
    return aliases


def _extract_likely_column_references(query: str) -> set[str]:
    references: set[str] = set()

    # where column == value, column contains value, column between (...)
    for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s*(?:==|=~|!=|!~|>=|<=|>|<|\bin~?\b|\bcontains\b|\bhas(?:_any)?\b|\bbetween\b)",
        query,
        flags=re.IGNORECASE,
    ):
        references.add(match.group(1))

    # sum(Revenue), startofmonth(fkdat), tostring(column)
    for match in re.finditer(
        r"\b(?:sum|avg|min|max|countif|dcount|any|anyif|startofday|startofweek|startofmonth|"
        r"startofquarter|startofyear|endofmonth|tolower|toupper|tostring|tolong|toreal)\s*"
        r"\(\s*([A-Za-z_]\w*)",
        query,
        flags=re.IGNORECASE,
    ):
        references.add(match.group(1))

    references.update(_extract_group_by_references(query))
    references.update(_extract_order_references(query))
    references.update(_extract_join_references(query))
    return references


def _extract_group_by_references(query: str) -> set[str]:
    references: set[str] = set()
    for match in re.finditer(r"\bby\s+([^|;]+)", query, flags=re.IGNORECASE):
        for item in match.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                continue
            name_match = re.match(r"([A-Za-z_]\w*)\b", item)
            if name_match:
                references.add(name_match.group(1))
    return references


def _extract_order_references(query: str) -> set[str]:
    references: set[str] = set()
    patterns = (
        r"\|\s*order\s+by\s+([A-Za-z_]\w*)",
        r"\|\s*top\s+\d+\s+by\s+([A-Za-z_]\w*)",
    )
    for pattern in patterns:
        references.update(
            match.group(1)
            for match in re.finditer(pattern, query, flags=re.IGNORECASE)
        )
    return references


def _extract_join_references(query: str) -> set[str]:
    references: set[str] = set()
    for match in re.finditer(r"\bjoin\b[\s\S]*?\bon\s+([A-Za-z_]\w*)", query, flags=re.IGNORECASE):
        references.add(match.group(1))
    return references
