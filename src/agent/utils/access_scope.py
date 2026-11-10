# salesbot/utils/access_scope.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from django.conf import settings

from user_auth.models import (  # adjust import if your models live elsewhere
    UserDepoMap, UserZoneMap, UserTerritoryMap, UserDivisionMap
)

import re
import json

# ---- Configure your ADX column names here (or move to settings.py) ----
DEFAULT_ADX_AREA_COLUMNS = {
    "table": "SAPSalesInfos",     # first table/pipeline to inject after
    "depo": "DepotCode",          # <-- CHANGE to your ADX column
    "zone": "ZoneCode",           # <-- CHANGE to your ADX column
    "territory": "TerritoryCode", # <-- CHANGE to your ADX column
}

def _colmap() -> Dict[str, str]:
    return getattr(settings, "ADX_AREA_COLUMNS", DEFAULT_ADX_AREA_COLUMNS)

def _is_admin(user) -> bool:
    # Bypass for superuser or in Admin/Super Admin group
    if getattr(user, "is_superuser", False):
        return True
    try:
        return user.groups.filter(name__in=["Admin", "Super Admin"]).exists()
    except Exception:
        return False

@dataclass
class UserAreaScope:
    depots: List[str]
    zones: List[str]
    territories: List[str]
    restricted: bool  # True if NOT admin (i.e., filtering applies)
    divisions: List[str] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)

def get_user_area_scope(user) -> UserAreaScope:
    """
    Returns the logged-in user's area scope as codes (strings).
    If user is admin/superadmin => restricted=False and empty lists.
    """
    if _is_admin(user):
        return UserAreaScope(depots=[], zones=[], territories=[], restricted=False, divisions=[])

    depots = list(
        UserDepoMap.objects.filter(user=user)
        .select_related("depo")
        .values_list("depo__code", flat=True)
    )
    zones = list(
        UserZoneMap.objects.filter(user=user)
        .select_related("zone")
        .values_list("zone__code", flat=True)
    )
    territories = list(
        UserTerritoryMap.objects.filter(user=user)
        .select_related("territory")
        .values_list("territory__code", flat=True)
    )
    divisions = list(
        UserDivisionMap.objects.filter(user=user)
        .select_related("division")
        .values_list("division__code", flat=True)
    )

    return UserAreaScope(depots=depots, zones=zones, territories=territories, restricted=True, divisions=divisions)

def _quote_list(vals: List[str]) -> str:
    # Build a KQL-safe comma list: ("A","B","C")
    esc = [f"\"{str(v).replace('\"', '\\\"')}\"" for v in vals]
    return "(" + ",".join(esc) + ")"

def build_area_where_clauses(scope: UserAreaScope) -> List[str]:
    """
    Build safe KQL where-clauses using tostring(column) in ("D01","D02").
    Empty lists => no clause.
    """
    if not scope.restricted:
        return []

    cm = _colmap()
    clauses = []

    if scope.depots:
        clauses.append(f'| where tostring({cm["depo"]}) in {_quote_list(scope.depots)}')
    if scope.zones:
        clauses.append(f'| where tostring({cm["zone"]}) in {_quote_list(scope.zones)}')
    if scope.territories:
        clauses.append(f'| where tostring({cm["territory"]}) in {_quote_list(scope.territories)}')

    return clauses

def inject_area_filters_into_kql(kql: str, scope: UserAreaScope) -> str:
    """
    Injects area filters right after the first table pipeline line (e.g., "SAPSalesInfos | ...").
    If we can't find it, we append at the end — still safe.
    """
    clauses = build_area_where_clauses(scope)
    if not clauses:
        return kql

    lines = kql.splitlines(True)
    # Find line like: <TableOrView> |
    # Example: "SAPSalesInfos | where fkdat >= StartDate"
    table_line_idx = None
    table_name = _colmap().get("table", "")
    table_regex = re.compile(rf"^\s*{re.escape(table_name)}\s*\|", re.IGNORECASE) if table_name else None

    for i, line in enumerate(lines):
        if table_regex and table_regex.search(line):
            table_line_idx = i
            break
        # fallback: first pipeline line that looks like "<word> |"
        if table_line_idx is None and re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*\|", line):
            table_line_idx = i
            # don't break; prefer exact table if later line matches

    inject_text = "".join(cl + "\n" for cl in clauses)

    if table_line_idx is not None:
        lines.insert(table_line_idx + 1, inject_text)
        return "".join(lines)

    # No obvious place found; append filters at the end
    return kql.rstrip() + "\n" + inject_text
