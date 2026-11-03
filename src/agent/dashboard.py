"""
Usage analytics for the Django admin dashboard (templates/admin/index.html).

All numbers come from AgentQueryAudit (one row per chat request) plus
Conversation / SalesAuthUser for context. Rows for the selected period are
bucketed in Python so day/hour boundaries follow Bangladesh time regardless
of the database backend or the project's UTC TIME_ZONE.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.db.models import Avg, Count, Exists, OuterRef, Q, Sum
from django.http import JsonResponse
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET

from agent.models import AgentQueryAudit, ErrorCode, ValidationStatus
from conversation.models import Conversation

DISPLAY_TZ = ZoneInfo("Asia/Dhaka")
VIEW_PERMISSION = "agent.view_agentqueryaudit"
MAX_RANGE_DAYS = 366
TOP_USERS = 10
LATEST_LIMIT = 10
WEEKLY_AFTER_DAYS = 62
# Custom dates outside this window are treated as invalid (avoids date overflow).
CUSTOM_MIN_DATE = dt.date(2000, 1, 1)
CUSTOM_MAX_DATE = dt.date(2100, 12, 31)

RANGE_PRESETS = {"today": 1, "7d": 7, "30d": 30, "90d": 90}

# Outcome buckets shown in the "Outcome breakdown" card, in display order.
OUTCOMES = [
    ("answered", "Answered with data"),
    ("general", "General chat"),
    ("no_data", "No data returned"),
    ("kql_failed", "KQL generation failed"),
    ("adx_failed", "ADX execution failed"),
    ("summary_failed", "Summary failed"),
    ("access_denied", "Access denied"),
    ("error", "Unexpected error"),
]
_ERROR_TO_OUTCOME = {
    ErrorCode.NO_DATA: "no_data",
    ErrorCode.KQL_GENERATION_FAILED: "kql_failed",
    ErrorCode.KQL_VALIDATION_FAILED: "kql_failed",
    ErrorCode.ADX_EXECUTION_FAILED: "adx_failed",
    ErrorCode.LLM_SUMMARY_FAILED: "summary_failed",
    ErrorCode.ACCESS_DENIED: "access_denied",
}


def _outcome(success: bool, error_code: str) -> str:
    if error_code == ErrorCode.GENERAL_QUERY:
        return "general" if success else "error"
    if success:
        return "answered"
    return _ERROR_TO_OUTCOME.get(error_code, "error")


def outcome_q(key: str) -> Q | None:
    """Queryset filter selecting the same rows _outcome() puts in `key` (audit admin drill-down)."""
    general = Q(error_code=ErrorCode.GENERAL_QUERY)
    if key == "answered":
        return Q(success=True) & ~general
    if key == "general":
        return Q(success=True) & general
    if key == "error":
        return Q(success=False) & ~Q(error_code__in=list(_ERROR_TO_OUTCOME))
    codes = [code for code, outcome in _ERROR_TO_OUTCOME.items() if outcome == key]
    return Q(success=False, error_code__in=codes) if codes else None


def _resolve_period(params) -> tuple[dt.datetime, dt.datetime, str]:
    """Return (start, end, key) as aware datetimes; end is exclusive."""
    now_local = timezone.now().astimezone(DISPLAY_TZ)
    today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    key = params.get("range", "30d")

    if key == "custom":
        try:
            start_d = dt.date.fromisoformat(params.get("start", ""))
            end_d = dt.date.fromisoformat(params.get("end", ""))
            if end_d < start_d:
                start_d, end_d = end_d, start_d
            end_d = min(end_d, start_d + dt.timedelta(days=MAX_RANGE_DAYS - 1))
            if start_d < CUSTOM_MIN_DATE or end_d > CUSTOM_MAX_DATE:
                raise ValueError("custom date out of range")
        except (ValueError, OverflowError):
            key = "30d"
        else:
            start =dt.datetime.combine(start_d, dt.time.min, DISPLAY_TZ)
            end = dt.datetime.combine(end_d + dt.timedelta(days=1), dt.time.min, DISPLAY_TZ)
            return start, end, key

    if key == "fytd":
        # Fiscal year runs April 1 → March 31.
        fy_year = today.year if today.month >= 4 else today.year - 1
        start = today.replace(year=fy_year, month=4, day=1)
        return start, today + dt.timedelta(days=1), key

    days = RANGE_PRESETS.get(key)
    if days is None:
        key, days = "30d", 30
    return today - dt.timedelta(days=days - 1), today + dt.timedelta(days=1), key


def _pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or not previous:
        return None
    return round((current - previous) / previous * 100, 1)


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _display_name(user) -> str:
    return user.get_full_name().strip() or user.username


def build_dashboard_stats(params) -> dict:
    start, end, range_key = _resolve_period(params)
    span = end - start
    prev_start = start - span
    # Bucket size keeps the trend charts readable: hours for a single day,
    # days up to ~2 months, weeks beyond that.
    if span <= dt.timedelta(days=1):
        bucket = "hour"
    elif span.days <= WEEKLY_AFTER_DAYS:
        bucket = "day"
    else:
        bucket = "week"

    user_filter = params.get("user", "")
    user_filter = int(user_filter) if user_filter.isdigit() else None
    base = AgentQueryAudit.objects.all()
    if user_filter:
        base = base.filter(user_id=user_filter)

    rows = (
        base.filter(created_at__gte=start, created_at__lt=end)
        .values_list(
            "created_at", "success", "error_code", "user_id", "is_sales_query",
            "kql_validation_status", "total_latency_ms", "kql_generation_latency_ms",
            "adx_execution_latency_ms", "llm_summary_latency_ms", "total_tokens",
            "estimated_cost_usd",
        )
        .order_by()
    )

    # ── Buckets for the activity / latency series ────────────────────────────
    if bucket == "hour":
        step = dt.timedelta(hours=1)
    else:
        step = dt.timedelta(days=7 if bucket == "week" else 1)
    n = -(-span // step)  # ceil
    bucket_starts = [start + step * i for i in range(n)]
    bucket_labels = [b.strftime("%H:00" if bucket == "hour" else "%d %b") for b in bucket_starts]
    s_total, s_success, s_failed = [0] * n, [0] * n, [0] * n
    s_kql, s_adx, s_llm = ([[] for _ in range(n)] for _ in range(3))

    total = success = 0
    sales = general = 0
    tokens = 0
    cost = 0.0
    latencies: list[float] = []
    stage = {"kql": [], "adx": [], "llm": []}
    outcomes: Counter = Counter()
    validation: Counter = Counter()
    heat: Counter = Counter()
    per_user: dict = defaultdict(lambda: {"total": 0, "success": 0, "lat": [], "last": None})

    for (created, ok, code, uid, is_sales, val_status, lat, kql_ms, adx_ms,
         llm_ms, tok, usd) in rows.iterator(chunk_size=2000):
        local = created.astimezone(DISPLAY_TZ)
        total += 1
        success += ok
        tokens += tok or 0
        cost += float(usd or 0)
        if is_sales:
            sales += 1
        elif is_sales is False:
            general += 1
        outcomes[_outcome(ok, code)] += 1
        if val_status:
            validation[val_status] += 1
        heat[(local.weekday(), local.hour)] += 1

        i = (created - start) // step
        if 0 <= i < n:
            s_total[i] += 1
            if ok:
                s_success[i] += 1
            else:
                s_failed[i] += 1
            for series, value in ((s_kql, kql_ms), (s_adx, adx_ms), (s_llm, llm_ms)):
                if value is not None:
                    series[i].append(value)

        if lat is not None:
            latencies.append(lat)
        for name, value in (("kql", kql_ms), ("adx", adx_ms), ("llm", llm_ms)):
            if value is not None:
                stage[name].append(value)

        u = per_user[uid]
        u["total"] += 1
        u["success"] += ok
        if lat is not None:
            u["lat"].append(lat)
        if u["last"] is None or created > u["last"]:
            u["last"] = created

    latencies.sort()
    failed = total - success

    # ── Previous period (DB aggregates are enough for the deltas) ────────────
    prev = base.filter(created_at__gte=prev_start, created_at__lt=start).aggregate(
        total=Count("id"),
        success=Count("id", filter=Q(success=True)),
        users=Count("user", distinct=True),
        latency=Avg("total_latency_ms"),
        cost=Sum("estimated_cost_usd"),
    )
    prev_failed = (prev["total"] or 0) - (prev["success"] or 0)
    prev_rate = (prev["success"] / prev["total"] * 100) if prev["total"] else None

    success_rate = round(success / total * 100, 1) if total else None
    active_users = len([uid for uid in per_user if uid is not None])
    avg_latency = _avg(latencies)

    conv_qs = Conversation.objects.filter(created_at__gte=start, created_at__lt=end)
    if user_filter:
        conv_qs = conv_qs.filter(user_id=user_filter)

    User = get_user_model()
    kpis = {
        "total": {"value": total, "delta": _pct_change(total, prev["total"])},
        "success": {
            "value": success,
            "rate": success_rate,
            "delta_pts": (round(success_rate - prev_rate, 1)
                          if success_rate is not None and prev_rate is not None else None),
        },
        "failed": {
            "value": failed,
            "rate": round(failed / total * 100, 1) if total else None,
            "delta": _pct_change(failed, prev_failed),
        },
        "users": {
            "value": active_users,
            "registered": User.objects.filter(is_active=True).count(),
            "delta": _pct_change(active_users, prev["users"]),
        },
        "latency": {
            "avg_ms": avg_latency,
            "p95_ms": _percentile(latencies, 95),
            "delta": _pct_change(avg_latency, prev["latency"]),
        },
        "cost": {
            "usd": round(cost, 4),
            "tokens": tokens,
            "delta": _pct_change(cost, float(prev["cost"] or 0)),
        },
        "conversations": conv_qs.count(),
        "sales_queries": sales,
        "general_queries": general,
    }

    # ── Users ────────────────────────────────────────────────────────────────
    users_by_id = {u.pk: u for u in User.objects.filter(pk__in=[k for k in per_user if k])}
    user_rows = []
    for uid, u in per_user.items():
        user = users_by_id.get(uid)
        user_rows.append({
            "id": uid,
            "name": _display_name(user) if user else "Unknown / deleted",
            "username": user.username if user else "",
            "total": u["total"],
            "success": u["success"],
            "failed": u["total"] - u["success"],
            "success_rate": round(u["success"] / u["total"] * 100, 1),
            "avg_latency_ms": _avg(u["lat"]),
            "last_active": u["last"].isoformat() if u["last"] else None,
        })
    user_rows.sort(key=lambda r: (-r["total"], r["name"].lower()))

    # ── Latest queries ───────────────────────────────────────────────────────
    def _latest(qs):
        out = []
        for a in qs.select_related("user")[:LATEST_LIMIT]:
            out.append({
                "prompt": a.user_prompt[:220],
                "user": _display_name(a.user) if a.user else "Unknown",
                "success": a.success,
                "outcome": _outcome(a.success, a.error_code),
                "latency_ms": a.total_latency_ms,
                "rows": a.adx_row_count,
                "created_at": a.created_at.isoformat(),
                "url": reverse("admin:agent_agentqueryaudit_change", args=[a.pk]),
            })
        return out

    period_qs = base.filter(created_at__gte=start, created_at__lt=end).order_by("-created_at")

    return {
        "range": {
            "key": range_key,
            "start": start.date().isoformat(),
            "end": (end - dt.timedelta(days=1)).date().isoformat(),
            # Exact Bangladesh-time bounds (end exclusive) for the audit-log drill-down links.
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "bucket": bucket,
            "generated_at": timezone.now().isoformat(),
        },
        "kpis": kpis,
        "trend": {
            "labels": bucket_labels,
            "total": s_total,
            "success": s_success,
            "failed": s_failed,
        },
        "latency": {
            "labels": bucket_labels,
            "kql": [_avg(v) for v in s_kql],
            "adx": [_avg(v) for v in s_adx],
            "llm": [_avg(v) for v in s_llm],
            "avg": {k: _avg(v) for k, v in stage.items()},
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
        },
        "outcomes": [
            {"key": k, "label": label, "count": outcomes.get(k, 0)} for k, label in OUTCOMES
        ],
        "validation": [
            {"key": v.value, "label": v.label, "count": validation.get(v.value, 0)}
            for v in ValidationStatus
        ],
        "heatmap": [[d, h, c] for (d, h), c in heat.items()],
        "users": user_rows,
        "latest": _latest(period_qs),
        "latest_failed": _latest(period_qs.filter(success=False)),
        "filters": {"user": user_filter, "users": user_options()},
    }


@staff_member_required
@require_GET
def dashboard_stats_view(request):
    if not request.user.has_perm(VIEW_PERMISSION):
        return JsonResponse({"detail": "You don't have permission to view query audits."}, status=403)
    return JsonResponse(build_dashboard_stats(request.GET))


def user_options() -> list[dict]:
    """Users that have at least one audited query, for the dashboard's user filter."""
    User = get_user_model()
    users = User.objects.filter(
        Exists(AgentQueryAudit.objects.filter(user=OuterRef("pk")))
    ).only("pk", "username", "first_name", "last_name")
    return sorted(
        ({"id": u.pk, "name": _display_name(u)} for u in users),
        key=lambda u: u["name"].lower(),
    )
