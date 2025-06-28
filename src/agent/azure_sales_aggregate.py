from datetime import datetime
from collections import defaultdict
import calendar

# You’ll need search_client and COLUMN_MAP imported from your context.


def azure_sales_aggregate(params, search_client, COLUMN_MAP):
    """
    Aggregates SAP sales data by sum/avg/min/max/trend/groupby in-memory,
    streaming from Azure Cognitive Search (Cognitive Search has no server aggregation).
    Handles large data efficiently.
    """
    # 1. Resolve columns
    metric = params.get("metric", "revenue")
    metric_col = COLUMN_MAP.get(metric.lower(), metric)
    group_by = params.get("group_by")
    group_by_2 = params.get("group_by_2")
    agg_type = (params.get("aggregation") or "sum").lower()
    trend = agg_type == "trend"
    # 2. Build filter (no $apply)
    filter_str, _ = extract_filter_and_orderby_from_odata(
        build_odata_query(params))
    page_size = 1000
    count = 0

    # 3. Set up aggregation
    def agg_init(): return {"sum": 0, "count": 0,
                            "min": float("inf"), "max": float("-inf")}
    agg_map = defaultdict(agg_init)
    key_fields = []
    if trend:
        key_fields = [group_by] if group_by else []
        key_fields += ["__trend"]  # we’ll compute this from fkdat
    else:
        if group_by:
            key_fields = [group_by]
            if group_by_2:
                key_fields += [group_by_2]

    # 4. Page through search results
    done = False
    skip = 0
    while not done:
        # Always select metric, group_by fields, and fkdat if trend
        select_fields = [metric_col]
        if group_by:
            select_fields.append(group_by)
        if group_by_2:
            select_fields.append(group_by_2)
        if trend:
            select_fields.append("fkdat")
        resp = search_client.search(
            search_text="*",
            filter=filter_str,
            select=",".join(set(select_fields)),
            top=page_size,
            skip=skip,
        )
        batch_count = 0
        for doc in resp:
            # Compose key
            if trend:
                # Group by month/year or as needed
                fkdat = doc.get("fkdat")
                if not fkdat:
                    continue
                # Parse date string
                try:
                    dt = datetime.fromisoformat(fkdat.replace("Z", "+00:00"))
                except Exception:
                    continue
                period = dt.strftime("%Y-%m")
                key = []
                if group_by:
                    key.append(str(doc.get(group_by, "UNKNOWN")))
                key.append(period)
                key = tuple(key)
            else:
                key = tuple(str(doc.get(f, "UNKNOWN"))
                            for f in key_fields) if key_fields else None

            # Aggregate
            val = doc.get(metric_col)
            if val is None:
                continue
            try:
                val = float(val)
            except Exception:
                continue
            agg = agg_map[key]
            agg["sum"] += val
            agg["count"] += 1
            agg["min"] = min(agg["min"], val)
            agg["max"] = max(agg["max"], val)
            count += 1
            batch_count += 1
        skip += batch_count
        if batch_count < page_size:
            done = True

    # 5. Format output
    summary = {}
    for key, v in agg_map.items():
        base = {
            "sum": v["sum"],
            "avg": (v["sum"] / v["count"]) if v["count"] else 0,
            "min": v["min"] if v["count"] else None,
            "max": v["max"] if v["count"] else None,
            "count": v["count"],
        }
        # Key can be tuple (for groupby), else just metric
        if key_fields or trend:
            summary[key] = base
        else:
            summary = base  # Not grouped, just overall

    return {
        "aggregation": agg_type,
        "group_by": key_fields if key_fields else None,
        "summary": summary,
        "row_count": count,
    }

# Helper: plug your existing methods here, or inline them


def build_odata_query(params):
    # Your current odata builder (just filter, no $apply!)
    # ...
    pass


def extract_filter_and_orderby_from_odata(odata_query):
    filter_str = None
    order_by = None
    for part in odata_query.split('&'):
        if part.startswith("$filter="):
            filter_str = part[len("$filter="):]
        elif part.startswith("$orderby="):
            order_by = part[len("$orderby="):]
    return filter_str, order_by
