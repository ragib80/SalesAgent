def build_trend_kql(start: str, end: str, dim_col: str, top_n: int = 5) -> str:
    return f"""
// 1) input dates
let StartDate         = datetime({start});
let EndDate           = datetime({end});
// if only one month… previous month window
let PreviousStartDate = startofmonth(StartDate - 1d);
let PreviousEndDate   = endofmonth(PreviousStartDate);

// 2) roll up by month & dimension
let Monthly = {TABLE_NAME}
| where fkdat between (PreviousStartDate .. EndDate)
| summarize Revenue = sum(Revenue)
    by Period = startofmonth(fkdat), {dim_col};

// 3) compute growth
let Growth = Monthly
| summarize
    PrevRev = anyif(Revenue, Period == PreviousStartDate),
    CurrRev = anyif(Revenue, Period == StartDate)
  by {dim_col}
| extend GrowthPct = iff(PrevRev == 0, real(null), (CurrRev - PrevRev)*100.0/PrevRev)
| order by GrowthPct desc
| take {top_n};

// 4) output
Growth
""".strip()
