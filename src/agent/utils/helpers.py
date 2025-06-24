import re
from datetime import datetime, timedelta
import dateparser

MONTHS = {m.lower(): i+1 for i, m in enumerate([
    'January','February','March','April','May','June','July','August','September','October','November','December'])}

def extract_date_range_from_prompt(prompt, today=None):
    today = today or datetime.utcnow().date()

    # Look for 'March 2025', 'Apr 2024', 'Q2 2025', '2024', etc.
    month_pat = re.search(r'\b([JFMASOND][a-z]+)\s+(\d{4})\b', prompt, re.I)
    year_pat = re.search(r'\b(20\d{2})\b', prompt)
    qtr_pat  = re.search(r'\bQ([1-4])\s*(20\d{2})\b', prompt, re.I)
    date_range_pat = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|through|-|–)\s*(\d{4}-\d{2}-\d{2})', prompt)

    if date_range_pat:
        # "2024-02-01 to 2024-02-28"
        start, end = date_range_pat.groups()
        return start, end

    if qtr_pat:
        # "Q2 2025"
        q, y = int(qtr_pat.group(1)), int(qtr_pat.group(2))
        start_month = 3*(q-1) + 1
        start = datetime(y, start_month, 1).date()
        end_month = start_month + 2
        # next month's first day, then subtract one
        if end_month == 12:
            end = datetime(y, 12, 31).date()
        else:
            end = (datetime(y, end_month+1, 1) - timedelta(days=1)).date()
        return start.isoformat(), end.isoformat()

    if month_pat:
        # "March 2025"
        month, y = month_pat.group(1).lower(), int(month_pat.group(2))
        m = MONTHS.get(month[:3].lower(), None)
        if m:
            start = datetime(y, m, 1).date()
            if m == 12:
                end = datetime(y, 12, 31).date()
            else:
                end = (datetime(y, m+1, 1) - timedelta(days=1)).date()
            return start.isoformat(), end.isoformat()

    if year_pat:
        y = int(year_pat.group(1))
        start = datetime(y, 1, 1).date()
        end = datetime(y, 12, 31).date()
        return start.isoformat(), end.isoformat()

    # fallback: use LLM or dateparser
    date_range = dateparser.search.search_dates(prompt, settings={'PREFER_DATES_FROM': 'past'})
    if date_range and len(date_range) >= 2:
        start = date_range[0][1].date()
        end = date_range[1][1].date()
        return start.isoformat(), end.isoformat()
    elif date_range and len(date_range) == 1:
        start = end = date_range[0][1].date()
        return start.isoformat(), end.isoformat()

    # Default: 2024-01-01 to today
    return "2024-01-01", today.isoformat()



def ensure_azure_datetime(date_str):
    # If already contains T, assume it's a datetime string
    if 'T' in date_str:
        if not date_str.endswith('Z'):
            return date_str + 'Z'
        return date_str
    # Add time and Zulu if missing
    return f"{date_str}T00:00:00Z"
