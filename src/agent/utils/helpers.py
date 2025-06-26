import re
from datetime import datetime, timedelta
import dateparser

import re
from datetime import datetime, timedelta
import dateparser
from conversation.models.message import Message
# Month mapping for extracting months
MONTHS = {m.lower(): i + 1 for i, m in enumerate([
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
])}

def extract_date_range_from_prompt(prompt, today=None):
    today = today or datetime.utcnow().date()

    # Look for 'March 2025', 'Apr 2024', 'Q2 2025', '2024', etc.
    month_pat = re.search(r'\b([JFMASOND][a-z]+)\s+(\d{4})\b', prompt, re.I)
    year_pat = re.search(r'\b(20\d{2})\b', prompt)
    qtr_pat = re.search(r'\bQ([1-4])\s*(20\d{2})\b', prompt, re.I)
    date_range_pat = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|through|-|–)\s*(\d{4}-\d{2}-\d{2})', prompt)

    if date_range_pat:
        # "2024-02-01 to 2024-02-28"
        start, end = date_range_pat.groups()
        return start, end

    if qtr_pat:
        # "Q2 2025"
        q, y = int(qtr_pat.group(1)), int(qtr_pat.group(2))
        start_month = 3 * (q - 1) + 1
        start = datetime(y, start_month, 1).date()
        end_month = start_month + 2
        if end_month == 12:
            end = datetime(y, 12, 31).date()
        else:
            end = (datetime(y, end_month + 1, 1) - timedelta(days=1)).date()
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
                end = (datetime(y, m + 1, 1) - timedelta(days=1)).date()
            return start.isoformat(), end.isoformat()

    if year_pat:
        y = int(year_pat.group(1))
        start = datetime(y, 1, 1).date()
        end = datetime(y, 12, 31).date()
        return start.isoformat(), end.isoformat()

    # Fallback: use dateparser's search_dates to handle any date-based expressions
    date_range = dateparser.search_dates(prompt, settings={'PREFER_DATES_FROM': 'past'})
    if date_range and len(date_range) >= 2:
        # If two dates are found, assume they represent a range
        start = date_range[0][1].date()
        end = date_range[1][1].date()
        return start.isoformat(), end.isoformat()
    elif date_range and len(date_range) == 1:
        # If only one date is found, treat it as a single day range
        start = end = date_range[0][1].date()
        return start.isoformat(), end.isoformat()

    # Default: 2024-01-01 to today if no date is found
    return "2024-01-01", today.isoformat()


def extract_dates_from_past_messages(conversation):
    """
    Extracts date ranges from the last 10 conversation messages.
    The two most recent date ranges will be identified and used as the 'from' and 'to' dates.
    """
    # Get the last 10 messages from the conversation where 'ai' is the sender
    past_msgs = Message.objects.filter(
        conversation=conversation, sender="ai", is_deleted=False
    ).order_by('-created_at')[:10]  # Get the last ten messages

    date_ranges = []

    # Extract dates from the messages
    for msg in past_msgs:
        # Parse the message text for date ranges
        date_range = extract_date_range_from_prompt(msg.text)
        if date_range != ("2024-01-01", str(datetime.utcnow().date())):
            # Avoid default range
            date_ranges.append(date_range)

    # If no valid date range is found in the last messages, return None
    if not date_ranges:
        return None

    # Sort date ranges chronologically to find the latest two periods
    date_ranges = sorted(date_ranges, key=lambda x: (x[0], x[1]))

    # Return the latest two date ranges (from and to)
    if len(date_ranges) >= 2:
        return date_ranges[-2:]  # Return the last two date ranges (most recent ones)
    else:
        # If there's only one date range, return it as both from and to
        return date_ranges * 2  # Duplicate the range to make it from and to


def ensure_azure_datetime(date_str):
    # If already contains T, assume it's a datetime string
    if 'T' in date_str:
        if not date_str.endswith('Z'):
            return date_str + 'Z'
        return date_str
    # Add time and Zulu if missing
    return f"{date_str}T00:00:00Z"





def fill_dates(plan):
    now = datetime.now()
    if not plan.get("start_date"):
        plan["start_date"] = f"{now.year}-01-01"
    if not plan.get("end_date"):
        plan["end_date"] = f"{now.year}-12-31"
    return plan
