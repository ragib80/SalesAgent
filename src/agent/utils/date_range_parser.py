# date_range_parser.py - Enhanced Date Range Parser for SAP Sales Analyzer Bot
import json
import re
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import calendar

from langchain_openai import AzureChatOpenAI


class TimeGranularity(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class CalculationType(Enum):
    TOTAL = "total"
    AVERAGE = "average"
    COUNT = "count"
    MAX = "max"
    MIN = "min"


@dataclass
class DateRange:
    start_date: datetime
    end_date: datetime
    granularity: TimeGranularity
    calculation_type: CalculationType
    relative_period: Optional[str] = None
    original_query: Optional[str] = None


class DateRangeParser:
    """Enhanced date range parser using LLM for natural language processing"""
    
    def __init__(self, llm_client: AzureChatOpenAI):
        self.llm = llm_client
        self.today = datetime.now().date()
        
    def parse_date_range(self, user_query: str) -> DateRange:
        """
        Parse user query to extract date range information using LLM
        """
        try:
            # Use LLM to extract structured date information
            date_info = self._extract_date_info_with_llm(user_query)
            
            # Convert to DateRange object
            date_range = self._convert_to_date_range(date_info, user_query)
            
            return date_range
            
        except Exception as e:
            print(f"Error parsing date range: {e}")
            # Fallback to default range (last 30 days)
            return self._get_default_date_range(user_query)
    
    def _extract_date_info_with_llm(self, user_query: str) -> Dict[str, Any]:
        """Use LLM to extract date information from user query"""
        
        prompt = f"""
You are a date range extraction expert for sales data queries. Extract date information from the user query and return ONLY a valid JSON object.

Current date: {self.today.strftime('%Y-%m-%d')}

Rules:
1. Return ONLY valid JSON, no additional text or formatting
2. Use ISO format (YYYY-MM-DD) for specific dates
3. For relative periods, use descriptive strings like "last_month", "this_week", "yesterday"
4. Granularity options: "daily", "weekly", "monthly", "quarterly", "yearly"
5. Calculation type options: "total", "average", "count", "max", "min"

JSON Schema:
{{
    "start_date": "YYYY-MM-DD or relative indicator",
    "end_date": "YYYY-MM-DD or relative indicator", 
    "granularity": "daily|weekly|monthly|quarterly|yearly",
    "calculation_type": "total|average|count|max|min",
    "relative_period": "last_month|this_week|yesterday|etc or null",
    "confidence": 0.0-1.0
}}

Examples:
"Show me sales for last month" -> {{"start_date": "last_month_start", "end_date": "last_month_end", "granularity": "monthly", "calculation_type": "total", "relative_period": "last_month", "confidence": 0.9}}

"Average daily sales in January 2024" -> {{"start_date": "2024-01-01", "end_date": "2024-01-31", "granularity": "daily", "calculation_type": "average", "relative_period": null, "confidence": 0.95}}

"Weekly sales this quarter" -> {{"start_date": "this_quarter_start", "end_date": "this_quarter_end", "granularity": "weekly", "calculation_type": "total", "relative_period": "this_quarter", "confidence": 0.85}}

User query: "{user_query}"
"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}]).content
            
            # Clean response and extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                return json.loads(response)
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM date extraction failed: {e}")
            # Fallback to pattern-based extraction
            return self._fallback_date_extraction(user_query)
    
    def _fallback_date_extraction(self, user_query: str) -> Dict[str, Any]:
        """Fallback pattern-based date extraction when LLM fails"""
        
        query_lower = user_query.lower()
        
        # Default values
        result = {
            "start_date": "last_month_start",
            "end_date": "last_month_end", 
            "granularity": "monthly",
            "calculation_type": "total",
            "relative_period": "last_month",
            "confidence": 0.5
        }
        
        # Detect calculation type
        if any(word in query_lower for word in ["average", "avg", "mean"]):
            result["calculation_type"] = "average"
        elif any(word in query_lower for word in ["count", "number of"]):
            result["calculation_type"] = "count"
        elif any(word in query_lower for word in ["maximum", "max", "highest"]):
            result["calculation_type"] = "max"
        elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
            result["calculation_type"] = "min"
        
        # Detect granularity
        if any(word in query_lower for word in ["daily", "day", "per day"]):
            result["granularity"] = "daily"
        elif any(word in query_lower for word in ["weekly", "week", "per week"]):
            result["granularity"] = "weekly"
        elif any(word in query_lower for word in ["monthly", "month", "per month"]):
            result["granularity"] = "monthly"
        elif any(word in query_lower for word in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]):
            result["granularity"] = "quarterly"
        elif any(word in query_lower for word in ["yearly", "year", "annual"]):
            result["granularity"] = "yearly"
        
        # Detect relative periods
        if "last month" in query_lower:
            result.update({
                "start_date": "last_month_start",
                "end_date": "last_month_end",
                "relative_period": "last_month"
            })
        elif "this month" in query_lower:
            result.update({
                "start_date": "this_month_start", 
                "end_date": "this_month_end",
                "relative_period": "this_month"
            })
        elif "last week" in query_lower:
            result.update({
                "start_date": "last_week_start",
                "end_date": "last_week_end", 
                "relative_period": "last_week",
                "granularity": "weekly"
            })
        elif "this week" in query_lower:
            result.update({
                "start_date": "this_week_start",
                "end_date": "this_week_end",
                "relative_period": "this_week",
                "granularity": "weekly"
            })
        elif "yesterday" in query_lower:
            result.update({
                "start_date": "yesterday",
                "end_date": "yesterday",
                "relative_period": "yesterday",
                "granularity": "daily"
            })
        elif "today" in query_lower:
            result.update({
                "start_date": "today",
                "end_date": "today", 
                "relative_period": "today",
                "granularity": "daily"
            })
        
        # Check for specific date patterns
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, user_query)
        if len(dates) >= 2:
            result.update({
                "start_date": dates[0],
                "end_date": dates[1],
                "relative_period": None
            })
        elif len(dates) == 1:
            result.update({
                "start_date": dates[0],
                "end_date": dates[0],
                "relative_period": None
            })
        
        return result
    
    def _convert_to_date_range(self, date_info: Dict[str, Any], original_query: str) -> DateRange:
        """Convert extracted date info to DateRange object"""
        
        # Parse start and end dates
        start_date = self._parse_date_value(date_info.get("start_date"))
        end_date = self._parse_date_value(date_info.get("end_date"))
        
        # Ensure end_date is not before start_date
        if end_date < start_date:
            end_date = start_date
        
        # Convert to datetime objects
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        return DateRange(
            start_date=start_datetime,
            end_date=end_datetime,
            granularity=TimeGranularity(date_info.get("granularity", "monthly")),
            calculation_type=CalculationType(date_info.get("calculation_type", "total")),
            relative_period=date_info.get("relative_period"),
            original_query=original_query
        )
    
    def _parse_date_value(self, date_value: str) -> date:
        """Parse date value (absolute or relative) to date object"""
        
        if not date_value:
            return self.today
        
        # Handle absolute dates
        if re.match(r'\d{4}-\d{2}-\d{2}', date_value):
            return datetime.strptime(date_value, '%Y-%m-%d').date()
        
        # Handle relative dates
        today = self.today
        
        if date_value == "today":
            return today
        elif date_value == "yesterday":
            return today - timedelta(days=1)
        elif date_value == "this_week_start":
            days_since_monday = today.weekday()
            return today - timedelta(days=days_since_monday)
        elif date_value == "this_week_end":
            days_since_monday = today.weekday()
            return today + timedelta(days=6-days_since_monday)
        elif date_value == "last_week_start":
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday + 7)
            return last_monday
        elif date_value == "last_week_end":
            days_since_monday = today.weekday()
            last_sunday = today - timedelta(days=days_since_monday + 1)
            return last_sunday
        elif date_value == "this_month_start":
            return today.replace(day=1)
        elif date_value == "this_month_end":
            next_month = today.replace(day=28) + timedelta(days=4)
            return next_month - timedelta(days=next_month.day)
        elif date_value == "last_month_start":
            first_day_this_month = today.replace(day=1)
            last_month = first_day_this_month - timedelta(days=1)
            return last_month.replace(day=1)
        elif date_value == "last_month_end":
            first_day_this_month = today.replace(day=1)
            return first_day_this_month - timedelta(days=1)
        elif date_value == "this_quarter_start":
            quarter = (today.month - 1) // 3 + 1
            return date(today.year, (quarter - 1) * 3 + 1, 1)
        elif date_value == "this_quarter_end":
            quarter = (today.month - 1) // 3 + 1
            month = quarter * 3
            return date(today.year, month, calendar.monthrange(today.year, month)[1])
        elif date_value == "last_quarter_start":
            quarter = (today.month - 1) // 3 + 1
            if quarter == 1:
                return date(today.year - 1, 10, 1)
            else:
                return date(today.year, (quarter - 2) * 3 + 1, 1)
        elif date_value == "last_quarter_end":
            quarter = (today.month - 1) // 3 + 1
            if quarter == 1:
                return date(today.year - 1, 12, 31)
            else:
                month = (quarter - 1) * 3
                return date(today.year, month, calendar.monthrange(today.year, month)[1])
        else:
            # Default to today if unable to parse
            return today
    
    def _get_default_date_range(self, user_query: str) -> DateRange:
        """Get default date range when parsing fails"""
        
        # Default to last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        return DateRange(
            start_date=start_date,
            end_date=end_date,
            granularity=TimeGranularity.DAILY,
            calculation_type=CalculationType.TOTAL,
            relative_period="last_30_days",
            original_query=user_query
        )
    
    def validate_date_range(self, date_range: DateRange) -> bool:
        """Validate if date range is reasonable"""
        
        # Check if start date is not in the future
        if date_range.start_date.date() > self.today:
            return False
        
        # Check if date range is not too large (e.g., more than 5 years)
        max_days = 5 * 365  # 5 years
        if (date_range.end_date - date_range.start_date).days > max_days:
            return False
        
        # Check if end date is not before start date
        if date_range.end_date < date_range.start_date:
            return False
        
        return True
    
    def get_date_range_description(self, date_range: DateRange) -> str:
        """Get human-readable description of date range"""
        
        start_str = date_range.start_date.strftime('%Y-%m-%d')
        end_str = date_range.end_date.strftime('%Y-%m-%d')
        
        if date_range.relative_period:
            return f"{date_range.relative_period} ({start_str} to {end_str})"
        else:
            return f"{start_str} to {end_str}"


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the DateRangeParser
    pass

