# enhanced_kql_generator.py - Enhanced KQL Generator for Time-based Analytics
import re
from typing import Dict, List, Optional
from datetime import datetime

from agent_tools.date_range_parser import DateRange, TimeGranularity, CalculationType


class EnhancedKQLGenerator:
    """Enhanced KQL generator that works with DateRange objects for time-based analytics"""
    
    def __init__(self, table_name: str = "SAPSalesInfos"):
        self.table_name = table_name
        self.field_mappings = {
            "revenue": "Revenue",
            "quantity": "fkimg", 
            "volume": "volum",
            "dealer": "cname",
            "brand": "wgbez",
            "product name": "arktx",
            "product": "arktx",
            "category": "matkl",
            "division": "spart_text",
            "company code": "bukrs",
            "sales org": "vkorg",
            "dist channel": "vtweg",
            "distribution channel": "vtweg",
            "business area": "gsber",
            "depo": "gsber",
            "credit control area": "kkber",
            "dealer group": "kukla",
            "account group": "ktokd",
            "sales group": "vkgrp_c",
            "sales office": "vkbur_c",
            "payer id": "Payer_DL",
            "product code": "matnr",
            "unit": "meins",
            "volume unit": "voleh",
            "business group": "GK",
            "territory": "Territory",
            "sales zone": "Szone",
            "date": "fkdat",
            "fkdat": "fkdat"
        }
        
        self.gsber_mapping = {
            "Dhaka Factory": "1000",
            "Chittagong Factory": "1100",
            "Mirsarai Factory": "1200",
            "Dhaka Sales": "4000",
            "Chittagong Sales": "4010",
            "Sylhet Sales": "4020",
            "Comilla Sales": "4030",
            "Rajshahi Sales": "4040",
            "Bogra Sales": "4050",
            "Khulna Sales": "4060",
            "Mymensing Sales": "4070",
            "Barishal Sales": "4080",
            "Rangpur Sales": "4090",
            "Feni Sales": "4100",
            "Dhaka South": "4110",
            "Brahmanbaria Sales": "4120",
            "Dhaka North": "4130",
            "Test Business Area": "4500",
            "PPHD": "5000",
            "Berger Design Studio": "5010",
            "Berger Training Institute": "5020",
            "Berger Tech Consulting Ltd": "5100",
            "Jenson & Nicholson BD Ltd": "6000",
            "JNBL 2nd Unit Dhaka": "6100",
            "Berger Becker Bangladesh": "7000",
            "Berger Fosroc Limited": "8000",
            "Corporate": "9000"
        }
    
    def generate_time_based_kql(self, user_query: str, date_range: DateRange, 
                               additional_filters: Optional[Dict] = None) -> str:
        """
        Generate KQL query for time-based analytics using DateRange object
        """
        
        # Start building the query
        kql_parts = []
        
        # Add date range variables
        start_date_str = date_range.start_date.strftime('%Y-%m-%d')
        end_date_str = date_range.end_date.strftime('%Y-%m-%d')
        
        kql_parts.append(f"let StartDate = datetime({start_date_str});")
        kql_parts.append(f"let EndDate = datetime({end_date_str});")
        kql_parts.append("")
        
        # Start main query
        kql_parts.append(f"{self.table_name}")
        
        # Add date filter
        kql_parts.append("| where fkdat >= StartDate and fkdat <= EndDate")
        
        # Add additional filters if provided
        if additional_filters:
            for field, value in additional_filters.items():
                if field in self.field_mappings:
                    mapped_field = self.field_mappings[field]
                    if isinstance(value, str):
                        kql_parts.append(f"| where {mapped_field} == '{value}'")
                    else:
                        kql_parts.append(f"| where {mapped_field} == {value}")
        
        # Add business area mapping if mentioned in query
        business_area_filter = self._extract_business_area_filter(user_query)
        if business_area_filter:
            kql_parts.append(business_area_filter)
        
        # Add time-based grouping and aggregation
        time_grouping = self._get_time_grouping(date_range.granularity)
        aggregation = self._get_aggregation(date_range.calculation_type)
        
        if time_grouping:
            kql_parts.append(f"| extend TimeGroup = {time_grouping}")
            kql_parts.append(f"| summarize {aggregation} by TimeGroup")
            kql_parts.append("| order by TimeGroup asc")
        else:
            # No time grouping, just aggregate all data
            kql_parts.append(f"| summarize {aggregation}")
        
        return "\n".join(kql_parts) + ";"
    
    def _get_time_grouping(self, granularity: TimeGranularity) -> str:
        """Get KQL time grouping expression based on granularity"""
        
        if granularity == TimeGranularity.DAILY:
            return "startofday(fkdat)"
        elif granularity == TimeGranularity.WEEKLY:
            return "startofweek(fkdat)"
        elif granularity == TimeGranularity.MONTHLY:
            return "startofmonth(fkdat)"
        elif granularity == TimeGranularity.QUARTERLY:
            return "startofmonth(datetime_add('month', (month(fkdat)-1)/3*3, startofyear(fkdat)))"
        elif granularity == TimeGranularity.YEARLY:
            return "startofyear(fkdat)"
        else:
            return None
    
    def _get_aggregation(self, calculation_type: CalculationType) -> str:
        """Get KQL aggregation expression based on calculation type"""
        
        base_aggregations = {
            CalculationType.TOTAL: "TotalRevenue = sum(Revenue), TotalQuantity = sum(fkimg), TotalVolume = sum(volum)",
            CalculationType.AVERAGE: "AvgRevenue = avg(Revenue), AvgQuantity = avg(fkimg), AvgVolume = avg(volum)",
            CalculationType.COUNT: "RecordCount = count(), UniqueProducts = dcount(matnr), UniqueDealers = dcount(cname)",
            CalculationType.MAX: "MaxRevenue = max(Revenue), MaxQuantity = max(fkimg), MaxVolume = max(volum)",
            CalculationType.MIN: "MinRevenue = min(Revenue), MinQuantity = min(fkimg), MinVolume = min(volum)"
        }
        
        return base_aggregations.get(calculation_type, base_aggregations[CalculationType.TOTAL])
    
    def _extract_business_area_filter(self, user_query: str) -> Optional[str]:
        """Extract business area filter from user query"""
        
        query_lower = user_query.lower()
        
        for territory, gsber_value in self.gsber_mapping.items():
            if territory.lower() in query_lower:
                return f"| where gsber == '{gsber_value}'"
        
        return None
    
    def generate_enhanced_kql(self, user_query: str, date_range: DateRange) -> str:
        """
        Generate enhanced KQL with better field mapping and error handling
        """
        
        try:
            # Extract additional filters from user query
            additional_filters = self._extract_filters_from_query(user_query)
            
            # Generate time-based KQL
            kql = self.generate_time_based_kql(user_query, date_range, additional_filters)
            
            # Apply post-processing fixes
            kql = self._apply_kql_fixes(kql)
            
            return kql
            
        except Exception as e:
            print(f"Error generating enhanced KQL: {e}")
            # Fallback to simple query
            return self._generate_fallback_kql(date_range)
    
    def _extract_filters_from_query(self, user_query: str) -> Dict:
        """Extract additional filters from user query"""
        
        filters = {}
        query_lower = user_query.lower()
        
        # Extract product/brand filters
        if "product" in query_lower:
            # This would need more sophisticated extraction
            pass
        
        # Extract dealer filters  
        if "dealer" in query_lower:
            # This would need more sophisticated extraction
            pass
        
        # Extract division filters
        if "division" in query_lower:
            # This would need more sophisticated extraction
            pass
        
        return filters
    
    def _apply_kql_fixes(self, kql: str) -> str:
        """Apply common KQL fixes and optimizations"""
        
        # Fix unsupported functions
        kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
        kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)
        
        # Ensure proper datetime formatting
        kql = re.sub(r'(\d{4}-\d{2}-\d{2})', r'datetime(\1)', kql)
        
        # Remove duplicate datetime() calls
        kql = re.sub(r'datetime\(datetime\(([^)]+)\)\)', r'datetime(\1)', kql)
        
        return kql
    
    def _generate_fallback_kql(self, date_range: DateRange) -> str:
        """Generate simple fallback KQL when enhanced generation fails"""
        
        start_date_str = date_range.start_date.strftime('%Y-%m-%d')
        end_date_str = date_range.end_date.strftime('%Y-%m-%d')
        
        return f"""
let StartDate = datetime({start_date_str});
let EndDate = datetime({end_date_str});

{self.table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| summarize TotalRevenue = sum(Revenue), TotalQuantity = sum(fkimg), RecordCount = count()
| order by TotalRevenue desc;
"""
    
    def get_kql_templates(self) -> Dict[str, str]:
        """Get predefined KQL templates for common queries"""
        
        templates = {
            "daily_sales": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| extend Day = startofday(fkdat)
| summarize TotalRevenue = sum(Revenue), AvgRevenue = avg(Revenue), RecordCount = count() by Day
| order by Day asc;
""",
            
            "weekly_sales": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| extend Week = startofweek(fkdat)
| summarize TotalRevenue = sum(Revenue), AvgRevenue = avg(Revenue), RecordCount = count() by Week
| order by Week asc;
""",
            
            "monthly_sales": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| extend Month = startofmonth(fkdat)
| summarize TotalRevenue = sum(Revenue), AvgRevenue = avg(Revenue), RecordCount = count() by Month
| order by Month asc;
""",
            
            "quarterly_sales": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| extend Quarter = startofmonth(datetime_add('month', (month(fkdat)-1)/3*3, startofyear(fkdat)))
| summarize TotalRevenue = sum(Revenue), AvgRevenue = avg(Revenue), RecordCount = count() by Quarter
| order by Quarter asc;
""",
            
            "yearly_sales": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| extend Year = startofyear(fkdat)
| summarize TotalRevenue = sum(Revenue), AvgRevenue = avg(Revenue), RecordCount = count() by Year
| order by Year asc;
""",
            
            "business_area_analysis": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| summarize TotalRevenue = sum(Revenue), AvgRevenue = avg(Revenue), RecordCount = count() by gsber
| order by TotalRevenue desc;
""",
            
            "product_performance": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| summarize TotalRevenue = sum(Revenue), TotalQuantity = sum(fkimg), RecordCount = count() by arktx
| order by TotalRevenue desc
| take 20;
""",
            
            "dealer_performance": """
let StartDate = datetime({start_date});
let EndDate = datetime({end_date});

{table_name}
| where fkdat >= StartDate and fkdat <= EndDate
| summarize TotalRevenue = sum(Revenue), TotalQuantity = sum(fkimg), RecordCount = count() by cname
| order by TotalRevenue desc
| take 20;
"""
        }
        
        return templates
    
    def use_template(self, template_name: str, date_range: DateRange, **kwargs) -> str:
        """Use a predefined template with date range"""
        
        templates = self.get_kql_templates()
        
        if template_name not in templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = templates[template_name]
        
        # Format template with date range and other parameters
        formatted_kql = template.format(
            start_date=date_range.start_date.strftime('%Y-%m-%d'),
            end_date=date_range.end_date.strftime('%Y-%m-%d'),
            table_name=self.table_name,
            **kwargs
        )
        
        return formatted_kql.strip()


# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    # Create test date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    test_date_range = DateRange(
        start_date=start_date,
        end_date=end_date,
        granularity=TimeGranularity.DAILY,
        calculation_type=CalculationType.TOTAL
    )
    
    # Test KQL generator
    generator = EnhancedKQLGenerator()
    
    # Test template usage
    daily_kql = generator.use_template("daily_sales", test_date_range)
    print("Daily Sales KQL:")
    print(daily_kql)
    print("\n" + "="*50 + "\n")
    
    # Test enhanced KQL generation
    enhanced_kql = generator.generate_enhanced_kql("Show me daily sales for last month", test_date_range)
    print("Enhanced KQL:")
    print(enhanced_kql)

