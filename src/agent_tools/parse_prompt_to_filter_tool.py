import json
import re

class PromptToFilterTool:
    def __init__(self):
        pass

    def parse_prompt_to_filter(self, prompt: str):
        """Parses a natural language prompt to extract filter conditions for sales data.

        Args:
            prompt (str): The natural language prompt from the user.

        Returns:
            str: A JSON string representing the filter conditions.
        """
        filters = {}

        # Example: Extracting product names
        product_matches = re.findall(r'product\s+["\']?([\w\s]+?)["\']?', prompt, re.IGNORECASE)
        if product_matches:
            filters["product_name"] = product_matches[0].strip()

        # Example: Extracting date ranges (e.g., \'in Q1 2023\', \'from Jan to March\')
        # This is a very basic example and would need significant expansion for robust date parsing.
        date_matches = re.findall(r"in\s+(Q[1-4]\s+\d{4})|from\s+(\w+\s+to\s+\w+)", prompt, re.IGNORECASE)
        if date_matches:
            if date_matches[0][0]: # QX YYYY format
                filters["date_range"] = date_matches[0][0].strip()
            elif date_matches[0][1]: # from Month to Month format
                filters["date_range"] = date_matches[0][1].strip()

        # Example: Extracting sales amount conditions (e.g., \'sales greater than 1000\')
        sales_amount_matches = re.findall(r"sales\s+(greater\s+than|less\s+than|equal\s+to)\s+(\d+)", prompt, re.IGNORECASE)
        if sales_amount_matches:
            condition = sales_amount_matches[0][0].lower()
            amount = int(sales_amount_matches[0][1])
            if "greater than" in condition:
                filters["sales_amount_gt"] = amount
            elif "less than" in condition:
                filters["sales_amount_lt"] = amount
            elif "equal to" in condition:
                filters["sales_amount_eq"] = amount

        # More complex parsing would involve a more sophisticated NLP approach
        # or using a dedicated library for natural language to query conversion.

        return json.dumps(filters, indent=2)


