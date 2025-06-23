import json

class GroupByQueryTool:
    def __init__(self):
        pass

    def group_by_sales_data(self, data: str, group_by_field: str, aggregate_field: str = None, aggregate_type: str = "sum"):
        """Groups sales data by a specified field and performs aggregation."""
        try:
            data_list = json.loads(data.replace("\\\\\\'", "\"")) # Handle single quotes from agent output

            if not data_list:
                return "No data provided for grouping."

            grouped_results = {}
            for item in data_list:
                key = item.get(group_by_field)
                if key is None:
                    continue

                if key not in grouped_results:
                    grouped_results[key] = []
                grouped_results[key].append(item)
            
            if aggregate_field and aggregate_type:
                aggregated_output = {}
                for key, items in grouped_results.items():
                    values = [float(item[aggregate_field]) for item in items if aggregate_field in item and isinstance(item[aggregate_field], (int, float, str))]
                    
                    if not values:
                        aggregated_output[key] = f"No valid {aggregate_field} for {key}"
                        continue

                    if aggregate_type == "sum":
                        aggregated_output[key] = sum(values)
                    elif aggregate_type == "avg":
                        aggregated_output[key] = sum(values) / len(values)
                    elif aggregate_type == "count":
                        aggregated_output[key] = len(values)
                    elif aggregate_type == "min":
                        aggregated_output[key] = min(values)
                    elif aggregate_type == "max":
                        aggregated_output[key] = max(values)
                    else:
                        aggregated_output[key] = f"Unsupported aggregation type: {aggregate_type}"
                return json.dumps(aggregated_output, indent=2)
            else:
                # If no aggregation, just return the grouped raw data (can be very large)
                return json.dumps(grouped_results, indent=2)

        except json.JSONDecodeError:
            return "Error: Invalid JSON format in input data."
        except KeyError as e:
            return f"Error: Missing expected field in data: {e}. Please check your data structure or group_by_field/aggregate_field."
        except Exception as e:
            return f"An unexpected error occurred during grouping: {e}"


