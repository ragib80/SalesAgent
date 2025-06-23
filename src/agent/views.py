# import json
# import os
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.shortcuts import render

# from azure.ai.projects import AIProjectClient
# from azure.identity import DefaultAzureCredential
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from openai import AzureOpenAI

# # Import your refactored tool classes
# # These tools are not directly called here, but their definitions are used by the agent
# from agent_tools.groupby_query_tool import GroupByQueryTool
# from agent_tools.parse_prompt_to_filter_tool import PromptToFilterTool
# from agent_tools.summarize_sales_tool import SalesSummarizerTool
# from agent_tools.vector_search_tool import VectorSearchTool

# # Load environment variables (already done in settings.py, but good for clarity in a standalone script)
# # from dotenv import load_dotenv
# # load_dotenv()

# # Initialize clients once (or retrieve from Django settings if preferred)
# # For simplicity, we'll re-initialize here, but in a larger app, consider a singleton pattern
# project_client = AIProjectClient(
#     endpoint=os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT"),
#     credential=DefaultAzureCredential()
# )

# search_client = SearchClient(
#     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
#     index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
#     credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
# )

# openai_client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2024-02-01" # Use a compatible API version
# )

# # Initialize tool instances (these will be used by the agent in Azure, not directly by Django here)
# # We initialize them here to get their tool_definitions for agent creation if needed,
# # or if you were to call them directly from Django for some reason.
# groupby_tool_instance = GroupByQueryTool()
# prompt_filter_tool_instance = PromptToFilterTool()
# summarize_tool_instance = SalesSummarizerTool(openai_client, search_client)
# vector_search_tool_instance = VectorSearchTool(openai_client, search_client, os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))

# # --- Django Views ---

# def index(request):
#     return render(request, 'index.html')

# # @csrf_exempt # For simplicity, disable CSRF for this example. In production, use proper CSRF protection.
# # def analyze_sales(request):
# #     if request.method == 'POST':
# #         try:
# #             data = json.loads(request.body)
# #             user_query = data.get('query')
# #             agent_id = os.getenv("AZURE_AI_FOUNDRY_AGENT_ID") # Get agent ID from environment variable

# #             if not user_query:
# #                 return JsonResponse({'error': 'No query provided'}, status=400)
            
# #             if not agent_id:
# #                 return JsonResponse({'error': 'AZURE_AI_FOUNDRY_AGENT_ID environment variable not set. Please create the agent first.'}, status=500)

# #             # To interact with the agent, you create a thread and send messages
# #             thread = project_client.agents.threads.create()
# #             message = project_client.agents.messages.create(
# #                 thread_id=thread.id,
# #                 role="user",
# #                 content=user_query
# #             )

# #             # Run the agent to process the message
# #             run = project_client.agents.runs.create_and_process(
# #                 thread_id=thread.id,
# #                 agent_id=agent_id # Specify the agent ID
# #             )

# #             if run.status == "failed":
# #                 return JsonResponse({'error': f'Agent run failed: {run.last_error}'}, status=500)

# #             # Retrieve the agent's response messages
# #             messages = project_client.agents.messages.list(thread_id=thread.id)
# #             agent_response = "No response from agent."
# #             for msg in messages:
# #                 if msg.role == "assistant": # Assuming the agent's response is from the 'assistant' role
# #                     agent_response = msg.content
# #                     break
            
# #             # Clean up the thread (optional, depending on your conversation management)
# #             # project_client.agents.threads.delete(thread.id)

# #             return JsonResponse({'response': agent_response})

# #         except Exception as e:
# #             return JsonResponse({'error': str(e)}, status=500)
# #     else:
# #         return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)




# # Replace with your Azure AI Foundry Agent endpoint and API key
# AZURE_AI_AGENT_ENDPOINT = os.environ.get("AZURE_AI_AGENT_ENDPOINT")
# AZURE_AI_AGENT_API_KEY = os.environ.get("AZURE_AI_AGENT_API_KEY")

# def analyze_sales(request):
#     if request.method == "POST":
#         user_query = request.POST.get("query")
#         if not user_query:
#             return JsonResponse({"error": "No query provided"}, status=400)

#         headers = {
#             "Content-Type": "application/json",
#             "api-key": AZURE_AI_AGENT_API_KEY  # Or "Authorization": f"Bearer {AZURE_AI_AGENT_API_KEY}" depending on API
#         }
#         payload = {
#             "query": user_query
#             # Add any other parameters required by your agent API
#         }

#         try:
#             response = requests.post(AZURE_AI_AGENT_ENDPOINT, headers=headers, json=payload)
#             response.raise_for_status()  # Raise an exception for HTTP errors
#             agent_response = response.json()
#             return JsonResponse({"response": agent_response})
#         except requests.exceptions.RequestException as e:
#             return JsonResponse({"error": str(e)}, status=500)
#     return JsonResponse({"message": "Send a POST request with a 'query' parameter"})
