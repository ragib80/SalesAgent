# SAP Sales Analysis AI Agent

## Overview

This project is an AI-powered SAP Sales Analysis Chat Agent designed to analyze SAP sales data using natural language queries.

The system accepts user prompts, converts them into KQL (Kusto Query Language), executes the queries against Azure Data Explorer (ADX), and returns analytical insights to the user.

The goal is to evolve the current chatbot into a highly accurate, scalable, context-aware, and intelligent SAP Sales Analysis AI Agent using LangGraph, Azure OpenAI, Azure AI Search, RAG, Server-Sent Events (SSE), and advanced orchestration patterns.

## Current System Flow

```text
User Prompt
  -> AI Model
  -> KQL Generation
  -> Execute Query on Azure Data Explorer (ADX)
  -> Return Result to User
```

## Target System Flow

```text
User Prompt
  -> Django API
  -> LangGraph Orchestrator
  -> Intent Detection
  -> Conversation Memory
  -> RAG Context Retrieval
  -> Structured Query Planning
  -> KQL Generation
  -> KQL Validation
  -> ADX Query Execution
  -> Insight Generation
  -> Visualization / SSE Streaming Response
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Django |
| LLM | Azure OpenAI |
| Agent Orchestration | LangGraph |
| Analytics Database | Azure Data Explorer (ADX) |
| Query Language | KQL |
| Business Data | SAP Sales Data |
| Memory Layer | MS SQL Server |
| RAG / Knowledge Search | Azure AI Search / Azure Cognitive Search |
| Streaming | Server-Sent Events (SSE) |
| Visualization | Plotly / ECharts |
| Observability | LangSmith + OpenTelemetry |

## Project Goal

Build an enterprise-grade SAP Sales Analysis AI Agent capable of:

- Natural language understanding
- Intelligent KQL generation
- Structured query planning
- Query validation and auto-repair
- KPI and business-context understanding
- Multi-step reasoning
- Conversational memory
- RAG-based business grounding
- Data visualization
- Streaming responses
- Trend and anomaly detection
- Scalable orchestration
- Production-grade observability and monitoring

## Recommended Enterprise Architecture

```text
Frontend UI
  -> Django API Layer
  -> SSE Streaming Layer
  -> LangGraph Orchestrator
      -> Intent Analyzer
      -> Memory Node
      -> RAG Retriever
      -> Query Planner
      -> KQL Generator
      -> Query Validator
      -> Query Repair Node
      -> ADX Execution Node
      -> Visualization Node
      -> Summary Node
  -> Azure Data Explorer (ADX)
  -> Final Response / Chart / Dashboard
```

## LangGraph Design

LangGraph should be used as the orchestration layer around the current Django, Azure OpenAI, ADX, and KQL flow.

### Recommended Nodes

| Node | Responsibility |
|---|---|
| Intent Node | Detect whether the user is asking a sales-analysis question |
| Memory Node | Load conversation history and user context |
| Retriever Node | Fetch SAP business context from Azure AI Search |
| Planner Node | Create a structured analytical query plan |
| KQL Generator Node | Generate KQL from the validated plan |
| Validator Node | Validate generated KQL against schema, safety, and access rules |
| Repair Node | Repair invalid or failed KQL |
| Executor Node | Execute KQL against ADX |
| Visualization Node | Generate chart/table/dashboard instructions |
| Summary Node | Generate business insights from query results |
| SSE Node | Stream progress, partial responses, and final results to the UI |

### Recommended Graph Flow

```text
START
  -> intent_node
  -> memory_node
  -> rag_retriever_node
  -> planner_node
  -> plan_validator_node
  -> kql_generator_node
  -> kql_validator_node
  -> adx_executor_node
  -> visualization_node
  -> summary_node
  -> END
```

If validation or execution fails:

```text
kql_validator_node
  -> repair_node
  -> kql_validator_node

adx_executor_node
  -> repair_node
  -> kql_validator_node
```

## Server-Sent Events (SSE)

SSE is worth using for this project.

The SAP Sales Analysis Agent may take time to classify intent, retrieve context, generate KQL, execute ADX queries, repair errors, and summarize insights. SSE improves the user experience by streaming progress and partial output instead of making the user wait for one final blocking response.

### Best Uses for SSE

- Stream assistant text as it is generated
- Show LangGraph node progress
- Show query status such as "Generating KQL", "Validating query", "Executing ADX query", and "Preparing insights"
- Stream partial analytical summaries
- Send final chart/table metadata to the frontend
- Keep long-running requests from feeling frozen

### When SSE Is a Good Fit

Use SSE when communication is mostly one-way:

```text
Server -> Browser
```

This matches the chat-analysis flow well because the frontend sends a user prompt, then the server streams progress and results back.

### When WebSockets May Be Better

Use WebSockets instead of SSE only if the app needs frequent two-way real-time interaction, such as:

- Collaborative dashboards
- Live user interruptions while the agent is running
- Multiple users editing or controlling the same analytical session
- Bidirectional event streams

For this project, SSE is simpler and likely enough for the first production version.

### Recommended SSE Event Types

```text
event: status
data: {"message": "Analyzing user intent"}

event: kql
data: {"query": "..."}

event: result_preview
data: {"rows": [...]}

event: chart
data: {"chart_type": "bar", "x": "Dealer", "y": "Revenue"}

event: final
data: {"answer": "..."}

event: error
data: {"message": "Query validation failed"}
```

### Django SSE Concept

```python
from django.http import StreamingHttpResponse
import json


def sse_event(event_name, payload):
    data = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {data}\n\n"


def chat_stream_view(request):
    def event_stream():
        yield sse_event("status", {"message": "Analyzing request"})
        yield sse_event("status", {"message": "Generating KQL"})
        yield sse_event("status", {"message": "Executing ADX query"})
        yield sse_event("final", {"answer": "Analysis completed"})

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
```

## RAG Integration

RAG should be used for business knowledge, not for querying raw SAP sales transactions.

ADX should remain the source for transactional sales analysis. Azure AI Search should retrieve relevant business context before query planning and KQL generation.

### RAG Knowledge Sources

- KPI definitions
- SAP terminology
- Product hierarchy
- Brand aliases
- Dealer/customer aliases
- Fiscal calendar rules
- Business area/depot mappings
- Sales process documentation
- Regional sales policies
- Pricing definitions
- Analytical best practices
- Approved KQL examples

### RAG Architecture

```text
SAP Documents
  -> Chunking
  -> Embedding Generation
  -> Azure AI Search Index
  -> Retriever Node
  -> Query Planner
  -> KQL Generator
```

### Example

User prompt:

```text
Show inactive dealers for waterproofing products in Dhaka North this fiscal year.
```

Azure AI Search can retrieve:

```text
Inactive dealer = a dealer who purchased in the previous fiscal year but has no purchase in the current fiscal year.
Waterproofing product group = approved product category list.
Dhaka North = business area code 4130.
Fiscal year = April 1 to March 31.
```

The planner then creates a grounded query plan instead of guessing business definitions.

## Structured Query Planning

Before generating KQL, the agent should convert the user prompt into a structured analytical plan.

### Example User Prompt

```text
Show me top 10 products by revenue in Dhaka for last quarter.
```

### Example Structured Plan

```json
{
  "intent": "ranking",
  "metric": "Revenue",
  "aggregation": "sum",
  "dimensions": ["Product"],
  "filters": {
    "location": "Dhaka"
  },
  "time_range": {
    "label": "last quarter",
    "calendar_type": "fiscal"
  },
  "sorting": {
    "field": "TotalRevenue",
    "direction": "desc"
  },
  "limit": 10
}
```

### Benefits

- More accurate KQL generation
- Reduced hallucinations
- Easier debugging
- Better explainability
- Cleaner validation

## KQL Validation and Auto-Repair

Generated KQL should be treated as executable code and validated before execution.

### Validation Checks

| Area | Checks |
|---|---|
| Syntax | Invalid operators, broken joins, malformed filters |
| Schema | Unknown tables, unknown columns, wrong data types |
| Security | Unsafe commands, cross-cluster calls, unrestricted scans |
| Performance | Missing limits, missing date filters, inefficient scans |
| Access Control | Missing company, depot, territory, or zone restrictions |

### Auto-Repair Flow

```text
Generate KQL
  -> Validate Query
  -> If Valid: Execute Query
  -> If Invalid: Repair Query
  -> Validate Again
  -> Execute Query
```

## Security and Access Control

Security should not depend only on prompt instructions.

Recommended controls:

- Enforce user data scope in backend code
- Use ADX RBAC and Row-Level Security where possible
- Use Microsoft Entra ID / managed identity for service authentication
- Store secrets in Azure Key Vault or secure environment variables
- Block unsafe KQL operations
- Allowlist approved tables and columns
- Enforce mandatory query limits
- Log every generated KQL query
- Audit user, conversation ID, query, result size, latency, and errors
- Avoid exposing raw internal authorization rules to the LLM response

## Performance Recommendations

- Apply selective `where` filters early
- Always filter by authorized scope
- Always apply date filters when the question implies a period
- Use `project` to reduce unnecessary columns
- Prefer aggregated results over raw row returns
- Add `top`, `take`, and result-size limits
- Use ADX materialized views for common KPI aggregations
- Cache repeated query plans and frequent result sets
- Use async/background execution for large exports
- Stream progress with SSE for long-running requests
- Add/modify the existing models if it needed. (mention this in a new file if sql is chnaged)

## Suggested Folder Structure

```text
src/
  agent/
    graph/
      workflow.py
      nodes.py
      state.py
    planner/
      prompts.py
      schemas.py
    kql/
      generator.py
      validator.py
      repair.py
    rag/
      retriever.py
      indexer.py
    services/
      adx.py
      openai_client.py
      memory.py
      streaming.py
    visualization/
      chart_spec.py
    observability/
      tracing.py
    tests/
      test_planner.py
      test_kql_validator.py
      test_graph_flow.py
```

## Implementation Roadmap

### Phase 1 - Wrap Existing Flow into LangGraph

Objective: integrate the existing workflow into LangGraph without changing current behavior.

Goals:

- Convert the current pipeline into graph-based orchestration
- Separate responsibilities into nodes
- Enable future extensibility
- Maintain existing business logic
- Add basic SSE progress events

Expected outcome:

- Modular architecture
- Easier debugging
- Better observability
- Foundation for advanced AI workflows

### Phase 2 - Structured Query Planning

Objective: introduce an intermediate query planning layer before KQL generation.

The planner should:

- Understand user intent
- Identify KPIs
- Detect dimensions
- Extract filters
- Parse time ranges
- Detect grouping requirements
- Generate structured analytical plans

### Phase 3 - KQL Validation and Auto-Repair

Objective: add validation and automatic repair mechanisms for generated KQL queries.

Goals:

- Validate syntax
- Validate schema references
- Detect hallucinated columns/tables
- Prevent unsafe queries
- Retry failed queries automatically
- Repair invalid KQL

### Phase 4 - RAG Integration for SAP Business Knowledge

Objective: add Retrieval-Augmented Generation to improve business understanding.

Use Azure AI Search / Azure Cognitive Search for:

- KPI definitions
- SAP terminology
- Business rules
- Product hierarchy
- Fiscal calendar logic
- Approved analytical examples

### Phase 5 - Persistent Memory and Visualization

Objective: add conversational memory and advanced analytical capabilities.

Features:

- Conversation history
- User preferences
- Frequently used KPIs
- Session-aware analytics
- Charts
- KPI dashboards
- Trend graphs
- Time-series analysis

### Phase 6 - Production Hardening

Objective: make the agent production-ready.

Focus areas:

- Authentication and authorization
- Query audit logs
- Prompt/model versioning
- Evaluation datasets
- Regression tests for KQL generation
- Latency and token monitoring
- Error analytics
- Cost monitoring
- Deployment and rollback strategy

## Recommended Prompting Strategy

### Planner Prompt

```text
Analyze the user's analytical request and extract:
- Intent
- KPI
- Dimensions
- Filters
- Time range
- Aggregation
- Sorting
- Comparison requirements
- Visualization preference

Return structured JSON only.
```

### KQL Generator Prompt

```text
Generate production-ready KQL using:
- Approved schema
- Allowed tables and columns
- Validated query plan
- User access scope
- Performance-optimized filters
- Safe aggregations

Return only valid KQL.
```

### Summary Prompt

```text
Generate a concise SAP sales analysis summary using only the query result JSON.
Do not invent numbers.
Mention the analysis period and applied filters.
Highlight business insights, risks, and opportunities.
```

## Advanced Features

Future specialized agents may include:

- KPI Analysis Agent
- Forecasting Agent
- Trend Detection Agent
- Root Cause Analysis Agent
- Recommendation Agent
- Anomaly Detection Agent

## Recommended Production Stack

| Component | Recommendation |
|---|---|
| LLM | Azure OpenAI chat deployments, such as GPT-4.1 and GPT-5.x depending on availability |
| Orchestration | LangGraph |
| RAG / Knowledge Search | Azure AI Search / Azure Cognitive Search |
| Memory | MS SQL Server |
| Analytics DB | Azure Data Explorer |
| Streaming | Server-Sent Events (SSE) |
| Monitoring | LangSmith + OpenTelemetry |
| API Layer | Django |
| Visualization | Plotly / ECharts |

## Final Goal

Build a highly intelligent SAP Sales Analysis AI Agent capable of:

- Conversational analytics
- Business-aware reasoning
- Reliable KQL generation
- Enterprise-scale execution
- Real-time insight generation
- Streaming progress and responses
- Autonomous analytical workflows
- Executive-level business summarization
