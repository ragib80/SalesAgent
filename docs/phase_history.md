# Phase Implementation History

## Phase 1 - LangGraph Foundation

### Date
2026-06-08

### Objective
Wrap the existing SAP sales analysis flow in LangGraph without changing the current KQL generation, ADX execution, response formatting, or conversation persistence behavior.

### Architecture Changes
- Added a new `agent.graph` package for LangGraph orchestration.
- Introduced a typed graph state shared between workflow nodes.
- Wrapped the existing `agent.agent.handle_user_query` function in a graph node instead of rewriting current business logic.
- Routed the existing chat API through the graph runner.
- Added an optional SSE endpoint for graph progress events.

### Features Implemented
- Phase 1 LangGraph workflow with request initialization, existing-agent execution, and response finalization nodes.
- Cached graph compilation for reuse across requests.
- Structured graph events for `status`, `final`, and `error`.
- Additive SSE endpoint at `api/sales/query/stream/`.

### Refactoring Performed
- Updated `ChatAPIView` and `ExistingConversationAPIView` to call the graph wrapper.
- Preserved existing response fields: `answer`, `data`, `operation_plan`, and first-chat `uuid`.
- Fixed token cleanup in `ExistingConversationAPIView` so `clear_current_chat_user` only runs when a token was created.

### Files Added
- `src/agent/graph/__init__.py`
- `src/agent/graph/state.py`
- `src/agent/graph/nodes.py`
- `src/agent/graph/workflow.py`

### Files Modified
- `src/sales_analyzer/views.py`
- `src/sales_analyzer/urls.py`
- `src/agent/tests.py`
- `src/sales_analyzer/tests.py`
- `docs/phase_history.md`

### Breaking Changes
None. Existing `api/sales/query/` and `api/sales/query/existing/<uuid>/` response shapes are preserved.

### Migration Steps
- Existing clients can continue using the current chat endpoints.
- Clients that want progress events can opt into `api/sales/query/stream/`.

### Testing Performed
- Ran `..\venv\Scripts\python.exe manage.py test agent.tests sales_analyzer.tests`.
- Ran `..\venv\Scripts\python.exe manage.py check`.
- Attempted `..\venv\Scripts\python.exe manage.py test agent sales_analyzer`; full app test setup was blocked by the local SQL Server/ODBC test database connection.

### Known Issues or Limitations
- The graph currently wraps the existing monolithic `handle_user_query` function; deeper node separation will happen in later phases.
- SSE support is additive and backend-ready, but frontend adoption is not part of Phase 1.
- Full database-backed tests require a reachable SQL Server test database configuration.

### Lessons Learned
- `langgraph` was already available in project requirements, so no dependency change was required.
- The safest Phase 1 boundary is around the existing agent entry point because access control, KQL generation, retry logic, and summarization are currently tightly coupled there.

### Next Phase
Phase 2 should introduce structured query planning before KQL generation while keeping the Phase 1 graph entry point stable.

## Phase 2 - Structured Query Planning

### Date
2026-06-08

### Objective
Introduce a structured analytical query planning layer before KQL generation while preserving the existing chat API behavior, ADX execution path, access-scope enforcement, and response shape.

### Architecture Changes
- Added a new `agent.planner` package for Phase 2 query planning.
- Added a structured `QueryPlan` schema for intent, metrics, dimensions, filters, time range, aggregation, sorting, comparisons, visualization hints, confidence, and assumptions.
- Added a LangGraph `create_query_plan` node between request initialization and existing-agent execution.
- Extended graph state with `query_plan` and `query_plan_error`.
- Passed the generated plan into the existing agent and KQL generator as advisory context.

### Features Implemented
- Planner prompt that asks the LLM for strict JSON only.
- JSON extraction and normalization for planner responses.
- Support for single `metric` or multiple `metrics` in planner output.
- Fail-open planner behavior so existing query generation continues if planning fails.
- Phase 2 SSE/status event: `Structured query plan created` or `Structured query planning unavailable`.
- Advisory KQL prompt block named `STRUCTURED QUERY PLAN (Phase 2 advisory context)`.

### Refactoring Performed
- Updated `handle_user_query` to accept an optional `query_plan`.
- Updated `generate_kql` to include the structured plan in the KQL-generation prompt.
- Preserved existing date context, schema context, user access-scope rules, `bukrs` enforcement, ADX retry behavior, and response summarization.
- Updated graph tests to verify the planner node, planner fallback, and query-plan propagation.

### Files Added
- `src/agent/planner/__init__.py`
- `src/agent/planner/schemas.py`
- `src/agent/planner/prompts.py`
- `src/agent/planner/planner.py`

### Files Modified
- `src/agent/agent.py`
- `src/agent/graph/state.py`
- `src/agent/graph/nodes.py`
- `src/agent/graph/workflow.py`
- `src/agent/tests.py`
- `docs/phase_history.md`

### Breaking Changes
None. Existing `api/sales/query/`, `api/sales/query/existing/<uuid>/`, and `api/sales/query/stream/` response shapes are preserved.

### Migration Steps
- No database migration is required.
- Existing clients do not need to change.
- SSE clients may see one additional status event during query planning.

### Testing Performed
- Ran `..\venv\Scripts\python.exe manage.py test agent.tests sales_analyzer.tests`.
- Ran `..\venv\Scripts\python.exe manage.py check`.

### Known Issues or Limitations
- The structured plan currently grounds KQL generation but does not yet validate generated KQL.
- Planner output is advisory, so the existing KQL prompt remains the primary behavior controller.
- Planner quality depends on LLM JSON compliance; invalid planner output falls back to the existing path.
- Full database-backed app tests still depend on the local SQL Server/ODBC test database configuration.

### Lessons Learned
- The safest Phase 2 boundary is an advisory planning node because the current agent still owns security, scope filtering, date handling, and execution behavior.
- Passing the plan into `generate_kql` prepares the codebase for Phase 3 validation without requiring a broad rewrite.

### Next Phase
Phase 3 should add KQL validation and auto-repair using the structured plan, approved schema, blocked operations, and execution-error feedback.

## Phase 3 - KQL Validation and Auto-Repair

### Date
2026-06-08

### Objective
Add validation and automatic repair around generated KQL before ADX execution, while preserving the existing graph entry point, chat API response shape, access-scope behavior, and result summarization flow.

### Architecture Changes
- Added a new `agent.kql` package for KQL validation, deterministic cleanup, repair prompts, and validation result schemas.
- Inserted a validation-and-repair step inside `handle_user_query` after KQL generation and before ADX execution.
- Added ADX execution-error feedback into the repair flow so failed queries can be corrected before the second execution attempt.
- Kept Phase 3 inside the existing agent execution boundary so Phase 1/2 graph and API contracts remain stable.

### Features Implemented
- Static KQL validation for empty output, markdown/prose leakage, blocked management or mutation commands, cross-cluster/database references, unbalanced parentheses, missing semicolons, unknown tables/aliases, hallucinated columns, missing `bukrs == 1000`, and missing result-limit warnings.
- Deterministic KQL cleanup that strips fences/META lines, normalizes escaped newlines, replaces unsupported monthly `bin()` usage, enforces the mandatory company filter, and appends a trailing semicolon.
- LLM-based repair prompt using the original user request, structured query plan, approved schema, validation issues, and optional ADX error text.
- Fail-closed behavior when generated and repaired KQL remain invalid.
- Existing strict-regeneration fallback preserved after repair cannot produce a different executable query.

### Refactoring Performed
- Centralized existing post-generation KQL fixes in `_post_process_generated_kql`.
- Added `_validate_or_repair_kql` as the Phase 3 execution gate before ADX calls.
- Reused the Phase 2 `QueryPlan` as repair context instead of introducing a second planning shape.
- Preserved current narrative summary generation and friendly failure message behavior.

### Files Added
- `src/agent/kql/__init__.py`
- `src/agent/kql/schemas.py`
- `src/agent/kql/validator.py`
- `src/agent/kql/prompts.py`
- `src/agent/kql/repair.py`

### Files Modified
- `src/agent/agent.py`
- `src/agent/tests.py`
- `docs/phase_history.md`

### Breaking Changes
None. Existing `api/sales/query/`, `api/sales/query/existing/<uuid>/`, and `api/sales/query/stream/` response shapes are preserved.

### Migration Steps
- No database migration is required.
- Existing clients do not need to change.
- No new environment variables are required.

### Testing Performed
- Ran `cd src; ..\venv\Scripts\python.exe manage.py test agent.tests`.
- Ran `cd src; ..\venv\Scripts\python.exe manage.py test agent.tests sales_analyzer.tests`.
- Ran `cd src; ..\venv\Scripts\python.exe manage.py check`.
- Ran `.\venv\Scripts\python.exe -m compileall src\agent\kql src\agent\agent.py` from the repository root.

### Known Issues or Limitations
- The static validator is intentionally conservative; deep KQL syntax validation still relies on ADX execution feedback.
- KQL validation and repair currently run inside the existing agent node rather than as separate LangGraph nodes.
- SSE clients do not yet receive separate `Validating KQL` or `Repairing KQL` events.
- Full database-backed app tests still depend on the local SQL Server/ODBC test database configuration.

### Lessons Learned
- The safest Phase 3 integration point is directly before ADX execution because generation, access scope, existing retry behavior, and summarization are still concentrated in `handle_user_query`.
- A small deterministic cleanup pass catches common LLM formatting issues before spending another LLM call on repair.
- The Phase 2 structured plan is useful repair context without requiring new request or response contracts.

### Next Phase
Phase 4 should add RAG retrieval for SAP business knowledge using Azure AI Search, then feed retrieved definitions and approved examples into planning, KQL generation, and repair.

## Phase 4 - RAG Integration for SAP Business Knowledge

### Date
2026-06-09

### Objective
Add fail-open Azure AI Search RAG retrieval for SAP sales business knowledge while preserving the current hardcoded `SYSTEM_PROMPT_KQL` behavior as the authoritative fallback.

### Architecture Changes
- Added a new `agent.rag` package for SAP sales business/context knowledge document generation and Azure AI Search indexing helpers.
- Added a fail-open Azure AI Search retriever for Phase 4 business context retrieval.
- Added a `retrieve_business_context` LangGraph node before structured query planning.
- Extended graph state with `rag_context` and `rag_context_error`.
- Passed retrieved RAG context into structured planning, KQL generation, and KQL repair prompts.
- Added a Django management command to build repeatable seed documents for the planned `sap-sales-knowledge-v1` Azure AI Search index.
- Added `AZURE_SEARCH_RAG_ENABLED` as a feature flag. It defaults to disabled so the current hardcoded prompt path stays active unless Azure RAG is intentionally enabled.
- Preserved existing ADX execution, KQL validation, access-scope enforcement, response formatting, and API response shapes.

### Features Implemented
- Generated business/context documents from:
  - `FIELD_MAPPINGS`
  - `KUSTO_SCHEMA`
  - `GSBER_MAPPING`
  - `VTWEG_MAPPING`
  - curated business rules and KQL patterns from `SYSTEM_PROMPT_KQL`
- Added seed documents for field mappings, ADX schema, column definitions, depot/business-area mappings, distribution-channel mappings, KPI definitions, date rules, data-type rules, and approved KQL patterns.
- Added Azure OpenAI embedding generation support for `contentVector` using `AZURE_OPENAI_EMBED_DEPLOYMENT`.
- Added Azure AI Search upload support using `merge_or_upload_documents`.
- Added dry-run and JSON export support so the generated documents can be reviewed before upload.
- Generated a local seed document export at `docs/sap_sales_knowledge_seed_docs.json`.
- Added hybrid retrieval logic that tries vector + semantic search first, then semantic keyword search, then simple keyword search.
- Added advisory prompt rendering for retrieved business context.
- Added RAG status events for SSE clients: retrieved, skipped, unavailable, or no external context found.

### Refactoring Performed
- Inserted RAG retrieval as a separate graph node instead of mixing Azure Search calls into the existing agent execution function.
- Kept RAG context advisory only; `SYSTEM_PROMPT_KQL`, schema rules, current date context, `USER_AREA_SCOPE`, and validator rules remain authoritative if there is any conflict.
- Kept Azure Search calls fail-open so missing configuration, index errors, search failures, or empty results do not break the chat flow.
- Dealer/customer alias indexing was intentionally excluded because dealer cardinality is greater than 350,000 and should remain queryable through ADX, not mirrored into Azure AI Search.

### Files Added
- `src/agent/rag/__init__.py`
- `src/agent/rag/knowledge_seed.py`
- `src/agent/rag/indexing.py`
- `src/agent/rag/retriever.py`
- `src/agent/management/__init__.py`
- `src/agent/management/commands/__init__.py`
- `src/agent/management/commands/seed_sales_knowledge_index.py`
- `docs/sap_sales_knowledge_seed_docs.json`

### Files Modified
- `src/core/settings.py`
- `src/agent/agent.py`
- `src/agent/graph/state.py`
- `src/agent/graph/nodes.py`
- `src/agent/graph/workflow.py`
- `src/agent/planner/planner.py`
- `src/agent/planner/prompts.py`
- `src/agent/kql/repair.py`
- `src/agent/kql/prompts.py`
- `src/agent/tests.py`
- `docs/phase_history.md`

### Breaking Changes
None. Existing `api/sales/query/`, `api/sales/query/existing/<uuid>/`, and `api/sales/query/stream/` behavior remains unchanged.

### Migration Steps
- No database migration is required.
- No client/API migration is required.
- Azure AI Search RAG retrieval is feature-flagged off by default.
- When Azure AI Search is ready, confirm `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, and `AZURE_SEARCH_INDEX_NAME=sap-sales-knowledge-v1`, then run:
  `cd src; ..\venv\Scripts\python.exe manage.py seed_sales_knowledge_index --upload`
- Enable live RAG retrieval by setting:
  `AZURE_SEARCH_RAG_ENABLED=true`

### Testing Performed
- Ran `cd src; ..\venv\Scripts\python.exe manage.py test agent.tests`.
- Ran `cd src; ..\venv\Scripts\python.exe manage.py check`.
- Ran `cd src; ..\venv\Scripts\python.exe manage.py seed_sales_knowledge_index --sample-size 1`.
- Ran `cd src; ..\venv\Scripts\python.exe manage.py seed_sales_knowledge_index --sample-size 0 --output ..\docs\sap_sales_knowledge_seed_docs.json`.
- Attempted Azure AI Search upload with `seed_sales_knowledge_index --upload`; embedding generation succeeded, but Azure AI Search rejected the upload because the configured search service endpoint did not contain the `sap-sales-knowledge-v1` index at that time.
- Added tests for RAG context propagation through graph planning and agent execution.
- Added tests for planner prompt RAG context injection.
- Added tests for RAG context prompt formatting and fail-open graph behavior when retrieval fails.

### Known Issues or Limitations
- Azure AI Search retrieval is implemented but disabled by default through `AZURE_SEARCH_RAG_ENABLED=false`.
- The hardcoded `SYSTEM_PROMPT_KQL` remains the source of prompt truth and fallback for current KQL generation.
- The Azure Search seed command depends on the search service endpoint, admin key, index name, and embedding deployment being aligned.
- The seed documents intentionally do not include dealer/customer aliases or raw sales facts.
- Existing `FIELD_MAPPINGS` contains a duplicate Python key for `material code`; Python keeps the last value, so the generated seed reflects the current runtime mapping.
- The retriever is advisory; it improves business grounding but does not replace KQL validation, access-scope enforcement, or ADX execution feedback.

### Lessons Learned
- Azure AI Search should be used for low-cardinality business knowledge, schema context, mappings, date rules, KPI definitions, and approved KQL patterns.
- ADX should remain the source for high-cardinality dealer/customer lookup and all transactional sales analytics.
- A feature-gated, fail-open retriever is the safest Phase 4 boundary because it enables RAG without risking the existing production chat path.

### Next Phase
After Azure AI Search endpoint/index configuration is confirmed and `sap-sales-knowledge-v1` is seeded, enable `AZURE_SEARCH_RAG_ENABLED=true`, monitor retrieval quality, then tune document ranking, prompt context size, and RAG evaluation tests.
