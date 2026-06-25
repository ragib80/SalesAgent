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

## Phase 4.1 - Azure AI Search RAG Activated

### Date
2026-06-14

### Objective
Seed the `sap-sales-knowledge-v1` Azure AI Search index with SAP sales business knowledge documents and enable live RAG retrieval for all chat requests.

### Actions Taken
- Created the `sap-sales-knowledge-v1` index in Azure AI Search using the schema defined in `docs/sap-sales-knowledge-v1.json`.
- Ran `cd src; ..\venv\Scripts\python.exe manage.py seed_sales_knowledge_index --upload` successfully.
- Embedding generation via `text-embedding-ada-002` succeeded for all documents.
- Azure AI Search accepted all 87 documents via `merge_or_upload_documents`.
- Set `AZURE_SEARCH_RAG_ENABLED=true` in `src/core/.env`.

### Result
- 87 knowledge documents are now live in the `sap-sales-knowledge-v1` index covering: field mappings, ADX schema, depot/gsber codes, distribution channel codes, KPI definitions, date rules, data type rules, and approved KQL patterns.
- RAG retrieval runs on every chat request before query planning, injecting relevant business context into both the planner and KQL generator prompts.
- The `retrieve_business_context_node` in the LangGraph workflow now actively retrieves context; SSE clients see `Business context retrieved (N documents)` status events.

### Known Issues or Limitations
- `SYSTEM_PROMPT_KQL` remains the authoritative fallback; RAG context is advisory only.
- Dealer/customer aliases are intentionally excluded from the index due to high cardinality (350,000+ records); ADX remains the source for those.

## Phase 5.1 - SSE Streaming Connected to Frontend Chat

### Date
2026-06-14

### Objective
Connect the existing backend SSE endpoint (`api/sales/query/stream/`) to the frontend chat UI so users see live graph progress events during request processing instead of a silent wait.

### Architecture Changes
- Replaced the `sendMessage()` function in `chat.js` with an async `fetch`-based SSE reader that POSTs to `api/sales/query/stream/` for all messages (new and existing conversations).
- The previous approach used jQuery `$.ajax` to `/sales/query/` and `/sales/query/existing/<uuid>/` and waited for the full blocking response.
- The typing indicator now shows a live status line updated by `event: status` SSE events.
- `event: final` renders the answer and updates the conversation ID.
- `event: error` renders the error message as a bot bubble.

### Files Modified
- `src/sales_analyzer/static/js/chat.js`

### Breaking Changes
None. The non-streaming endpoints (`/sales/query/` and `/sales/query/existing/<uuid>/`) are preserved and still used by any non-SSE clients.

### SSE Event → UI Mapping
| Server event | Frontend action |
|---|---|
| `status` | Updates grey status text under the typing dots |
| `final` | Removes typing indicator, renders answer bubble, saves conversation UUID |
| `error` | Removes typing indicator, renders error as bot bubble |

## Phase 5.2 - LLM Token Streaming with ChatGPT-Style Animation

### Date
2026-06-14

### Objective
Enable true token-by-token LLM streaming so the final narrative response appears word by word as it is generated, matching the ChatGPT/Claude user experience.

### Architecture Changes
- Added `on_token: Callable[[str], None] | None` parameter to `handle_user_query()` in `agent.py`.
- When `on_token` is provided, both the non-sales general response path and the sales narrative summarization path use `llm.stream()` instead of `llm.invoke()`, calling `on_token(chunk)` for each generated token.
- Added `on_token: Any` field to `SalesAgentState` in `graph/state.py`.
- `execute_existing_agent_node` in `graph/nodes.py` passes `state.get("on_token")` to `handle_user_query`.
- `stream_sales_analysis_graph` in `graph/workflow.py` accepts and passes `on_token` in the initial state.
- `ChatStreamAPIView` in `views.py` was refactored to use `threading.Thread` + `queue.Queue`: the graph runs in a background thread and puts both SSE graph events and LLM token chunks into the queue; the SSE generator reads from the queue and yields the appropriate events.
- Frontend `chat.js` handles `event: token` by accumulating `streamingText` and scheduling a `requestAnimationFrame`-throttled render (max 60 fps) with a blinking block cursor appended.
- On `event: final`, a final clean `marked.parse()` render is performed and the cursor is removed.
- Added `.stream-cursor` CSS animation and `.chunk-fade-in` CSS animation to `chat_index.html`.

### Full Streaming Flow
```
User prompt
  → status: "Analyzing request"
  → status: "Business context retrieved"
  → status: "Structured query plan created"
  → status: "Preparing response"
  → token: "**Top" → token: " 10" → token: " Dealers" → ...  (word by word)
  → final: {answer: "...full text...", uuid: "..."}
```

### Files Added
None.

### Files Modified
- `src/agent/agent.py`
- `src/agent/graph/state.py`
- `src/agent/graph/nodes.py`
- `src/agent/graph/workflow.py`
- `src/sales_analyzer/views.py`
- `src/sales_analyzer/static/js/chat.js`
- `src/sales_analyzer/templates/sales/chat_index.html`

### Breaking Changes
None. The non-streaming `ChatAPIView` and `ExistingConversationAPIView` still use `llm.invoke()` (no `on_token` passed) and are unaffected.

### Known Issues or Limitations
- `requestAnimationFrame` throttling renders at most ~60 times per second; if the LLM sends tokens faster, some are batched into one frame, which is the correct and desired behavior.
- Partial markdown during streaming (e.g. unclosed `**bold`) is handled gracefully by `marked.js`; the `final` event always does a clean re-render from the complete text.
- The background graph thread uses a 180-second queue timeout; long-running ADX queries that exceed this will surface as a timeout error event to the client.

### Next Phase
Phase 5.3 should add persistent user memory (cross-session preferences, frequently used filters, analytical context) using the existing MS SQL Server conversation infrastructure.

## Phase 6 - Production Hardening

### Date
2026-06-14

### Objective
Add query audit logging, prompt/model versioning, structured latency and token monitoring, and cost tracking to make the agent observable and production-ready without changing any query generation, ADX execution, access-scope, or API response behavior.

### Architecture Changes
- Added a new `agent.observability` package with a `Timer` context manager and a fail-safe `write_audit_record()` helper.
- Added `AgentQueryAudit` Django model (`agent_query_audit` table in MS SQL Server) that records one row per `handle_user_query()` call.
- Added a `PROMPT_VERSION` constant to `agent.py` so every audit record can be correlated with the prompt version that generated it.
- Instrumented `handle_user_query()` with per-phase latency timers (KQL generation, ADX execution, LLM summary) and token capture from the summary LLM response.
- Replaced all `print()` debug statements in `generate_kql()` and `handle_user_query()` with structured `logger.*()` calls.
- Added `agent` and `agent.observability` named loggers to the Django `LOGGING` config; level is configurable via `AGENT_LOG_LEVEL` env var (defaults to `INFO`).
- Registered `AgentQueryAudit` in Django admin as a read-only audit table with list filters, search, and date hierarchy.

### Features Implemented
- **Audit table** (`AgentQueryAudit`): captures user, conversation ID, user prompt, generated KQL, prompt version, model name, query plan, RAG docs retrieved, is_sales_query flag, KQL validation status, per-phase latency (ms), summary token counts, estimated cost (USD), ADX row count, success flag, error code, and error message.
- **Latency breakdown**: `kql_generation_latency_ms`, `adx_execution_latency_ms`, `llm_summary_latency_ms`, and `total_latency_ms` are recorded on every request including failed ones.
- **Token tracking**: `summary_prompt_tokens`, `summary_completion_tokens`, and `total_tokens` are captured from the final summary LLM call via `response_metadata["token_usage"]` (non-streaming only; streaming requests record 0).
- **Cost estimate**: `estimated_cost_usd` is computed from summary tokens using configurable rates (`OPENAI_COST_PER_1K_INPUT`, `OPENAI_COST_PER_1K_OUTPUT`; defaults to GPT-4o pricing).
- **Prompt versioning**: `PROMPT_VERSION = "kql-v3"` written to every audit row; increment this string whenever `SYSTEM_PROMPT_KQL` changes significantly.
- **Structured logging**: `INFO`-level success log on every completed query with all key metrics; `WARNING` on KQL/ADX failures; `DEBUG` for internal step events.
- **Fail-safe audit**: `write_audit_record()` is called in a `finally` block and swallows all exceptions — a database write failure never interrupts the user response.
- **Error classification**: `error_code` field uses named choices (`KQL_VAL_FAIL`, `ADX_EXEC_FAIL`, `NO_DATA`, `GENERAL_QUERY`, etc.) for dashboard filtering.
- **KQL validation status**: `kql_validation_status` tracks whether KQL was `valid`, `adx_repaired`, or `failed` on each request.
- **Django admin**: read-only `AgentQueryAuditAdmin` with `list_display`, `list_filter`, `search_fields`, and `date_hierarchy`; add/change permissions disabled.

### Files Added
- `src/agent/observability/__init__.py`
- `src/agent/observability/timing.py`
- `src/agent/observability/audit.py`
- `src/agent/migrations/0001_initial.py`

### Files Modified
- `src/agent/models.py` — `AgentQueryAudit` model with `ValidationStatus` and `ErrorCode` choices
- `src/agent/admin.py` — `AgentQueryAuditAdmin` registration
- `src/agent/agent.py` — `PROMPT_VERSION` constant, `import time`, print→logger replacements, `handle_user_query()` instrumented with `Timer` + `write_audit_record`
- `src/core/settings.py` — structured LOGGING config with `agent` logger and `AGENT_LOG_LEVEL` env var
- `docs/phase_history.md`

### Breaking Changes
None. All existing API response shapes, KQL generation behavior, ADX execution logic, access-scope enforcement, and SSE streaming contracts are unchanged.

### Migration Steps
Run `python manage.py migrate agent` to create the `agent_query_audit` table in MS SQL Server.
No other migration steps are required. Existing clients do not need to change.

### Testing Performed
- Ran `python -m compileall src/agent/models.py src/agent/admin.py src/agent/observability/ src/agent/migrations/0001_initial.py` — all files compile without errors.
- Ran `python -c "import ast; ast.parse(open('src/agent/agent.py', encoding='utf-8').read())"` — agent.py parses cleanly.
- Ran `python manage.py check` — no system check errors.

### Known Issues or Limitations
- Token counts cover the final summary LLM call only; KQL-generation and classifier token usage is not individually tracked (the dominant cost is the summary call).
- `estimated_cost_usd` uses GPT-4o pricing as the default; adjust `OPENAI_COST_PER_1K_INPUT` / `OPENAI_COST_PER_1K_OUTPUT` env vars if using a different deployment.
- Streaming requests (`on_token` provided) record `total_tokens = 0` because LangChain streaming does not surface `token_usage` in the same response metadata path.
- The audit table grows without bound; add a SQL Server retention job to purge rows older than 90 days in production.

### Next Phase
Phase 5.3 — Persistent user memory (cross-session preferences, frequently used filters) using the existing MS SQL Server conversation infrastructure.
