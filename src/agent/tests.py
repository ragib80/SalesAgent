from unittest.mock import ANY, patch

from django.test import SimpleTestCase

from agent.agent import handle_user_query
from agent.graph.workflow import get_sales_analysis_workflow, run_sales_analysis_graph
from agent.kql.repair import repair_kql
from agent.kql.validator import validate_kql
from agent.planner.planner import build_query_plan
from agent.planner.schemas import QueryPlan
from agent.rag.indexing import add_content_vectors, build_embedding_text, merge_or_upload_in_batches
from agent.rag.knowledge_seed import build_sales_knowledge_documents
from agent.rag.retriever import RAGContext, RAGDocument, format_rag_context_for_prompt


class SalesAnalysisGraphTests(SimpleTestCase):
    def setUp(self):
        get_sales_analysis_workflow.cache_clear()

    @patch("agent.graph.nodes.retrieve_sales_knowledge_context")
    @patch("agent.graph.nodes.build_query_plan")
    @patch("agent.graph.nodes.handle_user_query")
    def test_graph_wraps_existing_agent_flow(
        self,
        mock_handle_user_query,
        mock_build_query_plan,
        mock_retrieve_sales_knowledge_context,
    ):
        mock_handle_user_query.return_value = "Sales analysis answer"
        mock_build_query_plan.return_value = QueryPlan(
            intent="ranking",
            metrics=["Revenue"],
            dimensions=["cname"],
        )
        mock_retrieve_sales_knowledge_context.return_value = RAGContext(
            query="show sales",
            documents=[RAGDocument(id="rule-sales", title="Sales means Revenue")],
            index_name="sap-sales-knowledge-v1",
        )

        state = run_sales_analysis_graph(
            "show sales",
            conversation_id="conversation-1",
            user="user-1",
        )

        self.assertEqual(state["result"], "Sales analysis answer")
        self.assertEqual(state["answer"], "Sales analysis answer")
        self.assertEqual(state["query_plan"]["intent"], "ranking")
        self.assertEqual(state["rag_context"]["documents"][0]["id"], "rule-sales")
        self.assertEqual(
            [event["event"] for event in state["events"]],
            ["status", "status", "status", "status", "final"],
        )
        mock_retrieve_sales_knowledge_context.assert_called_once_with("show sales")
        mock_build_query_plan.assert_called_once_with(
            "show sales",
            conversation_id="conversation-1",
            llm_client=ANY,
            rag_context=state["rag_context"],
        )
        mock_handle_user_query.assert_called_once_with(
            "show sales",
            conversation_id="conversation-1",
            user="user-1",
            query_plan=state["query_plan"],
            rag_context=state["rag_context"],
        )

    @patch("agent.graph.nodes.retrieve_sales_knowledge_context")
    @patch("agent.graph.nodes.build_query_plan")
    @patch("agent.graph.nodes.handle_user_query")
    def test_graph_captures_agent_errors_in_state(
        self,
        mock_handle_user_query,
        mock_build_query_plan,
        mock_retrieve_sales_knowledge_context,
    ):
        mock_build_query_plan.return_value = QueryPlan(intent="summary")
        mock_retrieve_sales_knowledge_context.return_value = RAGContext(query="show sales")
        mock_handle_user_query.side_effect = RuntimeError("ADX failed")

        with self.assertLogs("agent.graph.nodes", level="ERROR"):
            state = run_sales_analysis_graph("show sales")

        self.assertEqual(state["error"], "ADX failed")
        self.assertEqual(state["answer"], "")
        self.assertEqual(
            [event["event"] for event in state["events"]],
            ["status", "status", "status", "error", "final"],
        )

    @patch("agent.graph.nodes.retrieve_sales_knowledge_context")
    @patch("agent.graph.nodes.build_query_plan")
    @patch("agent.graph.nodes.handle_user_query")
    def test_graph_continues_when_planning_fails(
        self,
        mock_handle_user_query,
        mock_build_query_plan,
        mock_retrieve_sales_knowledge_context,
    ):
        mock_build_query_plan.side_effect = ValueError("invalid planner JSON")
        mock_handle_user_query.return_value = "Fallback answer"
        mock_retrieve_sales_knowledge_context.return_value = RAGContext(query="show sales")

        state = run_sales_analysis_graph("show sales")

        self.assertEqual(state["result"], "Fallback answer")
        self.assertEqual(state["query_plan_error"], "invalid planner JSON")
        mock_handle_user_query.assert_called_once_with(
            "show sales",
            conversation_id=None,
            user=None,
            query_plan=None,
            rag_context=None,
        )

    @patch("agent.graph.nodes.retrieve_sales_knowledge_context")
    @patch("agent.graph.nodes.build_query_plan")
    @patch("agent.graph.nodes.handle_user_query")
    def test_graph_continues_when_rag_retrieval_fails(
        self,
        mock_handle_user_query,
        mock_build_query_plan,
        mock_retrieve_sales_knowledge_context,
    ):
        mock_retrieve_sales_knowledge_context.side_effect = RuntimeError("search unavailable")
        mock_build_query_plan.return_value = QueryPlan(intent="summary")
        mock_handle_user_query.return_value = "Fallback answer"

        state = run_sales_analysis_graph("show sales")

        self.assertEqual(state["result"], "Fallback answer")
        self.assertEqual(state["rag_context_error"], "search unavailable")
        mock_build_query_plan.assert_called_once()
        mock_handle_user_query.assert_called_once_with(
            "show sales",
            conversation_id=None,
            user=None,
            query_plan=state["query_plan"],
            rag_context=None,
        )


class QueryPlannerTests(SimpleTestCase):
    def test_build_query_plan_normalizes_llm_json(self):
        class FakeLLM:
            def invoke(self, messages):
                return type(
                    "Response",
                    (),
                    {
                        "content": (
                            '{"intent":"ranking","metric":"Revenue",'
                            '"dimensions":["cname"],"limit":"10",'
                            '"confidence":0.8}'
                        )
                    },
                )()

        plan = build_query_plan("top 10 dealers by revenue", llm_client=FakeLLM())

        self.assertEqual(plan.intent, "ranking")
        self.assertEqual(plan.metrics, ["Revenue"])
        self.assertEqual(plan.dimensions, ["cname"])
        self.assertEqual(plan.limit, 10)
        self.assertEqual(plan.confidence, 0.8)

    def test_build_query_plan_includes_rag_context_when_available(self):
        class FakeLLM:
            def __init__(self):
                self.messages = []

            def invoke(self, messages):
                self.messages = messages
                return type(
                    "Response",
                    (),
                    {"content": '{"intent":"summary","metrics":["Revenue"]}'},
                )()

        fake_llm = FakeLLM()
        rag_context = RAGContext(
            query="show sales",
            documents=[
                RAGDocument(
                    id="rule-sales",
                    title="Sales means Revenue",
                    content="sales maps to Revenue",
                    sap_columns=["Revenue"],
                )
            ],
        ).to_dict()

        plan = build_query_plan(
            "show sales",
            llm_client=fake_llm,
            rag_context=rag_context,
        )

        self.assertEqual(plan.intent, "summary")
        self.assertIn("Sales means Revenue", fake_llm.messages[1]["content"])


class KQLValidationTests(SimpleTestCase):
    schema_types = {
        "bukrs": "long",
        "fkdat": "datetime",
        "Revenue": "real",
        "cname": "string",
    }

    def test_validator_accepts_known_safe_query(self):
        kql = """
SAPSalesInfos
| where bukrs == 1000
| summarize TotalRevenue = sum(Revenue) by cname
| top 10 by TotalRevenue desc;
""".strip()

        result = validate_kql(
            kql,
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
        )

        self.assertTrue(result.is_valid, result.as_repair_context())

    def test_validator_rejects_unknown_columns(self):
        kql = """
SAPSalesInfos
| where bukrs == 1000
| where FakeRevenue == 1
| take 10;
""".strip()

        result = validate_kql(
            kql,
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("unknown_column", {issue.code for issue in result.errors})

    def test_validator_blocks_management_commands(self):
        result = validate_kql(
            ".drop table SAPSalesInfos;",
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("blocked_operation", {issue.code for issue in result.errors})

    def test_validator_requires_bukrs_filter_on_table_scans(self):
        kql = """
SAPSalesInfos
| where Revenue > 0
| take 10;
""".strip()

        result = validate_kql(
            kql,
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("missing_bukrs_filter", {issue.code for issue in result.errors})


class KQLRepairTests(SimpleTestCase):
    schema_types = KQLValidationTests.schema_types

    def test_repair_kql_uses_llm_feedback_when_static_cleanup_is_not_enough(self):
        class FakeLLM:
            def invoke(self, messages):
                return type(
                    "Response",
                    (),
                    {
                        "content": (
                            "SAPSalesInfos\n"
                            "| where bukrs == 1000\n"
                            "| summarize TotalRevenue = sum(Revenue)\n"
                            "| take 10;"
                        )
                    },
                )()

        invalid_kql = "SAPSalesInfos\n| where MissingColumn == 1\n| take 10;"
        validation = validate_kql(
            invalid_kql,
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
        )

        repaired = repair_kql(
            invalid_kql,
            user_prompt="show revenue",
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
            validation_result=validation,
            llm_client=FakeLLM(),
            query_plan={"intent": "summary"},
        )

        repaired_validation = validate_kql(
            repaired,
            table_name="SAPSalesInfos",
            schema_types=self.schema_types,
        )
        self.assertTrue(repaired_validation.is_valid, repaired_validation.as_repair_context())


class KQLAgentIntegrationTests(SimpleTestCase):
    @patch("agent.agent.llm")
    @patch("agent.agent.repair_kql")
    @patch("agent.agent.adx")
    @patch("agent.agent.generate_kql")
    @patch("agent.agent.is_sales_analysis_query", return_value=True)
    def test_handle_user_query_repairs_invalid_kql_before_adx_execution(
        self,
        mock_is_sales_analysis_query,
        mock_generate_kql,
        mock_adx,
        mock_repair_kql,
        mock_llm,
    ):
        invalid_kql = "SAPSalesInfos\n| where MissingColumn == 1\n| take 10;"
        repaired_kql = (
            "SAPSalesInfos\n"
            "| where bukrs == 1000\n"
            "| summarize TotalRevenue = sum(Revenue)\n"
            "| take 10;"
        )
        mock_generate_kql.return_value = invalid_kql
        mock_repair_kql.return_value = repaired_kql
        mock_adx.return_value.run.return_value = (["TotalRevenue"], [[100.0]])
        mock_llm.invoke.return_value = type(
            "Response",
            (),
            {"content": "Summary answer"},
        )()

        answer = handle_user_query(
            "show revenue",
            query_plan={"intent": "summary", "metrics": ["Revenue"]},
        )

        self.assertEqual(answer, "Summary answer")
        mock_is_sales_analysis_query.assert_called_once()
        mock_repair_kql.assert_called_once()
        mock_adx.return_value.run.assert_called_once_with(repaired_kql)


class SalesKnowledgeSeedTests(SimpleTestCase):
    def test_seed_documents_include_business_context_sources(self):
        documents = build_sales_knowledge_documents()
        docs_by_id = {document["id"]: document for document in documents}

        self.assertIn("field-mappings-v1", docs_by_id)
        self.assertIn("adx-schema-sapsalesinfos-v1", docs_by_id)
        self.assertIn("column-revenue", docs_by_id)
        self.assertIn("column-gsber", docs_by_id)
        self.assertIn("gsber-dhaka-north-4130", docs_by_id)
        self.assertIn("vtweg-dealer-10", docs_by_id)
        self.assertIn("pattern-dropoff-leftanti", docs_by_id)
        self.assertIn("pattern-ytd-growth", docs_by_id)

        field_mapping = docs_by_id["field-mappings-v1"]
        self.assertIn("Revenue", field_mapping["sap_columns"])
        self.assertIn("cname", field_mapping["sap_columns"])
        self.assertIn("kunrg", field_mapping["sap_columns"])

    def test_seed_documents_do_not_index_dealer_aliases_or_sales_facts(self):
        documents = build_sales_knowledge_documents()
        doc_types = {document["doc_type"] for document in documents}
        ids = {document["id"] for document in documents}

        self.assertNotIn("dealer_alias", doc_types)
        self.assertNotIn("customer_alias", doc_types)
        self.assertFalse(any("dealer-alias" in doc_id for doc_id in ids))

        revenue_column = next(document for document in documents if document["id"] == "column-revenue")
        self.assertEqual(revenue_column["doc_type"], "column_definition")
        self.assertNotIn("contentVector", revenue_column)


class SalesKnowledgeIndexingTests(SimpleTestCase):
    def test_format_rag_context_for_prompt_renders_advisory_context(self):
        context = RAGContext(
            query="show sales",
            documents=[
                RAGDocument(
                    id="rule-sales",
                    title="Sales means Revenue",
                    content="Use Revenue for sales.",
                    sap_columns=["Revenue"],
                )
            ],
        )

        block = format_rag_context_for_prompt(context.to_dict())

        self.assertIn("RAG BUSINESS CONTEXT", block)
        self.assertIn("Sales means Revenue", block)
        self.assertIn("authoritative sources win", block)

    def test_build_embedding_text_uses_searchable_context_fields(self):
        text = build_embedding_text(
            {
                "title": "Title",
                "summary": "Summary",
                "content": "Content",
                "aliases": ["sales"],
                "keywords": ["Revenue"],
                "sap_columns": ["Revenue"],
                "kpi_names": ["sales"],
                "intent_tags": ["ranking"],
            }
        )

        self.assertIn("Title", text)
        self.assertIn("Content", text)
        self.assertIn("Revenue", text)
        self.assertIn("ranking", text)

    def test_add_content_vectors_returns_copies_with_embeddings(self):
        class FakeEmbeddings:
            def __init__(self):
                self.calls = []

            def create(self, model, input):
                self.calls.append((model, input))
                return type(
                    "Response",
                    (),
                    {
                        "data": [
                            type(
                                "Embedding",
                                (),
                                {"embedding": [0.1, 0.2, 0.3]},
                            )()
                        ]
                    },
                )()

        class FakeOpenAIClient:
            def __init__(self):
                self.embeddings = FakeEmbeddings()

        documents = [{"id": "doc-1", "title": "Revenue", "content": "Sales means Revenue."}]
        client = FakeOpenAIClient()

        enriched = add_content_vectors(
            documents,
            openai_client=client,
            embedding_deployment="embedding-deployment",
        )

        self.assertNotIn("contentVector", documents[0])
        self.assertEqual(enriched[0]["contentVector"], [0.1, 0.2, 0.3])
        self.assertEqual(client.embeddings.calls[0][0], "embedding-deployment")

    def test_merge_or_upload_in_batches_uses_search_client_batching(self):
        class FakeSearchClient:
            def __init__(self):
                self.batches = []

            def merge_or_upload_documents(self, documents):
                self.batches.append([document["id"] for document in documents])
                return [
                    type(
                        "IndexingResult",
                        (),
                        {"key": document["id"], "succeeded": True, "error_message": ""},
                    )()
                    for document in documents
                ]

        search_client = FakeSearchClient()
        documents = [{"id": f"doc-{index}"} for index in range(5)]

        results = merge_or_upload_in_batches(search_client, documents, batch_size=2)

        self.assertEqual(search_client.batches, [["doc-0", "doc-1"], ["doc-2", "doc-3"], ["doc-4"]])
        self.assertEqual(len(results), 5)
