from unittest.mock import patch
from types import SimpleNamespace
from uuid import uuid4

from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sales_analyzer.views import ChatAPIView


class ChatAPIViewGraphIntegrationTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.user = SimpleNamespace(is_authenticated=True, uuid=uuid4())

    @patch.object(ChatAPIView, "_create_message")
    @patch.object(ChatAPIView, "_create_new_conversation")
    @patch("sales_analyzer.views.run_sales_analysis_graph")
    def test_chat_api_uses_graph_and_preserves_response_shape(
        self,
        mock_run_graph,
        mock_create_new_conversation,
        mock_create_message,
    ):
        conversation_uuid = uuid4()
        mock_create_new_conversation.return_value = SimpleNamespace(uuid=conversation_uuid)
        mock_run_graph.return_value = {
            "result": "Graph answer",
            "answer": "Graph answer",
            "events": [{"event": "final", "message": "Analysis completed"}],
        }
        request = self.factory.post(
            "/api/sales/query/",
            {"prompt": "show sales"},
            format="json",
        )
        force_authenticate(request, user=self.user)

        response = ChatAPIView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["answer"], "Graph answer")
        self.assertIsNone(response.data["data"])
        self.assertEqual(response.data["uuid"], str(conversation_uuid))
        self.assertEqual(mock_create_message.call_count, 2)
        mock_run_graph.assert_called_once()
