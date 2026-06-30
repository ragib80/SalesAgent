import json
import queue
import threading

from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sales_analyzer.serializers import QueryRequestSerializer, QueryResponseSerializer, ChatRequestSerializer, ChatResponseSerializer,FirstChatResponseSerializer
from agent.azure_clients import search_client, openai_client
from django.conf import settings
from django.views.generic import TemplateView
from agent.graph.workflow import run_sales_analysis_graph, stream_sales_analysis_graph
from conversation.models.conversation import Conversation
from conversation.models.message import Message
from datetime import datetime
import traceback
from core.middleware.current_user import set_current_chat_user, clear_current_chat_user
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

class ChatView(TemplateView):
    template_name = 'sales/chat_index.html'


# class ChatAPIView(APIView):
#     def post(self, request):
#         ser = ChatRequestSerializer(data=request.data)
#         ser.is_valid(raise_exception=True)
#         prompt = ser.validated_data['prompt']

#         result = sales_metrics_engine(prompt)
#         answer = generate_llm_answer(prompt, result)

#         out = {
#             'answer': answer,
#             'data': result.get('result'),
#             'operation_plan': result.get('operation_plan'),
#         }
#         response_ser = ChatResponseSerializer(out)
#         return Response(response_ser.data, status=status.HTTP_200_OK)



# imports: Conversation, Message, serializers, handle_user_query,
# set_current_chat_user, clear_current_chat_user

def _extract_answer(result):
    """
    Handle different shapes the agent might return on the first turn.
    """
    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        # common keys across first/next turns
        for k in ("answer", "final_answer", "message", "text", "content"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # last-ditch: stringify something meaningful
        if "result" in result and isinstance(result["result"], (str, int, float)):
            return str(result["result"])
        if "operation_plan" in result:
            return "I've prepared an operation plan; please see details."

    # really nothing
    return ""


def _sse_event(event_name, payload):
    data = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {data}\n\n"


def _result_from_graph_state(graph_state):
    if graph_state.get("error"):
        raise RuntimeError(graph_state["error"])
    return graph_state.get("result")

class ChatAPIView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        token = None
        conversation = None
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data["prompt"]

            # conversation
            conversation_id = request.data.get("conversation_id")
            conversation = (self._get_or_create_conversation(request.user, conversation_id)
                            if conversation_id
                            else self._create_new_conversation(request.user, title=self._make_chat_title(prompt)))

            # bind user context ONCE
            token = set_current_chat_user(request.user)

            # run graph-wrapped agent
            graph_state = run_sales_analysis_graph(
                prompt,
                conversation_id=str(conversation.uuid),
                user=request.user,
            )
            result = _result_from_graph_state(graph_state)

            # ---- extract / synthesize answer ----
            answer = self._extract_answer(result)
            if not answer.strip():
                # Build a minimal first-turn answer so a bubble is saved
                answer = self._fallback_answer_from_result(result, prompt) or \
                         "I prepared the results for you — see details below."
            if "Sorry, I couldn't process" in answer:
                answer = "sorry i am a baby now, day by day im learning from your prompt. for a better user experience."

            # persist messages
            self._create_message(conversation, "user", prompt)
            self._create_message(conversation, "bot", answer)

            out = {
                "answer": answer,
                "data": result.get("result") if isinstance(result, dict) else None,
                "operation_plan": result.get("operation_plan") if isinstance(result, dict) else None,
                "uuid": str(conversation.uuid),
            }
            return Response(FirstChatResponseSerializer(out).data, status=status.HTTP_200_OK)

        except Exception as e:
            print("[ChatAPIView] ERROR:", repr(e))
            out = {
                "answer": "sorry i am a baby now, day by day im learning from your prompt. for a better user experience.",
                "data": None,
                "operation_plan": None,
                "uuid": str(conversation.uuid) if conversation else None,
            }
            return Response(FirstChatResponseSerializer(out).data, status=status.HTTP_200_OK)
        finally:
            if token:
                clear_current_chat_user(token)

    # ---------- helpers ----------
    def _extract_answer(self, result):
        if isinstance(result, str):
            return (result or "").strip()
        if isinstance(result, dict):
            for k in ("answer", "final_answer", "message", "text", "content"):
                v = result.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    def _fallback_answer_from_result(self, result, prompt):
        """Build a small, readable first-turn bubble from structured data."""
        if not isinstance(result, dict):
            return ""

        data = result.get("result") or {}
        # Try common shapes your agent returns
        title = (data.get("title") or data.get("heading") or "Result").strip() if isinstance(data, dict) else "Result"

        # Try to detect period
        period = ""
        if isinstance(data, dict):
            period_info = data.get("period") or data.get("date_range") or {}
            start = period_info.get("start") or period_info.get("from")
            end = period_info.get("end") or period_info.get("to")
            if start and end:
                period = f" ({start} - {end})"

        # Try totals
        bullets = []
        totals = data.get("totals") if isinstance(data, dict) else None
        if isinstance(totals, dict):
            # Common keys: revenue / total_revenue / value / amount
            rev = totals.get("revenue") or totals.get("total_revenue") or totals.get("value") or totals.get("amount")
            cur = totals.get("currency") or "BDT"
            if rev is not None:
                bullets.append(f"**Total Revenue:** {rev} {cur}")

        # As a last resort, if no totals, but there is any numeric key
        if not bullets and isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)) and k.lower() in ("revenue", "sales", "total", "sum"):
                    bullets.append(f"**{k.capitalize()}:** {v}")
                    break

        # Compose markdown similar to your screenshot
        md = f"### {title}{period}\n"
        if bullets:
            md += "\n" + "\n".join([f"- {b}" for b in bullets])
        else:
            md += "\n_I've computed the results; expand the details in the panel below._"

        return md

    def _make_chat_title(self, prompt: str) -> str:
        t = prompt.strip()
        return (t[:55] + '…') if len(t) > 55 else t

    def _create_new_conversation(self, user, title=None):
        return Conversation.objects.create(
            user=user,
            title=title or f"Chat - {datetime.now():%Y-%m-%d %H:%M:%S}",
            is_deleted=False,
        )

    def _get_or_create_conversation(self, user, conversation_uuid):
        existing = Conversation.objects.filter(user=user, uuid=conversation_uuid, is_deleted=False).first()
        return existing or self._create_new_conversation(user)

    def _create_message(self, conversation, sender_role, text):
        return Message.objects.create(conversation=conversation, sender=sender_role, text=text, is_deleted=False)


class ChatStreamAPIView(ChatAPIView):
    """Stream LLM tokens and graph progress events via SSE."""

    def post(self, request):
        ser = ChatRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        prompt = ser.validated_data["prompt"]

        conversation_id = request.data.get("conversation_id")
        conversation = (
            self._get_or_create_conversation(request.user, conversation_id)
            if conversation_id
            else self._create_new_conversation(request.user, title=self._make_chat_title(prompt))
        )

        conv_uuid = str(conversation.uuid)
        q = queue.Queue()
        _DONE = object()

        def on_token(chunk):
            q.put(("token", chunk))

        def graph_runner():
            ctx_token = None
            try:
                ctx_token = set_current_chat_user(request.user)
                for event in stream_sales_analysis_graph(
                    prompt,
                    conversation_id=conv_uuid,
                    user=request.user,
                    on_token=on_token,
                ):
                    q.put(("event", event))
                q.put(("done", _DONE))
            except Exception as exc:
                q.put(("error", str(exc)))
            finally:
                if ctx_token:
                    clear_current_chat_user(ctx_token)

        threading.Thread(target=graph_runner, daemon=True).start()

        def event_stream():
            streamed_text = ""
            final_sent = False
            graph_failed = False

            while True:
                try:
                    kind, value = q.get(timeout=180)
                except queue.Empty:
                    if not final_sent:
                        yield _sse_event("error", {"message": "Request timed out.", "uuid": conv_uuid})
                    break

                if kind == "token":
                    streamed_text += value
                    yield _sse_event("token", {"chunk": value})

                elif kind == "event":
                    event_name = value.get("event", "status")
                    payload = {k: v for k, v in value.items() if k != "event"}

                    if event_name == "error":
                        graph_failed = True

                    if event_name == "final":
                        if graph_failed:
                            yield _sse_event("error", {
                                "message": "sorry i am a baby now, day by day im learning from your prompt. for a better user experience.",
                                "uuid": conv_uuid,
                            })
                            final_sent = True
                            continue

                        answer = streamed_text.strip() or payload.get("answer") or ""
                        if not answer.strip():
                            answer = "I prepared the results for you — see details below."
                        if "Sorry, I couldn't process" in answer:
                            answer = "sorry i am a baby now, day by day im learning from your prompt. for a better user experience."

                        self._create_message(conversation, "user", prompt)
                        bot_msg = self._create_message(conversation, "bot", answer)

                        # Save KQL server-side — never sent to the client
                        kql_to_store = payload.get("result_kql") or ""
                        if kql_to_store and hasattr(bot_msg, "kql"):
                            bot_msg.kql = kql_to_store
                            bot_msg.save(update_fields=["kql"])

                        # Emit data event (no KQL) before final so the frontend
                        # can store preview rows before the bubble is finalized
                        result_cols = payload.get("result_cols")
                        result_rows = payload.get("result_rows")
                        if result_cols and result_rows is not None:
                            yield _sse_event("data", {
                                "cols": result_cols,
                                "rows": result_rows[:100],
                                "total_rows": payload.get("result_total_rows", len(result_rows)),
                                "message_id": bot_msg.pk,
                            })

                        final_sent = True
                        yield _sse_event("final", {
                            "answer": answer,
                            "uuid": conv_uuid,
                            "message_id": bot_msg.pk,
                        })

                    else:
                        yield _sse_event(event_name, payload)

                elif kind == "done":
                    if not final_sent:
                        answer = streamed_text.strip() or "I prepared the results for you — see details below."
                        self._create_message(conversation, "user", prompt)
                        self._create_message(conversation, "bot", answer)
                        yield _sse_event("final", {"answer": answer, "uuid": conv_uuid})
                    break

                elif kind == "error":
                    if not final_sent:
                        yield _sse_event("error", {"message": value or "Something went wrong.", "uuid": conv_uuid})
                    break

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response


class ExistingConversationAPIView(APIView):
    def post(self, request, conversation_uuid):
        token = None
        try:
            ser = ChatRequestSerializer(data=request.data)
            ser.is_valid(raise_exception=True)
            prompt = ser.validated_data['prompt']

            conversation = self.get_or_create_conversation(request.user, conversation_uuid)

            token = set_current_chat_user(request.user)  
            # > Run graph-wrapped agent
            graph_state = run_sales_analysis_graph(
                prompt,
                conversation_id=str(conversation.uuid),
                user=request.user,
            )
            result = _result_from_graph_state(graph_state)

            answer = result if isinstance(result, str) else result.get("answer", "")
            if "Sorry, I couldn't process" in answer:
                answer = "sorry i am a baby now, day by day im learning from your prompt. for a better user experience."

            self.create_message(conversation, 'user', prompt)
            self.create_message(conversation, 'bot', answer)

            out = {
                'answer': answer,
                'data': result.get('result') if isinstance(result, dict) else None,
                'operation_plan': result.get('operation_plan') if isinstance(result, dict) else None,
            }
            response_ser = ChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)

        except Exception as e:
            out = {
                'answer': "sorry i am a baby now, day by day im learning from your prompt. for a better user experience.",
                'data': None,
                'operation_plan': None
            }
            response_ser = ChatResponseSerializer(out)
            return Response(response_ser.data, status=status.HTTP_200_OK)
        
        finally:
            if token:
                clear_current_chat_user(token)

    def get_or_create_conversation(self, user, conversation_uuid):
        existing_conversation = Conversation.objects.filter(
            user=user, uuid=conversation_uuid, is_deleted=False
        ).first()
        return existing_conversation if existing_conversation else self.create_new_conversation(user)

    def create_new_conversation(self, user):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conversation = Conversation.objects.create(
            user=user, title=f"Chat - {current_time}", is_deleted=False
        )
        return conversation

    def create_message(self, conversation, sender_role, message_content):
        Message.objects.create(
            conversation=conversation,
            sender=sender_role,
            text=message_content,
            is_deleted=False
        )


def _serialize_rows(cols, rows):
    """Convert ADX result rows to JSON-serializable lists."""
    import datetime as _dt
    result = []
    for row in rows:
        row_dict = dict(zip(cols, row))
        for k, v in row_dict.items():
            if isinstance(v, (_dt.datetime, _dt.date)):
                row_dict[k] = v.strftime("%Y-%m-%d")
        result.append(list(row_dict.values()))
    return result


class DataQueryAPIView(APIView):
    """
    Return paginated ADX rows for a bot message that has a stored KQL.
    The KQL is retrieved from Message.kql — it never crosses the network.

    GET /api/sales/data/?message_id=<pk>&page=1&page_size=50&mode=table&search=<text>
    GET /api/sales/data/?message_id=<pk>&mode=chart
    """

    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        import re as _re
        import datetime as _dt
        import logging as _logging
        from agent.agent import (
            _enforce_bukrs_filter,
            _rewrite_kql_for_export,
            _verify_kql_with_llm,
            adx,
            get_user_area_scope,
        )

        _log = _logging.getLogger(__name__)

        message_id = request.query_params.get("message_id", "").strip()
        mode       = request.query_params.get("mode", "table")  # "chart" | "table"
        try:
            page      = max(1, int(request.query_params.get("page", 1)))
            page_size = min(200, max(10, int(request.query_params.get("page_size", 50))))
        except (ValueError, TypeError):
            return Response({"error": "Invalid page or page_size."}, status=status.HTTP_400_BAD_REQUEST)
        search = (request.query_params.get("search") or "").strip()

        if not message_id:
            return Response({"error": "message_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Ownership check
        try:
            msg = Message.objects.get(pk=message_id, conversation__user=request.user, is_deleted=False)
        except (Message.DoesNotExist, ValueError):
            return Response({"error": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        kql = (msg.kql or "").strip()
        if not kql:
            return Response({"error": "No data available for this message."}, status=status.HTTP_404_NOT_FOUND)

        # 2. Verify bukrs == 1000 is present
        if "bukrs" not in kql.lower() or "1000" not in kql:
            return Response({"error": "KQL missing required company filter."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 3. Re-enforce bukrs as defense-in-depth (idempotent when already present)
        kql = _enforce_bukrs_filter(kql)

        # 4. Re-enforce user area scope
        try:
            scope = get_user_area_scope(request.user)
            if getattr(scope, "restricted", False):
                if not scope.depots:
                    return Response({"error": "No depot assigned to your account."}, status=status.HTTP_403_FORBIDDEN)
                if "gsber" not in kql.lower():
                    return Response({"error": "Scope enforcement failed."}, status=status.HTTP_403_FORBIDDEN)
        except Exception:
            pass  # fail-open for scope check errors

        # ── Chart mode ────────────────────────────────────────────────────────
        # Regex rewrites the trailing | top N (preserving 'by' clause) to | top 100.
        if mode == "chart":
            kql_chart = _rewrite_kql_for_export(kql, mode="chart")
            _log.debug("[DataQueryAPIView/chart] kql: %.400s", kql_chart)
            try:
                cols, rows = adx().run(kql_chart)
            except Exception as exc:
                _log.warning("[DataQueryAPIView/chart] ADX error — attempting LLM fix: %s", exc)
                kql_chart_fixed = _verify_kql_with_llm(kql_chart)
                if kql_chart_fixed != kql_chart:
                    try:
                        cols, rows = adx().run(kql_chart_fixed)
                    except Exception as exc2:
                        _log.error("[DataQueryAPIView/chart] ADX error after LLM fix: %s", exc2)
                        return Response({"error": str(exc2)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                else:
                    _log.error("[DataQueryAPIView/chart] ADX error, LLM produced no change: %s", exc)
                    return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            serialized = _serialize_rows(cols, rows)
            return Response({"cols": list(cols), "rows": serialized}, status=status.HTTP_200_OK)

        # ── Table mode ────────────────────────────────────────────────────────
        # Regex converts | top N by col dir → | order by col dir (preserves sort, drops limit).
        kql_base = _rewrite_kql_for_export(kql, mode="table_base")
        _log.debug("[DataQueryAPIView/table] kql_base: %.400s", kql_base)

        # Apply search filter
        if search:
            safe_search = _re.sub(r'["\']', '', search)[:100]
            kql_base += f'\n| where * has "{safe_search}"'

        # Column sort override from DataTables (sort_col = column name, sort_dir = asc|desc)
        sort_col = (request.query_params.get("sort_col") or "").strip()
        sort_dir = (request.query_params.get("sort_dir") or "desc").lower()
        if sort_dir not in ("asc", "desc"):
            sort_dir = "desc"
        if sort_col and _re.match(r'^[\w\[\]\. ]+$', sort_col):
            # Replace any existing | order by with the user-requested sort
            kql_base = _re.sub(r'\|\s*order\s+by\s+[^\n]+', '', kql_base, flags=_re.IGNORECASE).strip()
            kql_base += f'\n| order by {sort_col} {sort_dir}'

        # Count total rows on page 1 only
        total_rows = None
        total_pages = None
        if page == 1:
            try:
                _, count_rows = adx().run(kql_base + "\n| count")
                total_rows = int(count_rows[0][0]) if count_rows else 0
                total_pages = -(-total_rows // page_size)
            except Exception as exc:
                _log.warning("[DataQueryAPIView/table] count query failed: %s", exc)

        # Paginate — ADX does not support bare | skip N; use take-only for page 1
        # and row_number() windowing for subsequent pages.
        offset = (page - 1) * page_size
        if offset == 0:
            kql_paged = kql_base + f"\n| take {page_size}"
        else:
            kql_paged = (
                kql_base
                + f"\n| serialize rn = row_number()"
                + f"\n| where rn between ({offset + 1} .. {offset + page_size})"
                + f"\n| project-away rn"
            )
        _log.debug("[DataQueryAPIView/table] kql_paged: %.400s", kql_paged)

        try:
            cols, rows = adx().run(kql_paged)
        except Exception as exc:
            _log.warning("[DataQueryAPIView/table] ADX error — attempting LLM fix on kql_base: %s", exc)
            kql_base_fixed = _verify_kql_with_llm(kql_base)
            if kql_base_fixed != kql_base:
                if offset == 0:
                    kql_paged_fixed = kql_base_fixed + f"\n| take {page_size}"
                else:
                    kql_paged_fixed = (
                        kql_base_fixed
                        + f"\n| serialize rn = row_number()"
                        + f"\n| where rn between ({offset + 1} .. {offset + page_size})"
                        + f"\n| project-away rn"
                    )
                try:
                    cols, rows = adx().run(kql_paged_fixed)
                except Exception as exc2:
                    _log.error("[DataQueryAPIView/table] ADX error after LLM fix: %s", exc2)
                    return Response({"error": str(exc2)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                _log.error("[DataQueryAPIView/table] ADX error, LLM produced no change: %s", exc)
                return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        serialized = _serialize_rows(cols, rows)
        result = {"cols": list(cols), "rows": serialized, "page": page, "page_size": page_size}
        if total_rows is not None:
            result["total_rows"] = total_rows
            result["total_pages"] = total_pages
        return Response(result, status=status.HTTP_200_OK)
