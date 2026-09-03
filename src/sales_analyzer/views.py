import io
import json
import queue
import threading

from django.http import HttpResponse, StreamingHttpResponse
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

# Human-readable display labels for raw SAP/ADX column names.
# Lookup is case-insensitive (keys are lowercase); unrecognised columns pass through unchanged.
_COLUMN_DISPLAY_LABELS = {
    "cname": "Dealer Name",
    "kunrg": "Dealer Code",
    "fkdat": "Invoice Date",
    "fkimg": "Quantity",
    "volum": "Volume",
    "voleh": "Volume Unit",
    "wgbez": "Brand",
    "arktx": "Product Name",
    "matnr": "Product Code",
    "matkl": "Product Category",
    "gsber": "Business Area",
    "bukrs": "Company Code",
    "szone": "Sales Zone",
    "vkorg": "Sales Org",
    "vtweg": "Distribution Channel",
    "spart_text": "Division",
    "spart": "Division Code",
    "vkbur_c": "Sales Office",
    "vkgrp_c": "Sales Group",
    "kukla": "Dealer Group",
    "ktokd": "Account Group",
    "payer_dl": "Payer ID",
    "vbeln": "Invoice Number",
    "kkber": "Credit Control Area",
    "gk": "Business Group",
    "meins": "Unit of Measure",
    "kunnr_sh": "Ship-to Party",
    "posnr": "Line Item No.",
    "invoicecount": "Invoice Count",
    "totalrevenue": "Total Revenue",
    "py_revenue": "PY Revenue",
    "cy_revenue": "CY Revenue",
    "growth_pct": "Growth %",
    "growthpct": "Growth %",
    "py_quantity": "PY Quantity",
    "cy_quantity": "CY Quantity",
    "py_volume": "PY Volume",
    "cy_volume": "CY Volume",
}


def _apply_col_labels(cols):
    """Return human-readable display labels for a list of ADX column names."""
    return [_COLUMN_DISPLAY_LABELS.get(c.lower(), c) for c in cols]


# Column names (lowercase) hidden from all table/chart/export output.
# Filtering happens at the presentation layer only — KQL WHERE filters
# (e.g. `where vtweg == 10`) are unaffected; just the projected column is dropped.
_HIDDEN_COLUMNS = frozenset({"vtweg"})


def _strip_hidden_cols(cols, rows):
    """Drop hidden columns (e.g. vtweg) from ADX result cols and rows.

    Returns (filtered_cols, filtered_rows) with rows as lists. Filtering by
    column index keeps every row aligned with its columns.
    """
    keep = [i for i, c in enumerate(cols) if c.lower() not in _HIDDEN_COLUMNS]
    if len(keep) == len(cols):
        return list(cols), [list(r) for r in rows]
    filtered_cols = [cols[i] for i in keep]
    filtered_rows = [[row[i] for i in keep] for row in rows]
    return filtered_cols, filtered_rows


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
                            total_rows = payload.get("result_total_rows", len(result_rows))
                            result_cols, result_rows = _strip_hidden_cols(result_cols, result_rows)
                            yield _sse_event("data", {
                                "cols": result_cols,
                                "col_labels": _apply_col_labels(result_cols),
                                "rows": result_rows[:100],
                                "total_rows": total_rows,
                                "message_id": bot_msg.pk,
                                "chart_meta": payload.get("result_chart_meta"),
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

            col_list, rows = _strip_hidden_cols(cols, rows)
            serialized = _serialize_rows(col_list, rows)
            return Response({"cols": col_list, "col_labels": _apply_col_labels(col_list), "rows": serialized}, status=status.HTTP_200_OK)

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

        col_list, rows = _strip_hidden_cols(cols, rows)
        serialized = _serialize_rows(col_list, rows)
        result = {"cols": col_list, "col_labels": _apply_col_labels(col_list), "rows": serialized, "page": page, "page_size": page_size}
        if total_rows is not None:
            result["total_rows"] = total_rows
            result["total_pages"] = total_pages
        return Response(result, status=status.HTTP_200_OK)


class ExcelExportAPIView(APIView):
    """
    Two-sheet Excel export: Summary cover page + Sales Data table.
    Sheet 1 — Summary  : title banner, analysis query, KPI cards, optional chart.
    Sheet 2 — Sales Data: fully formatted Excel Table with conditional formatting.

    GET /api/sales/export-excel/?message_id=<pk>
    """

    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    # ── Column classification sets ──────────────────────────────────────────
    _REV  = frozenset(['revenue','amount','sales','totalrevenue','cy_revenue',
                       'py_revenue','periodb_revenue'])
    _QTY  = frozenset(['fkimg','quantity','qty','invoicecount','py_quantity',
                       'cy_quantity','totalquantity'])
    _VOL  = frozenset(['volum','volume','py_volume','cy_volume','totalvolume'])
    _GRW  = frozenset(['growth_pct','growthpct','growth'])
    _DT   = frozenset(['fkdat','invoicedate'])
    _PER  = frozenset(['period','timeperiod','timperiod','fiscalperiod'])
    _CODE = frozenset(['kunrg','gsber','bukrs','vkorg','vtweg','spart','matnr',
                       'matkl','kunnr_sh','payer_dl','vbeln','posnr','gk','kkber','szone'])

    def _col_type(self, name, sample):
        """Dynamically classify a column by its name and a small value sample."""
        n = name.lower()
        if n in self._DT  or 'date' in n:                               return 'date'
        if n in self._PER or 'period' in n or 'month' in n or 'quarter' in n: return 'period'
        if n in self._REV or 'revenue' in n or 'amount' in n:           return 'revenue'
        if n in self._QTY or 'quantity' in n:                           return 'quantity'
        if n in self._VOL or 'volume' in n:                             return 'volume'
        if n in self._GRW or 'growth' in n or 'pct' in n:              return 'growth'
        if n in self._CODE:                                              return 'code'
        non_null = [v for v in sample if v is not None]
        if non_null and all(isinstance(v, (int, float)) for v in non_null):
            return 'numeric'
        return 'text'

    def _detect_period(self, col_list, col_types, rows):
        """Return 'Apr 2024 – Mar 2026' style string, or None."""
        import datetime as _dt
        for i, col in enumerate(col_list):
            if col_types.get(col) not in ('date', 'period'):
                continue
            dates = []
            for row in rows:
                v = row[i]
                if isinstance(v, (_dt.datetime, _dt.date)):
                    dates.append(v if isinstance(v, _dt.datetime)
                                 else _dt.datetime(v.year, v.month, 1))
                elif isinstance(v, str) and v:
                    for fmt in ('%Y-%m-%d', '%Y-%m', '%b %Y', '%B %Y'):
                        try:
                            dates.append(_dt.datetime.strptime(v[:10].strip(), fmt))
                            break
                        except ValueError:
                            pass
            if not dates:
                continue
            mn, mx = min(dates), max(dates)
            if mn.year == mx.year and mn.month == mx.month:
                return mn.strftime('%b %Y')
            return f"{mn.strftime('%b %Y')} – {mx.strftime('%b %Y')}"
        return None

    def _maybe_add_growth(self, col_list, col_labels, rows, col_types):
        """Append a computed Growth % column when CY_Revenue + PY_Revenue are both present."""
        cy = next((i for i, c in enumerate(col_list) if 'cy_revenue' in c.lower()), None)
        py = next((i for i, c in enumerate(col_list) if 'py_revenue' in c.lower()), None)
        if cy is None or py is None:
            return col_list, col_labels, rows, col_types
        new_rows = []
        for row in rows:
            cv = row[cy] if isinstance(row[cy], (int, float)) else 0
            pv = row[py] if isinstance(row[py], (int, float)) else 0
            # Store as decimal so Excel % format (×100) renders correctly: 0.155 → 15.5%
            g = round((cv - pv) / abs(pv), 4) if pv else None
            new_rows.append(list(row) + [g])
        k = 'growth_pct_computed'
        t = dict(col_types)
        t[k] = 'growth'
        return list(col_list) + [k], list(col_labels) + ['Growth %'], new_rows, t

    def _build_charts_sheet(self, wb, col_list, col_labels, col_types, rows):
        """
        Create a 'Charts' sheet containing pre-aggregated data and charts.

        Two charts are generated when the data supports them:
          • Line chart  — total revenue aggregated by time period (trend)
          • Column chart — total revenue per entity, top-15 (comparison)

        Aggregating here (rather than referencing raw Sales Data rows) means
        the charts are meaningful regardless of how many rows the query returns.
        """
        from collections import defaultdict
        import datetime as _dt
        from openpyxl.chart import BarChart, LineChart, Reference
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        rev_cols  = [(i, col_labels[i]) for i, c in enumerate(col_list)
                     if col_types.get(c) == 'revenue']
        time_cols = [(i,)              for i, c in enumerate(col_list)
                     if col_types.get(c) in ('date', 'period')]
        text_cols = [(i, col_labels[i]) for i, c in enumerate(col_list)
                     if col_types.get(c) == 'text']

        if not rev_cols:
            return

        rv_i, rv_lbl = rev_cols[0]

        # ── Aggregate by time period ─────────────────────────────────────────
        time_data = None
        if time_cols:
            ti = time_cols[0][0]
            agg: dict = defaultdict(float)
            for row in rows:
                t, v = row[ti], row[rv_i]
                if t is not None and isinstance(v, (int, float)):
                    key = t.date() if isinstance(t, _dt.datetime) else t
                    agg[key] += v
            sorted_keys = sorted(agg.keys())
            if len(sorted_keys) > 1:
                time_data = [(k, round(agg[k], 2)) for k in sorted_keys]

        # ── Aggregate by entity (top 15) ─────────────────────────────────────
        entity_data = None
        if text_cols:
            te_i, te_lbl = text_cols[0]
            agg = defaultdict(float)
            for row in rows:
                e, v = row[te_i], row[rv_i]
                if e is not None and isinstance(v, (int, float)):
                    agg[str(e)] += v
            top15 = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:15]
            if top15:
                entity_data = (top15, te_lbl)

        if not time_data and not entity_data:
            return

        ws = wb.create_sheet("Charts")
        ws.sheet_view.showGridLines = False

        _NAVY = "1F3864"

        def _hdr_cell(ws, row, col, val):
            c = ws.cell(row=row, column=col, value=val)
            c.font = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
            c.fill = PatternFill(start_color=_NAVY, end_color=_NAVY, fill_type="solid")
            c.alignment = Alignment(horizontal="center", vertical="center")

        DATA_A, DATA_B = 1, 2   # data written in cols A–B
        CHART_COL     = 4       # charts start at col D
        r = 1                   # current data row pointer
        chart_row     = 1       # chart anchor row

        # ── Write time-series data + line chart ──────────────────────────────
        if time_data:
            _hdr_cell(ws, r, DATA_A, "Period")
            _hdr_cell(ws, r, DATA_B, rv_lbl)
            hdr_row = r;  r += 1
            for t, v in time_data:
                ws.cell(row=r, column=DATA_A, value=t).number_format = 'DD-MMM-YYYY'
                ws.cell(row=r, column=DATA_B, value=v).number_format  = '#,##0.00'
                r += 1
            end_row = r - 1

            data_ref = Reference(ws, min_col=DATA_B, max_col=DATA_B,
                                 min_row=hdr_row, max_row=end_row)
            cats_ref = Reference(ws, min_col=DATA_A, max_col=DATA_A,
                                 min_row=hdr_row + 1, max_row=end_row)
            ch = LineChart()
            ch.style  = 10
            ch.title  = f"{rv_lbl} — Trend by Period"
            ch.y_axis.title = rv_lbl
            ch.x_axis.title = "Period"
            ch.add_data(data_ref, titles_from_data=True)
            ch.set_categories(cats_ref)
            ch.width, ch.height = 22, 14
            ws.add_chart(ch, f"{get_column_letter(CHART_COL)}{chart_row}")
            chart_row += 24   # move anchor down for next chart
            r         += 2    # gap before entity block

        # ── Write entity data + column chart ─────────────────────────────────
        if entity_data:
            rows15, te_lbl = entity_data
            ent_start = r
            _hdr_cell(ws, r, DATA_A, te_lbl)
            _hdr_cell(ws, r, DATA_B, rv_lbl)
            r += 1
            for name, val in rows15:
                ws.cell(row=r, column=DATA_A, value=name)
                ws.cell(row=r, column=DATA_B, value=val).number_format = '#,##0.00'
                r += 1
            ent_end = r - 1

            data_ref = Reference(ws, min_col=DATA_B, max_col=DATA_B,
                                 min_row=ent_start, max_row=ent_end)
            cats_ref = Reference(ws, min_col=DATA_A, max_col=DATA_A,
                                 min_row=ent_start + 1, max_row=ent_end)
            ch = BarChart()
            ch.type  = "col"
            ch.style = 10
            ch.title = f"Top {len(rows15)} {te_lbl} by {rv_lbl}"
            ch.y_axis.title = rv_lbl
            ch.add_data(data_ref, titles_from_data=True)
            ch.set_categories(cats_ref)
            ch.width, ch.height = 22, 14
            ws.add_chart(ch, f"{get_column_letter(CHART_COL)}{chart_row}")

        # Column widths for the data area
        ws.column_dimensions[get_column_letter(DATA_A)].width = 26
        ws.column_dimensions[get_column_letter(DATA_B)].width = 18

    # ── Main request handler ────────────────────────────────────────────────
    def get(self, request):
        import datetime as _dt
        import decimal as _decimal
        import logging as _logging
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.table import Table, TableStyleInfo
        from openpyxl.formatting.rule import ColorScaleRule
        from agent.agent import (
            _enforce_bukrs_filter, _rewrite_kql_for_export,
            _verify_kql_with_llm, adx, get_user_area_scope,
        )
        _log = _logging.getLogger(__name__)

        # ── Auth / KQL security (unchanged) ──────────────────────────────────
        message_id = request.query_params.get("message_id", "").strip()
        if not message_id:
            return Response({"error": "message_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            msg = Message.objects.get(pk=message_id, conversation__user=request.user, is_deleted=False)
        except (Message.DoesNotExist, ValueError):
            return Response({"error": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        kql = (msg.kql or "").strip()
        if not kql:
            return Response({"error": "No data available for this message."}, status=status.HTTP_404_NOT_FOUND)

        if "bukrs" not in kql.lower() or "1000" not in kql:
            return Response({"error": "KQL missing required company filter."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        kql = _enforce_bukrs_filter(kql)

        try:
            scope = get_user_area_scope(request.user)
            if getattr(scope, "restricted", False) and not scope.depots:
                return Response({"error": "No depot assigned to your account."}, status=status.HTTP_403_FORBIDDEN)
        except Exception:
            pass

        user_msg = Message.objects.filter(
            conversation=msg.conversation, sender='user',
            is_deleted=False, pk__lt=msg.pk,
        ).order_by('-pk').first()
        prompt_text = (user_msg.text if user_msg else "") or "N/A"

        kql_export = _rewrite_kql_for_export(kql, mode="table_base")
        try:
            cols, rows = adx().run(kql_export)
        except Exception as exc:
            _log.warning("[ExcelExportAPIView] ADX error — attempting LLM fix: %s", exc)
            kql_fixed = _verify_kql_with_llm(kql_export)
            if kql_fixed != kql_export:
                try:
                    cols, rows = adx().run(kql_fixed)
                except Exception as exc2:
                    _log.error("[ExcelExportAPIView] ADX error after LLM fix: %s", exc2)
                    return Response({"error": "Failed to retrieve data for export."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                _log.error("[ExcelExportAPIView] ADX error, no LLM fix: %s", exc)
                return Response({"error": "Failed to retrieve data for export."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        col_list, rows = _strip_hidden_cols(cols, rows)
        col_labels = _apply_col_labels(col_list)
        n_rows     = len(rows)

        if n_rows == 0:
            return Response({"error": "No data to export."}, status=status.HTTP_404_NOT_FOUND)

        now = datetime.now()

        # ── Dynamic analysis ──────────────────────────────────────────────────
        col_types = {
            col: self._col_type(col, [rows[j][i] for j in range(min(15, n_rows))])
            for i, col in enumerate(col_list)
        }
        period_str = self._detect_period(col_list, col_types, rows)

        # Add computed Growth % column if CY + PY revenue are both present
        col_list, col_labels, rows, col_types = self._maybe_add_growth(
            col_list, col_labels, rows, col_types)
        n_cols = len(col_list)
        n_rows = len(rows)


        # ── Style helpers ─────────────────────────────────────────────────────
        _NAVY, _NAVY2, _GOLD = "1F3864", "2E4A7A", "C9A84C"
        _LBLUE, _WHITE = "DCE6F1", "FFFFFF"
        _CARD_COLS = ["1F3864", "2E75B6", "17375E"]

        def _fill(h):
            return PatternFill(start_color=h, end_color=h, fill_type="solid")

        def _fnt(bold=False, sz=11, color="000000", italic=False):
            return Font(bold=bold, size=sz, color=color, italic=italic, name="Calibri")

        def _aln(h="left", v="center", wrap=False, indent=0):
            return Alignment(horizontal=h, vertical=v, wrap_text=wrap, indent=indent)

        # ════════════════════════════════════════════════════════════════════
        # SHEET 1 — Summary
        # ════════════════════════════════════════════════════════════════════
        wb   = Workbook()
        SPAN = max(n_cols, 8)

        ws_sum = wb.active
        ws_sum.title = "Summary"
        ws_sum.sheet_view.showGridLines = False
        for ci in range(1, SPAN + 1):
            ws_sum.column_dimensions[get_column_letter(ci)].width = 14

        def _fill_row(row_num, hex_color, n=SPAN):
            for ci in range(1, n + 1):
                ws_sum.cell(row=row_num, column=ci).fill = _fill(hex_color)

        def _banner(row_num, hex_color, value, font, align, height):
            _fill_row(row_num, hex_color)
            c = ws_sum.cell(row=row_num, column=1)
            c.value, c.font, c.alignment = value, font, align
            ws_sum.row_dimensions[row_num].height = height
            ws_sum.merge_cells(start_row=row_num, start_column=1,
                               end_row=row_num, end_column=SPAN)

        r = 1
        # Top spacer
        _fill_row(r, _NAVY); ws_sum.row_dimensions[r].height = 6; r += 1
        # Main title
        _banner(r, _NAVY, "SALES DATA EXPORT",
                _fnt(bold=True, sz=22, color=_WHITE), _aln("center", "center"), 38); r += 1
        # Subtitle
        _banner(r, _NAVY, "AI Sales Analysis Report",
                _fnt(sz=13, color=_GOLD, italic=True), _aln("center", "center"), 22); r += 1
        # Bottom of title block
        _fill_row(r, _NAVY); ws_sum.row_dimensions[r].height = 6; r += 1
        # Section label
        _banner(r, _NAVY2, "  ANALYSIS QUERY",
                _fnt(bold=True, sz=9, color=_GOLD), _aln("left", "center"), 20); r += 1

        # Prompt row (wrapped, light-blue background)
        chars_per_row = max(60, SPAN * 9)
        p_lines = max(2, min(10, len(prompt_text) // chars_per_row + 2))
        _fill_row(r, _LBLUE)
        ws_sum.merge_cells(start_row=r, start_column=1, end_row=r, end_column=SPAN)
        c = ws_sum.cell(row=r, column=1)
        c.value = prompt_text
        c.font = _fnt(sz=11, color="1A1A2E", italic=True)
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1)
        ws_sum.row_dimensions[r].height = max(36, p_lines * 16); r += 1

        # Spacer
        ws_sum.row_dimensions[r].height = 10; r += 1

        # KPI cards — 3 cards across SPAN columns
        CARD_LBL, CARD_VAL = r, r + 1
        ws_sum.row_dimensions[CARD_LBL].height = 18
        ws_sum.row_dimensions[CARD_VAL].height = 32
        cw = SPAN // 3
        c_s = [1, cw + 1, cw * 2 + 1]
        c_e = [cw, cw * 2, SPAN]
        cards = [
            ("GENERATED AT",  now.strftime("%d %b %Y  %H:%M")),
            ("TOTAL RECORDS",  f"{n_rows:,}"),
            ("PERIOD",         period_str or "—"),
        ]
        for idx, (lbl, val) in enumerate(cards):
            cs, ce, cc = c_s[idx], c_e[idx], _CARD_COLS[idx]
            for ri2 in (CARD_LBL, CARD_VAL):
                for ci2 in range(cs, ce + 1):
                    ws_sum.cell(row=ri2, column=ci2).fill = _fill(cc)
            ws_sum.merge_cells(start_row=CARD_LBL, start_column=cs,
                               end_row=CARD_LBL, end_column=ce)
            c = ws_sum.cell(row=CARD_LBL, column=cs)
            c.value, c.font, c.alignment = (
                lbl, _fnt(bold=True, sz=8, color=_GOLD), _aln("center", "center"))
            ws_sum.merge_cells(start_row=CARD_VAL, start_column=cs,
                               end_row=CARD_VAL, end_column=ce)
            c = ws_sum.cell(row=CARD_VAL, column=cs)
            c.value = val
            c.font = _fnt(bold=True, sz=16 if len(val) <= 16 else 12, color=_WHITE)
            c.alignment = _aln("center", "center")
        r += 2

        # Spacer + pointer
        ws_sum.row_dimensions[r].height = 10; r += 1
        _banner(r, _WHITE,
                "\U0001f4ca  Full dataset is available on the 'Sales Data' sheet  →",
                _fnt(sz=11, color=_NAVY, italic=True), _aln("center", "center"), 22); r += 1
        ws_sum.row_dimensions[r].height = 8; r += 1

        # ════════════════════════════════════════════════════════════════════
        # SHEET 2 — Sales Data
        # ════════════════════════════════════════════════════════════════════
        ws_data = wb.create_sheet("Sales Data")
        ws_data.sheet_view.showGridLines = False

        HDR = 1; DATA_S = 2; DATA_E = 1 + n_rows

        # Header row
        ws_data.append(col_labels)
        ws_data.row_dimensions[HDR].height = 22
        for ci in range(1, n_cols + 1):
            c = ws_data.cell(row=HDR, column=ci)
            c.font = _fnt(bold=True, sz=10, color=_WHITE)
            c.fill = _fill(_NAVY)
            c.alignment = _aln("center", "center", wrap=True)

        # String address avoids creating a phantom empty row (known openpyxl pitfall)
        ws_data.freeze_panes = "A2"

        # Number format map (growth stored as decimal → Excel % format renders ×100)
        _FMT = {
            'date':     'DD-MMM-YYYY',
            'period':   'DD-MMM-YYYY',
            'revenue':  '#,##0.00',
            'quantity': '#,##0',
            'volume':   '#,##0.00',
            'numeric':  '#,##0.##',
            'growth':   '+0.00%;-0.00%;0.00%',
        }

        def _safe_cell(val):
            """Return a value that openpyxl can safely write to a cell."""
            if val is None:
                return ""
            if isinstance(val, _dt.datetime):
                v = val.replace(tzinfo=None)
                # If there's no time component (midnight), return a plain date so
                # Excel shows "01-Apr-2024" instead of "2024-04-01 0:00:00"
                if v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0:
                    return v.date()
                return v
            if isinstance(val, _dt.date):
                return val
            if isinstance(val, _dt.timedelta):
                # timedelta has no native Excel type — render as total hours string
                total_h = val.total_seconds() / 3600
                return f"{total_h:.2f}h"
            if isinstance(val, _decimal.Decimal):
                # Kusto decimal columns → convert to float for Excel numeric formatting
                return float(val)
            if isinstance(val, (bool, int, float, str)):
                return val
            # Fallback: stringify anything else so the workbook never errors
            return str(val)

        # Write data rows (fast batch append — no per-cell ops in this loop)
        for row in rows:
            ws_data.append([_safe_cell(v) for v in row])

        # Apply number / date formats per column after appending all rows
        for ci, col in enumerate(col_list, start=1):
            fmt = _FMT.get(col_types.get(col))
            if not fmt:
                continue
            for ri in range(DATA_S, DATA_E + 1):
                ws_data.cell(row=ri, column=ci).number_format = fmt

        # Excel Table — banded rows + auto-filter (freeze_panes is a string so no conflict)
        tbl = Table(
            displayName="SalesData",
            ref=f"A{HDR}:{get_column_letter(n_cols)}{DATA_E}",
        )
        tbl.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        ws_data.add_table(tbl)

        # Conditional formatting: color scale on revenue; RYG scale on growth
        for ci, col in enumerate(col_list, start=1):
            ctype = col_types.get(col)
            rng = f"{get_column_letter(ci)}{DATA_S}:{get_column_letter(ci)}{DATA_E}"
            if ctype == 'revenue' and n_rows > 1:
                ws_data.conditional_formatting.add(rng, ColorScaleRule(
                    start_type='min',        start_color='FFFFFFFF',
                    mid_type='percentile',   mid_value=50, mid_color='FFBDD7EE',
                    end_type='max',          end_color='FF1F3864',
                ))
            elif ctype == 'growth' and n_rows > 1:
                ws_data.conditional_formatting.add(rng, ColorScaleRule(
                    start_type='min',  start_color='FFF8696B',   # red  (low / negative)
                    mid_type='num',    mid_value=0, mid_color='FFFFFFEB',  # yellow (zero)
                    end_type='max',    end_color='FF63BE7B',      # green (high / positive)
                ))

        # Column widths — based on column type and sampled data values
        for ci, col in enumerate(col_list, start=1):
            ctype = col_types.get(col, 'text')
            label = col_labels[ci - 1]
            clet  = get_column_letter(ci)
            if ctype == 'date':
                w = 14
            elif ctype == 'period':
                samp = [str(rows[j][ci - 1]) for j in range(min(5, n_rows)) if rows[j][ci - 1]]
                w = max(len(label) + 2, max((len(v) for v in samp), default=10) + 2, 12)
            elif ctype in ('revenue', 'volume'):
                w = 16
            elif ctype in ('quantity', 'numeric', 'code'):
                w = 14
            elif ctype == 'growth':
                w = 12
            elif ctype == 'text':
                samp = [str(rows[j][ci - 1]) for j in range(min(20, n_rows)) if rows[j][ci - 1]]
                avg  = int(sum(len(v) for v in samp) / len(samp)) if samp else len(label)
                w    = max(len(label) + 2, min(avg + 4, 40))
            else:
                w = max(12, len(label) + 2)
            ws_data.column_dimensions[clet].width = w

        # ── Charts sheet (aggregated data → meaningful charts) ────────────────
        try:
            self._build_charts_sheet(wb, col_list, col_labels, col_types, rows)
        except Exception as e:
            _log.warning("[ExcelExportAPIView] Charts sheet skipped: %s", e)

        # ── Serialize ─────────────────────────────────────────────────────────
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        filename = f"sales_export_{now:%Y%m%d_%H%M%S}.xlsx"
        response = HttpResponse(
            buf.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response
