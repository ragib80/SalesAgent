from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from agent.models import AgentQueryAudit


@admin.register(AgentQueryAudit)
class AgentQueryAuditAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "get_username",
        "conversation_link",
        "success",
        "is_sales_query",
        "kql_validation_status",
        "total_latency_ms",
        "adx_row_count",
        "total_tokens",
        "estimated_cost_usd",
        "error_code",
    )
    list_filter = (
        "success",
        "is_sales_query",
        "kql_validation_status",
        "error_code",
        "created_at",
    )
    search_fields = (
        "user__username",
        "user__email",
        "user_prompt",
        "generated_kql",
        "error_message",
    )
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    # Show custom methods instead of raw user FK and conversation UUID
    readonly_fields = (
        "id",
        "get_username",
        "conversation_link",
        "user_prompt",
        "generated_kql",
        "prompt_version",
        "model_name",
        "query_plan",
        "rag_docs_retrieved",
        "is_sales_query",
        "kql_validation_status",
        "total_latency_ms",
        "kql_generation_latency_ms",
        "adx_execution_latency_ms",
        "llm_summary_latency_ms",
        "summary_prompt_tokens",
        "summary_completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "adx_row_count",
        "success",
        "error_code",
        "error_message",
        "created_at",
    )

    # Explicit fieldsets so the detail page shows our custom fields
    fieldsets = (
        ("Request", {
            "fields": ("id", "get_username", "conversation_link", "user_prompt", "created_at"),
        }),
        ("KQL & Planning", {
            "fields": ("prompt_version", "model_name", "query_plan", "rag_docs_retrieved",
                       "is_sales_query", "generated_kql", "kql_validation_status"),
        }),
        ("Latency (ms)", {
            "fields": ("total_latency_ms", "kql_generation_latency_ms",
                       "adx_execution_latency_ms", "llm_summary_latency_ms"),
        }),
        ("Tokens & Cost", {
            "fields": ("summary_prompt_tokens", "summary_completion_tokens",
                       "total_tokens", "estimated_cost_usd"),
        }),
        ("Outcome", {
            "fields": ("success", "adx_row_count", "error_code", "error_message"),
        }),
    )

    @admin.display(description="User", ordering="user__username")
    def get_username(self, obj):
        if obj.user:
            return obj.user.username
        return "—"

    @admin.display(description="Conversation")
    def conversation_link(self, obj):
        if not obj.conversation_id:
            return "—"
        try:
            url = (
                reverse("admin:conversation_conversation_changelist")
                + f"?uuid={obj.conversation_id}"
            )
            return format_html('<a href="{}">{}</a>', url, obj.conversation_id)
        except Exception:
            return str(obj.conversation_id)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
