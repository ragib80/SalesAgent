import uuid

from django.conf import settings
from django.db import models


class ValidationStatus(models.TextChoices):
    VALID = "valid", "Valid on first try"
    REPAIRED = "repaired", "Repaired before ADX execution"
    ADX_REPAIRED = "adx_repaired", "Repaired after ADX error"
    FAILED = "failed", "Could not produce valid KQL"


class ErrorCode(models.TextChoices):
    KQL_GENERATION_FAILED = "KQL_GEN_FAIL", "KQL generation failed"
    KQL_VALIDATION_FAILED = "KQL_VAL_FAIL", "KQL validation/repair failed"
    ADX_EXECUTION_FAILED = "ADX_EXEC_FAIL", "ADX execution failed after retries"
    NO_DATA = "NO_DATA", "No rows returned from ADX"
    LLM_SUMMARY_FAILED = "LLM_SUM_FAIL", "LLM summarization failed"
    ACCESS_DENIED = "ACCESS_DENIED", "User access-scope violation"
    GENERAL_QUERY = "GENERAL_QUERY", "Non-sales query — no KQL generated"


class AgentQueryAudit(models.Model):
    """
    One record per chat request processed by handle_user_query().

    Captures latency, token usage, generated KQL, and outcome so the
    operations team can monitor quality, cost, and error rates.

    Token counts cover the final summary LLM call only (the dominant cost).
    KQL-generation and classifier tokens are not individually tracked here.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="query_audits",
    )
    conversation_id = models.UUIDField(null=True, blank=True, db_index=True)
    user_prompt = models.TextField()
    generated_kql = models.TextField(blank=True)

    # ── Versioning ────────────────────────────────────────────────────────────
    prompt_version = models.CharField(max_length=32, blank=True)
    model_name = models.CharField(max_length=64, blank=True)

    # ── Planning / RAG ───────────────────────────────────────────────────────
    query_plan = models.JSONField(null=True, blank=True)
    rag_docs_retrieved = models.PositiveSmallIntegerField(default=0)

    # ── Classification & KQL status ──────────────────────────────────────────
    is_sales_query = models.BooleanField(null=True)
    kql_validation_status = models.CharField(
        max_length=16, choices=ValidationStatus.choices, blank=True
    )

    # ── Latency breakdown (milliseconds) ─────────────────────────────────────
    total_latency_ms = models.FloatField(null=True, blank=True)
    kql_generation_latency_ms = models.FloatField(null=True, blank=True)
    adx_execution_latency_ms = models.FloatField(null=True, blank=True)
    llm_summary_latency_ms = models.FloatField(null=True, blank=True)

    # ── Token usage (summary call only — see class docstring) ─────────────────
    summary_prompt_tokens = models.PositiveIntegerField(default=0)
    summary_completion_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)
    estimated_cost_usd = models.DecimalField(
        max_digits=10, decimal_places=6, null=True, blank=True
    )

    # ── ADX result ───────────────────────────────────────────────────────────
    adx_row_count = models.PositiveIntegerField(null=True, blank=True)

    # ── Outcome ───────────────────────────────────────────────────────────────
    success = models.BooleanField(default=False)
    error_code = models.CharField(max_length=20, choices=ErrorCode.choices, blank=True)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "agent_query_audit"
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["success", "created_at"]),
            models.Index(fields=["created_at"]),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error_code})"
        return f"Audit {self.id} | {status} | {self.created_at:%Y-%m-%d %H:%M:%S}"
