import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="AgentQueryAudit",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="query_audits",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "conversation_id",
                    models.UUIDField(blank=True, db_index=True, null=True),
                ),
                ("user_prompt", models.TextField()),
                ("generated_kql", models.TextField(blank=True)),
                ("prompt_version", models.CharField(blank=True, max_length=32)),
                ("model_name", models.CharField(blank=True, max_length=64)),
                ("query_plan", models.JSONField(blank=True, null=True)),
                (
                    "rag_docs_retrieved",
                    models.PositiveSmallIntegerField(default=0),
                ),
                ("is_sales_query", models.BooleanField(null=True)),
                (
                    "kql_validation_status",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("valid", "Valid on first try"),
                            ("repaired", "Repaired before ADX execution"),
                            ("adx_repaired", "Repaired after ADX error"),
                            ("failed", "Could not produce valid KQL"),
                        ],
                        max_length=16,
                    ),
                ),
                ("total_latency_ms", models.FloatField(blank=True, null=True)),
                (
                    "kql_generation_latency_ms",
                    models.FloatField(blank=True, null=True),
                ),
                (
                    "adx_execution_latency_ms",
                    models.FloatField(blank=True, null=True),
                ),
                (
                    "llm_summary_latency_ms",
                    models.FloatField(blank=True, null=True),
                ),
                (
                    "summary_prompt_tokens",
                    models.PositiveIntegerField(default=0),
                ),
                (
                    "summary_completion_tokens",
                    models.PositiveIntegerField(default=0),
                ),
                ("total_tokens", models.PositiveIntegerField(default=0)),
                (
                    "estimated_cost_usd",
                    models.DecimalField(
                        blank=True,
                        decimal_places=6,
                        max_digits=10,
                        null=True,
                    ),
                ),
                ("adx_row_count", models.PositiveIntegerField(blank=True, null=True)),
                ("success", models.BooleanField(default=False)),
                (
                    "error_code",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("KQL_GEN_FAIL", "KQL generation failed"),
                            ("KQL_VAL_FAIL", "KQL validation/repair failed"),
                            ("ADX_EXEC_FAIL", "ADX execution failed after retries"),
                            ("NO_DATA", "No rows returned from ADX"),
                            ("LLM_SUM_FAIL", "LLM summarization failed"),
                            ("ACCESS_DENIED", "User access-scope violation"),
                            ("GENERAL_QUERY", "Non-sales query — no KQL generated"),
                        ],
                        max_length=20,
                    ),
                ),
                ("error_message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "agent_query_audit",
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="agentqueryaudit",
            index=models.Index(
                fields=["user", "created_at"],
                name="agent_query_audit_user_created_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="agentqueryaudit",
            index=models.Index(
                fields=["success", "created_at"],
                name="agent_query_audit_success_created_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="agentqueryaudit",
            index=models.Index(
                fields=["created_at"],
                name="agent_query_audit_created_idx",
            ),
        ),
    ]
