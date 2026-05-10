import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models
from core.models.base_model import AuditModel


class SalesAuthUser(AbstractUser, AuditModel):
    IDENTITY_PROVIDER_LOCAL = "local"
    IDENTITY_PROVIDER_MICROSOFT = "microsoft"
    IDENTITY_PROVIDER_LEGACY_AD = "legacy_ad"

    IDENTITY_PROVIDER_CHOICES = (
        (IDENTITY_PROVIDER_LOCAL, "Local"),
        (IDENTITY_PROVIDER_MICROSOFT, "Microsoft Entra ID"),
        (IDENTITY_PROVIDER_LEGACY_AD, "Legacy Active Directory"),
    )

    uuid = models.UUIDField(
        default=uuid.uuid4, editable=False, unique=True, db_index=True)
    identity_provider = models.CharField(
        max_length=32,
        choices=IDENTITY_PROVIDER_CHOICES,
        default=IDENTITY_PROVIDER_LOCAL,
    )
    azure_ad_tenant_id = models.CharField(
        max_length=64,
        blank=True,
        null=True,
    )
    azure_ad_object_id = models.CharField(
        max_length=64,
        blank=True,
        null=True,
    )
    last_microsoft_login = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"SalesAuthUser ({self.username} : {self.uuid})"

    class Meta:
        db_table = 'sales_auth_user'
        verbose_name = "SalesAuthUser"
        verbose_name_plural = "SalesAuthUsers"
        ordering = ['-created_at']
        indexes = [
            models.Index(
                fields=["azure_ad_tenant_id", "azure_ad_object_id"],
                name="sales_auth_ms_tid_oid_idx",
            ),
        ]
