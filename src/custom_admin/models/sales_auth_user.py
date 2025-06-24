import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models
from core.models.base_model import AuditModel


class SalesAuthUser(AbstractUser, AuditModel):
    uuid = models.UUIDField(
        default=uuid.uuid4, editable=False, unique=True, db_index=True)

    def __str__(self):
        return f"SalesAuthUser ({self.username} : {self.uuid})"

    class Meta:
        db_table = 'sales_auth_user'
        verbose_name = "SalesAuthUser"
        verbose_name_plural = "SalesAuthUsers"
        ordering = ['-created_at']
