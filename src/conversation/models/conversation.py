from django.db import models
import uuid
from core.models.base_model import AuditModel

from custom_admin.models.sales_auth_user import SalesAuthUser
from core.managers import ActiveManager


class Conversation(AuditModel):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(SalesAuthUser, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, null=True)
    last_message_at = models.DateTimeField(null=True)
    pinned = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)
    is_archive = models.BooleanField(default=False)

    objects = models.Manager()         # Default manager (includes deleted)
    active = ActiveManager()           # Custom manager (only active)

    def delete(self, using=None, keep_parents=False):
        # Soft delete this conversation
        self.is_deleted = True
        self.save()

        # Soft delete all related messages
        self.message_set.update(is_deleted=True)

    def __str__(self):
        return f"{self.user} ({self.title})"

    class Meta:
        db_table = 'conversation'  # table name
        verbose_name = "Conversation"
        verbose_name_plural = "Conversations"
        ordering = ['-created_at']
