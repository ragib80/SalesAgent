from django.db import models
import uuid
from core.models.base_model import AuditModel
from conversation.models.conversation import Conversation
from core.managers import ActiveManager


class Message(AuditModel):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    sender = models.CharField(max_length=255, choices=[
                              ('user', 'User'), ('bot', 'Bot')])
    # ai_model = models.ForeignKey(AIModel, on_delete=models.SET_NULL, null=True)
    text = models.TextField()
    image_url = models.TextField(null=True, blank=True)
    token_count = models.IntegerField(null=True, blank=True)
    response_time_ms = models.IntegerField(null=True, blank=True)
    is_deleted = models.BooleanField(default=False)  # <-- add this
    ai_model_response = models.TextField(null=True, blank=True)

    objects = models.Manager()         # Default manager
    active = ActiveManager()           # Custom manager

    def __str__(self):
        return f"{self.sender} ({self.conversation})"

    class Meta:
        db_table = 'message'  # table name
        verbose_name = "Message"
        verbose_name_plural = "Messages"
        # ordering = ['-created_at']
