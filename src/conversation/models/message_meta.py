from django.db import models
import uuid
from core.models.base_model import AuditModel
from conversation.models.conversation import Conversation
from conversation.models.message import Message
from custom_admin.models.sales_auth_user import SalesAuthUser
from core.managers import ActiveManager

class MessageMeta(AuditModel):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    message      = models.ForeignKey(Message, on_delete=models.CASCADE)
    role         = models.CharField(max_length=16, choices=[('user','User'),('bot','Bot')])
    meta_json    = models.JSONField()     # store the exact META line you used


    class Meta:
        db_table = 'message_meta'
        indexes = [models.Index(fields=['conversation', 'created_at'])]
