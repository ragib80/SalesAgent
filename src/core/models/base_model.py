from django.db import models
from django.utils.timezone import now
from core.middleware.current_user import get_current_user


class AuditModel(models.Model):
    created_at = models.DateTimeField(default=now, editable=False)
    created_by = models.UUIDField(null=True, editable=False)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.UUIDField(null=True)

    def save(self, *args, **kwargs):
        user = get_current_user()
        if user and hasattr(user, 'uuid '):  # your UUID field
            if not self.pk and not self.created_by:
                self.created_by = user.uuid
            self.updated_by = user.uuid
        super().save(*args, **kwargs)

    class Meta:
        abstract = True
