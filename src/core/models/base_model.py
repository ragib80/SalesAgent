# core/models/audit.py
import uuid as _uuid
from django.db import models
from django.utils.timezone import now
from core.middleware.current_user import get_current_user, get_current_chat_user

class AuditModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, editable=False)
    created_by = models.UUIDField(null=True, editable=False)

    updated_at = models.DateTimeField(auto_now=True, editable=False)
    updated_by = models.UUIDField(null=True, editable=False)

    class Meta:
        abstract = True

    def _resolve_actor(self):
        """
        Prefer chat/JWT user (API) and fall back to admin/session user.
        Returns a user or None.
        """
        return get_current_chat_user() or get_current_user()

    def _extract_uuid(self, user):
        """
        Try user.uuid (custom UUID pk) else user.pk/id if they are UUID/str.
        Returns a uuid.UUID or None.
        """
        if user is None:
            return None
        # try common attributes in order
        for attr in ("uuid", "pk", "id"):
            val = getattr(user, attr, None)
            if val is None:
                continue
            if isinstance(val, _uuid.UUID):
                return val
            try:
                # allow string UUIDs
                return _uuid.UUID(str(val))
            except Exception:
                continue
        return None

    def save(self, *args, **kwargs):
        actor = self._resolve_actor()
        actor_uuid = self._extract_uuid(actor)

        # set created_by once
        if not self.pk and self.created_by is None and actor_uuid:
            self.created_by = actor_uuid
        # always set updated_by when we have an actor
        if actor_uuid:
            self.updated_by = actor_uuid

        super().save(*args, **kwargs)


# from django.db import models
# from django.utils.timezone import now
# from core.middleware.current_user import get_current_user


# class AuditModel(models.Model):
#     created_at = models.DateTimeField(default=now, editable=False)
#     created_by = models.UUIDField(null=True, editable=False)
#     updated_at = models.DateTimeField(auto_now=True)
#     updated_by = models.UUIDField(null=True)

#     def save(self, *args, **kwargs):
#         user = get_current_user()
#         if user and hasattr(user, 'uuid '):  # your UUID field
#             if not self.pk and not self.created_by:
#                 self.created_by = user.uuid
#             self.updated_by = user.uuid
#         super().save(*args, **kwargs)

#     class Meta:
#         abstract = True
