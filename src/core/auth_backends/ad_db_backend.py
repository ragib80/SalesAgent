from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model
from django.db.models import Q
from django.conf import settings
from django.db import transaction
from core.services.ad_service import ActiveDirectoryService

class ADDBBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        if not username:
            return None

        User = get_user_model()
        try:
            user = User.objects.get(Q(username__iexact=username) | Q(email__iexact=username))
        except User.DoesNotExist:
            return None  # pre-provisioned only

        if not getattr(user, "is_active", True):
            return None

        # DEV MODE: bypass AD entirely
        if settings.DEBUG and getattr(settings, "AUTH_DEV_BYPASS_AD", True):
            return user

        # PRODUCTION (or dev with bypass off): require AD password
        if not password:
            return None

        ad = ActiveDirectoryService()
        if ad.authenticate_user(user.username, password):
            # Sync a few fields from AD on successful login
            prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
            changed = False
            if prof:
                if user.first_name != (prof.first_name or ""):
                    user.first_name = prof.first_name or ""; changed = True
                if user.last_name != (prof.last_name or ""):
                    user.last_name = prof.last_name or ""; changed = True
                if prof.email_address and user.email != prof.email_address:
                    user.email = prof.email_address; changed = True
            if changed:
                with transaction.atomic():
                    user.save(update_fields=["first_name", "last_name", "email"])
            return user

        # Optional: allow local password fallback (e.g., for legacy superusers)
        if getattr(settings, "AUTH_ALLOW_LOCAL_PASSWORD_FALLBACK", True) and user.has_usable_password():
            if user.check_password(password):
                return user

        return None

    def get_user(self, user_id):
        User = get_user_model()
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
