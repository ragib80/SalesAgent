# custom_admin/forms.py
from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from django.conf import settings
from core.services.ad_service import ActiveDirectoryService

User = get_user_model()


class SalesAuthUserCreateFromADForm(forms.ModelForm):
    """
    Add-user form: single field (ad_identifier). Pulls attributes from AD,
    creates a local user with an unusable password.
    """
    ad_identifier = forms.CharField(
        label="AD username or email",
        help_text="sAMAccountName / userPrincipalName / email",
        required=True,
    )

    class Meta:
        model = User
        fields = ()  # hide model fields on the add page

    def clean(self):
        cleaned = super().clean()
        ident = (cleaned.get("ad_identifier") or "").strip()
        if not ident:
            raise ValidationError({"ad_identifier": "This field is required."})

        ad = ActiveDirectoryService()
        prof = ad.find_user(ident)
        if not prof:
            raise ValidationError({"ad_identifier": "No AD user found for that value."})

        username = (prof.login_name or ident.split("@")[0]).lower()
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError({"ad_identifier": f"User '{username}' already exists."})

        # Prefill instance for save()
        self.instance.username   = username
        self.instance.email      = prof.email_address or ""
        self.instance.first_name = prof.first_name or ""
        self.instance.last_name  = prof.last_name or ""
        self._ad_profile = prof
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_unusable_password()  # AD-only auth
        if commit:
            user.save()
            if hasattr(self, "save_m2m"):
                self.save_m2m()
        return user


class SalesAuthUserChangeForm(forms.ModelForm):
    """
    Change-user form: adds a 'Sync from AD now' checkbox.
    """
    sync_from_ad = forms.BooleanField(
        label="Sync from AD now",
        required=False,
        help_text="Pull first/last name and email from AD using username or email.",
    )

    class Meta:
        model = User
        fields = (
            "username", "email", "first_name", "last_name",
            "is_active", "is_staff", "is_superuser", "groups", "user_permissions",
        )

    def save(self, commit=True):
        user = super().save(commit=False)
        if self.cleaned_data.get("sync_from_ad"):
            try:
                ad = ActiveDirectoryService()
                prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
                if prof:
                    if prof.first_name and user.first_name != prof.first_name:
                        user.first_name = prof.first_name
                    if prof.last_name and user.last_name != prof.last_name:
                        user.last_name = prof.last_name
                    if prof.email_address and user.email != prof.email_address:
                        user.email = prof.email_address
            except Exception:
                # Don't block saving if AD is down
                pass
        if commit:
            user.save()
            self.save_m2m()
        return user




# # custom_admin/forms.py
# from django import forms
# from django.core.exceptions import ValidationError
# from django.contrib.auth import get_user_model
# from core.services.ad_service import ActiveDirectoryService

# User = get_user_model()

# class SalesAuthUserCreateFromADForm(forms.ModelForm):
#     ad_identifier = forms.CharField(
#         label="AD username or email",
#         help_text="sAMAccountName / userPrincipalName / email",
#         required=True,
#     )

#     class Meta:
#         model = User
#         fields = ()  # hide model fields on the add page

#     def clean(self):
#         cleaned = super().clean()
#         ident = (cleaned.get("ad_identifier") or "").strip()
#         if not ident:
#             raise ValidationError({"ad_identifier": "This field is required."})

#         ad = ActiveDirectoryService()
#         prof = ad.find_user(ident)
#         if not prof:
#             raise ValidationError({"ad_identifier": "No AD user found for that value."})

#         username = (prof.login_name or ident.split("@")[0]).lower()
#         if User.objects.filter(username__iexact=username).exists():
#             raise ValidationError({"ad_identifier": f"User '{username}' already exists."})

#         # Prefill the instance so save() just persists it
#         self.instance.username   = username
#         self.instance.email      = prof.email_address or ""
#         self.instance.first_name = prof.first_name or ""
#         self.instance.last_name  = prof.last_name or ""
#         self._ad_profile = prof
#         return cleaned

#     def save(self, commit=True):
#         # IMPORTANT: this attaches form.save_m2m for the admin
#         user = super().save(commit=False)

#         # No local password for AD users
#         user.set_unusable_password()

#         if commit:
#             user.save()
#             # safe even if there are no m2m fields; it will be a no-op
#             if hasattr(self, "save_m2m"):
#                 self.save_m2m()
#         return user
