# custom_admin/forms.py
from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from core.services.ad_service import ActiveDirectoryService

User = get_user_model()

class SalesAuthUserCreateFromADForm(forms.ModelForm):
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

        # Prefill the instance so save() just persists it
        self.instance.username   = username
        self.instance.email      = prof.email_address or ""
        self.instance.first_name = prof.first_name or ""
        self.instance.last_name  = prof.last_name or ""
        self._ad_profile = prof
        return cleaned

    def save(self, commit=True):
        # IMPORTANT: this attaches form.save_m2m for the admin
        user = super().save(commit=False)

        # No local password for AD users
        user.set_unusable_password()

        if commit:
            user.save()
            # safe even if there are no m2m fields; it will be a no-op
            if hasattr(self, "save_m2m"):
                self.save_m2m()
        return user
