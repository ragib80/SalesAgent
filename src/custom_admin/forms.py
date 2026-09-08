# custom_admin/forms.py
import logging

from django.contrib.admin.widgets import FilteredSelectMultiple
from django import forms
from django.db import transaction
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from core.services.ad_service import ActiveDirectoryService
from core.services.microsoft_graph_service import (
    MicrosoftGraphDirectoryService,
    MicrosoftGraphServiceError,
)
from user_auth.models import (
    Depo,
    Zone,
    Territory,
    UserDepoMap,
    UserZoneMap,
    UserTerritoryMap,
)

User = get_user_model()
logger = logging.getLogger(__name__)


class SalesAuthUserCreateFromADForm(forms.ModelForm):
    """
    Add-user form: single field (ad_identifier). Pulls attributes from
    Microsoft Graph and creates a local user with an unusable password.
    """
    ad_identifier = forms.CharField(
        label="Microsoft username or email",
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

        try:
            graph = MicrosoftGraphDirectoryService()
            prof = graph.find_user(ident)
        except Exception as exc:
            logger.exception("Microsoft directory lookup failed for %s", ident)
            message = "Microsoft directory lookup is unavailable. Try again later."
            if isinstance(exc, ValidationError):
                raise
            if isinstance(exc, MicrosoftGraphServiceError):
                raise ValidationError({"ad_identifier": message}) from exc
            raise ValidationError({"ad_identifier": message}) from exc
        if not prof:
            raise ValidationError(
                {"ad_identifier": "No Microsoft directory user found for that value."}
            )

        username = (prof.login_name or ident.split("@")[0]).lower()
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError({"ad_identifier": f"User '{username}' already exists."})
        if prof.tenant_id and prof.object_id:
            existing = User.objects.filter(
                azure_ad_tenant_id=prof.tenant_id,
                azure_ad_object_id=prof.object_id,
            ).first()
            if existing:
                raise ValidationError(
                    {
                        "ad_identifier": (
                            f"Microsoft user is already linked to local user "
                            f"'{existing.username}'."
                        )
                    }
                )

        # Prefill instance for save()
        self.instance.username = username
        self.instance.email = prof.email_address or ""
        self.instance.first_name = prof.first_name or ""
        self.instance.last_name = prof.last_name or ""
        self.instance.identity_provider = getattr(
            User,
            "IDENTITY_PROVIDER_MICROSOFT",
            "microsoft",
        )
        self.instance.azure_ad_tenant_id = prof.tenant_id or ""
        self.instance.azure_ad_object_id = prof.object_id or ""
        self._graph_profile = prof
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_unusable_password()
        user.identity_provider = getattr(User, "IDENTITY_PROVIDER_MICROSOFT", "microsoft")
        if commit:
            user.save()
            if hasattr(self, "save_m2m"):
                self.save_m2m()
        return user


class SalesAuthUserChangeForm(forms.ModelForm):
    """
    Change-user form: adds a 'Sync from AD now' checkbox.
    """
    depos = forms.ModelMultipleChoiceField(
        label="Depos",
        queryset=Depo.objects.none(),
        required=False,
        widget=FilteredSelectMultiple("Depos", is_stacked=False),
    )
    zones = forms.ModelMultipleChoiceField(
        label="Zones",
        queryset=Zone.objects.none(),
        required=False,
        widget=FilteredSelectMultiple("Zones", is_stacked=False),
    )
    territories = forms.ModelMultipleChoiceField(
        label="Territories",
        queryset=Territory.objects.none(),
        required=False,
        widget=FilteredSelectMultiple("Territories", is_stacked=False),
    )
    sync_from_ad = forms.BooleanField(
        label="Sync from AD now",
        required=False,
        help_text="Pull first/last name and email from AD using username or email.",
    )

    class Meta:
        model = User
        fields = (
            "username", "email", "first_name", "last_name",
            "depos", "zones", "territories",
            "is_active", "is_staff", "is_superuser", "groups", "user_permissions",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["depos"].queryset = Depo.objects.order_by("code", "name")
        self.fields["zones"].queryset = Zone.objects.order_by("code", "name")
        self.fields["territories"].queryset = Territory.objects.order_by("code", "name")

        if self.instance and self.instance.pk:
            self.fields["depos"].initial = self.instance.depo_links.values_list("depo_id", flat=True)
            self.fields["zones"].initial = self.instance.zone_links.values_list("zone_id", flat=True)
            self.fields["territories"].initial = self.instance.territory_links.values_list("territory_id", flat=True)

    def _sync_user_links(self, user):
        selected_depo_ids = set(self.cleaned_data["depos"].values_list("id", flat=True))
        selected_zone_ids = set(self.cleaned_data["zones"].values_list("id", flat=True))
        selected_territory_ids = set(self.cleaned_data["territories"].values_list("id", flat=True))

        current_depo_ids = set(user.depo_links.values_list("depo_id", flat=True))
        current_zone_ids = set(user.zone_links.values_list("zone_id", flat=True))
        current_territory_ids = set(user.territory_links.values_list("territory_id", flat=True))

        UserDepoMap.objects.filter(user=user, depo_id__in=current_depo_ids - selected_depo_ids).delete()
        UserZoneMap.objects.filter(user=user, zone_id__in=current_zone_ids - selected_zone_ids).delete()
        UserTerritoryMap.objects.filter(
            user=user,
            territory_id__in=current_territory_ids - selected_territory_ids,
        ).delete()

        UserDepoMap.objects.bulk_create(
            [UserDepoMap(user=user, depo_id=depo_id) for depo_id in (selected_depo_ids - current_depo_ids)],
            ignore_conflicts=True,
        )
        UserZoneMap.objects.bulk_create(
            [UserZoneMap(user=user, zone_id=zone_id) for zone_id in (selected_zone_ids - current_zone_ids)],
            ignore_conflicts=True,
        )
        UserTerritoryMap.objects.bulk_create(
            [
                UserTerritoryMap(user=user, territory_id=territory_id)
                for territory_id in (selected_territory_ids - current_territory_ids)
            ],
            ignore_conflicts=True,
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
            with transaction.atomic():
                user.save()
                self.save_m2m()
                self._sync_user_links(user)
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
