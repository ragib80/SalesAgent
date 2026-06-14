# custom_admin/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import get_user_model
from django.db.models import Count

from custom_admin.forms import (
    SalesAuthUserCreateFromADForm,
    SalesAuthUserChangeForm,
)
from core.services.ad_service import ActiveDirectoryService


from user_auth.models import Depo, Zone, Territory, UserDepoMap, UserZoneMap, UserTerritoryMap, Division, UserDivisionMap

User = get_user_model()


# ---- Masters so autocomplete works ----
@admin.register(Depo)
class DepoAdmin(admin.ModelAdmin):
    search_fields = ["id", "code", "name"]
    list_display = ["id", "code", "name"]
    ordering = ["code"]
    list_per_page = 20


@admin.register(Zone)
class ZoneAdmin(admin.ModelAdmin):
    search_fields = ["id", "code", "name"]
    list_display = ["id", "code", "name"]
    ordering = ["code"]
    list_per_page = 20


@admin.register(Territory)
class TerritoryAdmin(admin.ModelAdmin):
    search_fields = ["id", "code", "name"]
    list_display = ["id", "code", "name"]
    ordering = ["code"]
    list_per_page = 20


@admin.register(Division)
class DivisionAdmin(admin.ModelAdmin):
    search_fields = ["id", "code", "name"]
    list_display = ["id", "code", "name"]
    ordering = ["code"]
    list_per_page = 20


# ---- Inlines on User ----
class UserDepoInline(admin.TabularInline):
    model = UserDepoMap
    fk_name = "user"
    extra = 1
    autocomplete_fields = ["depo"]
    show_change_link = True


class UserZoneInline(admin.TabularInline):
    model = UserZoneMap
    fk_name = "user"
    extra = 1
    autocomplete_fields = ["zone"]
    show_change_link = True


class UserTerritoryInline(admin.TabularInline):
    model = UserTerritoryMap
    fk_name = "user"
    extra = 1
    autocomplete_fields = ["territory"]
    show_change_link = True


class UserDivisionInline(admin.TabularInline):
    model = UserDivisionMap
    fk_name = "user"
    extra = 1
    autocomplete_fields = ["division"]
    show_change_link = True


@admin.register(User)
class SalesAuthUserAdmin(UserAdmin):
    """
    Unified user admin:
    - Add page = single AD identifier (pulls from AD)
    - Change page = 'Sync from AD now' toggle + mapping inlines
    - Bulk action = Sync selected from AD
    - List counts = annotated using related_name paths (correct + orderable)
    """
    add_form = SalesAuthUserCreateFromADForm
    form = SalesAuthUserChangeForm
    inlines = [UserDepoInline, UserZoneInline, UserTerritoryInline, UserDivisionInline]
    save_on_top = True

    list_display = (
        "email", "username", "first_name", "last_name",
        "identity_provider",
        "is_active", "is_staff", "is_superuser",
        "depo_count", "zone_count", "territory_count", "division_count",
        "date_joined",
    )
    list_per_page = 20
    list_filter = ("identity_provider", "is_active", "is_staff", "is_superuser", "date_joined")
    search_fields = (
        "email", "username", "first_name", "last_name",
        "azure_ad_tenant_id", "azure_ad_object_id",
    )
    ordering = ("-date_joined",)
    readonly_fields = (
        "identity_provider",
        "azure_ad_tenant_id",
        "azure_ad_object_id",
        "last_microsoft_login",
        "last_login",
        "date_joined",
    )

    # Correct counts via annotations on related_name; distinct avoids dupes
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.annotate(
            depo_ct=Count("depo_links", distinct=True),
            zone_ct=Count("zone_links", distinct=True),
            territory_ct=Count("territory_links", distinct=True),
            division_ct=Count("division_links", distinct=True),
        )

    # Read from annotations (orderable columns)
    def depo_count(self, obj):
        return getattr(obj, "depo_ct", 0)
    depo_count.short_description = "Depo"
    depo_count.admin_order_field = "depo_ct"

    def zone_count(self, obj):
        return getattr(obj, "zone_ct", 0)
    zone_count.short_description = "Zone"
    zone_count.admin_order_field = "zone_ct"

    def territory_count(self, obj):
        return getattr(obj, "territory_ct", 0)
    territory_count.short_description = "Territory"
    territory_count.admin_order_field = "territory_ct"

    def division_count(self, obj):
        return getattr(obj, "division_ct", 0)
    division_count.short_description = "Division"
    division_count.admin_order_field = "division_ct"

    # CHANGE view (add our sync checkbox)
    fieldsets = (
        (None, {"fields": ("username", "email", "first_name", "last_name")}),
        (_("Active Directory"), {"fields": ("sync_from_ad",)}),
        (_("Microsoft Identity"), {"fields": (
            "identity_provider",
            "azure_ad_tenant_id",
            "azure_ad_object_id",
            "last_microsoft_login",
        )}),
        (_("Permissions"), {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )

    # ADD view: only the single AD identifier field
    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("ad_identifier",),
        }),
    )

    actions = ["sync_from_ad_action"]

    # Bulk action: sync selected from AD
    def sync_from_ad_action(self, request, queryset):
        ad = ActiveDirectoryService()
        updated = 0
        for user in queryset:
            prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
            if not prof:
                continue
            changed = False
            if prof.first_name and user.first_name != prof.first_name:
                user.first_name = prof.first_name; changed = True
            if prof.last_name and user.last_name != prof.last_name:
                user.last_name = prof.last_name; changed = True
            if prof.email_address and user.email != prof.email_address:
                user.email = prof.email_address; changed = True
            if changed:
                user.save(update_fields=["first_name", "last_name", "email"])
                updated += 1
        self.message_user(request, f"Synced {updated} user(s) from AD.")
    sync_from_ad_action.short_description = "Sync selected users from AD"
