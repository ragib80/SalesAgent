from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _
from custom_admin.models import SalesAuthUser
from custom_admin.forms import SalesAuthUserCreateFromADForm

@admin.register(SalesAuthUser)
class SalesAuthUserAdmin(UserAdmin):
    model = SalesAuthUser

    # Use our one-field creation form
    add_form = SalesAuthUserCreateFromADForm

    list_display = ("email", "username", "first_name", "last_name", "is_active", "is_staff", "is_superuser", "date_joined")
    list_filter = ("is_active", "is_staff", "is_superuser", "date_joined")
    search_fields = ("email", "username", "first_name", "last_name")
    ordering = ("-date_joined",)
    readonly_fields = ("last_login", "date_joined")

    # Change view fieldsets (no password fields)
    fieldsets = (
        (None, {"fields": ("username", "email", "first_name", "last_name")}),
        (_("Permissions"), {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )

    # ADD view: show only the AD identifier field
    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("ad_identifier",),
        }),
    )
