"""
Permissions for frontend (non-admin) user management.

Each action maps to the standard Django model permission on SalesAuthUser:
  list / detail -> view_salesauthuser OR change_salesauthuser (Django admin semantics)
  create        -> add_salesauthuser
  update / sync -> change_salesauthuser (+ object rules in edit_block_reason)
"""
from django.contrib.auth.models import Permission
from django.db.models import Q
from rest_framework.permissions import BasePermission

VIEW_SALES_USER_PERMISSION = "custom_admin.view_salesauthuser"
ADD_SALES_USER_PERMISSION = "custom_admin.add_salesauthuser"
CHANGE_SALES_USER_PERMISSION = "custom_admin.change_salesauthuser"

# Members of these groups bypass coverage scoping (see agent.utils.access_scope)
UNRESTRICTED_GROUPS = ("Admin", "Super Admin")

# Holders of these (directly or via a group) are user managers: peers can't change them.
USER_MANAGER_CODENAMES = ("add_salesauthuser", "change_salesauthuser")


def _authenticated(user) -> bool:
    return bool(user and user.is_authenticated and user.is_active)


def can_view_sales_users(user) -> bool:
    return _authenticated(user) and (
        user.has_perm(VIEW_SALES_USER_PERMISSION) or user.has_perm(CHANGE_SALES_USER_PERMISSION)
    )


def can_add_sales_users(user) -> bool:
    return _authenticated(user) and user.has_perm(ADD_SALES_USER_PERMISSION)


def can_change_sales_users(user) -> bool:
    return _authenticated(user) and user.has_perm(CHANGE_SALES_USER_PERMISSION)


def user_management_perms(user) -> dict:
    return {
        "view": can_view_sales_users(user),
        "add": can_add_sales_users(user),
        "change": can_change_sales_users(user),
    }


def is_privileged(user) -> bool:
    """Admin-level accounts: only superusers may change them from the frontend."""
    if user.is_superuser or user.is_staff:
        return True
    return user.groups.filter(name__in=UNRESTRICTED_GROUPS).exists()


def is_user_manager(user) -> bool:
    """Can add or change users. Checked on the grants themselves so it holds for inactive accounts too."""
    return Permission.objects.filter(
        Q(user=user) | Q(group__user=user),
        content_type__app_label="custom_admin",
        codename__in=USER_MANAGER_CODENAMES,
    ).exists()


def edit_block_reason(actor, target):
    """Return why `actor` can't change `target`, or None if allowed."""
    if not can_change_sales_users(actor):
        return "You can view this user but don't have permission to change users."
    if actor.pk == target.pk:
        return "You can't change your own account here. Ask another administrator."
    if is_privileged(target) and not actor.is_superuser:
        return "This is an administrator account. Only a superuser can change it."
    if not actor.is_superuser and is_user_manager(target):
        return "This account can manage users. Only a superuser can change it."
    return None


class CanViewSalesUsers(BasePermission):
    message = "You don't have permission to view users."

    def has_permission(self, request, view):
        return can_view_sales_users(request.user)


class CanAddSalesUser(BasePermission):
    message = "You don't have permission to add users."

    def has_permission(self, request, view):
        return can_add_sales_users(request.user)


class CanChangeSalesUser(BasePermission):
    message = "You don't have permission to change users."

    def has_permission(self, request, view):
        return can_change_sales_users(request.user)


class CanUseUserManagement(BasePermission):
    """Any of view / add / change (e.g. coverage master lists for filters and forms)."""
    message = "You don't have access to user management."

    def has_permission(self, request, view):
        return any(user_management_perms(request.user).values())
