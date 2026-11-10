# custom_admin/views.py
"""
Frontend (non-admin) user management for Voice of Sales.

Every action is guarded by the matching SalesAuthUser model permission
(see custom_admin.permissions). Page views only return an HTML shell; all
data is served by the JWT-protected APIs below.

Never writable from here: username, email (both are used to link Microsoft
sign-ins, see user_auth.views.find_local_user_to_link), password, groups,
permissions, is_staff, is_superuser.
"""
import logging

from django.contrib.auth import get_user_model
from django.core.paginator import Paginator
from django.db import IntegrityError, transaction
from django.db.models import Count, Exists, F, IntegerField, OuterRef, Q, Subquery, Value
from django.db.models.functions import Coalesce
from django.contrib.auth.models import Group
from django.shortcuts import get_object_or_404
from django.views.generic import TemplateView
from rest_framework import serializers, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.middleware.current_user import set_current_chat_user, clear_current_chat_user
from core.services.microsoft_graph_service import MicrosoftGraphDirectoryService
from custom_admin.forms import SalesAuthUserCreateFromADForm
from custom_admin.permissions import (
    UNRESTRICTED_GROUPS,
    CanAddSalesUser,
    CanChangeSalesUser,
    CanUseUserManagement,
    CanViewSalesUsers,
    edit_block_reason,
)
from user_auth.models import (
    Depo,
    Division,
    Territory,
    Zone,
    UserDepoMap,
    UserDivisionMap,
    UserTerritoryMap,
    UserZoneMap,
)

User = get_user_model()
logger = logging.getLogger(__name__)

# level -> (master model, map model, payload key). The map FK field is named after the level
# and its reverse relation on the user is "<level>_links".
COVERAGE_LEVELS = {
    "depo": (Depo, UserDepoMap, "depo_ids"),
    "territory": (Territory, UserTerritoryMap, "territory_ids"),
    "zone": (Zone, UserZoneMap, "zone_ids"),
    "division": (Division, UserDivisionMap, "division_ids"),
}

LIST_ORDERING = {
    "name": ("first_name", "last_name", "username"),
    "username": ("username",),
    "joined": ("date_joined",),
    "last_login": ("last_login",),
}
MAX_PAGE_SIZE = 100


# ---------------------------------------------------------------- pages

class UserListPageView(TemplateView):
    template_name = "custom_admin/user_list.html"


class UserAddPageView(TemplateView):
    template_name = "custom_admin/user_form.html"
    extra_context = {"form_mode": "add"}


class UserEditPageView(TemplateView):
    template_name = "custom_admin/user_form.html"
    extra_context = {"form_mode": "edit"}


# ---------------------------------------------------------------- helpers

def _first_form_error(form):
    for errors in form.errors.values():
        if errors:
            return str(errors[0])
    return "Couldn't validate this user."


def _profile_payload(user):
    return {
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": user.get_full_name(),
    }


def _role(user, in_admin_group=None):
    if user.is_superuser:
        return "superuser"
    if user.is_staff:
        return "staff"
    if in_admin_group is None:
        in_admin_group = user.groups.filter(name__in=UNRESTRICTED_GROUPS).exists()
    return "admin" if in_admin_group else None


def _user_detail(user, actor):
    coverage = {}
    for level, (_model, _map_model, _key) in COVERAGE_LEVELS.items():
        links = getattr(user, f"{level}_links").select_related(level).order_by(f"{level}__code", f"{level}__name")
        coverage[level] = [
            {"id": obj.id, "code": obj.code, "name": obj.name}
            for obj in (getattr(link, level) for link in links)
        ]

    reason = edit_block_reason(actor, user)
    return {
        "uuid": str(user.uuid),
        **_profile_payload(user),
        "identity_provider": user.identity_provider,
        "identity_provider_label": user.get_identity_provider_display(),
        "is_microsoft_linked": bool(user.azure_ad_object_id),
        "is_active": user.is_active,
        "role": _role(user),
        "last_login": user.last_login,
        "last_microsoft_login": user.last_microsoft_login,
        "date_joined": user.date_joined,
        "coverage": coverage,
        "can_edit": reason is None,
        "edit_block_reason": reason,
    }


def _link_count(map_model):
    counts = (
        map_model.objects.filter(user=OuterRef("pk"))
        .order_by()
        .values("user")
        .annotate(c=Count("id"))
        .values("c")[:1]
    )
    return Coalesce(Subquery(counts, output_field=IntegerField()), Value(0))


def _validate_coverage_ids(attrs):
    for level, (model, _map_model, key) in COVERAGE_LEVELS.items():
        if key not in attrs:
            continue
        ids = set(attrs[key] or [])
        found = set(model.objects.filter(id__in=ids).values_list("id", flat=True))
        missing = ids - found
        if missing:
            raise serializers.ValidationError(
                {key: f"Unknown {level} id(s): {', '.join(map(str, sorted(missing)))}"}
            )
        attrs[key] = ids
    return attrs


def _sync_coverage(user, data):
    """Replace the user's coverage for each level present in `data`. Returns changed levels."""
    changed = []
    for level, (_model, map_model, key) in COVERAGE_LEVELS.items():
        if key not in data:
            continue
        wanted = data[key]
        current = set(map_model.objects.filter(user=user).values_list(f"{level}_id", flat=True))
        removed, added = current - wanted, wanted - current
        if removed:
            map_model.objects.filter(user=user, **{f"{level}_id__in": removed}).delete()
        if added:
            try:
                with transaction.atomic():
                    map_model.objects.bulk_create([map_model(user=user, **{f"{level}_id": pk}) for pk in added])
            except IntegrityError:
                # A concurrent save already added some of these; insert the rest one by one.
                for pk in added:
                    map_model.objects.get_or_create(user=user, **{f"{level}_id": pk})
        if removed or added:
            changed.append(level)
    return changed


def _email_conflict(form):
    """409 if the directory email already belongs to a local user (it would break Microsoft sign-in linking)."""
    email = (form.instance.email or "").strip()
    if email and User.objects.filter(email__iexact=email).exists():
        return Response(
            {"detail": f"Another user already uses {email}. Nothing was changed."},
            status=status.HTTP_409_CONFLICT,
        )
    return None


def _get_target(user_uuid):
    return get_object_or_404(User, uuid=user_uuid)


def _blocked(actor, target):
    reason = edit_block_reason(actor, target)
    if reason:
        return Response({"detail": reason}, status=status.HTTP_403_FORBIDDEN)
    return None


# ---------------------------------------------------------------- serializers

class IdentifierSerializer(serializers.Serializer):
    identifier = serializers.CharField(max_length=254, trim_whitespace=True)


def _id_list():
    return serializers.ListField(child=serializers.IntegerField(), required=False, max_length=5000)


class UserCreateSerializer(IdentifierSerializer):
    depo_ids = _id_list()
    territory_ids = _id_list()
    zone_ids = _id_list()
    division_ids = _id_list()

    def validate(self, attrs):
        return _validate_coverage_ids(attrs)


class UserUpdateSerializer(serializers.Serializer):
    """Whitelist of fields the frontend may change. Anything else is ignored."""
    first_name = serializers.CharField(max_length=150, allow_blank=True, required=False, trim_whitespace=True)
    last_name = serializers.CharField(max_length=150, allow_blank=True, required=False, trim_whitespace=True)
    is_active = serializers.BooleanField(required=False)
    depo_ids = _id_list()
    territory_ids = _id_list()
    zone_ids = _id_list()
    division_ids = _id_list()

    def validate(self, attrs):
        return _validate_coverage_ids(attrs)


# ---------------------------------------------------------------- APIs

class CoverageOptionsAPIView(APIView):
    permission_classes = [IsAuthenticated, CanUseUserManagement]

    def get(self, request):
        def rows(model):
            return list(model.objects.order_by("code", "name").values("id", "code", "name"))

        return Response({
            "depos": rows(Depo),
            "territories": rows(Territory),
            "zones": rows(Zone),
            "divisions": rows(Division),
        })


class UserCollectionAPIView(APIView):
    """GET: list users (view). POST: create a user from Microsoft Entra ID (add)."""
    http_method_names = ["get", "post", "head", "options"]

    def get_permissions(self):
        perm = CanAddSalesUser if self.request.method == "POST" else CanViewSalesUsers
        return [IsAuthenticated(), perm()]

    def get(self, request):
        params = request.query_params
        qs = User.objects.annotate(
            in_admin_group=Exists(Group.objects.filter(user=OuterRef("pk"), name__in=UNRESTRICTED_GROUPS)),
            **{f"{level}_count": _link_count(map_model) for level, (_m, map_model, _k) in COVERAGE_LEVELS.items()},
        )

        q = (params.get("q") or "").strip()[:100]
        if q:
            qs = qs.filter(
                Q(username__icontains=q) | Q(email__icontains=q)
                | Q(first_name__icontains=q) | Q(last_name__icontains=q)
            )

        status_filter = params.get("status")
        if status_filter == "active":
            qs = qs.filter(is_active=True)
        elif status_filter == "inactive":
            qs = qs.filter(is_active=False)

        provider = params.get("provider")
        if provider in dict(User.IDENTITY_PROVIDER_CHOICES):
            qs = qs.filter(identity_provider=provider)

        for level, (_model, map_model, _key) in COVERAGE_LEVELS.items():
            raw = params.get(level)
            if raw:
                try:
                    obj_id = int(raw)
                except ValueError:
                    return Response({"detail": f"Invalid {level} filter."}, status=status.HTTP_400_BAD_REQUEST)
                qs = qs.filter(Exists(map_model.objects.filter(user=OuterRef("pk"), **{f"{level}_id": obj_id})))

        if params.get("missing_depo") == "1":
            qs = qs.filter(depo_count=0, is_superuser=False, is_staff=False, in_admin_group=False)

        ordering = params.get("ordering") or "-joined"
        desc = ordering.startswith("-")
        fields = LIST_ORDERING.get(ordering.lstrip("-"), LIST_ORDERING["joined"])
        order_by = [F(f).desc(nulls_last=True) if desc else F(f).asc(nulls_last=True) for f in fields]
        qs = qs.order_by(*order_by, "id")

        try:
            page_size = min(max(int(params.get("page_size", 20)), 1), MAX_PAGE_SIZE)
        except ValueError:
            page_size = 20
        page = Paginator(qs, page_size).get_page(params.get("page"))

        results = [
            {
                "uuid": str(u.uuid),
                **_profile_payload(u),
                "identity_provider": u.identity_provider,
                "identity_provider_label": u.get_identity_provider_display(),
                "is_active": u.is_active,
                "role": _role(u, u.in_admin_group),
                "coverage_counts": {level: getattr(u, f"{level}_count") for level in COVERAGE_LEVELS},
                "last_login": u.last_login,
                "date_joined": u.date_joined,
            }
            for u in page.object_list
        ]
        return Response({
            "count": page.paginator.count,
            "page": page.number,
            "num_pages": page.paginator.num_pages,
            "page_size": page_size,
            "results": results,
        })

    def post(self, request):
        ser = UserCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        form = SalesAuthUserCreateFromADForm(data={"ad_identifier": data["identifier"]})
        if not form.is_valid():
            return Response({"detail": _first_form_error(form)}, status=status.HTTP_400_BAD_REQUEST)
        conflict = _email_conflict(form)
        if conflict:
            return conflict

        token = set_current_chat_user(request.user)  # audit: created_by
        try:
            with transaction.atomic():
                user = form.save()
                _sync_coverage(user, {key: data.get(key, set()) for _m, _mm, key in COVERAGE_LEVELS.values()})
        finally:
            clear_current_chat_user(token)

        logger.info("User %s created from frontend by %s", user.username, request.user.username)
        return Response(_user_detail(user, request.user), status=status.HTTP_201_CREATED)


class UserLookupAPIView(APIView):
    """Preview the Microsoft directory profile without creating anything."""
    permission_classes = [IsAuthenticated, CanAddSalesUser]

    def post(self, request):
        ser = IdentifierSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        form = SalesAuthUserCreateFromADForm(data={"ad_identifier": ser.validated_data["identifier"]})
        if not form.is_valid():
            return Response({"detail": _first_form_error(form)}, status=status.HTTP_400_BAD_REQUEST)
        conflict = _email_conflict(form)
        if conflict:
            return conflict

        return Response({"profile": _profile_payload(form.instance)})


class UserDetailAPIView(APIView):
    """GET: user detail (view). PATCH: update whitelisted fields + coverage (change)."""
    http_method_names = ["get", "patch", "head", "options"]

    def get_permissions(self):
        perm = CanChangeSalesUser if self.request.method == "PATCH" else CanViewSalesUsers
        return [IsAuthenticated(), perm()]

    def get(self, request, user_uuid):
        return Response(_user_detail(_get_target(user_uuid), request.user))

    def patch(self, request, user_uuid):
        target = _get_target(user_uuid)
        blocked = _blocked(request.user, target)
        if blocked:
            return blocked

        ser = UserUpdateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        changed_fields = [
            f for f in ("first_name", "last_name", "is_active")
            if f in data and getattr(target, f) != data[f]
        ]
        for f in changed_fields:
            setattr(target, f, data[f])

        token = set_current_chat_user(request.user)  # audit: updated_by
        try:
            with transaction.atomic():
                if changed_fields:
                    target.save(update_fields=changed_fields + ["updated_at", "updated_by"])
                changed_levels = _sync_coverage(target, data)
        finally:
            clear_current_chat_user(token)

        if changed_fields or changed_levels:
            logger.info(
                "User %s updated from frontend by %s (fields=%s coverage=%s)",
                target.username, request.user.username, changed_fields, changed_levels,
            )
        return Response(_user_detail(target, request.user))


class UserSyncAPIView(APIView):
    """Refresh name + email from Microsoft Entra ID (change)."""
    permission_classes = [IsAuthenticated, CanChangeSalesUser]

    def post(self, request, user_uuid):
        target = _get_target(user_uuid)
        blocked = _blocked(request.user, target)
        if blocked:
            return blocked

        lookup_key = target.azure_ad_object_id or target.email or target.username
        try:
            prof = MicrosoftGraphDirectoryService().find_user(lookup_key)
        except Exception:
            logger.exception("Microsoft directory lookup failed for %s", target.username)
            return Response(
                {"detail": "Microsoft directory lookup is unavailable. Try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if not prof:
            return Response(
                {"detail": "This account wasn't found in the Microsoft directory."},
                status=status.HTTP_404_NOT_FOUND,
            )
        if target.azure_ad_object_id and prof.object_id and prof.object_id != target.azure_ad_object_id:
            return Response(
                {"detail": "The directory returned a different Microsoft account. Nothing was changed."},
                status=status.HTTP_409_CONFLICT,
            )

        new_email = (prof.email_address or "").strip()
        if new_email and User.objects.filter(email__iexact=new_email).exclude(pk=target.pk).exists():
            return Response(
                {"detail": f"Another user already uses {new_email}. Nothing was changed."},
                status=status.HTTP_409_CONFLICT,
            )

        updates = {"first_name": prof.first_name, "last_name": prof.last_name, "email": new_email}
        changed_fields = [f for f, v in updates.items() if v and getattr(target, f) != v]
        for f in changed_fields:
            setattr(target, f, updates[f])

        if changed_fields:
            token = set_current_chat_user(request.user)
            try:
                target.save(update_fields=changed_fields + ["updated_at", "updated_by"])
            finally:
                clear_current_chat_user(token)
            logger.info(
                "User %s refreshed from Microsoft by %s (fields=%s)",
                target.username, request.user.username, changed_fields,
            )

        return Response({"changed": changed_fields, "user": _user_detail(target, request.user)})
