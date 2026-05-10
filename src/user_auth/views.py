# user_auth/views.py

import logging
import re
import secrets
from urllib.parse import urlencode, urlparse

import msal
from django.conf import settings
from django.contrib.auth import get_user_model, login as django_login
from django.core.exceptions import ImproperlyConfigured
from django.core.mail import send_mail
from django.db import transaction
from django.db.models import Q
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.html import escape
from django.utils.http import url_has_allowed_host_and_scheme
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from core.services.ad_service import ActiveDirectoryService
from user_auth.models import OTPToken

logger = logging.getLogger(__name__)

MICROSOFT_PROVIDER = "microsoft"
DEFAULT_LOGIN_NEXT = "/api/sales/index/"


def login_template_view(request):
    return render(request, "login.html")


def build_jwt_login_payload(user, designation=""):
    refresh = RefreshToken.for_user(user)
    refresh["uid"] = str(getattr(user, "uuid", ""))
    refresh["un"] = user.username
    refresh["fn"] = user.get_full_name()

    return {
        "access": str(refresh.access_token),
        "refresh": str(refresh),
        "username": user.username,
        "full_name": user.get_full_name(),
        "designation": designation or "",
        "user_uuid": str(getattr(user, "uuid", "")),
    }


def get_legacy_ad_designation(user):
    if not getattr(settings, "AUTH_ENABLE_LEGACY_AD_PASSWORD_LOGIN", False):
        return ""

    try:
        ad = ActiveDirectoryService()
        prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
        return prof.title or "" if prof else ""
    except Exception:
        logger.exception("AD profile lookup failed for %s", user.username)
        return ""


def get_safe_next_url(request):
    raw_next = (request.GET.get("next") or request.POST.get("next") or "").strip()
    if not raw_next:
        raw_next = DEFAULT_LOGIN_NEXT

    if url_has_allowed_host_and_scheme(
        raw_next,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return raw_next

    return DEFAULT_LOGIN_NEXT


def wants_json(request):
    if request.GET.get("format") == "json":
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept and "text/html" not in accept


def auth_error_response(request, detail, status_code=status.HTTP_400_BAD_REQUEST):
    if wants_json(request):
        return Response({"detail": detail}, status=status_code)

    query = urlencode({"auth_error": detail})
    return redirect(f"/welcome/?{query}")


def is_admin_next_url(next_url):
    return urlparse(next_url or "").path.startswith("/admin/")


def auth_error_redirect(request, detail, next_url="", status_code=status.HTTP_400_BAD_REQUEST):
    if wants_json(request):
        return Response({"detail": detail}, status=status_code)

    target = "/admin/login/" if is_admin_next_url(next_url) else "/welcome/"
    query = urlencode({"auth_error": detail})
    return redirect(f"{target}?{query}")


def microsoft_scopes():
    return list(getattr(settings, "MICROSOFT_AUTH_SCOPES", []))


def microsoft_app():
    client_id = getattr(settings, "MICROSOFT_AUTH_CLIENT_ID", "")
    tenant_id = getattr(settings, "MICROSOFT_AUTH_TENANT_ID", "")
    client_secret = getattr(settings, "MICROSOFT_AUTH_CLIENT_SECRET", "")

    missing = [
        name
        for name, value in (
            ("MICROSOFT_AUTH_CLIENT_ID", client_id),
            ("MICROSOFT_AUTH_TENANT_ID", tenant_id),
            ("MICROSOFT_AUTH_CLIENT_SECRET", client_secret),
        )
        if not value
    ]
    if missing:
        raise ImproperlyConfigured(
            "Missing Microsoft auth setting(s): " + ", ".join(missing)
        )

    authority = f"https://login.microsoftonline.com/{tenant_id}"
    return msal.ConfidentialClientApplication(
        client_id=client_id,
        authority=authority,
        client_credential=client_secret,
    )


def microsoft_redirect_uri():
    redirect_uri = getattr(settings, "MICROSOFT_AUTH_REDIRECT_URI", "")
    if not redirect_uri:
        raise ImproperlyConfigured("Missing MICROSOFT_AUTH_REDIRECT_URI")
    return redirect_uri


def get_claim_email(claims):
    for key in ("email", "preferred_username", "upn", "unique_name"):
        value = (claims.get(key) or "").strip()
        if "@" in value:
            return value
    return ""


def get_claim_name_parts(claims):
    first_name = (claims.get("given_name") or "").strip()
    last_name = (claims.get("family_name") or "").strip()
    if first_name or last_name:
        return first_name, last_name

    name = (claims.get("name") or "").strip()
    if not name:
        return "", ""

    parts = name.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def normalize_username(value):
    value = (value or "").strip().lower()
    if "@" in value:
        value = value.split("@", 1)[0]
    value = re.sub(r"[^a-z0-9@.+_-]+", "_", value)
    return value.strip("._-") or "microsoft_user"


def unique_username(base_username):
    User = get_user_model()
    username = normalize_username(base_username)
    candidate = username
    suffix = 1
    while User.objects.filter(username__iexact=candidate).exists():
        suffix += 1
        candidate = f"{username}{suffix}"
    return candidate


def find_user_by_local_password_identifier(identifier):
    identifier = (identifier or "").strip()
    if not identifier:
        return None

    User = get_user_model()
    matches = list(
        User.objects.filter(
            Q(username__iexact=identifier) | Q(email__iexact=identifier),
            is_active=True,
        ).distinct()[:2]
    )
    if len(matches) != 1:
        return None
    return matches[0]


def find_user_by_microsoft_identity(tid, oid):
    User = get_user_model()
    matches = list(
        User.objects.filter(
            azure_ad_tenant_id=tid,
            azure_ad_object_id=oid,
            is_active=True,
        )[:2]
    )
    if len(matches) > 1:
        raise ValueError("Multiple users are linked to this Microsoft account.")
    return matches[0] if matches else None


def find_local_user_to_link(claims):
    User = get_user_model()
    email = get_claim_email(claims)
    preferred_username = (claims.get("preferred_username") or "").strip()
    upn = (claims.get("upn") or "").strip()

    query = Q()
    if email:
        query |= Q(email__iexact=email)
    for candidate in {preferred_username, upn, normalize_username(preferred_username), normalize_username(email)}:
        if candidate:
            query |= Q(username__iexact=candidate)

    if not query:
        return None

    matches = list(User.objects.filter(query, is_active=True).distinct()[:2])
    if len(matches) > 1:
        raise ValueError("Multiple active local users match this Microsoft account.")
    return matches[0] if matches else None


def create_microsoft_user(claims, tid, oid):
    User = get_user_model()
    email = get_claim_email(claims)
    first_name, last_name = get_claim_name_parts(claims)
    username_seed = claims.get("preferred_username") or email or oid

    user = User(
        username=unique_username(username_seed),
        email=email,
        first_name=first_name,
        last_name=last_name,
        is_active=True,
        identity_provider=MICROSOFT_PROVIDER,
        azure_ad_tenant_id=tid,
        azure_ad_object_id=oid,
        last_microsoft_login=timezone.now(),
        last_login=timezone.now(),
    )
    user.set_unusable_password()
    user.save()
    return user


def update_user_from_microsoft_claims(user, claims, tid, oid):
    email = get_claim_email(claims)
    first_name, last_name = get_claim_name_parts(claims)
    now = timezone.now()
    changed_fields = []

    updates = {
        "identity_provider": MICROSOFT_PROVIDER,
        "azure_ad_tenant_id": tid,
        "azure_ad_object_id": oid,
        "last_microsoft_login": now,
        "last_login": now,
    }
    if email:
        updates["email"] = email
    if first_name:
        updates["first_name"] = first_name
    if last_name:
        updates["last_name"] = last_name

    for field, value in updates.items():
        if getattr(user, field, None) != value:
            setattr(user, field, value)
            changed_fields.append(field)

    if changed_fields:
        user.save(update_fields=changed_fields)
    return user


class LocalLoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = (request.data.get("username") or "").strip()
        password = request.data.get("password") or ""

        if not username or not password:
            return Response(
                {"detail": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = find_user_by_local_password_identifier(username)
        if not user or not user.has_usable_password() or not user.check_password(password):
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        user.last_login = timezone.now()
        user.save(update_fields=["last_login"])
        return Response(build_jwt_login_payload(user), status=status.HTTP_200_OK)


class MicrosoftLoginView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            state = secrets.token_urlsafe(32)
            next_url = get_safe_next_url(request)
            request.session["microsoft_auth_state"] = state
            request.session["microsoft_auth_next"] = next_url

            prompt = (request.GET.get("prompt") or getattr(settings, "MICROSOFT_AUTH_PROMPT", "")).strip()
            auth_kwargs = {
                "scopes": microsoft_scopes(),
                "state": state,
                "redirect_uri": microsoft_redirect_uri(),
            }
            if prompt:
                auth_kwargs["prompt"] = prompt

            auth_url = microsoft_app().get_authorization_request_url(**auth_kwargs)
            return redirect(auth_url)
        except ImproperlyConfigured as exc:
            logger.exception("Microsoft auth is not configured")
            return auth_error_redirect(
                request,
                str(exc),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.exception("Microsoft authorization URL generation failed")
            return auth_error_redirect(
                request,
                "Microsoft sign-in is unavailable. Try again later.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )


class MicrosoftCallbackView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        expected_state = request.session.pop("microsoft_auth_state", "")
        next_url = request.session.pop("microsoft_auth_next", DEFAULT_LOGIN_NEXT)
        returned_state = request.GET.get("state", "")

        if not expected_state or not secrets.compare_digest(expected_state, returned_state):
            return auth_error_redirect(request, "Invalid Microsoft login state.", next_url)

        if request.GET.get("error"):
            detail = request.GET.get("error_description") or request.GET.get("error")
            return auth_error_redirect(request, detail, next_url)

        code = request.GET.get("code")
        if not code:
            return auth_error_redirect(request, "Missing Microsoft authorization code.", next_url)

        try:
            result = microsoft_app().acquire_token_by_authorization_code(
                code,
                scopes=microsoft_scopes(),
                redirect_uri=microsoft_redirect_uri(),
            )
        except ImproperlyConfigured as exc:
            logger.exception("Microsoft auth is not configured")
            return auth_error_redirect(
                request,
                str(exc),
                next_url,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.exception("Microsoft token exchange failed unexpectedly")
            return auth_error_redirect(
                request,
                "Microsoft sign-in is unavailable. Try again later.",
                next_url,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if result.get("error"):
            logger.warning("Microsoft token exchange failed: %s", result)
            detail = result.get("error_description") or result.get("error")
            return auth_error_redirect(request, detail, next_url)

        claims = result.get("id_token_claims") or {}
        tid = (claims.get("tid") or "").strip()
        oid = (claims.get("oid") or "").strip()
        if not tid or not oid:
            return auth_error_redirect(
                request,
                "Microsoft ID token did not contain tenant and object identifiers.",
                next_url,
            )

        try:
            with transaction.atomic():
                user = find_user_by_microsoft_identity(tid, oid)
                if not user:
                    user = find_local_user_to_link(claims)

                    if user:
                        linked_tid = getattr(user, "azure_ad_tenant_id", None)
                        linked_oid = getattr(user, "azure_ad_object_id", None)
                        if (linked_tid and linked_tid != tid) or (linked_oid and linked_oid != oid):
                            return auth_error_redirect(
                                request,
                                "This local user is already linked to a different Microsoft account.",
                                next_url,
                            )
                    elif getattr(settings, "MICROSOFT_AUTH_AUTO_CREATE_USERS", False):
                        user = create_microsoft_user(claims, tid, oid)
                    else:
                        return auth_error_redirect(
                            request,
                            "No active local user is linked to this Microsoft account.",
                            next_url,
                            status_code=status.HTTP_403_FORBIDDEN,
                        )

                user = update_user_from_microsoft_claims(user, claims, tid, oid)
        except ValueError as exc:
            return auth_error_redirect(request, str(exc), next_url, status.HTTP_409_CONFLICT)

        if is_admin_next_url(next_url):
            if not user.is_active or not user.is_staff:
                return auth_error_redirect(
                    request,
                    "This Microsoft account is not allowed to access the admin site.",
                    next_url,
                    status.HTTP_403_FORBIDDEN,
                )

            django_login(
                request,
                user,
                backend="django.contrib.auth.backends.ModelBackend",
            )
            return redirect(next_url)

        payload = build_jwt_login_payload(user)
        if wants_json(request):
            return Response(payload, status=status.HTTP_200_OK)

        return render(
            request,
            "microsoft_login_complete.html",
            {
                "payload": payload,
                "next_url": next_url,
            },
        )


class ADTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["uid"] = str(getattr(user, "uuid", ""))
        token["un"] = user.username
        token["fn"] = user.get_full_name()
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        user = self.user
        data.update(build_jwt_login_payload(user, get_legacy_ad_designation(user)))
        return data


class CustomTokenObtainPairView(TokenObtainPairView):
    permission_classes = [AllowAny]
    serializer_class = ADTokenObtainPairSerializer

    def post(self, request, *args, **kwargs):
        if not getattr(settings, "AUTH_ENABLE_PASSWORD_TOKEN_ENDPOINT", False):
            return Response(
                {"detail": "Password token endpoint is disabled."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return super().post(request, *args, **kwargs)


class LoginInitiateView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        if not getattr(settings, "AUTH_ENABLE_LEGACY_AD_PASSWORD_LOGIN", False):
            return Response(
                {"detail": "Legacy AD password login is disabled."},
                status=status.HTTP_404_NOT_FOUND,
            )

        username = (request.data.get("username") or "").strip()
        password = request.data.get("password") or ""

        if not username or not password:
            return Response(
                {"detail": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        User = get_user_model()
        try:
            user = User.objects.get(Q(username__iexact=username) | Q(email__iexact=username))
        except User.DoesNotExist:
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        if not user.is_active:
            return Response({"detail": "Account is disabled."}, status=status.HTTP_401_UNAUTHORIZED)

        dev_bypass = getattr(settings, "AUTH_DEV_BYPASS_AD", False)
        if not dev_bypass:
            try:
                ad = ActiveDirectoryService()
                if not ad.authenticate_user(user.username, password):
                    return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
            except Exception:
                logger.exception("LDAP authentication error for %s", username)
                return Response(
                    {"detail": "Authentication service unavailable. Try again later."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

        otp_obj, plain_otp = OTPToken.create_for_user(
            user,
            otp_length=getattr(settings, "OTP_LENGTH", 6),
            expiry_minutes=getattr(settings, "OTP_EXPIRY_MINUTES", 5),
        )

        if dev_bypass and settings.DEBUG:
            return Response(
                {
                    "message": "Dev mode: OTP generated (not emailed).",
                    "session_token": otp_obj.session_token,
                    "dev_otp": plain_otp,
                },
                status=status.HTTP_200_OK,
            )

        if not user.email:
            return Response(
                {"detail": "No email address on file. Contact your administrator."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            full_name = user.get_full_name() or user.username
            otp = str(plain_otp)
            text_message = (
                "Voice of Sales\n\n"
                "Please do not click links or open attachments unless you are expecting it and know the content is safe.\n\n"
                f"Hello {full_name},\n\n"
                "Your one-time password is:\n\n"
                f"  {otp}\n\n"
                "This code expires in 5 minutes.\n"
                "If you did not request this, please contact IT support immediately."
            )
            html_message = f"""
            <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5;">
            <div style="font-size: 16px; font-weight: 700; margin-bottom: 10px;">
                Voice of Sales
            </div>
            <div style="color: #c00000; font-weight: 700; margin: 12px 0;">
                Please do not click links or open attachments unless you are expecting it and know the content is safe.
            </div>
            <p>Hello {escape(full_name)},</p>
            <p>Your one-time password is:</p>
            <div style="font-size: 20px; margin: 12px 0;">
                <strong>{escape(otp)}</strong>
            </div>
            <p>This code expires in 5 minutes.<br/>
            If you did not request this, please contact IT support immediately.</p>
            </div>
            """
            send_mail(
                subject="Voice of Sales - Your OTP Verification Code",
                message=text_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
                html_message=html_message,
            )
        except Exception:
            logger.exception("Failed to send OTP email to %s", user.email)
            return Response(
                {"detail": "Failed to send OTP email. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "OTP sent to your registered email address.",
                "session_token": otp_obj.session_token,
            },
            status=status.HTTP_200_OK,
        )


class OTPVerifyView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        if not getattr(settings, "AUTH_ENABLE_LEGACY_AD_PASSWORD_LOGIN", False):
            return Response(
                {"detail": "Legacy AD OTP login is disabled."},
                status=status.HTTP_404_NOT_FOUND,
            )

        session_token = (request.data.get("session_token") or "").strip()
        otp_code = (request.data.get("otp_code") or "").strip()

        if not session_token or not otp_code:
            return Response(
                {"detail": "session_token and otp_code are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            otp_obj = OTPToken.objects.select_related("user").get(session_token=session_token)
        except OTPToken.DoesNotExist:
            return Response(
                {"detail": "Invalid or expired session. Please log in again."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        if not otp_obj.verify(otp_code):
            return Response(
                {"detail": "Invalid or expired OTP code."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        otp_obj.is_used = True
        otp_obj.save(update_fields=["is_used"])

        user = otp_obj.user
        user.last_login = timezone.now()
        user.save(update_fields=["last_login"])
        return Response(
            build_jwt_login_payload(user, get_legacy_ad_designation(user)),
            status=status.HTTP_200_OK,
        )


class LogoutAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            raw = request.data.get("refresh_token") or request.headers.get("Authorization")
            if not raw:
                return Response({"detail": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)

            if isinstance(raw, str) and raw.startswith("Bearer "):
                raw = raw[7:]

            token = RefreshToken(raw)
            token.blacklist()
            return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

        except TokenError as e:
            msg = str(e)
            if "blacklisted" in msg.lower():
                return Response({"detail": "Already logged out."}, status=status.HTTP_200_OK)
            return Response({"detail": f"Token error: {msg}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception("Unexpected error during logout")
            return Response({"detail": "An error occurred during logout."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class WhoAmIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        u = request.user
        return Response({
            "user_uuid": str(getattr(u, "uuid", "")),
            "username": u.username,
            "full_name": u.get_full_name(),
            "email": u.email,
            "is_staff": u.is_staff,
            "is_superuser": u.is_superuser,
        })
