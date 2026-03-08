# user_auth/views.py

import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.db.models import Q
from django.shortcuts import render
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from django.utils.html import escape
from core.services.ad_service import ActiveDirectoryService
from user_auth.models import OTPToken

logger = logging.getLogger(__name__)


def login_template_view(request):
    # optional HTML page if you ever render a form
    return render(request, "login.html")


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-step token view (kept for backward compat; prefer MFA flow below)
# ──────────────────────────────────────────────────────────────────────────────

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

        designation = ""
        try:
            ad = ActiveDirectoryService()
            prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
            if prof:
                designation = prof.title or ""
        except Exception:
            designation = ""

        data["username"] = user.username
        data["full_name"] = user.get_full_name()
        data["designation"] = designation
        data.setdefault("user_uuid", str(getattr(user, "uuid", "")))
        return data


class CustomTokenObtainPairView(TokenObtainPairView):
    """
    Legacy single-step login (LDAP only, no OTP).
    Kept for backward compatibility — prefer /auth/login/ + /auth/otp/verify/ instead.
    """
    permission_classes = [AllowAny]
    serializer_class = ADTokenObtainPairSerializer


# ──────────────────────────────────────────────────────────────────────────────
# MFA — Step 1: verify credentials → generate & email OTP
# ──────────────────────────────────────────────────────────────────────────────

class LoginInitiateView(APIView):
    """
    POST { "username": "...", "password": "..." }

    1. Looks up the user in Django DB.
    2. Verifies credentials against LDAPS (skipped when AUTH_DEV_BYPASS_AD=True).
    3. Generates a 6-digit OTP, stores a hashed copy, and emails the plain code.
    4. Returns { "session_token": "...", "message": "..." }.

    Dev mode (AUTH_DEV_BYPASS_AD=True & DEBUG=True):
      Email is NOT sent.  The response includes "dev_otp" so you can test locally.
    """

    permission_classes = [AllowAny]

    def post(self, request):
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

        # ── LDAP authentication ──────────────────────────────────────────────
        if not dev_bypass:
            try:
                ad = ActiveDirectoryService()
                if not ad.authenticate_user(user.username, password):
                    return Response(
                        {"detail": "Invalid credentials."},
                        status=status.HTTP_401_UNAUTHORIZED,
                    )
            except Exception:
                logger.exception("LDAP authentication error for %s", username)
                return Response(
                    {"detail": "Authentication service unavailable. Try again later."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

        # ── Generate OTP ─────────────────────────────────────────────────────
        otp_obj, plain_otp = OTPToken.create_for_user(
            user,
            otp_length=getattr(settings, "OTP_LENGTH", 6),
            expiry_minutes=getattr(settings, "OTP_EXPIRY_MINUTES", 5),
        )

        # ── Send / expose OTP ────────────────────────────────────────────────
        if dev_bypass and settings.DEBUG:
            # Dev mode: return OTP in response instead of emailing it
            logger.warning(
                "DEV MODE: OTP for %s is %s (not emailed)", user.username, plain_otp
            )
            return Response(
                {
                    "message": "Dev mode: OTP generated (not emailed).",
                    "session_token": otp_obj.session_token,
                    "dev_otp": plain_otp,  # ONLY present in dev mode
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

            # Plain-text fallback (no red/bold possible here)
            text_message = (
                "Voice of Sales\n\n"
                "Please do not click links or open attachments unless you are expecting it and know the content is safe.\n\n"
                f"Hello {full_name},\n\n"
                "Your one-time password is:\n\n"
                f"  {otp}\n\n"
                "This code expires in 5 minutes.\n"
                "If you did not request this, please contact IT support immediately."
            )

            # HTML version (supports red + bold)
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
            # send_mail(
            #     subject="Your OTP Verification Code",
            #     message=(
            #         f"Hello {user.get_full_name() or user.username},\n\n"
            #         f"Your one-time password is:\n\n"
            #         f"{plain_otp}\n\n"
            #         f"This code expires in 5 minutes.\n"
            #         f"If you did not request this, please contact IT support immediately."
            #     ),
            #     from_email=settings.DEFAULT_FROM_EMAIL,
            #     recipient_list=[user.email],
            #     fail_silently=False,
            # )
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


# ──────────────────────────────────────────────────────────────────────────────
# MFA — Step 2: verify OTP → issue JWT tokens
# ──────────────────────────────────────────────────────────────────────────────

class OTPVerifyView(APIView):
    """
    POST { "session_token": "...", "otp_code": "..." }

    1. Looks up the OTPToken by session_token.
    2. Verifies the code (hash comparison, expiry, used-flag).
    3. Marks the OTP as used.
    4. Issues access + refresh JWT tokens with the same custom claims as the
       legacy view, plus username / full_name / designation.
    """

    permission_classes = [AllowAny]

    def post(self, request):
        session_token = (request.data.get("session_token") or "").strip()
        otp_code = (request.data.get("otp_code") or "").strip()

        if not session_token or not otp_code:
            return Response(
                {"detail": "session_token and otp_code are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            otp_obj = OTPToken.objects.select_related("user").get(
                session_token=session_token
            )
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

        # Mark consumed
        otp_obj.is_used = True
        otp_obj.save(update_fields=["is_used"])

        user = otp_obj.user

        # ── Build JWT with custom claims ──────────────────────────────────────
        refresh = RefreshToken.for_user(user)
        refresh["uid"] = str(getattr(user, "uuid", ""))
        refresh["un"] = user.username
        refresh["fn"] = user.get_full_name()

        designation = ""
        try:
            ad = ActiveDirectoryService()
            prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
            if prof:
                designation = prof.title or ""
        except Exception:
            pass  # designation stays empty; don't block login

        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "username": user.username,
                "full_name": user.get_full_name(),
                "designation": designation,
                "user_uuid": str(getattr(user, "uuid", "")),
            },
            status=status.HTTP_200_OK,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Logout & WhoAmI (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class LogoutAPIView(APIView):
    """
    Blacklists the provided refresh token. Accept either:
      - body:   { "refresh_token": "<token>" }
      - header: Authorization: Bearer <token>
    """
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
    """
    Handy endpoint to verify JWTs and inspect the current user.
    """
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
