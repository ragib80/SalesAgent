# user_auth/views.py

import logging
from django.shortcuts import render
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from core.services.ad_service import ActiveDirectoryService  
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError

logger = logging.getLogger(__name__)


def login_template_view(request):
    # optional HTML page if you ever render a form
    return render(request, "login.html")


# -------- Serializer that uses your auth backends (AD) and adds extra fields --------
class ADTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # (optional) keep compact claims if you like
        token["uid"] = str(getattr(user, "uuid", ""))
        token["un"] = user.username
        token["fn"] = user.get_full_name()
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        user = self.user

        # Pull designation (title) from AD; fall back to ""
        designation = ""
        try:
            ad = ActiveDirectoryService()
            prof = ad.find_user(user.username) or ad.find_user(user.email or user.username)
            if prof:
                designation = prof.title or ""
        except Exception:
            designation = ""

        # exact fields you asked for:
        data["username"] = user.username
        data["full_name"] = user.get_full_name()
        data["designation"] = designation

        # (optional) keep anything else you already returned, e.g. user_uuid
        data.setdefault("user_uuid", str(getattr(user, "uuid", "")))
        return data

# -------- Views --------
class CustomTokenObtainPairView(TokenObtainPairView):
    """
    POST { "username": "...", "password": "..." }
    returns refresh/access only if your AD backend authenticated the user
    (or bypassed if AUTH_DEV_BYPASS_AD=True).
    """
    permission_classes = [AllowAny]
    serializer_class = ADTokenObtainPairSerializer


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
            # If already blacklisted, treat as success (idempotent logout)
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
