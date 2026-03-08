from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from user_auth.views import (
    CustomTokenObtainPairView,
    LoginInitiateView,
    LogoutAPIView,
    OTPVerifyView,
    WhoAmIView,
)

urlpatterns = [
    # ── MFA login flow (use these) ────────────────────────────────────────────
    # Step 1: verify username/password via LDAPS → send OTP email
    path("login/", LoginInitiateView.as_view(), name="login_initiate"),
    # Step 2: verify OTP → receive JWT access + refresh tokens
    path("otp/verify/", OTPVerifyView.as_view(), name="otp_verify"),

    # ── Token management ──────────────────────────────────────────────────────
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("logout/", LogoutAPIView.as_view(), name="logout"),
    path("whoami/", WhoAmIView.as_view(), name="whoami"),

    # ── Legacy single-step login (kept for backward compat — no OTP) ─────────
    path("token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
]
