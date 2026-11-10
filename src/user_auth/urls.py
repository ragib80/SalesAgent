from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from user_auth.views import (
    CustomTokenObtainPairView,
    LoginInitiateView,
    LocalLoginView,
    LogoutAPIView,
    MicrosoftCallbackView,
    MicrosoftLoginView,
    OTPVerifyView,
    WhoAmIView,
)

urlpatterns = [
    path("login/", LocalLoginView.as_view(), name="local_login"),
    path("microsoft/login/", MicrosoftLoginView.as_view(), name="microsoft_login"),
    path("microsoft/callback/", MicrosoftCallbackView.as_view(), name="microsoft_callback"),

    # Legacy AD + OTP rollback endpoints. Disabled unless enabled in settings.
    path("auth/", LoginInitiateView.as_view(), name="login_initiate"),
    path("otp/verify/", OTPVerifyView.as_view(), name="otp_verify"),

    # Token management.
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("logout/", LogoutAPIView.as_view(), name="logout"),
    path("whoami/", WhoAmIView.as_view(), name="whoami"),

    # Legacy single-step password token endpoint. Disabled unless enabled in settings.
    path("token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
]
