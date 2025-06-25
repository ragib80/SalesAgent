from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView, TokenObtainPairView
from rest_framework.authtoken.views import obtain_auth_token
from user_auth.views import CustomTokenObtainPairView
from user_auth.views import LogoutAPIView

urlpatterns = [
    # Token obtain and refresh views for login and token refresh
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('token/', obtain_auth_token, name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', LogoutAPIView.as_view(), name='logout'),
]
