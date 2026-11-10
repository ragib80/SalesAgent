from django.urls import path

from custom_admin.views import (
    CoverageOptionsAPIView,
    UserAddPageView,
    UserCollectionAPIView,
    UserDetailAPIView,
    UserEditPageView,
    UserListPageView,
    UserLookupAPIView,
    UserSyncAPIView,
)

urlpatterns = [
    # Pages (HTML shells; data comes from the APIs below)
    path("users/", UserListPageView.as_view(), name="frontend-user-list"),
    path("users/add/", UserAddPageView.as_view(), name="frontend-user-add"),
    path("users/<uuid:user_uuid>/", UserEditPageView.as_view(), name="frontend-user-edit"),

    # JSON APIs (JWT)
    path("api/users/", UserCollectionAPIView.as_view(), name="frontend-user-api"),
    path("api/users/lookup/", UserLookupAPIView.as_view(), name="frontend-user-lookup"),
    path("api/users/<uuid:user_uuid>/", UserDetailAPIView.as_view(), name="frontend-user-detail-api"),
    path("api/users/<uuid:user_uuid>/sync/", UserSyncAPIView.as_view(), name="frontend-user-sync"),
    path("api/coverage-options/", CoverageOptionsAPIView.as_view(), name="frontend-coverage-options"),
]
