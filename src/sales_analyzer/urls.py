from django.urls import path
from .views import SalesQueryAPIView,ChatView

urlpatterns = [
    path('index/', ChatView.as_view(), name='chat'),
    path('query/', SalesQueryAPIView.as_view(), name='sales-query'),
]
