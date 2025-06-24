from django.urls import path
from .views import ChatView,ChatAPIView

urlpatterns = [
    path('index/', ChatView.as_view(), name='chat'),
    path('query/', ChatAPIView.as_view(), name='sales-query'),

    
]
