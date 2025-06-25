from django.urls import path
from .views import ChatView,ChatAPIView,ExistingConversationAPIView

urlpatterns = [
    path('index/', ChatView.as_view(), name='chat'),
    path('query/', ChatAPIView.as_view(), name='sales-query'),
   
    path('query/existing/<uuid:conversation_uuid>/', ExistingConversationAPIView.as_view(), name='existing-sales-query'),  # For existing conversations

    
]
