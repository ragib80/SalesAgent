from django.urls import path
from .views import (
    ConversationListCreateAPIView,
    ConversationRetrieveUpdateDestroyAPIView,
    ConversationMessagesAPIView,
    UserConversationsAPIView,
)

urlpatterns = [
    path('conversations/', ConversationListCreateAPIView.as_view(),
         name='conversation-list-create'),
    path('conversations/<uuid:pk>/', ConversationRetrieveUpdateDestroyAPIView.as_view(),
         name='conversation-retrieve-update-destroy'),
    path('conversations/<uuid:pk>/messages/',
         ConversationMessagesAPIView.as_view(), name='conversation-messages'),
    path('conversations/user_conversations/',
         UserConversationsAPIView.as_view(), name='user-conversations'),
]
