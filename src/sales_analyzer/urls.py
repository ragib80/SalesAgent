from django.urls import path
from sales_analyzer.views import ChatView,ChatAPIView,ExistingConversationAPIView
from sales_analyzer.prompt_helper_view import GetFilterValuesAPIView, ApplyFiltersAPIView

urlpatterns = [
    path('index/', ChatView.as_view(), name='chat'),
    path('query/', ChatAPIView.as_view(), name='sales-query'),
   
    path('query/existing/<uuid:conversation_uuid>/', ExistingConversationAPIView.as_view(), name='existing-sales-query'),  # For existing conversations
    
    path('filters/<str:field>/', GetFilterValuesAPIView.as_view(), name='get-filter-values'),
    path('apply-filters/', ApplyFiltersAPIView.as_view(), name='apply-filters'),
    
]
