from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import get_object_or_404
# Importing the custom function
from core.middleware.current_user import get_current_user
from rest_framework.pagination import PageNumberPagination


class ConversationPagination(PageNumberPagination):
    page_size = 15                # fixed at 15 per page
    page_query_param = 'page'     # ?page=1,2,3,...
    page_size_query_param = None  # do not allow client override
    max_page_size = 10000000000            # optional safety



# class ConversationMessagePagination(PageNumberPagination):
#     page_size = 5                    # latest 10 by default
#     page_size_query_param = 'page_size'
#     max_page_size = 10000000000

class ConversationMessagePagination(PageNumberPagination):
    page_size = 8                    # latest 10 by default
    page_query_param = 'page'     # ?page=1,2,3,...
    page_size_query_param = None  # do not allow client override
    max_page_size = 10000000000

def get_paginated_response(self, data):
        return Response({
            "meta": {
                "page": self.page.number,
                "page_size": self.get_page_size(self.request),
                "total_pages": self.page.paginator.num_pages,
                "total_items": self.page.paginator.count,
                "has_next": self.page.has_next(),
                "has_previous": self.page.has_previous(),
                "next_page": self.page.next_page_number() if self.page.has_next() else None,
                "previous_page": self.page.previous_page_number() if self.page.has_previous() else None,
                "next": self.get_next_link(),
                "previous": self.get_previous_link(),
            },
            "results": data
        })    
