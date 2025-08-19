from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer
from django.shortcuts import get_object_or_404
# Importing the custom function
from core.middleware.current_user import get_current_user
from rest_framework.pagination import PageNumberPagination
from core.utils.helper.pagination_helper import ConversationPagination,ConversationMessagePagination,get_paginated_response

# class ConversationPagination(PageNumberPagination):
#     page_size = 5                # fixed at 15 per page
#     page_query_param = 'page'     # ?page=1,2,3,...
#     page_size_query_param = None  # do not allow client override
#     max_page_size = 10000000000            # optional safety

# def get_paginated_response(self, data):
#         return Response({
#             "meta": {
#                 "page": self.page.number,
#                 "page_size": self.get_page_size(self.request),
#                 "total_pages": self.page.paginator.num_pages,
#                 "total_items": self.page.paginator.count,
#                 "has_next": self.page.has_next(),
#                 "has_previous": self.page.has_previous(),
#                 "next_page": self.page.next_page_number() if self.page.has_next() else None,
#                 "previous_page": self.page.previous_page_number() if self.page.has_previous() else None,
#                 "next": self.get_next_link(),
#                 "previous": self.get_previous_link(),
#             },
#             "results": data
#         })    

class ConversationListCreateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        List all conversations for the logged-in user, paginated (15 per page)
        """
        current_user = request.user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."},
                            status=status.HTTP_401_UNAUTHORIZED)

        conversations = (Conversation.active
                         .filter(user=current_user, is_deleted=False)
                         .order_by('-id'))  # stable ordering

        paginator = ConversationPagination()
        page = paginator.paginate_queryset(conversations, request, view=self)
        serializer = ConversationSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    def post(self, request):
        """
        Create a new conversation for the logged-in user
        """
        current_user = request.user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."},
                            status=status.HTTP_401_UNAUTHORIZED)

        data = request.data.copy()
        data['user'] = current_user.id
        serializer = ConversationSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ConversationRetrieveUpdateDestroyAPIView(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        """
        Retrieve a single conversation by its UUID
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        conversation = get_object_or_404(
            Conversation, uuid=pk, user=current_user)
        serializer = ConversationSerializer(conversation)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        Update an existing conversation
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        conversation = get_object_or_404(
            Conversation, uuid=pk, user=current_user)
        serializer = ConversationSerializer(
            conversation, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        Soft delete the conversation
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        conversation = get_object_or_404(
            Conversation, uuid=pk, user=current_user)
        conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ConversationMessagesAPIView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, pk):
        """
        GET /api/conversations/<uuid>/messages/?page=1&page_size=10
        Returns newest→oldest in each page (created_at desc). Frontend will reverse for chat view.
        """
        convo = get_object_or_404(Conversation, uuid=pk, user=request.user)

        qs = (Message.objects
              .filter(conversation=convo, is_deleted=False)
              .order_by('-created_at', '-id'))  # stable + fast

        paginator = ConversationMessagePagination()
        page = paginator.paginate_queryset(qs, request, view=self)
        serializer = MessageSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)
    # def get(self, request, pk):
    #     """
    #     List all messages for a specific conversation identified by its UUID
    #     """
    #     current_user = request.user  # Get the current user
    #     if current_user is None:
    #         return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

    #     # Get conversation using UUID, not primary key
    #     conversation = get_object_or_404(Conversation, uuid=pk, user=current_user)
    #     messages = Message.objects.filter(conversation=conversation, is_deleted=False)
    #     serializer = MessageSerializer(messages, many=True)
    #     return Response(serializer.data)

    def post(self, request, pk):
        """
        Create a new message for a specific conversation identified by its UUID
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        # Get conversation using UUID, not primary key
        conversation = get_object_or_404(Conversation, uuid=pk, user=current_user)

        text = request.data.get('content', '')
        if not text.strip():
            return Response({"detail": "Message content cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)

        # Instead of assigning a model instance to sender, assign the user's username or ID (e.g., current_user.username)
        message = Message.objects.create(
            conversation=conversation,
            sender=current_user.username,  # Assign the username or ID, not the model instance
            text=text
        )
        message.save()
        serializer = MessageSerializer(message)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UserConversationsAPIView(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        List all conversations for the logged-in user
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        conversations = Conversation.objects.filter(
            user=current_user, is_deleted=False)
        serializer = ConversationSerializer(conversations, many=True)
        return Response(serializer.data)
