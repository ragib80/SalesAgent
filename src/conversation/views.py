from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer
from django.shortcuts import get_object_or_404
# Importing the custom function
from core.middleware.current_user import get_current_user


class ConversationListCreateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        List all conversations for the logged-in user
        """
        current_user = request.user  # Get the current user directly from request.user
        print('current_user ',current_user)
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        conversations = Conversation.active.filter(
            user=current_user, is_deleted=False)
        print ("C U C ",conversations)
        serializer = ConversationSerializer(conversations, many=True)
        return Response(serializer.data)

    def post(self, request):
        """
        Create a new conversation for the logged-in user
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        data = request.data
        # Automatically assign the logged-in user
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
        List all messages for a specific conversation identified by its UUID
        """
        current_user = request.user  # Get the current user
        if current_user is None:
            return Response({"detail": "Authentication credentials were not provided."}, status=status.HTTP_401_UNAUTHORIZED)

        # Get conversation using UUID, not primary key
        conversation = get_object_or_404(Conversation, uuid=pk, user=current_user)
        messages = Message.objects.filter(conversation=conversation, is_deleted=False)
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)

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
