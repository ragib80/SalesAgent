# serializers.py

from rest_framework import serializers
from conversation.models.conversation import Conversation
from conversation.models.message import Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = '__all__'


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['uuid', 'user', 'title', 'last_message_at',
                  'pinned', 'is_deleted', 'is_archive', 'messages']
