from rest_framework import serializers

class QueryRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField()

class QueryResponseSerializer(serializers.Serializer):
    sql_query = serializers.CharField()
    results = serializers.ListField(child=serializers.DictField())
    analysis = serializers.CharField()




class ChatRequestSerializer(serializers.Serializer):
    prompt    = serializers.CharField()
    page      = serializers.IntegerField(required=False, default=1)
    page_size = serializers.IntegerField(required=False, default=20)

class ChatResponseSerializer(serializers.Serializer):
    answer      = serializers.CharField()
    data        = serializers.JSONField()