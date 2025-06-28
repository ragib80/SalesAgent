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
    answer         = serializers.CharField()
    data           = serializers.JSONField()
    operation_plan = serializers.JSONField(required=False)
    # result     = serializers.JSONField(required=False)  # optional if you want raw



class FirstChatResponseSerializer(serializers.Serializer):
    answer         = serializers.CharField()
    data           = serializers.JSONField()
    operation_plan = serializers.JSONField(required=False)
    uuid           = serializers.CharField()   # <-- Add this line