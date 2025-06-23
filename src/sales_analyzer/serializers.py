from rest_framework import serializers

class QueryRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField()

class QueryResponseSerializer(serializers.Serializer):
    sql_query = serializers.CharField()
    results = serializers.ListField(child=serializers.DictField())
    analysis = serializers.CharField()