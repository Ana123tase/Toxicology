from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    smiles = serializers.CharField(max_length=1000)