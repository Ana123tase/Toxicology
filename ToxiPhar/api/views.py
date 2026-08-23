from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from predictor.services import PredictionService, PredictionError
from .serializers import PredictionRequestSerializer

class PredictView(APIView):
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            result = PredictionService.predict(serializer.validated_data["smiles"])
        except PredictionError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(result, status=status.HTTP_200_OK)