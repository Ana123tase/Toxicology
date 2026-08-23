from django.shortcuts import render
from predictor.services import PredictionService, PredictionError

def predict_view(request):
    result = None
    error = None
    if request.method == "POST":
        smiles = request.POST.get("smiles", "").strip()
        try:
            result = PredictionService.predict(smiles)
        except PredictionError as e:
            error = str(e)
    return render(request, "web/predict.html", {"result": result, "error": error})