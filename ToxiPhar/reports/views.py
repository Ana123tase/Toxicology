from django.http import HttpResponse
from predictor.services import PredictionService
from .pdf_builder import build_report

def download_report(request):
    smiles = request.GET.get("smiles")
    result = PredictionService.predict(smiles)
    pdf_bytes = build_report(result)
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="toxicity_report.pdf"'
    return response