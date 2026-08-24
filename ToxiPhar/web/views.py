from django.shortcuts import render, redirect
from django.urls import reverse
from predictor.services import PredictionService, PredictionError, ModelConfigError


def home(request):
    return render(request, "web/landing.html")


def _risk_level(probability: float) -> str:
    if probability < 0.3:
        return "safe"
    if probability < 0.6:
        return "warn"
    return "danger"


def predict_view(request):
    if request.method == "POST":
        smiles = request.POST.get("smiles", "").strip()

        if not smiles:
            request.session["predict_error"] = "Enter a SMILES string before submitting."
            return redirect(reverse("web:predict"))

        try:
            result = PredictionService.predict(smiles)
        except PredictionError as e:
            request.session["predict_error"] = str(e)
            return redirect(reverse("web:predict"))
        except ModelConfigError:
            request.session["predict_error"] = "Prediction service is temporarily unavailable. Try again shortly."
            return redirect(reverse("web:predict"))
        except Exception:
            request.session["predict_error"] = "Something went wrong running that prediction."
            return redirect(reverse("web:predict"))

        request.session["predict_result"] = result
        return redirect(reverse("web:report"))

    error = request.session.pop("predict_error", None)
    return render(request, "web/predict.html", {"error": error})

def report_view(request):
    result = request.session.pop("predict_result", None)

    if result is None:
        # Someone hit /report/ directly with nothing to show.
        return render(request, "web/report.html", {"result": None})

    endpoints = []
    for name, data in result["predictions"].items():
        endpoints.append({
            "name": name,
            "probability": data["probability"],
            "predicted_class": data["predicted_class"],
            "risk_level": _risk_level(data["probability"]),
        })

    return render(request, "web/report.html", {"result": result, "endpoints": endpoints})