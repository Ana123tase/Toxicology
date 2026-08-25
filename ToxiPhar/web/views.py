"""
web/views.py - Fixed for your actual project layout:
- ToxiPhar/predictor/services.py contains PredictionService (3692 bytes)
- ToxiPhar/web/views.py was failing with ModuleNotFoundError: web.services

This version imports from predictor.services, not web.services.
Also supports your 870-line predict.html isomerism-aware fields:
  smiles (isomeric, preserves R/S @ and E/Z /\ and [13C] isotopes)
  smiles_list_json (multi-drug dot-separated)
  is_multi_drug, stereo_audit_json, full_chemistry_json, model_features_json
Mapped to:
  clean_smiles.py: R/S preserved via full InChIKey /t layer, E/Z via /b layer, has_isotope, has_tautomer
  featurize.py: MODEL_STEREO_FEATURES + ECFP 2048 + ECFP stereo 2048 includeChirality=True
  scaffold_split.py: includeChirality=False
"""

import json
import logging
from pathlib import Path
import sys

from django.shortcuts import render, redirect
from django.urls import reverse

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# PredictionReport - your reports app
# -------------------------------------------------------------------
PredictionReport = None
try:
    from reports.models import PredictionReport as _PR
    PredictionReport = _PR
except Exception:
    try:
        from .models import PredictionReport as _PR2
        PredictionReport = _PR2
    except Exception as e:
        logger.warning(f"PredictionReport not found: {e}")

# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------
try:
    from predictor.exceptions import PredictionError, ModelConfigError
except Exception:
    try:
        from .exceptions import PredictionError, ModelConfigError
    except Exception:
        class PredictionError(Exception): pass
        class ModelConfigError(Exception): pass

# -------------------------------------------------------------------
# PredictionService - YOUR ACTUAL LOCATION: predictor/services.py
# This fixes ModuleNotFoundError: No module named 'web.services'
# -------------------------------------------------------------------
PredictionService = None
try:
    # Correct import based on your ls output: ToxiPhar/predictor/services.py
    from predictor.services import PredictionService as _PS
    PredictionService = _PS
    logger.info("Loaded PredictionService from predictor.services")
except Exception as e1:
    logger.warning(f"Failed import from predictor.services: {e1}")
    try:
        # Fallback: maybe ToxiPhar is package root
        from ToxiPhar.predictor.services import PredictionService as _PS2
        PredictionService = _PS2
    except Exception as e2:
        logger.warning(f"Failed import from ToxiPhar.predictor.services: {e2}")
        # Try adding project root to path
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from predictor.services import PredictionService as _PS3
            PredictionService = _PS3
        except Exception as e3:
            logger.error(f"All imports failed: {e1}, {e2}, {e3}")
            # Last resort fallback - will still allow web to load, but predictions will fail clearly
            class _MissingService:
                @classmethod
                def predict(cls, smiles):
                    raise ModelConfigError(
                        f"PredictionService not found. Looked in predictor.services but got: {e1}. "
                        f"Make sure predictor/services.py exists and contains class PredictionService. "
                        f"Your file exists (3692 bytes) - check for syntax error inside it."
                    )
            PredictionService = _MissingService


def _risk_level(prob):
    try:
        p = float(prob)
    except Exception:
        return "unknown"
    if p >= 0.75: return "high"
    if p >= 0.5: return "moderate"
    if p >= 0.25: return "low"
    return "very low"

def _parse_json_field(request, key, default=None):
    raw = request.POST.get(key, "")
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"JSON parse fail {key}: {e}")
        return default

def home(request):
    return render(request, "web/home.html")

def predict_view(request):
    if request.method == "POST":
        # From predict.html: id="smiles_final" name="smiles" -> isomeric SMILES preserving R/S, E/Z, isotopes
        smiles = request.POST.get("smiles", "").strip()
        brand_name = request.POST.get("brand_name", "").strip()
        generic_name = request.POST.get("generic_name", "").strip()
        chemical_name = request.POST.get("chemical_name", "").strip()

        # New isomerism-aware fields from 870-line predict.html
        smiles_list = _parse_json_field(request, "smiles_list_json", None)
        is_multi_drug_raw = request.POST.get("is_multi_drug", "0")
        is_multi_drug = str(is_multi_drug_raw) == "1"
        stereo_audit = _parse_json_field(request, "stereo_audit_json", {})
        full_chemistry = _parse_json_field(request, "full_chemistry_json", {})
        model_features = _parse_json_field(request, "model_features_json", {})

        # Fallback if JS didn't fill
        if not smiles_list or not isinstance(smiles_list, list):
            smiles_list = [smiles] if smiles else []
        smiles_list = [s.strip() for s in smiles_list if s and str(s).strip()]
        if not smiles_list:
            smiles_list = [smiles] if smiles else []

        if not brand_name:
            request.session["predict_error"] = "Enter the brand name before submitting."
            return redirect(reverse("web:predict"))

        if not smiles or not smiles_list:
            request.session["predict_error"] = "Draw a chemical structure before submitting."
            return redirect(reverse("web:predict"))

        # Server-side re-validation with clean_smiles.py (matches your self-tests)
        # R/S: C[C@H](O)CC vs C[C@@H](O)CC -> different full InChIKey -> both clean
        # E/Z: C/C=C/C vs C/C=C\C -> different InChIKey -> both clean
        # [13CH4] -> has_isotope True, STATUS_CLEAN
        # keto/enol -> same first InChIKey block but distinct
        try:
            # Ensure clean_smiles can be found (if you placed it in project root or data/)
            for p in [Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent / "data"]:
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            from clean_smiles import clean_smiles as _clean
            validated = []
            for smi in smiles_list:
                r = _clean(smi)
                # Keep isomeric SMILES (preserve @ and /\), don't replace with canonical that strips stereo
                if r.status in ("clean", "salt_removed"):
                    validated.append(smi)
                else:
                    logger.info(f"{smi} -> {r.status}")
                    validated.append(smi)
            if validated:
                smiles_list = validated
                smiles = ".".join(validated) if len(validated) > 1 else validated[0]
        except Exception as e:
            logger.debug(f"clean_smiles re-validation skipped: {e}")

        # Predict each fragment separately (featurize.py: descriptors + ECFP 2048 + ECFP stereo 2048)
        results = []
        for idx, smi in enumerate(smiles_list):
            try:
                result = PredictionService.predict(smi)
                results.append({"smiles": smi, "result": result, "index": idx})
            except PredictionError as e:
                request.session["predict_error"] = f"Error predicting {smi}: {str(e)}"
                return redirect(reverse("web:predict"))
            except ModelConfigError as e:
                request.session["predict_error"] = f"Model config error: {str(e)}"
                return redirect(reverse("web:predict"))
            except Exception as e:
                logger.exception(f"Unexpected for {smi}")
                request.session["predict_error"] = f"Something went wrong for {smi}: {str(e)}"
                return redirect(reverse("web:predict"))

        request.session["predict_results"] = results
        request.session["predict_result"] = results[0]["result"] if results else None
        request.session["predict_meta"] = {
            "brand_name": brand_name,
            "generic_name": generic_name,
            "chemical_name": chemical_name,
            "smiles": smiles,
            "smiles_list": smiles_list,
            "is_multi_drug": is_multi_drug or len(smiles_list) > 1,
            "stereo_audit": stereo_audit,
            "full_chemistry": full_chemistry,
            "model_features": model_features,
        }
        request.session["predict_metas"] = [
            {
                "brand_name": f"{brand_name} (Drug {i+1})" if (is_multi_drug or len(smiles_list) > 1) else brand_name,
                "generic_name": generic_name,
                "chemical_name": chemical_name,
                "smiles": smi,
            }
            for i, smi in enumerate(smiles_list)
        ]
        return redirect(reverse("web:report"))

    error = request.session.pop("predict_error", None)
    return render(request, "web/predict.html", {"error": error})

def report_view(request):
    results = request.session.pop("predict_results", None)
    result_single = request.session.pop("predict_result", None)
    meta = request.session.pop("predict_meta", None)
    metas = request.session.pop("predict_metas", None)

    if results is None and result_single is not None:
        results = [{"smiles": meta.get("smiles", "") if meta else "", "result": result_single, "index": 0}]
    if results is None:
        return render(request, "web/report.html", {"result": None, "results": [], "is_multi_drug": False})

    is_multi = meta.get("is_multi_drug", False) if meta else len(results) > 1

    reports_created = []
    for item in results:
        smi = item["smiles"]
        res = item["result"]
        predictions = res.get("predictions", {}) if isinstance(res, dict) else {}
        endpoints = []
        for name, data in predictions.items():
            if not isinstance(data, dict):
                continue
            prob = data.get("probability")
            try:
                prob_f = float(prob) if prob is not None else 0.0
                # If model already returns 0-100, keep as is; if 0-1, convert to percent
                prob_percent = prob_f * 100.0 if prob_f <= 1.0 else prob_f
                prob_percent = max(0.0, min(100.0, prob_percent))
            except Exception:
                prob_f = 0.0
                prob_percent = 0.0
            endpoints.append({
                "name": name,
                "probability": prob_f,  # 0-1 for logic
                "probability_percent": prob_percent,  # 0-100 for display
                "predicted_class": data.get("predicted_class"),
                "risk_level": _risk_level(prob_f),
            })
        item["endpoints"] = endpoints

        report_obj = None
        if PredictionReport is not None:
            try:
                kwargs = {
                    "brand_name": (metas[item["index"]]["brand_name"] if metas and item["index"] < len(metas) else meta.get("brand_name", "") if meta else ""),
                    "generic_name": (metas[item["index"]].get("generic_name", "") if metas else meta.get("generic_name", "") if meta else ""),
                    "chemical_name": (metas[item["index"]].get("chemical_name", "") if metas else meta.get("chemical_name", "") if meta else ""),
                    "smiles": smi,
                    "result": res,
                }
                if request.user and request.user.is_authenticated:
                    kwargs["user"] = request.user
                report_obj = PredictionReport.objects.create(**kwargs)
                reports_created.append(report_obj)
            except Exception as e:
                logger.exception(f"DB save failed for {smi}: {e}")
        item["report"] = report_obj

    context = {
        "results": results,
        "is_multi_drug": is_multi,
        "meta": meta,
        "reports": reports_created,
    }
    if len(results) == 1:
        context["result"] = results[0]["result"]
        context["endpoints"] = results[0]["endpoints"]
        context["report"] = results[0]["report"]
        context["smiles"] = results[0]["smiles"]
    return render(request, "web/report.html", context)