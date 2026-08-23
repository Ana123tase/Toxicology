# apps.py
import logging
from pathlib import Path

from django.apps import AppConfig
from django.conf import settings
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

ENDPOINTS = ["DILI", "Ames", "hERG", "CYP3A4", "Teratogenicity"]

# Point this at wherever the *_catboost_production.cbm files live in prod
MODULES_DIR = Path(getattr(settings, "TOX_MODELS_DIR", Path(__file__).resolve().parent / "models"))


class PredictorConfig(AppConfig):
    name = "predictor"  
    models: dict = {}
     # NOT "toxicology"

    def ready(self):
        # Django calls ready() once per process. Guard against double-loading
        # under the autoreloader in DEBUG mode.
        if PredictorConfig.models:
            return

        for ep in ENDPOINTS:
            path = MODULES_DIR / f"{ep}_catboost_production.cbm"
            if not path.exists():
                logger.error("Missing model file for endpoint %s: %s", ep, path)
                continue
            m = CatBoostClassifier()
            m.load_model(str(path))
            PredictorConfig.models[ep] = m
            logger.info("Loaded model %s (%d features)", ep, len(m.feature_names_))

        missing = set(ENDPOINTS) - set(PredictorConfig.models)
        if missing:
            # Fail loudly at startup rather than 500ing on first request
            raise RuntimeError(f"Failed to load models for endpoints: {missing}")