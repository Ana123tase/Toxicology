from django.apps import AppConfig
from catboost import CatBoostClassifier
import os
import sys

# Make the parent Toxicology folder importable, so we can import
# clean_smiles.py and featurize.py directly from where they actually live —
# no copies, no duplication of your Phase 1 finalized code.
TOXICOLOGY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if TOXICOLOGY_ROOT not in sys.path:
    sys.path.insert(0, TOXICOLOGY_ROOT)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ENDPOINTS = ["DILI", "hERG", "CYP3A4", "Ames", "Teratogenicity"]

class PredictorConfig(AppConfig):
    name = "predictor"
    models = {}

    def ready(self):
        for endpoint in ENDPOINTS:
            path = os.path.join(MODEL_DIR, f"{endpoint}_catboost_production.cbm")
            model = CatBoostClassifier()
            model.load_model(path)
            PredictorConfig.models[endpoint] = model
        print(f"Loaded {len(PredictorConfig.models)} CatBoost models.")