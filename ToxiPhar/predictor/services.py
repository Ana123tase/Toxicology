import hashlib
import logging

import pandas as pd
from rdkit import Chem

from .apps import PredictorConfig, ENDPOINTS
from clean_smiles import clean_smiles
from featurize import smiles_to_mol, compute_descriptors, DESCRIPTOR_NAMES, ECFP_COLS, _MORGAN_GEN

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """User-facing error: bad input. Maps to a 4xx in the view."""
    pass


class ModelConfigError(Exception):
    """Server-facing error: model/feature mismatch. Maps to a 5xx."""
    pass


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def featurize_single(mol) -> pd.DataFrame:
    """Builds a single-row feature DataFrame from an already-parsed RDKit mol."""
    descriptor_values = compute_descriptors(mol)

    fp = _MORGAN_GEN.GetFingerprint(mol)
    ecfp_values = [0] * len(ECFP_COLS)
    for bit in fp.GetOnBits():
        ecfp_values[bit] = 1

    row = descriptor_values + ecfp_values
    columns = DESCRIPTOR_NAMES + ECFP_COLS
    return pd.DataFrame([row], columns=columns)


class PredictionService:
    @staticmethod
    def predict(raw_smiles: str) -> dict:
        if not raw_smiles or not isinstance(raw_smiles, str) or not raw_smiles.strip():
            raise PredictionError("SMILES input is empty or not a string.")
        raw_smiles = raw_smiles.strip()

        if Chem.MolFromSmiles(raw_smiles) is None:
            raise PredictionError("Invalid SMILES: could not be parsed by RDKit.")

        try:
            cleaning_result = clean_smiles(raw_smiles)
        except Exception as e:
            logger.warning("clean_smiles failed for input hash=%s: %s", _short_hash(raw_smiles), e)
            raise PredictionError(f"SMILES could not be standardized: {e}") from e

        # clean_smiles() returns a CleaningResult object, not a plain string —
        # pull the actual cleaned SMILES out of it, and surface its own
        # failure reason if it didn't produce one.
        cleaned_smiles = cleaning_result.smiles_clean
        if not cleaned_smiles:
            reason = cleaning_result.review_reason or cleaning_result.status or "unknown reason"
            logger.warning(
                "clean_smiles produced no output for input hash=%s: %s",
                _short_hash(raw_smiles), reason
            )
            raise PredictionError(f"SMILES could not be standardized: {reason}")

        mol = smiles_to_mol(cleaned_smiles)
        if mol is None:
            raise PredictionError("Standardized SMILES could not be parsed by RDKit.")

        features_df = featurize_single(mol)

        results = {}
        for endpoint in ENDPOINTS:
            model = PredictorConfig.models.get(endpoint)
            if model is None:
                raise ModelConfigError(f"No model loaded for endpoint '{endpoint}'.")

            missing_cols = set(model.feature_names_) - set(features_df.columns)
            if missing_cols:
                raise ModelConfigError(
                    f"Feature mismatch for '{endpoint}': model expects columns "
                    f"not present in generated features: {sorted(missing_cols)[:5]}..."
                )

            ordered_df = features_df[model.feature_names_]
            proba = model.predict_proba(ordered_df)[0][1]
            results[endpoint] = {
                "probability": round(float(proba), 4),
                "predicted_class": int(proba >= 0.5),
            }

        return {
            "smiles_input": raw_smiles,
            "smiles_cleaned": cleaned_smiles,
            "predictions": results,
        }