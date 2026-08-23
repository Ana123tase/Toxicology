import pandas as pd
from rdkit import Chem
from .apps import PredictorConfig, ENDPOINTS

# Import directly from the real files in Toxicology\ — added to sys.path in apps.py
from clean_smiles import clean_smiles
from featurize import smiles_to_mol, compute_descriptors, DESCRIPTOR_NAMES, ECFP_COLS, _MORGAN_GEN


class PredictionError(Exception):
    pass


def featurize_single(smiles: str) -> pd.DataFrame:
    """
    Builds a single-row feature DataFrame for one molecule, using the
    exact same descriptor computation and ECFP generator as featurize.py's
    batch pipeline — just skipping the CSV/dataset-file machinery, which
    a live web request doesn't need.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise PredictionError("Invalid SMILES: could not be parsed by RDKit.")

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
        if Chem.MolFromSmiles(raw_smiles) is None:
            raise PredictionError("Invalid SMILES: could not be parsed by RDKit.")

        cleaned = clean_smiles(raw_smiles)
        features_df = featurize_single(cleaned)

        results = {}
        for endpoint in ENDPOINTS:
            model = PredictorConfig.models[endpoint]
            ordered_df = features_df[model.feature_names_]
            proba = model.predict_proba(ordered_df)[0][1]
            results[endpoint] = {
                "probability": round(float(proba), 4),
                "predicted_class": int(proba >= 0.5),
            }

        return {
            "smiles_input": raw_smiles,
            "smiles_cleaned": cleaned,
            "predictions": results,
        }