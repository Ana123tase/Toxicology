"""
featurize.py - Production
"""

import time
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')

ECFP_RADIUS = 2
ECFP_NBITS = 2048
PACKED_NBYTES = ECFP_NBITS // 8

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_NBITS)
try:
    _MORGAN_GEN_STEREO = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_NBITS, includeChirality=True, useBondTypes=True)
except TypeError:
    _MORGAN_GEN_STEREO = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_NBITS, includeChirality=True)

RDKIT_DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}
DESCRIPTOR_NAMES = list(RDKIT_DESCRIPTORS.keys())
PACKED_COLS = [f"ecfp_packed_{i}" for i in range(PACKED_NBYTES)]
ECFP_COLS = [f"ecfp_{i}" for i in range(ECFP_NBITS)]
PACKED_COLS_STEREO = [f"ecfp_stereo_packed_{i}" for i in range(PACKED_NBYTES)]
ECFP_COLS_STEREO = [f"ecfp_stereo_{i}" for i in range(ECFP_NBITS)]

MODEL_STEREO_FEATURES = [
    "n_stereocenters",
    "n_assigned_stereocenters",
    "n_unassigned_stereocenters",
    "n_ez_bonds",
    "n_assigned_ez",
    "n_unassigned_ez",
    "has_chiral",
    "has_ez",
    "has_stereo",
    "has_unassigned_stereo",
    "has_unassigned_ez",
    "is_stereoisomer_defined",
    "has_isotope",
    "n_isotope_atoms",
]

# Also handle legacy names from clean_smiles.py
LEGACY_STEREO_COLS = [
    "n_stereocenters",
    "n_unassigned_stereocenters",
    "n_ez_stereocenters",
    "n_assigned_ez",
    "n_unassigned_ez",
    "has_chiral",
    "has_ez",
    "has_stereo",
    "has_unassigned_stereo",
    "is_stereoisomer_defined",
    "has_isotope",
    "n_isotope_atoms",
    "isotope_labels",
    "has_tautomer",
    "stereo_signature",
    "SMILES_isomeric",
]

def get_rdkit_version():
    try:
        from rdkit import rdBase
        return rdBase.rdkitVersion
    except Exception:
        return "unknown"

def smiles_to_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)

def compute_descriptors(mol):
    out = []
    for name in DESCRIPTOR_NAMES:
        try:
            out.append(RDKIT_DESCRIPTORS[name](mol))
        except Exception:
            out.append(np.nan)
    return out

def unpack_ecfp_matrix(packed_matrix: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed_matrix.astype(np.uint8), axis=1).astype(np.uint8)

def _ez_counts_simple(mol: Chem.Mol) -> tuple[int, int]:
    n_a = 0
    n_u = 0
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True, flagPossibleStereoCenters=True)
    except Exception:
        pass
    potential = set()
    try:
        from rdkit.Chem import FindPotentialStereo
        for info in FindPotentialStereo(mol):
            if "Bond" in str(getattr(info, 'type', '')):
                try:
                    potential.add(int(info.centeredOn))
                except Exception:
                    pass
    except Exception:
        pass
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        st = bond.GetStereo()
        if st == Chem.BondStereo.STEREOE or st == Chem.BondStereo.STEREOZ:
            n_a += 1
        elif st == Chem.BondStereo.STEREOANY:
            n_u += 1
        elif bond.GetIdx() in potential and st == Chem.BondStereo.STEREONONE:
            n_u += 1
    return n_a, n_u

def _isotope_audit(mol: Chem.Mol):
    labs = [f"{a.GetIsotope()}{a.GetSymbol()}_{a.GetIdx()}" for a in mol.GetAtoms() if a.GetIsotope() != 0]
    return (len(labs) > 0), len(labs), ";".join(labs)

def _tautomer_audit(mol: Chem.Mol) -> bool:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        enum = rdMolStandardize.TautomerEnumerator()
        return len(list(enum.Enumerate(mol))) > 1
    except Exception:
        return False

def _stereo_audit(mol: Chem.Mol) -> dict:
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        n_total = len(centers)
        n_unassigned = sum(1 for _, lab in centers if lab == "?")
        n_assigned = n_total - n_unassigned
    except Exception:
        n_total = n_assigned = n_unassigned = 0
    try:
        n_a_ez, n_u_ez = _ez_counts_simple(mol)
        n_ez = n_a_ez + n_u_ez
    except Exception:
        n_a_ez = n_u_ez = n_ez = 0
    try:
        has_iso, n_iso, iso_labs = _isotope_audit(mol)
    except Exception:
        has_iso = False; n_iso = 0; iso_labs = ""
    try:
        has_tau = _tautomer_audit(mol)
    except Exception:
        has_tau = False
    return {
        "n_stereocenters": n_total,
        "n_assigned_stereocenters": n_assigned,
        "n_unassigned_stereocenters": n_unassigned,
        "n_ez_bonds": n_ez,
        "n_assigned_ez": n_a_ez,
        "n_unassigned_ez": n_u_ez,
        "has_chiral": bool(n_total > 0),
        "has_ez": bool(n_ez > 0),
        "has_stereo": bool(n_total > 0 or n_ez > 0),
        "has_unassigned_stereo": bool(n_unassigned > 0 or n_u_ez > 0),
        "has_unassigned_ez": bool(n_u_ez > 0),
        "is_stereoisomer_defined": bool((n_total > 0 or n_ez > 0) and not (n_unassigned > 0 or n_u_ez > 0)),
        "has_isotope": bool(has_iso),
        "n_isotope_atoms": int(n_iso),
        "has_tautomer": bool(has_tau),
        "isotope_labels": iso_labs,
        # legacy compat
        "n_ez_stereocenters": n_ez,
    }

def _validate_stereo_fingerprints():
    tests = [("C[C@H](O)CC", "C[C@@H](O)CC", "R/S"), ("C/C=C/C", "C/C=C\\C", "E/Z")]
    for smi1, smi2, label in tests:
        try:
            m1 = Chem.MolFromSmiles(smi1)
            m2 = Chem.MolFromSmiles(smi2)
            if m1 is None or m2 is None:
                continue
            fp1 = _MORGAN_GEN_STEREO.GetFingerprint(m1)
            fp2 = _MORGAN_GEN_STEREO.GetFingerprint(m2)
            from rdkit import DataStructs
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            diff = len(set(fp1.GetOnBits()).symmetric_difference(set(fp2.GetOnBits())))
            if sim >= 1.0 or diff == 0:
                print(f"  [VALIDATION WARNING] {label} NOT distinguished sim={sim}")
                return False
            else:
                print(f"  [VALIDATION] {label} OK sim={sim:.3f} diff={diff}")
        except Exception as e:
            print(f"  [VALIDATION] {label} error {e}")
            return False
    return True

def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "SMILES", progress_every: int = 2000, dataset_name: str = "dataset") -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found")
    # Avoid duplicate column collision with existing audit cols from clean_smiles
    # We will compute fresh audit but reuse if already present to avoid dup
    n = len(df)
    missing_mask = df[smiles_col].isna() | (df[smiles_col].astype(str).str.strip() == "")
    fp_packed = np.zeros((n, PACKED_NBYTES), dtype=np.uint8)
    fp_packed_stereo = np.zeros((n, PACKED_NBYTES), dtype=np.uint8)
    desc_matrix = np.full((n, len(DESCRIPTOR_NAMES)), np.nan, dtype=np.float32)
    is_valid = np.zeros(n, dtype=bool)
    inchikey_stereo = [None] * n
    inchikey_conn = [None] * n
    stereo_rows = []
    n_invalid = 0

    for pos in range(n):
        if missing_mask.iloc[pos]:
            stereo_rows.append({k: None for k in MODEL_STEREO_FEATURES + ["has_tautomer", "isotope_labels", "n_ez_stereocenters"]})
            continue
        mol = smiles_to_mol(df[smiles_col].iloc[pos])
        if mol is None:
            n_invalid += 1
            stereo_rows.append({k: None for k in MODEL_STEREO_FEATURES + ["has_tautomer", "isotope_labels", "n_ez_stereocenters"]})
            continue
        fp = _MORGAN_GEN.GetFingerprint(mol)
        for b in fp.GetOnBits():
            fp_packed[pos, b // 8] |= 1 << (7 - (b % 8))
        try:
            fps = _MORGAN_GEN_STEREO.GetFingerprint(mol)
            for b in fps.GetOnBits():
                fp_packed_stereo[pos, b // 8] |= 1 << (7 - (b % 8))
        except Exception:
            pass
        desc_matrix[pos] = compute_descriptors(mol)
        is_valid[pos] = True
        try:
            ik = inchi.MolToInchiKey(mol)
            inchikey_stereo[pos] = ik
            inchikey_conn[pos] = ik.split("-")[0] if ik else None
        except Exception:
            pass
        stereo_rows.append(_stereo_audit(mol))

    print(f"  [{dataset_name}] missing={int(missing_mask.sum())} invalid={n_invalid} valid={int(is_valid.sum())}")

    # Start with original df columns but drop any that would collide with our computed cols to avoid duplicate
    # Keep original audit cols from clean_smiles for traceability, but we will overwrite with fresh calc
    cols_to_drop_from_original = set(MODEL_STEREO_FEATURES + ["has_tautomer", "isotope_labels", "n_ez_stereocenters"]).intersection(set(df.columns))
    if cols_to_drop_from_original:
        print(f"  [{dataset_name}] dropping {len(cols_to_drop_from_original)} existing audit cols from input to avoid duplicate: {cols_to_drop_from_original}")
        df_base = df.drop(columns=list(cols_to_drop_from_original))
    else:
        df_base = df

    meta_df = df_base.reset_index(drop=True).copy()
    meta_df["_is_valid"] = is_valid
    meta_df["_is_missing"] = missing_mask.values
    meta_df["_inchikey_stereo"] = inchikey_stereo
    meta_df["_inchikey_connectivity"] = inchikey_conn
    meta_df["_inchikey"] = inchikey_stereo

    stereo_df = pd.DataFrame(stereo_rows)
    desc_df = pd.DataFrame(desc_matrix, columns=DESCRIPTOR_NAMES)
    packed_df = pd.DataFrame(fp_packed, columns=PACKED_COLS, dtype=np.uint8)
    packed_stereo_df = pd.DataFrame(fp_packed_stereo, columns=PACKED_COLS_STEREO, dtype=np.uint8)

    full = pd.concat([meta_df, stereo_df, desc_df, packed_df, packed_stereo_df], axis=1)
    # Final safety: drop duplicate columns (keep first)
    full = full.loc[:, ~full.columns.duplicated()]
    return full

def build_model_ready(full_df: pd.DataFrame, target_col: str, dataset_name: str = "dataset", valid_labels=(0, 1)):
    has_valid = full_df["_is_valid"]
    label_notna = full_df[target_col].notna()
    label_ok = label_notna & full_df[target_col].isin(valid_labels) if valid_labels is not None else label_notna
    valid_df = full_df[has_valid & label_ok].copy()
    label_review_df = full_df[has_valid & ~label_ok].copy()
    keep_mask = pd.Series(True, index=valid_df.index)
    conflict_rows = []
    has_ikey = valid_df["_inchikey_stereo"].notna()
    groups = valid_df[has_ikey].groupby("_inchikey_stereo").groups
    n_conflict_rows = 0; n_redundant = 0
    for ik, idx in groups.items():
        if len(idx) < 2:
            continue
        idx = sorted(idx)
        labels = valid_df.loc[idx, target_col].dropna().unique()
        if len(labels) > 1:
            keep_mask.loc[idx] = False
            conflict_rows.append(valid_df.loc[idx])
            n_conflict_rows += len(idx)
        else:
            keep_mask.loc[idx[1:]] = False
            n_redundant += len(idx) - 1
    resolved_df = valid_df[keep_mask].copy()
    dense = unpack_ecfp_matrix(resolved_df[PACKED_COLS].to_numpy(dtype=np.uint8))
    dense_s = unpack_ecfp_matrix(resolved_df[PACKED_COLS_STEREO].to_numpy(dtype=np.uint8))
    ecfp_df = pd.DataFrame(dense, columns=ECFP_COLS, index=resolved_df.index)
    ecfp_s_df = pd.DataFrame(dense_s, columns=ECFP_COLS_STEREO, index=resolved_df.index)
    # ONLY CHANGE: bring back SMILES for D-MPNN/CatBoost consistency - SMILES is metadata, not used as numeric feature
    keep_from_resolved = [c for c in [target_col, "SMILES"] + DESCRIPTOR_NAMES + MODEL_STEREO_FEATURES if c in resolved_df.columns]
    model_df = pd.concat([
        resolved_df[keep_from_resolved].reset_index(drop=True),
        ecfp_df.reset_index(drop=True),
        ecfp_s_df.reset_index(drop=True)
    ], axis=1)
    # Safety dedup
    model_df = model_df.loc[:, ~model_df.columns.duplicated()]
    conflicts_df = pd.concat(conflict_rows, axis=0).reset_index(drop=True) if conflict_rows else pd.DataFrame()
    print(f"  [{dataset_name}] model-ready rows: {len(model_df)}")
    return model_df, conflicts_df, label_review_df

def featurize_file(input_path: str, output_dir: str, dataset_name: str, target_col: str, smiles_col: str = "SMILES", valid_labels=(0, 1)):
    print(f"\n=== {input_path} (target={target_col}) ===")
    _validate_stereo_fingerprints()
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Input: {len(df)} rows x {len(df.columns)} cols")
    full_df = featurize_dataframe(df, smiles_col=smiles_col, dataset_name=dataset_name)
    model_df, conflicts_df, label_review_df = build_model_ready(full_df, target_col=target_col, dataset_name=dataset_name, valid_labels=valid_labels)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    full_path = f"{output_dir}/{dataset_name}_features_full.parquet"
    model_path = f"{output_dir}/{dataset_name}_features_model.parquet"
    # Final dedup before parquet write
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    model_df = model_df.loc[:, ~model_df.columns.duplicated()]
    full_df.to_parquet(full_path, index=False)
    model_df.to_parquet(model_path, index=False)
    if len(conflicts_df) > 0:
        conflicts_df.to_csv(f"{output_dir}/{dataset_name}_duplicate_conflicts.csv", index=False)
    if len(label_review_df) > 0:
        label_review_df.to_csv(f"{output_dir}/{dataset_name}_label_review.csv", index=False)
    meta = {
        "model_features": {
            "group1_connectivity": DESCRIPTOR_NAMES + ECFP_COLS,
            "group2_stereo_stable": MODEL_STEREO_FEATURES + ECFP_COLS_STEREO,
        },
        "audit_only": ["has_tautomer", "isotope_labels"],
        "n_input": len(df), "n_full": len(full_df), "n_model": len(model_df),
    }
    Path(f"{output_dir}/{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved: {full_path}")
    print(f"  Saved: {model_path} (PRODUCTION-SAFE)")
    return model_df

DATASETS = [
    ("data/DILI_clean.csv", "DILI", "Y"),
    ("data/hERG_clean.csv", "hERG", "Y"),
    ("data/CYP3A4_clean.csv", "CYP3A4", "Y"),
    ("data/Ames_clean.csv", "Ames", "Overall"),
    ("data/Teratogenicity_clean.csv", "Teratogenicity", "Y"),
]

if __name__ == "__main__":
    for input_path, dataset_name, target_col in DATASETS:
        try:
            featurize_file(input_path, "data", dataset_name, target_col, smiles_col="SMILES")
        except FileNotFoundError:
            print(f"Not found: {input_path}")