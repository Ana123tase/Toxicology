"""
featurize.py
Converts cleaned SMILES CSVs into model-ready feature tables.

THIS VERSION FIXES THREE RISKS FROM THE PREVIOUS ONE BY CONSTRUCTION,
NOT BY DOCUMENTATION -- i.e. it's no longer possible to accidentally
train on bad data just by forgetting a filter step, because the bad
data never reaches the file a training script would read from.

Every dataset now produces up to FOUR files:

  {name}_features_full.parquet
      Everything: every input row (valid, invalid, missing), every
      flag column, ECFP stored PACKED (256 bytes/molecule -- 8x
      smaller than dense storage). This is the audit trail. Nothing
      is ever deleted from this file. NOT for training -- the packed
      columns are not usable as model features as-is.

  {name}_features_model.parquet
      Only what's safe to train on: invalid/missing rows removed,
      rows with a missing or unexpected target_col value removed (see
      LABEL VALIDATION below), duplicate-conflict groups removed (see
      DUPLICATE HANDLING below), ECFP already UNPACKED to dense
      ecfp_0..ecfp_2047 columns. A training script can load this file
      and go straight to building a feature matrix.

  {name}_duplicate_conflicts.csv
      Only written if conflicts were found: groups of rows that are
      the exact same molecule (by InChIKey) but disagree on the
      label. Never auto-resolved -- same "never guess" policy as
      clean_smiles.py. Written here so you can review and decide.

  {name}_label_review.csv
      Only written if found: rows with a chemically VALID molecule
      but a target_col value that's missing or outside the expected
      set (default {0, 1}). This is NOT auto-mapped to any class --
      e.g. a value like -1 might mean "inconclusive," "not tested,"
      or something dataset-specific, and guessing wrong would
      silently corrupt training. Excluded from the model file until
      you've confirmed what the value means and, if appropriate,
      remapped it upstream before re-running this script.

LABEL VALIDATION (only applied when building the _model file):
  - build_model_ready(..., valid_labels=(0, 1)) by default. Any row
    with a valid molecule whose target_col value is null OR not in
    valid_labels is excluded from the model file and written to
    _label_review.csv instead. Pass valid_labels=None to disable this
    check for a genuinely non-binary target.

DUPLICATE HANDLING (only applied when building the _model file):
  - Rows sharing an InChIKey are a duplicate group.
  - If every row in a group agrees on target_col: keep ONE
    representative (lowest original row index), drop the rest.
    Nothing lost -- they were redundant, not contradictory.
  - If a group disagrees on target_col: the ENTIRE group is excluded
    from the _model file and written to _duplicate_conflicts.csv
    instead. This is the case most likely to appear because of the
    metal-recovery merge step possibly reintroducing a molecule that
    already existed in the clean file under a different SMILES
    spelling.

Requires: pip install pyarrow rdkit pandas
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


def compute_descriptors(mol) -> list:
    out = []
    for name in DESCRIPTOR_NAMES:
        try:
            out.append(RDKIT_DESCRIPTORS[name](mol))
        except Exception:
            out.append(np.nan)
    return out


def unpack_ecfp_matrix(packed_matrix: np.ndarray) -> np.ndarray:
    """(n, 256) packed uint8 -> (n, 2048) dense 0/1 uint8. Verified
    bit-exact against RDKit's own dense fingerprint output."""
    return np.unpackbits(packed_matrix.astype(np.uint8), axis=1).astype(np.uint8)


def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "SMILES",
                         progress_every: int = 2000,
                         dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Returns the FULL audit-trail table: every input row, every flag,
    ECFP packed. This is the building block both output files are
    made from -- see featurize_file() for how the _model file is
    derived from this.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found. Available: {list(df.columns)}")

    collisions = set(DESCRIPTOR_NAMES + PACKED_COLS + ECFP_COLS).intersection(df.columns)
    if collisions:
        raise ValueError(f"Input already contains feature names {collisions}. Rename first.")

    n = len(df)
    missing_mask = df[smiles_col].isna() | (df[smiles_col].astype(str).str.strip() == "")

    fp_packed = np.zeros((n, PACKED_NBYTES), dtype=np.uint8)
    desc_matrix = np.full((n, len(DESCRIPTOR_NAMES)), np.nan, dtype=np.float32)
    is_valid = np.zeros(n, dtype=bool)
    inchi_keys = [None] * n
    n_invalid = 0
    t0 = time.time()

    for pos in range(n):
        if missing_mask.iloc[pos]:
            continue
        mol = smiles_to_mol(df[smiles_col].iloc[pos])
        if mol is None:
            n_invalid += 1
            continue

        fp = _MORGAN_GEN.GetFingerprint(mol)
        on_bits = fp.GetOnBits()
        packed_row = fp_packed[pos]
        for bit in on_bits:
            packed_row[bit // 8] |= 1 << (7 - (bit % 8))

        desc_matrix[pos] = compute_descriptors(mol)
        is_valid[pos] = True
        try:
            inchi_keys[pos] = inchi.MolToInchiKey(mol)
        except Exception:
            inchi_keys[pos] = None

        if progress_every and (pos + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  [{dataset_name}] {pos + 1}/{n} ({(pos + 1) / elapsed:.0f} rows/sec)")

    n_missing = int(missing_mask.sum())
    print(f"  [{dataset_name}] missing={n_missing}  invalid={n_invalid}  valid={int(is_valid.sum())}")

    is_dup = pd.Series(inchi_keys).duplicated(keep=False) & pd.Series(inchi_keys).notna()
    if is_dup.any():
        print(f"  [{dataset_name}] {int(is_dup.sum())} rows share an InChIKey with another row "
              f"(canonical duplicates) -- resolved when building the _model file")

    meta_df = df.reset_index(drop=True).copy()
    meta_df["_is_valid"] = is_valid
    meta_df["_is_missing"] = missing_mask.values
    meta_df["_is_duplicate_canonical"] = is_dup.values
    meta_df["_inchikey"] = inchi_keys

    desc_df = pd.DataFrame(desc_matrix, columns=DESCRIPTOR_NAMES)
    packed_df = pd.DataFrame(fp_packed, columns=PACKED_COLS, dtype=np.uint8)

    return pd.concat([meta_df, desc_df, packed_df], axis=1)


def build_model_ready(full_df: pd.DataFrame, target_col: str,
                       dataset_name: str = "dataset",
                       valid_labels=(0, 1)):
    """
    Derives the _model file from the _full table:
      1. Drop invalid/missing-SMILES rows (never enters the model file at all).
      2. Drop rows whose target_col is null or outside valid_labels --
         written to a separate label_review frame for manual review
         rather than silently included or dropped without a trace.
         Pass valid_labels=None to skip this check for a genuinely
         non-binary target.
      3. Resolve duplicate groups: agreeing groups -> keep 1 representative;
         conflicting groups -> excluded here, returned separately for review.
      4. Unpack ECFP to dense 0/1 columns -- packed columns never appear
         in this output, so they can't be fed to a model by mistake.

    Returns (model_df, conflicts_df, label_review_df).
    """
    has_valid_mol = full_df["_is_valid"]

    label_notna = full_df[target_col].notna()
    if valid_labels is not None:
        label_ok = label_notna & full_df[target_col].isin(valid_labels)
    else:
        label_ok = label_notna

    valid_df = full_df[has_valid_mol & label_ok].copy()
    n_dropped_invalid = int((~has_valid_mol).sum())

    label_review_df = full_df[has_valid_mol & ~label_ok].copy()
    if len(label_review_df) > 0:
        seen_values = full_df.loc[has_valid_mol & ~label_ok, target_col].value_counts(dropna=False).to_dict()
        print(f"  [{dataset_name}] *** {len(label_review_df)} rows have a valid molecule but a "
              f"missing/unexpected '{target_col}' value (expected one of {valid_labels}) -- "
              f"excluded from model file, written to label review CSV. "
              f"Value breakdown: {seen_values} ***")

    # -- Resolve duplicate groups ---------------------------------------
    keep_mask = pd.Series(True, index=valid_df.index)
    conflict_rows = []

    has_inchikey = valid_df["_inchikey"].notna()
    groups = valid_df[has_inchikey].groupby("_inchikey").groups

    n_conflict_groups = 0
    n_conflict_rows = 0
    n_redundant_dropped = 0

    for inchikey, idx in groups.items():
        if len(idx) < 2:
            continue
        idx = sorted(idx)
        labels = valid_df.loc[idx, target_col].dropna().unique()
        if len(labels) > 1:
            # Genuine conflict: same molecule, different label. Exclude
            # the whole group from the model file, log it for review.
            keep_mask.loc[idx] = False
            conflict_rows.append(valid_df.loc[idx])
            n_conflict_groups += 1
            n_conflict_rows += len(idx)
        else:
            # Agreeing duplicates: keep the first (lowest original index),
            # drop the rest -- they're redundant, not contradictory. Safe
            # to just take idx[0] here because valid_df was already
            # filtered to rows with a non-null, in-range label above, so
            # every row in this group already has a usable label.
            keep_mask.loc[idx[1:]] = False
            n_redundant_dropped += len(idx) - 1

    resolved_df = valid_df[keep_mask].copy()

    if n_redundant_dropped:
        print(f"  [{dataset_name}] dropped {n_redundant_dropped} redundant duplicate rows "
              f"(kept 1 representative per agreeing group)")
    if n_conflict_groups:
        print(f"  [{dataset_name}] *** {n_conflict_groups} duplicate groups "
              f"({n_conflict_rows} rows) have CONFLICTING labels -- excluded from "
              f"model file, written to conflicts CSV for manual review ***")

    # -- Unpack ECFP to dense columns, drop packed + internal flag cols --
    packed_matrix = resolved_df[PACKED_COLS].to_numpy(dtype=np.uint8)
    dense = unpack_ecfp_matrix(packed_matrix)
    ecfp_df = pd.DataFrame(dense, columns=ECFP_COLS, index=resolved_df.index)

    keep_meta_cols = [c for c in resolved_df.columns
                       if c not in PACKED_COLS and c not in ["_is_valid", "_is_missing"]]
    model_df = pd.concat([resolved_df[keep_meta_cols], ecfp_df], axis=1).reset_index(drop=True)

    conflicts_df = pd.concat(conflict_rows, axis=0).reset_index(drop=True) if conflict_rows else pd.DataFrame()

    print(f"  [{dataset_name}] model-ready rows: {len(model_df)} "
          f"(dropped {n_dropped_invalid} invalid/missing molecules, "
          f"{len(label_review_df)} missing/unexpected labels, "
          f"{n_redundant_dropped} redundant dupes, {n_conflict_rows} conflict rows)")

    return model_df, conflicts_df, label_review_df


def featurize_file(input_path: str, output_dir: str, dataset_name: str,
                    target_col: str, smiles_col: str = "SMILES",
                    valid_labels=(0, 1)):
    print(f"\n=== {input_path}  (target_col={target_col}) ===")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Input: {len(df)} rows x {len(df.columns)} cols")

    full_df = featurize_dataframe(df, smiles_col=smiles_col, dataset_name=dataset_name)
    model_df, conflicts_df, label_review_df = build_model_ready(
        full_df, target_col=target_col, dataset_name=dataset_name, valid_labels=valid_labels
    )

    full_path = f"{output_dir}/{dataset_name}_features_full.parquet"
    model_path = f"{output_dir}/{dataset_name}_features_model.parquet"
    conflicts_path = f"{output_dir}/{dataset_name}_duplicate_conflicts.csv"
    label_review_path = f"{output_dir}/{dataset_name}_label_review.csv"

    full_df.to_parquet(full_path, index=False)
    model_df.to_parquet(model_path, index=False)
    if len(conflicts_df) > 0:
        conflicts_df.to_csv(conflicts_path, index=False)
    if len(label_review_df) > 0:
        label_review_df.to_csv(label_review_path, index=False)

    meta = {
        "input_file": input_path, "target_col": target_col,
        "valid_labels": list(valid_labels) if valid_labels is not None else None,
        "rdkit_version": get_rdkit_version(), "python_version": platform.python_version(),
        "ecfp_radius": ECFP_RADIUS, "ecfp_nbits": ECFP_NBITS,
        "n_input": len(df), "n_full": len(full_df), "n_model_ready": len(model_df),
        "n_duplicate_conflict_rows": len(conflicts_df),
        "n_label_review_rows": len(label_review_df),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(f"{output_dir}/{dataset_name}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  Saved: {full_path}  (audit trail, packed, NOT for training)")
    print(f"  Saved: {model_path}  (safe to train on directly)")
    if len(conflicts_df) > 0:
        print(f"  Saved: {conflicts_path}  (needs your review)")
    if len(label_review_df) > 0:
        print(f"  Saved: {label_review_path}  (needs your review)")

    return model_df


DATASETS = [
    ("data/DILI_clean.csv", "DILI", "Y"),
    ("data/hERG_clean.csv", "hERG", "Y"),
    ("data/CYP3A4_clean.csv", "CYP3A4", "Y"),
    ("data/Ames_clean.csv", "Ames", "Overall"),
    ("data/Teratogenicity_clean.csv", "Teratogenicity", "Y"),
]

if __name__ == "__main__":
    failures = []
    for input_path, dataset_name, target_col in DATASETS:
        try:
            featurize_file(input_path, "data", dataset_name, target_col, smiles_col="SMILES")
        except FileNotFoundError:
            msg = f"Not found: {input_path}"
            print(msg)
            failures.append(msg)
        except Exception as e:
            msg = f"FAILED {input_path}: {e}"
            print(msg)
            failures.append(msg)

    print("\n=== SUMMARY ===")
    if failures:
        for f in failures:
            print(" -", f)
    else:
        print("All datasets processed. Use *_features_model.parquet for training --")
        print("Check each *_label_review.csv (if any) before trusting that dataset's split.")