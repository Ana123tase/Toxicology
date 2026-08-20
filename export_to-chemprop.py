"""
export_to-chemprop.py

Converts the scaffold-split, model-ready parquet files produced by
scaffold_split.py (data/{name}_train.parquet, _val.parquet,
_test.parquet) into the plain SMILES + target CSVs that chemprop's
CLI expects.

Why this exists: scaffold_split.py's output still carries every ECFP/
RDKit feature column featurize.py generated for CatBoost. Chemprop
builds its own learned representation directly from the SMILES graph
and only wants two columns -- feeding it the CatBoost feature columns
would be at best ignored, at worst misread as extra targets.

This script does NOT re-split, re-clean, or touch the SMILES/labels
in any way -- it only reads, subsets columns, drops target-missing
rows, and writes. The train/val/test membership is exactly what
scaffold_split.py already decided, so CatBoost and D-MPNN stay
comparable on the same held-out molecules.

USAGE: python export_chemprop.py
"""

import pandas as pd
from pathlib import Path

# (dataset_name, smiles_col, target_col)
# Mirrors scaffold_split.py's DATASETS list -- same names, same target
# columns (Ames uses 'Overall', everything else uses 'Y').
DATASETS = [
    ("DILI", "SMILES", "Y"),
    ("hERG", "SMILES", "Y"),
    ("CYP3A4", "SMILES", "Y"),
    ("Ames", "SMILES", "Overall"),
    ("Teratogenicity", "SMILES", "Y"),
]

DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]


def export_split(name: str, split: str, smiles_col: str, target_col: str) -> dict:
    """
    Read one {name}_{split}.parquet, subset to [smiles_col, target_col],
    drop rows with missing target, write {name}_{split}_chemprop.csv.

    Returns a summary dict for the final report table. Never raises on
    a missing file -- prints a clear skip message instead, since not
    every dataset necessarily has all three splits populated (e.g. a
    tiny scaffold group distribution could in principle leave val or
    test empty for some dataset).
    """
    in_path = DATA_DIR / f"{name}_{split}.parquet"
    out_path = DATA_DIR / f"{name}_{split}_chemprop.csv"

    if not in_path.exists():
        print(f"  [{name}/{split}] SKIPPED - file not found: {in_path}")
        return {"dataset": name, "split": split, "status": "missing_file"}

    df = pd.read_parquet(in_path)

    if smiles_col not in df.columns:
        print(f"  [{name}/{split}] SKIPPED - column '{smiles_col}' not found. "
              f"Available: {list(df.columns)[:10]}...")
        return {"dataset": name, "split": split, "status": "missing_smiles_col"}

    if target_col not in df.columns:
        print(f"  [{name}/{split}] SKIPPED - column '{target_col}' not found. "
              f"Available: {list(df.columns)[:10]}...")
        return {"dataset": name, "split": split, "status": "missing_target_col"}

    n_before = len(df)
    subset = df[[smiles_col, target_col]].copy()

    subset = subset.dropna(subset=[target_col])
    n_dropped = n_before - len(subset)

    subset.to_csv(out_path, index=False)

    value_counts = subset[target_col].value_counts(dropna=True).to_dict()

    print(f"  [{name}/{split}] {len(subset)} rows written "
          f"({n_dropped} dropped for missing target) -> {out_path}")
    print(f"    label balance: {value_counts}")

    return {
        "dataset": name,
        "split": split,
        "status": "ok",
        "n_rows": len(subset),
        "n_dropped": n_dropped,
        "label_balance": value_counts,
    }


if __name__ == "__main__":
    all_results = []

    for name, smiles_col, target_col in DATASETS:
        print(f"\n=== {name} (target_col={target_col}) ===")
        for split in SPLITS:
            result = export_split(name, split, smiles_col, target_col)
            all_results.append(result)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)
    for r in all_results:
        if r["status"] == "ok":
            print(f"  {r['dataset']:16s} {r['split']:6s} "
                  f"rows={r['n_rows']:6d}  dropped={r['n_dropped']:4d}  "
                  f"labels={r['label_balance']}")
        else:
            print(f"  {r['dataset']:16s} {r['split']:6s} STATUS={r['status']}")