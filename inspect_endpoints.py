"""
inspect_endpoints.py

Quick inspection of your 5 endpoint parquet files (output of featurize.py).
Run this from your toxicology project folder and paste the printed output
back into the chat -- it tells us exactly what data types/schemas we're
building the pipeline around, so nothing downstream breaks on a wrong
assumption about label type.

Usage:
    python inspect_endpoints.py --data_dir data
"""

import argparse
import glob
import os

import pandas as pd


def inspect_file(path: str) -> None:
    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [ERROR] Could not read file: {e}")
        return

    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Try to find likely SMILES column
    smiles_cols = [c for c in df.columns if "smiles" in c.lower()]
    print(f"  Likely SMILES column(s): {smiles_cols}")

    # Try to find likely label column(s) -- anything not smiles/id/features
    id_like = [c for c in df.columns if c.lower() in ("id", "mol_id", "index", "name")]

    # Prefixes that indicate a fingerprint/embedding/descriptor bit column,
    # not a label. Add to this tuple if your featurize.py uses other prefixes.
    feature_prefixes = ("feat", "embed", "fp", "desc", "mf_", "molformer", "ecfp", "bit_", "maccs")
    feature_like = [c for c in df.columns if c.lower().startswith(feature_prefixes)]

    candidate_label_cols = [
        c for c in df.columns
        if c not in smiles_cols and c not in id_like and c not in feature_like
    ]

    print(f"  Column type summary:")
    print(f"      SMILES-like columns   : {len(smiles_cols)}  -> {smiles_cols}")
    print(f"      ID-like columns       : {len(id_like)}  -> {id_like}")
    print(f"      Feature/fp columns    : {len(feature_like)}  (suppressed from detail -- these are fingerprint bits, not labels)")
    print(f"      Remaining (candidate label) columns: {len(candidate_label_cols)}  -> {candidate_label_cols}")
    print()

    if len(candidate_label_cols) > 15:
        print(f"  [WARNING] {len(candidate_label_cols)} candidate label columns found -- that's a lot.")
        print(f"  Your feature_prefixes filter in this script may be missing a prefix your featurize.py uses.")
        print(f"  Showing summary only, not full detail, to avoid flooding output.")
        for col in candidate_label_cols:
            print(f"      - {col}  (dtype: {df[col].dtype}, n_unique: {df[col].nunique(dropna=True)})")
        print()
        return

    for col in candidate_label_cols:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_null = series.isnull().sum()
        dtype = series.dtype

        print(f"  --- Column: '{col}' ---")
        print(f"      dtype: {dtype}")
        print(f"      n_unique: {n_unique}")
        print(f"      n_null: {n_null}")

        is_numeric = pd.api.types.is_numeric_dtype(series)

        if n_unique <= 10:
            print(f"      value_counts:\n{series.value_counts(dropna=False).to_string()}")
        elif is_numeric:
            print(f"      min: {series.min()}, max: {series.max()}, mean: {series.mean():.4f}")
        else:
            print(f"      [non-numeric, high-cardinality -- likely an ID/text column, not a label]")
            print(f"      sample values: {series.dropna().unique()[:5].tolist()}")
        print()

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Folder containing your endpoint parquet files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
        help="Glob pattern to match endpoint files",
    )
    args = parser.parse_args()

    search_path = os.path.join(args.data_dir, args.pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        print(f"No parquet files found matching: {search_path}")
        print("Pass the correct folder with --data_dir, e.g.:")
        print("    python inspect_endpoints.py --data_dir data")
        return

    print(f"Found {len(files)} parquet file(s):")
    for f in files:
        print(f"  - {f}")
    print()

    for f in files:
        inspect_file(f)


if __name__ == "__main__":
    main()