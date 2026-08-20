"""
merge_recovered_metals.py

Automatically merges the AUTO_RECOVERABLE rows from each
*_metal_recovery.csv back into the corresponding *_clean.csv, AND
removes those same rows from *_review.csv, so the review file only
ever shows rows that still genuinely need your attention.

Why this is safe to automate (unlike the general "inspect everything"
advice): the judgment call already happened inside
recover_metal_ions.py's classify_metal_row() function -- a row only
gets labeled 'auto_recoverable' if it passed a specific, narrow check
(exactly one organic parent, or identical copies of one, with only
whitelisted simple free-ion metals removed). There's no remaining
chemistry guesswork for a human to catch here; that's exactly why this
category exists separately from 'needs_manual_review'.

Rows labeled 'needs_manual_review' are NEVER touched by this script --
those still require actual judgment this script can't make, and they
are exactly what's left behind in *_review.csv after this runs.

This prints a short, plain-English summary of what got added (counts
+ a few example before/after rows) so you have visibility without
needing to read a CSV yourself.

USAGE: python merge_recovered_metals.py
"""

import pandas as pd

# (clean_csv, review_csv, metal_recovery_csv)
DATASETS = [
    ("data/DILI_clean.csv", "data/DILI_review.csv", "data/DILI_metal_recovery.csv"),
    ("data/hERG_clean.csv", "data/hERG_review.csv", "data/hERG_metal_recovery.csv"),
    ("data/CYP3A4_clean.csv", "data/CYP3A4_review.csv", "data/CYP3A4_metal_recovery.csv"),
    ("data/Ames_clean.csv", "data/Ames_review.csv", "data/Ames_metal_recovery.csv"),
    ("data/Teratogenicity_clean.csv", "data/Teratogenicity_review.csv", "data/Teratogenicity_metal_recovery.csv"),
]


def merge_one(clean_path: str, review_path: str, recovery_path: str):
    try:
        clean_df = pd.read_csv(clean_path)
    except FileNotFoundError:
        print(f"{clean_path}: not found, skipping.\n")
        return

    try:
        review_df = pd.read_csv(review_path)
    except FileNotFoundError:
        print(f"{review_path}: not found, skipping.\n")
        return

    try:
        recovery_df = pd.read_csv(recovery_path)
    except FileNotFoundError:
        print(f"{recovery_path}: not found (no metal_complex rows for this dataset), skipping.\n")
        return

    to_add = recovery_df[recovery_df["recommendation"] == "auto_recoverable"].copy()

    if len(to_add) == 0:
        print(f"{clean_path}: 0 auto-recoverable rows, nothing to merge.\n")
        return

    n_clean_before = len(clean_df)
    n_review_before = len(review_df)

    # Use the recovered (metal-stripped) SMILES as the SMILES going forward.
    to_add["SMILES"] = to_add["recovered_SMILES"]
    to_add["cleaning_status"] = "metal_stripped_recovered"  # traceable, distinct label

    # -- 1. Add the recovered rows to the clean dataset -----------------
    common_cols = [c for c in clean_df.columns if c in to_add.columns]
    to_add_aligned = to_add[common_cols]
    merged_clean = pd.concat([clean_df, to_add_aligned], ignore_index=True)
    merged_clean.to_csv(clean_path, index=False)

    # -- 2. Remove those exact rows from the review file -----------------
    # Matched by SMILES_raw, since that's the original, untouched value
    # every row in review_df and recovery_df both carry. This only
    # removes rows that were BOTH metal_complex AND auto_recoverable --
    # everything else (mixtures, ambiguous fragments, needs_manual_review
    # metal rows, etc.) stays in review_df untouched.
    recovered_raw_smiles = set(to_add["SMILES_raw"])
    still_needs_review = review_df[~review_df["SMILES_raw"].isin(recovered_raw_smiles)].copy()
    still_needs_review.to_csv(review_path, index=False)

    n_clean_after = len(merged_clean)
    n_review_after = len(still_needs_review)

    print(f"{clean_path}")
    print(f"  Clean rows:  {n_clean_before} -> {n_clean_after}  (+{n_clean_after - n_clean_before})")
    print(f"  Review rows: {n_review_before} -> {n_review_after}  (-{n_review_before - n_review_after})")
    print(f"  Example of what was recovered:")
    for _, row in to_add.head(2).iterrows():
        print(f"    raw:       {row['SMILES_raw']}")
        print(f"    recovered: {row['SMILES']}   (reason: {row['recovery_reason']})")
    print()


if __name__ == "__main__":
    for clean_path, review_path, recovery_path in DATASETS:
        merge_one(clean_path, review_path, recovery_path)

    print("Done. Each *_clean.csv now includes its auto-recoverable metal-salt")
    print("rows (tagged cleaning_status='metal_stripped_recovered' for traceability),")
    print("and each *_review.csv now contains ONLY rows that still genuinely need")
    print("your manual attention -- recovered rows have been removed from it.")