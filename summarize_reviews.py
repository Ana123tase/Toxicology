"""
summarize_reviews.py

Produces a compact, paste-back-friendly summary of everything still
sitting in each *_review.csv after cleaning + metal recovery: how many
rows fall into each leftover category, and a few representative
examples per category so there's enough detail to actually assess
what's being excluded, without dumping hundreds of raw rows.

Categories you should expect to see (all are LEFTOVER, i.e. NOT in
your *_clean.csv):
  mixture              genuine multi-component mixture/co-crystal,
                        can't tell which part is the real test compound
  metal_complex         bonded/complex metal, or a metal not on the
                        "genuinely simple salt" whitelist (chelates,
                        organometallics, etc.) -- see recover_metal_ions.py
  ambiguous_fragment    a fragment that's neither a recognized salt nor
                        clearly organic
  no_parent             every fragment was a salt/solvent -- nothing
                        organic left at all
  invalid_smiles        SMILES string didn't parse
  duplicate             same structure as another row -- watch the
                        "CONFLICT" flag specifically: that means the
                        SAME molecule has DIFFERENT labels across rows,
                        which is a real data-quality problem worth
                        knowing about, not just redundancy

Also writes a full, unabridged CSV per dataset
(*_review_full_detail.csv) in case you want to dig into a specific
category later -- the terminal output here is a summary, not the
complete picture.

USAGE: python summarize_reviews.py
"""

import pandas as pd

DATASETS = [
    ("DILI", "data/DILI_review.csv"),
    ("hERG", "data/hERG_review.csv"),
    ("CYP3A4", "data/CYP3A4_review.csv"),
    ("Ames", "data/Ames_review.csv"),
    ("Teratogenicity", "data/Teratogenicity_review.csv"),
]

N_EXAMPLES_PER_CATEGORY = 3


def summarize_one(name: str, path: str):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"=== {name} ===\n  {path} not found, skipping.\n")
        return None

    print(f"{'=' * 65}")
    print(f"=== {name}  ({path}) ===")
    print('=' * 65)
    print(f"Total leftover rows: {len(df)}\n")

    if len(df) == 0:
        print("Nothing left to review for this dataset.\n")
        return df

    status_counts = df["cleaning_status"].value_counts()
    print("Breakdown by category:")
    for status, count in status_counts.items():
        print(f"  {status:25s} {count}")

    # Flag duplicate CONFLICTS specifically -- same structure, different
    # labels. This is more important than plain (non-conflicting)
    # duplicates, since it points at a real data-quality issue rather
    # than just redundant rows.
    if "duplicate" in status_counts.index and "review_reason" in df.columns:
        dup_rows = df[df["cleaning_status"] == "duplicate"]
        n_conflict = dup_rows["review_reason"].str.contains(
            "conflicting", case=False, na=False
        ).sum()
        if n_conflict > 0:
            print(f"\n  *** {n_conflict} of those duplicates are LABEL CONFLICTS ***")
            print(f"  *** (same molecule, different label values -- worth flagging) ***")

    print(f"\nRepresentative examples ({N_EXAMPLES_PER_CATEGORY} per category):")
    for status in status_counts.index:
        subset = df[df["cleaning_status"] == status].head(N_EXAMPLES_PER_CATEGORY)
        print(f"\n  --- {status} ---")
        for _, row in subset.iterrows():
            smiles = row.get("SMILES_raw", row.get("SMILES", "?"))
            reason = row.get("review_reason", "")
            # Truncate long SMILES so the summary stays readable
            smiles_display = smiles if len(str(smiles)) <= 70 else str(smiles)[:67] + "..."
            print(f"    {smiles_display}")
            if reason and pd.notna(reason):
                print(f"      reason: {reason}")

    print()
    return df


if __name__ == "__main__":
    all_dfs = []
    for name, path in DATASETS:
        df = summarize_one(name, path)
        if df is not None and len(df) > 0:
            df = df.copy()
            df["dataset"] = name
            all_dfs.append(df)
            out_path = path.replace("_review.csv", "_review_full_detail.csv")
            df.to_csv(out_path, index=False)

    print(f"{'=' * 65}")
    print("Full unabridged detail per dataset saved to *_review_full_detail.csv")
    print("(the summary above is what to paste back for analysis)")
    print('=' * 65)