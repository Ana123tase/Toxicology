"""
rerun_cleaning.py

Re-runs clean_dataset() from clean_smiles.py for all 5 datasets, this
time WITH --target-col supplied. Without it, clean_smiles.py cannot
verify whether a duplicate group's labels agree, so it conservatively
sends every row in every duplicate group to review -- including the
representative row that could otherwise be kept. Supplying the label
column lets it auto-collapse agreeing duplicate groups down to one
representative each, recovering real rows for free.

This calls clean_dataset() directly (importing it) rather than
shelling out via subprocess, so it's simpler to run on Windows and
easier to read what's happening.

USAGE: python rerun_cleaning.py
"""

from clean_smiles import clean_dataset

# (input_path, output_path, smiles_col, target_col)
# target_col is the label column PyTDC/your source used for that
# endpoint -- confirmed from inspect_all.py output earlier:
#   DILI, hERG, CYP3A4, Teratogenicity -> 'Y'
#   Ames -> 'Overall' (different convention, from the Mendeley file)
DATASETS = [
    ("data/DILI_raw.csv", "data/DILI_clean.csv", "SMILES", "Y"),
    ("data/hERG_raw.csv", "data/hERG_clean.csv", "SMILES", "Y"),
    ("data/CYP3A4_raw.csv", "data/CYP3A4_clean.csv", "SMILES", "Y"),
    ("data/Ames_raw.csv", "data/Ames_clean.csv", "SMILES", "Overall"),
    ("data/Teratogenicity_raw.csv", "data/Teratogenicity_clean.csv", "SMILES", "Y"),
]

if __name__ == "__main__":
    results = []
    for input_path, output_path, smiles_col, target_col in DATASETS:
        print(f"\n{'#' * 70}")
        print(f"# {input_path}  (target_col={target_col})")
        print('#' * 70)
        clean_df, review_df = clean_dataset(
            input_path=input_path,
            output_path=output_path,
            smiles_col=smiles_col,
            target_col=target_col,
        )
        results.append((input_path, len(clean_df), len(review_df)))

    print(f"\n{'=' * 70}")
    print("SUMMARY (compare these clean-row counts to your earlier run)")
    print('=' * 70)
    for input_path, n_clean, n_review in results:
        print(f"  {input_path:35s} clean={n_clean:6d}  review={n_review:6d}")