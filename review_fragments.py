import pandas as pd
from pathlib import Path


DATASETS = [
    "DILI",
    "CYP3A4",
    "hERG",
    "Ames",
    "Teratogenicity",
]


def create_review_file(name):
    input_file = Path(f"data/{name}_clean.csv")
    output_file = Path(f"data/{name}_review.csv")

    df = pd.read_csv(input_file)

    review = df[
        df["cleaning_status"] == "review_required"
    ].copy()

    if review.empty:
        print(f"{name}: no rows require review.")
        return

    # Keep useful columns first.
    preferred = [
        "SMILES_raw",
        "SMILES",
        "cleaning_status",
        "num_fragments",
        "num_removed",
        "removed_fragments",
        "inchikey",
    ]

    # Add any remaining columns afterward.
    columns = [c for c in preferred if c in review.columns]
    columns += [c for c in review.columns if c not in columns]

    review = review[columns]

    review.to_csv(output_file, index=False)

    print(
        f"{name}: {len(review)} rows requiring review "
        f"-> {output_file}"
    )


if __name__ == "__main__":
    print("=== Fragment Review Files ===")

    for dataset in DATASETS:
        create_review_file(dataset)

    print("\nDone.")