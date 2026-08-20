from pathlib import Path
from datasets import load_dataset
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Generic Hugging Face downloader
# ============================================================

def download_hf_dataset(repo, output_file, name):
    print()
    print("=" * 60)
    print(f"Downloading {name}")
    print("=" * 60)

    try:
        dataset = load_dataset(repo)

        print(dataset)

        # Most of these datasets have a single train split
        if "train" in dataset:
            df = dataset["train"].to_pandas()
        else:
            split = list(dataset.keys())[0]
            df = dataset[split].to_pandas()

        print(f"Rows: {len(df):,}")
        print(f"Columns: {df.columns.tolist()}")

        output_path = DATA_DIR / output_file

        df.to_csv(output_path, index=False)

        print(f"Saved to:")
        print(output_path)

        return df

    except Exception as e:
        print(f"FAILED: {name}")
        print(e)

        return None


# ============================================================
# DILI
# ============================================================

def download_dili():

    return download_hf_dataset(
        "jablonkagroup/drug_induced_liver_injury",
        "DILI_raw.csv",
        "DILI"
    )


# ============================================================
# CYP3A4
# ============================================================

def download_cyp3a4():

    return download_hf_dataset(
        "scikit-fingerprints/TDC_cyp3a4_veith",
        "CYP3A4_raw.csv",
        "CYP3A4 Veith"
    )


# ============================================================
# hERG
# ============================================================

def download_herg():

    return download_hf_dataset(
        "scikit-fingerprints/TDC_herg_karim",
        "hERG_raw.csv",
        "hERG Karim"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TOXICOLOGY DATASET DOWNLOADER")
    print("=" * 60)

    print(f"\nData directory:")
    print(DATA_DIR)

    download_dili()

    download_cyp3a4()

    download_herg()

    print()
    print("=" * 60)
    print("CURRENT DATA DIRECTORY")
    print("=" * 60)

    for file in DATA_DIR.glob("*.csv"):
        print(file.name)

    print()
    print("Finished.")


if __name__ == "__main__":
    main()