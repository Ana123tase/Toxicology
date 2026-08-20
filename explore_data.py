"""
exprole_data.py
Runs the same inspection checklist across all 5 raw toxicity datasets:
DILI, hERG, CYP3A4, Ames, Teratogenicity.

For each file, this prints: shape, columns, a data preview, detected
SMILES/label columns, class balance, NaN count, and duplicate count.

Column names are auto-detected where possible since the Hugging Face
mirrors may not use TDC's 'Drug'/'Y' convention. If auto-detection
fails for a file, it prints the raw columns so you can hardcode the
right names into CANDIDATE overrides below and re-run.
"""

import pandas as pd

# If auto-detection picks the wrong column for any file, hardcode the
# correct names here, e.g. 'DILI': {'smiles_col': 'smiles', 'label_col': 'label'}
OVERRIDES = {
    # 'DILI': {'smiles_col': None, 'label_col': None},
    # 'hERG': {'smiles_col': None, 'label_col': None},
    # 'CYP3A4': {'smiles_col': None, 'label_col': None},
    # 'Teratogenicity': {'smiles_col': None, 'label_col': None},
}

FILES = {
    'DILI': 'data/DILI_raw.csv',
    'hERG': 'data/hERG_raw.csv',
    'CYP3A4': 'data/CYP3A4_raw.csv',
    'Ames': 'data/Ames_raw.csv',
    'Teratogenicity': 'data/Teratogenicity_raw.csv',
}

SMILES_CANDIDATES = ['Drug', 'SMILES', 'smiles', 'Smiles', 'canonical_smiles',
                      'Canonical_SMILES', 'mol', 'molecule']
LABEL_CANDIDATES = ['Y', 'Label', 'label', 'Overall', 'Class', 'class',
                     'Activity', 'activity', 'target', 'Target']


def detect_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def inspect_file(name, path):
    print(f"\n{'=' * 60}")
    print(f"=== {name}  ({path}) ===")
    print('=' * 60)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"FILE NOT FOUND: {path}  -- check the path/filename.")
        return
    except Exception as e:
        print(f"FAILED TO READ FILE: {e}")
        return

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(df.head(3))

    override = OVERRIDES.get(name, {})
    smiles_col = override.get('smiles_col') or detect_column(df.columns, SMILES_CANDIDATES)
    label_col = override.get('label_col') or detect_column(df.columns, LABEL_CANDIDATES)

    print(f"\nDetected SMILES column: {smiles_col!r}")
    print(f"Detected label column:  {label_col!r}")

    if smiles_col is None or label_col is None:
        print("\n*** Could not auto-detect one or both columns. ***")
        print("*** Look at the columns list above, then add an entry to ***")
        print("*** OVERRIDES at the top of this script and re-run.      ***")
        return

    print(f"\n--- Label balance ({label_col}) ---")
    print(df[label_col].value_counts(dropna=False))

    print(f"\n--- Data quality ({smiles_col}) ---")
    print(f"NaN count: {df[smiles_col].isna().sum()}")
    print(f"Duplicate count: {df[smiles_col].duplicated().sum()}")
    print(f"\nSample SMILES:")
    print(df[smiles_col].dropna().head(5).tolist())


    for name, path in [('DILI','data/DILI_raw.csv'), ('hERG','data/hERG_raw.csv'),
                        ('CYP3A4','data/CYP3A4_raw.csv'), ('Ames','data/Ames_raw.csv'),
                        ('Teratogenicity','data/Teratogenicity_raw.csv')]:
        df = pd.read_csv(path)
        smiles_col = 'SMILES'
        n_multifrag = df[smiles_col].str.contains(r'\.').sum()
        print(f"{name}: {n_multifrag} / {len(df)} SMILES contain '.' (multi-fragment)")


if __name__ == "__main__":
    for name, path in FILES.items():
        inspect_file(name, path)

    print(f"\n{'=' * 60}")
    print("Done. Copy this entire terminal output and paste it back.")
    print('=' * 60)