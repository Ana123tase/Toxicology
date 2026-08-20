from rdkit import Chem
import pandas as pd
from pathlib import Path

sdf_file = Path("DIT_model/Data/Train 2D.sdf")
output_file = Path("data/Teratogenicity_raw.csv")

supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)

rows = []

for i, mol in enumerate(supplier):

    if mol is None:
        print(f"Skipping molecule {i}")
        continue

    smiles = Chem.MolToSmiles(mol)

    # Repository definition:
    # first 67 = positive
    # last 45  = negative
    label = 1 if i < 67 else 0

    rows.append({
        "SMILES": smiles,
        "Y": label
    })

df = pd.DataFrame(rows)

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("Shape:", df.shape)
print("\nLabels:")
print(df["Y"].value_counts().sort_index())

print("\nMissing values:")
print(df.isna().sum())

print("\nSaved:")
print(output_file.resolve())