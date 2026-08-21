# build_MolFormer_features.py
# Phase 3a: RDKit descriptors + ECFP 2048 + MolFormer 768 = 2828 — production version

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = "molformer.db"
DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]
DATASETS = ["DILI", "hERG", "CYP3A4", "Ames", "Teratogenicity"]

LABEL_COLS = {
    "DILI": "Y",
    "hERG": "Y",
    "CYP3A4": "Y",
    "Ames": "Overall",
    "Teratogenicity": "Y",
}

# Must match benchmark_p1.txt's "RDKit descriptors + ECFP 2048-bit" baseline
# exactly, so MolFormer is the only added variable in the comparison.
DESCRIPTOR_COLS = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "FractionCSP3", "HeavyAtomCount",
]

SQLITE_IN_CHUNK = 500
EMBEDDING_DIM = 768
EXPECTED_ECFP = 2048

MISSING_LOG_DIR = DATA_DIR / "missing_logs"
MISSING_LOG_DIR.mkdir(exist_ok=True)

def fetch_embeddings(conn, smiles_list):
    uniq = list(dict.fromkeys(smiles_list))
    res = {}
    for i in range(0, len(uniq), SQLITE_IN_CHUNK):
        chunk = uniq[i:i+SQLITE_IN_CHUNK]
        ph = ",".join("?" for _ in chunk)
        cur = conn.execute(f"SELECT smiles, vector FROM embeddings WHERE smiles IN ({ph})", chunk)
        for smi, blob in cur.fetchall():
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0]==EMBEDDING_DIM and vec.size!=0 and np.isfinite(vec).all():
                res[smi]=vec
    return res

def check_leakage_all_pairs(name):
    paths = {s: DATA_DIR / f"{name}_{s}.parquet" for s in SPLITS}
    if not all(p.exists() for p in paths.values()):
        return False

    train_set = set(pd.read_parquet(paths["train"], columns=["SMILES"])["SMILES"].tolist())
    val_set = set(pd.read_parquet(paths["val"], columns=["SMILES"])["SMILES"].tolist())
    test_set = set(pd.read_parquet(paths["test"], columns=["SMILES"])["SMILES"].tolist())

    has_leak = False

    ov_train_val = train_set.intersection(val_set)
    ov_train_test = train_set.intersection(test_set)
    ov_val_test = val_set.intersection(test_set)

    if ov_train_val:
        print(f"[{name}] LEAKAGE {len(ov_train_val)} train↔val")
        has_leak=True
    else:
        print(f"[{name}] No leakage train↔val")

    if ov_train_test:
        print(f"[{name}] LEAKAGE {len(ov_train_test)} train↔test")
        has_leak=True
    else:
        print(f"[{name}] No leakage train↔test")

    if ov_val_test:
        print(f"[{name}] LEAKAGE {len(ov_val_test)} val↔test")
        has_leak=True
    else:
        print(f"[{name}] No leakage val↔test")

    return has_leak

def build_one(conn, name, split):
    path = DATA_DIR / f"{name}_{split}.parquet"
    if not path.exists():
        print(f"[{name}/{split}] missing")
        return
    df = pd.read_parquet(path)
    label_col = LABEL_COLS[name]
    if label_col not in df.columns:
        raise ValueError(f"[{name}/{split}] LABEL ERROR: need {label_col}")

    ecfp_cols = [c for c in df.columns if c.startswith("ecfp_")]
    if len(ecfp_cols)!=EXPECTED_ECFP:
        raise ValueError(f"[{name}/{split}] ECFP ERROR: need {EXPECTED_ECFP} got {len(ecfp_cols)}")

    missing_desc = [c for c in DESCRIPTOR_COLS if c not in df.columns]
    if missing_desc:
        raise ValueError(f"[{name}/{split}] DESCRIPTOR ERROR: missing columns {missing_desc}")

    df = df[["SMILES", label_col] + DESCRIPTOR_COLS + ecfp_cols].copy()
    embs = fetch_embeddings(conn, df["SMILES"].tolist())

    miss = ~df["SMILES"].isin(embs.keys())
    if miss.sum()>0:
        df[miss][["SMILES", label_col]].to_csv(MISSING_LOG_DIR / f"{name}_{split}_missing.csv", index=False)
        df = df[~miss].reset_index(drop=True)

    if len(df)==0:
        return
    mf = np.stack([embs[s] for s in df["SMILES"]])
    mf_cols = [f"mf_{i}" for i in range(EMBEDDING_DIM)]
    combined = pd.concat([df.reset_index(drop=True), pd.DataFrame(mf, columns=mf_cols)], axis=1)
    combined.to_parquet(DATA_DIR / f"{name}_{split}_combined.parquet", index=False)
    print(f"[{name}/{split}] {combined.shape[0]}x{combined.shape[1]} -> combined")

def main():
    conn = sqlite3.connect(DB_PATH)
    for name in DATASETS:
        if check_leakage_all_pairs(name):
            print(f"[{name}] Skipping due to leakage\n")
            continue
        for split in SPLITS:
            build_one(conn, name, split)
        print()
    conn.close()

if __name__ == "__main__":
    main()