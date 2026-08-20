"""
scaffold_split.py
Splits by Bemis-Murcko scaffold, not randomly.

Fixed for real pipeline:
- SMILES default, not Drug
- scaffold computed once, reused for leakage check
- seed actually used (shuffle equal-size groups)
- diversity diagnostic before split
- wired to the *_features_model.parquet files featurize.py actually writes
- leakage check now actually runs (previous version defined it but never called it,
  and scaffold_split() didn't even return the index lists it needs)
"""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str, include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except Exception:
        return ""


def compute_scaffold_map(df: pd.DataFrame, smiles_col: str = "SMILES"):
    """Compute scaffolds once, return map + grouping."""
    scaffold_to_indices = defaultdict(list)
    idx_to_scaffold = {}

    for idx, smi in zip(df.index, df[smiles_col]):
        scaf = get_scaffold(smi) if pd.notna(smi) and str(smi).strip() != "" else ""
        scaffold_to_indices[scaf].append(idx)
        idx_to_scaffold[idx] = scaf

    return scaffold_to_indices, idx_to_scaffold


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
):
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    rng = np.random.RandomState(seed)

    scaffold_to_indices, idx_to_scaffold = compute_scaffold_map(df, smiles_col)

    # Diversity diagnostic before split
    n_total = len(df)
    n_scaffolds = len(scaffold_to_indices)
    if n_scaffolds == 0:
        raise ValueError("No scaffolds computed -- input frame is empty")
    sizes = sorted([len(v) for v in scaffold_to_indices.values()], reverse=True)
    print(f" [scaffold] {n_total} mols | {n_scaffolds} distinct scaffolds | largest group {sizes[0]} | top5 {sizes[:5]}")
    print(f" [scaffold] singleton scaffolds: {sum(1 for s in sizes if s == 1)} "
          f"({sum(1 for s in sizes if s == 1) / n_scaffolds:.1%})")
    if n_scaffolds < 10:
        print(f" [scaffold] WARNING: very low scaffold diversity - split may fail")

    # Shuffle scaffold groups before sorting, then stable sort by size descending.
    # Equal-size groups appear in different order per seed -- this is what makes
    # the seed actually matter.
    scaffold_groups = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups = sorted(scaffold_groups, key=lambda idxs: len(idxs), reverse=True)

    n_train_target = int(frac_train * n_total)
    n_val_target = int(frac_val * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    # Return the original index lists too -- without these, no caller can ever
    # run the leakage check below (this is what the previous version got wrong).
    return train_df, val_df, test_df, idx_to_scaffold, train_idx, val_idx, test_idx


def check_no_scaffold_leakage(idx_to_scaffold_map, train_idx_orig, val_idx_orig, test_idx_orig):
    """Reuse already-computed scaffolds, don't recompute per molecule."""
    def scaffolds_from_orig_indices(orig_indices):
        return {idx_to_scaffold_map[i] for i in orig_indices}

    train_scaf = scaffolds_from_orig_indices(train_idx_orig)
    val_scaf = scaffolds_from_orig_indices(val_idx_orig)
    test_scaf = scaffolds_from_orig_indices(test_idx_orig)

    assert not (train_scaf & val_scaf), f"Leak train/val: {train_scaf & val_scaf}"
    assert not (train_scaf & test_scaf), f"Leak train/test: {train_scaf & test_scaf}"
    assert not (val_scaf & test_scaf), f"Leak val/test: {val_scaf & test_scaf}"
    print(" [scaffold] No scaffold leakage - OK")


def report_split_diagnostics(train_df, val_df, test_df, target_col, dataset_name,
                              frac_train, frac_val, frac_test,
                              min_eval_size=30, frac_tolerance=0.03):
    """
    Two things a clean 'No scaffold leakage' print can still hide:
      1. A val/test split that's badly skewed or single-class on the label
         -- scaffold and label aren't correlated by design, so this can
         happen even with a perfectly correct split.
      2. A val/test set so small in absolute terms that any metric computed
         on it is mostly noise, even if the *fraction* looks right.
    Neither is a bug in the split -- both are things worth knowing before
    trusting a downstream AUC number.
    """
    n_total = len(train_df) + len(val_df) + len(test_df)

    for split_name, split_df, target_frac in [
        ("train", train_df, frac_train), ("val", val_df, frac_val), ("test", test_df, frac_test)
    ]:
        actual_frac = len(split_df) / n_total if n_total else 0.0
        if abs(actual_frac - target_frac) > frac_tolerance:
            print(f" [scaffold] NOTE [{dataset_name}/{split_name}]: actual size "
                  f"{actual_frac:.1%} vs target {target_frac:.1%} -- scaffold groups "
                  f"didn't pack evenly for this dataset")

        if split_name in ("val", "test") and 0 < len(split_df) < min_eval_size:
            print(f" [scaffold] WARNING [{dataset_name}/{split_name}]: only {len(split_df)} "
                  f"molecules -- metrics computed on this split will be high-variance")

        if target_col and target_col in split_df.columns and len(split_df) > 0:
            counts = split_df[target_col].value_counts(dropna=True)
            n_classes = counts.shape[0]
            if n_classes < 2:
                print(f" [scaffold] WARNING [{dataset_name}/{split_name}]: only "
                      f"{n_classes} distinct label value(s) present ({dict(counts)}) -- "
                      f"AUC/AP on this split will be undefined")
            else:
                frac_str = ", ".join(f"{k}={v} ({v / counts.sum():.1%})" for k, v in counts.items())
                print(f" [scaffold] [{dataset_name}/{split_name}] label balance: {frac_str}")


def split_file(input_path: str, output_dir: str, smiles_col: str = "SMILES",
               target_col: str = None,
               frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):
    print(f"\n=== {input_path} ===")
    df = pd.read_parquet(input_path)

    # *_features_model.parquet (the intended input) has already had invalid/missing
    # rows removed, so it has no _is_valid column -- that branch only fires if
    # someone points this at *_features_full.parquet instead.
    if "_is_valid" in df.columns:
        valid_df = df[df["_is_valid"]].copy().reset_index(drop=True)
        invalid_df = df[~df["_is_valid"]].copy()
        print(f" Input: {len(df)} rows ({len(valid_df)} valid, {len(invalid_df)} invalid)")
    else:
        valid_df = df
        invalid_df = pd.DataFrame()

    train_df, val_df, test_df, scaf_map, train_idx, val_idx, test_idx = scaffold_split(
        valid_df, smiles_col=smiles_col, frac_train=frac_train,
        frac_val=frac_val, frac_test=frac_test, seed=seed,
    )

    print(f" Split: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    # Actually run the leakage check -- previously defined but never called.
    check_no_scaffold_leakage(scaf_map, train_idx, val_idx, test_idx)

    report_split_diagnostics(train_df, val_df, test_df, target_col,
                              Path(input_path).stem, frac_train, frac_val, frac_test)

    # Save
    out_base = Path(output_dir) / Path(input_path).stem.replace("_features_model", "").replace("_features_full", "")
    train_df.to_parquet(f"{out_base}_train.parquet", index=False)
    val_df.to_parquet(f"{out_base}_val.parquet", index=False)
    test_df.to_parquet(f"{out_base}_test.parquet", index=False)
    if len(invalid_df) > 0:
        invalid_df.to_parquet(f"{out_base}_invalid.parquet", index=False)

    print(f" Saved: {out_base}_train/val/test.parquet")
    return train_df, val_df, test_df


# Wired to what featurize.py actually writes: the model-ready file, which has
# invalid/missing rows already dropped, duplicate conflicts already resolved,
# and ECFP already unpacked to dense columns -- exactly what a split for
# training should be built from. Do NOT point this at *_features_full.parquet:
# that file still has unresolved duplicate-conflict rows and packed fingerprints.
# target_col paired per dataset (matches featurize.py's DATASETS list) so
# report_split_diagnostics can check label balance per split, not just
# scaffold membership.
DATASETS = [
    ("data/DILI_features_model.parquet", "Y"),
    ("data/hERG_features_model.parquet", "Y"),
    ("data/CYP3A4_features_model.parquet", "Y"),
    ("data/Ames_features_model.parquet", "Overall"),
    ("data/Teratogenicity_features_model.parquet", "Y"),
]

if __name__ == "__main__":
    # Quick verification that seed matters now
    print("=== Seed test ===")
    dummy = pd.DataFrame({
        "SMILES": ["c1ccccc1", "c1ccccc1C", "c1ccccc1O", "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C",
                   "CCO", "CCCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(cc1)c1ccccc1"],
        "Y": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    })
    t1, _, _, _, _, _, _ = scaffold_split(dummy, seed=42)
    t2, _, _, _, _, _, _ = scaffold_split(dummy, seed=99)
    print(f" seed 42 train: {t1['SMILES'].tolist()}")
    print(f" seed 99 train: {t2['SMILES'].tolist()}")
    print(f" seed changes split: {t1['SMILES'].tolist() != t2['SMILES'].tolist()} (should be True for equal-size groups)")

    for path, target_col in DATASETS:
        try:
            split_file(path, "data/", smiles_col="SMILES", target_col=target_col, seed=42)
        except FileNotFoundError:
            print(f" Not found, skipping: {path}")
        except Exception as e:
            print(f" FAILED {path}: {e}")