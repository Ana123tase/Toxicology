"""
train_skeleton.py
Skeleton for shared multi-task CatBoost training across the locked endpoint
panel: DILI, hERG, Ames, CYP3A4.

Locked-in decisions this reflects:
  - Model: shared multi-task CatBoost (validated against single-task
    baselines to confirm the multi-task bet actually helps)
  - Retraining: warm-start incremental retraining (not online learning)
  - Features: output of featurize.py (RDKit + ECFP 2048-bit)
  - Split: scaffold_split.py (not random)

This file is a SKELETON: it runs end-to-end on dummy data to prove the
plumbing works, but the actual DILI/hERG/Ames/CYP3A4 dataframes need to
replace the dummy data once the Dataverse download completes.

NOTE on "multi-task": CatBoost's native multi-label mode requires all
tasks to have a label for every row, which won't be true here (DILI,
hERG, Ames, CYP3A4 are different compound sets with mostly non-overlapping
coverage). Two honest options once real data is in:
  (a) Train separate single-task CatBoost models per endpoint (simplest,
      always valid baseline), or
  (b) Train one shared model on the UNION of compounds with multi-label
      targets, masking missing labels per task during evaluation.
This skeleton implements (a) first, since it's what "single-task baseline"
in your locked decisions requires anyway

IMPORTANT: this version featurizes each split via build_model_ready(),
not featurize_dataframe() directly. featurize_dataframe() returns the
FULL audit table -- raw label column, metadata flags (_is_valid,
_is_missing, _inchikey), and PACKED (not dense) ECFP bytes. An earlier
version of this skeleton fed that table straight into CatBoost with
"every column except SMILES is a feature," which meant the label column
itself was included as an input feature (Y predicting Y), on top of
training on packed fingerprint bytes and unresolved duplicate-conflict
rows -- exactly what featurize.py's three-file split exists to prevent.
build_model_ready() is what actually applies those protections, so it's
used here instead.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from featurize import featurize_dataframe, build_model_ready, DESCRIPTOR_NAMES, ECFP_COLS
from scaffold_split import scaffold_split, check_no_scaffold_leakage


# Feature columns are an explicit, known set -- descriptors + dense ECFP bits --
# rather than "every column that isn't the SMILES column." That way any new
# metadata column featurize.py adds later (or the label column itself) can
# never silently end up in the feature matrix by default.
FEATURE_COLS = DESCRIPTOR_NAMES + ECFP_COLS


def train_single_task(
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    train_labels: pd.Series,
    val_labels: pd.Series,
    feature_cols: list,
    task_name: str = "task",
):
    """
    Train one CatBoost binary classifier for a single endpoint.
    Uses CatBoost's native class-imbalance handling (auto_class_weights).
    """
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",  # native imbalance handling
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        train_feat[feature_cols], train_labels,
        eval_set=(val_feat[feature_cols], val_labels),
        use_best_model=True,
    )

    val_probs = model.predict_proba(val_feat[feature_cols])[:, 1]
    auc = roc_auc_score(val_labels, val_probs)
    ap = average_precision_score(val_labels, val_probs)

    print(f"[{task_name}] Val AUC: {auc:.3f} | Val AP: {ap:.3f} "
          f"| n_train={len(train_labels)} n_val={len(val_labels)}")

    return model, {"auc": auc, "ap": ap}


def warm_start_retrain(model: CatBoostClassifier, new_feat: pd.DataFrame,
                        new_labels: pd.Series, feature_cols: list):
    """
    Incremental retraining stub: continue training an existing model on
    newly labeled data, rather than retraining from scratch or doing
    live/online learning. Matches the locked retraining strategy.
    """
    model.fit(
        new_feat[feature_cols], new_labels,
        init_model=model,   # warm start from existing trees
        verbose=False,
    )
    return model


def _featurize_split_for_training(raw_split: pd.DataFrame, smiles_col: str,
                                   label_col: str, task_name: str, split_name: str):
    """
    raw split -> full audit table -> model-ready table.
    Applies the same invalid/missing drop + duplicate-conflict resolution +
    ECFP unpacking that featurize.py's own pipeline applies -- run per split
    so nothing about val/test leaks into how train rows get resolved.
    """
    full_df = featurize_dataframe(raw_split, smiles_col=smiles_col,
                                   dataset_name=f"{task_name}_{split_name}")
    model_df, conflicts_df, label_review_df = build_model_ready(
        full_df, target_col=label_col, dataset_name=f"{task_name}_{split_name}"
    )
    if len(conflicts_df) > 0:
        print(f"[{task_name}/{split_name}] WARNING: {len(conflicts_df)} rows excluded "
              f"as duplicate-label conflicts within this split")
    if len(label_review_df) > 0:
        print(f"[{task_name}/{split_name}] WARNING: {len(label_review_df)} rows excluded "
              f"for missing/unexpected '{label_col}' values within this split")
    return model_df


def run_pipeline_for_endpoint(raw_df: pd.DataFrame, task_name: str,
                               smiles_col: str = "SMILES", label_col: str = "Y"):
    """
    Full path: raw df -> scaffold split -> featurize (model-ready) -> train -> report.
    This is the function that gets called once per real endpoint
    (DILI, hERG, Ames, CYP3A4) once their data is downloaded.
    """
    print(f"\n=== {task_name} ===")

    # 1. Scaffold split BEFORE featurizing -- labels get carried along on the
    #    raw rows, so no separate label-realignment step is needed later.
    train_raw, val_raw, test_raw, scaf_map, train_idx, val_idx, test_idx = scaffold_split(
        raw_df, smiles_col=smiles_col, seed=42
    )
    check_no_scaffold_leakage(scaf_map, train_idx, val_idx, test_idx)

    for split_name, split_df in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        if len(split_df) == 0:
            raise ValueError(
                f"[{task_name}] Scaffold split produced an EMPTY {split_name} set. "
                f"This usually means one or more scaffolds are too large relative to "
                f"your val/test fractions. Inspect scaffold group sizes before proceeding."
            )

    # 2. Featurize each split independently via build_model_ready -- drops
    #    invalid/missing rows, resolves duplicate-SMILES conflicts, unpacks
    #    ECFP to dense bits. Label column comes along naturally as a normal
    #    column, so no SMILES-string label realignment is needed.
    train_feat = _featurize_split_for_training(train_raw, smiles_col, label_col, task_name, "train")
    val_feat = _featurize_split_for_training(val_raw, smiles_col, label_col, task_name, "val")
    test_feat = _featurize_split_for_training(test_raw, smiles_col, label_col, task_name, "test")

    for split_name, feat_df in [("train", train_feat), ("val", val_feat), ("test", test_feat)]:
        if len(feat_df) == 0:
            raise ValueError(
                f"[{task_name}] Featurization left an EMPTY {split_name} set after "
                f"dropping invalid/missing/conflicting rows. Check for bad SMILES or "
                f"label conflicts concentrated in one scaffold group."
            )

    train_labels = train_feat[label_col].reset_index(drop=True)
    val_labels = val_feat[label_col].reset_index(drop=True)
    test_labels = test_feat[label_col].reset_index(drop=True)

    missing_cols = [c for c in FEATURE_COLS if c not in train_feat.columns]
    if missing_cols:
        raise ValueError(f"[{task_name}] Expected feature columns missing from "
                          f"featurized output: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")

    # 3. Train single-task baseline
    model, val_metrics = train_single_task(
        train_feat, val_feat, train_labels, val_labels, FEATURE_COLS, task_name
    )

    # 4. Held-out test evaluation
    test_probs = model.predict_proba(test_feat[FEATURE_COLS])[:, 1]
    test_auc = roc_auc_score(test_labels, test_probs) if test_labels.nunique() > 1 else float("nan")
    print(f"[{task_name}] Test AUC: {test_auc:.3f} (n_test={len(test_labels)})")

    return model, {"val": val_metrics, "test_auc": test_auc}


if __name__ == "__main__":
    # Dummy end-to-end smoke test -- NOT real DILI data.
    # Purpose: prove featurize -> split -> train -> eval all connect correctly.
    # Labels here are pure random noise, so a CORRECT pipeline should land
    # close to AUC ~0.5. If you see AUC near 1.0 on this dummy run, that is
    # now a red flag for a leak, not a sign of a great model -- the previous
    # version of this skeleton reliably showed ~1.0 because Y was literally
    # included as an input feature.
    from rdkit import Chem  # local import, only needed for this validity check

    np.random.seed(0)
    base_rings = [
        "c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "c1ccsc1", "C1CCCCC1",
        "c1ccoc1", "C1CCCC1", "c1cn[nH]c1", "C1CCCCCC1", "c1ccc2[nH]ccc2c1",
        "c1nccnc1", "C1CCOC1", "c1csc(n1)", "C1CCNCC1", "c1ccc2ncccc2c1",
    ]
    substituents = ["", "C", "O", "N", "Cl", "F", "CC", "CO", "CN",
                    "C(=O)O", "OC", "NC", "C(C)C", "CCO", "CCN", "OCC",
                    "C(=O)N", "S(=O)(=O)N", "Br", "CBr"]

    candidates = set()
    for ring in base_rings:
        for sub in substituents:
            candidates.add(ring if sub == "" else f"{sub}{ring}")

    unique_smiles = [s for s in candidates if Chem.MolFromSmiles(s) is not None]
    unique_smiles = unique_smiles[:120]
    n_rejected = len(candidates) - len(unique_smiles)
    if n_rejected:
        print(f"[dummy data] Rejected {n_rejected} invalid synthetic SMILES at generation time.")

    dummy_df = pd.DataFrame({
        "SMILES": unique_smiles,
        "Y": np.random.randint(0, 2, size=len(unique_smiles)),
    })
    print(f"Dummy dataset: {len(dummy_df)} unique molecules")

    model, metrics = run_pipeline_for_endpoint(dummy_df, task_name="DUMMY_ENDPOINT")
    print("\nPipeline ran end-to-end successfully on dummy data.")
    print(f"Metrics: {metrics}")