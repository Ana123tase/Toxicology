"""
train_MolFormer_eval.py
FINAL EVALUATION-ONLY TRAINER — CatBoost + MoLFormer Fusion

Like train_catboost.py but for MoLFormer Fusion.

- Uses YOUR trusted splits: data/{ENDPOINT}_{split}.parquet (SMILES, Y, ecfp_*)
- Loads molformer.db 30,909 embeddings (768 dim)
- Builds Fusion: ECFP 2048 + MoLFormer 768 = 2816
- Trains on train only, evaluates on val/test
- Does NOT train production model on 100% — evaluation only
- Supports warm-start incremental retraining (init_model)

Usage:
    python train_MolFormer_eval.py
    python train_MolFormer_eval.py --dataset DILI --warm-start --increment-iterations 100
    python train_MolFormer_eval.py --dataset hERG --full-retrain
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

DATA_DIR = Path(os.environ.get("ADMET_DATA_DIR", "data"))
MODULES_DIR = Path(os.environ.get("ADMET_MODULES_DIR", "modules"))
DB_PATH = Path("molformer.db")

INCREMENTS_DIR = MODULES_DIR / "increments"
REGISTRY_PATH = MODULES_DIR / "registry_molformer.jsonl"
KEEP_INCREMENTS = int(os.environ.get("ADMET_KEEP_INCREMENTS", "20"))

DATASETS = ["DILI", "hERG", "CYP3A4", "Ames", "Teratogenicity"]

TREE_COUNTS = {
    "DILI": 232,
    "hERG": 499,
    "CYP3A4": 442,
    "Ames": 479,
    "Teratogenicity": 108
}

BASE_PARAMS = dict(
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    eval_metric="AUC",
    random_seed=42,
)

FULL_TRAIN_ITERATIONS = None # we use locked counts
DEFAULT_INCREMENT_ITERATIONS = 100
EARLY_STOPPING_ROUNDS = 50

log = logging.getLogger("molformer_train")

class TrainingError(Exception):
    pass

def _configure_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

def _file_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _atomic_write_bytes(path: Path, write_fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def _atomic_write_text(path: Path, text: str):
    _atomic_write_bytes(path, lambda tmp: tmp.write_text(text))

def _append_registry(entry: dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_molformer_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT smiles, vector FROM embeddings")
    emb_dict = {}
    for smi, blob in cur.fetchall():
        emb_dict[smi] = np.frombuffer(blob, dtype=np.float32)
    conn.close()
    log.info(f"Loaded {len(emb_dict)} MolFormer embeddings")
    return emb_dict

def load_trusted_splits_eval(dataset_name: str, emb_dict):
    dfs = {}
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{dataset_name}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        dfs[split] = pd.read_parquet(path)

    full = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    smiles_col = "SMILES"
    label_col = "Y"
    ecfp_cols = [c for c in full.columns if c.startswith("ecfp_")]

    def get_emb(smi):
        v = emb_dict.get(smi)
        if v is None:
            v = emb_dict.get(str(smi).strip())
        return v

    # Build per-split arrays
    result = {}
    for split_name in ["train", "val", "test"]:
        df = dfs[split_name].copy()
        df["molformer"] = df[smiles_col].apply(get_emb)
        before = len(df)
        df = df[df["molformer"].notna()].copy()
        after = len(df)
        if after < before:
            log.warning(f"[{dataset_name}/{split_name}] dropped {before-after} without MolFormer emb")

        X_ecfp = df[ecfp_cols].values.astype(np.float32)
        X_molformer = np.vstack(df["molformer"].values) if len(df)>0 else np.empty((0,768))
        X_fusion = np.hstack([X_ecfp, X_molformer]).astype(np.float32)
        y = df[label_col].values

        result[split_name] = (X_ecfp, X_fusion, y, df)

    return result, ecfp_cols

def _safe_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred) if len(np.unique(y_true))>1 else float("nan")

def _safe_ap(y_true, y_pred):
    return average_precision_score(y_true, y_pred) if len(np.unique(y_true))>1 else float("nan")

def train_one(dataset_name: str, warm_start=False, increment_iterations=None, tag=None, emb_dict=None):
    if emb_dict is None:
        emb_dict = load_molformer_db()

    MODULES_DIR.mkdir(exist_ok=True)

    splits, ecfp_cols = load_trusted_splits_eval(dataset_name, emb_dict)
    X_ecfp_train, X_fusion_train, y_train, _ = splits["train"]
    X_ecfp_val, X_fusion_val, y_val, _ = splits["val"]
    X_ecfp_test, X_fusion_test, y_test, _ = splits["test"]

    if len(np.unique(y_train)) < 2:
        raise TrainingError(f"[{dataset_name}] train has <2 classes")

    log.info(f"=== {dataset_name}{' (warm-start)' if warm_start else ''} ===")
    log.info(f"Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)} | Fusion dim {X_fusion_train.shape[1]}")

    model_path = MODULES_DIR / f"{dataset_name}_molformer.cbm"
    metrics_path = MODULES_DIR / f"{dataset_name}_molformer_metrics.json"

    init_model = None
    prior_metrics = None
    if warm_start and model_path.exists():
        if metrics_path.exists():
            prior_metrics = json.loads(metrics_path.read_text())
        init_model = CatBoostClassifier()
        init_model.load_model(str(model_path))
        log.info(f"[{dataset_name}] warm-start from {model_path} best={prior_metrics.get('best_iteration') if prior_metrics else '?'}")

    n_iter = TREE_COUNTS[dataset_name]
    iterations = (increment_iterations or DEFAULT_INCREMENT_ITERATIONS) if init_model else n_iter

    model = CatBoostClassifier(
        iterations=iterations,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=100,
        **BASE_PARAMS
    )

    fit_kwargs = dict(eval_set=(X_fusion_val, y_val), use_best_model=True)
    if init_model:
        fit_kwargs["init_model"] = init_model

    model.fit(X_fusion_train, y_train, **fit_kwargs)

    # ECFP-only baseline for delta (fair comparison)
    model_ecfp = CatBoostClassifier(iterations=n_iter, **BASE_PARAMS, verbose=False)
    model_ecfp.fit(X_ecfp_train, y_train)

    val_pred_fusion = model.predict_proba(X_fusion_val)[:,1]
    val_pred_ecfp = model_ecfp.predict_proba(X_ecfp_val)[:,1]
    test_pred_fusion = model.predict_proba(X_fusion_test)[:,1] if len(y_test)>0 else None
    test_pred_ecfp = model_ecfp.predict_proba(X_ecfp_test)[:,1] if len(y_test)>0 else None

    val_auc_fusion = _safe_auc(y_val, val_pred_fusion)
    val_auc_ecfp = _safe_auc(y_val, val_pred_ecfp)
    test_auc_fusion = _safe_auc(y_test, test_pred_fusion) if test_pred_fusion is not None else float("nan")

    log.info(f"[{dataset_name}] Val ECFP AUC {val_auc_ecfp:.3f} | Fusion AUC {val_auc_fusion:.3f} Delta {val_auc_fusion-val_auc_ecfp:+.3f} | Test Fusion AUC {test_auc_fusion:.3f}")

    # Snapshot before overwrite
    if model_path.exists():
        INCREMENTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(model_path, INCREMENTS_DIR / f"{dataset_name}_{stamp}_molformer.cbm")
        if metrics_path.exists():
            shutil.copy2(metrics_path, INCREMENTS_DIR / f"{dataset_name}_{stamp}_molformer_metrics.json")

    _atomic_write_bytes(model_path, lambda tmp: model.save_model(str(tmp)))

    train_hash = _file_sha256(DATA_DIR / f"{dataset_name}_train.parquet")
    metrics = {
        "dataset": dataset_name,
        "target_col": "Y",
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "val_auc_ecfp": val_auc_ecfp,
        "val_auc_fusion": val_auc_fusion,
        "delta_auc": float(val_auc_fusion - val_auc_ecfp),
        "test_auc_fusion": test_auc_fusion,
        "best_iteration": int(model.get_best_iteration()),
        "fusion_dim": int(X_fusion_train.shape[1]),
        "ecfp_dim": 2048,
        "molformer_dim": 768,
        "warm_started": init_model is not None,
        "train_data_sha256": train_hash,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
    }
    _atomic_write_text(metrics_path, json.dumps(metrics, indent=2))
    _append_registry(metrics)

    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", dest="datasets", default=None, help="Dataset name")
    parser.add_argument("--warm-start", action="store_true", help="Continue from saved model")
    parser.add_argument("--full-retrain", action="store_true", help="Force from scratch")
    parser.add_argument("--increment-iterations", type=int, default=None, help="Extra trees on warm-start")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.warm_start and args.full_retrain:
        parser.error("--warm-start and --full-retrain mutually exclusive")

    emb_dict = load_molformer_db()

    selected = DATASETS if not args.datasets else args.datasets
    failures = []
    all_metrics = []

    for name in selected:
        try:
            m = train_one(name, warm_start=args.warm_start and not args.full_retrain,
                          increment_iterations=args.increment_iterations, tag=args.tag, emb_dict=emb_dict)
            all_metrics.append(m)
        except Exception as e:
            log.error(f"Skipping {name} - {e}\n{traceback.format_exc()}")
            failures.append(name)

    # Write benchmark_p3.txt summary
    if all_metrics:
        with open("benchmark_p3.txt", "w") as f:
            f.write("Toxicology ADMET Phase 3 — MolFormer Fusion Benchmark (EVAL ONLY)\n")
            f.write("="*70+"\n")
            for m in all_metrics:
                f.write(f"{m['dataset']:20s} train={m['n_train']} val={m['n_val']} test={m['n_test']} "
                        f"ECFP AUC={m['val_auc_ecfp']:.3f} Fusion AUC={m['val_auc_fusion']:.3f} "
                        f"Delta={m['delta_auc']:+.4f} Test AUC={m['test_auc_fusion']:.3f}\n")
            f.write("\nDecision (keep Fusion if delta > 0.01):\n")
            for m in all_metrics:
                dec = "KEEP MOLFORMER FUSION" if m['delta_auc']>0.01 else "KEEP ECFP-ONLY"
                f.write(f" {m['dataset']:20s} delta={m['delta_auc']:+.4f} -> {dec}\n")

    if failures:
        log.error(f"Completed with failures: {failures}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())