# train_mlp_molformer.py
"""
Phase 3 isolation test: MLP (small feedforward net) on MoLFormer 768-dim
embeddings ONLY (no ECFP, no RDKit descriptors). Same scaffold splits as
prior Phase 3 benchmarks. Does not touch production modules/ or prior
benchmark files.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_DIR = Path("data")
OUT_DIR = Path("modules_p3_mlp")
BENCHMARK_OUT = Path("benchmark_p3_mlp.txt")

DATASETS = [
    ("DILI", "Y"),
    ("hERG", "Y"),
    ("CYP3A4", "Y"),
    ("Ames", "Overall"),
    ("Teratogenicity", "Y"),
]

RANDOM_SEED = 42

log = logging.getLogger("admet_mlp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def _load_split(dataset_name, split_name):
    path = DATA_DIR / f"{dataset_name}_{split_name}_combined.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _mf_cols(df):
    cols = [c for c in df.columns if c.startswith("mf_")]
    if len(cols) != 768:
        raise ValueError(f"expected 768 mf_ columns, found {len(cols)}")
    return cols


def _clean(df, feature_cols, target_col, dataset_name, split_name):
    df = df.copy()
    n_nan = int(df[feature_cols].isna().sum().sum())
    if n_nan:
        log.warning("[%s/%s] filling %d NaN feature values with 0", dataset_name, split_name, n_nan)
    df[feature_cols] = df[feature_cols].fillna(0)

    n_bad = int(df[target_col].isna().sum())
    if n_bad:
        log.warning("[%s/%s] dropping %d rows with missing target", dataset_name, split_name, n_bad)
        df = df[df[target_col].notna()].reset_index(drop=True)
    return df


def _safe_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred) if y_true.nunique() > 1 else float("nan")


def _safe_ap(y_true, y_pred):
    return average_precision_score(y_true, y_pred) if y_true.nunique() > 1 else float("nan")


def train_one(dataset_name: str, target_col: str) -> dict:
    OUT_DIR.mkdir(exist_ok=True)

    train = _load_split(dataset_name, "train")
    val = _load_split(dataset_name, "val")
    test = _load_split(dataset_name, "test")

    feature_cols = _mf_cols(train)

    train = _clean(train, feature_cols, target_col, dataset_name, "train")
    val = _clean(val, feature_cols, target_col, dataset_name, "val")
    test = _clean(test, feature_cols, target_col, dataset_name, "test") if len(test) else test

    if train[target_col].nunique() < 2:
        raise ValueError(f"[{dataset_name}] training split has fewer than 2 classes")

    log.info("=== %s (MLP on MoLFormer embeddings) ===", dataset_name)
    log.info("Train %d | Val %d | Test %d", len(train), len(val), len(test))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols]) if len(test) else None

    y_train = train[target_col]
    y_val = val[target_col]
    y_test = test[target_col] if len(test) else None

    # Small, regularized, early-stopping to guard against overfitting on
    # small endpoints like Teratogenicity (89 train molecules).
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=1e-3,                 # L2 regularization
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1] if X_test is not None else None

    val_auc = _safe_auc(y_val, val_pred)
    val_ap = _safe_ap(y_val, val_pred)
    test_auc = _safe_auc(y_test, test_pred) if test_pred is not None else float("nan")

    log.info("[%s] Val AUC %.3f AP %.3f | Test AUC %.3f", dataset_name, val_auc, val_ap, test_auc)

    metrics = {
        "dataset": dataset_name,
        "target_col": target_col,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "val_auc": val_auc,
        "val_ap": val_ap,
        "test_auc": test_auc,
        "n_iter": int(model.n_iter_),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / f"{dataset_name}_metrics_mlp.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    all_metrics = []
    failures = []
    for name, target in DATASETS:
        try:
            all_metrics.append(train_one(name, target))
        except Exception as e:
            log.error("Skipping %s - %s", name, e)
            failures.append(name)

    lines = []
    lines.append("=" * 60)
    lines.append("ADMET Benchmark P3 - MLP on MoLFormer embeddings")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 60)
    lines.append("Features: MoLFormer 768-dim embeddings ONLY, StandardScaler-normalized")
    lines.append("Model: MLPClassifier hidden=(128,64), alpha=1e-3, early_stopping=True")
    lines.append("=" * 60)
    for m in all_metrics:
        lines.append(f"{m['dataset']} (target={m['target_col']}):")
        lines.append(f"  n_train={m['n_train']} | n_val={m['n_val']} | n_test={m['n_test']}")
        lines.append(f"  Val  AUC={m['val_auc']:.3f}  AP={m['val_ap']:.3f}")
        lines.append(f"  Test AUC={m['test_auc']:.3f}  (honest scaffold hold-out)")
        lines.append(f"  n_iter={m['n_iter']}")
    lines.append("-" * 60)
    lines.append("SUMMARY (Test AUC):")
    for m in all_metrics:
        lines.append(f"  {m['dataset']}: {m['test_auc']:.3f}")
    if failures:
        lines.append(f"FAILED: {failures}")

    report = "\n".join(lines)
    BENCHMARK_OUT.write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()