"""
Trainer for the toxicity/ADMET endpoint models (DILI, hERG, CYP3A4, Ames,
Teratogenicity).

Retraining strategy: warm-start incremental retraining (NOT live/online
learning). Each retrain call is still an explicit, offline batch job — you
decide when it runs and on what data. What "incremental" means here is that
a retrain can continue boosting on top of the existing model's trees
(CatBoost's `init_model`) instead of always starting from iteration 0. This
assumes your data pipeline is the source of truth for what "current
training data" means: before running --warm-start you should have already
refreshed data/{dataset}_train.parquet (etc.) with whatever new molecules
you want folded in. This script does not fetch or diff new data itself.

Usage:
    # normal full training (unchanged behavior from before), all datasets
    python train_catboost.py

    # warm-start incremental retrain of a single endpoint on refreshed data
    python train.py --dataset hERG --warm-start --increment-iterations 300

    # force a from-scratch retrain even if a prior model exists
    python train.py --dataset DILI --full-retrain

Environment overrides:
    ADMET_DATA_DIR      default "data"
    ADMET_MODULES_DIR   default "modules"
    ADMET_KEEP_INCREMENTS  how many prior snapshots per dataset to retain
                           (default 20; set 0 to keep all)
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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from featurize import DESCRIPTOR_NAMES, ECFP_COLS

FEATURE_COLS = DESCRIPTOR_NAMES + ECFP_COLS

DATASETS = [
    ("DILI", "Y"),
    ("hERG", "Y"),
    ("CYP3A4", "Y"),
    ("Ames", "Overall"),
    ("Teratogenicity", "Y"),
]

DATA_DIR = Path(os.environ.get("ADMET_DATA_DIR", "data"))
MODULES_DIR = Path(os.environ.get("ADMET_MODULES_DIR", "modules"))
# Snapshots of every previous model+metrics before they get overwritten.
# This is what makes warm-start retraining auditable/rollback-able instead
# of a silent in-place mutation of the production model file.
INCREMENTS_DIR = MODULES_DIR / "increments"
REGISTRY_PATH = MODULES_DIR / "registry.jsonl"  # append-only log, not a rewritten array
# How many prior snapshots to keep per dataset. Unbounded snapshotting will
# eventually fill disk on a long-lived warm-start cadence; 0 disables pruning.
KEEP_INCREMENTS = int(os.environ.get("ADMET_KEEP_INCREMENTS", "20"))

# Left exactly as in the original script. Not touched — see the note in
# chat about why blind tuning on these dataset sizes is more likely to hurt
# than help without held-out evidence.
BASE_PARAMS = dict(
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    eval_metric="AUC",
    random_seed=42,
)
FULL_TRAIN_ITERATIONS = 2500
DEFAULT_INCREMENT_ITERATIONS = 300  # additional trees appended on warm-start
# Safe addition: use_best_model already selects the best-val-AUC iteration
# regardless of how long training runs, so stopping early once val AUC
# stalls only saves compute -- it cannot pick a worse model than before.
EARLY_STOPPING_ROUNDS = 150

log = logging.getLogger("admet_train")


class TrainingError(Exception):
    """Raised for data/schema problems that should stop this dataset's run
    but not necessarily the whole batch."""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _load_split(dataset_name: str, split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset_name}_{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_columns(dataset_name: str, split_name: str, df: pd.DataFrame, target_col: str) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise TrainingError(
            f"[{dataset_name}/{split_name}] missing {len(missing)} expected feature "
            f"column(s), e.g. {missing[:5]}. Data pipeline is out of sync with "
            f"featurize.py's current DESCRIPTOR_NAMES/ECFP_COLS."
        )
    if target_col not in df.columns:
        raise TrainingError(f"[{dataset_name}/{split_name}] missing target column {target_col!r}")


def _fill_nans(dataset_name: str, split_name: str, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # A valid molecule can still fail an individual RDKit descriptor
    # calculation (e.g. an unusual valence/macrocycle) even though
    # build_model_ready's _is_valid filter only checks that the molecule
    # parsed at all -- so a few NaNs here are expected, not a bug. Report
    # how many before filling, rather than silently overwriting them: 0 is
    # not a neutral value for something like MolWt or LogP, so it's worth
    # knowing whether this is affecting 3 rows or 3000 before trusting it.
    n_nan = int(df[FEATURE_COLS].isna().sum().sum())
    if n_nan:
        n_rows = int(df[FEATURE_COLS].isna().any(axis=1).sum())
        log.warning(
            "[%s/%s] filling %d NaN descriptor values with 0 (%d affected rows)",
            dataset_name, split_name, n_nan, n_rows,
        )
    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    # Unlike feature NaNs, a NaN label can't be safely defaulted to
    # anything -- it would silently inject a fake class-0 (or class-1)
    # example. Drop those rows and say so loudly.
    n_bad_labels = int(df[target_col].isna().sum())
    if n_bad_labels:
        log.warning(
            "[%s/%s] dropping %d row(s) with missing target label %r",
            dataset_name, split_name, n_bad_labels, target_col,
        )
        df = df[df[target_col].notna()].reset_index(drop=True)
    return df


def _safe_auc(y_true, y_pred) -> float:
    return roc_auc_score(y_true, y_pred) if y_true.nunique() > 1 else float("nan")


def _safe_ap(y_true, y_pred) -> float:
    # average_precision_score doesn't raise on a single-class y_true the way
    # roc_auc_score does -- it just returns a degenerate, misleading number
    # (e.g. ~1.0 if everything is positive). Guard it the same way as AUC so
    # a small/skewed val split (Teratogenicity's is 11 molecules) doesn't
    # produce a clean-looking but meaningless AP.
    return average_precision_score(y_true, y_pred) if y_true.nunique() > 1 else float("nan")


def _atomic_write_bytes(path: Path, write_fn) -> None:
    """Write via a temp file + os.replace so a crash mid-write never leaves
    a truncated/corrupt file at `path`. write_fn(tmp_path) must produce the
    final content at tmp_path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, lambda tmp: tmp.write_text(text))


def _append_registry(entry: dict) -> None:
    # Append-only JSONL instead of read-modify-write of a single JSON array:
    # avoids the read/parse/rewrite race between concurrent runs, and avoids
    # rewriting an ever-growing file on every single training call.
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _prune_increments(dataset_name: str) -> None:
    if KEEP_INCREMENTS <= 0 or not INCREMENTS_DIR.exists():
        return
    snapshots = sorted(
        INCREMENTS_DIR.glob(f"{dataset_name}_*_catboost.cbm"),
        key=lambda p: p.stat().st_mtime,
    )
    excess = snapshots[:-KEEP_INCREMENTS] if len(snapshots) > KEEP_INCREMENTS else []
    for cbm_path in excess:
        stamp = cbm_path.name[len(dataset_name) + 1: -len("_catboost.cbm")]
        metrics_snap = INCREMENTS_DIR / f"{dataset_name}_{stamp}_metrics.json"
        cbm_path.unlink(missing_ok=True)
        metrics_snap.unlink(missing_ok=True)
        log.info("[%s] pruned old increment snapshot %s", dataset_name, stamp)


def _check_feature_schema_match(dataset_name: str, prior_metrics: dict | None) -> None:
    if prior_metrics is None:
        return
    prior_cols = prior_metrics.get("feature_cols")
    if prior_cols is None:
        log.warning(
            "[%s] prior metrics file has no recorded feature_cols; cannot verify "
            "warm-start feature-schema compatibility. Proceeding, but this is a gap "
            "worth backfilling.",
            dataset_name,
        )
        return
    if prior_cols != FEATURE_COLS:
        added = [c for c in FEATURE_COLS if c not in prior_cols]
        removed = [c for c in prior_cols if c not in FEATURE_COLS]
        raise TrainingError(
            f"[{dataset_name}] refusing warm-start: current FEATURE_COLS does not match "
            f"the feature schema the saved model was trained on "
            f"(added={added[:5]}{'...' if len(added) > 5 else ''}, "
            f"removed={removed[:5]}{'...' if len(removed) > 5 else ''}). "
            f"CatBoost's init_model will not error on a mismatched feature space -- it "
            f"will silently produce a broken model. Run --full-retrain instead, or "
            f"revert featurize.py to the schema this model expects."
        )


def train_one(
    dataset_name: str,
    target_col: str,
    warm_start: bool = False,
    increment_iterations: int | None = None,
    tag: str | None = None,
) -> dict:
    MODULES_DIR.mkdir(exist_ok=True)

    train = _load_split(dataset_name, "train")
    val = _load_split(dataset_name, "val")
    test = _load_split(dataset_name, "test")

    for split_name, df in (("train", train), ("val", val), ("test", test)):
        _validate_columns(dataset_name, split_name, df, target_col)

    train = _fill_nans(dataset_name, "train", train, target_col)
    val = _fill_nans(dataset_name, "val", val, target_col)
    test = _fill_nans(dataset_name, "test", test, target_col) if len(test) else test

    if train[target_col].nunique() < 2:
        raise TrainingError(f"[{dataset_name}] training split has fewer than 2 classes after cleaning")

    log.info("=== %s%s ===", dataset_name, " (warm-start)" if warm_start else "")
    log.info("Train %d | Val %d | Test %d", len(train), len(val), len(test))

    train_data_path = DATA_DIR / f"{dataset_name}_train.parquet"
    train_data_hash = _file_sha256(train_data_path) if train_data_path.exists() else None

    model_path = MODULES_DIR / f"{dataset_name}_catboost.cbm"
    metrics_path = MODULES_DIR / f"{dataset_name}_metrics.json"

    init_model = None
    prior_metrics = None
    if warm_start:
        if not model_path.exists():
            log.info(
                "[%s] --warm-start requested but no existing model at %s; "
                "training from scratch instead.", dataset_name, model_path,
            )
        else:
            if metrics_path.exists():
                prior_metrics = json.loads(metrics_path.read_text())
            _check_feature_schema_match(dataset_name, prior_metrics)
            init_model = CatBoostClassifier()
            init_model.load_model(str(model_path))
            prior_best = prior_metrics.get("best_iteration") if prior_metrics else "?"
            log.info(
                "[%s] warm-starting from %s (prior best_iteration=%s)",
                dataset_name, model_path, prior_best,
            )

    iterations = (
        (increment_iterations or DEFAULT_INCREMENT_ITERATIONS)
        if init_model is not None
        else FULL_TRAIN_ITERATIONS
    )

    model = CatBoostClassifier(
        iterations=iterations,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=100,
        **BASE_PARAMS,
    )

    fit_kwargs = dict(
        eval_set=(val[FEATURE_COLS], val[target_col]),
        use_best_model=True,
    )
    if init_model is not None:
        fit_kwargs["init_model"] = init_model

    model.fit(train[FEATURE_COLS], train[target_col], **fit_kwargs)

    val_pred = model.predict_proba(val[FEATURE_COLS])[:, 1]
    test_pred = model.predict_proba(test[FEATURE_COLS])[:, 1] if len(test) > 0 else None

    val_auc = _safe_auc(val[target_col], val_pred)
    val_ap = _safe_ap(val[target_col], val_pred)
    test_auc = _safe_auc(test[target_col], test_pred) if test_pred is not None else float("nan")

    log.info("[%s] Val AUC %.3f AP %.3f | Test AUC %.3f", dataset_name, val_auc, val_ap, test_auc)

    # Snapshot whatever model/metrics currently sit at the canonical path
    # BEFORE overwriting them, so a bad warm-start round is never a
    # destructive, unrecoverable action.
    if model_path.exists():
        INCREMENTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(model_path, INCREMENTS_DIR / f"{dataset_name}_{stamp}_catboost.cbm")
        if metrics_path.exists():
            shutil.copy2(metrics_path, INCREMENTS_DIR / f"{dataset_name}_{stamp}_metrics.json")
        _prune_increments(dataset_name)

    # Atomic writes: a kill/crash mid-save can no longer leave a truncated
    # .cbm or a half-written metrics.json at the canonical path.
    _atomic_write_bytes(model_path, lambda tmp: model.save_model(str(tmp)))

    metrics = {
        "dataset": dataset_name,
        "target_col": target_col,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "val_auc": val_auc,
        "val_ap": val_ap,
        "test_auc": test_auc,
        "best_iteration": int(model.get_best_iteration()),
        "feature_cols": FEATURE_COLS,
        "warm_started": init_model is not None,
        "warm_started_from_best_iteration": (
            prior_metrics.get("best_iteration") if prior_metrics else None
        ),
        "train_data_sha256": train_data_hash,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
    }
    _atomic_write_text(metrics_path, json.dumps(metrics, indent=2))

    _append_registry({
        "dataset": dataset_name,
        "trained_at_utc": metrics["trained_at_utc"],
        "warm_started": metrics["warm_started"],
        "n_train": len(train),
        "val_auc": val_auc,
        "test_auc": test_auc,
        "best_iteration": metrics["best_iteration"],
        "train_data_sha256": train_data_hash,
        "tag": tag,
    })

    log.info("Saved: %s", model_path)
    log.info("Saved: %s", metrics_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", action="append", dest="datasets", default=None,
        help="Name of a dataset to train (e.g. hERG). Repeatable. Default: all datasets.",
    )
    parser.add_argument(
        "--warm-start", action="store_true",
        help="Continue training on top of the existing saved model (CatBoost init_model) "
             "instead of retraining from scratch. Falls back to a full retrain if no "
             "existing model is found.",
    )
    parser.add_argument(
        "--full-retrain", action="store_true",
        help="Force a from-scratch retrain even if a saved model exists. Mutually "
             "exclusive with --warm-start.",
    )
    parser.add_argument(
        "--increment-iterations", type=int, default=None,
        help=f"Additional boosting rounds to append during a warm-start retrain "
             f"(default: {DEFAULT_INCREMENT_ITERATIONS}). Ignored for full retrains.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Free-text note stored in this run's metrics/registry entry "
             "(e.g. 'added Q3 batch of 40 new compounds').",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.warm_start and args.full_retrain:
        parser.error("--warm-start and --full-retrain are mutually exclusive")
    if args.increment_iterations is not None and args.increment_iterations <= 0:
        parser.error("--increment-iterations must be a positive integer")

    selected = DATASETS
    if args.datasets:
        wanted = set(args.datasets)
        selected = [(n, t) for n, t in DATASETS if n in wanted]
        missing = wanted - {n for n, _ in selected}
        if missing:
            parser.error(f"Unknown dataset(s): {sorted(missing)}. "
                         f"Choices: {[n for n, _ in DATASETS]}")

    failures = []
    for name, target in selected:
        try:
            train_one(
                name, target,
                warm_start=args.warm_start and not args.full_retrain,
                increment_iterations=args.increment_iterations,
                tag=args.tag,
            )
        except FileNotFoundError as e:
            log.warning("Skipping %s - missing data file %s. Run scaffold_split.py first.", name, e)
            failures.append(name)
        except TrainingError as e:
            log.error("Skipping %s - %s", name, e)
            failures.append(name)
        except Exception:
            # Don't let one broken dataset take down the rest of the batch job,
            # but do surface a full traceback and a non-zero exit code so a
            # scheduler/CI job notices instead of reporting silent success.
            log.error("Skipping %s - unexpected error:\n%s", name, traceback.format_exc())
            failures.append(name)

    if failures:
        log.error("Completed with failures: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())