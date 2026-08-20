"""
train_catboost_for_production.py

Production fit + warm-start incremental retraining for CatBoost models,
built to stay the exact equivalent of train.py's validated run -- same
features, same NaN handling, same core hyperparameters -- while adding
support for incremental learning as new labeled molecules arrive.

=============================================================================
BUG FIXES vs. the previous version of this file
=============================================================================
1. Re-running --mode full used to silently overwrite the physical v0.cbm
   file in place, even when incremental rounds (v1, v2, ...) already
   existed and were warm-started from it. That left those later rounds'
   lineage entries pointing at a parent file that no longer matched what
   they were actually built from, with no error and no record of it.
   --mode full now refuses to touch an existing lineage that has
   incremental rounds unless you pass --confirm-rebuild, and always
   archives the prior lineage + v0 files (timestamped, not deleted)
   before writing fresh ones.
2. train_incremental() now verifies the parent .cbm's sha256 against the
   hash lineage recorded for it before warm-starting, so a corrupted or
   out-of-band-modified parent model is refused rather than silently
   built on top of.
3. Target-column validation (no NaNs, binary labels) is now applied to
   both the full-fit data and every incremental increment, before any
   CatBoost fit is attempted.
4. The incremental eval split is now guarded against degenerate batches:
   if the new increment is single-class, too small for the requested
   validation_fraction to produce at least one row per split, or a
   stratified split isn't possible, validation is disabled for that
   round (with a clear warning) instead of raising deep inside sklearn
   or CatBoost.
5. All JSON/model/alias writes are now atomic (temp file + os.replace),
   matching the versioned-file design already used elsewhere in this
   script.
6. A per-dataset file lock (fcntl) guards lineage.json against two runs
   for the same dataset racing each other.
7. Plain print() calls were replaced with the logging module for
   timestamps, levels, and easier redirection in a production setting.

=============================================================================
WHAT DID NOT CHANGE (on purpose, and why)
=============================================================================
depth=6, learning_rate=0.05, auto_class_weights="Balanced" are still
copied directly from train.py's validated CatBoostClassifier call and are
NOT retuned here. This script trains on 100% of the data with no
held-out set (in full mode), so there is no way to measure whether
changing these would help or hurt -- any change made here would be a
guess wearing a confident face. If you want to chase better AUC/AP, do
it in train.py against its val/test split; whatever wins there flows
into this script automatically because it reads train.py's validated
params and iteration count. Two extra CatBoost regularization knobs
(l2_leaf_reg, bagging_temperature) are exposed as opt-in CLI flags,
defaulted to None (= CatBoost's own default, i.e. untouched) so you can
experiment with them in train.py without this file ever silently
changing behavior.

NOTE on auto_class_weights="Balanced" during incremental rounds: CatBoost
recomputes "Balanced" weights from whatever data you hand it in that
call -- so each incremental round's new trees are weighted against that
round's own new-batch class balance, not the accumulated dataset's
balance. That's not "wrong", but it does mean the loss the new trees are
optimized under can drift round over round in a way this script doesn't
control for. If that matters for your use case, it belongs in train.py's
validation loop (e.g. computing explicit class_weights from the full
accumulated label distribution), not as a silent default change here.

=============================================================================
Warm-start incremental retraining
=============================================================================
This is offline batch incremental learning, not live/online learning:
you accumulate a new batch of labeled molecules, run this script once in
--mode incremental, and get a new production model that continues
training on top of the previous one via CatBoost's init_model warm-start
mechanism (new trees are added on top of the existing ensemble; old
trees are not touched or retrained).

Every round is versioned and never overwrites a prior model:
  modules/{name}_catboost_production_v0.cbm   <- full fit (train+val+test)
  modules/{name}_catboost_production_v1.cbm   <- +increment 1 (warm start)
  modules/{name}_catboost_production_v2.cbm   <- +increment 2 (warm start)
  modules/{name}_lineage.json                 <- full history + latest pointer
  modules/{name}_catboost_production.cbm      <- alias/copy of latest, for
                                                   any existing consumer that
                                                   expects a fixed filename

validation_fraction > 0 holds out a stratified slice of just the new
increment as an eval_set with use_best_model=True, so the new trees
stop being added once they stop helping on unseen new data -- this is
real validation signal for the new trees being added, without touching
or re-scoring anything about the original validated base model.

A soft warning fires if total tree count grows very large across many
incremental rounds (default threshold 5000): warm-starting forever lets
old data's influence dilute relative to recent batches in a way that
isn't reweighted, so periodically doing a fresh --mode full run (after
re-running train.py on the full accumulated dataset) is the recommended
reset point rather than warm-starting indefinitely.

Requires: pip install catboost pandas pyarrow scikit-learn
Requires featurize.py importable from the same directory.
Requires modules/{name}_metrics.json to exist for --mode full
(i.e. train.py must have been run first for that dataset).
"""

import argparse
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    fcntl = None
import hashlib
import json
import logging
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from catboost import CatBoostClassifier, __version__ as catboost_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from featurize import DESCRIPTOR_NAMES, ECFP_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_catboost")

FEATURE_COLS = DESCRIPTOR_NAMES + ECFP_COLS

DATASETS = [
    ("DILI", "Y"),
    ("hERG", "Y"),
    ("CYP3A4", "Y"),
    ("Ames", "Overall"),
    ("Teratogenicity", "Y"),
]

# Copied directly from train.py's validated CatBoostClassifier call.
# "iterations" is deliberately absent -- for --mode full it comes from
# each dataset's best_iteration in metrics.json (see read_best_iteration);
# for --mode incremental it's an early-stopped ceiling, not a fixed count.
BASE_CATBOOST_PARAMS = {
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": 100,
}

TOTAL_TREES_WARNING_THRESHOLD = 5000
MIN_ROWS_PER_SPLIT_FOR_VALIDATION = 2  # need at least this many rows in val to mean anything


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".partial")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_model_atomic(model: CatBoostClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    model.save_model(str(tmp))
    os.replace(str(tmp), str(path))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


from contextlib import contextmanager


@contextmanager
def dataset_lock(dataset_name: str, out_dir: Path):
    lock_dir = out_dir / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{dataset_name}.lock"

    if HAS_FCNTL:
        # POSIX: original fcntl logic
        with open(lock_path, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"Another training run for '{dataset_name}' appears to be in progress "
                    f"(lock held on {lock_path}). Wait for it to finish, or remove the lock "
                    f"file if you're sure it's stale."
                )
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    else:
        # Windows / platforms without fcntl: try msvcrt, otherwise no-op with warning
        try:
            import msvcrt
            with open(lock_path, "w") as f:
                locked = False
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    raise RuntimeError(
                        f"Another training run for '{dataset_name}' appears to be in progress "
                        f"(lock held on {lock_path}). Wait for it to finish, or remove the lock "
                        f"file if you're sure it's stale."
                    )
                try:
                    yield
                finally:
                    if locked:
                        try:
                            f.seek(0)
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
        except ImportError:
            log.warning(
                f"File locking not available on this platform (fcntl/msvcrt missing) - "
                f"running without lock for {dataset_name}. Avoid parallel runs for same dataset."
            )
            yield


def validate_target_col(df: pd.DataFrame, target_col: str, source_desc: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"{source_desc}: missing target column '{target_col}'")
    if len(df) == 0:
        raise ValueError(f"{source_desc}: no rows")
    n_missing = df[target_col].isna().sum()
    if n_missing:
        raise ValueError(f"{source_desc}: {n_missing} rows with missing '{target_col}' label")
    uniq = set(pd.unique(df[target_col].astype(float)))
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"{source_desc}: target column '{target_col}' has non-binary values: {sorted(uniq)}")


def fill_descriptor_nans(df: pd.DataFrame) -> tuple:
    """
    Matches train.py exactly: fills NaNs across FEATURE_COLS with 0 and
    reports how many values/rows were affected before doing so.
    """
    df = df.copy()
    n_nan = int(df[FEATURE_COLS].isna().sum().sum())
    n_rows_affected = 0
    if n_nan:
        n_rows_affected = int(df[FEATURE_COLS].isna().any(axis=1).sum())
        log.info(f"  filling {n_nan} NaN descriptor values with 0 ({n_rows_affected} affected rows)")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df, n_nan, n_rows_affected


def read_best_iteration(dataset_name: str, metrics_dir: str = "modules") -> int:
    """
    Reads best_iteration from train.py's {name}_metrics.json and returns
    the tree count to use (best_iteration + 1, since CatBoost's
    best_iteration is 0-indexed). Refuses to guess if the file or field
    is missing -- this is the number that makes the production fit
    faithful to what train.py actually validated, so a silent default
    would defeat the point of this script.
    """
    path = Path(metrics_dir) / f"{dataset_name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: {path} not found. Run train.py for this "
            f"dataset first -- this script will not guess an iteration count."
        )
    metrics = json.loads(path.read_text())
    if "best_iteration" not in metrics:
        raise KeyError(
            f"{dataset_name}: {path} exists but has no 'best_iteration' key. "
            f"Refusing to default to an arbitrary iteration count."
        )
    iterations = int(metrics["best_iteration"]) + 1
    if iterations < 1:
        raise ValueError(f"{dataset_name}: {path} best_iteration resolves to {iterations} iterations, which is invalid")
    return iterations


def _lineage_path(dataset_name: str, out_dir: Path) -> Path:
    return out_dir / f"{dataset_name}_lineage.json"


def _load_lineage(dataset_name: str, out_dir: Path) -> dict:
    path = _lineage_path(dataset_name, out_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"dataset": dataset_name, "latest_version": None, "rounds": []}


def _save_lineage(lineage: dict, dataset_name: str, out_dir: Path) -> None:
    atomic_write_json(_lineage_path(dataset_name, out_dir), lineage)


def _update_alias(cbm_path: Path, dataset_name: str, out_dir: Path) -> Path:
    """Keeps modules/{name}_catboost_production.cbm pointing at the latest
    version, for any downstream code that expects a fixed filename."""
    alias_path = out_dir / f"{dataset_name}_catboost_production.cbm"
    atomic_copy(cbm_path, alias_path)
    return alias_path


def _archive_existing_lineage(dataset_name: str, out_dir: Path, lineage: dict) -> None:
    """Archives the current lineage.json plus every versioned .cbm/_meta.json
    file it references, so a --mode full rebuild never destroys history --
    it only moves it aside."""
    archive_dir = out_dir / "archive" / f"{dataset_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    lineage_file = _lineage_path(dataset_name, out_dir)
    if lineage_file.exists():
        shutil.copy(str(lineage_file), str(archive_dir / lineage_file.name))
    for round_info in lineage.get("rounds", []):
        for key in ("cbm_path", "meta_path"):
            p = Path(round_info.get(key, ""))
            if p.exists():
                shutil.copy(str(p), str(archive_dir / p.name))
    log.info(f"  Archived existing lineage + versioned files to {archive_dir}")


# --------------------------------------------------------------------------
# Mode: full  (100% of train+val+test, matches train.py's validated config)
# --------------------------------------------------------------------------

def train_full(dataset_name: str, target_col: str,
                data_dir: str = "data", metrics_dir: str = "modules",
                out_dir: str = "modules",
                l2_leaf_reg: Optional[float] = None, bagging_temperature: Optional[float] = None,
                confirm_rebuild: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    with dataset_lock(dataset_name, out_dir):
        log.info(f"=== {dataset_name}: full production fit on 100% of validated data ===")

        lineage = _load_lineage(dataset_name, out_dir)
        has_incremental_rounds = any(r.get("type") == "incremental" for r in lineage.get("rounds", []))
        if lineage["rounds"] and has_incremental_rounds and not confirm_rebuild:
            raise RuntimeError(
                f"{dataset_name}: existing lineage already has incremental rounds warm-started "
                f"from v0. Rebuilding v0 from scratch here would invalidate what they were built "
                f"on. If that's really what you want, pass confirm_rebuild=True (or --confirm-rebuild "
                f"on the CLI) -- the existing lineage and versioned files will be archived, not deleted."
            )
        if lineage["rounds"]:
            _archive_existing_lineage(dataset_name, out_dir, lineage)
            lineage = {"dataset": dataset_name, "latest_version": None, "rounds": []}

        base = Path(data_dir) / dataset_name
        train = pd.read_parquet(f"{base}_train.parquet")
        val = pd.read_parquet(f"{base}_val.parquet")
        test = pd.read_parquet(f"{base}_test.parquet")
        for df, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            if target_col not in df.columns:
                raise ValueError(f"{dataset_name} {split_name}.parquet: missing target column '{target_col}'")

        full = pd.concat([train, val, test], ignore_index=True)
        log.info(f"  Combined {len(train)} train + {len(val)} val + {len(test)} test "
                 f"= {len(full)} rows (all used for training)")

        missing_cols = [c for c in FEATURE_COLS + [target_col] if c not in full.columns]
        if missing_cols:
            raise ValueError(f"{dataset_name}: missing expected columns {missing_cols[:5]}"
                              f"{'...' if len(missing_cols) > 5 else ''}")

        full, n_nan, n_rows_affected = fill_descriptor_nans(full)
        validate_target_col(full, target_col, f"{dataset_name} combined train+val+test")

        iterations = read_best_iteration(dataset_name, metrics_dir)
        log.info(f"  Using validated iterations={iterations} (from {dataset_name}_metrics.json)")

        params = dict(BASE_CATBOOST_PARAMS)
        params["iterations"] = iterations
        params["eval_metric"] = "AUC"
        if l2_leaf_reg is not None:
            params["l2_leaf_reg"] = l2_leaf_reg
        if bagging_temperature is not None:
            params["bagging_temperature"] = bagging_temperature

        model = CatBoostClassifier(**params)
        # No eval_set, no use_best_model: iterations is already fixed at the
        # validated tree count, and there's no held-out portion left once
        # train+val+test are combined.
        model.fit(full[FEATURE_COLS], full[target_col])

        version = 0
        cbm_path = out_dir / f"{dataset_name}_catboost_production_v{version}.cbm"
        meta_path = out_dir / f"{dataset_name}_catboost_production_v{version}_meta.json"
        save_model_atomic(model, cbm_path)

        label_counts = full[target_col].value_counts(dropna=False).to_dict()
        meta = {
            "dataset": dataset_name,
            "version": version,
            "round_type": "full",
            "target_col": target_col,
            "n_rows_trained_on": len(full),
            "n_train_source": len(train),
            "n_val_source": len(val),
            "n_test_source": len(test),
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "descriptor_nan_values_filled_with_zero": n_nan,
            "descriptor_nan_rows_affected": n_rows_affected,
            "feature_cols_count": len(FEATURE_COLS),
            "feature_cols": FEATURE_COLS,
            "catboost_params": params,
            "total_trees": int(model.tree_count_),
            "model_hash": file_hash(cbm_path),
            "source_validated_metrics_file": str(Path(metrics_dir) / f"{dataset_name}_metrics.json"),
            "note": (
                "Trained on 100% of this dataset's model-ready molecules "
                "(train+val+test recombined). No accuracy/AUC is recorded here -- "
                "that validation, and the iteration count and hyperparameters "
                "used above, come from the run recorded in "
                "source_validated_metrics_file."
            ),
            "catboost_version": catboost_version,
            "python_version": platform.python_version(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(meta_path, meta)
        alias_path = _update_alias(cbm_path, dataset_name, out_dir)

        lineage["latest_version"] = version
        lineage["rounds"].append({
            "version": version,
            "type": "full",
            "cbm_path": str(cbm_path),
            "meta_path": str(meta_path),
            "model_hash": meta["model_hash"],
            "n_rows": len(full),
            "total_trees": int(model.tree_count_),
            "timestamp": meta["timestamp"],
        })
        _save_lineage(lineage, dataset_name, out_dir)

        log.info(f"  Saved: {cbm_path}")
        log.info(f"  Saved: {meta_path}")
        log.info(f"  Alias updated: {alias_path}")
        return cbm_path


# --------------------------------------------------------------------------
# Mode: incremental (warm-start on new data only)
# --------------------------------------------------------------------------

def _safe_validation_split(increment: pd.DataFrame, target_col: str, validation_fraction: float):
    """Returns (train_inc, val_inc, disabled_reason). val_inc is None and
    disabled_reason is set if a meaningful stratified split isn't possible
    for this batch -- callers should fall back to no early stopping rather
    than let sklearn/CatBoost raise partway through."""
    if not validation_fraction or validation_fraction <= 0:
        return increment, None, "validation_fraction=0"

    counts = increment[target_col].value_counts()
    if len(counts) < 2:
        return increment, None, f"increment is single-class ({counts.to_dict()}); AUC/AP are undefined"
    if counts.min() < MIN_ROWS_PER_SPLIT_FOR_VALIDATION:
        return increment, None, (
            f"smallest class in the increment has only {int(counts.min())} row(s); "
            f"too few to hold any out for validation"
        )

    n_val = int(round(len(increment) * validation_fraction))
    if n_val < 1 or (len(increment) - n_val) < 1:
        return increment, None, f"validation_fraction={validation_fraction} on {len(increment)} rows leaves an empty split"

    try:
        train_inc, val_inc = train_test_split(
            increment, test_size=validation_fraction,
            stratify=increment[target_col], random_state=42,
        )
    except ValueError as e:
        return increment, None, f"stratified split failed ({e})"

    if val_inc[target_col].nunique() < 2:
        return increment, None, "validation split ended up single-class; AUC/AP are undefined"

    return train_inc, val_inc, None


def train_incremental(dataset_name: str, target_col: str, increment_path: str,
                       out_dir: str = "modules",
                       max_new_iterations: int = 1000,
                       validation_fraction: float = 0.1,
                       early_stopping_rounds: int = 50,
                       l2_leaf_reg: Optional[float] = None, bagging_temperature: Optional[float] = None,
                       total_trees_warning: int = TOTAL_TREES_WARNING_THRESHOLD):
    """
    Warm-starts from the latest saved version for this dataset and trains
    only on the new increment, adding trees on top of the existing
    ensemble via CatBoost's init_model. Old trees are never retrained or
    touched.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    increment_path = Path(increment_path)
    if not increment_path.exists():
        raise FileNotFoundError(f"increment_path {increment_path} does not exist")

    with dataset_lock(dataset_name, out_dir):
        lineage = _load_lineage(dataset_name, out_dir)
        if lineage["latest_version"] is None:
            raise FileNotFoundError(
                f"{dataset_name}: no existing production model found in {out_dir}. "
                f"Run --mode full at least once before incremental training."
            )
        parent_version = lineage["latest_version"]
        parent_round = next(r for r in lineage["rounds"] if r["version"] == parent_version)
        parent_cbm_path = Path(parent_round["cbm_path"])
        parent_total_trees = parent_round["total_trees"]

        if not parent_cbm_path.exists():
            raise FileNotFoundError(f"Parent checkpoint {parent_cbm_path} referenced by lineage is missing")
        expected_hash = parent_round.get("model_hash")
        if expected_hash and file_hash(parent_cbm_path) != expected_hash:
            raise RuntimeError(
                f"{dataset_name}: parent checkpoint {parent_cbm_path} does not match the hash "
                f"recorded in lineage for v{parent_version}. Refusing to warm-start from a model "
                f"that may be corrupted or was modified outside this script."
            )

        log.info(f"=== {dataset_name}: incremental warm-start from v{parent_version} "
                 f"({parent_total_trees} existing trees) ===")

        # Warm start requires an identical feature set/order to the parent
        # model -- verify before touching anything expensive.
        parent_model = CatBoostClassifier()
        parent_model.load_model(str(parent_cbm_path))
        if list(parent_model.feature_names_) != FEATURE_COLS:
            raise ValueError(
                f"{dataset_name}: current FEATURE_COLS does not match the feature "
                f"set the parent model (v{parent_version}) was trained on. Warm "
                f"start requires an identical, identically-ordered feature set. "
                f"If featurize.py changed, you need a fresh --mode full run, not "
                f"an incremental one."
            )

        increment = pd.read_parquet(increment_path)
        missing_cols = [c for c in FEATURE_COLS + [target_col] if c not in increment.columns]
        if missing_cols:
            raise ValueError(f"{dataset_name}: increment data missing columns "
                              f"{missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")

        increment, n_nan, n_rows_affected = fill_descriptor_nans(increment)
        validate_target_col(increment, target_col, str(increment_path))
        log.info(f"  New increment: {len(increment)} rows")

        train_inc, val_inc, disabled_reason = _safe_validation_split(increment, target_col, validation_fraction)
        has_val = val_inc is not None
        if has_val:
            log.info(f"  Holding out {len(val_inc)} of the new rows for early stopping "
                     f"({len(train_inc)} used for training this round)")
        else:
            log.warning(f"  Validation disabled for this round: {disabled_reason}. "
                        f"Training up to max_new_iterations with no early stopping -- "
                        f"consider validating this round some other way.")

        params = dict(BASE_CATBOOST_PARAMS)
        params["iterations"] = max_new_iterations
        params["eval_metric"] = "AUC"
        if l2_leaf_reg is not None:
            params["l2_leaf_reg"] = l2_leaf_reg
        if bagging_temperature is not None:
            params["bagging_temperature"] = bagging_temperature

        model = CatBoostClassifier(**params)
        fit_kwargs = {"init_model": str(parent_cbm_path)}
        if has_val:
            fit_kwargs["eval_set"] = (val_inc[FEATURE_COLS], val_inc[target_col])
            fit_kwargs["use_best_model"] = True
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        model.fit(train_inc[FEATURE_COLS], train_inc[target_col], **fit_kwargs)

        round_metrics = {}
        if has_val:
            val_probs = model.predict_proba(val_inc[FEATURE_COLS])[:, 1]
            round_metrics = {
                "held_out_auc": float(roc_auc_score(val_inc[target_col], val_probs)),
                "held_out_ap": float(average_precision_score(val_inc[target_col], val_probs)),
                "held_out_rows": len(val_inc),
            }
            log.info(f"  Held-out AUC (this increment only): {round_metrics['held_out_auc']:.4f}")
            log.info(f"  Held-out AP  (this increment only): {round_metrics['held_out_ap']:.4f}")

        new_version = parent_version + 1
        cbm_path = out_dir / f"{dataset_name}_catboost_production_v{new_version}.cbm"
        meta_path = out_dir / f"{dataset_name}_catboost_production_v{new_version}_meta.json"
        save_model_atomic(model, cbm_path)

        total_trees = int(model.tree_count_)
        if total_trees >= total_trees_warning:
            log.warning(f"  total tree count is now {total_trees} across "
                        f"{new_version + 1} rounds. Consider a fresh --mode full run "
                        f"(re-run train.py on the full accumulated dataset first) "
                        f"instead of continuing to warm-start indefinitely.")

        meta = {
            "dataset": dataset_name,
            "version": new_version,
            "round_type": "incremental",
            "parent_version": parent_version,
            "parent_cbm_path": str(parent_cbm_path),
            "parent_model_hash": expected_hash,
            "target_col": target_col,
            "increment_source_path": str(increment_path),
            "n_increment_rows": len(increment),
            "n_increment_train_rows": len(train_inc),
            "validation_disabled_reason": disabled_reason,
            "descriptor_nan_values_filled_with_zero": n_nan,
            "descriptor_nan_rows_affected": n_rows_affected,
            "catboost_params": params,
            "total_trees": total_trees,
            "new_trees_added": total_trees - parent_total_trees,
            "model_hash": None,  # filled after save below
            **round_metrics,
            "note": (
                "Warm-started from the parent model listed above via CatBoost's "
                "init_model; only new trees added in this round were fit on the "
                "increment above. held_out_auc/ap (if present) reflect only this "
                "round's new data, not the full accumulated dataset, and are not "
                "comparable to train.py's original val/test AUC. auto_class_weights "
                "='Balanced' was recomputed from this round's increment only -- see "
                "module docstring."
            ),
            "catboost_version": catboost_version,
            "python_version": platform.python_version(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta["model_hash"] = file_hash(cbm_path)
        atomic_write_json(meta_path, meta)
        alias_path = _update_alias(cbm_path, dataset_name, out_dir)

        lineage["latest_version"] = new_version
        lineage["rounds"].append({
            "version": new_version,
            "type": "incremental",
            "parent_version": parent_version,
            "cbm_path": str(cbm_path),
            "meta_path": str(meta_path),
            "model_hash": meta["model_hash"],
            "n_increment_rows": len(increment),
            "total_trees": total_trees,
            **round_metrics,
            "timestamp": meta["timestamp"],
        })
        _save_lineage(lineage, dataset_name, out_dir)

        log.info(f"  Saved: {cbm_path}")
        log.info(f"  Saved: {meta_path}")
        log.info(f"  Alias updated: {alias_path}")
        return cbm_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    parser.add_argument("--dataset", default=None,
                         help="Dataset name (e.g. DILI). Required for --mode incremental. "
                              "Omit for --mode full to run all DATASETS.")
    parser.add_argument("--increment-data", default=None,
                         help="Path to a parquet file of NEW labeled rows, required for "
                              "--mode incremental.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metrics-dir", default="modules")
    parser.add_argument("--out-dir", default="modules")
    parser.add_argument("--confirm-rebuild", action="store_true",
                         help="Required for --mode full if a lineage with incremental "
                              "rounds already exists. The old lineage is archived, not deleted.")
    parser.add_argument("--max-new-iterations", type=int, default=1000,
                         help="Ceiling on new trees added per incremental round; the "
                              "actual number is usually lower due to early stopping.")
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                         help="Fraction of the NEW increment held out for early stopping "
                              "on incremental rounds. 0 disables early stopping.")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--l2-leaf-reg", type=float, default=None,
                         help="Optional override. Leave unset to use CatBoost's default "
                              "(untouched, matches previously validated behavior).")
    parser.add_argument("--bagging-temperature", type=float, default=None,
                         help="Optional override. Leave unset to use CatBoost's default "
                              "(untouched, matches previously validated behavior).")
    args = parser.parse_args()

    if args.mode == "full":
        targets = [(args.dataset, dict(DATASETS).get(args.dataset))] if args.dataset else DATASETS
        failures = []
        for name, target in targets:
            if target is None:
                log.warning(f"Unknown dataset '{name}', skipping.")
                continue
            try:
                train_full(name, target, data_dir=args.data_dir,
                           metrics_dir=args.metrics_dir, out_dir=args.out_dir,
                           l2_leaf_reg=args.l2_leaf_reg,
                           bagging_temperature=args.bagging_temperature,
                           confirm_rebuild=args.confirm_rebuild)
            except FileNotFoundError as e:
                msg = f"Skipping {name}: {e}"
                log.warning(msg)
                failures.append(msg)
            except Exception as e:
                msg = f"FAILED {name}: {e}"
                log.error(msg)
                failures.append(msg)
        log.info("=== SUMMARY ===")
        if failures:
            for f in failures:
                log.info(f" - {f}")
            raise SystemExit(1)
        else:
            log.info("All production models trained on 100% of validated data.")

    else:  # incremental
        if not args.dataset or not args.increment_data:
            parser.error("--mode incremental requires --dataset and --increment-data")
        target = dict(DATASETS).get(args.dataset)
        if target is None:
            parser.error(f"Unknown dataset '{args.dataset}'")
        train_incremental(
            args.dataset, target, args.increment_data,
            out_dir=args.out_dir,
            max_new_iterations=args.max_new_iterations,
            validation_fraction=args.validation_fraction,
            early_stopping_rounds=args.early_stopping_rounds,
            l2_leaf_reg=args.l2_leaf_reg,
            bagging_temperature=args.bagging_temperature,
        )


if __name__ == "__main__":
    main()
    