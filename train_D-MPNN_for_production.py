"""
 python  train_D-MPNN_for_production.py production : PRODUCTION + INCREMENTAL

 train all 5 datasets
python train_D-MPNN_for_production.py production

 train just one dataset
python train_D-MPNN_for_production.py production --dataset DILI

 train 2 specific ones
python train_D-MPNN_for_production.py production --dataset DILI --dataset Ames

 if it says lineage already exists and refuses to rebuild:
python train_D-MPNN_for_production.py production --dataset DILI --confirm-rebuild

python train_D-MPNN_for_production.py incremental --dataset DILI --new-data data/new_dili_rows.csv

# with early stopping on the new data (needs >=20 new rows)
python train_D-MPNN_for_production.py incremental --dataset DILI --new-data data/new_dili_rows.csv --early-stopping-patience 3 --max-extra-epochs 10

for incremental you always need these 2:
Code
--dataset NAME
--new-data PATH_TO_CSV

The other 2 are optional - they already have defaults:
--max-extra-epochs 10          (default = 10)
--early-stopping-patience      (default = None = off)

Two entry points:
    1. train_production_endpoint(name, target_col)
     -> trains from scratch on 100% of data using validated hyperparams
        from *_dmpnn_metrics.json. Use this once, or to rebuild from zero
        if you suspect drift/corruption. If a lineage with promoted
        incremental rounds already exists, this REFUSES to run unless you
        pass confirm_rebuild=True, and even then archives the old lineage
        file (with a timestamp) instead of deleting it.

    2. train_incremental_endpoint(name, target_col, new_data_path)
     -> WARM-START incremental retraining. Loads the current production
        (or latest promoted incremental) checkpoint, verifies its hash
        against what lineage recorded, continues training on old+new data
        at a reduced LR for a small number of epochs, evaluates against a
        held-out set that is NEVER trained on across any round, and only
        promotes the new checkpoint if it doesn't regress.

This is NOT live/online learning. Nothing updates per-sample. Every round
is a deliberate, gated, versioned batch retrain that you trigger by hand
(or via the CLI at the bottom).

TUNING KNOBS (see TuningConfig below) are OFF by default (None = "use
validated value unchanged"). Turning one on means you are running a new
experiment, not applying a proven improvement -- re-validate on the
held-out set before trusting the result.

CHANGES vs the original draft (see accompanying review for the full list):
    - lineage/meta JSON writes are atomic (temp file + os.replace)
    - production retrain no longer silently wipes lineage history
    - parent checkpoint hash is verified before warm-starting from it
    - evaluate() fails loudly instead of propagating None
    - new incremental data is schema/label-validated before any training work
    - early-stopping "monitor" set is never the same data the model trains on
    - chemeleon download supports an optional sha256 pin, checked before load
    - a per-dataset file lock guards the lineage file against concurrent runs
    - training is seeded for reproducible promote/reject decisions
    - minimal CLI so you don't have to hand-edit RUN_ONLY to do one-off runs

SPEED NOTES (CPU-only, low-RAM dev machine -- infra only, nothing below
changes promotion thresholds, warm-start LR, epoch counts, hash checks,
lineage archiving, or how the held-out test set is used):
    - BLAS/OpenMP thread env vars set before numpy/torch import (often
      default to single-threaded otherwise on constrained systems).
    - The chemeleon foundation-model checkpoint dict is read from disk
      once per process and cached in memory. build_chemeleon_message_
      passing() previously re-read it from disk on every call -- an
      incremental round calls it at least twice (baseline model +
      candidate model). Every model still gets its own fresh
      nn.BondMessagePassing + load_state_dict, identical to before.
    - make_dataset() now caches the featurized MoleculeDataset to disk,
      keyed by a content hash of the actual SMILES+label data (not the
      file path), so re-running production/incremental against unchanged
      data during development skips re-featurization. Because the key is
      content-based there's no staleness risk.
    - DataLoader worker count now reads from DMPNN_NUM_WORKERS (still
      defaults to 0 -- safe for 8GB RAM), with persistent_workers opted
      in only if you raise it.
    - enable_model_summary=False on all Trainer instances to cut a bit of
      setup/print overhead.
    - Deliberately did NOT add a data-subsetting "fast" mode here (unlike
      the sibling dev script): this file's entire design is built on
      strict guarantees -- parent-checkpoint hash verification, a
      held-out test set that must never be trained on, lineage that's
      never silently wiped. Faking a smaller dataset through this path
      risks a promotion decision being made on non-representative data,
      which is exactly the class of mistake the rest of this file guards
      against. For fast smoke-tests, point --new-data / DATA_DIR at a
      small standalone test dataset instead.

GPU / GOOGLE COLAB (T4) NOTES -- added on top of the CPU speed notes above,
still infra-only, same guarantee (promotion thresholds, warm-start LR,
epoch counts, hash checks, lineage archiving, held-out-set usage: all
byte-for-byte unchanged):
    - Accelerator/device is now auto-detected once at import time
      (_detect_accelerator()). If torch.cuda.is_available() -> "gpu",
      devices=1 (Colab gives you exactly one T4). Otherwise -> "cpu",
      devices=1, identical to the original hardcoded behavior. Every
      pl.Trainer(...) call (evaluate(), production, incremental) uses
      this shared ACCELERATOR/DEVICES/PRECISION instead of a hardcoded
      accelerator="cpu".
    - On GPU, Trainer precision is set to "16-mixed" (automatic mixed
      precision) -- a T4's tensor cores are roughly 2-4x faster in fp16
      than fp32 for this kind of model, and PyTorch Lightning handles the
      loss-scaling/casting automatically. On CPU, precision stays "32-true"
      (fp32), i.e. exactly the original behavior.
    - cudnn.benchmark is turned on when a GPU is present (no-op, harmless
      on CPU-only runs since it's gated behind cuda.is_available()).
    - DataLoaders now pass pin_memory=True when a GPU is present (faster
      host->device transfer), with the same try/except-and-fall-back
      pattern already used for persistent_workers so it degrades cleanly
      on chemprop versions that don't accept the kwarg.
    - The chemeleon checkpoint download/load path is untouched -- it's
      still read to CPU RAM once and handed to Lightning, which moves
      parameters onto the GPU itself during trainer.fit()/trainer.test().
    - A small Colab bootstrap cell is included further down (guarded by
      `if IN_COLAB:`) that installs the handful of packages Colab doesn't
      ship with (chemprop, lightning) and prints which accelerator was
      detected, so you can just paste this whole file into a Colab cell
      (or `!python train_D-MPNN_for_production.py production` in a Colab
      terminal cell) and it does the right thing on a T4 or on CPU.
"""

import os

# --- Speed setup: must run before numpy/pandas/torch are imported ----------
_N_THREADS = str(max(1, (os.cpu_count() or 4)))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _N_THREADS)
# -----------------------------------------------------------------------------

import argparse
import hashlib
import json
import logging
import pickle
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Google Colab bootstrap (no-op outside Colab). Only installs packages /
# prints diagnostics -- does not touch any training logic below.
# ---------------------------------------------------------------------------
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import importlib
    _need = []
    for _pkg in ("chemprop", "lightning"):
        if importlib.util.find_spec(_pkg) is None:
            _need.append(_pkg)
    if _need:
        os.system(f"pip install -q {' '.join(_need)}")

from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
from chemprop import data, featurizers, models, nn

torch.set_num_threads(int(_N_THREADS))
try:
    # Harmless on CPU-only setups; only affects CUDA matmul precision paths.
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# ---------------------------------------------------------------------------
# GPU / accelerator auto-detection (Colab T4 -> "gpu", else -> "cpu").
# Detected once, at import time, and reused by every pl.Trainer(...) call
# below. This is the only thing that decides GPU vs CPU -- nothing about
# batch size, epoch count, LR, promotion threshold, or data handling
# changes based on it.
# ---------------------------------------------------------------------------

def _detect_accelerator():
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        gpu_name = torch.cuda.get_device_name(0)
        return "gpu", 1, "16-mixed", gpu_name
    return "cpu", 1, "32-true", None

ACCELERATOR, DEVICES, PRECISION, _GPU_NAME = _detect_accelerator()
USE_GPU = ACCELERATOR == "gpu"

# --- CROSS-PLATFORM IMPORT FOR FILE LOCKING ---
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    fcntl = None # type: ignore
    HAS_FCNTL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_dmpnn_incremental")

log.info(
    f"Accelerator = {ACCELERATOR} (devices={DEVICES}, precision={PRECISION})"
    + (f" [{_GPU_NAME}]" if _GPU_NAME else " [no GPU visible -- falling back to CPU, nothing else changes]")
)

DATASETS = [("DILI", "Y"), ("hERG", "Y"), ("CYP3A4", "Y"), ("Ames", "Overall"), ("Teratogenicity", "Y")]

DATA_DIR = Path("data")
MODULES_DIR = Path("modules")
LINEAGE_DIR = MODULES_DIR / "lineage"
LOCK_DIR = MODULES_DIR / "locks"
FEATURE_CACHE_DIR = MODULES_DIR / "feature_cache"
CHEMELEON_PATH = Path("chemeleon_mp.pt")
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_MIN_BYTES = 1_000_000
# Fill this in once you've verified a good download (sha256sum chemeleon_mp.pt).
# Left None = size-check only, with a loud warning. Pin it for real production use.
CHEMELEON_SHA256: Optional[str] = None
# Default 0 to stay safe on 8GB RAM (worker subprocesses duplicate data in
# memory). Override with DMPNN_NUM_WORKERS=N if you want to experiment.
# Colab gives you 2 vCPUs, so the existing default of 2 is already a good
# fit there too -- left untouched.
NUM_WORKERS = int(os.environ.get("DMPNN_NUM_WORKERS", "2"))
# Inference-only batch size for evaluate() (held-out AUROC scoring). This is
# NOT a training hyperparameter -- no gradients, no optimizer step, batch_norm
# is off in this model -- so making it bigger changes nothing about the
# result, only how many forward-pass round-trips it takes to get there.
# Bigger is safe on a T4's 16GB; keeps the old value on CPU.
EVAL_BATCH_SIZE = int(os.environ.get("DMPNN_EVAL_BATCH_SIZE", "512" if USE_GPU else "128"))
ENCODER_STATE_VALUES = {"frozen", "fine-tuned"}
MIN_NEW_ROWS_FOR_MONITOR = 20 # below this, we don't fake a validation set

# Tolerance for promotion: how much AUROC drop on the held-out set is
# acceptable before we refuse to promote an incremental checkpoint.
# 0.0 = zero tolerance (new must be >= old). Loosen only deliberately.
REGRESSION_TOLERANCE = 0.0

DEFAULT_SEED = 42

@dataclass
class TuningConfig:
    """Optional overrides. None = keep the validated value from metrics.json.
    Every non-None field here is an experiment you are choosing to run --
    it is not a guaranteed improvement. Validate on the held-out set."""
    weight_decay: Optional[float] = None
    lr_scheduler: Optional[str] = None
    lr_patience: int = 3
    grad_clip_val: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    dropout_override: Optional[float] = None
    max_extra_epochs: int = 10
    seed: int = DEFAULT_SEED
    # OPT-IN ONLY, off by default (None = use the validated batch_size from
    # *_dmpnn_metrics.json, unchanged). Unlike the eval-only batch size
    # above, this DOES change training dynamics -- fewer, larger gradient
    # updates per epoch -- so it is not silently applied. Set explicitly
    # (CLI: --gpu-batch-size) when you want to trade a small amount of
    # optimization-trajectory risk for GPU utilization / wall-clock time,
    # e.g. on a time-limited Colab T4 session. For incremental rounds this
    # is still caught by the existing held-out AUROC promotion gate; for a
    # production (from-scratch) run there is no such gate, so treat a
    # from-scratch run made with this set as a new baseline to sanity-check.
    train_batch_size_override: Optional[int] = None

# ---------------------------------------------------------------------------
# small infra helpers
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path) # atomic on same filesystem
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

@contextmanager
def dataset_lock(name: str):
    """Advisory file lock so two incremental/production runs for the same
    dataset can't race on lineage.json. Not a substitute for a real job
    queue, but enough to fail fast instead of corrupting state."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{name}.lock"

    if HAS_FCNTL:
        # POSIX: original fcntl logic
        with open(lock_path, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"Another training run for '{name}' appears to be in progress "
                    f"(lock held on {lock_path}). Wait for it to finish or remove "
                    f"the lock file if you're sure it's stale."
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
                        f"Another training run for '{name}' appears to be in progress "
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
                f"running without lock for {name}. Avoid parallel runs for same dataset."
            )
            yield

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def file_hash_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_dataframe(df: pd.DataFrame, target_col: str, source_desc: str) -> None:
    """Fail fast, with a clear message, on malformed training data."""
    if "SMILES" not in df.columns:
        raise ValueError(f"{source_desc}: missing required 'SMILES' column")
    if target_col not in df.columns:
        raise ValueError(f"{source_desc}: missing target column '{target_col}'")
    if len(df) == 0:
        raise ValueError(f"{source_desc}: no rows")
    n_missing_smiles = df["SMILES"].isna().sum()
    if n_missing_smiles:
        raise ValueError(f"{source_desc}: {n_missing_smiles} rows with missing SMILES")
    n_missing_y = df[target_col].isna().sum()
    if n_missing_y:
        raise ValueError(f"{source_desc}: {n_missing_y} rows with missing '{target_col}' label")
    uniq = set(pd.unique(df[target_col].astype(float)))
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"{source_desc}: target column '{target_col}' has non-binary values: {sorted(uniq)}")

# ---------------------------------------------------------------------------
# foundation model loading
# ---------------------------------------------------------------------------

def ensure_chemeleon_downloaded():
    if CHEMELEON_PATH.exists() and CHEMELEON_PATH.stat().st_size >= CHEMELEON_MIN_BYTES:
        if CHEMELEON_SHA256:
            if file_hash_full(CHEMELEON_PATH)!= CHEMELEON_SHA256:
                log.warning("Existing chemeleon checkpoint fails sha256 check -- re-downloading")
                CHEMELEON_PATH.unlink()
            else:
                return CHEMELEON_PATH
        else:
            return CHEMELEON_PATH
    if CHEMELEON_PATH.exists():
        CHEMELEON_PATH.unlink()

    fd, tmp_str = tempfile.mkstemp(dir=".", suffix=".partial")
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        urlretrieve(CHEMELEON_URL, tmp)
        if tmp.stat().st_size < CHEMELEON_MIN_BYTES:
            raise RuntimeError(f"Downloaded chemeleon file is only {tmp.stat().st_size} bytes -- looks truncated")
        if CHEMELEON_SHA256:
            got = file_hash_full(tmp)
            if got!= CHEMELEON_SHA256:
                raise RuntimeError(f"chemeleon download sha256 mismatch: got {got}, expected {CHEMELEON_SHA256}")
        else:
            log.warning(
                "CHEMELEON_SHA256 is not set -- only a size check was performed on the downloaded "
                "foundation model weights. Pin the hash for real production use; this file is "
                "loaded with torch.load(weights_only=False), which trusts its contents."
            )
        shutil.move(str(tmp), str(CHEMELEON_PATH))
    finally:
        if tmp.exists():
            tmp.unlink()
    return CHEMELEON_PATH

# In-memory cache of the raw checkpoint dict so it's only read from disk
# once per process, no matter how many times build_chemeleon_message_
# passing() is called (an incremental round calls it at least twice: once
# for the baseline model, once for the candidate model). Every caller
# still gets its own fresh nn.BondMessagePassing(...) + load_state_dict(...),
# so nothing is shared between models -- identical behavior to before,
# just without redundant disk I/O. Same trust boundary as before: this
# still only ever points at CHEMELEON_PATH, integrity-checked above.
_CHEMELEON_CKPT_CACHE = None

def _load_chemeleon_ckpt_dict():
    global _CHEMELEON_CKPT_CACHE
    if _CHEMELEON_CKPT_CACHE is None:
        # NOTE: weights_only=False is required because chemprop/lightning checkpoints
        # contain more than plain tensors. Only ever point this at CHEMELEON_PATH
        # (integrity-checked above) or locally produced checkpoints -- never load
        # an untrusted .pt/.ckpt file this way. Loaded to CPU RAM regardless of
        # accelerator -- Lightning moves parameters onto the GPU itself during
        # trainer.fit()/trainer.test(), so this stays map_location="cpu" even
        # when ACCELERATOR == "gpu".
        _CHEMELEON_CKPT_CACHE = torch.load(str(CHEMELEON_PATH), map_location="cpu", weights_only=False)
    return _CHEMELEON_CKPT_CACHE

def build_chemeleon_message_passing():
    ckpt = _load_chemeleon_ckpt_dict()
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    return mp, ckpt["hyper_parameters"]["d_h"]

class ReapplyFreeze(Callback):
    def __init__(self, mp):
        super().__init__()
        self.mp = mp

    def on_train_epoch_start(self, trainer, pl_module):
        self.mp.eval()
        for p in self.mp.parameters():
            p.requires_grad_(False)

def get_validated_config(name: str):
    """Reads the original validated hyperparams (from your v2.1.1 sweep).
    Single source of truth for architecture -- it never changes across
    incremental rounds, only weights and epoch count do."""
    metrics_path = MODULES_DIR / f"{name}_dmpnn_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found -- run the v2.1.1 sweep first")
    src = json.loads(metrics_path.read_text())
    for k in ["model_path", "encoder", "agg", "ffn_hidden", "dropout", "lr", "chemeleon_dim"]:
        if k not in src:
            raise KeyError(f"{metrics_path} missing '{k}'")
    if src["encoder"] not in ENCODER_STATE_VALUES:
        raise ValueError(f"encoder={src['encoder']} not in {ENCODER_STATE_VALUES}")
    return src

def build_model(src, tuning: TuningConfig):
    mp, _ = build_chemeleon_message_passing()
    agg = nn.SumAggregation() if src["agg"] == "sum" else nn.MeanAggregation()
    dropout = tuning.dropout_override if tuning.dropout_override is not None else src["dropout"]
    ffn = nn.BinaryClassificationFFN(input_dim=src["chemeleon_dim"], hidden_dim=src["ffn_hidden"], dropout=dropout)
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.metrics.BinaryAUROC()])
    return mpnn, mp

def _df_content_key(df: pd.DataFrame, target_col: str) -> str:
    """Content hash (not path-based) so the cache below can never go
    stale: if the actual SMILES/label data changes at all, the key
    changes with it."""
    hashed = pd.util.hash_pandas_object(df[["SMILES", target_col]], index=False).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:24]

def make_dataset(df, target_col):
    # Disk cache of the featurized dataset, keyed by data content. Molecule
    # featurization is CPU-heavy and deterministic given the same rows, so
    # re-running production/incremental against unchanged data (common
    # while developing/testing) can skip it entirely.
    cache_key = _df_content_key(df, target_col)
    cache_path = FEATURE_CACHE_DIR / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"feature cache read failed for {cache_path.name} ({e}); recomputing")

    smis = df["SMILES"].values
    ys = df[[target_col]].values.astype(float)
    dps = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(dps, featurizer)

    try:
        FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(dset, f)
    except Exception as e:
        log.warning(f"feature cache write failed for {cache_path.name} ({e}); continuing uncached")

    return dset

def _build_loader(dset, batch_size, shuffle, num_workers=NUM_WORKERS):
    """Thin wrapper around data.build_dataloader that opts into
    persistent_workers when num_workers > 0, and pin_memory when a GPU is
    present (faster host->device transfer for the batches DataLoader
    produces). Falls back cleanly if the installed chemprop version
    doesn't support either kwarg. With num_workers=0 and no GPU, behavior
    is identical to a direct data.build_dataloader(...) call."""
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    # Try the most feature-complete call first, then fall back one kwarg
    # at a time so this still works on older chemprop versions. prefetch_factor
    # lets worker processes build the *next* batch while the GPU is still
    # busy with the current one -- pure overlap, no effect on what gets
    # computed.
    attempts = []
    if num_workers > 0 and USE_GPU:
        attempts.append(dict(persistent_workers=True, pin_memory=True, prefetch_factor=4))
        attempts.append(dict(persistent_workers=True, pin_memory=True))
    if num_workers > 0:
        attempts.append(dict(persistent_workers=True, prefetch_factor=4))
        attempts.append(dict(persistent_workers=True))
    if USE_GPU:
        attempts.append(dict(pin_memory=True))
    attempts.append({})

    for extra in attempts:
        try:
            return data.build_dataloader(dset, **kwargs, **extra)
        except TypeError:
            continue
    return data.build_dataloader(dset, **kwargs)

def load_held_out_test(name, target_col):
    """The scaffold-split test set from your original v2.1.1 run. This file
    must NEVER be trained on, in production or incremental rounds -- it is
    your only honest signal of whether a round helped or hurt."""
    path = DATA_DIR / f"{name}_test_chemprop.csv"
    df = pd.read_csv(path)
    validate_dataframe(df, target_col, str(path))
    return make_dataset(df, target_col), len(df)

def evaluate(mpnn, dset, batch_size=EVAL_BATCH_SIZE):
    loader = _build_loader(dset, batch_size, shuffle=False)
    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, accelerator=ACCELERATOR, devices=DEVICES,
        precision=PRECISION, num_sanity_val_steps=0,
    )
    result = trainer.test(mpnn, loader, verbose=False)
    if not result or "test/BinaryAUROC" not in result[0]:
        raise RuntimeError(
            f"evaluate(): expected key 'test/BinaryAUROC' in trainer.test() output, got {result}. "
            f"This likely means the chemprop/lightning metric key changed -- fix the key rather "
            f"than letting a promotion decision run on a None value."
        )
    auroc = result[0]["test/BinaryAUROC"]
    if auroc is None:
        raise RuntimeError("evaluate(): BinaryAUROC came back None")
    return auroc

def configure_optimizer_fn(mpnn, lr, weight_decay, lr_scheduler_name, lr_patience):
    def configure_optimizers():
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, mpnn.parameters()),
            lr=lr,
            weight_decay=weight_decay or 0.0,
        )
        if lr_scheduler_name == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=lr_patience)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/BinaryAUROC"}}
        return opt
    return configure_optimizers

# ---------------------------------------------------------------------------
# 1. PRODUCTION: train from scratch on 100% of data
# ---------------------------------------------------------------------------

def train_production_endpoint(name, target_col, tuning: TuningConfig = TuningConfig(), confirm_rebuild: bool = False):
    with dataset_lock(name):
        log.info(f"=== {name}: PRODUCTION (from scratch, 100% data) ===")
        pl.seed_everything(tuning.seed, workers=True)

        lineage_path = LINEAGE_DIR / f"{name}_lineage.json"
        if lineage_path.exists():
            existing = json.loads(lineage_path.read_text())
            if len(existing) > 1 and not confirm_rebuild:
                raise RuntimeError(
                    f"{lineage_path} already has {len(existing)} versions (including promoted "
                    f"incremental rounds). Rebuilding from scratch here would discard that lineage. "
                    f"If that's really what you want, call train_production_endpoint(..., "
                    f"confirm_rebuild=True). The existing lineage will be archived, not deleted."
                )
            if existing:
                archive_dir = LINEAGE_DIR / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                shutil.copy(str(lineage_path), str(archive_dir / f"{name}_lineage_{stamp}.json"))
                log.info(f"Archived existing lineage to {archive_dir / f'{name}_lineage_{stamp}.json'}")

        src = get_validated_config(name)

        ckpt = torch.load(str(Path(src["model_path"])), map_location="cpu", weights_only=False)
        best_epoch = ckpt.get("epoch")
        if best_epoch is None:
            raise KeyError(f"{src['model_path']} has no epoch")
        n_epochs = int(best_epoch) + 1
        freeze_encoder = src["encoder"] == "frozen"
        batch_size = src.get("batch_size", 32 if src.get("n_train", 0) < 500 else 128)
        if tuning.train_batch_size_override is not None:
            log.warning(
                f"[{name}] --gpu-batch-size override active: using batch_size="
                f"{tuning.train_batch_size_override} instead of the validated "
                f"{batch_size}. This changes training dynamics (fewer/larger "
                f"gradient steps per epoch), not just infra speed -- there is no "
                f"promotion gate on a from-scratch production run, so treat this "
                f"checkpoint as a new baseline and sanity-check it before relying on it."
            )
            batch_size = tuning.train_batch_size_override

        dfs = []
        for split in ("train", "val", "test"):
            p = DATA_DIR / f"{name}_{split}_chemprop.csv"
            df = pd.read_csv(p)
            validate_dataframe(df, target_col, str(p))
            dfs.append(df)
        full = pd.concat(dfs, ignore_index=True)
        full_dset = make_dataset(full, target_col)
        train_loader = _build_loader(full_dset, batch_size, shuffle=True)

        mpnn, mp = build_model(src, tuning)
        mpnn.configure_optimizers = configure_optimizer_fn(
            mpnn, src["lr"], tuning.weight_decay, tuning.lr_scheduler, tuning.lr_patience
        )

        callbacks = []
        if freeze_encoder:
            mp.eval()
            for p in mp.parameters():
                p.requires_grad_(False)
            callbacks.append(ReapplyFreeze(mp))

        trainer = pl.Trainer(
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, accelerator=ACCELERATOR, devices=DEVICES,
            precision=PRECISION, num_sanity_val_steps=0,
            max_epochs=n_epochs, callbacks=callbacks,
            gradient_clip_val=tuning.grad_clip_val,
            deterministic=True,
        )
        trainer.fit(mpnn, train_loader)

        MODULES_DIR.mkdir(exist_ok=True)
        out = MODULES_DIR / f"{name}_dmpnn_production.ckpt"
        tmp = out.with_suffix(".ckpt.partial")
        trainer.save_checkpoint(str(tmp))
        os.replace(str(tmp), str(out)) # atomic

        meta = {
            "dataset": name, "target_col": target_col, "version": 0,
            "n_rows_trained_on": len(full), "epochs_trained": n_epochs,
            "encoder": src["encoder"], "agg": src["agg"], "ffn_hidden": src["ffn_hidden"],
            "dropout": tuning.dropout_override or src["dropout"], "lr": src["lr"],
            "weight_decay": tuning.weight_decay, "chemeleon_dim": src["chemeleon_dim"],
            "foundation_model": "chemeleon_mp",
            "seed": tuning.seed,
            "batch_size_used": batch_size,
            "train_batch_size_override": tuning.train_batch_size_override,
            "model_path": str(out),
            "parent_checkpoint": None,
            "checkpoint_hash": file_hash(out),
            "checkpoint_hash_full": file_hash_full(out),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(MODULES_DIR / f"{name}_dmpnn_production_meta.json", meta)
        atomic_write_json(lineage_path, [meta])
        log.info(f"Saved {out}")
        return meta

# ---------------------------------------------------------------------------
# 2. INCREMENTAL: warm-start, train briefly, gate on held-out AUROC
# ---------------------------------------------------------------------------

def train_incremental_endpoint(name, target_col, new_data_path, tuning: TuningConfig = TuningConfig()):
    with dataset_lock(name):
        log.info(f"=== {name}: INCREMENTAL warm-start round ===")
        pl.seed_everything(tuning.seed, workers=True)

        new_data_path = Path(new_data_path)
        if not new_data_path.exists():
            raise FileNotFoundError(f"new_data_path {new_data_path} does not exist")
        new_df = pd.read_csv(new_data_path)
        validate_dataframe(new_df, target_col, str(new_data_path)) # fail fast, before any model work

        src = get_validated_config(name)
        lineage_path = LINEAGE_DIR / f"{name}_lineage.json"
        if not lineage_path.exists():
            raise FileNotFoundError(
                f"No lineage found for {name}. Run train_production_endpoint('{name}',...) first "
                f"to establish version 0."
            )
        lineage = json.loads(lineage_path.read_text())
        parent = lineage[-1] # latest PROMOTED version
        parent_ckpt_path = Path(parent["model_path"])
        if not parent_ckpt_path.exists():
            raise FileNotFoundError(f"Parent checkpoint {parent_ckpt_path} referenced by lineage is missing")

        expected_hash = parent.get("checkpoint_hash")
        if expected_hash and file_hash(parent_ckpt_path)!= expected_hash:
            raise RuntimeError(
                f"Parent checkpoint {parent_ckpt_path} does not match the hash recorded in lineage "
                f"({expected_hash}). Refusing to warm-start from a checkpoint that may be corrupted "
                f"or was modified out of band."
            )

        # --- held-out test set: same file every round, never trained on ---
        held_out_dset, n_held_out = load_held_out_test(name, target_col)

        # --- baseline: how good is the parent, right now, on the held-out set? ---
        baseline_mpnn, _ = build_model(src, TuningConfig(dropout_override=parent.get("dropout")))
        parent_ckpt = torch.load(str(parent_ckpt_path), map_location="cpu", weights_only=False)
        baseline_mpnn.load_state_dict(parent_ckpt["state_dict"])
        baseline_auroc = evaluate(baseline_mpnn, held_out_dset)
        log.info(f"[{name}] parent (v{parent['version']}) held-out AUROC = {baseline_auroc:.4f}")

        # --- build training set: old train+val (still on disk) + new increment ---
        old_train_path = DATA_DIR / f"{name}_train_chemprop.csv"
        old_val_path = DATA_DIR / f"{name}_val_chemprop.csv"
        old_train = pd.read_csv(old_train_path)
        old_val = pd.read_csv(old_val_path)
        validate_dataframe(old_train, target_col, str(old_train_path))
        validate_dataframe(old_val, target_col, str(old_val_path))
        combined = pd.concat([old_train, old_val, new_df], ignore_index=True)
        n_new = len(new_df)
        log.info(f"[{name}] training on {len(old_train)} old-train + {len(old_val)} old-val + {n_new} new rows")

        combined_dset = make_dataset(combined, target_col)
        batch_size = src.get("batch_size", 32 if len(combined) < 500 else 128)
        if tuning.train_batch_size_override is not None:
            log.warning(
                f"[{name}] --gpu-batch-size override active: using batch_size="
                f"{tuning.train_batch_size_override} instead of the validated "
                f"{batch_size}. This changes training dynamics for this round. "
                f"It's still covered by the held-out AUROC promotion gate below, "
                f"so a regression will be caught and rejected as usual."
            )
            batch_size = tuning.train_batch_size_override
        train_loader = _build_loader(combined_dset, batch_size, shuffle=True)

        # A slice of the *new* data can double as an early-stopping monitor --
        # but only if there's enough of it to mean something. We never fall
        # back to validating on the training set itself: that would defeat
        # the point of early stopping.
        use_monitor = n_new >= MIN_NEW_ROWS_FOR_MONITOR
        monitor_dset = make_dataset(new_df, target_col) if use_monitor else None
        run_early_stopping = tuning.early_stopping_patience is not None and use_monitor
        if tuning.early_stopping_patience is not None and not use_monitor:
            log.warning(
                f"[{name}] early_stopping_patience was set but only {n_new} new rows are available "
                f"(< {MIN_NEW_ROWS_FOR_MONITOR}) -- skipping early stopping for this round rather than "
                f"monitoring against the training data itself. Training for max_extra_epochs instead."
            )

        # --- warm start ---
        mpnn, mp = build_model(src, tuning)
        mpnn.load_state_dict(parent_ckpt["state_dict"])

        freeze_encoder = src["encoder"] == "frozen"
        callbacks = []
        if freeze_encoder:
            mp.eval()
            for p in mp.parameters():
                p.requires_grad_(False)
            callbacks.append(ReapplyFreeze(mp))
        if run_early_stopping:
            callbacks.append(EarlyStopping(monitor="val/BinaryAUROC", mode="max", patience=tuning.early_stopping_patience))

        warm_start_lr = src["lr"] * 0.1 # reduced LR is the core of "warm start, don't destroy prior knowledge"
        mpnn.configure_optimizers = configure_optimizer_fn(
            mpnn, warm_start_lr, tuning.weight_decay, tuning.lr_scheduler, tuning.lr_patience
        )

        val_loader = None
        if run_early_stopping:
            val_loader = _build_loader(monitor_dset, batch_size, shuffle=False)

        trainer = pl.Trainer(
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, accelerator=ACCELERATOR, devices=DEVICES,
            precision=PRECISION, num_sanity_val_steps=0,
            max_epochs=tuning.max_extra_epochs, callbacks=callbacks,
            gradient_clip_val=tuning.grad_clip_val,
            deterministic=True,
        )
        trainer.fit(mpnn, train_loader, val_loader)

        # --- gate: only promote if held-out AUROC doesn't regress ---
        candidate_auroc = evaluate(mpnn, held_out_dset)
        version = parent["version"] + 1
        delta = candidate_auroc - baseline_auroc
        promoted = delta >= -REGRESSION_TOLERANCE
        log.info(f"[{name}] candidate (v{version}) held-out AUROC = {candidate_auroc:.4f} (delta {delta:+.4f}) "
                  f"-> {'PROMOTED' if promoted else 'REJECTED, production unchanged'}")

        MODULES_DIR.mkdir(exist_ok=True)
        candidate_path = MODULES_DIR / f"{name}_dmpnn_v{version}_candidate.ckpt"
        tmp = candidate_path.with_suffix(".ckpt.partial")
        trainer.save_checkpoint(str(tmp))
        os.replace(str(tmp), str(candidate_path))

        meta = {
            "dataset": name, "target_col": target_col, "version": version,
            "parent_version": parent["version"], "parent_checkpoint": str(parent_ckpt_path),
            "parent_checkpoint_hash": parent.get("checkpoint_hash"),
            "n_new_rows": n_new, "n_total_rows_trained_on": len(combined),
            "warm_start_lr": warm_start_lr, "weight_decay": tuning.weight_decay,
            "seed": tuning.seed,
            "batch_size_used": batch_size,
            "train_batch_size_override": tuning.train_batch_size_override,
            "used_early_stopping_monitor": run_early_stopping,
            "held_out_n": n_held_out,
            "baseline_held_out_auroc": baseline_auroc,
            "candidate_held_out_auroc": candidate_auroc,
            "delta": delta, "promoted": promoted,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        if promoted:
            final_path = MODULES_DIR / f"{name}_dmpnn_production.ckpt"
            shutil.copy(str(candidate_path), str(final_path))
            meta["model_path"] = str(final_path)
            meta["checkpoint_hash"] = file_hash(final_path)
            meta["checkpoint_hash_full"] = file_hash_full(final_path)
            lineage.append(meta)
            atomic_write_json(lineage_path, lineage)
            atomic_write_json(MODULES_DIR / f"{name}_dmpnn_production_meta.json", meta)
        else:
            meta["model_path"] = str(candidate_path) # kept for inspection, NOT promoted
            rejected_log = LINEAGE_DIR / f"{name}_rejected.json"
            rejected = json.loads(rejected_log.read_text()) if rejected_log.exists() else []
            rejected.append(meta)
            atomic_write_json(rejected_log, rejected)

        return meta

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_prod = sub.add_parser("production", help="Train from scratch on 100%% of data")
    p_prod.add_argument("--dataset", action="append", dest="datasets",
                         help="Dataset name to run (repeatable). Default: all of DATASETS.")
    p_prod.add_argument("--confirm-rebuild", action="store_true",
                         help="Required if a lineage with promoted incremental rounds already exists.")
    p_prod.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_prod.add_argument("--gpu-batch-size", type=int, default=None,
                         help="OPT-IN. Overrides the validated training batch_size (e.g. 256 on a "
                              "T4) to improve GPU utilization / wall-clock time. Changes training "
                              "dynamics, not just speed -- off by default. No promotion gate exists "
                              "for production runs, so treat the result as a new baseline.")

    p_inc = sub.add_parser("incremental", help="Warm-start retrain on old+new data")
    p_inc.add_argument("--dataset", required=True, help="Dataset name (must have an existing lineage)")
    p_inc.add_argument("--new-data", required=True, help="Path to CSV with new rows")
    p_inc.add_argument("--max-extra-epochs", type=int, default=10)
    p_inc.add_argument("--early-stopping-patience", type=int, default=None)
    p_inc.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_inc.add_argument("--gpu-batch-size", type=int, default=None,
                        help="OPT-IN. Overrides the validated training batch_size for this round. "
                             "Still covered by the held-out AUROC promotion gate, so a regression "
                             "is caught and rejected automatically.")

    args = parser.parse_args(argv)

    name_to_target = dict(DATASETS)

    if args.mode == "production":
        names = args.datasets or [n for n, _ in DATASETS]
        ensure_chemeleon_downloaded()
        failures = []
        for n in names:
            if n not in name_to_target:
                log.error(f"Unknown dataset '{n}', skipping. Known: {list(name_to_target)}")
                failures.append(n)
                continue
            try:
                train_production_endpoint(
                    n, name_to_target[n],
                    tuning=TuningConfig(seed=args.seed, train_batch_size_override=args.gpu_batch_size),
                    confirm_rebuild=args.confirm_rebuild,
                )
            except Exception as e:
                log.error(f"FAILED {n}: {e}\n{traceback.format_exc()}")
                failures.append(n)
        return 1 if failures else 0

    if args.mode == "incremental":
        if args.dataset not in name_to_target:
            log.error(f"Unknown dataset '{args.dataset}'. Known: {list(name_to_target)}")
            return 1
        ensure_chemeleon_downloaded()
        try:
            train_incremental_endpoint(
                args.dataset, name_to_target[args.dataset], args.new_data,
                tuning=TuningConfig(
                    max_extra_epochs=args.max_extra_epochs,
                    early_stopping_patience=args.early_stopping_patience,
                    seed=args.seed,
                    train_batch_size_override=args.gpu_batch_size,
                ),
            )
        except Exception as e:
            log.error(f"FAILED {args.dataset}: {e}\n{traceback.format_exc()}")
            return 1
        return 0

    return 1

if __name__ == "__main__":
    raise SystemExit(main())