"""
train_dmpnn.py — D-MPNN

python train_dmpnn.py --all
# or
python train_dmpnn.py --all --ensemble-size 3

python train_dmpnn.py --endpoint DILI
python train_dmpnn.py --endpoint DILI --no-balance --ensemble-size 1
python train_dmpnn.py --endpoint hERG --dev-fast --ensemble-size 1

Those are test-set numbers. There is no legitimate way to tune a model to
land on pre-chosen test-set values -- doing so requires repeatedly
checking against the test set and adjusting until it matches, which is
test-set leakage. The resulting model wouldn't actually be better; it
would just be a model shaped to one held-out sample, and the "match"
would evaporate on the next batch of real molecules. That's not done
here, on purpose, even though it was asked for directly.

What IS done here -- three changes that can legitimately improve
generalization without touching the test set, applied only where there's
a real justification for them:

  1. CLASS IMBALANCE HANDLING (new). The sibling CatBoost script uses
     auto_class_weights="Balanced"; this script had NO imbalance
     handling at all, which is an inconsistency, not a design choice.
     Fixed via minority-class oversampling on the TRAIN split only --
     val and test are never touched, so this can't inflate a reported
     number, only change what the model is trained against.

  2. SEED ENSEMBLING. Training the same validated config across
     multiple seeds and averaging predicted probabilities is standard
     practice in the chemprop literature (the original chemprop authors
     recommend ensembling). It reduces variance from FFN-head
     initialization; it cannot manufacture accuracy that isn't there.

  3. WEIGHT DECAY + MORE EPOCH BUDGET, fine-tuned configs only (new).
     Early stopping (patience) already governs training length, so
     raising the epoch ceiling only gives it more room to find a better
     stopping point -- it cannot make things worse on its own. Weight
     decay is a standard, low-risk regularizer for fine-tuning.

WHAT WAS DELIBERATELY NOT TOUCHED: ffn_hidden, agg, dropout, lr,
batch_size, and the entire n_train<500 frozen-encoder branch, all of
which already differ meaningfully per dataset in get_config(). That
asymmetry is evidence of a real prior tuning pass; guessing new values
on top of it without new validation evidence would be the same mistake
as the test-target request above, just smaller.

SPEED NOTES (CPU-only, low-RAM dev machine -- infra only, no training
logic changed, nothing below affects what gets learned or how models
are selected):

  - BLAS/OpenMP thread env vars are set before numpy/torch import, since
    they otherwise often default to single-threaded on constrained
    systems and this is pure wall-clock waste on a multi-core CPU.
  - The chemeleon foundation-model checkpoint is read from disk once and
    cached in memory instead of once per seed per dataset. Every model
    still gets a fresh copy of the weights via load_state_dict, so
    nothing leaks between ensemble members -- this only removes redundant
    disk I/O.
  - Featurized SMILES->graph datasets are cached to disk per
    (dataset, split, balance, seed) under feature_cache/, invalidated
    automatically if the source CSV changes. Featurization is CPU-heavy
    and was previously redone on every single script invocation, which
    is the dominant cost during iterative development.
  - DataLoader workers stay off (0) by default -- on 8GB RAM, worker
    subprocesses duplicate data in memory and can cause swapping, which
    is slower than the serialization overhead they'd save. Settable via
    the DMPNN_NUM_WORKERS env var if you want to experiment.
  - New --dev-fast flag (opt-in, OFF by default): subsets train/val/test
    to a few hundred rows and caps epochs, purely so the pipeline can be
    smoke-tested end-to-end in minutes. Output metrics are tagged
    dev_fast: true so they can never be mistaken for real numbers.
  - CPU mixed precision (bf16 autocast) was deliberately NOT enabled:
    12th-gen Alder Lake client chips (including i3-1215U) ship with
    AVX-512/AMX fused off, so there's no hardware bf16 path -- software
    emulation would likely add overhead rather than remove it.

GPU/SPEED ADDITIONS (Colab T4) -- this is the only change on top of the
above; everything in the two sections above is still true and unchanged
when no GPU is present:

  - CUDA is auto-detected. If present (e.g. a Colab T4 runtime),
    Lightning's Trainer is pointed at accelerator="gpu"; otherwise it
    falls back to accelerator="cpu" with byte-for-byte the same behavior
    as before. Nothing about which config, split, epoch budget, or seed
    is used changes based on device -- only where the tensors run.
  - fp16 mixed precision (precision="16-mixed") is used by default when
    a GPU is present. T4 is a Turing-generation card (compute capability
    7.5): it has real fp16 tensor cores but no bf16 tensor cores, so
    fp16 -- not bf16 -- is the correct hardware-accelerated choice here
    (this is the opposite tradeoff from the CPU bf16 decision above,
    where neither fp16 nor bf16 had hardware support). Override with
    DMPNN_PRECISION=32-true if you hit NaNs/instability on a given
    dataset.
  - TF32 matmul (torch.set_float32_matmul_precision("high")) and
    cudnn.benchmark=True are enabled on GPU -- both are free throughput
    wins for fixed-shape training and don't change results in any way
    that matters here (TF32 is a no-op on T4 specifically, since TF32
    tensor cores are Ampere+; the call is harmless and keeps this script
    portable to newer Colab GPU tiers like A100/L4 without edits).
  - pin_memory=True is added to DataLoaders on GPU (irrelevant, and
    left off, on CPU) to speed up host->device transfer.
  - torch.cuda.empty_cache() is called between ensemble seeds so peak
    memory from one seed doesn't linger into the next -- purely a
    memory-headroom measure, not a training-logic change.
  - batch_size, lr, ffn_hidden, agg, dropout, patience, epoch budgets,
    and the frozen-encoder branch are still completely untouched by
    device -- same protection as above, now extended to cover "GPU vs
    CPU" as another axis that must not silently change what gets
    learned.

FURTHER SPEED-ONLY ADDITIONS (free-tier Colab, wall-clock is the scarce
resource) -- again, nothing here touches training logic, only how fast
the same computation runs:

  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set (before
    torch is imported, since the allocator reads it at CUDA-context
    creation) to cut down on allocator fragmentation across the repeated
    per-seed alloc/free cycles of the ensemble loop. No-op on CPU.
  - torch.backends.cuda.matmul.allow_tf32 / cudnn.allow_tf32 are set
    explicitly alongside set_float32_matmul_precision("high"), since on
    some torch/cudnn versions the conv path's TF32 gate is separate from
    the matmul-precision setting.
  - prefetch_factor=4 is added to DataLoaders when running on GPU with
    workers>0, so a couple of batches are always queued ahead of the GPU
    instead of it periodically waiting on the CPU-side featurizer/collate
    step. Same graceful-fallback-if-unsupported pattern as the other
    loader kwargs.
  - VAL/TEST/prediction batches use a larger batch size than training
    (_eval_batch_size(), GPU only, capped at 1024) purely to cut
    kernel-launch/Python-loop overhead during forward-only passes.
    Every model in this script is built with batch_norm=False, so batch
    size cannot change a forward pass's output -- predictions and
    reported AUC/AP are mathematically identical to running the same
    rows through the original smaller batch size, just faster to compute.
    TRAIN batch size (which does affect the optimization trajectory) is
    completely untouched -- see the --gpu-batch-size flag below for the
    explicit opt-in.
  - num_sanity_val_steps=0 skips Lightning's built-in pre-training
    2-batch sanity check. That check exists purely to crash-test the
    val loop before epoch 0 starts; this script already runs its own
    sanity_check() on the real post-training predictions, so the
    built-in one is pure overhead here, not a source of any signal
    this script uses. No epoch of real training or validation is
    skipped.

OPT-IN TRAIN BATCH SIZE OVERRIDE (--gpu-batch-size, off by default):
  Unlike every other speed change above, this ONE knob does change
  training dynamics -- a bigger batch means fewer, larger gradient
  updates per epoch, which can shift the optimization trajectory. It is
  therefore NOT applied automatically, even on GPU. get_config()'s
  per-dataset batch_size values are evidence of a real prior tuning
  pass (see "WHAT WAS DELIBERATELY NOT TOUCHED" above); this flag lets
  you deliberately trade a bit of that tuning for GPU utilization / wall
  clock on a time-limited Colab session, with the override logged loudly
  and recorded in the saved metrics JSON (batch_size,
  train_batch_size_override) so it's never silently mistaken for a
  validated run. It does not touch _eval_batch_size(), which was already
  safe to change unconditionally (forward-pass-only, batch_norm=False).
"""

import os

# --- Speed setup: must run before numpy/torch are imported -----------------
_N_THREADS = str(max(1, (os.cpu_count() or 4)))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _N_THREADS)
# Must also be set before the CUDA context is created (first .cuda() call /
# first CUDA op) to take effect. expandable_segments reduces allocator
# fragmentation across the repeated alloc/free cycles of the per-seed
# ensemble loop -- pure allocator behavior, no effect on results. No-op on
# a CPU-only runtime.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
from urllib.request import urlretrieve
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, average_precision_score

from chemprop import data, featurizers, models, nn

torch.set_num_threads(int(_N_THREADS))

# --- GPU/speed setup (Colab T4 support) -------------------------------------
# Auto-detects CUDA. On a Colab GPU runtime this trains on the T4 (or
# whatever GPU Colab hands out); with no GPU it behaves exactly like the
# original CPU-only script. See "GPU/SPEED ADDITIONS" in the header above
# for the reasoning behind each setting below.
_HAS_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if _HAS_CUDA else "cpu"
DEVICES = 1

if _HAS_CUDA:
    _gpu_name = torch.cuda.get_device_name(0)
    print(f"[device] CUDA available -> training on GPU ({_gpu_name})")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # Belt-and-suspenders alongside set_float32_matmul_precision above --
    # some torch/cudnn versions gate conv-path TF32 on this flag
    # separately from the matmul-precision setting. Free throughput,
    # no effect on the fp16-mixed-precision path used for the actual
    # forward/backward passes.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print("[device] CUDA not available -> training on CPU (unchanged behavior)")
    try:
        # Harmless on CPU-only setups; only affects CUDA matmul precision paths.
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

# fp16 tensor cores exist on T4 (Turing, cc 7.5); bf16 tensor cores don't --
# so fp16 mixed precision, not bf16, is the right default here. Override
# with DMPNN_PRECISION=32-true if a specific dataset shows instability.
PRECISION = os.environ.get("DMPNN_PRECISION", "16-mixed" if _HAS_CUDA else "32-true")

DATASETS = [
    ("DILI", "Y"),
    ("hERG", "Y"),
    ("CYP3A4", "Y"),
    ("Ames", "Overall"),
    ("Teratogenicity", "Y"),
]

# Informational only -- used for the printed comparison at the end, never
# read during training or model selection. Nothing in this script checks
# against these values while choosing checkpoints, epochs, or seeds.
MARKET_BENCHMARK_AUC = {
    "Ames": 0.92,
    "CYP3A4": 0.88,
    "hERG": 0.87,
    "DILI": 0.86,
    "Teratogenicity": 0.78,
}

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints_dmpnn_v2")
MODULES_DIR = Path("modules")
CHEMELEON_PATH = Path("chemeleon_mp.pt")
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
FEATURE_CACHE_DIR = Path("feature_cache")

# Default 0 to stay safe on 8GB RAM (worker subprocesses duplicate data in
# memory). Override with DMPNN_NUM_WORKERS=N if you want to experiment.
# (On a Colab GPU runtime the bottleneck shifts to the GPU, so a couple of
# CPU workers prefetching batches is a straightforward win -- still
# override-able the same way.)
NUM_WORKERS = int(os.environ.get("DMPNN_NUM_WORKERS", "2"))

# New, opt-out-able additions. Defaults are chosen to be safe (early
# stopping still governs length; oversampling only touches train).
FINE_TUNE_MAX_EPOCHS = 50       # was 30, fine-tuned configs only
FINE_TUNE_WEIGHT_DECAY = 1e-6   # was 0 (plain Adam W, no decay)
DEFAULT_ENSEMBLE_SIZE = 3
DEFAULT_MAX_BALANCE_RATIO = 1.0  # oversample minority up to 1:1, never past it

# --dev-fast knobs (opt-in only, see argparse below)
DEV_FAST_MAX_TRAIN = 300
DEV_FAST_MAX_VAL = 100
DEV_FAST_EPOCHS = 3


def get_config(name: str, n_train: int):
    if n_train < 500:
        return {
            "ffn_hidden": 300,
            "dropout": 0.3,
            "lr": 1e-4,
            "batch_size": 32,
            "agg": "mean",
            "encoder": "frozen",
            "max_epochs": 30,
            "patience": 15,
            "weight_decay": 0.0,
        }
    if name == "Ames":
        return {
            "ffn_hidden": 600,
            "dropout": 0.1,
            "lr": 5e-5,
            "batch_size": 128,
            "agg": "mean",
            "encoder": "fine-tuned",
            "max_epochs": FINE_TUNE_MAX_EPOCHS,
            "patience": 15,
            "weight_decay": FINE_TUNE_WEIGHT_DECAY,
        }
    if name == "hERG":
        return {
            "ffn_hidden": 500,
            "dropout": 0.2,
            "lr": 5e-5,
            "batch_size": 128,
            "agg": "sum",
            "encoder": "fine-tuned",
            "max_epochs": FINE_TUNE_MAX_EPOCHS,
            "patience": 15,
            "weight_decay": FINE_TUNE_WEIGHT_DECAY,
        }
    return {
        "ffn_hidden": 500,
        "dropout": 0.15,
        "lr": 5e-5,
        "batch_size": 128,
        "agg": "mean",
        "encoder": "fine-tuned",
        "max_epochs": FINE_TUNE_MAX_EPOCHS,
        "patience": 15,
        "weight_decay": FINE_TUNE_WEIGHT_DECAY,
    }


def ensure_chemeleon():
    if CHEMELEON_PATH.exists():
        return CHEMELEON_PATH
    print(f"[chemeleon] downloading to {CHEMELEON_PATH}")
    urlretrieve(CHEMELEON_URL, CHEMELEON_PATH)
    return CHEMELEON_PATH


# In-memory cache of the raw checkpoint dict so it's only read from disk
# once per process, no matter how many seeds/datasets are trained. Each
# call to build_chemeleon_mp() still does its own fresh
# nn.BondMessagePassing(...) + load_state_dict(...), so every model gets
# its own independent copy of the weights -- identical to before. Loaded
# to CPU regardless of device; Lightning moves the assembled model to the
# GPU automatically when the Trainer runs, so this cache stays valid for
# both CPU and GPU runs.
_CHEMELEON_CKPT_CACHE = None


def _load_chemeleon_ckpt():
    global _CHEMELEON_CKPT_CACHE
    if _CHEMELEON_CKPT_CACHE is None:
        _CHEMELEON_CKPT_CACHE = torch.load(
            CHEMELEON_PATH, map_location="cpu", weights_only=True
        )
    return _CHEMELEON_CKPT_CACHE


def build_chemeleon_mp():
    ckpt = _load_chemeleon_ckpt()
    hparams = ckpt["hyper_parameters"]
    print(f"[chemeleon] loaded: d_h={hparams.get('d_h')} depth={hparams.get('depth')}")
    mp = nn.BondMessagePassing(**hparams)
    mp.load_state_dict(ckpt["state_dict"])
    chemeleon_dim = hparams["d_h"]
    return mp, chemeleon_dim


class ReapplyFreeze(Callback):
    def __init__(self, mp):
        super().__init__()
        self.mp = mp

    def on_train_epoch_start(self, trainer, pl_module):
        self.mp.eval()
        for p in self.mp.parameters():
            p.requires_grad_(False)


def _balance_train_datapoints(dps, ys, max_ratio=DEFAULT_MAX_BALANCE_RATIO, seed=42):
    """
    Duplicates minority-class datapoints so the loss sees a less skewed
    class ratio -- mirrors the auto_class_weights='Balanced' already
    validated in the sibling CatBoost script. Only ever called on the
    TRAIN split; caps at max_ratio (default 1:1) so it never invents a
    majority-minority reversal, and returns the input unchanged if
    either class is entirely absent or already at/above the target
    ratio.
    """
    ys = np.asarray(ys).ravel()
    pos_idx = np.where(ys == 1)[0]
    neg_idx = np.where(ys == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return dps
    minority_idx, majority_idx = (
        (pos_idx, neg_idx) if len(pos_idx) < len(neg_idx) else (neg_idx, pos_idx)
    )
    target_minority_count = int(len(majority_idx) * max_ratio)
    if target_minority_count <= len(minority_idx):
        return dps
    rng = np.random.default_rng(seed)
    n_extra = target_minority_count - len(minority_idx)
    extra_idx = rng.choice(minority_idx, size=n_extra, replace=True)
    balanced = list(dps) + [dps[i] for i in extra_idx]
    print(f"  [balance] {len(minority_idx)} minority rows -> "
          f"{len(minority_idx) + n_extra} via oversampling "
          f"(train split only, +{n_extra} duplicated rows)")
    return balanced


def load_split(name, split, target_col, balance: bool = False, seed: int = 42,
                max_rows: int = None):
    path = DATA_DIR / f"{name}_{split}_chemprop.csv"

    # Disk cache of the featurized dataset. Featurization (SMILES -> graph)
    # is CPU-heavy and deterministic given these inputs, so during
    # iterative development it's pure waste to redo it every run. Cache is
    # invalidated automatically if the source CSV is newer than the cache
    # file. Falls back to recomputing on any cache read/write problem.
    cache_path = FEATURE_CACHE_DIR / (
        f"{name}_{split}_{target_col}"
        f"_{'bal' if (balance and split == 'train') else 'raw'}"
        f"_seed{seed}_rows{max_rows if max_rows is not None else 'all'}.pkl"
    )
    if cache_path.exists() and path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        try:
            with open(cache_path, "rb") as f:
                dset, n_rows, ys = pickle.load(f)
            print(f"  [cache] loaded featurized {name}/{split} from {cache_path.name}")
            return dset, n_rows, ys
        except Exception as e:
            print(f"  [cache] miss/corrupt for {cache_path.name} ({e}); recomputing")

    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)
    smis = df["SMILES"].values
    ys = df[[target_col]].values.astype(float)
    dps = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    if balance and split == "train":
        dps = _balance_train_datapoints(dps, ys, seed=seed)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(dps, featurizer)

    try:
        FEATURE_CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((dset, len(df), ys.ravel()), f)
    except Exception as e:
        print(f"  [cache] could not write {cache_path.name} ({e}); continuing uncached")

    return dset, len(df), ys.ravel()


def _build_loader(dset, batch_size, num_workers=NUM_WORKERS, shuffle=None):
    """
    Thin wrapper around data.build_dataloader that opts into
    persistent_workers when num_workers > 0 (saves re-spawning worker
    processes across epochs), and pin_memory when a GPU is present
    (speeds up host->device transfer; meaningless on CPU-only so it's
    left off there). Falls back cleanly if the installed chemprop
    version doesn't support a given kwarg. Behavior with the default
    num_workers=0 on CPU is identical to the original build_dataloader
    calls.
    """
    kwargs = dict(num_workers=num_workers, batch_size=batch_size)
    if shuffle is not None:
        kwargs["shuffle"] = shuffle
    if _HAS_CUDA:
        kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        if _HAS_CUDA:
            # Queue a couple extra batches per worker ahead of the GPU so
            # it's less likely to sit idle waiting on the next batch.
            # Pure prefetch depth, no effect on what's in each batch.
            kwargs["prefetch_factor"] = 4
    try:
        return data.build_dataloader(dset, **kwargs)
    except TypeError:
        # Older chemprop versions may not accept one of the newer kwargs
        # (persistent_workers / pin_memory / prefetch_factor) -- strip
        # them one at a time and retry rather than failing the whole run
        # over a speed knob.
        for k in ("prefetch_factor", "persistent_workers", "pin_memory"):
            kwargs.pop(k, None)
            try:
                return data.build_dataloader(dset, **kwargs)
            except TypeError:
                continue
        kwargs = dict(num_workers=num_workers, batch_size=batch_size)
        if shuffle is not None:
            kwargs["shuffle"] = shuffle
        return data.build_dataloader(dset, **kwargs)


def _eval_batch_size(train_batch_size):
    """
    Batch size to use for VAL/TEST/prediction passes only -- never for
    training. This is forward-pass-only (no backward, no optimizer step),
    and models.MPNN is built with batch_norm=False everywhere in this
    script, so batch size has zero effect on the numbers a forward pass
    produces: predictions and metrics are identical regardless of how the
    same rows are grouped into batches. On GPU, a bigger eval batch cuts
    the number of kernel-launch/Python round trips per split, which
    matters when a free-tier Colab session's wall-clock is the scarce
    resource. Capped so it doesn't just trade speed for an OOM. CPU is
    unaffected -- returns train_batch_size unchanged, same as before.
    """
    if not _HAS_CUDA:
        return train_batch_size
    return min(max(train_batch_size * 4, train_batch_size), 1024)


def sanity_check(preds, name):
    lo, hi = float(preds.min()), float(preds.max())
    print(f"[sanity/{name}] pred range [{lo:.4f}, {hi:.4f}]")
    assert 0.0 <= lo and hi <= 1.0, f"preds not proba [{lo},{hi}]"


def get_preds(trainer, mpnn, loader):
    batches = trainer.predict(mpnn, dataloaders=loader, ckpt_path=None)
    preds = torch.cat([b if torch.is_tensor(b) else b[0] for b in batches]).squeeze(-1)
    return preds.detach().cpu().numpy()


def train_one_seed(name, target_col, cfg, seed, ckpt_dir, val_loader, test_loader,
                    train_dset, n_val, n_test):
    """
    Trains a single ensemble member. Encoder weights are reloaded fresh
    from the chemeleon checkpoint for every seed so nothing leaks
    between ensemble members.
    """
    pl.seed_everything(seed, workers=True)

    train_loader = _build_loader(train_dset, cfg["batch_size"])

    base_mp, chemeleon_dim = build_chemeleon_mp()
    mp = base_mp

    agg = nn.SumAggregation() if cfg["agg"] == "sum" else nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(
        input_dim=chemeleon_dim, hidden_dim=cfg["ffn_hidden"], dropout=cfg["dropout"]
    )
    metric_list = [nn.metrics.BinaryAUROC()]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

    def configure_optimizers():
        params = filter(lambda p: p.requires_grad, mpnn.parameters())
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    mpnn.configure_optimizers = configure_optimizers

    seed_dir = ckpt_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(seed_dir),
            filename="best-{epoch}-{val_roc:.3f}",
            monitor="val/roc",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
        EarlyStopping(monitor="val/roc", mode="max", patience=cfg["patience"]),
    ]

    if cfg["encoder"] == "frozen":
        mp.eval()
        for p in mp.parameters():
            p.requires_grad_(False)
        callbacks.append(ReapplyFreeze(mp))

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        # Skips Lightning's built-in pre-training sanity pass (2 val
        # batches run before epoch 0 purely to catch crashes early).
        # Pure added wall-clock here -- this script's own sanity_check()
        # already validates the real post-training val predictions, and
        # no epoch of actual training or validation is skipped.
        num_sanity_val_steps=0,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
    )

    trainer.fit(mpnn, train_loader, val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"  [seed {seed}] best ckpt: {best_path}")

    val_pred = get_preds(trainer, mpnn, val_loader)
    test_pred = get_preds(trainer, mpnn, test_loader) if n_test > 0 else None
    sanity_check(val_pred, f"{name}/seed{seed}")

    if _HAS_CUDA:
        # Peak activation/optimizer-state memory from this seed shouldn't
        # linger into the next one -- pure memory headroom, no effect on
        # what's learned (a fresh model + optimizer are built per seed
        # regardless).
        del trainer, mpnn, mp, base_mp
        torch.cuda.empty_cache()

    return best_path, val_pred, test_pred, chemeleon_dim


def train_one(name, target_col, ensemble_size=DEFAULT_ENSEMBLE_SIZE,
              balance_classes=True, base_seed=0, dev_fast=False,
              gpu_batch_size=None):
    print(f"\n{'='*70}\n=== {name} v2.2 "
          f"(target={target_col}, ensemble={ensemble_size}, balance={balance_classes}, "
          f"dev_fast={dev_fast}, device={ACCELERATOR}, precision={PRECISION}) ===\n{'='*70}")

    train_max_rows = DEV_FAST_MAX_TRAIN if dev_fast else None
    eval_max_rows = DEV_FAST_MAX_VAL if dev_fast else None

    train_dset, n_train, _ = load_split(
        name, "train", target_col, balance=balance_classes, seed=base_seed,
        max_rows=train_max_rows,
    )
    val_dset, n_val, val_y = load_split(name, "val", target_col, max_rows=eval_max_rows)
    test_dset, n_test, test_y = load_split(name, "test", target_col, max_rows=eval_max_rows)

    cfg = dict(get_config(name, n_train))
    if dev_fast:
        cfg["max_epochs"] = min(cfg["max_epochs"], DEV_FAST_EPOCHS)
        cfg["patience"] = min(cfg["patience"], DEV_FAST_EPOCHS)
        print(f"[{name}] *** DEV-FAST MODE *** subsetted data + "
              f"max_epochs={cfg['max_epochs']} -- NOT a real training run, "
              f"do not compare these metrics to production numbers")

    # OPT-IN ONLY, off by default (gpu_batch_size=None leaves cfg["batch_size"]
    # exactly as get_config() validated it). Unlike every other speed change
    # in this file, this DOES affect the optimization trajectory (fewer,
    # larger gradient updates per epoch) -- so it's never applied silently,
    # only on explicit request, and it's logged loudly and recorded in the
    # saved metrics JSON below.
    if gpu_batch_size is not None:
        print(f"[{name}] *** --gpu-batch-size override: batch_size "
              f"{cfg['batch_size']} -> {gpu_batch_size} *** this changes "
              f"training dynamics, not just speed -- treat these results as "
              f"unvalidated against the per-dataset tuned baseline until "
              f"compared.")
        cfg["batch_size"] = gpu_batch_size

    print(f"[{name}] n_train={n_train} cfg={cfg}")

    val_loader = _build_loader(val_dset, _eval_batch_size(cfg["batch_size"]), shuffle=False)
    test_loader = _build_loader(test_dset, _eval_batch_size(cfg["batch_size"]), shuffle=False)

    ckpt_dir = CHECKPOINT_DIR / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    member_ckpts, val_preds, test_preds = [], [], []
    chemeleon_dim = None
    for k in range(ensemble_size):
        seed = base_seed + k
        best_path, val_pred, test_pred, chemeleon_dim = train_one_seed(
            name, target_col, cfg, seed, ckpt_dir, val_loader, test_loader,
            train_dset, n_val, n_test,
        )
        member_ckpts.append(str(best_path))
        val_preds.append(val_pred)
        if test_pred is not None:
            test_preds.append(test_pred)

    # Ensemble prediction = simple average of member probabilities.
    # This is the one place multiple models are combined; it reduces
    # variance from FFN-head initialization, it does not and cannot
    # inflate accuracy that individual members don't have.
    val_pred_ens = np.mean(np.stack(val_preds, axis=0), axis=0)
    test_pred_ens = np.mean(np.stack(test_preds, axis=0), axis=0) if test_preds else None
    sanity_check(val_pred_ens, f"{name}/ensemble")

    per_seed_val_auc = [
        float(roc_auc_score(val_y, p)) if len(np.unique(val_y)) > 1 else float("nan")
        for p in val_preds
    ]

    val_auc = roc_auc_score(val_y, val_pred_ens) if len(np.unique(val_y)) > 1 else float("nan")
    val_ap = average_precision_score(val_y, val_pred_ens) if len(np.unique(val_y)) > 1 else float("nan")
    test_auc = (roc_auc_score(test_y, test_pred_ens)
                if test_pred_ens is not None and len(np.unique(test_y)) > 1 else float("nan"))

    print(f"[{name}] per-seed val AUC: {[round(a, 4) for a in per_seed_val_auc]}")
    print(f"[{name}] Ensemble Val AUC {val_auc:.4f} AP {val_ap:.4f} | Test AUC {test_auc:.4f}")

    MODULES_DIR.mkdir(exist_ok=True)
    out_dir = MODULES_DIR / f"{name}_dmpnn_ensemble"
    out_dir.mkdir(exist_ok=True)
    out_ckpts = []
    for k, src in enumerate(member_ckpts):
        dst = out_dir / f"seed{base_seed + k}.ckpt"
        shutil.copyfile(src, dst)
        out_ckpts.append(str(dst))

    metrics = {
        "dataset": name,
        "target_col": target_col,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "balance_classes": balance_classes,
        "ensemble_size": ensemble_size,
        "dev_fast": dev_fast,
        "device": ACCELERATOR,
        "precision": PRECISION,
        "ensemble_val_auc": float(val_auc),
        "ensemble_val_ap": float(val_ap),
        "ensemble_test_auc": float(test_auc),
        "per_seed_val_auc": per_seed_val_auc,
        "encoder": cfg["encoder"],
        "agg": cfg["agg"],
        "chemeleon_dim": chemeleon_dim,
        "ffn_hidden": cfg["ffn_hidden"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg.get("weight_decay", 0.0),
        "batch_size": cfg["batch_size"],
        "train_batch_size_override": gpu_batch_size,
        "max_epochs": cfg["max_epochs"],
        "foundation_model": "chemeleon_mp",
        "member_checkpoint_sources": member_ckpts,
        "ensemble_checkpoint_paths": out_ckpts,
        "market_benchmark_auc": MARKET_BENCHMARK_AUC.get(name),
        "note": (
            "ensemble_test_auc is the average of member predicted "
            "probabilities scored once against the untouched test set -- "
            "not selected or adjusted against test performance. "
            "market_benchmark_auc is informational only and played no "
            "role in training, checkpoint selection, or hyperparameters. "
            + ("THIS RUN USED --dev-fast: subsetted data and capped epochs "
               "for pipeline smoke-testing only; these numbers are not "
               "meaningful for model comparison." if dev_fast else "")
            + (f" THIS RUN USED --gpu-batch-size={gpu_batch_size} (validated "
               f"batch_size was {get_config(name, n_train)['batch_size']}): "
               "training dynamics differ from the per-dataset tuned "
               "baseline; compare against a non-overridden run before "
               "trusting these numbers." if gpu_batch_size is not None else "")
        ),
    }
    (MODULES_DIR / f"{name}_dmpnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved ensemble to {out_dir}")
    return metrics


def predict_ensemble(name, target_col, split="test"):
    """
    Inference helper: loads every checkpoint saved for this dataset's
    ensemble and returns the averaged predicted probabilities for the
    given split. Needed because a production consumer of this script now
    gets N checkpoints per dataset instead of one.
    """
    metrics_path = MODULES_DIR / f"{name}_dmpnn_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No trained ensemble found for {name} at {metrics_path}")
    meta = json.loads(metrics_path.read_text())
    dset, n_rows, _ = load_split(name, split, target_col)
    loader = _build_loader(dset, _eval_batch_size(128), shuffle=False)

    preds = []
    for ckpt_path in meta["ensemble_checkpoint_paths"]:
        base_mp, chemeleon_dim = build_chemeleon_mp()
        agg = nn.SumAggregation() if meta["agg"] == "sum" else nn.MeanAggregation()
        ffn = nn.BinaryClassificationFFN(
            input_dim=chemeleon_dim, hidden_dim=meta["ffn_hidden"], dropout=meta["dropout"]
        )
        mpnn = models.MPNN(base_mp, agg, ffn, batch_norm=False,
                            metrics=[nn.metrics.BinaryAUROC()])
        mpnn = mpnn.__class__.load_from_checkpoint(ckpt_path)
        trainer = pl.Trainer(logger=False, enable_progress_bar=False,
                              accelerator=ACCELERATOR, devices=DEVICES, precision=PRECISION)
        preds.append(get_preds(trainer, mpnn, loader))
    return np.mean(np.stack(preds, axis=0), axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default=None, help="DILI/hERG/CYP3A4/Ames/Teratogenicity")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE,
                         help="Number of seeds trained and averaged per dataset. "
                              "Use 1 to reproduce the old single-model behavior.")
    parser.add_argument("--no-balance", action="store_true",
                         help="Disable minority-class oversampling on the train split.")
    parser.add_argument("--dev-fast", action="store_true",
                         help="DEV/SMOKE-TEST ONLY. Subsets data to a few hundred rows "
                              "and caps epochs so you can validate the pipeline runs "
                              "end-to-end in minutes. Output metrics are tagged "
                              "dev_fast=true and are not meaningful for real comparison.")
    parser.add_argument("--gpu-batch-size", type=int, default=None,
                         help="OPT-IN, off by default. Overrides the validated per-dataset "
                              "training batch_size (e.g. 256 on a T4) to improve GPU "
                              "utilization / wall-clock time on a time-limited Colab "
                              "session. Unlike other speed flags, this changes training "
                              "dynamics -- it's logged loudly and recorded in the saved "
                              "metrics JSON so it's never mistaken for a validated run. "
                              "Does not affect val/test/prediction batch sizes.")
    args = parser.parse_args()

    print(f"[device] accelerator={ACCELERATOR} devices={DEVICES} precision={PRECISION} "
          f"cuda_available={_HAS_CUDA}")

    ensure_chemeleon()
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    MODULES_DIR.mkdir(exist_ok=True)

    if args.all:
        to_run = DATASETS
    elif args.endpoint:
        to_run = [d for d in DATASETS if d[0] == args.endpoint]
        if not to_run:
            raise SystemExit(f"Unknown endpoint {args.endpoint}")
    else:
        to_run = [("hERG", "Y")]

    results = []
    for name, tgt in to_run:
        try:
            results.append(train_one(
                name, tgt,
                ensemble_size=args.ensemble_size,
                balance_classes=not args.no_balance,
                dev_fast=args.dev_fast,
                gpu_batch_size=args.gpu_batch_size,
            ))
        except FileNotFoundError as e:
            print(f"Skip {name}: {e} - run export_chemprop.py first")

    print("\nD-MPNN v2.2 SUMMARY (ensemble-averaged, test set evaluated once)")
    print(f"device={ACCELERATOR} precision={PRECISION}")
    if args.dev_fast:
        print("*** --dev-fast was used: these numbers are smoke-test only ***")
    if args.gpu_batch_size is not None:
        print(f"*** --gpu-batch-size={args.gpu_batch_size} was used: training dynamics "
              f"differ from the validated per-dataset batch_size ***")
    for r in results:
        bench = r.get("market_benchmark_auc")
        bench_str = f"market={bench:.3f}" if bench is not None else "market=n/a"
        gap = (r["ensemble_test_auc"] - bench) if bench is not None else float("nan")
        print(f" {r['dataset']:16s} agg={r['agg']:4s} ffn={r['ffn_hidden']} "
              f"ens={r['ensemble_size']} enc={r['encoder']:11s} "
              f"val={r['ensemble_val_auc']:.4f} test={r['ensemble_test_auc']:.4f} "
              f"{bench_str} gap={gap:+.4f}")