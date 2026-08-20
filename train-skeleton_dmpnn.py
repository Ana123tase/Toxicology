"""
train-skeleton_dmpnn.py

Trains one single-task D-MPNN per endpoint (DILI, hERG, CYP3A4, Ames,
Teratogenicity), using chemprop's Python API directly against the
already-exported, already-scaffold-split CSVs from export_chemprop.py.
No CLI commands, no chemprop's own splitting -- train/val/test
membership is exactly what scaffold_split.py decided, so results stay
directly comparable to CatBoost's benchmark_p1 numbers.

STILL A SKELETON, not the production trainer. Its job is a fast,
single-file health check: does the full chemprop pipeline (CheMeleon
load -> freeze policy -> fit -> checkpoint -> predict) run cleanly
end-to-end on your real, already-exported data? Output is one flat
dmpnn_benchmark_p2.txt.

The actual production D-MPNN trainer -- the one that produces
modules/{name}_dmpnn_metrics.json in the same format as
train_catboost.py's metrics.json, evaluated with the same sklearn
functions CatBoost's script uses, one clean model artifact per
endpoint in modules/ -- is train_dmpnn.py, not this file. Run this
skeleton first (fast, catches broken environments/data early); trust
train_dmpnn.py's numbers for anything you'd report or feed into
Phase 2's ensemble.

Decisions locked in for this script (mirrors train_skeleton.py's
"Decisions locked in" convention):
  - Task structure: single-task per endpoint (matches CatBoost baseline)
  - Ensemble size: 1 (ensembling is a later, separate step)
  - Foundation model: CheMeleon pretrained message-passing encoder,
    loaded for every endpoint
  - Encoder freeze policy: size-driven, not blanket. Endpoints with a
    small train split (< FREEZE_THRESHOLD_TRAIN_ROWS rows) use a
    FROZEN CheMeleon encoder -- fine-tuning the full 8.7M-parameter
    MPNN on ~89-370 molecules risks overfitting the encoder itself,
    not just the FFN head. Endpoints with enough data get an
    UNFROZEN (fine-tuned) encoder.
  - Evaluation checkpoint: best-val-loss checkpoint (ckpt_path="best"),
    not whatever's in memory when early stopping fires -- matches
    CatBoost's use_best_model=True so the two aren't compared unfairly.

USAGE: python train-skeleton_dmpnn.py
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from chemprop import data, featurizers, models, nn

# (dataset_name, target_col) -- matches export_chemprop.py / scaffold_split.py
DATASETS = [
    ("DILI", "Y"),
    ("hERG", "Y"),
    ("CYP3A4", "Y"),
    ("Ames", "Overall"),
    ("Teratogenicity", "Y"),
]

# Set this to a subset of names (e.g. ["Teratogenicity"]) to smoke-test
# one endpoint at a time before running the full sweep. None = run all.
RUN_ONLY = None

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints_dmpnn")
CHEMELEON_PATH = Path("chemeleon_mp.pt")
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

# Train-split row count below which the CheMeleon encoder is frozen
# instead of fine-tuned. 1000 cleanly separates DILI (370) and
# Teratogenicity (89) from hERG/CYP3A4/Ames (thousands) given your
# actual split sizes.
FREEZE_THRESHOLD_TRAIN_ROWS = 1000

NUM_WORKERS = 0          # required 0 on Windows -- see chemprop --help warning
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5  # epochs without val_loss improvement before stopping


def ensure_chemeleon_downloaded() -> Path:
    """Download the CheMeleon checkpoint once; reuse the cached file after."""
    if CHEMELEON_PATH.exists():
        print(f"[chemeleon] using cached checkpoint: {CHEMELEON_PATH}")
        return CHEMELEON_PATH
    print(f"[chemeleon] downloading checkpoint to {CHEMELEON_PATH} ...")
    urlretrieve(CHEMELEON_URL, CHEMELEON_PATH)
    print("[chemeleon] download complete")
    return CHEMELEON_PATH


def build_chemeleon_message_passing() -> nn.BondMessagePassing:
    """
    Load a fresh CheMeleon-initialized BondMessagePassing module.
    Built fresh per endpoint so freezing one endpoint's copy can never
    accidentally affect another endpoint's training.
    """
    checkpoint = torch.load(CHEMELEON_PATH, weights_only=True)
    mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
    mp.load_state_dict(checkpoint["state_dict"])
    return mp


class ReapplyFreeze(Callback):
    """
    Lightning's Trainer calls model.train() at the start of every
    training epoch, which would silently undo a one-time mp.eval()
    call. requires_grad_(False) alone still blocks weight updates, but
    to keep the encoder's dropout/eval behavior consistent throughout
    training too (not just at epoch 0), this callback re-applies both
    at the start of every epoch for frozen-encoder runs.
    """

    def __init__(self, mp: nn.BondMessagePassing):
        self.mp = mp

    def on_train_epoch_start(self, trainer, pl_module):
        self.mp.eval()
        self.mp.apply(lambda module: module.requires_grad_(False))


def load_split_csv(name: str, split: str, target_col: str):
    path = DATA_DIR / f"{name}_{split}_chemprop.csv"
    df = pd.read_csv(path)
    smis = df["SMILES"].values
    ys = df[[target_col]].values
    datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # FIX: your installed chemprop version's MoleculeDataset.__init__()
    # doesn't accept a `cache` keyword argument (that constructor param
    # was added/removed across chemprop v2 releases -- version skew
    # between what this script was written against and what's actually
    # in your env). MoleculeDataset exposes `cache` as a settable
    # property instead: constructing without it defaults to cache=False
    # (recompute featurized graphs every epoch), and setting dset.cache
    # = True afterward pre-computes and stores them, which is the exact
    # same "RAM for speed" behavior the original cache=True kwarg gave
    # us. Wrapped in try/except so this keeps working even if a future
    # chemprop release removes the property too -- it just silently
    # falls back to no caching (slower, but never a crash) instead of
    # failing the whole run.
    dset = data.MoleculeDataset(datapoints, featurizer)
    try:
        dset.cache = True
    except AttributeError:
        print(f"[{name}/{split}] this chemprop version has no MoleculeDataset.cache "
              f"property -- continuing without graph caching (slower, not incorrect)")

    return dset, len(df)


# NOTE: batch size is intentionally left at chemprop's default, not
# scaled by n_train. A larger batch size trains faster on CPU but is
# NOT strictly accuracy-neutral -- it changes how many gradient
# updates happen per epoch and the noise per update, which can shift
# convergence (most visible on small splits like Teratogenicity's 89
# rows). Since accuracy and comparability to the CatBoost baseline
# outrank speed for this run, batch size stays untouched. The cache
# setting above is the only speedup applied: it changes nothing about
# training, it only avoids recomputing the same featurized graphs
# every epoch.


def _extract_auc_like_metric(results: list) -> tuple:
    """
    trainer.test() returns a list of dicts whose exact key names
    (e.g. "test/roc" vs "test/auc" vs something else) depend on the
    installed chemprop/Lightning version. Rather than hardcoding one
    key and silently returning None if it's wrong (which the earlier
    version of this script did), scan for any key containing "roc" or
    "auc" and report both the value and which key matched -- so a
    version mismatch is visible in the printed output instead of
    silently producing an N/A that looks like a missing test set.
    """
    if not results:
        return None, None
    d = results[0]
    for key, value in d.items():
        lowered = key.lower()
        if "roc" in lowered or "auc" in lowered:
            return value, key
    return None, None


def train_one_endpoint(name: str, target_col: str) -> dict:
    print(f"\n{'=' * 70}")
    print(f"=== {name} (target_col={target_col}) ===")
    print('=' * 70)

    train_dset, n_train = load_split_csv(name, "train", target_col)
    val_dset, n_val = load_split_csv(name, "val", target_col)
    test_dset, n_test = load_split_csv(name, "test", target_col)

    freeze_encoder = n_train < FREEZE_THRESHOLD_TRAIN_ROWS
    print(f"[{name}] n_train={n_train} n_val={n_val} n_test={n_test} "
          f"-> encoder {'FROZEN' if freeze_encoder else 'FINE-TUNED'}")

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader = data.build_dataloader(val_dset, num_workers=NUM_WORKERS, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

    mp = build_chemeleon_message_passing()
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim)
    metric_list = [nn.metrics.BinaryAUROC()]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

    callbacks = [
        ModelCheckpoint(
            str(CHECKPOINT_DIR / name),
            "best-{epoch}-{val_loss:.3f}",
            "val_loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOP_PATIENCE),
    ]

    if freeze_encoder:
        mp.eval()
        mp.apply(lambda module: module.requires_grad_(False))
        callbacks.append(ReapplyFreeze(mp))

    # PERF: no-op on your current machine (no GPU), but harmless to leave
    # in -- if this script is ever run on a CUDA machine, it'll pick up
    # the GPU and mixed precision automatically without code changes.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        precision="16-mixed" if accelerator == "gpu" else 32,
    )

    trainer.fit(mpnn, train_loader, val_loader)

    # FIX: no such argument as weights_only on Trainer.test() as a
    # public kwarg pass-through in older versions -- it IS supported in
    # current Lightning and we now set it explicitly to False. Lightning's
    # checkpoint restore otherwise defaults to torch.load(weights_only=True)
    # (PyTorch 2.6+ default), which blocks unpickling chemprop/torchmetrics
    # internals (e.g. BinaryAUROC, torchmetrics.metric.jit_distributed_available)
    # stored in the checkpoint's hyperparameters/metric state. Allowlisting
    # each blocked global one at a time doesn't scale -- new metrics or
    # torchmetrics internals keep surfacing new unpicklable globals. Since
    # this checkpoint was written by this same script in this same run
    # (fully trusted, not an external file), it's safe to bypass the
    # weights-only restriction entirely for this restore. Also still
    # explicitly evaluates the BEST checkpoint by val_loss, not whatever
    # is left in memory after early stopping -- matches CatBoost's
    # use_best_model=True so the comparison across models is fair.
    results = trainer.test(
        dataloaders=test_loader,
        ckpt_path="best",
        weights_only=False,  # trusted: checkpoint written by this same run
    )

    test_auc, matched_key = _extract_auc_like_metric(results)
    if test_auc is not None:
        print(f"[{name}] Test AUC ({matched_key}): {test_auc:.4f}")
    else:
        print(f"[{name}] Could not find an AUC-like key in test results. "
              f"Full results dict: {results}")

    return {
        "dataset": name,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "encoder": "frozen" if freeze_encoder else "fine-tuned",
        "test_auc": test_auc,
        "test_auc_key": matched_key,
    }


if __name__ == "__main__":
    ensure_chemeleon_downloaded()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    datasets_to_run = DATASETS if RUN_ONLY is None else [
        d for d in DATASETS if d[0] in RUN_ONLY
    ]

    all_results = []
    for name, target_col in datasets_to_run:
        result = train_one_endpoint(name, target_col)
        all_results.append(result)

    print(f"\n{'=' * 70}")
    print("D-MPNN SUMMARY (single-task, ensemble-size=1)")
    print('=' * 70)
    for r in all_results:
        auc_str = f"{r['test_auc']:.4f}" if r["test_auc"] is not None else "N/A"
        print(f"  {r['dataset']:16s} encoder={r['encoder']:11s} "
              f"n_train={r['n_train']:6d}  test_auc={auc_str}")

    report_path = Path("dmpnn_benchmark_p2.txt")
    with open(report_path, "w") as fh:
        fh.write("D-MPNN single-task results (ensemble_size=1, CheMeleon foundation model)\n")
        fh.write("SKELETON RUN -- see train_dmpnn.py for production metrics.json output\n")
        fh.write("=" * 70 + "\n")
        for r in all_results:
            auc_str = f"{r['test_auc']:.4f}" if r["test_auc"] is not None else "N/A"
            fh.write(f"{r['dataset']:16s} encoder={r['encoder']:11s} "
                      f"n_train={r['n_train']:6d}  test_auc={auc_str}\n")
    print(f"\nSaved summary: {report_path}")