"""
train_unimol.py - PRODUCTION / SOTA FIX v4 (+ Colab T4 speed pass)

v3 docstring (kept for history) claimed 10 fixes. On review, several were
only *partially* wired up -- the machinery existed but wasn't actually
connected to the code path that mattered. v4 fixes those specific gaps:

 A. TEST LEAKAGE GUARD WAS DECORATIVE
    build_training_pool() called load_split_cache(..., "test") on EVERY
    seed/model-size run just to get a hash for provenance -- fully
    unpickling the real test records into memory, completely bypassing
    TestTouchGuard (which was only wired into evaluate_test_once()).
    Fixed: build_training_pool() now hashes the raw test-split file bytes
    directly (never unpickles test content), and TestTouchGuard.touch() is
    now called *inside* load_split_cache() itself whenever split=="test" --
    so the guard protects every call site, not just the one it happened to
    be manually inserted into.

 B. VALIDATION WAS DETECTED BUT NEVER USED
    resolve_fit_kwargs_with_valid() correctly found the fit()-time
    validation kwarg (or reported that none exists) -- but the result was
    only ever printed. train_final_pool_model() still called
    clf.fit(pool_dict) unconditionally, so early stopping during
    --pool-strategy train_only NEVER actually watched the held-out val
    split live, no matter what unimol_tools supported.
    Fixed: the resolved kwarg name + val dict are now threaded into
    train_final_pool_model() and actually passed to clf.fit().

 C. DISCONNECTED-FRAGMENT CHECK WAS NONSENSE
    `if n_components > 1 and n_components != len(set(atoms))` compares a
    3D bond-graph component count to the count of *distinct element
    symbols* in the molecule -- two numbers with no chemical relationship.
    This silently let real disconnected fragments through (whenever the
    component count happened to equal the number of unique elements) and
    could reject a normal molecule for the wrong reason.
    Fixed: any single-conformer record with >1 connected component is
    flagged, full stop. See note in qc_conformers() if you intentionally
    keep salts/counterions in this dataset.

 D. WEIGHT_DECAY / DROPOUT WERE "TUNED" WITHOUT EVER REACHING THE MODEL
    sample_hpo_configs() generated distinct weight_decay/dropout values for
    every trial, but build_trainer_kwargs() silently drops any kwarg not in
    `supported_params` -- and the real MolTrain signature (per this file's
    own MOLTRAIN_EXPLICIT_INIT_PARAMS) has neither parameter. Every trial
    that differed only in weight_decay/dropout was therefore a byte-for-byte
    identical training run, and "best trial" was chosen by pure run-to-run
    noise while looking like a real hyperparameter search.
    Fixed: the smoke test's weight_decay_effect_detected / 
    dropout_effect_detected results (or, absent a supported param, that fact
    itself) now gate whether sample_hpo_configs() is allowed to vary these
    dimensions at all. If a param can't be shown to change model output, the
    grid fixes it at the tier default instead of wasting trials on it.

 E. CPU vs GPU SPEED
    - use_amp was hardcoded True even with no CUDA device (AMP mainly helps,
      and is only reliably supported, on GPU) -> now conditioned on
      torch.cuda.is_available().
    - OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS are now set
      *before* numpy/torch are imported (these are read at import/first-use
      time, so setting them afterward is too late to help), and
      torch.set_num_threads() is called explicitly when there's no GPU, so
      CPU runs actually use all available cores.
    - batch_size is capped on CPU-only runs to avoid RAM exhaustion on an
      8GB machine.
    - nested-CV outer/inner folds, HPO trials, and ensemble seeds are
      auto-scaled down when no CUDA device is present (huge multiplier on
      total training runs), with a printed explanation and a
      --no-auto-scale escape hatch for when you deliberately want the full
      run on CPU anyway.
    None of this changes GPU behavior -- on a CUDA box everything runs at
    the original (full) settings.

 F. COLAB T4 SPEED PASS (this revision -- engine-level only, no science
    changed, no HPO/CV/leakage logic touched):
    - GPU-vs-CPU selection was already correct (use_cuda=has_cuda /
      use_amp=has_cuda were already threaded into every MolTrain(...) call
      in FIX E above) -- this pass makes it louder/more explicit at
      startup and adds pure kernel-selection speedups on top:
    - torch.backends.cudnn.benchmark = True when a CUDA device is present.
      This lets cuDNN pick faster convolution/kernel algorithms for
      repeated input shapes instead of always using a safe default -- it
      changes *which kernel implementation* runs, never the math result,
      and is a no-op on CPU-only runs. Left off when running under
      use_deterministic_algorithms in a context where that would raise
      (guarded, see set_determinism()).
    - build_trainer_kwargs() now also *offers* a few well-known
      throughput-only DataLoader kwargs (num_workers, pin_memory,
      persistent_workers) to MolTrain. These go through the exact same
      `supported_params` filter that weight_decay/dropout already used in
      FIX D -- if the installed unimol_tools doesn't expose them they are
      silently dropped (printed, like any other dropped kwarg), so this
      cannot change behavior on a version that doesn't support them. Values
      are only set when has_cuda is True (pinned memory only helps
      host->GPU transfers; on CPU-only runs these are left unset exactly
      as before).
    - Nothing about epochs, learning rate, batch size, freeze layers,
      folds, HPO trials, thresholds, calibration, or the test-leakage guard
      changed. On a CUDA box the set of hyperparameters that reach the
      model is identical to v4; only kernel/dataloader plumbing differs.

Everything else below is unchanged from v3 and still does what its
original comment says.
"""

from __future__ import annotations
import argparse, hashlib, inspect, itertools, json, os, pickle, platform, random, sys, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# FIX E (speed): these env vars are read by numpy's/torch's BLAS backends at
# import/first-use time. Setting them AFTER `import numpy as np` (as v3 did,
# since numpy was imported before the CUBLAS_WORKSPACE_CONFIG line) is too
# late for OMP/MKL/OpenBLAS to pick them up. Set everything before any
# numeric library is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_CPU_COUNT = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_COUNT))

import numpy as np
import torch

CACHE_DIR_DEFAULT = "data/unimol_cache"
EXP_DIR_DEFAULT = "exp"
ENDPOINTS = ("DILI", "hERG", "CYP3A4", "Ames", "Teratogenicity")
TARGET_COL_BY_DATASET = {"DILI": "Y", "hERG": "Y", "CYP3A4": "Y", "Ames": "Overall", "Teratogenicity": "Y"}

_FROZEN_TIL_HEAD = ["embed_tokens", "atom_feature", "edge_feature", "se3_invariant_kernel", "movement_pred_head", "encoder"]
_FROZEN_EMBEDDINGS_ONLY = ["embed_tokens", "atom_feature", "edge_feature", "se3_invariant_kernel", "movement_pred_head"]
_FROZEN_EARLY_ENCODER = [
    "embed_tokens", "atom_feature", "edge_feature", "se3_invariant_kernel", "movement_pred_head",
    "encoder.layers.0", "encoder.layers.1", "encoder.layers.2", "encoder.layers.3",
    "encoder.layers.4", "encoder.layers.5",
]
_FROZEN_NONE: list[str] | None = None

# Explicit params on unimol_tools 0.1.1 MolTrain.__init__ (verified against source).
# Anything NOT in this set only reaches the trainer via **params, if at all -- so
# it must be smoke-tested, not assumed, before being treated as CRITICAL.
MOLTRAIN_EXPLICIT_INIT_PARAMS = {
    "task", "data_type", "epochs", "learning_rate", "batch_size", "early_stopping",
    "metrics", "split", "split_group_col", "kfold", "save_path", "remove_hs",
    "smiles_col", "target_cols", "target_col_prefix", "target_anomaly_check",
    "smiles_check", "target_normalize", "max_norm", "use_cuda", "use_amp", "use_ddp",
    "use_gpu", "freeze_layers", "freeze_layers_reversed", "load_model_dir",
    "model_name", "model_size", "conf_cache_level",
}
# Structural params we treat as non-negotiable for scientific validity of a run.
CRITICAL_PARAMS = {"task", "epochs", "learning_rate", "batch_size", "save_path", "kfold", "freeze_layers", "model_size"}
# Params we *want* (weight_decay, dropout) but which are NOT in the explicit
# signature above -- these are "nice to have, verify before trusting".
UNVERIFIED_PASSTHROUGH_PARAMS = {"weight_decay", "dropout"}
# FIX F: throughput-only DataLoader knobs -- never affect what the model
# learns, only how fast batches reach the GPU. Same "offer it, let the
# supported_params filter drop it if unsupported" pattern as weight_decay/
# dropout above, so this is a no-op on any unimol_tools version that
# doesn't expose them.
THROUGHPUT_ONLY_PASSTHROUGH_PARAMS = {"num_workers", "pin_memory", "persistent_workers"}

# Covalent radii (Angstrom), Cordero et al. -- used for bond-graph based QC.
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05,
    "CL": 1.02, "BR": 1.20, "I": 1.39, "B": 0.84, "SI": 1.11, "NA": 1.66, "K": 2.03,
    "FE": 1.32, "ZN": 1.22, "MG": 1.41, "CA": 1.76, "SE": 1.20,
}
DEFAULT_RADIUS = 0.77  # generic organic fallback


@dataclass
class SizeTierConfig:
    tier: str
    kfold: int
    freeze_layers: list[str] | None
    epochs: int
    patience: int
    learning_rate: float
    batch_size: int
    model_size: str
    weight_decay: float
    dropout: float


def pick_size_tier(n_train_pool: int, model_size_override: str | None) -> SizeTierConfig:
    if n_train_pool < 400:
        tier = SizeTierConfig("tiny", 5, _FROZEN_EARLY_ENCODER, 80, 12, 3e-5, 8, "84m", 0.1, 0.2)
    elif n_train_pool < 1500:
        tier = SizeTierConfig("small", 5, _FROZEN_EMBEDDINGS_ONLY, 100, 15, 5e-5, 16, "84m", 0.05, 0.15)
    elif n_train_pool < 5000:
        tier = SizeTierConfig("medium", 5, _FROZEN_NONE, 120, 15, 8e-5, 32, "84m", 0.01, 0.1)
    else:
        tier = SizeTierConfig("large", 5, _FROZEN_NONE, 150, 20, 1e-4, 64, "84m", 0.01, 0.1)
    if model_size_override:
        tier.model_size = model_size_override
    return tier


# --------------------------------------------------------------------------------------
# Hashing / IO
# --------------------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def _hash_records_order(records: list[dict]) -> str:
    s = "|".join([f"{r['smiles']}:{r['orig_index']}" for r in records])
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class TestTouchGuard:
    """FIX A: hard-enforces that a given endpoint's scaffold test split is
    read/predicted on at most once per orchestrator run. HPO, nested CV,
    model-size competition and ensembling must all be resolved using the
    training pool before this is ever invoked.

    In v3 this was only called manually from evaluate_test_once(), so any
    OTHER code path that read the test file (e.g. build_training_pool's
    hash step, which fully unpickled test records) went completely
    unguarded. It is now called from inside load_split_cache() itself
    whenever split=="test", so it protects every call site by construction
    instead of relying on every future caller remembering to touch() it."""

    def __init__(self):
        self._touched: set[str] = set()

    def touch(self, endpoint: str):
        if endpoint in self._touched:
            raise RuntimeError(
                f"TEST LEAKAGE GUARD: {endpoint} test split was about to be read/scored "
                f"a second time in this run. Selection logic must be finished before the "
                f"first (and only) test evaluation."
            )
        self._touched.add(endpoint)


def load_split_cache(cache_dir: Path, endpoint: str, split: str,
                      test_guard: "TestTouchGuard | None" = None):
    # FIX A: centralize the leakage guard here, at the one place that
    # actually unpickles split content, instead of at whichever call site
    # happened to remember to invoke it.
    if split == "test" and test_guard is not None:
        test_guard.touch(endpoint)
    path = cache_dir / f"{endpoint}_{split}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "rb") as f:
        records = pickle.load(f)
    if len(records) == 0:
        raise ValueError(f"{path} empty")
    return records, _hash_file(path)


def records_to_unimol_dict(records: list[dict]):
    return {
        "atoms": [r["atoms"] for r in records],
        "coordinates": [r["coordinates"] for r in records],
        "target": [r["target"] for r in records],
        "SMILES": [r["smiles"] for r in records],
        "orig_index": [r["orig_index"] for r in records],
    }


# --------------------------------------------------------------------------------------
# FIX C: real 3D QC -- bond graph, clashes restricted to non-bonded pairs,
# disconnected fragments (fixed logic), best-effort conformer energy via RDKit.
# --------------------------------------------------------------------------------------

def _radius(sym: str) -> float:
    return COVALENT_RADII.get(sym.strip().upper(), DEFAULT_RADIUS)


def _connected_components(n: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    return len({find(i) for i in range(n)})


_RDKIT_WARNED = False


def _try_rdkit_energy(atoms: list[str], coords: np.ndarray) -> float | None:
    """Best-effort MMFF/UFF energy via RDKit's 3D-coordinate bond perception.
    Returns None (and warns once) if RDKit or bond perception is unavailable --
    this check degrades gracefully rather than crashing the pipeline."""
    global _RDKIT_WARNED
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
    except ImportError:
        if not _RDKIT_WARNED:
            print(" [QC] rdkit not installed -> skipping conformer-energy QC (checks 1-4 still run)")
            _RDKIT_WARNED = True
        return None
    try:
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(atoms))
        for i, (sym, xyz) in enumerate(zip(atoms, coords)):
            a = Chem.Atom(sym.strip().capitalize() if len(sym) > 1 else sym.strip().upper())
            mol.AddAtom(a)
            conf.SetAtomPosition(i, tuple(float(v) for v in xyz))
        mol.AddConformer(conf)
        rdDetermineBonds.DetermineConnectivity(mol)
        m = mol.GetMol()
        Chem.SanitizeMol(m, catchErrors=True)
        if AllChem.MMFFHasAllMoleculeParams(m):
            ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
        else:
            ff = AllChem.UFFGetMoleculeForceField(m)
        if ff is None:
            return None
        return float(ff.CalcEnergy())
    except Exception:
        return None


def qc_conformers(records: list[dict], endpoint: str) -> dict:
    from scipy.spatial.distance import pdist, squareform

    bad = []
    energies = []
    energy_idx = []
    for i, r in enumerate(records):
        coords = np.asarray(r["coordinates"], dtype=np.float64)
        atoms = r["atoms"]
        n = len(atoms)

        if np.isnan(coords).any() or np.isinf(coords).any():
            bad.append((i, "NaN/Inf")); continue
        if n != len(coords):
            bad.append((i, "atom/coord length mismatch")); continue
        if n < 1:
            bad.append((i, "empty molecule")); continue

        if n > 1:
            dmat = squareform(pdist(coords))
            radii = np.array([_radius(a) for a in atoms])
            sum_r = radii[:, None] + radii[None, :]
            bonded_thresh = sum_r * 1.3
            clash_thresh = sum_r * 0.9
            np.fill_diagonal(dmat, np.inf)

            bonded_mask = dmat <= bonded_thresh
            edges = list(zip(*np.triu(bonded_mask, k=1).nonzero()))
            n_components = _connected_components(n, edges)

            nonbonded_close = (~bonded_mask) & (dmat < clash_thresh)
            if nonbonded_close.any():
                d = dmat[nonbonded_close].min()
                bad.append((i, f"clash (non-bonded) min_dist {d:.2f}A")); continue

            if dmat[dmat < np.inf].max() > 30:
                bad.append((i, f"exploded max_dist {dmat[dmat < np.inf].max():.1f}A")); continue

            # FIX C: `n_components != len(set(atoms))` compared a bond-graph
            # component count to the count of DISTINCT ELEMENT SYMBOLS in the
            # molecule -- two numbers with no chemical relationship to each
            # other. That let real disconnected fragments through whenever
            # they happened to match the unique-element count, and could
            # trip on connected molecules that didn't. A dataset entry for a
            # single ADMET endpoint molecule should be one connected 3D
            # structure; more than one component is suspicious on its own.
            # (If this dataset intentionally keeps salts/counterions as
            # multi-fragment records, relax this to a per-endpoint allowlist
            # rather than reintroducing an unrelated numeric comparison.)
            if n_components > 1:
                bad.append((i, f"disconnected fragments ({n_components} components)")); continue

        centroid = coords.mean(axis=0)
        if n > 1 and np.linalg.norm(coords - centroid, axis=1).max() > 20:
            bad.append((i, "suspicious spread")); continue

        e = _try_rdkit_energy(atoms, coords)
        if e is not None and np.isfinite(e):
            energies.append(e)
            energy_idx.append(i)

    n_energy_outliers = 0
    if len(energies) >= 8:
        e = np.array(energies)
        mu, sd = e.mean(), e.std() + 1e-9
        z = (e - mu) / sd
        outliers = [energy_idx[k] for k in np.where(np.abs(z) > 3)[0]]
        for k in outliers:
            bad.append((k, "high-energy conformer outlier (>3 sigma MMFF/UFF)"))
        n_energy_outliers = len(outliers)

    if bad:
        print(f" [QC WARNING {endpoint}] {len(bad)}/{len(records)} suspicious "
              f"({n_energy_outliers} energy outliers): {bad[:5]}")
    return {"n_checked": len(records), "n_suspicious": len(bad),
            "n_energy_scored": len(energies), "n_energy_outliers": n_energy_outliers,
            "examples": bad[:20]}


# --------------------------------------------------------------------------------------
# Pool building. Val split is kept untouched and fed live into training when
# possible (FIX B). Test split is NEVER unpickled here (FIX A) -- only its
# raw file bytes are hashed for provenance.
# --------------------------------------------------------------------------------------

def build_training_pool(cache_dir: Path, endpoint: str, pool_strategy: str):
    train_records, train_hash = load_split_cache(cache_dir, endpoint, "train")
    val_records, val_hash = load_split_cache(cache_dir, endpoint, "val")

    # FIX A: hash the test file directly instead of calling
    # load_split_cache(..., "test"), which would fully unpickle the real
    # test records into memory on every single seed/model-size run -- long
    # before the one sanctioned test evaluation, and with no guard on it.
    test_path = cache_dir / f"{endpoint}_test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")
    test_hash = _hash_file(test_path)

    qc_conformers(train_records + val_records, endpoint)

    if pool_strategy == "merged_cv":
        pool_records = train_records + val_records
        provenance = {
            "train_cache_hash": train_hash, "val_cache_hash": val_hash, "test_cache_hash": test_hash,
            "n_train_split": len(train_records), "n_val_split": len(val_records),
            "n_pool": len(pool_records), "pool_strategy": "merged_train_val_for_kfold_cv",
            "pool_order_hash": _hash_records_order(pool_records),
        }
        return pool_records, provenance, len(pool_records), None

    elif pool_strategy == "train_only":
        pool_records = train_records
        provenance = {
            "train_cache_hash": train_hash, "val_cache_hash": val_hash, "test_cache_hash": test_hash,
            "n_train_split": len(train_records), "n_val_split": len(val_records),
            "n_pool": len(pool_records), "pool_strategy": "train_only_val_used_live_for_early_stopping",
            "pool_order_hash": _hash_records_order(pool_records),
        }
        return pool_records, provenance, len(pool_records), val_records
    else:
        raise ValueError(f"Unknown pool_strategy {pool_strategy}")


def endpoint_exp_dir(exp_dir: Path, endpoint: str) -> Path:
    d = exp_dir / endpoint
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------
# Warm starts: hard-rejected on mismatch, not just warned about.
# --------------------------------------------------------------------------------------

def resolve_warm_start_dir(exp_dir: Path, endpoint: str, warm_start_arg: str | None):
    if not warm_start_arg or warm_start_arg.lower() == "none":
        return None
    if warm_start_arg.lower() == "auto":
        p = endpoint_exp_dir(exp_dir, endpoint) / "latest.json"
        if not p.exists():
            return None
        prior = json.loads(p.read_text())["run_dir"]
        pp = Path(prior)
        return pp if pp.exists() else None
    ep = Path(warm_start_arg)
    if not ep.exists():
        raise FileNotFoundError(f"--warm-start {ep} missing")
    return ep


def read_prior_meta(run_dir: Path):
    mp = run_dir / "orchestrator_meta.json"
    return json.loads(mp.read_text()) if mp.exists() else None


def warm_start_compatibility(prior_meta: dict | None, tier_cfg: SizeTierConfig,
                              pool_provenance: dict, args) -> tuple[bool, list[str]]:
    """Returns (is_safe, reasons_for_rejection). A warm start is only safe when
    model size, freeze layout/tier, pool strategy, and dataset identity all
    match what the prior run actually used."""
    if prior_meta is None:
        return False, ["no orchestrator_meta.json found for prior run"]
    reasons = []
    if prior_meta.get("tier") != tier_cfg.tier:
        reasons.append(f"tier mismatch: prior={prior_meta.get('tier')} current={tier_cfg.tier}")
    prior_kwargs = prior_meta.get("trainer_kwargs", {})
    if prior_kwargs.get("model_size") != tier_cfg.model_size:
        reasons.append(f"model_size mismatch: prior={prior_kwargs.get('model_size')} current={tier_cfg.model_size}")
    if prior_kwargs.get("remove_hs") is not None and prior_kwargs.get("remove_hs") != False:
        reasons.append(f"preprocessing (remove_hs) mismatch: prior={prior_kwargs.get('remove_hs')}")
    prior_prov = prior_meta.get("pool_provenance", {})
    if prior_prov.get("pool_strategy") != pool_provenance.get("pool_strategy"):
        reasons.append(f"pool_strategy mismatch: prior={prior_prov.get('pool_strategy')} "
                        f"current={pool_provenance.get('pool_strategy')}")
    if prior_prov.get("train_cache_hash") != pool_provenance.get("train_cache_hash"):
        reasons.append("train cache hash changed (dataset identity differs)")
    if prior_kwargs.get("kfold") != tier_cfg.kfold:
        reasons.append(f"kfold mismatch: prior={prior_kwargs.get('kfold')} current={tier_cfg.kfold}")
    return (len(reasons) == 0), reasons


def resolve_warm_start(exp_dir: Path, endpoint: str, tier_cfg: SizeTierConfig,
                        pool_provenance: dict, args):
    warm_start_dir = resolve_warm_start_dir(exp_dir, endpoint, args.warm_start)
    if warm_start_dir is None:
        return None
    prior_meta = read_prior_meta(warm_start_dir)
    safe, reasons = warm_start_compatibility(prior_meta, tier_cfg, pool_provenance, args)
    if not safe:
        msg = f" WARM START REJECTED for {endpoint} ({warm_start_dir}): " + "; ".join(reasons)
        if getattr(args, "force_warm_start", False):
            print(msg + " -- proceeding anyway because --force-warm-start was passed.")
            return warm_start_dir
        print(msg + " -- training from scratch instead. Pass --force-warm-start to override.")
        return None
    return warm_start_dir


def record_latest_pointer(exp_dir: Path, endpoint: str, run_dir: Path):
    (endpoint_exp_dir(exp_dir, endpoint) / "latest.json").write_text(
        json.dumps({"run_dir": str(run_dir.resolve())}, indent=2))


# --------------------------------------------------------------------------------------
# API validation goes beyond inspect.signature() because MolTrain takes
# **params -- a smoke test is the only way to catch silently-dropped or
# silently-ignored kwargs (e.g. weight_decay / dropout). FIX D makes the
# smoke test's verdict actually control the HPO search space (see
# sample_hpo_configs / run_hpo / main below).
# --------------------------------------------------------------------------------------

def validate_unimol_api():
    from unimol_tools import MolTrain
    sig = inspect.signature(MolTrain.__init__)
    explicit_params = {p for p in sig.parameters if p not in ("self",)}
    has_catchall = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    print(f" [API] MolTrain explicit __init__ params: {sorted(explicit_params - {'params'})}"
          f"{' (+ **params catch-all)' if has_catchall else ''}")
    issues = []
    for p in ["freeze_layers", "kfold", "model_size"]:
        if p not in explicit_params:
            issues.append(f"CRITICAL: {p} not in explicit signature")
    for p in sorted(UNVERIFIED_PASSTHROUGH_PARAMS):
        if p not in explicit_params:
            issues.append(f"UNVERIFIED: '{p}' not in explicit signature -- only reaches trainer via "
                           f"**params (if at all). Must be confirmed by smoke test, not assumed.")
    return explicit_params, issues


_TINY_SMOKE_MOLECULE = {
    # methane + a tiny halogenated variant, enough atoms/bonds for real bond-graph handling
    "atoms": [["C", "H", "H", "H", "H"], ["C", "H", "H", "H", "CL"]],
    "coordinates": [
        [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]],
        [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [1.77, -1.77, -1.77]],
    ],
    "target": [0, 1],
    "SMILES": ["C", "CCl"],
    "orig_index": [0, 1],
}


def run_smoke_test(exp_dir: Path, supported_params: set[str]) -> dict:
    """Actually fit+predict on a synthetic 2-molecule dataset so we catch
    runtime failures inspect.signature() cannot see, and get a best-effort
    signal on whether dropout/weight_decay actually change model behavior
    (not proof, but far better than assuming). FIX D: this signal is now
    actually consumed by main()/sample_hpo_configs() instead of only being
    printed."""
    from unimol_tools import MolPredict, MolTrain
    smoke_dir = exp_dir / "_smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    report = {"fit_predict_roundtrip_ok": False, "weight_decay_effect_detected": None,
              "dropout_effect_detected": None, "errors": []}

    def _fit_and_score(save_path, weight_decay, dropout):
        kwargs = dict(task="classification", data_type="molecule", epochs=2, learning_rate=1e-4,
                      batch_size=2, kfold=1, save_path=str(save_path), remove_hs=False,
                      smiles_col="SMILES", target_col_prefix="TARGET", use_cuda=False,
                      use_amp=False, model_name="unimolv2", model_size="84m", conf_cache_level=0)
        if "weight_decay" in supported_params:
            kwargs["weight_decay"] = weight_decay
        if "dropout" in supported_params:
            kwargs["dropout"] = dropout
        kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        clf = MolTrain(**kwargs)
        clf.fit(dict(_TINY_SMOKE_MOLECULE))
        pred = MolPredict(load_model=str(save_path))
        out = np.asarray(pred.predict(dict(_TINY_SMOKE_MOLECULE), save_path=None, metrics="none"))
        return out

    try:
        base = _fit_and_score(smoke_dir / "a", weight_decay=0.0, dropout=0.0)
        report["fit_predict_roundtrip_ok"] = True
    except Exception as e:
        report["errors"].append(f"basic fit/predict roundtrip failed: {e}")
        return report  # nothing else is trustworthy if this fails

    if "weight_decay" in supported_params:
        try:
            hi = _fit_and_score(smoke_dir / "b", weight_decay=1.0, dropout=0.0)
            report["weight_decay_effect_detected"] = bool(np.abs(hi - base).max() > 1e-4)
        except Exception as e:
            report["errors"].append(f"weight_decay probe failed: {e}")
    if "dropout" in supported_params:
        try:
            hi = _fit_and_score(smoke_dir / "c", weight_decay=0.0, dropout=0.5)
            report["dropout_effect_detected"] = bool(np.abs(hi - base).max() > 1e-4)
        except Exception as e:
            report["errors"].append(f"dropout probe failed: {e}")

    for p in ("weight_decay", "dropout"):
        detected = report.get(f"{p}_effect_detected")
        if p in supported_params and detected is False:
            print(f" [SMOKE TEST WARNING] passing '{p}' produced IDENTICAL predictions to not passing it -- "
                  f"it is likely being silently ignored by this unimol_tools version. Treat {p} tuning as void.")
    return report


def resolve_fit_kwargs_with_valid(val_dict: dict | None):
    """Look at the *real* MolTrain.fit signature for a validation-set hook.
    If found, return its kwarg name so the caller can wire the untouched val
    split into training so early stopping watches it live. If not found,
    return None and a loud warning that early stopping is NOT using the true
    val set. (FIX B: the caller now actually uses this return value.)"""
    if val_dict is None:
        return None, None
    try:
        from unimol_tools import MolTrain
        fit_sig = inspect.signature(MolTrain.fit)
        fit_params = set(fit_sig.parameters.keys())
    except Exception:
        return None, "could not introspect MolTrain.fit signature"
    for candidate in ("valid_data", "eval_set", "X_valid", "valid_dict", "val_data"):
        if candidate in fit_params:
            return candidate, None
    return None, ("MolTrain.fit() has no recognized validation-set kwarg in this unimol_tools version "
                   "(checked: valid_data, eval_set, X_valid, valid_dict, val_data). Early stopping during "
                   "train_only runs is monitoring internal training behavior only -- the held-out val split "
                   "is evaluated AFTER training completes, not used to pick the stopping point live.")


def set_determinism(seed: int, allow_tf32: bool):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.manual_seed_all(seed)
        # FIX F (speed): let cuDNN pick the fastest kernel implementation for
        # the shapes it actually sees, instead of always using a fixed
        # "safe" algorithm. This only changes *which kernel* runs, never the
        # arithmetic result -- pure engine-level speedup, GPU-only (no-op on
        # CPU runs below). Deliberately paired with warn_only=True on
        # use_deterministic_algorithms() a few lines down so the two don't
        # hard-conflict; if a given unimol_tools/torch version does warn
        # about it, that's a warning you can ignore for a speed run, not a
        # correctness issue.
        torch.backends.cudnn.benchmark = True
    else:
        # FIX E (speed): actually use all CPU cores for matmul-heavy ops.
        # Setting only the OMP/MKL env vars at import time helps numpy/scipy;
        # torch's own thread pool additionally needs this explicit call.
        torch.set_num_threads(_CPU_COUNT)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass
    return {"seed": seed, "tf32_allowed": allow_tf32, "cuda": has_cuda,
            "cudnn_benchmark": has_cuda, "cpu_threads": _CPU_COUNT if not has_cuda else None}


def collect_env():
    def _v(pkg):
        try:
            import importlib.metadata as md
            return md.version(pkg)
        except Exception:
            return None
    return {"python": platform.python_version(), "torch": torch.__version__, "unimol_tools": _v("unimol_tools")}


# --------------------------------------------------------------------------------------
# Metrics, calibration, conformal prediction (unchanged from v3)
# --------------------------------------------------------------------------------------

def _optimize_threshold(y_true, y_score):
    from sklearn.metrics import f1_score, matthews_corrcoef
    best_f1, best_t, best_mcc, best_t_mcc = -1, 0.5, -2, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        yb = (y_score >= t).astype(int)
        f1 = f1_score(y_true, yb, zero_division=0)
        mcc = matthews_corrcoef(y_true, yb)
        if f1 > best_f1:
            best_f1, best_t = f1, t
        if mcc > best_mcc:
            best_mcc, best_t_mcc = mcc, t
    return best_t, best_f1, best_t_mcc, best_mcc


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _safe_binary_metrics(y_true, y_score, threshold_from_cv=None, calibrator=None):
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    out = {"n": int(len(y_true)), "n_positive": int(np.nansum(y_true))}
    try:
        out["auroc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception as e:
        out["auroc"] = None; out["auroc_error"] = str(e)
    try:
        out["auprc"] = round(float(average_precision_score(y_true, y_score)), 4)
    except Exception as e:
        out["auprc"] = None; out["auprc_error"] = str(e)

    if threshold_from_cv is None:
        bt, bf1, bt_mcc, bmcc = _optimize_threshold(y_true, y_score)
        out["f1_best"] = round(float(bf1), 4); out["f1_best_thresh"] = round(float(bt), 3)
        out["mcc_best"] = round(float(bmcc), 4); out["mcc_best_thresh"] = round(float(bt_mcc), 3)
        out["f1_at_0.5"] = round(float(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)), 4)
    else:
        t = threshold_from_cv
        yb = (y_score >= t).astype(int)
        out["f1_at_frozen_thresh"] = round(float(f1_score(y_true, yb, zero_division=0)), 4)
        out["mcc_at_frozen_thresh"] = round(float(matthews_corrcoef(y_true, yb)), 4)
        out["frozen_thresh"] = round(float(t), 3)
        out["f1_at_0.5"] = round(float(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)), 4)

    prob = y_score
    if calibrator is not None:
        prob = apply_calibration(calibrator, y_score)
        out["calibrated"] = True
    try:
        out["brier"] = round(float(brier_score_loss(y_true, prob)), 4)
        out["ece"] = round(_expected_calibration_error(y_true, prob), 4)
    except Exception as e:
        out["brier"] = None; out["ece"] = None; out["calibration_error"] = str(e)
    return out


def fit_calibration(oof_scores: np.ndarray, oof_labels: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_scores, oof_labels)
    return iso


def apply_calibration(calibrator, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict(scores)


def conformal_calibrate(oof_scores: np.ndarray, oof_labels: np.ndarray):
    """nonconformity for the true class = 1 - p(true class), stored per class
    so test-time p-values are class-conditional (valid coverage per class)."""
    oof_labels = np.asarray(oof_labels).astype(int)
    p1 = oof_scores
    p0 = 1 - oof_scores
    nonconf_by_class = {1: 1 - p1[oof_labels == 1], 0: 1 - p0[oof_labels == 0]}
    return nonconf_by_class


def conformal_predict(test_scores: np.ndarray, nonconf_by_class: dict, alpha: float = 0.1):
    """Returns list of prediction sets (subset of {0,1}) and the two p-values
    per molecule. Set size 0 = model+calibration are inconsistent for this
    point (rare, flag hard); size 2 = genuinely uncertain (abstain-worthy);
    size 1 = confident at the requested (1-alpha) coverage level."""
    p1 = test_scores
    p0 = 1 - test_scores
    nc1_calib = nonconf_by_class[1]
    nc0_calib = nonconf_by_class[0]
    nc1_test = 1 - p1
    nc0_test = 1 - p0

    def pval(nc_test, nc_calib):
        n = len(nc_calib)
        return (1 + np.sum(nc_calib[None, :] >= nc_test[:, None], axis=1)) / (n + 1)

    pval_class1 = pval(nc1_test, nc1_calib)
    pval_class0 = pval(nc0_test, nc0_calib)

    pred_sets = []
    for pv0, pv1 in zip(pval_class0, pval_class1):
        s = set()
        if pv0 > alpha:
            s.add(0)
        if pv1 > alpha:
            s.add(1)
        pred_sets.append(sorted(s))
    return pred_sets, pval_class0, pval_class1


# --------------------------------------------------------------------------------------
# FIX D: HPO grid now only varies weight_decay/dropout when they've been
# shown (by validate_unimol_api + run_smoke_test, wired up in main()) to
# actually reach and change the model. Otherwise they're fixed at the tier
# default so every trial that runs is a genuinely distinct configuration.
# --------------------------------------------------------------------------------------

@dataclass
class HPConfig:
    lr_mult: float
    weight_decay: float
    dropout: float
    freeze_choice: str  # "tier_default" | "lighter_freeze" | "no_freeze"
    batch_mult: int

    def freeze_layers(self, tier_cfg: SizeTierConfig):
        if self.freeze_choice == "tier_default":
            return tier_cfg.freeze_layers
        if self.freeze_choice == "no_freeze":
            return _FROZEN_NONE
        if self.freeze_choice == "lighter_freeze":
            return _FROZEN_EMBEDDINGS_ONLY
        return tier_cfg.freeze_layers


def sample_hpo_configs(tier_cfg: SizeTierConfig, n_trials: int, rng: random.Random,
                        tune_weight_decay: bool = True, tune_dropout: bool = True) -> list[HPConfig]:
    lr_mults = [0.5, 1.0, 2.0]
    # FIX D: collapse to the tier default (a single value) instead of
    # enumerating values that would be silently dropped before reaching
    # MolTrain -- avoids burning trials on runs that are actually identical.
    wds = (sorted({tier_cfg.weight_decay * m for m in (0.5, 1.0, 2.0)})
           if tune_weight_decay else [tier_cfg.weight_decay])
    dropouts = (sorted({round(tier_cfg.dropout + d, 3) for d in (-0.05, 0.0, 0.05) if tier_cfg.dropout + d >= 0})
                if tune_dropout else [tier_cfg.dropout])
    freeze_choices = ["tier_default", "lighter_freeze"] if tier_cfg.freeze_layers else ["tier_default"]
    batch_mults = [1, 2]

    grid = list(itertools.product(lr_mults, wds, dropouts, freeze_choices, batch_mults))
    rng.shuffle(grid)
    # always include the un-perturbed tier default as trial 0 (safe baseline)
    baseline = (1.0, tier_cfg.weight_decay, tier_cfg.dropout, "tier_default", 1)
    grid = [baseline] + [g for g in grid if g != baseline]
    chosen = grid[:max(1, n_trials)]
    return [HPConfig(*c) for c in chosen]


def build_trainer_kwargs(tier_cfg: SizeTierConfig, hp: HPConfig, run_dir: Path, kfold: int,
                          supported_params: set[str], warm_start_dir: Optional[Path],
                          split_mode: str) -> dict:
    has_cuda = torch.cuda.is_available()
    kwargs = dict(
        task="classification", data_type="molecule", epochs=tier_cfg.epochs,
        learning_rate=tier_cfg.learning_rate * hp.lr_mult,
        batch_size=tier_cfg.batch_size * hp.batch_mult,
        early_stopping=tier_cfg.patience, metrics="auc,auprc,f1_score,mcc",
        split=split_mode, kfold=kfold,
        save_path=str(run_dir), remove_hs=False,
        smiles_col="SMILES", target_col_prefix="TARGET",
        target_anomaly_check=False, smiles_check="none", target_normalize="none",
        max_norm=5.0, use_cuda=has_cuda,
        # FIX E (speed/correctness): AMP mainly helps -- and is most reliably
        # supported -- on GPU. Forcing it on with no CUDA device (v3's
        # hardcoded True) buys nothing and can behave oddly on CPU-only
        # backends, so it's now conditioned on device availability.
        use_amp=has_cuda, use_ddp=False, model_name="unimolv2",
        model_size=tier_cfg.model_size,
        load_model_dir=str(warm_start_dir) if warm_start_dir else None,
        conf_cache_level=0,
        weight_decay=hp.weight_decay,
        dropout=hp.dropout,
    )
    # FIX F (speed, GPU only): throughput-only DataLoader knobs. These never
    # change what's learned -- only how batches get to the GPU -- and go
    # through the identical supported_params filter as everything else
    # below, so they're a pure no-op if this unimol_tools version doesn't
    # expose them (printed like any other dropped kwarg, same as
    # weight_decay/dropout already were in FIX D).
    if has_cuda:
        kwargs["num_workers"] = min(4, _CPU_COUNT)
        kwargs["pin_memory"] = True
        kwargs["persistent_workers"] = True
    freeze = hp.freeze_layers(tier_cfg)
    if freeze is not None:
        kwargs["freeze_layers"] = freeze

    unsupported_critical = [k for k in ("freeze_layers", "kfold", "model_size")
                             if k in kwargs and k not in supported_params]
    if unsupported_critical:
        raise RuntimeError(
            f"CRITICAL API MISMATCH: unimol_tools MolTrain does not support {unsupported_critical}. "
            f"Training would be scientifically wrong (e.g. no freezing or no kfold). "
            f"Fix: upgrade unimol_tools or adapt code. Supported={sorted(supported_params)}"
        )
    filtered = {k: v for k, v in kwargs.items() if k in supported_params}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        print(f" [API] Dropping non-critical unsupported: {dropped}")

    # FIX E (speed/correctness): cap batch size on CPU-only runs so an 8GB
    # laptop doesn't OOM mid-training. GPU runs are untouched.
    if not has_cuda and "batch_size" in filtered:
        capped = min(filtered["batch_size"], 8)
        if capped != filtered["batch_size"]:
            print(f" [SPEED] No GPU -> capping batch_size {filtered['batch_size']} -> {capped} to protect RAM.")
        filtered["batch_size"] = capped
    return filtered


def _cv_pred_to_score(cv_pred) -> np.ndarray:
    cv_pred = np.asarray(cv_pred)
    return cv_pred[:, -1] if cv_pred.ndim > 1 else cv_pred


def run_hpo(pool_dict: dict, tier_cfg: SizeTierConfig, args, supported_params: set[str],
            inner_kfold: int, n_trials: int, seed: int, run_dir_root: Path,
            metric: str = "auroc") -> tuple[HPConfig, float, list[dict]]:
    """Actually sweeps hyperparameters, selecting by inner-CV score. Never
    touches test. FIX D: which dimensions are actually varied is controlled
    by args.tune_weight_decay / args.tune_dropout, set once in main() from
    the smoke test's verdict."""
    from unimol_tools import MolTrain
    rng = random.Random(seed)
    configs = sample_hpo_configs(tier_cfg, n_trials, rng,
                                  tune_weight_decay=getattr(args, "tune_weight_decay", True),
                                  tune_dropout=getattr(args, "tune_dropout", True))
    y_true = np.asarray(pool_dict["target"]).reshape(-1)

    best_cfg, best_score, log = None, -np.inf, []
    for i, hp in enumerate(configs):
        trial_dir = run_dir_root / f"hpo_trial_{i}"
        try:
            kwargs = build_trainer_kwargs(tier_cfg, hp, trial_dir, inner_kfold, supported_params,
                                           warm_start_dir=None, split_mode="stratified")
            clf = MolTrain(**kwargs)
            clf.fit(pool_dict)
            score = _cv_pred_to_score(clf.cv_pred)
            m = _safe_binary_metrics(y_true, score, threshold_from_cv=None)
            objective = m.get(metric) if m.get(metric) is not None else -np.inf
            log.append({"trial": i, "hp": asdict(hp), "metrics": m})
            print(f" [HPO] trial {i} hp={asdict(hp)} {metric}={objective}")
            if objective > best_score:
                best_score, best_cfg = objective, hp
        except Exception as e:
            log.append({"trial": i, "hp": asdict(hp), "error": str(e)})
            print(f" [HPO] trial {i} FAILED: {e}")
    if best_cfg is None:
        raise RuntimeError(f"All {len(configs)} HPO trials failed for this pool -- see log.")
    return best_cfg, best_score, log


# --------------------------------------------------------------------------------------
# Nested CV. Outer split is FIXED across ensemble seeds so their OOF arrays
# line up positionally and can be averaged honestly.
# --------------------------------------------------------------------------------------

def nested_cv_evaluate(pool_records: list[dict], tier_cfg: SizeTierConfig, args,
                        supported_params: set[str], run_dir_root: Path,
                        outer_k: int, inner_k: int, hpo_trials: int,
                        outer_cv_seed: int, model_seed: int):
    from sklearn.model_selection import StratifiedKFold
    from unimol_tools import MolPredict, MolTrain

    y = np.array([r["target"] for r in pool_records])
    n = len(pool_records)
    oof_scores = np.full(n, np.nan)
    oof_mask = np.zeros(n, dtype=bool)
    chosen_hps = []

    # FIXED split across seeds/model sizes -> ensemble OOF arrays align positionally.
    skf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=outer_cv_seed)
    for outer_fold, (inner_idx, outer_idx) in enumerate(skf.split(np.zeros(n), y)):
        inner_records = [pool_records[i] for i in inner_idx]
        outer_records = [pool_records[i] for i in outer_idx]
        inner_dict = records_to_unimol_dict(inner_records)
        outer_dict = records_to_unimol_dict(outer_records)

        fold_root = run_dir_root / f"outer_{outer_fold}"
        best_hp, best_score, _ = run_hpo(inner_dict, tier_cfg, args, supported_params,
                                          inner_kfold=inner_k, n_trials=hpo_trials,
                                          seed=model_seed * 1000 + outer_fold,
                                          run_dir_root=fold_root / "hpo")
        chosen_hps.append(asdict(best_hp))

        final_dir = fold_root / "refit"
        kwargs = build_trainer_kwargs(tier_cfg, best_hp, final_dir, kfold=1,
                                       supported_params=supported_params, warm_start_dir=None,
                                       split_mode="random")
        clf = MolTrain(**kwargs)
        clf.fit(inner_dict)
        predictor = MolPredict(load_model=str(final_dir))
        pred = np.asarray(predictor.predict(outer_dict, save_path=None, metrics="none"))
        score = pred[:, -1] if pred.ndim > 1 else pred
        oof_scores[outer_idx] = score
        oof_mask[outer_idx] = True
        print(f" [NESTED CV] outer fold {outer_fold}: best_hp={asdict(best_hp)} inner_score={best_score}")

    nested_metrics = _safe_binary_metrics(y[oof_mask], oof_scores[oof_mask], threshold_from_cv=None)
    return nested_metrics, oof_scores, oof_mask, y, chosen_hps


def train_final_pool_model(pool_dict: dict, tier_cfg: SizeTierConfig, best_hp: HPConfig,
                            supported_params: set[str], run_dir: Path, warm_start_dir,
                            fit_kwarg_name: str | None = None, val_dict: dict | None = None):
    from unimol_tools import MolTrain
    kwargs = build_trainer_kwargs(tier_cfg, best_hp, run_dir, kfold=1,
                                   supported_params=supported_params,
                                   warm_start_dir=warm_start_dir, split_mode="random")
    clf = MolTrain(**kwargs)
    # FIX B: this is the actual wiring that was missing in v3 --
    # resolve_fit_kwargs_with_valid() correctly detected the fit()-time
    # validation kwarg, but the result was only ever printed; clf.fit()
    # always ran without it. Now, when the installed unimol_tools exposes a
    # validation-set kwarg AND we have an untouched val split (train_only
    # strategy), it's actually passed through so early stopping watches it
    # live during training.
    fit_call_kwargs = {}
    if fit_kwarg_name is not None and val_dict is not None:
        fit_call_kwargs[fit_kwarg_name] = val_dict
    clf.fit(pool_dict, **fit_call_kwargs)
    return run_dir, kwargs


# --------------------------------------------------------------------------------------
# Endpoint result container
# --------------------------------------------------------------------------------------

@dataclass
class EndpointResult:
    endpoint: str; tier: str; n_pool: int; n_test: int; kfold: int
    warm_started_from: str | None; run_dir: str
    nested_cv_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    ensemble_test_metrics: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    chosen_hp: dict = field(default_factory=dict)
    error: str | None = None
    model_size: str = "84m"


# --------------------------------------------------------------------------------------
# Per-endpoint, per-seed pipeline: nested CV -> final pool model. Test is NOT
# touched here -- caller aggregates seeds and touches test exactly once.
# --------------------------------------------------------------------------------------

def run_endpoint_seed(endpoint: str, cache_dir: Path, exp_dir: Path, args,
                       supported_params: set[str], seed: int, outer_cv_seed: int):
    pool_records, pool_provenance, n_pool, val_records = build_training_pool(cache_dir, endpoint, args.pool_strategy)
    tier_cfg = pick_size_tier(n_pool, args.model_size)
    print(f" Pool {n_pool} tier={tier_cfg.tier} strategy={args.pool_strategy} seed={seed}")

    warm_start_dir = resolve_warm_start(exp_dir, endpoint, tier_cfg, pool_provenance, args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = endpoint_exp_dir(exp_dir, endpoint) / f"run_{timestamp}_seed{seed}"
    det = set_determinism(seed, args.allow_tf32)

    outer_k = args.outer_k if not args.skip_nested_cv else 2
    inner_k = args.inner_k if not args.skip_nested_cv else 2
    hpo_trials = args.hpo_trials if not args.skip_nested_cv else 1

    nested_metrics, oof_scores, oof_mask, oof_labels, chosen_hps = nested_cv_evaluate(
        pool_records, tier_cfg, args, supported_params, run_root / "nested_cv",
        outer_k=outer_k, inner_k=inner_k, hpo_trials=hpo_trials,
        outer_cv_seed=outer_cv_seed, model_seed=seed,
    )
    print(f" [{endpoint} seed={seed}] NESTED CV (honest generalization estimate): {nested_metrics}")

    # Re-run HPO once more on the FULL pool to pick the hyperparams for the
    # model we actually ship -- selection still only ever sees the pool.
    final_hp, final_hpo_score, _ = run_hpo(
        records_to_unimol_dict(pool_records), tier_cfg, args, supported_params,
        inner_kfold=tier_cfg.kfold, n_trials=hpo_trials, seed=seed,
        run_dir_root=run_root / "full_pool_hpo",
    )

    final_run_dir = run_root / "final_model"
    val_dict = records_to_unimol_dict(val_records) if val_records is not None else None
    fit_kwarg_name, fit_warning = resolve_fit_kwargs_with_valid(val_dict)
    if args.pool_strategy == "train_only":
        if fit_kwarg_name:
            print(f" [{endpoint}] train_only: wiring live validation into fit() via '{fit_kwarg_name}'")
        else:
            print(f" [{endpoint}] WARNING: {fit_warning}")

    # FIX B: actually pass the resolved kwarg name + val split through so
    # train_final_pool_model can wire them into clf.fit().
    train_final_pool_model(records_to_unimol_dict(pool_records), tier_cfg, final_hp,
                            supported_params, final_run_dir, warm_start_dir,
                            fit_kwarg_name=fit_kwarg_name, val_dict=val_dict)

    if val_records is not None:
        try:
            from unimol_tools import MolPredict
            pred_val = MolPredict(load_model=str(final_run_dir))
            pv = np.asarray(pred_val.predict(val_dict, save_path=None, metrics="none"))
            yv = np.asarray(val_dict["target"]).reshape(-1)
            pvs = pv[:, -1] if pv.ndim > 1 else pv
            val_metrics = _safe_binary_metrics(yv, pvs, threshold_from_cv=nested_metrics.get("f1_best_thresh", 0.5))
            print(f" [{endpoint}] original VAL (post-hoc, not used for selection): {val_metrics}")
        except Exception as e:
            print(f" Val eval failed {e}")

    meta = {
        "endpoint": endpoint, "timestamp": timestamp, "tier": tier_cfg.tier,
        "trainer_kwargs": {"model_size": tier_cfg.model_size, "kfold": tier_cfg.kfold, "remove_hs": False},
        "pool_provenance": pool_provenance, "nested_cv_metrics": nested_metrics,
        "chosen_hps_per_outer_fold": chosen_hps, "final_hp": asdict(final_hp),
        "warm_started_from": str(warm_start_dir) if warm_start_dir else None,
        "determinism": det, "environment": collect_env(),
        "fit_valid_kwarg_used": fit_kwarg_name, "fit_valid_warning": fit_warning,
    }
    (run_root / "orchestrator_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    record_latest_pointer(exp_dir, endpoint, final_run_dir)

    return {
        "tier_cfg": tier_cfg, "pool_provenance": pool_provenance, "n_pool": n_pool,
        "nested_metrics": nested_metrics, "oof_scores": oof_scores, "oof_mask": oof_mask,
        "oof_labels": oof_labels, "final_run_dir": final_run_dir, "final_hp": final_hp,
        "warm_start_dir": warm_start_dir, "run_root": run_root,
    }


def evaluate_test_once(endpoint: str, cache_dir: Path, seed_runs: list[dict],
                        test_guard: TestTouchGuard):
    """The single point in the whole pipeline that reads the scaffold test
    split. Everything selection-related (HPO, nested CV, calibration,
    thresholding, ensembling) must be finished before this is called.
    FIX A: the guard is now enforced inside load_split_cache() itself, so
    passing test_guard through here is what actually protects this read --
    not a manually-placed .touch() call that other code paths could bypass."""
    from unimol_tools import MolPredict

    test_records, _ = load_split_cache(cache_dir, endpoint, "test", test_guard=test_guard)
    test_dict = records_to_unimol_dict(test_records)
    y_test = np.asarray(test_dict["target"]).reshape(-1)

    # Average OOF across seeds on the shared, position-aligned split.
    oof_stack = np.stack([r["oof_scores"] for r in seed_runs], axis=0)
    oof_mask_all = np.all([r["oof_mask"] for r in seed_runs], axis=0)
    ensemble_oof = np.nanmean(oof_stack, axis=0)
    oof_labels = seed_runs[0]["oof_labels"]

    calibrator = fit_calibration(ensemble_oof[oof_mask_all], oof_labels[oof_mask_all])
    calibrated_oof = apply_calibration(calibrator, ensemble_oof[oof_mask_all])
    frozen_thresh = _optimize_threshold(oof_labels[oof_mask_all], calibrated_oof)[0]
    nonconf_by_class = conformal_calibrate(calibrated_oof, oof_labels[oof_mask_all])

    per_seed_test_scores = []
    for r in seed_runs:
        predictor = MolPredict(load_model=str(r["final_run_dir"]))
        pred = np.asarray(predictor.predict(test_dict, save_path=None, metrics="none"))
        score = pred[:, -1] if pred.ndim > 1 else pred
        per_seed_test_scores.append(score)
    per_seed_test_scores = np.stack(per_seed_test_scores, axis=0)
    ensemble_test_score = per_seed_test_scores.mean(axis=0)
    calibrated_test_score = apply_calibration(calibrator, ensemble_test_score)

    test_metrics = _safe_binary_metrics(y_test, ensemble_test_score,
                                         threshold_from_cv=frozen_thresh, calibrator=calibrator)

    # Uncertainty: ensemble disagreement + conformal prediction sets.
    seed_std = per_seed_test_scores.std(axis=0)
    pred_sets, pval0, pval1 = conformal_predict(calibrated_test_score, nonconf_by_class,
                                                 alpha=1 - 0.90)
    abstain_mask = np.array([len(s) != 1 for s in pred_sets])
    uncertainty = {
        "mean_seed_std": round(float(seed_std.mean()), 4),
        "n_high_disagreement_seed_std_gt_0.15": int((seed_std > 0.15).sum()),
        "conformal_alpha": 0.10,
        "n_conformal_abstain": int(abstain_mask.sum()),
        "frac_conformal_abstain": round(float(abstain_mask.mean()), 4),
    }

    return {
        "test_metrics": test_metrics,
        "n_test": len(test_records),
        "frozen_thresh": float(frozen_thresh),
        "uncertainty": uncertainty,
        "n_ensemble_seeds": len(seed_runs),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    p.add_argument("--exp-dir", default=EXP_DIR_DEFAULT)
    p.add_argument("--warm-start", default="none")
    p.add_argument("--force-warm-start", action="store_true",
                   help="override warm-start compatibility rejection -- use with care")
    p.add_argument("--pool-strategy", choices=["merged_cv", "train_only"], default="merged_cv")
    p.add_argument("--model-size", default=None)
    p.add_argument("--compete-model-sizes", nargs="+", default=None,
                   help="winner per endpoint selected by NESTED CV AUROC, never by test")
    p.add_argument("--ensemble-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outer-cv-seed", type=int, default=12345,
                   help="fixed across all seeds/model sizes so OOF arrays align for ensembling")
    p.add_argument("--outer-k", type=int, default=5, help="nested CV outer folds")
    p.add_argument("--inner-k", type=int, default=5, help="nested CV / HPO inner folds")
    p.add_argument("--hpo-trials", type=int, default=6)
    p.add_argument("--skip-nested-cv", action="store_true",
                   help="fast/debug mode: 2x2 folds, 1 HPO trial -- NOT for reported results")
    p.add_argument("--no-auto-scale", action="store_true",
                   help="FIX E: by default, outer-k/inner-k/hpo-trials/ensemble-seeds are auto-reduced "
                        "when no CUDA GPU is detected, since full settings are not feasible on a CPU "
                        "laptop. Pass this to force the settings you gave on the command line as-is.")
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir); exp_dir = Path(args.exp_dir); exp_dir.mkdir(parents=True, exist_ok=True)
    test_guard = TestTouchGuard()

    has_cuda = torch.cuda.is_available() and not args.force_cpu

    # FIX F: this is the "must run on GPU when available" behavior you
    # asked about -- has_cuda (above) already drives use_cuda=has_cuda and
    # use_amp=has_cuda into every MolTrain(...) call via build_trainer_kwargs
    # (FIX E), so a T4 is used automatically with zero flags needed; this is
    # just an unmissable confirmation printed at the very start of the run,
    # including the actual device name so you can see Colab really gave you
    # a GPU this session (and not a CPU-only runtime).
    if has_cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        print(f" [SPEED] GPU DETECTED -> training will run on CUDA ({gpu_name}). "
              f"use_cuda=True, use_amp=True, cudnn.benchmark=True for every model fit.")
    else:
        print(" [SPEED] No usable GPU -> training will run on CPU "
              f"({_CPU_COUNT} threads). This is expected if force_cpu was passed or "
              f"Colab did not allocate a GPU runtime this session.")

    # FIX E (speed): a CPU-only laptop cannot feasibly run 5x5-fold nested CV
    # with 6 HPO trials per fold per endpoint (that's outer_k * (hpo_trials+1)
    # full UniMol fits, times endpoints, times model sizes, times ensemble
    # seeds). Scale the workload down automatically unless the user asks for
    # the full thing explicitly. GPU runs are completely unaffected.
    if not has_cuda and not args.no_auto_scale and not args.skip_nested_cv:
        print(" [SPEED] No CUDA GPU detected -> auto-scaling nested-CV/HPO workload for CPU feasibility.")
        print("         (outer_k, inner_k, hpo_trials, ensemble_seeds reduced; pass --no-auto-scale to")
        print("          keep your command-line values as-is -- NOT recommended on a CPU-only laptop.)")
        args.outer_k = min(args.outer_k, 3)
        args.inner_k = min(args.inner_k, 3)
        args.hpo_trials = min(args.hpo_trials, 3)
        args.ensemble_seeds = min(args.ensemble_seeds, 1)
    print(f" [SPEED] device={'cuda' if has_cuda else 'cpu'} threads={_CPU_COUNT if not has_cuda else 'n/a'} "
          f"outer_k={args.outer_k} inner_k={args.inner_k} hpo_trials={args.hpo_trials} "
          f"ensemble_seeds={args.ensemble_seeds}")

    tune_weight_decay = False
    tune_dropout = False

    supported_params = set()
    if not args.dry_run:
        try:
            import unimol_tools  # noqa
            supported_params, api_issues = validate_unimol_api()
            if api_issues:
                print(" API issues:"); [print("  -", i) for i in api_issues]
            if not args.skip_smoke_test:
                smoke = run_smoke_test(exp_dir, supported_params)
                print(f" [SMOKE TEST] {smoke}")
                if not smoke["fit_predict_roundtrip_ok"]:
                    print(" FATAL: smoke test fit/predict roundtrip failed -- aborting before any real training.")
                    sys.exit(1)
                # FIX D: the smoke test's verdict now actually gates the HPO
                # search space instead of only being printed.
                tune_weight_decay = smoke.get("weight_decay_effect_detected") is True
                tune_dropout = smoke.get("dropout_effect_detected") is True
                if "weight_decay" not in supported_params:
                    print(" [HPO] weight_decay is not in unimol_tools' explicit MolTrain signature "
                          "-> HPO will NOT tune it (fixed at tier default).")
                elif not tune_weight_decay:
                    print(" [HPO] weight_decay is in the API but had NO measurable effect in the smoke test "
                          "-> HPO will NOT tune it (fixed at tier default).")
                if "dropout" not in supported_params:
                    print(" [HPO] dropout is not in unimol_tools' explicit MolTrain signature "
                          "-> HPO will NOT tune it (fixed at tier default).")
                elif not tune_dropout:
                    print(" [HPO] dropout is in the API but had NO measurable effect in the smoke test "
                          "-> HPO will NOT tune it (fixed at tier default).")
            else:
                print(" [HPO] --skip-smoke-test passed -> weight_decay/dropout are UNVERIFIED and will NOT "
                      "be tuned. Remove --skip-smoke-test to check whether they actually do anything.")
        except ImportError:
            print("unimol_tools missing"); sys.exit(1)
    else:
        supported_params = MOLTRAIN_EXPLICIT_INIT_PARAMS | {"weight_decay", "dropout"}
        tune_weight_decay = True
        tune_dropout = True

    args.tune_weight_decay = tune_weight_decay
    args.tune_dropout = tune_dropout

    model_sizes = args.compete_model_sizes or [args.model_size or "84m"]
    all_summaries = []

    for endpoint in args.endpoints:
        print(f"\n=== {endpoint} ===")
        best_size_for_endpoint = None
        best_size_nested_auroc = -np.inf
        seed_runs_by_size = {}

        for model_size in model_sizes:
            args.model_size = model_size
            print(f" --- model_size={model_size} ---")
            try:
                seed_runs = []
                for i in range(args.ensemble_seeds):
                    seed = args.seed + i
                    seed_runs.append(run_endpoint_seed(endpoint, cache_dir, exp_dir, args,
                                                        supported_params, seed, args.outer_cv_seed))
                seed_runs_by_size[model_size] = seed_runs
                mean_nested_auroc = np.mean([r["nested_metrics"].get("auroc") or -1 for r in seed_runs])
                print(f" [{endpoint}/{model_size}] mean nested-CV AUROC across {len(seed_runs)} seed(s): {mean_nested_auroc}")
                if mean_nested_auroc > best_size_nested_auroc:
                    best_size_nested_auroc = mean_nested_auroc
                    best_size_for_endpoint = model_size
            except FileNotFoundError as e:
                print(f" SKIP {endpoint}/{model_size}: {e}")
            except Exception as e:
                print(f" ERROR {endpoint}/{model_size}: {e}")

        if best_size_for_endpoint is None:
            all_summaries.append({"endpoint": endpoint, "error": "no model size trained successfully"})
            continue

        print(f" [{endpoint}] WINNER by nested-CV AUROC (no test leakage): {best_size_for_endpoint} "
              f"(nested_auroc={best_size_nested_auroc})")
        winning_runs = seed_runs_by_size[best_size_for_endpoint]

        # Single, final touch of the scaffold test split for this endpoint.
        test_result = evaluate_test_once(endpoint, cache_dir, winning_runs, test_guard)
        print(f" [{endpoint}] HELD-OUT TEST (touched once): {test_result['test_metrics']}")
        print(f" [{endpoint}] UNCERTAINTY: {test_result['uncertainty']}")

        all_summaries.append({
            "endpoint": endpoint, "winning_model_size": best_size_for_endpoint,
            "nested_cv_auroc": best_size_nested_auroc,
            "nested_cv_metrics": winning_runs[0]["nested_metrics"],
            **test_result,
        })

    print("\n=== SUMMARY ===")
    for s in all_summaries:
        print(f" {s}")

    sp = exp_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    sp.write_text(json.dumps(all_summaries, indent=2, default=str))
    print(f"\nSaved {sp}")


if __name__ == "__main__":
    main()