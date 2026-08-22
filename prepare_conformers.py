"""
prepare_conformers.py

Step 1 of the Uni-Mol pipeline: converts your scaffold-split parquet files
(SMILES + label + descriptors + ECFP) into 3D-conformer input that Uni-Mol
can consume. RDKit descriptors/ECFP columns are ignored here -- Uni-Mol
learns its own representation from raw 3D structure, it doesn't take
precomputed features as input.

Follows the same "never guess, log for review" policy as featurize.py:
    - A molecule that fails conformer embedding is NOT given fabricated/random
    coordinates. It's dropped from the Uni-Mol-ready cache and logged to a
    review CSV, exactly like label conflicts and duplicate conflicts are
    handled in featurize.py. Your original parquet files are never touched.
    - Nothing is silently discarded without a trace you can inspect.

SPEED NOTE (Colab edition):
    Conformer embedding + force-field optimization (MMFF/UFF) is CPU-bound
    classical chemistry -- RDKit has no GPU path for it, so a T4 does not
    accelerate this particular step. GPU_STATUS below just reports what's
    visible in the runtime so you know it's there and idle for now; it gets
    used by your D-MPNN / MoLFormer training scripts later, not this one.

    All the actual speedup here is CPU parallelism: one worker process per
    molecule batch, each worker doing single-threaded RDKit work internally
    (mixing RDKit's own multithreading with process-level parallelism causes
    oversubscription and is usually *slower*, not faster).

    N_WORKERS now uses ALL vCPUs Colab gives the runtime. The previous
    "-2 headroom" cap existed because the original target was an 8GB/i3
    laptop that also had to run your IDE and OS alongside the script --
    Colab's CPU is dedicated to the notebook process, so there's nothing to
    reserve headroom for. Override with UNIMOL_PREP_WORKERS if you ever want
    to cap it (e.g. to leave a core free for another cell running in
    parallel).

    No science changed: same conformer count, same fallback policy, same
    outputs. This only changes how fast you get there.

Input: data/{name}_train.parquet, data/{name}_val.parquet, data/{name}_test.parquet
        (the output of scaffold_split.py)
Output: data/unimol_cache/{name}_{split}.pkl
        data/unimol_cache/{name}_{split}_conformer_failures.csv (only if any failures)
        data/unimol_cache/{name}_{split}_meta.json

Each .pkl is a list of dicts:
    {
        "smiles": str,
        "atoms": List[str] -- element symbols, length n_atoms (incl. Hs)
        "coordinates": np.ndarray -- shape (n_atoms, 3), lowest-energy conformer
        "target": int or float,
        "orig_index": int -- row index in the source split parquet
    }

Requires: pip install rdkit pandas numpy tqdm
"""

import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Same (path_stub, target_col) pairing used in featurize.py / scaffold_split.py.
# Kept here explicitly (not imported) so this script has no hard import
# dependency on your other files -- but the target_col values MUST match
# scaffold_split.py's DATASETS list. If you change one, change both.
TARGET_COL_BY_DATASET = {
    "DILI": "Y",
    "hERG": "Y",
    "CYP3A4": "Y",
    "Ames": "Overall",
    "Teratogenicity": "Y",
}

SPLITS = ("train", "val", "test")

# Conformer generation settings (unchanged from original)
N_CONFS_ATTEMPT = 5  # generate this many candidate conformers per molecule
MAX_EMBED_ATTEMPTS = 2  # 1 normal ETKDGv3 attempt, then 1 fallback with random coords
RANDOM_SEED = 42

# --- GPU probe (informational only -- see SPEED NOTE above) --------------
# This step has no GPU-executable code path (RDKit conformer/force-field
# work is CPU-only). We still check and print status so you can confirm
# Colab handed you a T4 before your later (actually GPU-bound) training
# steps, without silently pretending this script uses it.
def _probe_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, None
    except ImportError:
        return False, None


GPU_AVAILABLE, GPU_NAME = _probe_gpu()
DEVICE = "cuda" if GPU_AVAILABLE else "cpu"

# --- Speed settings (Colab: full-core CPU parallelism) -------------------
# Colab's CPU is dedicated to this notebook, so use every vCPU the runtime
# gives you -- no "-2 for OS/IDE" reservation needed (that was for sharing
# an 8GB laptop with other running programs).
DEFAULT_WORKERS = max(1, os.cpu_count() or 2)
N_WORKERS = int(os.environ.get("UNIMOL_PREP_WORKERS", DEFAULT_WORKERS))
# Rows handed to each worker per task-queue pop. Slightly larger than the
# laptop default to cut IPC/pickling overhead now that more workers are
# typically active at once. 16-24 is a good range for molecules this size.
CHUNKSIZE = 16
# -------------------------------------------------------------------------


def _convert_label(v):
    """Explicitly convert label as read from parquet to int/float, never raises."""
    if pd.isna(v):
        return v
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return int(f) if f.is_integer() else f
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        pass
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    # Unexpected non-numeric label: don't fabricate, don't raise - return as-is
    # for review per "never guess" policy.
    return v


def embed_and_optimize(mol: Chem.Mol, n_confs: int = N_CONFS_ATTEMPT):
    """
    Generate up to n_confs 3D conformers, optimize each with a force field,
    and return the lowest-energy conformer's coordinates.

    Returns (coords: np.ndarray[n_atoms, 3], atom_symbols: List[str]) on
    success, or (None, None) if embedding genuinely fails after fallback.

    Fallback policy: if standard ETKDGv3 embedding produces zero usable
    conformers (common for some macrocycles, unusual valence, unresolved
    stereochemistry), retry once with useRandomCoords=True. This is still
    RDKit's own geometry-generation algorithm, just a different starting
    strategy -- not fabricated/interpolated coordinates. If that also
    fails, the molecule is dropped and logged, per policy.

    Note: selecting the lowest MMFF/UFF energy conformer is reasonable for
    a single-conformer pipeline, but it is not necessarily the biologically
    relevant / bioactive conformation. This is a known limitation.
    """
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    # Kept at 1 deliberately: parallelism happens at the process level
    # (one molecule per worker), not inside RDKit's own conformer search.
    # Setting this >1 here while also using multiprocessing would oversubscribe
    # CPU threads and make things slower, not faster.
    params.numThreads = 1
    params.useRandomCoords = False

    conf_ids = []
    for attempt in range(MAX_EMBED_ATTEMPTS):
        params.useRandomCoords = (attempt > 0)
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
        if len(conf_ids) > 0:
            break

    if len(conf_ids) == 0:
        return None, None

    # Optimize each conformer. Try MMFF94 first (better parameterized for
    # organics); some atom types (e.g. certain metal-containing molecules --
    # relevant given recover_metal_ions.py in this project) aren't covered
    # by MMFF, so fall back to UFF which has broader atom-type coverage.
    energies = {}
    for cid in conf_ids:
        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid
            )
            if ff is None:
                raise ValueError("MMFF unavailable for this molecule")
            ff.Minimize(maxIts=500)
            energies[cid] = ff.CalcEnergy()
        except Exception:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if ff is None:
                    continue
                ff.Minimize(maxIts=500)
                energies[cid] = ff.CalcEnergy()
            except Exception:
                continue

    if not energies:
        return None, None

    best_cid = min(energies, key=energies.get)
    conf = mol.GetConformer(best_cid)
    coords = conf.GetPositions()  # (n_atoms, 3)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    return np.asarray(coords, dtype=np.float32), atom_symbols


def _process_one(task):
    """
    Worker-process entry point. Must be a top-level function (not a closure/
    lambda/method) so it can be pickled and sent to each worker process.

    task: (orig_idx, smi, label_raw)
    Returns: ("ok", record_dict) or ("fail", failure_dict)
    """
    orig_idx, smi, label_raw = task

    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return "fail", {"orig_index": orig_idx, "smiles": smi, "reason": "MolFromSmiles failed"}

    coords, atoms = embed_and_optimize(mol)
    if coords is None:
        return "fail", {"orig_index": orig_idx, "smiles": smi, "reason": "conformer embedding failed"}

    label = _convert_label(label_raw)

    record = {
        "smiles": smi,
        "atoms": atoms,
        "coordinates": coords,
        "target": label,
        "orig_index": orig_idx,
    }
    return "ok", record


def process_split(path: str, dataset_name: str, split_name: str, target_col: str,
                   smiles_col: str = "SMILES", progress_every: int = 200):
    print(f"\n=== {dataset_name} / {split_name} ===")
    df = pd.read_parquet(path)
    n = len(df)
    print(f"  Input: {n} rows  |  workers={N_WORKERS}  chunksize={CHUNKSIZE}")

    # Build the task list once, up front, using itertuples (much faster than
    # repeated df.iloc[pos] lookups in the original loop, which re-resolves
    # column labels on every access).
    tasks = [
        (row.Index, getattr(row, smiles_col), getattr(row, target_col))
        for row in df.itertuples(index=True)
    ]

    records = []
    failures = []
    t0 = time.time()
    done = 0

    if N_WORKERS <= 1 or n < 50:
        # Not worth spinning up a pool for tiny splits (e.g. your 112-row
        # Teratogenicity test fold) -- process overhead would dominate.
        results_iter = (_process_one(t) for t in tasks)
    else:
        pool = Pool(processes=N_WORKERS)
        results_iter = pool.imap_unordered(_process_one, tasks, chunksize=CHUNKSIZE)

    try:
        for status, payload in results_iter:
            if status == "ok":
                records.append(payload)
            else:
                failures.append(payload)
            done += 1
            if progress_every and done % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  [{dataset_name}/{split_name}] {done}/{n} "
                      f"({done / elapsed:.1f} rows/sec, {len(failures)} failures so far)")
    finally:
        if N_WORKERS > 1 and n >= 50:
            pool.close()
            pool.join()

    print(f"  [{dataset_name}/{split_name}] done: {len(records)} usable, {len(failures)} failed "
          f"({len(failures) / n:.1%} failure rate)")

    return records, failures, n


def save_outputs(records, failures, n_input, dataset_name, split_name, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / f"{dataset_name}_{split_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    print(f"  Saved: {pkl_path} ({len(records)} molecules)")

    if failures:
        fail_path = out_dir / f"{dataset_name}_{split_name}_conformer_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"  Saved: {fail_path} ({len(failures)} rows -- review before assuming "
              f"this split is representative)")

    meta = {
        "dataset": dataset_name,
        "split": split_name,
        "n_input_rows": n_input,
        "n_usable_molecules": len(records),
        "n_conformer_failures": len(failures),
        "failure_rate": round(len(failures) / n_input, 4) if n_input else 0.0,
        "n_confs_attempted": N_CONFS_ATTEMPT,
        "random_seed": RANDOM_SEED,
        "n_workers_used": N_WORKERS,
        "gpu_available": GPU_AVAILABLE,
        "gpu_name": GPU_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_dir / f"{dataset_name}_{split_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


def main():
    if GPU_AVAILABLE:
        print(f"GPU detected: {GPU_NAME} (idle for this step -- conformer generation "
              f"is CPU-only; the GPU will be used by your D-MPNN/MoLFormer scripts)")
    else:
        print("No GPU detected -- running on CPU (this step wouldn't use a GPU anyway)")
    print(f"CPU workers for this run: {N_WORKERS} (all available vCPUs)")

    output_dir = "data/unimol_cache"
    summary = []

    for dataset_name, target_col in TARGET_COL_BY_DATASET.items():
        for split_name in SPLITS:
            input_path = f"data/{dataset_name}_{split_name}.parquet"
            if not Path(input_path).exists():
                print(f"\nNot found, skipping: {input_path}")
                continue

            try:
                records, failures, n_input = process_split(
                    input_path, dataset_name, split_name, target_col
                )
                save_outputs(records, failures, n_input, dataset_name, split_name, output_dir)
                summary.append({
                    "dataset": dataset_name, "split": split_name,
                    "n_input": n_input, "n_usable": len(records),
                    "n_failed": len(failures),
                })
            except Exception as e:
                print(f"  FAILED {input_path}: {e}")
                summary.append({
                    "dataset": dataset_name, "split": split_name,
                    "n_input": None, "n_usable": None, "n_failed": None, "error": str(e),
                })

    print("\n=== SUMMARY ===")
    for row in summary:
        print(f"  {row}")

    high_failure = [r for r in summary if r.get("n_input") and r["n_failed"] / r["n_input"] > 0.05]
    if high_failure:
        print("\n  NOTE: the following splits lost >5% of molecules to conformer failure --")
        print("  check the *_conformer_failures.csv before trusting downstream metrics,")
        print("  especially for your small endpoints (DILI, Teratogenicity), where every")
        print("  dropped molecule is a larger fraction of an already-small test set:")
        for row in high_failure:
            print(f"  - {row['dataset']}/{row['split']}: "
                  f"{row['n_failed']}/{row['n_input']} failed")


if __name__ == "__main__":
    # Required on Windows/macOS for multiprocessing with the "spawn" start
    # method; harmless no-op on Linux ("fork") -- and Colab notebooks run
    # Linux, but this guard is kept for safety/portability regardless.
    main()