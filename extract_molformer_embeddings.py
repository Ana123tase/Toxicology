"""
extract_molformer_embeddings.py

Single responsibility: SMILES -> pinned-revision MoLFormer embedding,
cached in molformer.db. Classical RDKit descriptors and the ECFP
fingerprint are owned by featurize.py / *_features_model.parquet, not
this script — join on `smiles` at train time instead of duplicating
that logic here.
"""
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# --- Pinned model version: DO NOT change without re-embedding everything ---
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
MODEL_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
DB_PATH = "molformer.db"
BATCH_SIZE = 4

DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]
SMILES_COLUMN = "SMILES"

DATASETS = ["DILI", "hERG", "CYP3A4", "Ames", "Teratogenicity"]
SQLITE_IN_CHUNK = 500

def configure_cpu_threads():
    n_cores = os.cpu_count() or 4
    n_threads = max(1, min(6, n_cores))
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Using {n_threads} CPU threads (detected {n_cores} logical cores).")

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            smiles TEXT PRIMARY KEY,
            vector BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skipped_smiles (
            smiles TEXT PRIMARY KEY,
            reason TEXT,
            skipped_at TEXT
        )
    """)
    cur = conn.execute("SELECT key, value FROM metadata")
    meta = dict(cur.fetchall())
    if not meta:
        conn.execute("INSERT INTO metadata (key, value) VALUES (?,?)",
                      ("model_revision", MODEL_REVISION))
        conn.execute("INSERT INTO metadata (key, value) VALUES (?,?)",
                      ("model_name", MODEL_NAME))
        conn.execute("INSERT INTO metadata (key, value) VALUES (?,?)",
                      ("pooling", "masked_mean"))
        conn.execute("INSERT INTO metadata (key, value) VALUES (?,?)",
                      ("embedding_dim", "768"))
        conn.commit()
    else:
        if meta.get("model_revision")!= MODEL_REVISION:
            raise RuntimeError(
                f"molformer.db was built with revision {meta.get('model_revision')}, "
                f"but current script uses {MODEL_REVISION}. Incompatible."
            )
        if meta.get("model_name") and meta.get("model_name")!= MODEL_NAME:
            raise RuntimeError(
                f"molformer.db was built with model {meta.get('model_name')}, "
                f"but current script uses {MODEL_NAME}. Incompatible — do not mix."
            )
        if meta.get("pooling") and meta.get("pooling")!= "masked_mean":
            raise RuntimeError(
                f"molformer.db was built with pooling={meta.get('pooling')}, "
                f"but current script uses masked_mean. Incompatible — do not mix."
            )
    return conn

def already_embedded(conn, smiles_list):
    found = set()
    for i in range(0, len(smiles_list), SQLITE_IN_CHUNK):
        chunk = smiles_list[i:i + SQLITE_IN_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(
            f"SELECT smiles FROM embeddings WHERE smiles IN ({placeholders})",
            chunk,
        )
        found.update(row[0] for row in cur.fetchall())
    return found

def already_skipped(conn, smiles_list):
    found = set()
    for i in range(0, len(smiles_list), SQLITE_IN_CHUNK):
        chunk = smiles_list[i:i + SQLITE_IN_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(
            f"SELECT smiles FROM skipped_smiles WHERE smiles IN ({placeholders})",
            chunk,
        )
        found.update(row[0] for row in cur.fetchall())
    return found

def save_skipped(conn, smiles_list, reason):
    if not smiles_list:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO skipped_smiles (smiles, reason, skipped_at) "
        "VALUES (?,?, datetime('now'))",
        [(s, reason) for s in smiles_list],
    )
    conn.commit()

def save_batch(conn, smiles_batch, vectors):
    for smi, vec in zip(smiles_batch, vectors):
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (smiles, vector) VALUES (?,?)",
            (smi, np.asarray(vec, dtype=np.float32).tobytes()),
        )
    conn.commit()

def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def embed_and_store(conn, tokenizer, model, smiles_list, label):
    already = already_embedded(conn, smiles_list)
    skipped = already_skipped(conn, smiles_list)
    to_process = [s for s in smiles_list if s not in already and s not in skipped]
    total = len(to_process)
    if total == 0:
        print(f"[{label}] All {len(smiles_list)} molecules already embedded or skipped. Skipping.")
        return

    to_process = sorted(to_process, key=len)
    print(f"[{label}] {total} molecules need embedding (of {len(smiles_list)} total).")
    start_time = time.time()

    max_len = tokenizer.model_max_length
    check_truncation = max_len is not None and max_len < int(1e6)

    for i in range(0, total, BATCH_SIZE):
        batch = to_process[i:i + BATCH_SIZE]
        orig_len = len(batch)

        if check_truncation:
            enc = tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
            long_idx = [j for j, ids in enumerate(enc["input_ids"]) if len(ids) > max_len]
            if long_idx:
                long_idx_set = set(long_idx)
                truncated = [batch[j] for j in long_idx]
                print(f"[{label}] WARNING: {len(truncated)} SMILES > max_length={max_len} — skipping")
                save_skipped(conn, truncated, f"truncated: exceeds max_length={max_len}")
                batch = [s for j, s in enumerate(batch) if j not in long_idx_set]
                if not batch:
                    done = min(i + orig_len, total)
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate / 60 if rate > 0 else float("inf")
                    print(f"[{label}] {done}/{total} done ({rate:.2f} mol/sec, ETA {eta:.1f} min)")
                    continue

        try:
            inputs = tokenizer(batch, padding=True, truncation=check_truncation, return_tensors="pt")
            with torch.inference_mode():
                outputs = model(**inputs)
            pooled = masked_mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            save_batch(conn, batch, pooled.detach().cpu().numpy())

        except Exception as e:
            print(f"[{label}] Batch error at {i}: {e} — retrying per-molecule")
            for smi in batch:
                try:
                    inp = tokenizer([smi], padding=True, truncation=check_truncation, return_tensors="pt")
                    with torch.inference_mode():
                        out = model(**inp)
                    p = masked_mean_pool(out.last_hidden_state, inp["attention_mask"])
                    save_batch(conn, [smi], p.detach().cpu().numpy())
                except Exception as e2:
                    # Not persisted to skipped_smiles: unlike the truncation
                    # case above, this failure mode (OOM, transient tokenizer
                    # issue, etc.) isn't guaranteed permanent, so we let it
                    # be retried on the next run instead of skipping it forever.
                    print(f"[{label}] Skipping {smi[:80]} for this run: {e2}")

        done = min(i + orig_len, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else float("inf")
        print(f"[{label}] {done}/{total} done ({rate:.2f} mol/sec, ETA {eta:.1f} min)")

def main():
    configure_cpu_threads()
    print("Loading MoLFormer (pinned revision)...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True, deterministic_eval=True
    )
    model.eval()
    print("Model loaded.\n")

    conn = init_db()

    for name in DATASETS:
        for split in SPLITS:
            path = DATA_DIR / f"{name}_{split}_chemprop.csv"
            label = f"{name}/{split}"
            if not path.exists():
                print(f"[{label}] File not found: {path} — skipping.\n")
                continue
            df = pd.read_csv(path)
            if SMILES_COLUMN not in df.columns:
                print(f"[{label}] Column '{SMILES_COLUMN}' not found. Available: {list(df.columns)} — skipping.\n")
                continue
            smiles_list = df[SMILES_COLUMN].dropna().unique().tolist()
            embed_and_store(conn, tokenizer, model, smiles_list, label)
            print()

    conn.close()
    print("Done. All embeddings saved to", DB_PATH)

if __name__ == "__main__":
    main()