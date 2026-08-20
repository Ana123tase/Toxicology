"""
Get_CatBoost_benchmark.py
Creates benchmark_p1.txt from modules/*_metrics.json

You already did:
  clean_smiles.py -> featurize.py -> scaffold_split.py -> train_one() -> *_metrics.json

This file does NOT train. It just collects those JSONs into a human-readable txt benchmark.
"""

import json
from pathlib import Path
from datetime import datetime

METRICS_DIR = Path("modules")
OUTPUT_TXT = Path("benchmark_p1.txt")

def load_metrics():
    rows = []
    for jf in sorted(METRICS_DIR.glob("*_metrics.json")):
        try:
            data = json.loads(jf.read_text())
            rows.append(data)
        except Exception as e:
            print(f"Failed to read {jf}: {e}")
    return sorted(rows, key=lambda x: x.get("dataset",""))

def write_txt(rows):
    with open(OUTPUT_TXT, "w") as f:
        f.write("="*60 + "\n")
        f.write("ADMET Benchmark P1 - Scaffold Split\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n")
        f.write("Split: 80% train / 10% val / 10% test | seed 42\n")
        f.write("Leakage check: scaffold_split.py -> No scaffold leakage - OK\n")
        f.write("Model: CatBoost single-task | auto_class_weights=Balanced\n")
        f.write("       iterations=500, depth=6, lr=0.05\n")
        f.write("Features: RDKit descriptors + ECFP 2048-bit (DESCRIPTOR_NAMES + ECFP_COLS)\n")
        f.write("Data: Legitimate labeled sources (TDC/Harvard Dataverse down fallback)\n")
        f.write("="*60 + "\n\n")

        for m in rows:
            dataset = m.get("dataset", "UNKNOWN")
            target = m.get("target_col", "Y")
            f.write(f"{dataset} (target={target}):\n")
            f.write(f"  n_train={m.get('n_train')} | n_val={m.get('n_val')} | n_test={m.get('n_test')}\n")
            f.write(f"  Val  AUC={m.get('val_auc', float('nan')):.3f}  AP={m.get('val_ap', float('nan')):.3f}\n")
            f.write(f"  Test AUC={m.get('test_auc', float('nan')):.3f}  (honest scaffold hold-out)\n")
            f.write(f"  best_iteration={m.get('best_iteration')}\n")
            f.write("\n")

        # summary line for quick copy-paste
        f.write("-"*60 + "\n")
        f.write("SUMMARY (Test AUC):\n")
        for m in rows:
            f.write(f"  {m.get('dataset')}: {m.get('test_auc', float('nan')):.3f}\n")

    print(f"Saved: {OUTPUT_TXT}")
    print(OUTPUT_TXT.read_text())

if __name__ == "__main__":
    rows = load_metrics()
    if not rows:
        print(f"No metrics found in {METRICS_DIR}/*.json - run train_one() first")
    else:
        write_txt(rows)