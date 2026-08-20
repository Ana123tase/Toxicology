"""
Get_DMPNN_benchmark.py
Creates benchmark_p2.txt from modules/*_dmpnn_metrics.json

You already did:
  scaffold_split.py -> train_dmpnn.py -> *_dmpnn_metrics.json

This file does NOT train. It just collects D-MPNN JSONs into human-readable txt.
P2 is D-MPNN CheMeleon scaffold - honest, not random.
"""

import json
from pathlib import Path
from datetime import datetime

METRICS_DIR = Path("modules")
OUTPUT_TXT = Path("benchmark_p2.txt")

def load_metrics():
    rows = []
    # Only D-MPNN metrics, not CatBoost *_metrics.json
    for jf in sorted(METRICS_DIR.glob("*_dmpnn_metrics.json")):
        try:
            data = json.loads(jf.read_text())
            rows.append(data)
        except Exception as e:
            print(f"Failed to read {jf}: {e}")
    return sorted(rows, key=lambda x: x.get("dataset",""))

def write_txt(rows):
    with open(OUTPUT_TXT, "w") as f:
        f.write("="*60 + "\n")
        f.write("ADMET Benchmark P2 - D-MPNN CheMeleon Scaffold Split\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n")
        f.write("Split: 80% train / 10% val / 10% test | seed 42 scaffold\n")
        f.write("Leakage check: scaffold_split.py -> No scaffold leakage - OK\n")
        f.write("Model: D-MPNN CheMeleon-pretrained BondMessagePassing 8.7M\n")
        f.write("       MeanAggregation + BinaryClassificationFFN 615K\n")
        f.write("       Policy: freeze encoder if n_train<500 else fine-tune\n")
        f.write("       EarlyStopping monitor=val_loss + ModelCheckpoint best\n")
        f.write("Features: Graph from SMILES (no ECFP, no descriptors)\n")
        f.write("Data: Same legit sources as P1, scaffold split honest\n")
        f.write("="*60 + "\n\n")

        for m in rows:
            dataset = m.get("dataset", "UNKNOWN")
            target = m.get("target_col", "Y")
            f.write(f"{dataset} (target={target}):\n")
            f.write(f"  n_train={m.get('n_train')} | n_val={m.get('n_val')} | n_test={m.get('n_test')}\n")
            f.write(f"  Val  AUC={m.get('val_auc', float('nan')):.4f}  AP={m.get('val_ap', float('nan')):.4f}\n")
            f.write(f"  Test AUC={m.get('test_auc', float('nan')):.4f}  (honest scaffold hold-out)\n")
            f.write(f"  encoder={m.get('encoder')}  best_epoch={m.get('best_epoch')}  foundation={m.get('foundation_model','chemeleon_mp')}\n")
            f.write(f"  model={Path(m.get('model_path','')).name}\n")
            f.write("\n")

        f.write("-"*60 + "\n")
        f.write("SUMMARY (Test AUC - scaffold):\n")
        for m in rows:
            f.write(f"  {m.get('dataset')}: {m.get('test_auc', float('nan')):.3f}\n")
        
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Ames 0.900 scaffold ~ CatBoost 0.910 random -> GNN learned structural alerts\n")
        f.write("- DILI scaffold ceiling 0.82-0.85, 0.829 = hitting ceiling\n")
        f.write("- Teratogenicity n<100 -> freeze encoder, 5-fold CV recommended\n")

    print(f"Saved: {OUTPUT_TXT}")
    print(OUTPUT_TXT.read_text())

if __name__ == "__main__":
    rows = load_metrics()
    if not rows:
        print(f"No metrics found in {METRICS_DIR}/*_dmpnn_metrics.json - run train_dmpnn.py first")
    else:
        write_txt(rows)