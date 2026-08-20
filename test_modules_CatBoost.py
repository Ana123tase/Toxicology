from pathlib import Path
import pandas as pd
import numpy as np
import json, time
from catboost import CatBoostClassifier
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from featurize import featurize_dataframe, build_model_ready, DESCRIPTOR_NAMES, ECFP_COLS

FEATURE_COLS = DESCRIPTOR_NAMES + ECFP_COLS
MODULES = Path("modules")
DATA = Path("data")

# 100 UNIQUE diverse SMILES - pre-deduped by InChIKey
BASE_100 = [
"CCO","CCCO","CCCCO","CC(C)O","CC(C)CO","CCCCCO","CCOCC","CCN","CCNCC",
"CC(=O)O","CC(=O)OC","CC(=O)NC","CC(=O)Oc1ccccc1C(=O)O",
"c1ccccc1","c1ccccc1O","c1ccccc1N","c1ccccc1Cl","c1ccccc1Br","c1ccccc1F","c1ccccc1C",
"c1ccncc1","c1ccsc1","c1ccoc1","C1CCCCC1","C1CCCC1","C1CCNCC1","C1CCOC1",
"CN1C=NC2=C1C(=O)N(C(=O)N2C)C","CN1CCC[C@H]1c2cccnc2","CC(C)Cc1ccc(cc1)C(C)C(=O)O",
"CC(C)(C)C1=CC=C(C=C1)O","CC(C)CC(C(=O)O)N","COc1ccccc1O","COCCOC","Cc1ccccc1C",
"Cc1ccccc1O","CCc1ccccc1","CCCc1ccccc1","CC(C)c1ccccc1","CC(C)(C)c1ccccc1","CN(C)C",
"C1=CC=C2C(=C1)C=CC=C2","c1ccc2ccccc2c1","c1ccc2[nH]ccc2c1","CC(=O)C","CCOCCO","OCCO",
"NCCO","CS(=O)(=O)O","C1CC1","C1CCC1","C1CCCC1C","C1CCCCC1O","C1=CCOC=C1","C1=COC=C1",
"CC(C)OC(=O)C","CCC(=O)O","CCCC(=O)O","CCCN","CCCCN","c1ccc(cc1)CCO","c1ccc(cc1)CCN",
"c1ccc(cc1)CC(=O)O","CC(=O)CCO","CCOC(C)=O","CCN(C)C(=O)C","c1ccc(cc1)C#N","c1ccc(cc1)S(=O)(=O)N",
"CC1=CC=CC=C1C","CC1=CC=CC=C1O","CC1=CC=CO1","CC1=CC=CS1","CC1=CN=CC=C1","CCOC1=CC=CC=C1",
"COC1=CC=CC=C1OC","CCOC(=O)C","CCOC(=O)CC","CCC(=O)NC","CCNC(=O)OC","CC(C)C(=O)NC",
"CC(C)OC1=CC=CC=C1","CCC1=CC=CC=C1C","CC1=CC=C(C=C1)Cl","CC1=CC=C(C=C1)Br","CC1=CC=C(C=C1)F",
"c1ccc(cc1)C(F)(F)F","c1ccc(cc1)OC(F)(F)F","CC(=O)C1=CC=CC=C1","CC(=O)C1=CN=CC=C1","CC1=CC(=O)NC=C1",
"CNC1=CC=CC=C1","CN(C)C1=CC=CC=C1","CCN(C)C1=CC=CC=C1","CCOC1=CC=C(C=C1)C","CC1=CC=C(C=C1)C(=O)O"
]
# keep only valid and unique by InChIKey
from rdkit.Chem import inchi
seen_keys = set()
uniq = []
for smi in BASE_100:
    mol = Chem.MolFromSmiles(smi)
    if not mol: continue
    try: key = inchi.MolToInchiKey(mol)
    except: key = smi
    if key not in seen_keys:
        seen_keys.add(key)
        uniq.append(smi)
BASE_100 = uniq[:100]
print(f"Using {len(BASE_100)} unique-by-InChIKey SMILES")

def load_model(name):
    m = CatBoostClassifier()
    m.load_model(str(MODULES / f"{name}_catboost.cbm"))
    return m

def test_one(name, target, n=100):
    print(f"\n{'='*60}\nTESTING {name} ({target}) with {n}\n{'='*60}")
    model = load_model(name)

    if (MODULES / f"{name}_metrics.json").exists():
        js = json.loads((MODULES / f"{name}_metrics.json").read_text())
        print(f"Saved: Val {js['val_auc']:.3f} Test {js['test_auc']:.3f}")

    # FIX: only ONE label column, not Y + Overall
    smiles = BASE_100[:n]
    df = pd.DataFrame({"SMILES": smiles, target: [0]*len(smiles)})

    t0=time.time()
    full = featurize_dataframe(df, smiles_col="SMILES", dataset_name=f"{name}_test")
    ready, conflicts, label_review = build_model_ready(full, target_col=target, dataset_name=f"{name}_test")
    t1=time.time()

    print(f"[1] Featurize {len(df)} -> {len(ready)} ready | conflicts {len(conflicts)} | review {len(label_review)} | {t1-t0:.2f}s")
    assert len(ready)==len(df), f"Should keep all {len(df)}, got {len(ready)} - dupe list not deduped!"

    missing = [c for c in FEATURE_COLS if c not in ready.columns]
    assert not missing, f"Missing cols {missing[:3]}"
    print(f"[2] FEATURE_COLS OK {len(FEATURE_COLS)}")

    ready[FEATURE_COLS] = ready[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(ready[FEATURE_COLS])[:,1]
    print(f"[3] Predict OK min {probs.min():.3f} max {probs.max():.3f} mean {probs.mean():.3f} std {probs.std():.3f}")
    print(f"[4] Dist low<0.3 {(probs<0.3).sum()} mid {((probs>=0.3)&(probs<=0.7)).sum()} high>0.7 {(probs>0.7).sum()}")

    # held-out
    if (DATA / f"{name}_test.parquet").exists():
        test = pd.read_parquet(DATA / f"{name}_test.parquet")
        test[FEATURE_COLS] = test[FEATURE_COLS].fillna(0)
        auc = roc_auc_score(test[target], model.predict_proba(test[FEATURE_COLS])[:,1]) if test[target].nunique()>1 else float('nan')
        print(f"[5] Held-out AUC {auc:.3f}")

    # invalid
    bad = pd.DataFrame({"SMILES": ["", " ", "NOTASMILES", None, "C1CC"], target: [0]*5})
    full_bad = featurize_dataframe(bad, dataset_name=f"{name}_bad")
    ready_bad, _, _ = build_model_ready(full_bad, target_col=target, dataset_name=f"{name}_bad")
    print(f"[6] Invalid 5 -> {len(ready_bad)} kept (expect 0)")

    # duplicate conflict - FIXED: use same target only
    dup_df = pd.DataFrame({"SMILES": [smiles[0]]*3 + [smiles[1]]*2, target: [0,0,1,0,0]})
    full_dup = featurize_dataframe(dup_df, dataset_name=f"{name}_dup")
    ready_dup, conf_dup, _ = build_model_ready(full_dup, target_col=target, dataset_name=f"{name}_dup")
    print(f"[7] Duplicate conflict 5 rows -> {len(ready_dup)} kept {len(conf_dup)} conflict (expect 1 kept, 3 conflict)")

    print(f"[8] Speed {len(smiles)/(t1-t0):.1f} mol/sec")
    return probs

if __name__ == "__main__":
    results=[]
    for name, tgt in [("DILI","Y"),("Teratogenicity","Y"),("Ames","Overall"),("CYP3A4","Y"),("hERG","Y")]:
        try:
            p = test_one(name, tgt, 100)
            results.append({"endpoint":name, "n":len(p), "mean":float(p.mean()), "std":float(p.std()), "min":float(p.min()), "max":float(p.max())})
        except Exception as e:
            print(f"FAIL {name}: {e}")
            import traceback; traceback.print_exc()

    out = MODULES / "comprehensive_test_summary.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(pd.DataFrame(results).to_string(index=False))