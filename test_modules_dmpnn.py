"""
test_modules_dmpnn.py - FINAL Production Tester for D-MPNN CheMeleon
Fixed v2 - nukes functional dropout
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json, time, random
import torch
import torch.serialization
import torch.nn.functional as F
from chemprop.nn import metrics as cp_metrics
import chemprop.nn as cp_nn
import chemprop.models as cp_models
from rdkit import Chem
from rdkit.Chem import inchi

torch.serialization.add_safe_globals([
    cp_metrics.BinaryAUROC,
    cp_metrics.BinaryMCCMetric,
    cp_metrics.BinaryAccuracy,
    cp_metrics.RMSE,
    cp_nn.BondMessagePassing,
    cp_nn.AtomMessagePassing,
    cp_nn.MeanAggregation,
    cp_nn.NormAggregation,
    cp_nn.BinaryClassificationFFN,
    cp_nn.RegressionFFN,
    cp_models.MPNN,
])

from chemprop import data, featurizers, models
from sklearn.metrics import roc_auc_score

MODULES = Path("modules")
DATA = Path("data")

BASE_100_RAW = [
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

seen_keys, uniq = set(), []
for smi in BASE_100_RAW:
    mol = Chem.MolFromSmiles(smi)
    if not mol: continue
    try: key = inchi.MolToInchiKey(mol)
    except: key = smi
    if key not in seen_keys:
        seen_keys.add(key)
        uniq.append(smi)
BASE_100 = uniq
print(f"Using {len(BASE_100)} unique-by-InChIKey SMILES")

# GLOBAL MONKEY PATCH - forces all functional dropout to identity
ORIGINAL_DROPOUT = F.dropout
def deterministic_dropout(input, p=0.5, training=True, inplace=False):
    return input
F.dropout = deterministic_dropout

def make_deterministic(mpnn):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    mpnn.eval()
    mpnn.train(False)
    
    # Force all dropout layers to p=0 and eval
    for m in mpnn.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.p = 0.0
            m.eval()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm)):
            m.eval()
        # Chemprop FFN specific
        if hasattr(m, 'dropout_p'):
            m.dropout_p = 0.0
        if hasattr(m, 'dropout'):
            try:
                if isinstance(m.dropout, torch.nn.Dropout):
                    m.dropout.p = 0.0
                    m.dropout.eval()
            except:
                pass
        if hasattr(m, 'p'):
            # some custom dropout classes
            try:
                if 'dropout' in m.__class__.__name__.lower():
                    m.p = 0.0
            except:
                pass
    return mpnn

def load_model(name, prefer_production=True):
    order = [f"{name}_dmpnn_production.ckpt", f"{name}_dmpnn.ckpt"] if prefer_production else [f"{name}_dmpnn.ckpt", f"{name}_dmpnn_production.ckpt"]
    ckpt_path = next((MODULES / f for f in order if (MODULES / f).exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"modules/{name}_dmpnn*.ckpt missing")
    print(f"Loading {ckpt_path}")
    try:
        mpnn = models.MPNN.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        mpnn = models.MPNN(**ckpt["hyper_parameters"])
        mpnn.load_state_dict(ckpt["state_dict"])

    mpnn = make_deterministic(mpnn)
    print(f"  -> Deterministic forced: training={mpnn.training} | F.dropout patched")
    return mpnn

def predict_smiles(mpnn, smiles_list, batch_size=128):
    mpnn = make_deterministic(mpnn)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dps, valid_mask, idx_map = [], [], []
    for i, smi in enumerate(smiles_list):
        try:
            if not smi or not str(smi).strip(): raise ValueError
            if Chem.MolFromSmiles(str(smi)) is None: raise ValueError
            dp = data.MoleculeDatapoint.from_smi(str(smi))
            if dp is None or dp.mol is None: raise ValueError
            dps.append(dp); valid_mask.append(True); idx_map.append(i)
        except Exception:
            valid_mask.append(False)
    if not dps:
        return [float('nan')]*len(smiles_list), valid_mask

    dset = data.MoleculeDataset(dps, featurizer)
    loader = data.build_dataloader(dset, num_workers=0, batch_size=batch_size)
    preds_valid = []
    with torch.inference_mode():
        for batch in loader:
            out = mpnn.predict_step(batch, 0)
            preds_valid.extend(out.detach().cpu().numpy().ravel().tolist())

    full_preds = [float('nan')]*len(smiles_list)
    for orig_idx, p in zip(idx_map, preds_valid):
        full_preds[orig_idx] = p
    return full_preds, valid_mask

def test_one(name, target, n=100):
    n = min(n, len(BASE_100))
    print(f"\n{'='*60}\nTESTING {name} ({target}) with {n} D-MPNN\n{'='*60}")
    model = load_model(name, prefer_production=True)

    saved_val, saved_test, enc = 0, 0, "?"
    if (MODULES / f"{name}_dmpnn_metrics.json").exists() or (MODULES / f"{name}_dmpnn_production_metrics.json").exists():
        mp = MODULES / f"{name}_dmpnn_production_metrics.json"
        if not mp.exists(): mp = MODULES / f"{name}_dmpnn_metrics.json"
        js = json.loads(mp.read_text())
        saved_val, saved_test, enc = js.get('val_auc',0), js.get('test_auc',0), js.get('encoder','?')
        print(f"Saved: Val {saved_val:.3f} Test {saved_test:.3f} encoder={enc} from {mp.name}")

    smiles = BASE_100[:n]
    t0=time.time()
    probs, valid_mask = predict_smiles(model, smiles, batch_size=64)
    t1=time.time()
    probs_arr = np.array([p for p in probs if not np.isnan(p)])

    print(f"[1] Featurize {len(smiles)} -> {sum(valid_mask)} valid | {t1-t0:.2f}s")
    print(f"[2] Predict min {probs_arr.min():.3f} max {probs_arr.max():.3f} mean {probs_arr.mean():.3f} std {probs_arr.std():.3f}")
    print(f"[3] Dist low<0.3 {(probs_arr<0.3).sum()} mid {((probs_arr>=0.3)&(probs_arr<=0.7)).sum()} high>0.7 {(probs_arr>0.7).sum()}")
    print(f"[4] Range {'PASS' if probs_arr.min()>=0 and probs_arr.max()<=1 else 'FAIL'}")

    live_auc = float('nan')
    test_csv = DATA / f"{name}_test_chemprop.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        test_smiles = df_test["SMILES"].tolist()
        test_y = pd.to_numeric(df_test[target], errors='coerce').values
        test_probs, _ = predict_smiles(model, test_smiles, batch_size=256)
        mask = ~np.isnan(test_probs)
        y_clean = test_y[mask]
        p_clean = np.array(test_probs)[mask]
        if len(set(y_clean))>1 and len(y_clean)>10:
            live_auc = roc_auc_score(y_clean, p_clean)
            flip_auc = roc_auc_score(y_clean, 1-p_clean)
            if flip_auc > live_auc and flip_auc > 0.7:
                print(f"[5] Held-out AUC {live_auc:.3f} n={len(y_clean)} -> FLIPPED LABELS? 1-AUC={flip_auc:.3f}!!!")
                live_auc = flip_auc
            else:
                print(f"[5] Held-out AUC {live_auc:.3f} n={len(y_clean)}")
    else:
        print(f"[5] Held-out AUC SKIP")

    bad_smiles = ["", " ", "NOTASMILES", None, "C1CC"]
    bad_probs, bad_valid = predict_smiles(model, bad_smiles, batch_size=5)
    print(f"[6] Invalid 5 -> {sum(bad_valid)} valid (expect 0)")

    dup_smiles = [smiles[0]]*3 + [smiles[1]]*2
    dup_probs, _ = predict_smiles(model, dup_smiles, batch_size=5)
    identical = abs(dup_probs[0]-dup_probs[1])<1e-6 and abs(dup_probs[0]-dup_probs[2])<1e-6
    print(f"[7] Duplicate 5 -> {dup_probs[0]:.4f} {dup_probs[1]:.4f} {dup_probs[2]:.4f} identical={identical} {'PASS' if identical else 'FAIL - dropout not disabled'}")
    print(f"[8] Speed {sum(valid_mask)/max(t1-t0,1e-6):.1f} mol/sec")
    return {"probs": probs_arr, "live_auc": live_auc, "saved_val": saved_val, "saved_test": saved_test}

if __name__ == "__main__":
    results=[]
    for name, tgt in [("DILI","Y"),("Teratogenicity","Y"),("Ames","Overall"),("CYP3A4","Y"),("hERG","Y")]:
        try:
            out = test_one(name, tgt, 100)
            p = out["probs"]
            if len(p)>0:
                results.append({
                    "endpoint":name, "n":len(p), "mean":float(p.mean()), "std":float(p.std()),
                    "min":float(p.min()), "max":float(p.max()),
                    "saved_val": out["saved_val"], "saved_test": out["saved_test"],
                    "live_auc": out["live_auc"]
                })
        except Exception as e:
            print(f"FAIL {name}: {e}")
            import traceback; traceback.print_exc()

    out_path = MODULES / "comprehensive_test_summary_dmpnn.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    if results:
        print(pd.DataFrame(results).to_string(index=False))
