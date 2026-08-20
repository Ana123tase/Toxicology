# tox-predict-ai

A production-grade molecular toxicity prediction model, built in phases from a
simple baseline to a full hybrid deep-learning system with an explainable API
and a deployable web app.


## 0. How this repo is organized

```
tox-predict-ai/
├── data/
│   ├── raw/              # original downloaded dataset(s), untouched
│   ├── processed/        # cleaned, split, deduplicated data
│   └── feature_store/    # frozen ECFP/RDKit features (Phase 1 deliverable)
├── src/                  # all reusable python code (no notebooks logic lives only here)
├── notebooks/            # exploration only — nothing production-critical lives only in a notebook
├── models/
│   ├── p1_catboost/
│   ├── p2_dmpnn/
│   └── p3_molformer/
├── benchmarks/           # benchmark_p1.csv, benchmark_p2.csv, benchmark_p3.csv
├── api/                  # Phase 4-5 serving code
└── docs/                 # write-ups, pricing page, PDF reports
```

Rule we're following throughout: **nothing moves to the next phase until its
benchmark CSV is committed and reviewed.** This is what makes the project
credible to a company evaluating it — every claim ("ensemble beats single
model") is backed by a logged number, not a vibe.

---

## Features

- **Multi-endpoint toxicity prediction** — probability of activity across
  multiple biological assays (e.g. receptor binding, cellular stress
  response), not just a single toxic/non-toxic label
- **Confidence scoring** — how much to trust each prediction
- **Applicability domain check** — flags when a molecule is too dissimilar
  from training data to predict reliably
- **Explainability** — highlights which atoms/substructures drove the
  prediction, instead of a black-box score

## Example usage

```python
from tox_predict import ToxPredictor

model = ToxPredictor.load("models/latest")
result = model.predict("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin

print(result)
# {
#   "prob": 0.12,
#   "confidence": 0.87,
#   "domain": "in_domain",
#   "neighbors": [...],
#   "atom_map": [...]
# }
```

## Built with

RDKit · CatBoost · D-MPNN · MoLFormer · Python

## Project structure

```
tox-predict-ai/
├── data/            # raw data, processed data, feature store
├── src/             # core pipeline code
├── notebooks/       # exploration/analysis
├── models/          # trained model artifacts
├── benchmarks/       # benchmark results
├── api/             # production serving code
└── docs/            # roadmap, write-ups, reports
```

## Development

# Abbreviations
DILI       Drug-Induced Liver Injury
CYP3A4     Cytochrome P450 3A4
hERG       Human Ether-à-go-go-Related Gene
ECFP       Extended-Connectivity Fingerprint
D-MPNN     Directed Message Passing Neural Network
SMILES     Simplified Molecular Input Line Entry System
SAR        Structure–Activity Relationship
RDKit      Rational Drug Design Toolkit
ECFP       Extended-Connectivity Fingerprint
TDC        Therapeutics Data Commons
SDF        Structure Data File
D-MPNN     Directed Message Passing Neural Network
CID        Compound Identifier (PubChem Compound ID)

# Development roadmap

**Current status:** 🚧 Phase 1 in progress — data pipeline being built. No
trained models yet.

Full phase-by-phase plan for `tox-predict-ai`. We're following a strict
gating rule: **no phase starts until the previous phase's benchmark CSV is
logged and reviewed.** Every added layer of complexity has to prove it
improves on the last, with numbers, not assumptions.

| Phase | Goal | Deliverable |
|---|---|---|
| **1** | Dataset → RDKit/ECFP features → CatBoost baseline | `benchmark_p1.csv` (AUC/F1 for 4 tox endpoints) + frozen feature store |
| **2** | Add D-MPNN (graph neural net) → ensemble with CatBoost | D-MPNN weights + `benchmark_p2.csv` proving ensemble > single model |
| **3** | Add MoLFormer (pretrained chemical transformer) → hybrid fusion | `molformer.db` + fusion model + `benchmark_p3.csv` (final model) |
| **4** | Confidence, applicability domain, explainability | API returns `{prob, confidence, domain, neighbors, atom_map}` |
| **5** | Fast production API → load testing → deployment | `/predict` < 300ms, PDF report, pricing page |
| **6** | Web app on top of the API | Usable front-end for non-technical users |

---

## Phase 1 — detailed plan

### Dataset & endpoints

**Dataset:** [Tox21](https://tripod.nih.gov/tox21/challenge/) — a public
NIH/EPA dataset of ~8,000 compounds tested across 12 toxicity assay
endpoints. Widely cited benchmark, small enough to iterate on quickly.

**Endpoints (4, default subset):**
- `NR-AR` (androgen receptor)
- `NR-AhR` (aryl hydrocarbon receptor)
- `SR-ARE` (antioxidant response element — oxidative stress)
- `SR-p53` (DNA damage response)

Can be swapped for a different subset or a company-provided dataset — this
just needs to be decided before featurization, since the feature store is
frozen at the end of Phase 1 and every later phase builds on it.

### Steps

1. **Environment** — Python env with `rdkit`, `catboost`, `scikit-learn`,
   `pandas`, `numpy`, pinned in `requirements.txt` for reproducibility.
2. **Data** — download Tox21, dedupe by canonical SMILES, drop unparseable
   molecules, stratified train/val/test split with a fixed random seed
   (this split is the fixed ruler every later phase is measured against).
3. **Featurization** — canonical SMILES → ECFP (Morgan fingerprints, radius
   2, 2048 bits) + optional RDKit physicochemical descriptors (MW, LogP,
   TPSA, etc.).
4. **Feature store** — versioned parquet files in `data/feature_store/`.
   Frozen means no silent regeneration later; changes get a new version.
5. **Model** — CatBoost classifier(s) with class-imbalance handling (tox
   datasets skew heavily toward "non-toxic").
6. **Benchmark** — AUC-ROC and F1 per endpoint → `benchmarks/benchmark_p1.csv`,
   plus a short note on what we observed (e.g. hardest endpoint, and why).

### Open items before coding starts

- Confirm dataset/endpoints above.
- Confirm working environment (this container vs. local clone).
- Any deadline or company-specific output requirement to design around now.

## Why this project

Most toxicity screening tools are closed-source, expensive, or don't explain
their predictions — a real problem in a domain where a wrong or opaque
answer can mean an unsafe compound moving forward. This project aims to be
accurate, fast, and transparent enough to actually trust.


## Dataset Sources

The toxicity datasets used in this project were obtained from independent,
downloadable sources rather than relying on the TDC API or Harvard Dataverse.

### 1. DILI

The Drug-Induced Liver Injury (DILI) dataset contains 475 compounds with
molecular structures represented as SMILES and binary toxicity labels.

The dataset was obtained from an independent Hugging Face mirror of the
TDC DILI dataset:

https://huggingface.co/datasets/scikit-fingerprints/TDC_dili

The downloaded file was saved locally as:

`data/DILI_raw.csv`

Format:

`SMILES, Y`

---

### 2. CYP3A4

The CYP3A4 dataset used in this project is the Veith CYP3A4 inhibition
dataset containing 12,328 compounds.

It was obtained from the independent Hugging Face dataset mirror:

https://huggingface.co/datasets/scikit-fingerprints/TDC_cyp3a4_veith

The downloaded file was saved as:

`data/CYP3A4_raw.csv`

Format:

`SMILES, Y`

---

### 3. hERG

The hERG dataset is the Karim hERG dataset containing 13,445 compounds.

It was obtained from the independent Hugging Face dataset mirror:

https://huggingface.co/datasets/scikit-fingerprints/TDC_herg_karim

The downloaded file was saved as:

`data/hERG_raw.csv`

Format:

`SMILES, Y`

---

### 4. Teratogenicity

The teratogenicity dataset was obtained from the public
LiSH7450/DIT_model repository associated with the study:

"Structure-activity relationship (SAR) model for predicting teratogenic
risk of antiepileptic drugs in pregnancy by using support vector machine."

The repository provides the molecular structures as `Train 2D.sdf`.

The repository documentation explicitly defines the first 67 molecules as
positive samples and the last 45 molecules as negative samples, giving
112 compounds in total.

The SDF file was processed using RDKit to generate canonical SMILES.
The labels were assigned according to the dataset definition:

- 67 positive / high-risk compounds → `Y = 1`
- 45 negative / low-risk compounds → `Y = 0`

The resulting dataset was saved as:

`data/Teratogenicity_raw.csv`

Format:

`SMILES, Y`

---

### 5. Ames Mutagenicity

The Ames mutagenicity endpoint uses the Hansen et al. benchmark dataset,
which contains 6,512 compounds with binary Ames mutagenicity outcomes.

The dataset consists of 3,053 mutagenic and 3,009 non-mutagenic compounds.
The benchmark provides molecular structures as canonical SMILES and binary
activity labels:

- `1` = mutagenic
- `0` = non-mutagenic

The publicly available Toxicity Benchmark provides the dataset in CSV/SMILES
format:

https://doc.ml.tu-berlin.de/toxbenchmark/

The Hansen benchmark was originally compiled from multiple sources,
including CCRIS, Helma et al., Kazius et al., Feng et al., VITIC and
GeneTox.

The dataset is used locally as the Ames mutagenicity dataset for this
project.

## Author

Boticario DUKUNDABERA Anastase: undergraduate project in Pharmacy
computational toxicology.
