# Toxicology ADMET Model — Project Status: developing a drug toxicity prediction module as an AI tool to be used in drug discover and drug design as a Pharmacy student, I have been requested this by industrial pharmacies and I am developing a money making project

# ALTHOUTH I AM DEVELOPING A PRODUCTION LEVEL PROJECT, THIS IS A LEARNING PROJECT, SO I MUST DO EACH AND EVERYSTEP BY MY SELF YOU JUST GIVE ME THE CODE AND TELL ME HOW TO RUN IT, DON'T RUN ANYTHING IN BACKGROUND, INSTEAD EXPLAIN EVERTHING TO DO TO ME AND TELL ME WHY AND HOW
I must run and learn everthing by my self in my local development laptop
My module must be a market leader coz I am coming to beat the market, I must be ahead of my competitors

My points of focus are only DILI, hERG, Ames, CYP3A4 and Teratogenicity
My development laptop is Lenovo  12th Gen Intel(R) Core(TM) i3-1215U (1.20 GHz) 8GB RAM, but I don' care


My project has 5 available endpoints:

DILI — Drug-Induced Liver Injury
hERG — hERG cardiac-channel inhibition
CYP3A4 — CYP3A4 activity/inhibition
Ames — Ames mutagenicity
Teratogenicity — teratogenicity

## Roadmap (6 phases)
1. Dataset → RDKit/ECFP → CatBoost → benchmark (`benchmark_p1.txt` + frozen feature store): This is already done, now moving to phase 2
2. Add D-MPNN and ensemble it with CatBoost→ benchmark → CatBoost+D-MPNN ensemble (`benchmark_p2.txt`): start
3. Add MoLFormer and make a cascade → hybrid fusion (`molformer.db` + `benchmark_p3.txt`)
NB: Here all algorithms (CatBoost, D-MPNN AND MolFormer) must participate in all mechanisms (Ensemble, cascade and hybrid)
N.B: At each level it checks if confidence is enough
4. Confidence + applicability domain + explainability (API: prob, confidence, domain, neighbors, atom_map)
5. Fast production API (<300ms) + load testing + PDF report + pricing page
6. Web app

## Decisions locked across the phases
- **Endpoint panel:** DILI, hERG, Ames, CYP3A4 and Teratogen (real ADMET panel, not arbitrary Tox21 endpoints — chosen for commercial relevance)
- **Model architecture:** Shared multi-task CatBoost (validated against single-task baselines in the benchmark, to confirm the multi-task bet actually helps)
- **Retraining strategy,** Warm-start incremental retraining (not live/online learning) — future learning: this is a must have for all of my algorithms used to build models of my project
- **Features:** RDKit + ECFP, radius 2, 2048 bits
- **Model library:** CatBoost, D-MPNN and MolFormer
- **Data source:** TDC (Therapeutics Data Commons), pulling from Harvard Dataverse but it is down now, we will do development without it

## What's actually been done

**Environment**
- Python env set up (conda + pip, Python 3.12) — resolved conda/pip binary conflicts
**SMILES cleaning module — `clean_smiles.py` (DONE, finalized and run on real data)**
- I have benchmark_p1.txt
- Featurization with featurize.py
- CATBOOST Training is successfully complete for all 5 production models. ✅
**500-drug CatBoost inference audit — DONE**

And all five .cbm files were successfully saved in:

modules/

Specifically:

DILI_catboost_production.cbm
hERG_catboost_production.cbm
CYP3A4_catboost_production.cbm
Ames_catboost_production.cbm
Teratogenicity_catboost_production.cbm

The important part is that no train/validation/test split was used for this final fit. My production models were trained using 100% of the available model-ready molecules.


- D-MPNN done but not tested coz my it will take my laptop a year of training

## Where we are right now (Updated 2026-08-19)

1.  500 unique check on CatBoost: done
2. D-MPNN development with chemprop: now we are doing training but it is taking very long so I am going to continue development of MolFormer

ensembling and stacking will be done later after development and training of all models

You have:

DILI: 463/463 molecules
hERG: 13,053/13,053
CYP3A4: 12,169/12,169
Ames: 5,517/5,517
Teratogenicity: 112/112

my previous train_catboost.py run supplied the validated tree counts:

DILI             232
hERG             499
CYP3A4           442
Ames             479
Teratogenicity   108

So now — the final CATBOOST production training stage is done.

**Key learning you flagged:**
> Different predictions do NOT mean accurate predictions. Accuracy must be checked against labelled held-out test set using *_metrics.json (AUC, F1), not by looking at uniqueness.

**Next is still Phase 1 original plan:** TDC pulls for DILI, hERG, Ames, CYP3A4 → save raw → clean_smiles → review.

## Fallback plan if Harvard Dataverse is still down and currently it is
Don't build this yet — only if TDC stays broken:
- Try DeepChem's MoleculeNet loaders (independent host): alrady done
- Or find direct CSV downloads from the original papers behind DILI/hERG/Ames/CYP3A4/Teratogens

## Near-term step sequence (after DILI pull succeeds)
Now I am heading straight to MolFormer development

## What matters most for "best in market" (keep priorities straight)
1. Benchmark credibility — honest, reproducible numbers, ideally matching/beating published literature
2. Correct evaluation methodology — **scaffold splits** done
3. Explainability + confidence (Phase 4) — matters more to buyers than raw accuracy
4. Architecture sophistication (D-MPNN, MoLFormer) only pays off once 1–3 are solid — a fancy model with sloppy evaluation loses to a simple model with rigorous evaluation: all algorithms (CatBoost, D-MPNN AND MolFormer) must participate in all mechanisms (Ensemble, cascade and hybrid)