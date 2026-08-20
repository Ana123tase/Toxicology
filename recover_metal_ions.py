"""
recover_metal_ions.py

For every row in a *_review.csv flagged as `metal_complex`, decide
whether it's actually simple: a plain ionic counterion (like a lone
Na+, K+, Ca2+) sitting next to one clean organic parent -- chemically
no different from the Cl-/Br- salts clean_smiles.py already strips
automatically -- versus a genuine bonded/coordinated metal complex,
which is NOT safe to auto-recover.

The rule used to tell these apart:
  - A SIMPLE IONIC COUNTERION is a metal fragment that is a single,
    isolated atom (no bonds to anything, since it split off as its
    own fragment) carrying a nonzero formal charge. That's the exact
    definition of "free ion in solution."
  - A GENUINE METAL COMPLEX has the metal atom BONDED to other atoms
    within its own fragment (a multi-atom fragment) -- e.g. a real
    coordination complex, or an organometallic compound. These are
    never auto-recovered here; they still require a human to look at
    them, same as clean_smiles.py's original conservative design.
  - Multiple organic fragments are allowed IF, and only if, they are
    all structurally IDENTICAL -- the same "nothing to guess" precedent
    clean_smiles.py itself already uses for e.g. "CCN.CCN". This
    matters a lot in practice: divalent/trivalent metals (Ca2+, Mg2+,
    Al3+) very commonly pair with 2 or 3 copies of the SAME organic
    anion to balance charge (e.g. calcium diacetate, aluminum
    triacetate). Without this check, every one of those completely
    ordinary salts would be wrongly routed to manual review just
    because the metal isn't monovalent -- a real gap in an earlier
    version of this script, caught by testing against synthetic
    divalent/trivalent cases before this was ever run on real data.
    DIFFERENT organic fragments alongside a metal are still always
    rejected -- that's a genuine mixture, not resolved here.

This script does NOT modify your existing clean/review CSVs. It writes
a new *_metal_recovery.csv per dataset for YOU to inspect before
deciding to merge any of it back into your clean dataset. Nothing is
auto-merged.

Note on charge: recovered structures are NOT neutralized. If the
organic parent was e.g. an acetate anion (CC(=O)[O-]) paired with the
metal, the recovered SMILES stays as the anion, not neutral acetic
acid. This matches how clean_smiles.py already treats non-metal salts
elsewhere (e.g. "CC[NH3+].[Cl-]" -> "CC[NH3+]", charge preserved) --
it's an existing, consistent policy, not something new introduced
here. If you want neutral forms, that's a separate, deliberate
post-processing decision -- don't let it happen silently as a side
effect of this script.

USAGE: python recover_metal_ions.py
"""

import pandas as pd
from rdkit import Chem

import clean_smiles as cs


def is_simple_ionic_metal(fragment: Chem.Mol) -> bool:
    """
    True only if this fragment is a single metal atom carrying a
    nonzero formal charge -- i.e. a free ion, not bonded to anything.
    A metal atom with any bonds (part of a larger fragment) is NOT
    considered simple, regardless of charge.
    """
    if fragment.GetNumAtoms() != 1:
        return False
    atom = fragment.GetAtomWithIdx(0)
    if atom.GetSymbol() not in cs.METAL_SYMBOLS:
        return False
    return atom.GetFormalCharge() != 0


def classify_metal_row(raw_smiles: str):
    """
    Re-derive fragment categories for one metal_complex row (using the
    same _categorize_fragment logic as clean_smiles.py) and decide:
      'auto_recoverable'  -> exactly one UNIQUE organic parent structure
                             (one copy, or several identical copies --
                             e.g. calcium diacetate), and all metal
                             fragments are simple free ions
      'needs_manual_review' -> anything else (multiple DIFFERENT
                             parents, a bonded/complex metal fragment,
                             an ambiguous fragment also present, or no
                             organic parent at all)
    Returns (recommendation, recovered_smiles_or_None, reason).
    """
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return "needs_manual_review", None, "could not re-parse SMILES"

    fragments = list(Chem.GetMolFrags(mol, asMols=True))
    has_siblings = len(fragments) > 1
    categories = [cs._categorize_fragment(f, has_siblings) for f in fragments]

    metals = [f for f, c in zip(fragments, categories) if c == "metal"]
    parents = [f for f, c in zip(fragments, categories) if c == "parent"]
    ambiguous = [f for f, c in zip(fragments, categories) if c == "ambiguous"]

    if ambiguous:
        return "needs_manual_review", None, "also has an ambiguous fragment, not just a metal"

    if not parents:
        return "needs_manual_review", None, "no organic parent fragment remained"

    # Multiple parents are only OK if they're all the SAME structure --
    # e.g. calcium diacetate (Ca2+ + two identical acetate anions).
    # Different parents alongside a metal is a genuine mixture, not
    # something this script resolves.
    canon_parents = [cs._canonical(p) for p in parents]
    n_unique_parents = len(set(canon_parents))
    if n_unique_parents != 1:
        return ("needs_manual_review", None,
                f"{n_unique_parents} DIFFERENT organic parent candidates "
                f"(expected exactly 1 unique structure)")

    if not all(is_simple_ionic_metal(m) for m in metals):
        return ("needs_manual_review", None,
                "metal fragment is bonded/multi-atom -- looks like a real "
                "coordination complex or organometallic, not a simple counterion")

    recovered = canon_parents[0]
    metal_symbols = ",".join(sorted({m.GetAtomWithIdx(0).GetSymbol() for m in metals}))
    return "auto_recoverable", recovered, f"simple ionic counterion(s) removed: {metal_symbols}"


def recover_dataset(review_path: str, output_path: str):
    df = pd.read_csv(review_path)
    metal_rows = df[df["cleaning_status"] == "metal_complex"].copy()

    if len(metal_rows) == 0:
        print(f"{review_path}: no metal_complex rows found, skipping.")
        return

    recs = metal_rows["SMILES_raw"].apply(classify_metal_row)
    metal_rows["recommendation"] = recs.apply(lambda r: r[0])
    metal_rows["recovered_SMILES"] = recs.apply(lambda r: r[1])
    metal_rows["recovery_reason"] = recs.apply(lambda r: r[2])

    n_auto = (metal_rows["recommendation"] == "auto_recoverable").sum()
    n_manual = (metal_rows["recommendation"] == "needs_manual_review").sum()

    metal_rows.to_csv(output_path, index=False)

    print(f"{review_path}")
    print(f"  Total metal_complex rows: {len(metal_rows)}")
    print(f"  Auto-recoverable (simple ionic counterion): {n_auto}")
    print(f"  Still needs manual review (bonded/complex metal): {n_manual}")
    print(f"  Written: {output_path}  <- inspect this before merging anything")
    print()


DATASETS = [
    ("data/DILI_review.csv", "data/DILI_metal_recovery.csv"),
    ("data/hERG_review.csv", "data/hERG_metal_recovery.csv"),
    ("data/CYP3A4_review.csv", "data/CYP3A4_metal_recovery.csv"),
    ("data/Ames_review.csv", "data/Ames_metal_recovery.csv"),
    ("data/Teratogenicity_review.csv", "data/Teratogenicity_metal_recovery.csv"),
]

if __name__ == "__main__":
    for review_path, output_path in DATASETS:
        try:
            recover_dataset(review_path, output_path)
        except FileNotFoundError:
            print(f"{review_path}: not found, skipping (run rerun_cleaning.py first)\n")