#!/usr/bin/env python3
"""
clean_smiles.py
================

Conservative SMILES cleaning and structural QC stage (NOT aggressive standardization) for a multi-endpoint
drug toxicity ML project (Ames, CYP3A4, hERG, DILI, Teratogenicity).

This script does ONE job: turn raw SMILES into a conservatively cleaned,
traceable, audit-ready dataset. It does NOT compute descriptors or
fingerprints, and it does NOT split data into train/test sets or build
models. Those belong in later pipeline stages.

    RAW DATA
        |
    SMILES VALIDATION
        |
    FRAGMENT ANALYSIS
        |
    SAFE SALT STANDARDIZATION      (non-metal counterions only)
        |
    MIXTURE / METAL / AMBIGUITY DETECTION
        |
    STANDARDIZED SMILES
        |
    STEREOCENTER AUDIT              (counted, never resolved/guessed)
        |
    DUPLICATE ANALYSIS             (InChIKey-based)
        |
    FINAL CLEAN DATASET + REVIEW DATASET + AUDIT REPORT

GUIDING PRINCIPLE: NEVER GUESS.
If the cleaner can confidently determine the chemical structure, it
standardizes it. If it cannot, the record is routed to the review
dataset with a human-readable reason -- it is never dropped and never
silently "fixed" with a heuristic. A smaller, chemically reliable
clean dataset is preferred over a larger dataset that may contain
mis-standardized structures.

CLEANING STATUS VALUES
-----------------------
clean               valid SMILES, single unambiguous organic parent,
                     nothing needed to be removed
salt_removed        one or more recognized non-metal counterions/
                     solvent fragments were removed; exactly one
                     organic parent remained
mixture             two or more distinct, non-salt organic fragments
                     remain after salt removal -- this is a genuine
                     multi-component mixture/co-crystal, not a salt
metal_complex       the record contains a metal atom, either as part
                     of a coordination complex or as a metal ion
                     fragment (e.g. a Na+/K+/Ca2+ counterion). Metals
                     are NEVER auto-removed or auto-standardized --
                     see the DESIGN NOTES below
ambiguous_fragment  a non-metal fragment could not be confidently
                     classified as either a known salt/solvent or a
                     plausible organic parent
no_parent           every fragment was a recognized non-metal salt/
                     solvent -- nothing organic is left to be a parent
invalid_smiles      RDKit could not parse the string, or it was empty/
                     missing/NaN, or an unexpected error occurred
                     while processing this specific row (see
                     `error_detail`)
duplicate           this row's standardized structure (by InChIKey)
                     is shared with at least one other row. See
                     DUPLICATE HANDLING below for what happens to the
                     representative vs. the other members of the group

STEREOCENTER AUDITING
-----------------------
For every row that resolves to a single structure (`clean` or
`salt_removed`), `n_stereocenters` and `n_unassigned_stereocenters`
are recorded via RDKit's `FindMolChiralCenters(includeUnassigned=True)`.

This module does NOT strip, resolve, or guess at stereochemistry --
whatever stereo information is in the input SMILES is preserved as-is
in `SMILES`. The audit columns exist purely for visibility: for
endpoints like hERG and CYP3A4, an unassigned stereocenter means the
raw data genuinely does not tell you which enantiomer was tested, and
enantiomers can have materially different activity. That is a
modeling-stage decision (e.g. whether to exclude, flag, or treat such
rows specially), not something this cleaning stage should decide
silently.

For rows that do NOT resolve to a single structure (`mixture`,
`metal_complex`, `ambiguous_fragment`, `no_parent`, `invalid_smiles`,
and rows later relabeled `duplicate`), both fields are `None` --
deliberately distinct from `0`. `None` means "not applicable, no
single resolved structure to check"; `0` means "checked, and there
are no stereocenters." Don't conflate the two downstream.

DUPLICATE HANDLING
-------------------
Tautomers: grouped via tautomer_group_id (audit only), NOT automatically deduplicated.
Duplicates remain based on full InChIKey (including stereo, isotopes), so R/S and E/Z isomers are preserved as distinct rows.

Duplicate detection runs AFTER structural cleaning and is based on
InChIKey, never on label values -- labels only ever influence which
duplicate rows are auto-collapsible, never how a molecule is
standardized (see LEAKAGE PROTECTION).

* Rows are only compared for duplication if their cleaning status is
  `clean` or `salt_removed` (i.e. an unambiguous parent structure
  exists). Rows sent to review are never merged with anything.
* If an optional `--target-col` is supplied and every non-missing
  value for a duplicate group agrees, one representative row (the
  lowest original row index) keeps its `clean`/`salt_removed` status
  and is retained in the clean dataset. The other rows in the group
  are relabeled `duplicate` and moved to the review dataset purely
  for traceability (they are not lost, just not double-counted as
  independent clean training rows).
* If `--target-col` values conflict within a group, or no
  `--target-col` was supplied at all (so agreement cannot be
  verified), EVERY row in that group is relabeled `duplicate` and
  routed to review. Conflicting observations are never silently
  dropped or averaged.
* `duplicate_group_id` is assigned to every row in every duplicate
  group (representative included) for traceability, and left blank
  for structures that occur only once.

LEAKAGE PROTECTION
--------------------
The chemical-cleaning logic (`clean_smiles()`) receives nothing but a
single SMILES string -- by construction it cannot see, and therefore
cannot be influenced by, any label or metadata column. The following
columns, if present in the input file, are NEVER read by any part of
the standardization/classification logic, and are never used to
decide whether a molecule is cleaned, removed, or retained:

    Y, Overall, TA98, TA100, TA102, TA1535, TA1537, Id, source, Partition

The only place a label column can matter at all is the OPTIONAL
`--target-col` duplicate-agreement check described above, which only
ever decides whether a *structural* duplicate is auto-collapsible; it
never changes how any individual structure is standardized.

DESIGN NOTES FOR REVIEWERS
-----------------------------
* V4 change: added stereocenter auditing (see STEREOCENTER AUDITING
  above). This does not change any classification/removal behavior --
  it is purely additional reporting on rows that already resolve
  successfully.
* V3 change: standalone organic acids that are common salt counterions
  in pharma but also plausible standalone toxicology test compounds
  (acetic, citric, succinic, fumaric, tartaric, formic acid, TFA) are
  NEVER auto-classified as removable salts, even though RDKit's own
  default SaltRemover table recognizes all of them as salts. They are
  always treated as an organic parent candidate instead -- see
  PROTECTED_ORGANIC_ACID_SMILES. A bare acetic-acid row stays `clean`;
  an acetic-acid fragment alongside another organic fragment now
  produces `mixture` rather than being auto-resolved to
  `salt_removed`.
* "Salt/solvent" removal draws first from RDKit's own maintained
  SMARTS table (`rdkit.Chem.SaltRemover`, backed by RDKit's
  `Salts.txt`) plus a small supplementary literal list for common
  organic solvents and inorganic acids/bases not covered by that
  table. Both lists are filtered/curated to exclude anything
  containing a metal atom -- see METAL_SYMBOLS.
* Metals are handled completely separately and are NEVER folded into
  the auto-removable salt category, even for extremely common cases
  like a plain Na+ or K+ counterion. This is a deliberate, spec-driven
  choice: "do not automatically remove metals or metal-containing
  fragments." Any record containing a metal atom is always routed to
  `metal_complex` for manual review, whether that metal is a simple
  ionic counterion or part of a true coordination complex. This
  cleaner does not attempt to distinguish those two cases, since doing
  so reliably from SMILES alone (without 3D/bonding metadata) is
  itself a guess.
* Fragment matching against the salt/solvent tables is whole-fragment
  matching only -- a match must cover every atom in the fragment, so
  a chlorine atom that is part of a larger organic molecule is never
  mistaken for a chloride counterion.
* Organic-solvent recognition (ethanol, DMSO, THF, ...) is only
  applied when at least one other fragment is present to be the
  actual parent -- a bare "CCO" row on its own is a valid standalone
  compound, not an error, and is treated as `clean`.
* When multiple non-salt, non-metal fragments remain, this script
  never falls back to "keep the largest fragment." That heuristic
  silently discards real data in edge cases (a co-crystal of two
  actives, or an assay reported as a mixture). It is always routed to
  `mixture` for human review. The one deliberate exception: if the
  remaining fragments are all structurally IDENTICAL (e.g. "X.X"),
  there is nothing to guess -- it is folded into `clean`/
  `salt_removed` as that single structure.
* This module never crashes on a single malformed row; every row is
  processed inside its own error boundary.
* This module never modifies or overwrites the input file.

USAGE
-----
    python clean_smiles.py --input data/raw.csv --output data/clean.csv
    python clean_smiles.py --input data/raw.csv --output data/clean.csv \
        --smiles-col smiles --report data/clean_report.json
    python clean_smiles.py --input data/raw.csv --output data/clean.csv \
        --target-col Y
    python clean_smiles.py --self-test

Run `python clean_smiles.py --help` for the full option list.

Outputs (for --output path/to/foo_clean.csv):
    path/to/foo_clean.csv     structures that passed cleaning
    path/to/foo_review.csv    every record needing manual review/
                               exclusion, with a reason
    path/to/foo_report.json   machine-readable audit summary
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import SaltRemover

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger("clean_smiles")

STATUS_CLEAN = "clean"
STATUS_SALT_REMOVED = "salt_removed"
STATUS_MIXTURE = "mixture"
STATUS_METAL_COMPLEX = "metal_complex"
STATUS_AMBIGUOUS_FRAGMENT = "ambiguous_fragment"
STATUS_NO_PARENT = "no_parent"
STATUS_INVALID_SMILES = "invalid_smiles"
STATUS_DUPLICATE = "duplicate"

ALL_STATUSES = [
    STATUS_CLEAN, STATUS_SALT_REMOVED, STATUS_MIXTURE, STATUS_METAL_COMPLEX,
    STATUS_AMBIGUOUS_FRAGMENT, STATUS_NO_PARENT, STATUS_INVALID_SMILES, STATUS_DUPLICATE,
]
_CLEAN_ELIGIBLE_STATUSES = {STATUS_CLEAN, STATUS_SALT_REMOVED}
LEAKAGE_COLUMNS = frozenset({
    "Y", "Overall", "TA98", "TA100", "TA102", "TA1535", "TA1537",
    "Id", "source", "Partition",
})
METAL_SYMBOLS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu",
})
def _contains_metal(fragment: Chem.Mol) -> bool:
    return any(atom.GetSymbol() in METAL_SYMBOLS for atom in fragment.GetAtoms())
def _has_carbon(fragment: Chem.Mol) -> bool:
    return any(atom.GetSymbol() == "C" for atom in fragment.GetAtoms())
SUPPLEMENTARY_SOLVENT_SMILES = frozenset({
    "CO", "CCO", "CC(C)O", "CC(C)=O", "CS(C)=O", "O=CN(C)C", "C1CCOC1",
})
SUPPLEMENTARY_SAFE_SALT_SMILES = frozenset({
    "O", "[NH4+]", "Cl", "[Cl-]", "Br", "[Br-]", "I", "[I-]", "[OH-]",
    "OS(=O)(=O)O", "OP(=O)(O)O", "O=[N+]([O-])O", "OC(=O)O", "CS(=O)(=O)O",
    "Cc1ccc(cc1)S(=O)(=O)O", "OC(=O)C(=O)O",
})
PROTECTED_ORGANIC_ACID_SMILES = frozenset({
    "CC(=O)O", "CC(=O)[O-]", "OC(=O)CC(O)(CC(=O)O)C(=O)O", "OC(=O)CCC(=O)O",
    "OC(=O)/C=C/C(=O)O", "OC(=O)C=CC(=O)O", "OC(=O)C(O)C(O)C(=O)O", "OC=O", "[O-]C=O",
    "OC(=O)C(F)(F)F", "[O-]C(=O)C(F)(F)F",
})
def _mols_from_smiles_set(smiles_set: frozenset[str]) -> dict[str, Chem.Mol]:
    out = {}
    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Reference SMILES failed to parse and was skipped: %r", smi)
            continue
        out[smi] = mol
    return out
def _load_metal_free_salt_patterns() -> list[Chem.Mol]:
    remover = SaltRemover.SaltRemover()
    protected_mols = list(_mols_from_smiles_set(PROTECTED_ORGANIC_ACID_SMILES).values())
    safe_patterns = []
    for pattern in remover.salts:
        try:
            if any(atom.GetSymbol() in METAL_SYMBOLS for atom in pattern.GetAtoms()):
                continue
            protects_an_acid = False
            for protected in protected_mols:
                match = protected.GetSubstructMatch(pattern)
                if match and len(match) == protected.GetNumAtoms():
                    protects_an_acid = True
                    break
            if protects_an_acid:
                continue
        except Exception as exc:
            logger.debug("Could not inspect salt pattern, excluding it: %s", exc)
            continue
        safe_patterns.append(pattern)
    return safe_patterns
_METAL_FREE_SALT_PATTERNS: list[Chem.Mol] = _load_metal_free_salt_patterns()
_SOLVENT_MOLS: dict[str, Chem.Mol] = _mols_from_smiles_set(SUPPLEMENTARY_SOLVENT_SMILES)
_SAFE_SALT_MOLS: dict[str, Chem.Mol] = _mols_from_smiles_set(SUPPLEMENTARY_SAFE_SALT_SMILES)
_PROTECTED_ACID_MOLS: dict[str, Chem.Mol] = _mols_from_smiles_set(PROTECTED_ORGANIC_ACID_SMILES)
_SOLVENT_CANONICAL: frozenset[str] = frozenset(Chem.MolToSmiles(m, canonical=True) for m in _SOLVENT_MOLS.values())
_SAFE_SALT_CANONICAL: frozenset[str] = frozenset(Chem.MolToSmiles(m, canonical=True) for m in _SAFE_SALT_MOLS.values())
_PROTECTED_ACID_CANONICAL: frozenset[str] = frozenset(Chem.MolToSmiles(m, canonical=True) for m in _PROTECTED_ACID_MOLS.values())
def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)
def _stereo_counts(mol: Chem.Mol) -> tuple[int, int]:
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    n_total = len(centers)
    n_unassigned = sum(1 for _, label in centers if label == "?")
    return n_total, n_unassigned


def _canonical_isomeric(mol: Chem.Mol) -> str:
    """Isomeric SMILES - preserves R/S, E/Z. Audit only."""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def _ez_counts(mol: Chem.Mol) -> tuple[int, int]:
    """
    Production-simple E/Z audit - fully RDKit, no heuristics.
    Assigned: BondStereo == E or Z
    Unassigned: BondStereo == ANY, or bond is potential stereo but STEREONONE
    """
    n_assigned = 0
    n_unassigned = 0
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True, flagPossibleStereoCenters=True)
    except Exception:
        pass

    # RDKit potential stereo bonds
    potential = set()
    try:
        from rdkit.Chem import FindPotentialStereo
        for info in FindPotentialStereo(mol):
            try:
                # info.type is StereoType, info.centeredOn is bond idx for double bonds
                if "Bond" in str(info.type):
                    potential.add(int(info.centeredOn))
            except Exception:
                pass
    except Exception:
        pass

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        st = bond.GetStereo()
        if st == Chem.BondStereo.STEREOE or st == Chem.BondStereo.STEREOZ:
            n_assigned += 1
        elif st == Chem.BondStereo.STEREOANY:
            n_unassigned += 1
        elif bond.GetIdx() in potential and st == Chem.BondStereo.STEREONONE:
            # stereogenic but unspecified
            n_unassigned += 1
    return n_assigned, n_unassigned

def _isotope_audit(mol: Chem.Mol):
    labs = [f"{a.GetIsotope()}{a.GetSymbol()}_{a.GetIdx()}" for a in mol.GetAtoms() if a.GetIsotope() != 0]
    return (len(labs) > 0), len(labs), ";".join(labs)

_tautomer_enumerator = None
def _get_tautomer_enumerator():
    global _tautomer_enumerator
    if _tautomer_enumerator is not None:
        return _tautomer_enumerator
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        return _tautomer_enumerator
    except Exception:
        return None

def _tautomer_audit(mol: Chem.Mol):
    try:
        enum = _get_tautomer_enumerator()
        if enum is None:
            return False, None, None
        try:
            has = len(list(enum.Enumerate(mol))) > 1
        except Exception:
            has = False
        can = enum.Canonicalize(mol)
        can_smi = Chem.MolToSmiles(can, canonical=True)
        try:
            ik = Chem.MolToInchiKey(can)
            first = ik.split("-")[0] if ik else can_smi
        except Exception:
            first = can_smi
        return has, can_smi, first
    except Exception:
        return False, None, None

def _stereo_signature(mol: Chem.Mol) -> str:
    try:
        iso = _canonical_isomeric(mol)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        c_str = ";".join([f"{i}:{lab}" for i, lab in sorted(centers)]) if centers else "none"
        ez_parts = []
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic():
                st = b.GetStereo()
                if st == Chem.BondStereo.STEREOE:
                    ez_parts.append(f"{b.GetIdx()}:E")
                elif st == Chem.BondStereo.STEREOZ:
                    ez_parts.append(f"{b.GetIdx()}:Z")
                elif st == Chem.BondStereo.STEREOANY:
                    ez_parts.append(f"{b.GetIdx()}:ANY")
        ez_str = ";".join(ez_parts) if ez_parts else "none"
        return f"{iso} | CHIRAL[{c_str}] | EZ[{ez_str}]"
    except Exception:
        try:
            return _canonical_isomeric(mol)
        except Exception:
            return ""


def _matches_smarts_table(fragment: Chem.Mol) -> bool:
    n_atoms = fragment.GetNumAtoms()
    for pattern in _METAL_FREE_SALT_PATTERNS:
        match = fragment.GetSubstructMatch(pattern)
        if match and len(match) == n_atoms:
            return True
    return False
def _categorize_fragment(fragment: Chem.Mol, has_siblings: bool) -> str:
    if _contains_metal(fragment):
        return "metal"
    frag_canonical = _canonical(fragment)
    if frag_canonical in _PROTECTED_ACID_CANONICAL:
        return "parent"
    if frag_canonical in _SAFE_SALT_CANONICAL or _matches_smarts_table(fragment):
        return "removable"
    if has_siblings and frag_canonical in _SOLVENT_CANONICAL:
        return "removable"
    if _has_carbon(fragment):
        return "parent"
    return "ambiguous"

@dataclasses.dataclass
class CleaningResult:
    smiles_clean: Optional[str]
    status: str
    num_fragments: int
    num_removed: int
    removed_fragments: str
    retained_fragments: str
    inchikey: Optional[str]
    review_reason: str
    error_detail: str = ""
    n_stereocenters: Optional[int] = None
    n_unassigned_stereocenters: Optional[int] = None
    # === AUDIT ONLY (isomers preserved, never removed) ===
    n_ez_stereocenters: Optional[int] = None
    n_assigned_ez: Optional[int] = None
    n_unassigned_ez: Optional[int] = None
    has_chiral: Optional[bool] = None
    has_ez: Optional[bool] = None
    has_stereo: Optional[bool] = None
    has_unassigned_stereo: Optional[bool] = None
    is_stereoisomer_defined: Optional[bool] = None
    has_isotope: Optional[bool] = None
    n_isotope_atoms: Optional[int] = None
    isotope_labels: Optional[str] = None
    has_tautomer: Optional[bool] = None
    tautomer_canonical_smiles: Optional[str] = None
    tautomer_canonical_firstblock: Optional[str] = None
    stereo_signature: Optional[str] = None
    SMILES_isomeric: Optional[str] = None
    inchikey_no_stereo: Optional[str] = None
    tautomer_group_id: Optional[int] = None

def _safe_inchikey(mol: Chem.Mol) -> Optional[str]:
    try:
        key = Chem.MolToInchiKey(mol)
        return key or None
    except Exception as exc:
        logger.debug("InChIKey generation failed: %s", exc)
        return None

def clean_smiles(smiles: object) -> CleaningResult:
    try:
        if pd.isna(smiles):
            return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None, "missing SMILES value")
        smiles = str(smiles).strip()
        if not smiles:
            return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None, "missing SMILES value")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None, "RDKit could not parse this SMILES string")
        fragments = list(Chem.GetMolFrags(mol, asMols=True))
        num_fragments = len(fragments)
        has_siblings = num_fragments > 1
        categories = [_categorize_fragment(f, has_siblings) for f in fragments]
        metals = [f for f, c in zip(fragments, categories) if c == "metal"]
        removable = [f for f, c in zip(fragments, categories) if c == "removable"]
        parents = [f for f, c in zip(fragments, categories) if c == "parent"]
        ambiguous = [f for f, c in zip(fragments, categories) if c == "ambiguous"]
        removed_str = ".".join(_canonical(f) for f in removable)
        num_removed = len(removable)
        if metals:
            retained_str = ".".join(_canonical(f) for f in (parents + ambiguous + metals))
            return CleaningResult(None, STATUS_METAL_COMPLEX, num_fragments, num_removed, removed_str, retained_str, None, "contains a metal atom (as a simple ion or part of a coordination complex); metals are never automatically removed or standardized and require manual review")
        if ambiguous:
            retained_str = ".".join(_canonical(f) for f in (parents + ambiguous))
            return CleaningResult(None, STATUS_AMBIGUOUS_FRAGMENT, num_fragments, num_removed, removed_str, retained_str, None, "one or more fragments could not be confidently classified as a known salt/solvent or a plausible organic parent")
        if not parents:
            return CleaningResult(None, STATUS_NO_PARENT, num_fragments, num_removed, removed_str, "", None, "no organic parent fragment remained after removing recognized salts/solvents")
        if len(parents) == 1:
            cleaned = _canonical(parents[0])
            status = STATUS_CLEAN if num_removed == 0 else STATUS_SALT_REMOVED
            n_stereo, n_unassigned = _stereo_counts(parents[0])
            # --- AUDIT ONLY, preserve isomer ---
            try:
                n_a_ez, n_u_ez = _ez_counts(parents[0])
                n_ez = n_a_ez + n_u_ez
                has_chiral = n_stereo > 0
                has_ez = n_ez > 0
                has_stereo = has_chiral or has_ez
                has_unassigned = (n_unassigned > 0 or n_u_ez > 0)
                is_defined = has_stereo and not has_unassigned
                has_iso, n_iso, iso_labs = _isotope_audit(parents[0])
                has_tau, tau_smi, tau_first = _tautomer_audit(parents[0])
                sig = _stereo_signature(parents[0])
                iso_smi = _canonical_isomeric(parents[0])
                ik = _safe_inchikey(parents[0])
                ik_first = ik.split("-")[0] if ik else None
            except Exception:
                n_ez = n_a_ez = n_u_ez = 0
                has_chiral = has_ez = has_stereo = has_unassigned = is_defined = False
                has_iso = False; n_iso = 0; iso_labs = ""; has_tau = False; tau_smi = None; tau_first = None; sig = None; iso_smi = None; ik_first = None
                ik = _safe_inchikey(parents[0])
            return CleaningResult(cleaned, status, num_fragments, num_removed, removed_str, "", ik, "", n_stereocenters=n_stereo, n_unassigned_stereocenters=n_unassigned, n_ez_stereocenters=n_ez, n_assigned_ez=n_a_ez, n_unassigned_ez=n_u_ez, has_chiral=has_chiral, has_ez=has_ez, has_stereo=has_stereo, has_unassigned_stereo=has_unassigned, is_stereoisomer_defined=is_defined, has_isotope=has_iso, n_isotope_atoms=n_iso, isotope_labels=iso_labs, has_tautomer=has_tau, tautomer_canonical_smiles=tau_smi, tautomer_canonical_firstblock=tau_first, stereo_signature=sig, SMILES_isomeric=iso_smi, inchikey_no_stereo=ik_first)
        canon_parents = [_canonical(p) for p in parents]
        if len(set(canon_parents)) == 1:
            cleaned = canon_parents[0]
            status = STATUS_CLEAN if num_removed == 0 else STATUS_SALT_REMOVED
            n_stereo, n_unassigned = _stereo_counts(parents[0])
            try:
                n_a_ez, n_u_ez = _ez_counts(parents[0])
                n_ez = n_a_ez + n_u_ez
                has_chiral = n_stereo > 0
                has_ez = n_ez > 0
                has_stereo = has_chiral or has_ez
                has_unassigned = (n_unassigned > 0 or n_u_ez > 0)
                is_defined = has_stereo and not has_unassigned
                has_iso, n_iso, iso_labs = _isotope_audit(parents[0])
                has_tau, tau_smi, tau_first = _tautomer_audit(parents[0])
                sig = _stereo_signature(parents[0])
                iso_smi = _canonical_isomeric(parents[0])
                ik = _safe_inchikey(parents[0])
                ik_first = ik.split("-")[0] if ik else None
            except Exception:
                n_ez = n_a_ez = n_u_ez = 0
                has_chiral = has_ez = has_stereo = has_unassigned = is_defined = False
                has_iso = False; n_iso = 0; iso_labs = ""; has_tau = False; tau_smi = None; tau_first = None; sig = None; iso_smi = None; ik_first = None
                ik = _safe_inchikey(parents[0])
            return CleaningResult(cleaned, status, num_fragments, num_removed, removed_str, "", ik, "", n_stereocenters=n_stereo, n_unassigned_stereocenters=n_unassigned, n_ez_stereocenters=n_ez, n_assigned_ez=n_a_ez, n_unassigned_ez=n_u_ez, has_chiral=has_chiral, has_ez=has_ez, has_stereo=has_stereo, has_unassigned_stereo=has_unassigned, is_stereoisomer_defined=is_defined, has_isotope=has_iso, n_isotope_atoms=n_iso, isotope_labels=iso_labs, has_tautomer=has_tau, tautomer_canonical_smiles=tau_smi, tautomer_canonical_firstblock=tau_first, stereo_signature=sig, SMILES_isomeric=iso_smi, inchikey_no_stereo=ik_first)
        retained_str = ".".join(canon_parents)
        return CleaningResult(None, STATUS_MIXTURE, num_fragments, num_removed, removed_str, retained_str, None, "multiple distinct, non-salt organic components remain; cannot determine which is biologically relevant without guessing")
    except Exception as exc:
        logger.warning("Unexpected error cleaning %r: %s", smiles, exc)
        return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None, "unexpected processing error", str(exc))

def _resolve_duplicates(df: pd.DataFrame, status_col: str, inchikey_col: str, target_col: Optional[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["duplicate_group_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    eligible = df[status_col].isin(_CLEAN_ELIGIBLE_STATUSES) & df[inchikey_col].notna()
    if not eligible.any():
        return df, {"duplicate_groups": 0, "duplicate_rows_flagged": 0, "conflicting_duplicate_groups": 0, "conflicting_duplicate_rows": 0}
    groups: dict[str, list[int]] = {}
    for idx in df.index[eligible]:
        key = df.at[idx, inchikey_col]
        groups.setdefault(key, []).append(idx)
    duplicate_keys = sorted((key for key, idxs in groups.items() if len(idxs) > 1), key=lambda k: min(groups[k]))
    n_dup_groups = 0; n_dup_rows_flagged = 0; n_conflict_groups = 0; n_conflict_rows = 0
    for group_id, key in enumerate(duplicate_keys):
        member_idx = sorted(groups[key])
        df.loc[member_idx, "duplicate_group_id"] = group_id
        n_dup_groups += 1
        conflict = None
        if target_col is not None and target_col in df.columns:
            values = df.loc[member_idx, target_col].dropna().unique()
            conflict = len(values) > 1
        unverifiable = target_col is None or target_col not in df.columns
        if conflict or unverifiable:
            for idx in member_idx:
                df.at[idx, status_col] = STATUS_DUPLICATE
                if conflict:
                    df.at[idx, "review_reason"] = f"duplicate structure (InChIKey group {group_id}) with conflicting values in '{target_col}'; manual review required"
                else:
                    df.at[idx, "review_reason"] = f"duplicate structure (InChIKey group {group_id}); no --target-col supplied, so agreement could not be verified"
            n_dup_rows_flagged += len(member_idx)
            if conflict:
                n_conflict_groups += 1; n_conflict_rows += len(member_idx)
        else:
            representative, *others = member_idx
            for idx in others:
                df.at[idx, status_col] = STATUS_DUPLICATE
                df.at[idx, "review_reason"] = f"duplicate of row {representative} (InChIKey group {group_id}); same value in '{target_col}' -- representative row retained in the clean dataset"
            n_dup_rows_flagged += len(others)
    return df, {"duplicate_groups": n_dup_groups, "duplicate_rows_flagged": n_dup_rows_flagged, "conflicting_duplicate_groups": n_conflict_groups, "conflicting_duplicate_rows": n_conflict_rows}

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except Exception:
        return None
def clean_dataset(input_path: str, output_path: str, smiles_col: str = "SMILES", report_path: Optional[str] = None, review_path: Optional[str] = None, target_col: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_path); output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    if report_path is None:
        report_path = output_path.with_name(stem + "_report.json")
    else:
        report_path = Path(report_path); report_path.parent.mkdir(parents=True, exist_ok=True)
    if review_path is None:
        review_stem = stem[:-6] + "_review" if stem.endswith("_clean") else stem + "_review"
        review_path = output_path.with_name(review_stem + output_path.suffix)
    else:
        review_path = Path(review_path); review_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path); total_rows = len(df)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {input_path}. Available columns: {list(df.columns)}")
    if target_col is not None and target_col not in df.columns:
        logger.warning("--target-col '%s' not found in input; duplicate groups will be conservatively routed to review instead of auto-collapsed.", target_col)
    df["SMILES_raw"] = df[smiles_col]
    logger.info("Cleaning %d rows...", total_rows)
    results = df["SMILES_raw"].apply(clean_smiles)
    df["SMILES"] = results.apply(lambda r: r.smiles_clean)
    df["cleaning_status"] = results.apply(lambda r: r.status)
    df["inchikey"] = results.apply(lambda r: r.inchikey)
    df["num_fragments"] = results.apply(lambda r: r.num_fragments)
    df["num_removed"] = results.apply(lambda r: r.num_removed)
    df["removed_fragments"] = results.apply(lambda r: r.removed_fragments)
    df["retained_fragments"] = results.apply(lambda r: r.retained_fragments)
    df["review_reason"] = results.apply(lambda r: r.review_reason)
    df["error_detail"] = results.apply(lambda r: r.error_detail)
    df["n_stereocenters"] = results.apply(lambda r: r.n_stereocenters)
    df["n_unassigned_stereocenters"] = results.apply(lambda r: r.n_unassigned_stereocenters)
    # --- AUDIT ONLY (single implementation) ---
    df["n_ez_stereocenters"] = results.apply(lambda r: getattr(r, "n_ez_stereocenters", None))
    df["n_assigned_ez"] = results.apply(lambda r: getattr(r, "n_assigned_ez", None))
    df["n_unassigned_ez"] = results.apply(lambda r: getattr(r, "n_unassigned_ez", None))
    df["has_chiral"] = results.apply(lambda r: getattr(r, "has_chiral", None))
    df["has_ez"] = results.apply(lambda r: getattr(r, "has_ez", None))
    df["has_unassigned_stereo"] = results.apply(lambda r: getattr(r, "has_unassigned_stereo", None))
    df["has_isotope"] = results.apply(lambda r: getattr(r, "has_isotope", None))
    df["n_isotope_atoms"] = results.apply(lambda r: getattr(r, "n_isotope_atoms", None))
    df["has_tautomer"] = results.apply(lambda r: getattr(r, "has_tautomer", None))
    # traceability only - NOT model features
    df["stereo_signature"] = results.apply(lambda r: getattr(r, "stereo_signature", None))
    df["SMILES_isomeric"] = results.apply(lambda r: getattr(r, "SMILES_isomeric", None))
    df["tautomer_canonical_smiles"] = results.apply(lambda r: getattr(r, "tautomer_canonical_smiles", None))
    df["tautomer_canonical_firstblock"] = results.apply(lambda r: getattr(r, "tautomer_canonical_firstblock", None))
    df["inchikey_no_stereo"] = results.apply(lambda r: getattr(r, "inchikey_no_stereo", None))
    pre_dup_status_counts = df["cleaning_status"].value_counts().to_dict()
    df, dup_stats = _resolve_duplicates(df, status_col="cleaning_status", inchikey_col="inchikey", target_col=target_col)
    # --- tautomer grouping (audit only): grouped, NOT deduplicated ---
    # Tautomers share tautomer_group_id but keep distinct InChIKeys -> remain separate rows in clean set
    df["tautomer_group_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    try:
        eligible_tau = df["cleaning_status"].isin(_CLEAN_ELIGIBLE_STATUSES) & df["tautomer_canonical_firstblock"].notna()
        if eligible_tau.any():
            tau_map = {}; gid = 0
            for idx in df.index[eligible_tau]:
                key = df.at[idx, "tautomer_canonical_firstblock"]
                if key not in tau_map:
                    tau_map[key] = gid; gid += 1
                df.at[idx, "tautomer_group_id"] = tau_map[key]
    except Exception:
        pass
    final_status_counts = {s: int((df["cleaning_status"] == s).sum()) for s in ALL_STATUSES}
    clean_mask = df["cleaning_status"].isin(_CLEAN_ELIGIBLE_STATUSES)
    clean_df = df[clean_mask].copy(); review_df = df[~clean_mask].copy()
    assert len(clean_df) + len(review_df) == total_rows
    n_invalid = final_status_counts[STATUS_INVALID_SMILES]
    n_valid_smiles = total_rows - n_invalid
    n_unassigned_stereo_rows = int((df["n_unassigned_stereocenters"].fillna(0) > 0).sum())
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rdkit_version": rdBase.rdkitVersion,
        "script_git_commit": _git_commit_hash(),
        "input_file": str(input_path),
        "input_file_sha256": _file_sha256(input_path),
        "clean_output_file": str(output_path),
        "review_output_file": str(review_path),
        "smiles_column": smiles_col,
        "target_column": target_col,
        "total_records": total_rows,
        "valid_smiles": n_valid_smiles,
        "invalid_smiles": n_invalid,
        "clean_structures": final_status_counts[STATUS_CLEAN],
        "salts_removed": final_status_counts[STATUS_SALT_REMOVED],
        "mixtures": final_status_counts[STATUS_MIXTURE],
        "metal_complexes": final_status_counts[STATUS_METAL_COMPLEX],
        "ambiguous_structures": final_status_counts[STATUS_AMBIGUOUS_FRAGMENT],
        "no_parent_structures": final_status_counts[STATUS_NO_PARENT],
        "duplicate_groups": dup_stats["duplicate_groups"],
        "duplicates_flagged": dup_stats["duplicate_rows_flagged"],
        "conflicting_duplicate_groups": dup_stats["conflicting_duplicate_groups"],
        "conflicting_duplicate_rows": dup_stats["conflicting_duplicate_rows"],
        "rows_with_unassigned_stereocenters": n_unassigned_stereo_rows,
        "records_retained": len(clean_df),
        "records_excluded": len(review_df),
        "pre_duplicate_status_counts": pre_dup_status_counts,
        "final_status_counts": final_status_counts,
    }
    logger.info("=" * 70)
    logger.info("FILE: %s", input_path)
    logger.info("=" * 70)
    logger.info("Total records:      %d", total_rows)
    for status_name in ALL_STATUSES:
        logger.info("%-22s %d", status_name + ":", final_status_counts[status_name])
    logger.info("Retained (clean):   %d", len(clean_df))
    logger.info("Excluded (review):  %d", len(review_df))
    logger.info("Duplicate groups: %d (%d rows flagged, %d conflicting groups / %d conflicting rows)", dup_stats["duplicate_groups"], dup_stats["duplicate_rows_flagged"], dup_stats["conflicting_duplicate_groups"], dup_stats["conflicting_duplicate_rows"])
    logger.info("Rows with unassigned stereocenters (among resolved structures): %d", n_unassigned_stereo_rows)
    if final_status_counts[STATUS_MIXTURE]: logger.warning("%d rows are unresolved mixtures.", final_status_counts[STATUS_MIXTURE])
    if final_status_counts[STATUS_METAL_COMPLEX]: logger.warning("%d rows contain a metal and require manual review.", final_status_counts[STATUS_METAL_COMPLEX])
    if final_status_counts[STATUS_AMBIGUOUS_FRAGMENT]: logger.warning("%d rows have an ambiguous fragment.", final_status_counts[STATUS_AMBIGUOUS_FRAGMENT])
    if final_status_counts[STATUS_INVALID_SMILES]: logger.warning("%d rows had invalid/missing SMILES.", final_status_counts[STATUS_INVALID_SMILES])
    if n_unassigned_stereo_rows: logger.warning("%d clean/salt_removed rows have at least one unassigned stereocenter -- the raw data does not specify which enantiomer. Worth reviewing before hERG/CYP3A4 modeling.", n_unassigned_stereo_rows)
    clean_df.to_csv(output_path, index=False)
    review_df.to_csv(review_path, index=False)
    with open(report_path, "w") as fh: json.dump(summary, fh, indent=2)
    logger.info("Saved clean dataset:  %s (%d rows)", output_path, len(clean_df))
    logger.info("Saved review dataset: %s (%d rows)", review_path, len(review_df))
    logger.info("Saved audit report:   %s", report_path)
    logger.info("=" * 70)
    return clean_df, review_df

def _run_fragment_self_tests() -> list[str]:
    cases = [
        ("c1ccccc1", STATUS_CLEAN, "c1ccccc1"),
        ("CC(=O)Oc1ccccc1C(=O)O", STATUS_CLEAN, None),
        ("CCN.Cl", STATUS_SALT_REMOVED, "CCN"),
        ("CCN.CCO", STATUS_SALT_REMOVED, "CCN"),
        ("O.O.CCN", STATUS_SALT_REMOVED, "CCN"),
        ("CC(=O)O", STATUS_CLEAN, None),
        ("OC(=O)CC(O)(CC(=O)O)C(=O)O", STATUS_CLEAN, None),
        ("OC(=O)CCC(=O)O", STATUS_CLEAN, None),
        ("OC(=O)C=CC(=O)O", STATUS_CLEAN, None),
        ("OC(=O)C(O)C(O)C(=O)O", STATUS_CLEAN, None),
        ("OC=O", STATUS_CLEAN, None),
        ("OC(=O)C(F)(F)F", STATUS_CLEAN, None),
        ("CCN.CC(=O)O", STATUS_MIXTURE, None),
        ("CCN.CS(=O)(=O)O", STATUS_SALT_REMOVED, "CCN"),
        ("CCN.OC(=O)C(=O)O", STATUS_SALT_REMOVED, "CCN"),
        ("CC[NH3+].[Cl-]", STATUS_SALT_REMOVED, "CC[NH3+]"),
        ("CCO", STATUS_CLEAN, "CCO"),
        ("CCN.c1ccccc1O", STATUS_MIXTURE, None),
        ("[Na+].CC(=O)[O-]", STATUS_METAL_COMPLEX, None),
        ("[Na+].[Cl-]", STATUS_METAL_COMPLEX, None),
        ("[Fe+2].CC(=O)[O-].CC(=O)[O-]", STATUS_METAL_COMPLEX, None),
        ("OS(=O)(=O)O", STATUS_NO_PARENT, None),
        ("[NH4+]", STATUS_NO_PARENT, None),
        ("O", STATUS_NO_PARENT, None),
        ("not_real_smiles!!", STATUS_INVALID_SMILES, None),
        ("", STATUS_INVALID_SMILES, None),
        (None, STATUS_INVALID_SMILES, None),
        ("CCN.[He]", STATUS_AMBIGUOUS_FRAGMENT, None),
        ("N#N", STATUS_AMBIGUOUS_FRAGMENT, None),
        ("CCN.CCN", STATUS_CLEAN, "CCN"),
        ("CCN.CCN.Cl", STATUS_SALT_REMOVED, "CCN"),
    ]
    failures = []
    for smi, expected_status, expected_smiles in cases:
        result = clean_smiles(smi)
        ok_status = result.status == expected_status
        ok_smiles = expected_smiles is None or result.smiles_clean == expected_smiles
        if not (ok_status and ok_smiles):
            failures.append(f"  [fragment] input={smi!r}\n    expected: status={expected_status!r} smiles={expected_smiles!r}\n    actual:   status={result.status!r} smiles={result.smiles_clean!r}")
    return failures
def _run_stereocenter_self_tests() -> list[str]:
    failures = []
    cases = [("C[C@H](N)C(=O)O", 1, 0), ("CC(N)C(=O)O", 1, 1), ("CCO", 0, 0)]
    for smi, expected_total, expected_unassigned in cases:
        result = clean_smiles(smi)
        if not (result.n_stereocenters == expected_total and result.n_unassigned_stereocenters == expected_unassigned):
            failures.append(f"  [stereo] input={smi!r}\n    expected: n_stereocenters={expected_total} n_unassigned={expected_unassigned}\n    actual:   n_stereocenters={result.n_stereocenters!r} n_unassigned={result.n_unassigned_stereocenters!r}")
    mixture_result = clean_smiles("CCN.c1ccccc1O")
    if not (mixture_result.n_stereocenters is None and mixture_result.n_unassigned_stereocenters is None):
        failures.append(f"  [stereo] mixture row should report None/None for stereocenter fields, got n_stereocenters={mixture_result.n_stereocenters!r} n_unassigned={mixture_result.n_unassigned_stereocenters!r}")
    return failures
def _run_duplicate_self_tests() -> list[str]:
    failures = []
    df = pd.DataFrame({"SMILES_raw": ["CCO", "OCC"], "Y": [1, 1]})
    results = df["SMILES_raw"].apply(clean_smiles)
    df["cleaning_status"] = results.apply(lambda r: r.status)
    df["inchikey"] = results.apply(lambda r: r.inchikey)
    df["review_reason"] = ""
    resolved, stats = _resolve_duplicates(df, "cleaning_status", "inchikey", target_col="Y")
    if not (resolved.loc[0, "cleaning_status"] == STATUS_CLEAN and resolved.loc[1, "cleaning_status"] == STATUS_DUPLICATE and stats["duplicate_groups"] == 1 and stats["conflicting_duplicate_groups"] == 0):
        failures.append("  [duplicate] same-target duplicate did not collapse as expected")
    df2 = pd.DataFrame({"SMILES_raw": ["CCO", "OCC"], "Y": [1, 0]})
    results2 = df2["SMILES_raw"].apply(clean_smiles)
    df2["cleaning_status"] = results2.apply(lambda r: r.status)
    df2["inchikey"] = results2.apply(lambda r: r.inchikey)
    df2["review_reason"] = ""
    resolved2, stats2 = _resolve_duplicates(df2, "cleaning_status", "inchikey", target_col="Y")
    if not ((resolved2["cleaning_status"] == STATUS_DUPLICATE).all() and stats2["conflicting_duplicate_groups"] == 1 and stats2["conflicting_duplicate_rows"] == 2):
        failures.append("  [duplicate] conflicting-target duplicate was not fully flagged")
    df3 = pd.DataFrame({"SMILES_raw": ["CCO", "OCC"]})
    results3 = df3["SMILES_raw"].apply(clean_smiles)
    df3["cleaning_status"] = results3.apply(lambda r: r.status)
    df3["inchikey"] = results3.apply(lambda r: r.inchikey)
    df3["review_reason"] = ""
    resolved3, stats3 = _resolve_duplicates(df3, "cleaning_status", "inchikey", target_col=None)
    if not (resolved3["cleaning_status"] == STATUS_DUPLICATE).all():
        failures.append("  [duplicate] no-target-col duplicate was not conservatively flagged")
    df4 = pd.DataFrame({"SMILES_raw": ["CCO", "CCN"], "Y": [1, 0]})
    results4 = df4["SMILES_raw"].apply(clean_smiles)
    df4["cleaning_status"] = results4.apply(lambda r: r.status)
    df4["inchikey"] = results4.apply(lambda r: r.inchikey)
    df4["review_reason"] = ""
    resolved4, stats4 = _resolve_duplicates(df4, "cleaning_status", "inchikey", target_col="Y")
    if not (resolved4["duplicate_group_id"].isna().all() and stats4["duplicate_groups"] == 0):
        failures.append("  [duplicate] unique structures were incorrectly grouped")
    return failures

def _run_new_audit_self_tests() -> list[str]:
    """Production regression: E/Z, R/S, isotope, tautomer - isomers preserved."""
    failures = []
    # Fully RDKit-based E/Z - no heuristic fallback
    for smi, exp_a, exp_u in [("C/C=C/C",1,0), ("CC=CC",0,1), ("CC=C",0,0), ("C/C=C\\C",1,0)]:
        r = clean_smiles(smi)
        if r.n_assigned_ez != exp_a or r.n_unassigned_ez != exp_u:
            failures.append(f"  [ez] {smi!r} exp a={exp_a} u={exp_u} got a={r.n_assigned_ez} u={r.n_unassigned_ez}")
    # R/S preserved - full InChIKey includes stereo
    df_rs = pd.DataFrame({"SMILES_raw":["C[C@H](O)CC","C[C@@H](O)CC"],"Y":[1,1]})
    res = df_rs["SMILES_raw"].apply(clean_smiles)
    df_rs["cleaning_status"]=res.apply(lambda x:x.status); df_rs["inchikey"]=res.apply(lambda x:x.inchikey)
    df_rs["review_reason"]=""
    resolved,_=_resolve_duplicates(df_rs,"cleaning_status","inchikey",target_col="Y")
    if (resolved["cleaning_status"] != STATUS_CLEAN).any():
        failures.append("  [duplicate] R/S incorrectly collapsed - should preserve enantiomers")
    # E/Z preserved
    df_ez = pd.DataFrame({"SMILES_raw":["C/C=C/C","C/C=C\\C"],"Y":[1,1]})
    res_ez = df_ez["SMILES_raw"].apply(clean_smiles)
    df_ez["cleaning_status"]=res_ez.apply(lambda x:x.status); df_ez["inchikey"]=res_ez.apply(lambda x:x.inchikey)
    df_ez["review_reason"]=""
    resolved_ez,_=_resolve_duplicates(df_ez,"cleaning_status","inchikey",target_col="Y")
    if (resolved_ez["cleaning_status"] != STATUS_CLEAN).any():
        failures.append("  [duplicate] E/Z incorrectly collapsed")
    # isotope
    r_iso = clean_smiles("[13CH4]")
    if not getattr(r_iso, 'has_isotope', False) or r_iso.status != STATUS_CLEAN:
        failures.append("  [isotope] [13CH4] should be clean")
    # tautomer - audit only, grouped not deduped
    r_tau1 = clean_smiles("CC(=O)CC(=O)C")
    r_tau2 = clean_smiles("CC(O)=CC(=O)C")
    if getattr(r_tau1, 'has_tautomer', False) is False:
        failures.append("  [tautomer] acetylacetone should flag has_tautomer")
    # if both have same first block, they should get same group id (when in dataset) but remain distinct rows
    if getattr(r_tau1, 'tautomer_canonical_firstblock', None) != getattr(r_tau2, 'tautomer_canonical_firstblock', None):
        failures.append("  [tautomer] keto/enol should share first block")
    return failures


def run_self_test() -> bool:
    failures = (_run_fragment_self_tests() + _run_stereocenter_self_tests() + _run_duplicate_self_tests() + _run_new_audit_self_tests())
    if failures:
        print("SELF-TEST FAILURES:\n" + "\n".join(failures))
        return False
    print("ALL SELF-TESTS PASSED")
    return True
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative SMILES cleaning for toxicity ML. If run without --input/--output, automatically cleans all endpoints in data/ .")
    parser.add_argument("--input", "-i", default=None, help="Path to input CSV (optional - if omitted, batch mode cleans all endpoints in data/).")
    parser.add_argument("--output", "-o", default=None, help="Path to write clean CSV (optional).")
    parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name (default: SMILES).")
    parser.add_argument("--report", default=None, help="Path to write JSON audit report.")
    parser.add_argument("--review-output", default=None, help="Path to write review CSV.")
    parser.add_argument("--target-col", default=None, help="Label column for duplicate agreement check (optional).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests and exit.")
    return parser

# Default endpoints - batch mode (original behavior you had)
DEFAULT_ENDPOINTS = [
    ("data/Ames_raw.csv", "data/Ames_clean.csv", "Overall"),
    ("data/CYP3A4_raw.csv", "data/CYP3A4_clean.csv", "Y"),
    ("data/hERG_raw.csv", "data/hERG_clean.csv", "Y"),
    ("data/DILI_raw.csv", "data/DILI_clean.csv", "Y"),
    ("data/Teratogenicity_raw.csv", "data/Teratogenicity_clean.csv", "Y"),
]

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.self_test:
        return 0 if run_self_test() else 1

    # CLI mode: if --input and --output supplied, clean single file
    if args.input and args.output:
        clean_dataset(input_path=args.input, output_path=args.output, smiles_col=args.smiles_col, report_path=args.report, review_path=args.review_output, target_col=args.target_col)
        return 0

    # CLI mode: only --input supplied without --output -> error with helpful message
    if args.input and not args.output:
        parser.error("--output is required when --input is supplied. Or run without arguments to clean all endpoints in data/.")

    # AUTO-RUN BATCH MODE (original behavior): no args -> clean all endpoints
    print("\n=== AUTO-RUN BATCH MODE: cleaning all endpoints in data/ ===")
    for inp, out, tgt in DEFAULT_ENDPOINTS:
        if not Path(inp).exists():
            print(f"  Skipping {inp} (not found)")
            continue
        try:
            print(f"\n--- {inp} -> {out} (target_col={tgt}) ---")
            clean_dataset(input_path=inp, output_path=out, smiles_col=args.smiles_col, target_col=tgt)
        except Exception as e:
            print(f"  FAILED {inp}: {e}")
            import traceback
            traceback.print_exc()
    print("\n=== BATCH CLEANING DONE ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())