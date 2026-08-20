#!/usr/bin/env python3
"""
clean_smiles.py
================

Chemical standardization and quality-control stage for a multi-endpoint
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

# RDKit is chatty on stderr by default (parse warnings etc.). We
# capture failures ourselves per-row, so silence RDKit's own
# C++-level logger and rely on our own `logging` module instead.
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger("clean_smiles")


# =======================================================================
# STATUS CONSTANTS
# =======================================================================

STATUS_CLEAN = "clean"
STATUS_SALT_REMOVED = "salt_removed"
STATUS_MIXTURE = "mixture"
STATUS_METAL_COMPLEX = "metal_complex"
STATUS_AMBIGUOUS_FRAGMENT = "ambiguous_fragment"
STATUS_NO_PARENT = "no_parent"
STATUS_INVALID_SMILES = "invalid_smiles"
STATUS_DUPLICATE = "duplicate"

ALL_STATUSES = [
    STATUS_CLEAN,
    STATUS_SALT_REMOVED,
    STATUS_MIXTURE,
    STATUS_METAL_COMPLEX,
    STATUS_AMBIGUOUS_FRAGMENT,
    STATUS_NO_PARENT,
    STATUS_INVALID_SMILES,
    STATUS_DUPLICATE,
]

# Statuses that represent an unambiguous, single organic parent
# structure -- these are the only rows eligible for the clean dataset
# (before duplicate resolution potentially demotes some of them).
_CLEAN_ELIGIBLE_STATUSES = {STATUS_CLEAN, STATUS_SALT_REMOVED}

# Columns this script's cleaning/classification logic must NEVER read.
# Documented here for reviewers; the architecture (clean_smiles() only
# ever receives a bare string) is what actually enforces this, not
# this list -- see LEAKAGE PROTECTION in the module docstring.
LEAKAGE_COLUMNS = frozenset({
    "Y", "Overall", "TA98", "TA100", "TA102", "TA1535", "TA1537",
    "Id", "source", "Partition",
})


# =======================================================================
# METAL RECOGNITION
# =======================================================================

# Deliberately conservative: common metallic elements seen in drug
# salts and metal-based therapeutics/complexes. Metalloids (B, Si, Ge,
# As, Sb, Te) are intentionally EXCLUDED -- they behave chemically
# like organic-compatible elements in most drug structures (e.g.
# boronic acids) and are not what "do not auto-remove metals" is
# guarding against.
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
    """True if any atom in `fragment` is one of METAL_SYMBOLS."""
    return any(atom.GetSymbol() in METAL_SYMBOLS for atom in fragment.GetAtoms())


def _has_carbon(fragment: Chem.Mol) -> bool:
    return any(atom.GetSymbol() == "C" for atom in fragment.GetAtoms())


# =======================================================================
# SALT / SOLVENT RECOGNITION (non-metal only)
# =======================================================================

# Non-ionic solvents that legitimately co-crystallize with an API but
# are not covered by RDKit's ionic-salt SMARTS table. These are only
# treated as removable when at least one other fragment is present to
# be the actual parent -- see `_categorize_fragment`. None of these
# contain a metal atom.
SUPPLEMENTARY_SOLVENT_SMILES = frozenset({
    "CO",          # methanol
    "CCO",         # ethanol
    "CC(C)O",      # isopropanol
    "CC(C)=O",     # acetone
    "CS(C)=O",     # DMSO
    "O=CN(C)C",    # DMF
    "C1CCOC1",     # THF
})

# Simple, unambiguous, non-metal counterions/small inorganic species.
# Unlike the solvent list above, these are treated as removable
# regardless of whether other fragments are present -- a bare "Cl"
# (chloride) or bare "O" (water) row on its own has no organic parent
# and should be `no_parent`, not `clean`.
#
# Deliberately NOT included here: simple carboxylic acids that are
# also plausible standalone toxicology test compounds in their own
# right (acetic, citric, succinic, fumaric, tartaric, formic acid,
# TFA). Those are handled separately by PROTECTED_ORGANIC_ACID_SMILES
# below -- see that block for why.
SUPPLEMENTARY_SAFE_SALT_SMILES = frozenset({
    "O",                        # water
    "[NH4+]",                   # ammonium
    "Cl", "[Cl-]",               # chloride
    "Br", "[Br-]",               # bromide
    "I", "[I-]",                 # iodide
    "[OH-]",                     # hydroxide
    "OS(=O)(=O)O",                # sulfuric acid / sulfate (protonated)
    "OP(=O)(O)O",                 # phosphoric acid / phosphate (protonated)
    "O=[N+]([O-])O",              # nitric acid
    "OC(=O)O",                    # carbonic acid
    "CS(=O)(=O)O",                 # methanesulfonic acid (mesylate)
    "Cc1ccc(cc1)S(=O)(=O)O",        # p-toluenesulfonic acid (tosylate)
    "OC(=O)C(=O)O",                 # oxalic acid
})

# Simple organic (carboxylic) acids that are common salt-forming
# counterions in pharma BUT are also legitimate standalone assay
# compounds in a toxicology dataset -- e.g. acetic acid or TFA could
# themselves be a row someone is testing, not just a counterion on
# some other drug. Silently stripping them (as RDKit's own default
# SaltRemover table would do) risks turning a real test compound into
# `no_parent` or wrongly merging it away in a mixture.
#
# These are therefore NEVER classified as "removable", in either
# direction:
#   - standalone ("CC(=O)O" alone)          -> treated as `parent`,
#     status `clean`, not `no_parent`.
#   - alongside another organic fragment
#     ("CCN.CC(=O)O")                        -> treated as `parent`
#     too, so the row becomes `mixture` (two real candidate
#     compounds) rather than being auto-resolved to `salt_removed`.
#     This is a deliberate "never guess" choice: we no longer assume
#     the acid is just playing counterion.
#
# This requires actively suppressing RDKit's own built-in salt SMARTS
# table for these specific structures too (see PROTECTED_ORGANIC_ACID
# usage in `_categorize_fragment`), since that table recognizes all of
# these as salts by default.
PROTECTED_ORGANIC_ACID_SMILES = frozenset({
    "CC(=O)O", "CC(=O)[O-]",                        # acetic acid / acetate
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",                     # citric acid
    "OC(=O)CCC(=O)O",                                  # succinic acid
    "OC(=O)/C=C/C(=O)O", "OC(=O)C=CC(=O)O",             # fumaric acid
    "OC(=O)C(O)C(O)C(=O)O",                              # tartaric acid
    "OC=O", "[O-]C=O",                                    # formic acid / formate
    "OC(=O)C(F)(F)F", "[O-]C(=O)C(F)(F)F",                 # TFA / trifluoroacetate
})


def _load_metal_free_salt_patterns() -> list[Chem.Mol]:
    """
    Load RDKit's built-in, maintained salt/counterion SMARTS patterns
    (the same table used internally by `SaltRemover`), then drop any
    pattern that contains a metal atom OR that whole-fragment-matches
    one of the protected standalone organic acids.

    Dropping metal-containing patterns is what enforces "never
    automatically remove metals": RDKit's own table happily matches
    Na+, K+, Ca2+, Al3+, Zn2+, etc. as counterions, but this pipeline's
    whole-fragment salt check must never fire on those -- they are
    handled exclusively by the metal-complex path instead.

    Dropping the protected-acid patterns is what enforces "never
    automatically classify standalone organic acids (acetic, citric,
    succinic, fumaric, tartaric, formic, TFA, ...) as removable
    salts": RDKit's default table also recognizes all of these as
    salts, which is exactly the behavior we need to override so they
    fall through to ordinary `parent` classification instead.
    """
    remover = SaltRemover.SaltRemover()
    protected_mols = list(_mols_from_smiles_set(PROTECTED_ORGANIC_ACID_SMILES).values())
    safe_patterns = []
    for pattern in remover.salts:
        try:
            if any(atom.GetSymbol() in METAL_SYMBOLS for atom in pattern.GetAtoms()):
                continue
            # Exclude this SMARTS pattern if it would whole-fragment-
            # match one of the protected acids -- i.e. the acid (as an
            # actual molecule) is the substructure-match TARGET and
            # the salt pattern is the QUERY, mirroring exactly how
            # `_matches_smarts_table` matches real fragments below.
            protects_an_acid = False
            for protected in protected_mols:
                match = protected.GetSubstructMatch(pattern)
                if match and len(match) == protected.GetNumAtoms():
                    protects_an_acid = True
                    break
            if protects_an_acid:
                continue
        except Exception as exc:  # pragma: no cover - defensive
            # If we can't confidently tell this pattern is metal-free
            # and acid-free, exclude it rather than risk auto-removing
            # something it shouldn't.
            logger.debug("Could not inspect salt pattern, excluding it: %s", exc)
            continue
        safe_patterns.append(pattern)
    return safe_patterns


def _mols_from_smiles_set(smiles_set: frozenset[str]) -> dict[str, Chem.Mol]:
    out = {}
    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:  # pragma: no cover - defensive, catches typos
            logger.warning("Reference SMILES failed to parse and was skipped: %r", smi)
            continue
        out[smi] = mol
    return out


_METAL_FREE_SALT_PATTERNS: list[Chem.Mol] = _load_metal_free_salt_patterns()
_SOLVENT_MOLS: dict[str, Chem.Mol] = _mols_from_smiles_set(SUPPLEMENTARY_SOLVENT_SMILES)
_SAFE_SALT_MOLS: dict[str, Chem.Mol] = _mols_from_smiles_set(SUPPLEMENTARY_SAFE_SALT_SMILES)
_PROTECTED_ACID_MOLS: dict[str, Chem.Mol] = _mols_from_smiles_set(PROTECTED_ORGANIC_ACID_SMILES)
_SOLVENT_CANONICAL: frozenset[str] = frozenset(
    Chem.MolToSmiles(m, canonical=True) for m in _SOLVENT_MOLS.values()
)
_SAFE_SALT_CANONICAL: frozenset[str] = frozenset(
    Chem.MolToSmiles(m, canonical=True) for m in _SAFE_SALT_MOLS.values()
)
_PROTECTED_ACID_CANONICAL: frozenset[str] = frozenset(
    Chem.MolToSmiles(m, canonical=True) for m in _PROTECTED_ACID_MOLS.values()
)


def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def _stereo_counts(mol: Chem.Mol) -> tuple[int, int]:
    """
    Count stereocenters in a resolved molecule.

    Returns (n_stereocenters, n_unassigned_stereocenters). Uses
    includeUnassigned=True so a stereocenter present in the graph but
    NOT specified in the input SMILES is still counted -- that's
    exactly the case worth flagging (raw data didn't say which
    enantiomer). Stereochemistry itself is never modified here; this
    is reporting only.
    """
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    n_total = len(centers)
    n_unassigned = sum(1 for _, label in centers if label == "?")
    return n_total, n_unassigned


def _matches_smarts_table(fragment: Chem.Mol) -> bool:
    """Whole-fragment match against the metal-free RDKit salt SMARTS table."""
    n_atoms = fragment.GetNumAtoms()
    for pattern in _METAL_FREE_SALT_PATTERNS:
        match = fragment.GetSubstructMatch(pattern)
        if match and len(match) == n_atoms:
            return True
    return False


def _categorize_fragment(fragment: Chem.Mol, has_siblings: bool) -> str:
    """
    Classify a single fragment as one of:
      "metal"      contains a metal atom -- never auto-processed
      "removable"  a recognized non-metal salt/solvent
      "parent"     a plausible organic candidate parent
      "ambiguous"  none of the above -- unrecognized, not clearly
                   organic; must not be guessed at

    `has_siblings` controls whether the supplementary *solvent* list
    applies (solvents are only "not the parent" when something else
    is present to actually be the parent). The safe-salt list (water,
    chloride, ...) applies regardless of sibling fragments, since
    those are never plausible parents in a toxicology dataset.

    Protected organic acids (acetic, citric, succinic, fumaric,
    tartaric, formic, TFA) are checked BEFORE the salt/SMARTS checks
    and always classify as "parent", never "removable" -- they are
    common salt counterions in pharma, but also plausible standalone
    toxicology test compounds, so this pipeline never assumes they're
    "just" a counterion. See PROTECTED_ORGANIC_ACID_SMILES.
    """
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


# =======================================================================
# PER-ROW CLEANING
# =======================================================================

@dataclasses.dataclass
class CleaningResult:
    smiles_clean: Optional[str]
    status: str
    num_fragments: int
    num_removed: int
    removed_fragments: str   # canonical SMILES of removed salts/solvents, "."-joined
    retained_fragments: str  # canonical SMILES of fragments kept for review, "."-joined
    inchikey: Optional[str]
    review_reason: str
    error_detail: str = ""
    # Stereocenter audit -- populated ONLY for rows that resolve to a
    # single structure (`clean` / `salt_removed`). None (not 0) means
    # "no single resolved structure to check", distinct from an
    # actual count of 0. See STEREOCENTER AUDITING in the module
    # docstring.
    n_stereocenters: Optional[int] = None
    n_unassigned_stereocenters: Optional[int] = None


def _safe_inchikey(mol: Chem.Mol) -> Optional[str]:
    """InChIKey generation can fail for exotic structures; never raise."""
    try:
        key = Chem.MolToInchiKey(mol)
        return key or None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("InChIKey generation failed: %s", exc)
        return None


def clean_smiles(smiles: object) -> CleaningResult:
    """
    Clean and classify a single SMILES value.

    IMPORTANT (leakage protection): this function's only input is the
    SMILES value itself. It has no access to, and therefore cannot be
    influenced by, any label/metadata column -- see LEAKAGE PROTECTION
    in the module docstring.

    This function never raises: any unexpected RDKit exception is
    caught and reported as `invalid_smiles` (with the exception text
    in `error_detail`) so that one malformed row can never abort a
    batch job.
    """
    try:
        if pd.isna(smiles):
            return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None,
                                   "missing SMILES value")

        smiles = str(smiles).strip()
        if not smiles:
            return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None,
                                   "missing SMILES value")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CleaningResult(None, STATUS_INVALID_SMILES, 0, 0, "", "", None,
                                   "RDKit could not parse this SMILES string")

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

        # -- Metals are never auto-removed or auto-standardized, full --
        # -- stop, regardless of what else is in the molecule. --------
        if metals:
            retained_str = ".".join(_canonical(f) for f in (parents + ambiguous + metals))
            return CleaningResult(
                None, STATUS_METAL_COMPLEX, num_fragments, num_removed,
                removed_str, retained_str, None,
                "contains a metal atom (as a simple ion or part of a "
                "coordination complex); metals are never automatically "
                "removed or standardized and require manual review",
            )

        # -- A fragment we can't confidently place anywhere. Do not ---
        # -- guess whether it's a novel salt or a real parent. --------
        if ambiguous:
            retained_str = ".".join(_canonical(f) for f in (parents + ambiguous))
            return CleaningResult(
                None, STATUS_AMBIGUOUS_FRAGMENT, num_fragments, num_removed,
                removed_str, retained_str, None,
                "one or more fragments could not be confidently classified "
                "as a known salt/solvent or a plausible organic parent",
            )

        # -- Every fragment was a recognized non-metal salt/solvent. --
        if not parents:
            return CleaningResult(
                None, STATUS_NO_PARENT, num_fragments, num_removed,
                removed_str, "", None,
                "no organic parent fragment remained after removing "
                "recognized salts/solvents",
            )

        # -- Exactly one organic parent candidate: clean success. -----
        if len(parents) == 1:
            cleaned = _canonical(parents[0])
            status = STATUS_CLEAN if num_removed == 0 else STATUS_SALT_REMOVED
            n_stereo, n_unassigned = _stereo_counts(parents[0])
            return CleaningResult(
                cleaned, status, num_fragments, num_removed, removed_str, "",
                _safe_inchikey(parents[0]), "",
                n_stereocenters=n_stereo, n_unassigned_stereocenters=n_unassigned,
            )

        # -- Multiple parent candidates. If they are all structurally -
        # -- identical (e.g. "X.X"), this is not actually ambiguous -- -
        # -- resolve it to that single structure. ----------------------
        canon_parents = [_canonical(p) for p in parents]
        if len(set(canon_parents)) == 1:
            cleaned = canon_parents[0]
            status = STATUS_CLEAN if num_removed == 0 else STATUS_SALT_REMOVED
            n_stereo, n_unassigned = _stereo_counts(parents[0])
            return CleaningResult(
                cleaned, status, num_fragments, num_removed, removed_str, "",
                _safe_inchikey(parents[0]), "",
                n_stereocenters=n_stereo, n_unassigned_stereocenters=n_unassigned,
            )

        # -- Genuinely different organic components remain: a mixture -
        # -- or co-crystal. We deliberately do NOT guess which one is --
        # -- the biologically relevant molecule. ------------------------
        retained_str = ".".join(canon_parents)
        return CleaningResult(
            None, STATUS_MIXTURE, num_fragments, num_removed, removed_str,
            retained_str, None,
            "multiple distinct, non-salt organic components remain; "
            "cannot determine which is biologically relevant without "
            "guessing",
        )

    except Exception as exc:  # noqa: BLE001 - intentional catch-all
        logger.warning("Unexpected error cleaning %r: %s", smiles, exc)
        return CleaningResult(
            None, STATUS_INVALID_SMILES, 0, 0, "", "", None,
            "unexpected processing error", str(exc),
        )


# =======================================================================
# DUPLICATE RESOLUTION (structure-first, label-aware only for arbitration)
# =======================================================================

def _resolve_duplicates(
    df: pd.DataFrame,
    status_col: str,
    inchikey_col: str,
    target_col: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Given a dataframe that already has per-row cleaning status and
    inchikey columns populated, find structural duplicates and decide
    which rows stay eligible for the clean dataset.

    Returns the (possibly status-mutated) dataframe plus a stats dict.
    Mutates only `status_col`, `review_reason`, and a new
    `duplicate_group_id` column -- never touches `smiles_clean`/the
    standardized structure itself, and never re-runs any chemistry.

    `target_col` is used ONLY here, ONLY to decide whether a group of
    structural duplicates agrees well enough to auto-collapse. It is
    never passed to, or used by, `clean_smiles()`.
    """
    df = df.copy()
    df["duplicate_group_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    eligible = df[status_col].isin(_CLEAN_ELIGIBLE_STATUSES) & df[inchikey_col].notna()
    if not eligible.any():
        return df, {
            "duplicate_groups": 0, "duplicate_rows_flagged": 0,
            "conflicting_duplicate_groups": 0, "conflicting_duplicate_rows": 0,
        }

    # Deterministic group ordering: sort each InChIKey's member rows by
    # original row index, then order groups themselves by the index at
    # which they first appear. This avoids depending on pandas'
    # internal tie-breaking in value_counts()/groupby ordering, which
    # is not a documented stable contract across versions -- important
    # for a pipeline that promises reproducible, auditable output.
    groups: dict[str, list[int]] = {}
    for idx in df.index[eligible]:
        key = df.at[idx, inchikey_col]
        groups.setdefault(key, []).append(idx)

    duplicate_keys = sorted(
        (key for key, idxs in groups.items() if len(idxs) > 1),
        key=lambda k: min(groups[k]),
    )

    n_dup_groups = 0
    n_dup_rows_flagged = 0
    n_conflict_groups = 0
    n_conflict_rows = 0

    for group_id, key in enumerate(duplicate_keys):
        member_idx = sorted(groups[key])
        df.loc[member_idx, "duplicate_group_id"] = group_id
        n_dup_groups += 1

        conflict = None
        if target_col is not None and target_col in df.columns:
            values = df.loc[member_idx, target_col].dropna().unique()
            conflict = len(values) > 1
        # If target_col is None (or missing from the data), agreement
        # cannot be verified -- treat like a conflict: never guess
        # which row to keep. See DUPLICATE HANDLING in the docstring.
        unverifiable = target_col is None or target_col not in df.columns

        if conflict or unverifiable:
            for idx in member_idx:
                df.at[idx, status_col] = STATUS_DUPLICATE
                if conflict:
                    df.at[idx, "review_reason"] = (
                        f"duplicate structure (InChIKey group {group_id}) with "
                        f"conflicting values in '{target_col}'; manual review required"
                    )
                else:
                    df.at[idx, "review_reason"] = (
                        f"duplicate structure (InChIKey group {group_id}); no "
                        f"--target-col supplied, so agreement could not be "
                        f"verified"
                    )
            n_dup_rows_flagged += len(member_idx)
            if conflict:
                n_conflict_groups += 1
                n_conflict_rows += len(member_idx)
        else:
            representative, *others = member_idx  # lowest index kept
            for idx in others:
                df.at[idx, status_col] = STATUS_DUPLICATE
                df.at[idx, "review_reason"] = (
                    f"duplicate of row {representative} (InChIKey group "
                    f"{group_id}); same value in '{target_col}' -- "
                    f"representative row retained in the clean dataset"
                )
            n_dup_rows_flagged += len(others)

    return df, {
        "duplicate_groups": n_dup_groups,
        "duplicate_rows_flagged": n_dup_rows_flagged,
        "conflicting_duplicate_groups": n_conflict_groups,
        "conflicting_duplicate_rows": n_conflict_rows,
    }


# =======================================================================
# DATASET-LEVEL PROCESSING
# =======================================================================

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def clean_dataset(
    input_path: str,
    output_path: str,
    smiles_col: str = "SMILES",
    report_path: Optional[str] = None,
    review_path: Optional[str] = None,
    target_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean a full CSV dataset and write clean/review/report files to disk.

    Parameters
    ----------
    input_path : str
        Path to the original, untouched CSV file. Only ever read.
    output_path : str
        Destination for the clean CSV (e.g. "data/foo_clean.csv").
    smiles_col : str
        Name of the column containing SMILES strings in the input file.
    report_path : str, optional
        Where to write the JSON audit report. Defaults to
        `<output stem>_report.json` next to the output file.
    review_path : str, optional
        Where to write the review CSV. Defaults to
        `<output stem with "_clean" -> "_review">.csv`, or
        `<output stem>_review.csv` if "_clean" isn't in the stem.
    target_col : str, optional
        Name of a label column used ONLY to decide whether structural
        duplicates agree and can be auto-collapsed to one
        representative. Never used to decide how a molecule is
        cleaned. If omitted, duplicate groups are conservatively
        routed to review rather than guessed at.

    Returns
    -------
    (clean_df, review_df) : tuple of pandas.DataFrame
        The two dataframes that were written to `output_path` and
        `review_path`, respectively.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem
    if report_path is None:
        report_path = output_path.with_name(stem + "_report.json")
    else:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    if review_path is None:
        review_stem = stem[:-6] + "_review" if stem.endswith("_clean") else stem + "_review"
        review_path = output_path.with_name(review_stem + output_path.suffix)
    else:
        review_path = Path(review_path)
        review_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path)
    total_rows = len(df)

    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )
    if target_col is not None and target_col not in df.columns:
        logger.warning(
            "--target-col '%s' not found in input; duplicate groups will "
            "be conservatively routed to review instead of auto-collapsed.",
            target_col,
        )

    # Preserve the untouched original value before anything overwrites
    # `smiles_col` (relevant when smiles_col is already named "SMILES").
    df["SMILES_raw"] = df[smiles_col]

    logger.info("Cleaning %d rows...", total_rows)
    # NOTE: only the raw SMILES Series is handed to clean_smiles() --
    # no other column, and in particular no label column, is visible
    # to the chemistry-classification logic. See LEAKAGE PROTECTION.
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

    pre_dup_status_counts = df["cleaning_status"].value_counts().to_dict()

    df, dup_stats = _resolve_duplicates(
        df, status_col="cleaning_status", inchikey_col="inchikey", target_col=target_col,
    )

    final_status_counts = {s: int((df["cleaning_status"] == s).sum()) for s in ALL_STATUSES}

    clean_mask = df["cleaning_status"].isin(_CLEAN_ELIGIBLE_STATUSES)
    clean_df = df[clean_mask].copy()
    review_df = df[~clean_mask].copy()

    # No record may disappear silently.
    assert len(clean_df) + len(review_df) == total_rows, (
        "internal error: clean + review row counts do not sum to the "
        "original row count"
    )

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
    logger.info(
        "Duplicate groups: %d (%d rows flagged, %d conflicting groups / "
        "%d conflicting rows)",
        dup_stats["duplicate_groups"], dup_stats["duplicate_rows_flagged"],
        dup_stats["conflicting_duplicate_groups"], dup_stats["conflicting_duplicate_rows"],
    )
    logger.info(
        "Rows with unassigned stereocenters (among resolved structures): %d",
        n_unassigned_stereo_rows,
    )

    if final_status_counts[STATUS_MIXTURE]:
        logger.warning("%d rows are unresolved mixtures.", final_status_counts[STATUS_MIXTURE])
    if final_status_counts[STATUS_METAL_COMPLEX]:
        logger.warning(
            "%d rows contain a metal and require manual review.",
            final_status_counts[STATUS_METAL_COMPLEX],
        )
    if final_status_counts[STATUS_AMBIGUOUS_FRAGMENT]:
        logger.warning(
            "%d rows have an ambiguous fragment.",
            final_status_counts[STATUS_AMBIGUOUS_FRAGMENT],
        )
    if final_status_counts[STATUS_INVALID_SMILES]:
        logger.warning(
            "%d rows had invalid/missing SMILES.", final_status_counts[STATUS_INVALID_SMILES],
        )
    if n_unassigned_stereo_rows:
        logger.warning(
            "%d clean/salt_removed rows have at least one unassigned "
            "stereocenter -- the raw data does not specify which "
            "enantiomer. Worth reviewing before hERG/CYP3A4 modeling.",
            n_unassigned_stereo_rows,
        )

    clean_df.to_csv(output_path, index=False)
    review_df.to_csv(review_path, index=False)
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved clean dataset:  %s (%d rows)", output_path, len(clean_df))
    logger.info("Saved review dataset: %s (%d rows)", review_path, len(review_df))
    logger.info("Saved audit report:   %s", report_path)
    logger.info("=" * 70)

    return clean_df, review_df


# =======================================================================
# SELF TEST (run with --self-test; no external data file required)
# =======================================================================

def _run_fragment_self_tests() -> list[str]:
    """Per-row classification tests. Returns a list of failure strings."""
    cases = [
        # (input, expected_status, expected_smiles_or_None)  --------------
        # -- ordinary molecule -------------------------------------------
        ("c1ccccc1", STATUS_CLEAN, "c1ccccc1"),
        # aspirin: only assert status, not the exact canonical string --
        # RDKit's atom-traversal order for canonicalization is an
        # internal implementation detail, not something this test
        # should hard-code and risk breaking on an RDKit version bump.
        ("CC(=O)Oc1ccccc1C(=O)O", STATUS_CLEAN, None),
        # -- salts (non-metal counterion) ---------------------------------
        ("CCN.Cl", STATUS_SALT_REMOVED, "CCN"),
        ("CCN.CCO", STATUS_SALT_REMOVED, "CCN"),          # ethanol solvate
        ("O.O.CCN", STATUS_SALT_REMOVED, "CCN"),
        # -- protected standalone organic acids -----------------------------
        # These are common salt counterions in pharma, but the cleaner
        # must NOT assume that -- they may be the actual test compound.
        ("CC(=O)O", STATUS_CLEAN, None),                      # acetic acid alone
        ("OC(=O)CC(O)(CC(=O)O)C(=O)O", STATUS_CLEAN, None),    # citric acid alone
        ("OC(=O)CCC(=O)O", STATUS_CLEAN, None),                 # succinic acid alone
        ("OC(=O)C=CC(=O)O", STATUS_CLEAN, None),                  # fumaric acid alone
        ("OC(=O)C(O)C(O)C(=O)O", STATUS_CLEAN, None),               # tartaric acid alone
        ("OC=O", STATUS_CLEAN, None),                                 # formic acid alone
        ("OC(=O)C(F)(F)F", STATUS_CLEAN, None),                        # TFA alone
        # Paired with another organic fragment, a protected acid is now
        # treated as a second candidate parent (never guess that it's
        # "just" a counterion) -- this is a mixture, not a salt.
        ("CCN.CC(=O)O", STATUS_MIXTURE, None),
        # Sanity check that this change is scoped to the protected acid
        # list only -- ordinary sulfonic-acid salt formers (mesylate,
        # tosylate) and oxalic acid are unaffected and still auto-strip.
        ("CCN.CS(=O)(=O)O", STATUS_SALT_REMOVED, "CCN"),
        ("CCN.OC(=O)C(=O)O", STATUS_SALT_REMOVED, "CCN"),
        # -- protonated molecule + counterion ------------------------------
        ("CC[NH3+].[Cl-]", STATUS_SALT_REMOVED, "CC[NH3+]"),
        # -- pure solvent/standalone organic (no siblings -> stays clean) --
        ("CCO", STATUS_CLEAN, "CCO"),
        # -- mixtures -------------------------------------------------------
        ("CCN.c1ccccc1O", STATUS_MIXTURE, None),
        # -- metal complexes / metal-containing salts ------------------------
        ("[Na+].CC(=O)[O-]", STATUS_METAL_COMPLEX, None),      # sodium acetate
        ("[Na+].[Cl-]", STATUS_METAL_COMPLEX, None),            # NaCl
        ("[Fe+2].CC(=O)[O-].CC(=O)[O-]", STATUS_METAL_COMPLEX, None),  # ferrous acetate
        # -- inorganic compounds ---------------------------------------------
        ("OS(=O)(=O)O", STATUS_NO_PARENT, None),                # sulfuric acid alone
        ("[NH4+]", STATUS_NO_PARENT, None),
        ("O", STATUS_NO_PARENT, None),
        # -- invalid SMILES ----------------------------------------------------
        ("not_real_smiles!!", STATUS_INVALID_SMILES, None),
        ("", STATUS_INVALID_SMILES, None),
        (None, STATUS_INVALID_SMILES, None),
        # -- ambiguous fragments -------------------------------------------------
        ("CCN.[He]", STATUS_AMBIGUOUS_FRAGMENT, None),   # helium: not metal, not organic
        ("N#N", STATUS_AMBIGUOUS_FRAGMENT, None),         # nitrogen gas alone
        # -- identical repeated fragments resolve unambiguously -------------------
        ("CCN.CCN", STATUS_CLEAN, "CCN"),
        ("CCN.CCN.Cl", STATUS_SALT_REMOVED, "CCN"),
    ]

    failures = []
    for smi, expected_status, expected_smiles in cases:
        result = clean_smiles(smi)
        ok_status = result.status == expected_status
        ok_smiles = expected_smiles is None or result.smiles_clean == expected_smiles
        if not (ok_status and ok_smiles):
            failures.append(
                f"  [fragment] input={smi!r}\n"
                f"    expected: status={expected_status!r} smiles={expected_smiles!r}\n"
                f"    actual:   status={result.status!r} smiles={result.smiles_clean!r}"
            )
    return failures


def _run_stereocenter_self_tests() -> list[str]:
    """
    Stereocenter-audit tests. Checks that stereocenters are counted
    correctly and that unresolved rows report None (not 0), which is
    what downstream code should rely on to distinguish "achiral" from
    "not applicable".
    """
    failures = []
    cases = [
        # (input, expected_n_stereocenters, expected_n_unassigned)
        ("C[C@H](N)C(=O)O", 1, 0),   # defined stereocenter (L-alanine)
        ("CC(N)C(=O)O", 1, 1),        # same skeleton, undefined
        ("CCO", 0, 0),                 # achiral
    ]
    for smi, expected_total, expected_unassigned in cases:
        result = clean_smiles(smi)
        if not (
            result.n_stereocenters == expected_total
            and result.n_unassigned_stereocenters == expected_unassigned
        ):
            failures.append(
                f"  [stereo] input={smi!r}\n"
                f"    expected: n_stereocenters={expected_total} n_unassigned={expected_unassigned}\n"
                f"    actual:   n_stereocenters={result.n_stereocenters!r} "
                f"n_unassigned={result.n_unassigned_stereocenters!r}"
            )

    # A row that does NOT resolve to a single structure must report
    # None for both fields, never 0 -- 0 would falsely imply "checked,
    # achiral" when really nothing was checked at all.
    mixture_result = clean_smiles("CCN.c1ccccc1O")
    if not (
        mixture_result.n_stereocenters is None
        and mixture_result.n_unassigned_stereocenters is None
    ):
        failures.append(
            "  [stereo] mixture row should report None/None for stereocenter "
            f"fields, got n_stereocenters={mixture_result.n_stereocenters!r} "
            f"n_unassigned={mixture_result.n_unassigned_stereocenters!r}"
        )

    return failures


def _run_duplicate_self_tests() -> list[str]:
    """Dataset-level duplicate-resolution tests using an in-memory frame."""
    failures = []

    # Case 1: same structure (different SMILES spellings), same target
    # -> one representative retained, the other demoted to `duplicate`.
    df = pd.DataFrame({
        "SMILES_raw": ["CCO", "OCC"],  # both ethanol
        "Y": [1, 1],
    })
    results = df["SMILES_raw"].apply(clean_smiles)
    df["cleaning_status"] = results.apply(lambda r: r.status)
    df["inchikey"] = results.apply(lambda r: r.inchikey)
    df["review_reason"] = ""
    resolved, stats = _resolve_duplicates(df, "cleaning_status", "inchikey", target_col="Y")
    if not (
        resolved.loc[0, "cleaning_status"] == STATUS_CLEAN
        and resolved.loc[1, "cleaning_status"] == STATUS_DUPLICATE
        and stats["duplicate_groups"] == 1
        and stats["conflicting_duplicate_groups"] == 0
    ):
        failures.append("  [duplicate] same-target duplicate did not collapse as expected")

    # Case 2: same structure, conflicting target -> both flagged, neither
    # silently kept.
    df2 = pd.DataFrame({
        "SMILES_raw": ["CCO", "OCC"],
        "Y": [1, 0],
    })
    results2 = df2["SMILES_raw"].apply(clean_smiles)
    df2["cleaning_status"] = results2.apply(lambda r: r.status)
    df2["inchikey"] = results2.apply(lambda r: r.inchikey)
    df2["review_reason"] = ""
    resolved2, stats2 = _resolve_duplicates(df2, "cleaning_status", "inchikey", target_col="Y")
    if not (
        (resolved2["cleaning_status"] == STATUS_DUPLICATE).all()
        and stats2["conflicting_duplicate_groups"] == 1
        and stats2["conflicting_duplicate_rows"] == 2
    ):
        failures.append("  [duplicate] conflicting-target duplicate was not fully flagged")

    # Case 3: no target_col supplied -> conservatively flag both, since
    # agreement can't be verified (never guess).
    df3 = pd.DataFrame({"SMILES_raw": ["CCO", "OCC"]})
    results3 = df3["SMILES_raw"].apply(clean_smiles)
    df3["cleaning_status"] = results3.apply(lambda r: r.status)
    df3["inchikey"] = results3.apply(lambda r: r.inchikey)
    df3["review_reason"] = ""
    resolved3, stats3 = _resolve_duplicates(df3, "cleaning_status", "inchikey", target_col=None)
    if not (resolved3["cleaning_status"] == STATUS_DUPLICATE).all():
        failures.append("  [duplicate] no-target-col duplicate was not conservatively flagged")

    # Case 4: unique structures -> no duplicate_group_id assigned.
    df4 = pd.DataFrame({"SMILES_raw": ["CCO", "CCN"], "Y": [1, 0]})
    results4 = df4["SMILES_raw"].apply(clean_smiles)
    df4["cleaning_status"] = results4.apply(lambda r: r.status)
    df4["inchikey"] = results4.apply(lambda r: r.inchikey)
    df4["review_reason"] = ""
    resolved4, stats4 = _resolve_duplicates(df4, "cleaning_status", "inchikey", target_col="Y")
    if not (resolved4["duplicate_group_id"].isna().all() and stats4["duplicate_groups"] == 0):
        failures.append("  [duplicate] unique structures were incorrectly grouped")

    return failures


def run_self_test() -> bool:
    """
    Exercise the cleaner against fixed, known cases (per-row
    classification, stereocenter auditing, and dataset-level duplicate
    resolution) and assert the expected outcome for each. Intended as
    a quick, reviewable correctness check that requires no data file
    on disk.
    """
    failures = (
        _run_fragment_self_tests()
        + _run_stereocenter_self_tests()
        + _run_duplicate_self_tests()
    )

    if failures:
        print("SELF-TEST FAILURES:\n" + "\n".join(failures))
        return False

    print("ALL SELF-TESTS PASSED")
    return True


# =======================================================================
# CLI
# =======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, fragment-analyze, and conservatively "
                     "standardize SMILES data for a toxicology ML pipeline. "
                     "Produces <output>_clean.csv-style clean/review/report "
                     "files; never guesses at ambiguous structures.",
    )
    parser.add_argument("--input", "-i", help="Path to the input CSV file.")
    parser.add_argument("--output", "-o", help="Path to write the clean CSV.")
    parser.add_argument(
        "--smiles-col", default="SMILES",
        help="Name of the SMILES column in the input file (default: SMILES).",
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to write the JSON audit report "
             "(default: <output stem>_report.json).",
    )
    parser.add_argument(
        "--review-output", default=None,
        help="Path to write the review CSV "
             "(default: <output stem>_review.csv).",
    )
    parser.add_argument(
        "--target-col", default=None,
        help="Optional label column used ONLY to decide whether "
             "structural duplicates agree well enough to auto-collapse "
             "to one representative row. Never used to decide how a "
             "molecule is cleaned. If omitted, duplicate groups are "
             "conservatively routed to review.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the built-in correctness test suite and exit "
             "(no --input/--output required).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.self_test:
        return 0 if run_self_test() else 1

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --self-test is given.")

    clean_dataset(
        input_path=args.input,
        output_path=args.output,
        smiles_col=args.smiles_col,
        report_path=args.report,
        review_path=args.review_output,
        target_col=args.target_col,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

'''
python clean_smiles.py --input data\Ames_raw.csv --output data\Ames_clean.csv
python clean_smiles.py --input data\Teratogenicity_raw.csv --output data\Teratogenicity_clean.csv
python clean_smiles.py --input data\hERG_raw.csv --output data\hERG_clean.csv
python clean_smiles.py --input data\CYP3A4_raw.csv --output data\CYP3A4_clean.csv
python clean_smiles.py --input data\DILI_raw.csv --output data\DILI_clean.csv    
'''