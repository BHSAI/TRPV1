# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 21:35:44 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
Visualize important Morgan bits as highlighted substructures.

- Input:
    * A CSV with a SMILES column (e.g., IC50 train or external test)
    * A list of important bit indices from SHAP (TOP_BITS)

- Output:
    * One SVG per (bit, example molecule) with the atoms contributing to that bit highlighted
    * A frequency report: how many molecules in the set contain each bit

Edit the CONFIG section to switch between IC50/EC50 and train/external.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors

# ────────── CONFIG ──────────

TASK = "EC50"  # or "EC50" (just for naming)
SET  = "external"  # "train" or "external"

CSV_PATH = {
    ("IC50", "train"):    "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv",
    ("IC50", "external"): "TRPV1_IC50_scaffold_split/TRPV1_IC50_exttest_scaffold.csv",
    ("EC50", "train"):    "TRPV1_EC50_scaffold_split/TRPV1_EC50_train_scaffold.csv",
    ("EC50", "external"): "TRPV1_EC50_scaffold_split/TRPV1_EC50_exttest_scaffold.csv",
}[(TASK, SET)]

SMILES_COL = "SMILES"

# >>> Replace this with the bit indices from your SHAP analysis <<<
TOP_BITS = [378, 926, 1816, 41, 1243, 875,310]  # example list for IC50

N_BITS   = 2048
RADIUS   = 2
EXAMPLES_PER_BIT = 3

OUT_DIR = Path(f"TRPV1_{TASK}_Morgan_SHAP_bits_{SET}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────── HELPERS ──────────

def get_atoms_for_bit(mol: Chem.Mol, bit_id: int,
                      radius: int = RADIUS,
                      nBits: int = N_BITS):
    """
    Given a molecule and a Morgan bit ID, return the atom indices
    associated with that bit using RDKit's bitInfo.
    """
    bitInfo = {}
    _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=nBits, bitInfo=bitInfo
    )
    if bit_id not in bitInfo:
        return []

    atoms = set()
    for center, rad in bitInfo[bit_id]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
        for bond_idx in env:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
    return list(atoms)


def draw_mol_with_highlight(mol: Chem.Mol,
                            highlight_atoms,
                            out_path: Path,
                            legend: str = ""):
    """
    Draw an SVG with highlighted atoms.
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=highlight_atoms, legend=legend
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    out_path.write_text(svg, encoding="utf-8")


# ────────── MAIN ──────────

def main():
    print(f"Loading {TASK} {SET} set from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if SMILES_COL not in df.columns:
        raise ValueError(f"Column '{SMILES_COL}' not found in {CSV_PATH}")

    smiles_list = df[SMILES_COL].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    # 1) Visual examples per bit
    for bit in TOP_BITS:
        count = 0
        print(f"\n=== Bit {bit} ===")
        for idx, mol in enumerate(mols):
            if mol is None:
                continue

            hl_atoms = get_atoms_for_bit(mol, bit)
            if not hl_atoms:
                continue

            out_file = OUT_DIR / f"bit{bit}_mol{count+1}_idx{idx}.svg"
            legend = f"Bit {bit} | mol idx {idx}"
            draw_mol_with_highlight(mol, hl_atoms, out_file, legend=legend)
            print(f"  Saved {out_file}")
            count += 1

            if count >= EXAMPLES_PER_BIT:
                break

        if count == 0:
            print(f"  [WARN] Bit {bit} not found in any molecule in this set.")

    # 2) Bit frequency check
    print("\n=== Bit frequency in this set ===")
    for bit in TOP_BITS:
        freq = 0
        for mol in mols:
            if mol is None:
                continue
            bitInfo = {}
            _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=RADIUS, nBits=N_BITS, bitInfo=bitInfo
            )
            if bit in bitInfo:
                freq += 1
        print(f"Bit {bit} appears in {freq} molecules "
              f"({freq / len(mols):.2%} of this {SET} set)")

    print(f"\nDone. SVGs are in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
