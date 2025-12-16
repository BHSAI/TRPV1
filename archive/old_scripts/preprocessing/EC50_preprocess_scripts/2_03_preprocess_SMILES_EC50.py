"""
Preprocess TRPV1 IC50 data:
  * filter non‑organic structures
  * deep standardisation (cleanup, charge, tautomer, stereo, isotope)
  * deduplicate by InChIKey
Outputs CSV with two new cols:
  CANONICAL_SMILES, InChIKey
"""
import pandas as pd, logging, time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as rms
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

INPUT_CSV  = "./TRPV1_chembl_EC50_cleaned_v1.csv"
OUTPUT_CSV = "./TRPV1_EC50_cleaned_RDK.csv"

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
_CARBON    = Chem.MolFromSmarts("[#6]")
_WHITELIST = {"H","C","N","O","P","S","F","Cl","Br","I"}

# ---------- helpers ---------------------------------------------------------

def is_valid_mol(smiles):
    # --- new robust guards ----------------------------------------------
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if smiles == "" or smiles.lower() in {"nan", "na", "."}:
        return None
    # --------------------------------------------------------------------
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    """Return sanitised Mol or None if inorganic / unsupported element."""
    if not mol or not mol.HasSubstructMatch(_CARBON):
        return None
    if any(a.GetSymbol() not in _WHITELIST for a in mol.GetAtoms()):
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logging.debug(f"Sanitisation failed for {smiles}: {e}")
        return None

_TAUT_ENUM   = rms.TautomerEnumerator()
_UNCHARGER   = rms.Uncharger()
def standardize_mol(mol):
    """RDKit standardisation; return canonical isomeric SMILES or None."""
    try:
        # InChI round‑trip once
        mol = Chem.MolFromInchi(Chem.MolToInchi(mol))

        mol = rms.Cleanup(mol)                 # fix valence, arom, etc.
        mol = rms.FragmentParent(mol)          # get largest fragment
        mol = _UNCHARGER.uncharge(mol)         # neutralise
        mol = rms.ChargeParent(mol)            # ensure integral charge
        mol = rms.IsotopeParent(mol)           # strip isotopes
        mol = rms.StereoParent(mol)            # normalise stereo
        mol = _TAUT_ENUM.Canonicalize(mol)     # pick canonical tautomer
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        logging.debug(f"Standardise fail: {e}")
        return None

def to_inchikey(smiles):
    try:
        return InchiToInchiKey(MolToInchi(Chem.MolFromSmiles(smiles)))
    except Exception as e:
        logging.debug(f"InChIKey fail for {smiles}: {e}")
        return None

def process_smiles(smiles):
    mol = is_valid_mol(smiles)
    if not mol:
        return None, None
    canon = standardize_mol(mol)
    if not canon:
        return None, None
    return canon, to_inchikey(canon)

# ---------- main routine ----------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df):,} rows")

    # vectorised apply -> list of tuples (canon, ikey)
    canon_ikey = df["SMILES"].apply(process_smiles)
    df[["CANONICAL_SMILES", "InChIKey"]] = pd.DataFrame(canon_ikey.tolist(),
                                                        index=df.index)

    df = (df.dropna(subset=["InChIKey"])
            .drop_duplicates("InChIKey")
            .reset_index(drop=True))

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved {len(df):,} cleaned molecules → {OUTPUT_CSV} "
                 f"({time.time()-t0:.1f}s)")
