\# TRPV1 IC50 data processing



This document describes how the raw ChEMBL export was processed into the

final train/test sets used in the manuscript.



\## Inputs



\- `data/raw/TRPV1\_IC50\_Chembl\_5-2025\_raw.csv`



\## Pipeline (IC50)



1\. Standardization \& basic clean-up → `TRPV1\_IC50\_cleaned\_RDK.csv`

2\. Duplicate / near-duplicate checks (InChIKey + Tanimoto)  

3\. Murcko scaffold split (80/20) into:

&nbsp;  - `TRPV1\_IC50\_train\_scaffold.csv`

&nbsp;  - `TRPV1\_IC50\_exttest\_scaffold.csv`



See `code/preprocessing/\*.py` for exact implementations.



