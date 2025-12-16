## IC50 preprocessing pipeline

1. `03_preprocess_SMILES_IC50.py`  
   - Input: `data/raw/TRPV1_IC50_Chembl_5-2025_raw.csv`  
   - Output: `data/intermediate/TRPV1_IC50_cleaned_RDK.csv`

2. `03b_duplicate_check_spyder_IC50.py`  
   - Input: `data/intermediate/TRPV1_IC50_cleaned_RDK.csv`  
   - Output: `data/intermediate/TRPV1_IC50_SMILES_cleaned_RDK.csv`

3. `03c_similarity_check_IC50.py`  
   - Input: `data/intermediate/TRPV1_IC50_cleaned_RDK.csv`  
   - Output: `data/intermediate/identical_pairs_IC50.csv`

4. `04_scaff_split_IC50.py`  
   - Input: `data/intermediate/TRPV1_IC50_SMILES_cleaned_RDK.csv`  
   - Output:  
     - `data/pre-processed/TRPV1_IC50_train_scaffold.csv`  
     - `data/pre-processed/TRPV1_IC50_exttest_scaffold.csv`  

## EC50 preprocessing pipeline

1. `2_03_preprocess_SMILES_EC50.py`  
   - Input: `data/raw/TRPV1_EC50_Chembl_5-2025_raw.csv`  
   - Output: `data/intermediate/TRPV1_EC50_cleaned_RDK.csv`

2. `2_03b_duplicate_check_spyder_EC50.py`  
   - Input: `data/intermediate/TRPV1_EC50_cleaned_RDK.csv`  
   - Output: `data/intermediate/TRPV1_EC50_SMILES_cleaned_RDK.csv`

3. `2_03c_similarity_check_EC50.py`  
   - Input: `data/intermediate/TRPV1_EC50_cleaned_RDK.csv`  
   - Output: `data/intermediate/identical_pairs_EC50.csv`

4. `2_04_scaff_split_EC50.py`  
   - Input: `data/intermediate/TRPV1_EC50_SMILES_cleaned_RDK.csv`  
   - Output:  
     - `data/pre-processed/TRPV1_EC50_train_scaffold.csv`  
     - `data/pre-processed/TRPV1_EC50_exttest_scaffold.csv`
