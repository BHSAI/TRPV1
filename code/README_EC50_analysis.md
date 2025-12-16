# TRPV1 EC50 pipeline

This folder contains the full EC50 analysis pipeline, from
fingerprint/descriptor generation to statistical comparison and
interpretation.

## Pipeline order

1. **01_TRPV1_EC50_5x5CV_fingerprints.py**  
   5×5 repeated stratified CV for 7 models × 3 fingerprints
   (Morgan, RDKit, MACCS). Writes per-fold and summary CSVs.

2. **02_TRPV1_EC50_5x5CV_Mordred.py**  
   Computes Mordred descriptors, applies filtering and scaling,
   and runs 5×5 CV across the same 7 models.

3. **03_TRPV1_EC50_RM_ANOVA_Tukey.py**  
   Repeated-measures ANOVA + Tukey HSD for method comparison
   across models and fingerprints using MCC, G-mean, ROC-AUC,
   and PR-AUC.

4. **04a_TRPV1_EC50_5x5CV_dashboard.py**  
   Generates per-fingerprint dashboards (boxplots + adjusted
   p-value heatmaps) (Figure 6).

5. **04b_multimetric_heatmap.py**  
   Produces the multi-metric heatmap summarizing mean performance
   across all model–descriptor combinations (Figure 5).

6. **04c_allboxplots_4fps_7ML_supp.py**  
   Creates MCC and G-mean boxplots across all 7 models and 4
   molecular representations for Supplementary Figure S2.

7. **05a_TRPV1_EC50_champions_RM_ANOVA_Tukey.py**
   Compares top-models per-fingerprint as dashboards (boxplots + adjusted
   p-value heatmaps) (Figure 7).

8. **12_TRPV1_EC50_master_table_mean.py**  
   Assembles master mean-score tables across fingerprints and
   models for downstream visualization.

9. **13_TRPV1_EC50_MorganLR_SDC_AD.py**  
   Trains the final Morgan + Logistic Regression model, computes
   applicability domain using SDC, and summarizes performance
   inside vs outside AD on the external scaffold test set.

10. **14d_TRPV1_EC50_external_bar_plot.py**  
   Generates bar plots of external test performance metrics
   (MCC, G-mean, ROC-AUC, PR-AUC) used in Supplementary Figure S5.

11. **15_TRPV1_EC50_MorganLR_SHAP_external.py**  
    Computes SHAP values for the final Morgan + Logistic
    Regression model on the external test set and generates
    summary and bar plots (Figures 8 and 9).

12. **16_TRPV1_scaffold_EC50_SHAP_bit_visuals.py**  
    Visualizes selected Morgan bits as atom-highlighted scaffold
    depictions for interpretation of SHAP-identified fragments.
