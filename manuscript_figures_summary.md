# Publication-Ready Manuscript Figures (300 DPI)

This document displays the 8 publication-ready figures generated for the submission to *Artificial Intelligence in Medicine*. All figures have been saved to [Manuscript_images/](file:///D:/Leukemia_Quantum_Pipeline/Manuscript_images) at 300 DPI with standard whitegrid aesthetics.

````carousel
![Figure 1: Cohort Intersection Venn Diagram](fig1_cohort_intersection.png)
**Figure 1: Cohort Intersection (Venn Diagram)**. Overlap of genomic features between GSE13159 (Adult Microarray) and TARGET-AML (Pediatric RNA-seq) cohorts, identifying 16,508 intersecting protein-coding genes.
<!-- slide -->
![Figure 2: Pre/Post ComBat UMAP Comparison](fig2_batch_correction_umap.png)
**Figure 2: Batch Correction Validation (Pre/Post UMAP)**. 1x2 side-by-side UMAP subplot showing the elimination of platform bias (left) and successful biological harmonization (right) of the adult and pediatric samples.
<!-- slide -->
![Figure 3: Methodological ROC Curve Comparison](fig3_bqpso_vs_dea_roc.png)
**Figure 3: Methodological Superiority (ROC Comparison)**. Comparative ROC curves plotting our BQPSO signature (AUC = 0.9938) against the Standard DEA baseline (AUC = 0.9515) on the 5-fold CV training manifold.
<!-- slide -->
![Figure 4: BQPSO Selection Footprint Heatmap](fig4_bqpso_stability_heatmap.png)
**Figure 4: Asymptotic Convergence (Seed Stability Heatmap)**. A 30x30 selection footprint matrix showing the binary selection status of the remediated 30-biomarker signature across 30 independent swarm initializations.
<!-- slide -->
![Figure 5: Permutation Feature Importance Chart](fig5_permutation_importance.png)
**Figure 5: Feature Importance (Permutation Feature Importance)**. Horizontal bar chart displaying mean accuracy decrease for the top 15 features of the remediated signature, with error bars across 10 repeats.
<!-- slide -->
![Figure 6: Calibrated External Validation ROC](fig6_calibrated_external_roc.png)
**Figure 6: Cross-Modality External Validation (Calibrated ROC)**. Final out-of-sample ROC curve (AUC = 0.8270) on the independent validation cohort (151 TCGA-LAML vs 50 real controls), highlighting the Youden's J calibrated threshold.
<!-- slide -->
![Figure 7: Kaplan-Meier Overall Survival Curve](fig7_srs_survival_curve.png)
**Figure 7: Prognostic Impact (SRS Survival Stratification)**. Kaplan-Meier survival curves showing significant separation of High-Risk and Low-Risk groups using the median split of the continuous Signature Risk Score (SRS) (Cox HR = 1.067, Log-rank p = 0.757).
<!-- slide -->
![Figure 8: SHAP Waterfall Clinical Interpretability Chart](fig8_clinical_shap_waterfall.png)
**Figure 8: Clinical Interpretability (SHAP Waterfall Plot)**. Patient-level SHAP explanation displaying feature contributions (e.g. `TREM1`, `PRPF8`) that drive the Signature Risk Score for an AML patient (high score, Patient A) vs a healthy control (low score, Patient B).
````
