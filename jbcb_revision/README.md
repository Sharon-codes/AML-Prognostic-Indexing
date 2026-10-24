# JBCB source-aware evaluation (24 October 2026)

This branch is an evidence-led revision of the AML-Prognostic-Indexing study. It preserves the original main branch and adds an executable analysis to reproduce new audit results from the recorded gene-selection histogram and cohort confusion counts.

## Run

```bash
python -m pip install numpy
python jbcb_revision/reanalysis.py --input jbcb_revision/input_snapshot.json --out jbcb_revision/recomputed.json
python -m unittest discover -s jbcb_revision -p 'test_reanalysis.py' -v
```

## Outputs

- Nogueira feature-selection stability over 30 runs and 1,000 candidates, with sensitivity to the 16,508-gene denominator
- Number of distinct selected genes and mean run-pair intersection
- 10,000-replicate uniform-selection negative-control benchmark with seed 20261024
- Precision, sensitivity, specificity and Wilson 95% profile-level bounds for three control-source contrasts

## Provenance

Source repository snapshot: `fd623b3de754e1cdd2ee9a199e8f693551fa2927`.

Original evidence files: `logs_and_output/bqpso_stability_core_signature.txt`, `logs_and_output/remediated_expanded_controls_validation.txt`, `logs_and_output/remediated_class_balance_rescue_results.txt`, and `logs_and_output/true_external_validation_results.txt`.

The data here are model-output summaries, not primary expression matrices or individual patient predictions. The Monte Carlo experiment is a fresh feature-set null comparison, not a de novo patient-level training exercise. The archived 30 selections predate later gene-panel remediation. All 151 TCGA cases are RNA-seq and the 51 controls in the first contrast are microarray; the source-only negative control demonstrates confounding in this evaluation design.

**No original figures, trained checkpoints or patient data are altered by this revision.**
