# Cross-experiment transportability of student assignment noncompletion prediction in online learning

**Version 1.0**

Repository: https://github.com/ZidongZhou/cross-experiment-transportability

This repository contains the analysis code, aggregate result tables and figures for a study of 14-day recorded assignment noncompletion in the public ASSISTments multi-experiment archive. The analyses cover within-archive validation, leave-one-experiment-out (LOEO) evaluation, predictor comparisons, probability calibration, outcome-horizon sensitivity and fixed-capacity record selection.

## Data

The study uses the public ASSISTments multi-experiment archive distributed with *Exploring Common Trends in Online Educational Experiments Data* (OSF DOI: [10.17605/OSF.IO/59SHV](https://doi.org/10.17605/OSF.IO/59SHV)). Participant-level source data are not included in this repository.

Download `experiment_dataset_2021-09-23.zip` from the source repository and place it at `data/experiment_dataset_2021-09-23.zip`, or set `TKL_RAW_ZIP` to its absolute path. Use the inner experiment archive, rather than an outer download wrapper. Before analysis, the code verifies the downloaded archive against the expected SHA256 checksum recorded in `code/analysis_pipeline.py`.

The analysis table contains one assignment event per learner within an experiment. Prediction is made at the learner's recorded assignment start in that experiment. Predictors are restricted to pre-experiment learner history and static context available at that time.

## Installation

The pinned environment is specified in `environment.yml` (Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, SciPy 1.17.0, statsmodels 0.14.6, Matplotlib 3.10.8, joblib 1.5.3 and LightGBM 4.6.0).

```bash
conda env create -f environment.yml
conda activate transportability-analysis
```

## Reproduce the analyses

Run all commands from the repository root. Set `TKL_N_JOBS` according to available CPU cores and memory.

**macOS/Linux (bash):**

```bash
export TKL_RAW_ZIP=/absolute/path/to/experiment_dataset_2021-09-23.zip
export TKL_N_JOBS=4
```

**Windows (PowerShell):**

```powershell
$env:TKL_RAW_ZIP = "C:\path\to\experiment_dataset_2021-09-23.zip"
$env:TKL_N_JOBS = "4"
```

Run the four stages in this order:

```bash
python code/analysis_pipeline.py
python code/build_model_predictions.py
python code/additional_analyses.py
python code/make_figures.py
```

The stages have the following inputs and outputs:

1. **`analysis_pipeline.py`** verifies and reads the source archive, reconstructs the analysis cohort, fits the primary models and validation designs, and writes the primary validation tables, cohort summaries, calibration summaries, sensitivity analyses and `results/horizon_lightgbm.csv`. The 7-day, 30-day and archive-closure LightGBM sensitivity models use the same five-fold experiment-blocked, learner-purged design as the corresponding logistic-regression analyses.
2. **`build_model_predictions.py`** reconstructs the cohort and fits the additional out-of-fold prediction sets used for the LBFGS comparator, learner-history and context-ablation comparisons, and the alternate-seed LightGBM reference. It writes local intermediate prediction files under `analysis_work/` and aggregate cohort and predictor tables under `results/`. The builder replaces the preliminary prediction workfile written by the primary pipeline.
3. **`additional_analyses.py`** checks the local prediction-workfile provenance and computes paired experiment-level bootstrap intervals, validation-design contrasts, predictor comparisons, calibration benchmarks, capacity intervals, shortlist-stability analyses and `results/Supplementary_Table_S12.csv`.
4. **`make_figures.py`** reads the aggregate result tables and writes Figures 1–4 and Supplementary Figures S1–S2.

The record-level prediction workfile is generated locally and excluded from version control. It is not distributed in this repository. Its provenance record checks the source-data checksum, generating-code checksums, fixed model configuration, row keys and predicted probabilities. If the raw archive, prediction code, model configuration or prediction values change, rebuild the workfile by running `build_model_predictions.py` before `additional_analyses.py`. Keep the four stages in the order shown: rerunning `analysis_pipeline.py` after the prediction builder replaces the complete prediction workfile with the primary pipeline's preliminary predictions, so the builder must then be run again.

### Joint-purge analysis

The three aggregate tables for the joint learner-, class- and teacher-purged LOEO analysis are included in `results/`. Their SHA256 digests, together with the model-specification digest, are listed in `results/aggregate_output_checksums.sha256`. Before using these tables, the analysis code checks file integrity and verifies that their experiment coverage, sample sizes, event counts, eligibility flags and model labels match the reconstructed cohort. If a table is missing or fails validation, the joint-purge analysis is recomputed from the source data.

To explicitly recompute the joint-purge analysis, set the following variable before running `additional_analyses.py`:

**macOS/Linux (bash):**

```bash
TKL_RECOMPUTE_JOINT_PURGE=1 python code/additional_analyses.py
```

**Windows (PowerShell):**

```powershell
$env:TKL_RECOMPUTE_JOINT_PURGE = "1"
python code/additional_analyses.py
Remove-Item Env:TKL_RECOMPUTE_JOINT_PURGE
```

### Additional analyses

- **Validation-design dependence and transport:** `validation_gap_uncertainty.csv` reports paired experiment-level median LightGBM-minus-SGD logistic-regression AUROC gaps and 95% experiment-bootstrap intervals for all eight validation designs. `validation_gap_contrasts.csv` reports paired gap changes for every pair of designs. `validation_model_contrasts.csv` reports paired AUROC changes for both primary models across every pair of designs.
- **Uncertainty for paired differences:** `feature_ablation_uncertainty.csv` reports experiment-bootstrap intervals for learner-history versus full predictors, context ablation under row-random and learner-purged LOEO, and the between-design context-effect contrast. `capacity_difference_uncertainty.csv` reports experiment-cluster bootstrap intervals for precision and recall differences at every review capacity. Figures 1, 2 and 4 display the corresponding intervals.
- **Standard logistic-regression comparator:** `LogisticRegression(solver='lbfgs', C=1.0, max_iter=5000, tol=1e-4)` is evaluated under row-random and learner-purged LOEO. The associated result tables report discrimination, calibration and changes in the LightGBM-logistic gap.
- **Local-prevalence calibration reference:** `local_prevalence_reference_by_experiment.csv` assigns later records the event prevalence observed in the matured calibration cohort. `local_recalibration_comparison.csv` compares this reference with no update and local updating. Additional tables report calibration, waiting-period exclusion and evaluation counts, and experiments in which intercept updating worsened Brier score.
- **Shortlist-overlap reference:** `shortlist_overlap_summary.csv` reports overlap in the 80 AUROC-eligible experiments. The reference compares two LightGBM fits with different fixed random seeds; `shortlist_overlap_reference_contrasts.csv` reports paired differences from the cross-model overlap.
- **Analysis eligibility:** `Supplementary_Table_S12.csv` lists all 88 experiments, their sizes and prevalences, and eligibility and record counts for the AUROC, calibration and chronological local-recalibration analyses.

## Outputs

- **`results/`** contains aggregate tables used by the manuscript, supplementary material and figures. It does not contain row-level predictions or participant-level source records.
- **`figures/`** contains Figures 1–4 and Supplementary Figures S1–S2 in PDF, EPS, SVG and 600-DPI PNG formats.
- **`FIGURE_LEGENDS.md`** contains the figure titles and descriptions.
- **`analysis_work/`** is created locally for intermediate prediction files and is excluded from version control.

Key result files include:

| Analysis | Result files |
| --- | --- |
| Cohort eligibility | `experiment_cohort.csv`, `Supplementary_Table_S12.csv` |
| Validation performance and model gaps | `validation_performance.csv`, `experiment_level_performance.csv`, `validation_gap_uncertainty.csv`, `validation_gap_contrasts.csv`, `validation_model_contrasts.csv` |
| Predictor comparisons | `feature_blocks.csv`, `feature_blocks_loeo.csv`, `lightgbm_context_loeo.csv`, `feature_auc_summary.csv`, `feature_ablation_uncertainty.csv` |
| Standard logistic-regression comparator | `logistic_lbfgs_comparison.csv`, `logistic_lbfgs_gap_comparison.csv`, `logistic_lbfgs_calibration_slope.csv`, `logistic_lbfgs_calibration_experiment_level.csv` |
| Calibration and local updating | `calibration_summary.csv`, `local_recalibration.csv`, `local_recalibration_comparison.csv`, `local_recalibration_pairwise_contrasts.csv`, `local_prevalence_reference_by_experiment.csv`, `local_recalibration_worsening_summary.csv` |
| Capacity and shortlist stability | `capacity_model_uncertainty.csv`, `capacity_difference_uncertainty.csv`, `shortlist_overlap_summary.csv`, `shortlist_overlap_reference.csv`, `shortlist_overlap_reference_contrasts.csv` |
| Outcome-horizon sensitivity | `sensitivity_summary.csv`, `horizon_lightgbm.csv` |

`Supplementary_Table_S12.csv` lists all 88 experiments and indicates eligibility for experiment-specific AUROC, calibration and chronological local recalibration analyses. For the 69 local-recalibration experiments, it also reports the calibration, waiting-period exclusion and evaluation sample sizes.

## Statistical methods

The primary endpoint is the absence of a recorded completion within 14 days after the recorded assignment start. Experiment identifiers define the held-out target experiments; learner, class and teacher identifiers define grouped and purged validation designs. All preprocessing is fitted on training folds only.

Experiment-specific AUROC eligibility requires at least 50 evaluable records and at least 10 observations in each outcome class. Calibration eligibility requires at least 100 records and at least 50 observations in each outcome class. Chronological local recalibration uses the earliest 20% of records as the calibration cohort, waits until the 14-day outcomes for that cohort have matured, excludes records beginning during the waiting period, and evaluates on later records. Calibration and evaluation cohorts must each contain at least 30 records and at least five observations in each outcome class.

Paired uncertainty is estimated by resampling whole experiments (10,000 bootstrap replicates unless otherwise specified). Capacity precision and recall intervals are calculated by resampling experiments and recomputing pooled numerators and denominators. The reported intervals are pointwise 95% intervals without multiplicity adjustment. Pooled calibration intercepts and slopes are estimated from all record-level out-of-fold predictions; experiment-level medians and interquartile ranges are calculated across the 70 calibration-eligible experiments. These are different estimands and should not be compared as if they were the same summary. Mean Brier comparisons for chronological local updating use the separate set of 69 local-recalibration-eligible experiments.

The historical logistic-regression comparator uses `SGDClassifier` with L2 regularisation. The standard-solver comparator uses `LogisticRegression(solver='lbfgs', C=1.0, max_iter=5000, tol=1e-4)` with L2 regularisation. The regularisation setting is fixed and is not selected using held-out target experiments. LightGBM uses the fixed settings recorded in `results/model_specification.csv`.

The local-prevalence reference assigns each later evaluation record the observed noncompletion prevalence in the corresponding experiment's calibration cohort. Calibration and evaluation membership is checked by comparing record keys in memory; participant-derived record hashes are not written to public result tables. Intercept-only updating worsened Brier score in 14/69 SGD logistic-regression experiments, 19/69 LBFGS logistic-regression experiments and 17/69 LightGBM experiments. The experiment-specific changes and their magnitudes are reported in the local-recalibration result tables.

The `auprc` field in historical result tables records non-interpolated average precision computed by `sklearn.metrics.average_precision_score` (reported as AP in the manuscript).

## Reproducibility and citation

The primary model seed is `20260807`; the alternate LightGBM seed and additional-analysis bootstrap seed are `20260808`. Software versions are pinned in `environment.yml`. The `CITATION.cff` file provides software citation metadata and links to this repository. The software is archived on Zenodo. Please use the version-specific Zenodo DOI when citing this release.

## License

The included `LICENSE` file reserves copyright and does not grant permission to reuse, modify or redistribute this code. Any broader reuse permission requires approval from the relevant copyright holders. The public source data remain subject to the terms of their original repository.
