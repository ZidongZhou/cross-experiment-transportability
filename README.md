# Cross-experiment transportability of student assignment noncompletion prediction in online learning

**Version 1.0**

Repository: [GitHub](https://github.com/ZidongZhou/cross-experiment-transportability)

Archived software release: [Zenodo, DOI 10.5281/zenodo.24374175](https://doi.org/10.5281/zenodo.24374175)

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

1. **`analysis_pipeline.py`** verifies and reads the source archive, reconstructs the analysis cohort, fits the primary models and validation designs, and writes the primary validation tables, cohort summaries, calibration summaries, sensitivity analyses, `results/horizon_lightgbm.csv` and the initial `results/convergence_diagnostics.csv`. This first set of diagnostics covers the SGD fits made by the primary pipeline. It records iteration counts and any `ConvergenceWarning` without changing model settings or predictions. The 7-day, 30-day and archive-closure LightGBM sensitivity models use the same five-fold experiment-blocked, learner-purged design as the corresponding logistic-regression analyses.
2. **`build_model_predictions.py`** reconstructs the cohort and fits the additional out-of-fold prediction sets used for the LBFGS comparator, learner-history and context-ablation comparisons, and the alternate-seed LightGBM reference. It writes local intermediate prediction files under `analysis_work/` and aggregate cohort and predictor tables under `results/`. The builder replaces the preliminary prediction workfile written by the primary pipeline and appends the SGD fit diagnostics from this stage to `results/convergence_diagnostics.csv`.
3. **`additional_analyses.py`** checks the local prediction-workfile provenance and computes paired experiment-level bootstrap intervals, validation-design contrasts, predictor comparisons, calibration benchmarks, capacity intervals, shortlist-stability analyses and `results/Supplementary_Table_S12.csv`. If the joint-purge results need to be refitted, the SGD fit diagnostics from that refit are also appended to the same file.
4. **`make_figures.py`** reads the aggregate result tables and writes EPS-only versions of Figures 1–4 and Supplementary Figures S1–S2.

The record-level prediction workfile is generated locally and is not included in the repository. Its provenance record stores the source-data and code SHA256 values, model configuration, row keys and prediction fingerprint. Run the four stages in the order shown; if you rerun `analysis_pipeline.py`, run `build_model_predictions.py` again before `additional_analyses.py`.

### Joint-purge analysis

The three aggregate tables for the joint learner-, class- and teacher-purged LOEO analysis are included in `results/`. Their SHA256 values, together with the model-specification file, are recorded in `results/aggregate_output_checksums.sha256`. The analysis checks these files against the reconstructed cohort before reusing them.

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
- **`figures/`** contains only the vector EPS versions of Figures 1–4 and Supplementary Figures S1–S2. Running `code/make_figures.py` regenerates these six EPS files.
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
| Capacity and shortlist stability | `capacity_screening.csv`, `capacity_model_uncertainty.csv`, `capacity_difference_uncertainty.csv`, `shortlist_overlap_summary.csv`, `shortlist_overlap_reference.csv`, `shortlist_overlap_reference_contrasts.csv` |
| Outcome-horizon sensitivity | `sensitivity_summary.csv`, `horizon_lightgbm.csv` |
| Model fitting diagnostics | `convergence_diagnostics.csv` |

The archived `convergence_diagnostics.csv` contains 505 SGD fit records: 324 from `analysis_pipeline.py` and 181 from `build_model_predictions.py`. None reached `max_iter=1500` or raised `ConvergenceWarning`; the maximum observed `n_iter_` was 55.

`Supplementary_Table_S12.csv` lists all 88 experiments and indicates eligibility for experiment-specific AUROC, calibration and chronological local recalibration analyses. For the 69 local-recalibration experiments, it also reports the calibration, waiting-period exclusion and evaluation sample sizes.

## Statistical methods

The primary endpoint is the absence of a recorded completion within 14 days after the recorded assignment start. Experiment identifiers define the held-out target experiments; learner, class and teacher identifiers define grouped and purged validation designs. All preprocessing is fitted on training folds only.

Experiment-specific AUROC eligibility requires at least 50 evaluable records and at least 10 observations in each outcome class. Calibration eligibility requires at least 100 records and at least 50 observations in each outcome class. Chronological local recalibration uses the earliest 20% of records as the calibration cohort, waits until the 14-day outcomes for that cohort have matured, excludes records beginning during the waiting period, and evaluates on later records. Calibration and evaluation cohorts must each contain at least 30 records and at least five observations in each outcome class. For each within-experiment review capacity \(c\) in 5%, 10%, 15%, 20%, 25% and 30%, the selected count is \(k=\max(1,\lceil c n\rceil)\); ties are resolved by the stable descending probability sort. Class- and teacher-level screening excludes groups with fewer than 20 records within an experiment. Precision and recall for these two ranking units therefore describe the retained eligible groups; experiment-level screening uses all records in the cohort.

Paired uncertainty is estimated by resampling whole experiments (10,000 bootstrap replicates unless otherwise specified). Capacity precision and recall intervals are calculated by resampling experiments and recomputing pooled numerators and denominators. The reported intervals are pointwise 95% intervals without multiplicity adjustment. Pooled calibration intercepts and slopes are estimated from all record-level out-of-fold predictions; experiment-level medians and interquartile ranges are calculated across the 70 calibration-eligible experiments. These are different estimands and should not be compared as if they were the same summary. Mean Brier comparisons for chronological local updating use the separate set of 69 local-recalibration-eligible experiments.

The historical logistic-regression comparator uses `SGDClassifier` with L2 regularisation (`alpha=1e-4`, `max_iter=1500`, `tol=1e-3`, `average=True`, `random_state=20260807`). The standard-solver comparator uses `LogisticRegression(solver='lbfgs', C=1.0, max_iter=5000, tol=1e-4)` with L2 regularisation. The regularisation setting is fixed and is not selected using held-out target experiments. LightGBM uses the fixed settings recorded in `results/model_specification.csv`.

The local-prevalence reference assigns each later evaluation record the observed noncompletion prevalence in the corresponding experiment's calibration cohort. Intercept-only updating worsened Brier score in 14/69 SGD logistic-regression experiments, 19/69 LBFGS logistic-regression experiments and 17/69 LightGBM experiments. The experiment-specific changes are reported in the local-recalibration result tables.

The `auprc` field in historical result tables records non-interpolated average precision computed by `sklearn.metrics.average_precision_score` (reported as AP in the manuscript).

## Reproducibility and citation

The primary model seed is `20260807`; the alternate LightGBM seed and additional-analysis bootstrap seed are `20260808`. Software versions are pinned in `environment.yml`. The `CITATION.cff` file records the software citation and the version 1.0 Zenodo DOI ([10.5281/zenodo.24374175](https://doi.org/10.5281/zenodo.24374175)).

## License

The included `LICENSE` file reserves copyright and does not grant permission to reuse, modify or redistribute this code. Any broader reuse permission requires approval from the relevant copyright holders. The public source data remain subject to the terms of their original repository.
