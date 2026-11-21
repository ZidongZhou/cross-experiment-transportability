#!/usr/bin/env python3
"""Additional analyses for transportability, calibration, predictor ablation and review capacity.

Run after code/build_model_predictions.py. The raw ASSISTments archive is required because
per-experiment predictions are not distributed in the public reproducibility package.
Only aggregate, experiment-level outputs are written to results/. Record-level
predictions remain in the local, git-ignored analysis_work/ directory.
"""
from __future__ import annotations
from pathlib import Path
import math
import os
import hashlib
import json
from itertools import combinations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import analysis_pipeline as ap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
WORK = ROOT / "analysis_work"
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)
SEED = 20260808
B = 10_000
KEY_COLUMNS = ["experiment_id", "student_id", "class_id", "teacher_id", "start", "y"]
REQUIRED_PREDICTION_COLUMNS = [
    "row_random__logistic_regression",
    "row_random__logistic_regression_lbfgs",
    "row_random__lightgbm",
    "row_random__lightgbm_no_context",
    "learner_purged_loeo__logistic_regression",
    "learner_purged_loeo__logistic_regression_lbfgs",
    "learner_purged_loeo__lightgbm",
    "learner_purged_loeo__lightgbm_seed_20260808",
    "learner_purged_loeo__lightgbm_no_context",
    "learner_purged_loeo__logistic_regression_history",
]
JOINT_OUTPUT_CHECKSUMS = OUT / "aggregate_output_checksums.sha256"
JOINT_OUTPUT_NAMES = (
    "combined_purge_by_experiment.csv",
    "combined_purge_performance.csv",
    "combined_purge_experiment_performance.csv",
    "model_specification.csv",
)

VALIDATION_ORDER = [
    "Row random", "Learner grouped", "Learner + class grouped",
    "Learner + teacher grouped", "Standard LOEO",
    "Training-size-matched LOEO", "Learner-purged LOEO",
    "Learner + class + teacher-purged LOEO",
]


def metric_row(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
    }


def bootstrap_ci(values, statistic="median", rng=None, n_boot=B):
    """Percentile CI from resampling experiment-level values, not records."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(SEED)
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be median or mean")
    observed = float(np.median(x) if statistic == "median" else np.mean(x))
    draw = rng.integers(0, len(x), size=(n_boot, len(x)))
    sampled = x[draw]
    boot = np.median(sampled, axis=1) if statistic == "median" else np.mean(sampled, axis=1)
    return observed, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def _prediction_content_sha256(frame, columns):
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype="uint64").tobytes()
    return hashlib.sha256(hashed).hexdigest()


def load_dataset():
    ap.verify_raw_archive(ap.RAW)
    # Reuse an existing local prediction workfile only when the source archive, generating code,
    # model configuration, record keys and probability values all match.
    prediction_workfile_path = WORK / "predictions.pkl"
    provenance_path = WORK / "prediction_provenance.json"
    if prediction_workfile_path.exists():
        if not provenance_path.exists():
            raise FileNotFoundError(
                f"{provenance_path} is missing. Re-run code/build_model_predictions.py "
                "to create a complete provenance record."
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if int(provenance.get("schema_version", 0)) != 1:
            raise ValueError("Prediction provenance schema is missing or unsupported; rebuild predictions.")
        current_raw_sha256 = ap.sha256_file(ap.RAW)
        if provenance.get("raw_sha256") != current_raw_sha256:
            raise ValueError("Existing predictions were generated from a different raw archive; rebuild predictions.")
        pipeline_path = Path(ap.__file__).resolve()
        builder_path = ROOT / "code" / "build_model_predictions.py"
        if provenance.get("analysis_pipeline_sha256") != ap.sha256_file(pipeline_path):
            raise ValueError("The analysis pipeline changed after predictions were generated; rebuild predictions.")
        if provenance.get("prediction_builder_sha256") != ap.sha256_file(builder_path):
            raise ValueError("The prediction builder changed after predictions were generated; rebuild predictions.")
        predwork = pd.read_pickle(prediction_workfile_path)
        if int(provenance.get("n_records", -1)) != len(predwork):
            raise ValueError("Prediction workfile row count does not match its provenance record.")
        if int(provenance.get("n_experiments", -1)) != int(predwork.experiment_id.nunique()):
            raise ValueError("Prediction workfile experiment count does not match its provenance record.")
        if int(provenance.get("events", -1)) != int(predwork.y.sum()):
            raise ValueError("Prediction workfile event count does not match its provenance record.")
        if predwork[KEY_COLUMNS].duplicated().any():
            raise ValueError("Prediction workfile workfile contains duplicate record keys.")
        prediction_columns = [c for c in predwork.columns if c not in KEY_COLUMNS]
        if prediction_columns != provenance.get("prediction_columns"):
            raise ValueError("Prediction workfile columns do not match their provenance record; rebuild predictions.")
        if not set(REQUIRED_PREDICTION_COLUMNS).issubset(prediction_columns):
            raise ValueError("Prediction workfile workfile is missing required model predictions; rebuild predictions.")
        probabilities = predwork[prediction_columns].to_numpy(dtype=float)
        if not (np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()):
            raise ValueError("Prediction workfile workfile contains invalid probabilities; rebuild predictions.")
        key_hash = _prediction_content_sha256(predwork, KEY_COLUMNS)
        if key_hash != provenance.get("key_sha256"):
            raise ValueError("Prediction workfile keys/outcomes do not match their provenance record.")
        prediction_hash = _prediction_content_sha256(predwork, KEY_COLUMNS + prediction_columns)
        if prediction_hash != provenance.get("prediction_sha256"):
            raise ValueError("Prediction workfile probabilities do not match their provenance record; rebuild predictions.")
        configuration = {
            "features": ap.FEATURE_SETS["Full transportable"],
            "feature_sets": {
                "No context": ap.FEATURE_SETS["No context"],
                "Complete learner history": ap.FEATURE_SETS["Complete learner history"],
            },
            "estimators": {
                "sgd": ap.make_model("Logistic regression").get_params(deep=True),
                "lbfgs": ap.make_model("Logistic regression (LBFGS)").get_params(deep=True),
                "lightgbm": ap.make_model("LightGBM").get_params(deep=True),
                "lightgbm_alternate_seed": ap.make_model("LightGBM seed 20260808").get_params(deep=True),
            },
            "seed": ap.SEED,
        }
        config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if config_hash != provenance.get("model_configuration_sha256"):
            raise ValueError("Model configuration changed after predictions were generated; rebuild predictions.")
        d = predwork[KEY_COLUMNS].copy().reset_index(drop=True)
        d["y"] = d.y.astype(int)
        return None, d
    alogs, priors, slog, _ = ap.load_raw(ap.RAW)
    full = ap.derive_dataset(alogs, priors, slog)
    d = full[full.evaluable14].copy().reset_index(drop=True)
    d["y"] = d.y14.astype(int)
    return full, d


def load_prediction_workfile(d):
    path = WORK / "predictions.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run code/build_model_predictions.py to create the complete local OOF prediction workfile."
        )
    predwork = pd.read_pickle(path)
    pd.testing.assert_frame_equal(
        predwork[KEY_COLUMNS].reset_index(drop=True),
        d[KEY_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )
    required = REQUIRED_PREDICTION_COLUMNS
    missing = [c for c in required if c not in predwork]
    if missing:
        raise KeyError(f"Prediction workfile lacks required columns: {missing}")
    return predwork


def joint_purge(d):
    required_features = set(ap.FEATURE_SETS["Full transportable"])
    missing_features = sorted(required_features.difference(d.columns))
    if missing_features:
        raise ValueError(
            "Joint-purge refitting requires the reconstructed predictor table; "
            f"missing feature columns: {missing_features}"
        )
    idx = np.arange(len(d))
    exp = d.experiment_id.to_numpy()
    stud = d.student_id.to_numpy()
    cls = d.class_id.to_numpy()
    teach = d.teacher_id.to_numpy()
    splits, stats = [], []
    for e in pd.unique(exp):
        te = idx[exp == e]
        base = idx[exp != e]
        mask = (
            ~np.isin(stud[base], np.unique(stud[te]))
            & ~np.isin(cls[base], np.unique(cls[te]))
            & ~np.isin(teach[base], np.unique(teach[te]))
        )
        tr = base[mask]
        assert len(np.intersect1d(np.unique(stud[tr]), np.unique(stud[te]))) == 0
        assert len(np.intersect1d(np.unique(cls[tr]), np.unique(cls[te]))) == 0
        assert len(np.intersect1d(np.unique(teach[tr]), np.unique(teach[te]))) == 0
        splits.append((tr, te))
        stats.append({
            "experiment_id": e, "standard_train_n": len(base),
            "combined_purged_train_n": len(tr),
            "combined_purged_fraction": 1 - len(tr) / len(base), "test_n": len(te),
        })
    fit_jobs = [
        (tr, te, {
            "validation": "Learner + class + teacher-purged LOEO",
            "fold": fold + 1,
            "experiment_id": str(d.iloc[te[0]].experiment_id),
        })
        for fold, (tr, te) in enumerate(splits)
    ]
    jobs = [
        delayed(ap.fit_predict)(
            tr, te, d, ap.FEATURE_SETS["Full transportable"],
            fit_context=context,
        )
        for tr, te, context in fit_jobs
    ]
    res = Parallel(n_jobs=ap.N_JOBS, backend="threading")(jobs) if ap.N_JOBS != 1 else [
        ap.fit_predict(
            tr, te, d, ap.FEATURE_SETS["Full transportable"],
            fit_context=context,
        )
        for tr, te, context in fit_jobs
    ]
    preds = {m: np.full(len(d), np.nan) for m in ("Logistic regression", "LightGBM")}
    for te, out in res:
        for m, p in out.items():
            preds[m][te] = p
    for m, p in preds.items():
        assert np.isfinite(p).all(), f"Missing predictions in joint purge: {m}"
    pd.DataFrame(stats).to_csv(OUT / "combined_purge_by_experiment.csv", index=False)
    rows = [dict(validation="Learner + class + teacher-purged LOEO", model=m, **metric_row(d.y, p))
            for m, p in preds.items()]
    pd.DataFrame(rows).to_csv(OUT / "combined_purge_performance.csv", index=False)
    ap.per_experiment_metrics(d, preds, "Learner + class + teacher-purged LOEO").to_csv(
        OUT / "combined_purge_experiment_performance.csv", index=False
    )
    write_joint_purge_checksums()
    return preds


def paired_model_delta_table(expperf, combperf):
    allperf = pd.concat([expperf, combperf], ignore_index=True)
    rng = np.random.default_rng(SEED)
    rows = []
    pivots = {}
    for validation in VALIDATION_ORDER:
        z = allperf[(allperf.validation == validation) & allperf.eligible_auroc].pivot(
            index="experiment_id", columns="model", values="auroc"
        ).dropna()
        if not {"LightGBM", "Logistic regression"}.issubset(z.columns):
            continue
        delta = (z["LightGBM"] - z["Logistic regression"]).to_numpy()
        if len(delta) != 80:
            raise ValueError(f"{validation}: expected 80 AUROC-eligible experiments, found {len(delta)}")
        med, lo, hi = bootstrap_ci(delta, "median", rng)
        rows.append({
            "validation": validation, "n_experiments": len(delta),
            "median_delta_auroc": med, "q1": float(np.quantile(delta, .25)),
            "q3": float(np.quantile(delta, .75)), "bootstrap_95_low": lo,
            "bootstrap_95_high": hi, "proportion_logistic_better": float(np.mean(delta < 0)),
        })
        pivots[validation] = z
    gap = pd.DataFrame(rows)
    gap.to_csv(OUT / "validation_gap_uncertainty.csv", index=False)
    pairs = list(combinations(VALIDATION_ORDER, 2))
    contrast_rows = []
    for a, b in pairs:
        if a not in pivots or b not in pivots:
            continue
        common = pivots[a].index.intersection(pivots[b].index)
        da = (pivots[a].loc[common, "LightGBM"] - pivots[a].loc[common, "Logistic regression"]).to_numpy()
        db = (pivots[b].loc[common, "LightGBM"] - pivots[b].loc[common, "Logistic regression"]).to_numpy()
        change = db - da
        med, lo, hi = bootstrap_ci(change, "median", rng)
        mean, mean_lo, mean_hi = bootstrap_ci(change, "mean", rng)
        contrast_rows.append({
            "from_validation": a, "to_validation": b, "n_common_experiments": len(common),
            "median_change_in_gap": med, "median_change_95_low": lo, "median_change_95_high": hi,
            "mean_change_in_gap": mean, "mean_change_95_low": mean_lo, "mean_change_95_high": mean_hi,
        })
    pd.DataFrame(contrast_rows).to_csv(OUT / "validation_gap_contrasts.csv", index=False)
    return gap, pivots


def paired_validation_model_contrasts(expperf, combperf):
    """Estimate paired experiment-level AUROC changes for each model across designs."""
    print("Estimating paired AUROC changes across validation designs...", flush=True)
    allperf = pd.concat([expperf, combperf], ignore_index=True)
    pairs = list(combinations(VALIDATION_ORDER, 2))
    rng = np.random.default_rng(SEED + 23)
    rows = []
    for model in ("Logistic regression", "LightGBM"):
        sub = allperf[(allperf.model == model) & allperf.eligible_auroc].copy()
        wide = sub.pivot(index="experiment_id", columns="validation", values="auroc")
        for from_design, to_design in pairs:
            if from_design not in wide or to_design not in wide:
                continue
            paired = wide[[from_design, to_design]].dropna()
            delta = (paired[to_design] - paired[from_design]).to_numpy(dtype=float)
            med, med_lo, med_hi = bootstrap_ci(delta, "median", rng)
            mean, mean_lo, mean_hi = bootstrap_ci(delta, "mean", rng)
            rows.append({
                "model": model,
                "from_validation": from_design,
                "to_validation": to_design,
                "n_common_experiments": len(delta),
                "median_change_auroc": med,
                "q1_change_auroc": float(np.quantile(delta, .25)),
                "q3_change_auroc": float(np.quantile(delta, .75)),
                "median_change_bootstrap_95_low": med_lo,
                "median_change_bootstrap_95_high": med_hi,
                "mean_change_auroc": mean,
                "mean_change_bootstrap_95_low": mean_lo,
                "mean_change_bootstrap_95_high": mean_hi,
                "median_direction": "decrease" if med < 0 else ("increase" if med > 0 else "no median change"),
                "interpretation": ("evidence of decrease" if med_hi < 0 else
                                   "evidence of increase" if med_lo > 0 else
                                   "no clear evidence of a directional change"),
            })
    out = pd.DataFrame(rows)
    expected = 2 * len(pairs)
    assert len(out) == expected, f"Expected {expected} per-model validation contrasts, found {len(out)}"
    assert (out.n_common_experiments == 80).all(), "All contrasts should use the same 80 AUROC-eligible experiments"
    out.to_csv(OUT / "validation_model_contrasts.csv", index=False)
    return out


def experiment_auc(d, predictions, label, model, feature_set):
    rows = []
    for e, g in d.groupby("experiment_id", sort=False):
        y = g.y.to_numpy()
        n, events = len(g), int(y.sum())
        eligible = n >= 50 and min(events, n - events) >= 10
        if not eligible:
            continue
        p = np.asarray(predictions)[g.index.to_numpy()]
        rows.append({
            "experiment_id": e, "validation": label, "model": model,
            "feature_set": feature_set, "n": n, "events": events,
            "auroc": roc_auc_score(y, p),
        })
    return pd.DataFrame(rows)


def paired_feature_summary(df, comparison, value_col="delta_auroc", rng=None):
    x = df[value_col].to_numpy(dtype=float)
    med, lo, hi = bootstrap_ci(x, "median", rng)
    mean, mean_lo, mean_hi = bootstrap_ci(x, "mean", rng)
    return {
        "comparison": comparison, "n_experiments": len(x),
        "median_difference": med, "q1": float(np.quantile(x, .25)),
        "q3": float(np.quantile(x, .75)), "median_bootstrap_95_low": lo,
        "median_bootstrap_95_high": hi, "mean_difference": mean,
        "mean_bootstrap_95_low": mean_lo, "mean_bootstrap_95_high": mean_hi,
    }


def run_feature_uncertainty(d, predwork):
    print("Estimating predictor-set uncertainty...", flush=True)
    # All predictions are generated once by build_model_predictions.py.
    hist_sgd = predwork["learner_purged_loeo__logistic_regression_history"].to_numpy()
    noctx_row = predwork["row_random__lightgbm_no_context"].to_numpy()
    noctx_loeo = predwork["learner_purged_loeo__lightgbm_no_context"].to_numpy()
    full_sgd = predwork["learner_purged_loeo__logistic_regression"].to_numpy()
    full_lgb_row = predwork["row_random__lightgbm"].to_numpy()
    full_lgb_loeo = predwork["learner_purged_loeo__lightgbm"].to_numpy()

    tables = [
        experiment_auc(d, full_sgd, "Learner-purged LOEO", "Logistic regression (SGD)", "Full transportable"),
        experiment_auc(d, hist_sgd, "Learner-purged LOEO", "Logistic regression (SGD)", "Complete learner history"),
        experiment_auc(d, predwork["learner_purged_loeo__logistic_regression_lbfgs"].to_numpy(),
                       "Learner-purged LOEO", "Logistic regression (LBFGS)", "Full transportable"),
        experiment_auc(d, full_lgb_row, "Row random", "LightGBM", "Full transportable"),
        experiment_auc(d, noctx_row, "Row random", "LightGBM", "No context"),
        experiment_auc(d, full_lgb_loeo, "Learner-purged LOEO", "LightGBM", "Full transportable"),
        experiment_auc(d, noctx_loeo, "Learner-purged LOEO", "LightGBM", "No context"),
    ]
    per = pd.concat(tables, ignore_index=True)
    per.to_csv(OUT / "feature_auc_experiment_level.csv", index=False)
    rng = np.random.default_rng(SEED + 1)
    auc_rows = []
    for (model, validation, feature_set), g in per.groupby(["model", "validation", "feature_set"]):
        med, lo, hi = bootstrap_ci(g.auroc.to_numpy(), "median", rng)
        auc_rows.append({
            "model": model, "validation": validation, "feature_set": feature_set,
            "n_experiments": len(g), "median_auroc": med,
            "q1_auroc": float(g.auroc.quantile(.25)), "q3_auroc": float(g.auroc.quantile(.75)),
            "bootstrap_95_low": lo, "bootstrap_95_high": hi,
        })
    pd.DataFrame(auc_rows).to_csv(OUT / "feature_auc_summary.csv", index=False)

    wide = per.pivot_table(index="experiment_id", columns=["model", "validation", "feature_set"], values="auroc")
    rows, long_rows = [], []
    comparisons = [
        ("Logistic regression (SGD)", "Learner-purged LOEO", "Complete learner history", "Full transportable",
         "Logistic SGD: learner history minus full set"),
        ("LightGBM", "Row random", "No context", "Full transportable",
         "LightGBM: context removal under row-random"),
        ("LightGBM", "Learner-purged LOEO", "No context", "Full transportable",
         "LightGBM: context removal under learner-purged LOEO"),
    ]
    deltas = {}
    for model, validation, a, b, label in comparisons:
        x = wide[(model, validation, a)] - wide[(model, validation, b)]
        z = x.dropna().rename("delta_auroc").reset_index()
        if len(z) != 80:
            raise ValueError(f"{label}: expected 80 paired AUROC-eligible experiments, found {len(z)}")
        z["comparison"] = label
        long_rows.append(z)
        deltas[label] = x
        rows.append(paired_feature_summary(z, label, rng=rng))
    row_effect = deltas["LightGBM: context removal under row-random"]
    loeo_effect = deltas["LightGBM: context removal under learner-purged LOEO"]
    common = row_effect.dropna().index.intersection(loeo_effect.dropna().index)
    interaction = (row_effect.loc[common] - loeo_effect.loc[common]).rename("delta_auroc").reset_index()
    interaction["comparison"] = "LightGBM: row-random context effect minus learner-purged LOEO context effect"
    long_rows.append(interaction)
    rows.append(paired_feature_summary(interaction, interaction.comparison.iloc[0], rng=rng))
    pd.concat(long_rows, ignore_index=True).to_csv(OUT / "feature_ablation_experiment_level.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "feature_ablation_uncertainty.csv", index=False)
    return {"hist_sgd": hist_sgd, "noctx_row": noctx_row, "noctx_loeo": noctx_loeo}

def run_lbfgs_sensitivity(d, predwork):
    print("Evaluating the standard logistic-regression comparator...", flush=True)
    model_name = "Logistic regression (LBFGS)"
    sgd = {
        "Row random": predwork["row_random__logistic_regression"].to_numpy(),
        "Learner-purged LOEO": predwork["learner_purged_loeo__logistic_regression"].to_numpy(),
    }
    lgb = {
        "Row random": predwork["row_random__lightgbm"].to_numpy(),
        "Learner-purged LOEO": predwork["learner_purged_loeo__lightgbm"].to_numpy(),
    }
    lb = {
        "Row random": predwork["row_random__logistic_regression_lbfgs"].to_numpy(),
        "Learner-purged LOEO": predwork["learner_purged_loeo__logistic_regression_lbfgs"].to_numpy(),
    }
    rows = []
    for validation in ("Row random", "Learner-purged LOEO"):
        for label, pred in (("Logistic regression (SGD)", sgd[validation]),
                            (model_name, lb[validation]), ("LightGBM", lgb[validation])):
            ci, cs = ap.calibration_coefficients(d.y.to_numpy(), pred)
            rows.append({
                "validation": validation, "model": label,
                **metric_row(d.y.to_numpy(), pred),
                "calibration_intercept": ci, "calibration_slope": cs,
            })
    pd.DataFrame(rows).to_csv(OUT / "logistic_lbfgs_comparison.csv", index=False)

    rng = np.random.default_rng(SEED + 2)
    gap_rows = []
    for validation in ("Row random", "Learner-purged LOEO"):
        auc_sgd = experiment_auc(d, sgd[validation], validation, "SGD", "Full")
        auc_lb = experiment_auc(d, lb[validation], validation, "LBFGS", "Full")
        auc_lgb = experiment_auc(d, lgb[validation], validation, "LightGBM", "Full")
        z = auc_sgd[["experiment_id", "auroc"]].rename(columns={"auroc": "sgd"}).merge(
            auc_lb[["experiment_id", "auroc"]].rename(columns={"auroc": "lbfgs"}), on="experiment_id"
        ).merge(auc_lgb[["experiment_id", "auroc"]].rename(columns={"auroc": "lgbm"}), on="experiment_id")
        for comparator in ("sgd", "lbfgs"):
            gap = z.lgbm - z[comparator]
            med, lo, hi = bootstrap_ci(gap.to_numpy(), "median", rng)
            gap_rows.append({
                "validation": validation,
                "comparison": f"LightGBM minus {'Logistic regression (SGD)' if comparator == 'sgd' else model_name}",
                "comparison_type": "Model gap", "n_experiments": len(z),
                "median_difference_auroc": med, "bootstrap_95_low": lo,
                "bootstrap_95_high": hi, "proportion_logistic_better": float(np.mean(gap < 0)),
            })
        change = z.lbfgs - z.sgd
        med, lo, hi = bootstrap_ci(change.to_numpy(), "median", rng)
        gap_rows.append({
            "validation": validation, "comparison": "LBFGS minus SGD logistic regression",
            "comparison_type": "Solver change", "n_experiments": len(z),
            "median_difference_auroc": med, "bootstrap_95_low": lo,
            "bootstrap_95_high": hi, "proportion_logistic_better": np.nan,
        })
    pd.DataFrame(gap_rows).to_csv(OUT / "logistic_lbfgs_gap_comparison.csv", index=False)

    slope_rows = []
    experiment_calibration_rows = []
    calibration_eligible = []
    for experiment_id, group in d.groupby("experiment_id", sort=False):
        n = len(group)
        events = int(group.y.sum())
        if n >= 100 and events >= 50 and (n - events) >= 50:
            calibration_eligible.append(experiment_id)
            idx = group.index.to_numpy()
            for validation in ("Row random", "Learner-purged LOEO"):
                for model_label, pred_map in (("Logistic regression (SGD)", sgd),
                                              (model_name, lb),
                                              ("LightGBM", lgb)):
                    intercept, slope = ap.calibration_coefficients(group.y.to_numpy(), pred_map[validation][idx])
                    experiment_calibration_rows.append({
                        "experiment_id": experiment_id, "validation": validation,
                        "model": model_label, "n": n, "events": events,
                        "calibration_intercept": intercept, "calibration_slope": slope,
                    })
    experiment_calibration = pd.DataFrame(experiment_calibration_rows)
    experiment_calibration.to_csv(OUT / "logistic_lbfgs_calibration_experiment_level.csv", index=False)

    # Report pooled calibration estimates for both validation designs and all three models.
    for validation in ("Row random", "Learner-purged LOEO"):
        for model_label, pred_map in (("Logistic regression (SGD)", sgd),
                                      (model_name, lb),
                                      ("LightGBM", lgb)):
            intercept, slope = ap.calibration_coefficients(d.y.to_numpy(), pred_map[validation])
            slope_rows.append({
                "validation": validation, "model": model_label, "scope": "Pooled",
                "calibration_intercept": intercept, "calibration_slope": slope,
                "n_experiments": int(d.experiment_id.nunique()),
            })
    pd.DataFrame(slope_rows).to_csv(OUT / "logistic_lbfgs_calibration_slope.csv", index=False)

    # Add the standard-solver experiment-level distribution to the common calibration table.
    strict = experiment_calibration[
        (experiment_calibration.validation == "Learner-purged LOEO")
        & (experiment_calibration.model == model_name)
    ]
    calibration_rows = []
    for metric, column in (("Calibration intercept", "calibration_intercept"),
                           ("Calibration slope", "calibration_slope")):
        values = strict[column].dropna().to_numpy(dtype=float)
        calibration_rows.append({
            "scope": "Experiment-level", "model": model_name, "metric": metric,
            "value": float(np.median(values)),
            "q1": float(np.quantile(values, .25)), "q3": float(np.quantile(values, .75)),
            "n_experiments": int(len(values)),
        })
    pooled_intercept, pooled_slope = ap.calibration_coefficients(d.y.to_numpy(), lb["Learner-purged LOEO"])
    calibration_rows.extend([
        {"scope": "Pooled", "model": model_name, "metric": "Calibration intercept",
         "value": pooled_intercept, "q1": np.nan, "q3": np.nan, "n_experiments": int(d.experiment_id.nunique())},
        {"scope": "Pooled", "model": model_name, "metric": "Calibration slope",
         "value": pooled_slope, "q1": np.nan, "q3": np.nan, "n_experiments": int(d.experiment_id.nunique())},
    ])
    calibration_path = OUT / "calibration_summary.csv"
    if calibration_path.exists():
        current = pd.read_csv(calibration_path)
        current = current[current.model != model_name]
        pd.concat([current, pd.DataFrame(calibration_rows)], ignore_index=True).to_csv(calibration_path, index=False)
    return {"row": lb["Row random"], "loeo": lb["Learner-purged LOEO"], "sgd": sgd, "lgb": lgb}

def local_prevalence_reference(d, return_membership=False):
    rows = []
    membership = {}
    cohorts = ap.local_recalibration_cohorts(d)
    for experiment_id, cohort in cohorts.items():
        early = cohort["calibration"]
        later = cohort["evaluation"]
        update_time = cohort["update_time"]
        membership[experiment_id] = (cohort["calibration_keys"], cohort["evaluation_keys"])
        prevalence = float(early.y.mean())
        y_later = later.y.to_numpy()
        p_ref = np.full(len(later), prevalence, dtype=float)
        n_total = int((d.experiment_id == experiment_id).sum())
        rows.append({
            "experiment_id": experiment_id,
            "n_total_evaluable": n_total,
            "n_calibration": len(early),
            "n_excluded_while_waiting": n_total - len(early) - len(later),
            "n_evaluation": len(later),
            "calibration_prevalence": prevalence,
            "evaluation_prevalence": float(np.mean(y_later)),
            "prevalence_reference_brier": brier_score_loss(y_later, p_ref),
            "prevalence_reference_log_loss": log_loss(y_later, p_ref),
            "update_time": update_time,
        })
    result = pd.DataFrame(rows)
    return (result, membership) if return_membership else result


def export_supplementary_table_s12(d, local_reference):
    """Export experiment-level eligibility and chronological cohort counts."""
    cohort = d.groupby("experiment_id", sort=False).agg(
        n_evaluable_14d=("y", "size"),
        noncompletion_events_14d=("y", "sum"),
        noncompletion_prevalence_14d=("y", "mean"),
    ).reset_index()
    cohort["noncompletion_events_14d"] = cohort.noncompletion_events_14d.astype(int)
    cohort["eligible_for_auroc"] = (
        (cohort.n_evaluable_14d >= 50)
        & (cohort.noncompletion_events_14d >= 10)
        & ((cohort.n_evaluable_14d - cohort.noncompletion_events_14d) >= 10)
    )
    cohort["eligible_for_calibration"] = (
        (cohort.n_evaluable_14d >= 100)
        & (cohort.noncompletion_events_14d >= 50)
        & ((cohort.n_evaluable_14d - cohort.noncompletion_events_14d) >= 50)
    )
    local_columns = [
        "experiment_id", "n_calibration", "n_excluded_while_waiting", "n_evaluation"
    ]
    local = local_reference[local_columns].copy()
    local["eligible_for_local_recalibration"] = True
    table = cohort.merge(local, on="experiment_id", how="left", validate="one_to_one")
    table["eligible_for_local_recalibration"] = table.eligible_for_local_recalibration.notna()
    table = table.rename(columns={
        "n_calibration": "n_calibration_records",
        "n_excluded_while_waiting": "n_waiting_period_excluded",
        "n_evaluation": "n_evaluation_records",
    }).sort_values("experiment_id").reset_index(drop=True)
    for column in ("n_calibration_records", "n_waiting_period_excluded", "n_evaluation_records"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    assert table.experiment_id.is_unique, "Supplementary Table S12 contains duplicate experiment IDs"
    assert len(table) == 88, f"Expected 88 experiments, found {len(table)}"
    assert int(table.eligible_for_auroc.sum()) == 80
    assert int(table.eligible_for_calibration.sum()) == 70
    assert int(table.eligible_for_local_recalibration.sum()) == 69
    local_rows = table.loc[table.eligible_for_local_recalibration]
    assert local_rows[["n_calibration_records", "n_waiting_period_excluded", "n_evaluation_records"]].notna().all().all()
    assert (local_rows["n_calibration_records"] + local_rows["n_waiting_period_excluded"] +
            local_rows["n_evaluation_records"] == local_rows["n_evaluable_14d"]).all()
    assert int(local_rows["n_evaluation_records"].sum()) == int(local_reference.n_evaluation.sum())
    table.to_csv(OUT / "Supplementary_Table_S12.csv", index=False)
    return table


def build_local_recalibration_worsened_table(lr, model_names):
    """Build experiment-level details for models whose intercept update worsened Brier."""
    worsened_rows = []
    for model in model_names:
        model_sub = lr.loc[lr["model"] == model].copy()
        wide = model_sub.pivot(
            index="experiment_id", columns="probability_version", values="brier"
        )
        required_versions = {"Intercept update", "No local update"}
        if not required_versions.issubset(wide.columns):
            raise ValueError(f"{model}: local-recalibration Brier table is missing required versions.")
        delta = wide["Intercept update"] - wide["No local update"]
        worsened = delta[delta > 0].sort_values(ascending=False)
        detail_columns = ["n_calibration", "n_evaluation", "update_time"]
        inconsistent = (
            model_sub.groupby("experiment_id")[detail_columns]
            .nunique(dropna=False).gt(1).any(axis=1)
        )
        if inconsistent.any():
            raise ValueError(
                f"{model}: calibration/evaluation metadata varies within an experiment."
            )
        details = model_sub.drop_duplicates("experiment_id").set_index("experiment_id")
        if not details.index.is_unique:
            raise ValueError(f"{model}: experiment-level recalibration details are not unique.")
        if not set(worsened.index).issubset(details.index):
            raise AssertionError(f"{model}: worsened experiments are missing calibration/evaluation details.")
        for experiment_id, value in worsened.items():
            worsened_rows.append({
                "model": model,
                "experiment_id": experiment_id,
                "n_calibration": int(details.loc[experiment_id, "n_calibration"]),
                "n_evaluation": int(details.loc[experiment_id, "n_evaluation"]),
                "update_time": details.loc[experiment_id, "update_time"],
                "brier_increase": float(value),
                "brier_before": float(wide.loc[experiment_id, "No local update"]),
                "brier_after": float(wide.loc[experiment_id, "Intercept update"]),
            })
    return pd.DataFrame(worsened_rows, columns=[
        "model", "experiment_id", "n_calibration", "n_evaluation",
        "update_time", "brier_increase", "brier_before", "brier_after",
    ])


def run_local_prevalence_analysis(d, purged_predictions):
    print("Evaluating local calibration benchmarks...", flush=True)
    lr, recalibration_membership = ap.local_recalibration(d, purged_predictions, return_membership=True)
    lr.to_csv(OUT / "local_recalibration_by_experiment.csv", index=False)
    ref, prevalence_membership = local_prevalence_reference(d, return_membership=True)
    ref.to_csv(OUT / "local_prevalence_reference_by_experiment.csv", index=False)

    # Compare the actual record keys in memory; no participant-derived hashes are
    # written to the public aggregate result tables.
    assert recalibration_membership == prevalence_membership, (
        "The local-prevalence reference and recalibration models use different records"
    )
    assert ref.experiment_id.nunique() == 69, f"Expected 69 eligible experiments, got {ref.experiment_id.nunique()}"
    assert ref.experiment_id.is_unique, "Prevalence reference has duplicate experiment rows"
    for model in purged_predictions:
        sub = lr[lr.model == model].copy()
        assert sub.experiment_id.nunique() == 69, f"{model}: expected 69 recalibration experiments"
        assert set(sub.experiment_id) == set(ref.experiment_id), f"{model}: eligible experiments differ from prevalence reference"
        by_exp = sub.drop_duplicates("experiment_id").set_index("experiment_id")
        ref_idx = ref.set_index("experiment_id")
        for col in ("n_calibration", "n_evaluation", "update_time"):
            left = by_exp.loc[ref_idx.index, col]
            right = ref_idx[col]
            if col == "update_time":
                assert (pd.to_datetime(left) == pd.to_datetime(right)).all(), f"{model}: update times differ"
            else:
                assert (left.astype(int).to_numpy() == right.astype(int).to_numpy()).all(), f"{model}: {col} differs"
        assert (ref.n_calibration + ref.n_excluded_while_waiting + ref.n_evaluation == ref.n_total_evaluable).all()

    counts = pd.DataFrame([{
        "n_experiments": len(ref), "calibration_records": int(ref.n_calibration.sum()),
        "excluded_while_waiting_records": int(ref.n_excluded_while_waiting.sum()),
        "evaluation_records": int(ref.n_evaluation.sum()),
        "total_records_in_eligible_experiments": int(ref.n_total_evaluable.sum()),
        "min_experiment_size": int(ref.n_total_evaluable.min()),
        "max_experiment_size": int(ref.n_total_evaluable.max()),
    }])
    counts.to_csv(OUT / "local_recalibration_counts.csv", index=False)

    summaries = []
    for model in purged_predictions:
        sub = lr[lr.model == model]
        wide = sub.pivot(index="experiment_id", columns="probability_version", values="brier")
        for version in ("No local update", "Intercept update", "Intercept + slope update"):
            x = wide[version].dropna()
            mean_brier, mean_lo, mean_hi = bootstrap_ci(x.to_numpy(), "mean", np.random.default_rng(SEED + len(summaries)))
            summaries.append({
                "model": model, "probability_version": version, "n_experiments": len(x),
                "mean_experiment_brier": mean_brier, "median_experiment_brier": float(x.median()),
                "mean_brier_bootstrap_95_low": mean_lo, "mean_brier_bootstrap_95_high": mean_hi,
                "improved_vs_no_update": int((wide[version] < wide["No local update"]).sum()) if version != "No local update" else 0,
                "unchanged_vs_no_update": int((wide[version] == wide["No local update"]).sum()) if version != "No local update" else 0,
                "worsened_vs_no_update": int((wide[version] > wide["No local update"]).sum()) if version != "No local update" else 0,
            })
    base_summary = pd.DataFrame(summaries)
    prevalence_summary = pd.DataFrame([{
        "model": "Prevalence-only reference", "probability_version": "Calibration-cohort prevalence",
        "n_experiments": len(ref), "mean_experiment_brier": ref.prevalence_reference_brier.mean(),
        "median_experiment_brier": ref.prevalence_reference_brier.median(),
        "mean_brier_bootstrap_95_low": bootstrap_ci(ref.prevalence_reference_brier.to_numpy(), "mean", np.random.default_rng(SEED + 90))[1],
        "mean_brier_bootstrap_95_high": bootstrap_ci(ref.prevalence_reference_brier.to_numpy(), "mean", np.random.default_rng(SEED + 90))[2],
        "improved_vs_no_update": np.nan, "worsened_vs_no_update": np.nan,
    }])
    comparison_summary = pd.concat([base_summary, prevalence_summary], ignore_index=True)
    comparison_summary.to_csv(OUT / "local_recalibration_comparison.csv", index=False)

    # Keep the compact supplementary summary aligned with all three models and the reference.
    compact = lr.groupby(["model", "probability_version"]).agg(
        experiments=("experiment_id", "nunique"),
        evaluation_records=("n_evaluation", "sum"),
        mean_brier=("brier", "mean"),
        mean_log_loss=("log_loss", "mean"),
        median_brier=("brier", "median"),
    ).reset_index()
    compact = pd.concat([compact, pd.DataFrame([{
        "model": "Prevalence-only reference",
        "probability_version": "Calibration-cohort prevalence",
        "experiments": int(len(ref)),
        "evaluation_records": int(ref.n_evaluation.sum()),
        "mean_brier": float(ref.prevalence_reference_brier.mean()),
        "mean_log_loss": float(ref.prevalence_reference_log_loss.mean()),
        "median_brier": float(ref.prevalence_reference_brier.median()),
    }])], ignore_index=True)
    compact.to_csv(OUT / "local_recalibration.csv", index=False)
    export_supplementary_table_s12(d, ref)

    change_rows = []
    for model in purged_predictions:
        wide = lr[lr.model == model].pivot(index="experiment_id", columns="probability_version", values="brier")
        for version in ("Intercept update", "Intercept + slope update"):
            delta = wide[version] - wide["No local update"]
            mean, lo, hi = bootstrap_ci(delta.to_numpy(), "mean", np.random.default_rng(SEED + len(change_rows)))
            med, mlo, mhi = bootstrap_ci(delta.to_numpy(), "median", np.random.default_rng(SEED + 20 + len(change_rows)))
            change_rows.append({"model": model, "probability_version": version, "n_experiments": len(delta),
                                "mean_change_brier": mean, "mean_bootstrap_95_low": lo, "mean_bootstrap_95_high": hi,
                                "median_change_brier": med, "median_bootstrap_95_low": mlo, "median_bootstrap_95_high": mhi,
                                "improved": int((delta < 0).sum()), "unchanged": int((delta == 0).sum()),
                                "worsened": int((delta > 0).sum())})
    pd.DataFrame(change_rows).to_csv(OUT / "local_recalibration_change.csv", index=False)

    pair_rows = []
    for model in purged_predictions:
        wide = lr[lr.model == model].pivot(index="experiment_id", columns="probability_version", values="brier")
        for reference in ("No local update", "Calibration-cohort prevalence"):
            target = wide["Intercept update"]
            baseline = (wide[reference] if reference == "No local update"
                        else ref.set_index("experiment_id").prevalence_reference_brier)
            delta = (target - baseline).dropna()
            mean, lo, hi = bootstrap_ci(delta.to_numpy(), "mean", np.random.default_rng(SEED + len(pair_rows)))
            med, mlo, mhi = bootstrap_ci(delta.to_numpy(), "median", np.random.default_rng(SEED + 40 + len(pair_rows)))
            pair_rows.append({"model": model, "comparison": f"Intercept update minus {reference}",
                              "n_experiments": len(delta), "mean_difference_brier": mean,
                              "mean_bootstrap_95_low": lo, "mean_bootstrap_95_high": hi,
                              "median_difference_brier": med, "median_bootstrap_95_low": mlo,
                              "median_bootstrap_95_high": mhi})
    pd.DataFrame(pair_rows).to_csv(OUT / "local_recalibration_pairwise_contrasts.csv", index=False)

    worsened_table = build_local_recalibration_worsened_table(
        lr, purged_predictions.keys()
    )
    worsened_table.to_csv(
        OUT / "local_recalibration_worsened_experiments.csv", index=False
    )
    worsening_summary = []
    for model in purged_predictions:
        wide = lr[lr.model == model].pivot(index="experiment_id", columns="probability_version", values="brier")
        delta = wide["Intercept update"] - wide["No local update"]
        positive = delta[delta > 0]
        worsening_summary.append({"model": model, "n_experiments": len(delta),
                                  "improved": int((delta < 0).sum()), "unchanged": int((delta == 0).sum()),
                                  "worsened": int((delta > 0).sum()),
                                  "median_brier_increase_among_worsened": float(positive.median()) if len(positive) else np.nan,
                                  "mean_brier_increase_among_worsened": float(positive.mean()) if len(positive) else np.nan,
                                  "max_brier_increase": float(positive.max()) if len(positive) else np.nan})
    pd.DataFrame(worsening_summary).to_csv(OUT / "local_recalibration_worsening_summary.csv", index=False)
    return ref

def capacity_experiment_table(d, prediction_map, capacities=(.05, .10, .15, .20, .25, .30)):
    rows = []
    for e, g in d.groupby("experiment_id", sort=False):
        ii = g.index.to_numpy()
        y = g.y.to_numpy()
        n, events = len(g), int(y.sum())
        for capacity in capacities:
            k = max(1, int(math.ceil(capacity * n)))
            for model, p in prediction_map.items():
                selected = ii[np.argsort(-p[ii], kind="mergesort")[:k]]
                tp = int(d.loc[selected, "y"].sum())
                rows.append({
                    "experiment_id": e, "capacity": capacity, "model": model,
                    "experiment_n": n, "total_events": events, "selected_n": k,
                    "selected_events": tp, "precision": tp / k,
                    "recall": tp / events if events > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def run_capacity_uncertainty(d, prediction_map):
    print("Estimating capacity-specific precision and recall uncertainty...", flush=True)
    capexp = capacity_experiment_table(d, prediction_map)
    capexp.to_csv(OUT / "capacity_experiment_metrics.csv", index=False)
    rng = np.random.default_rng(SEED + 3)
    model_rows, diff_rows, pooled_rows = [], [], []
    for capacity, sub in capexp.groupby("capacity", sort=True):
        exp_ids = pd.unique(sub.experiment_id)
        by_model = {m: sub[sub.model == m].set_index("experiment_id").loc[exp_ids] for m in prediction_map}
        n_exp = len(exp_ids)
        draws = rng.integers(0, n_exp, size=(B, n_exp))
        boot_metrics = {}
        observed_metrics = {}
        for model, frame in by_model.items():
            sel = frame.selected_n.to_numpy(dtype=float)
            tp = frame.selected_events.to_numpy(dtype=float)
            ev = frame.total_events.to_numpy(dtype=float)
            obs_precision = tp.sum() / sel.sum()
            obs_recall = tp.sum() / ev.sum()
            b_precision = tp[draws].sum(axis=1) / sel[draws].sum(axis=1)
            b_recall = tp[draws].sum(axis=1) / ev[draws].sum(axis=1)
            boot_metrics[model] = {"precision": b_precision, "recall": b_recall}
            observed_metrics[model] = {"precision": obs_precision, "recall": obs_recall}
            model_rows.append({
                "capacity": capacity, "model": model, "n_experiments": n_exp,
                "precision": obs_precision, "precision_bootstrap_95_low": float(np.quantile(b_precision, .025)),
                "precision_bootstrap_95_high": float(np.quantile(b_precision, .975)),
                "recall": obs_recall, "recall_bootstrap_95_low": float(np.quantile(b_recall, .025)),
                "recall_bootstrap_95_high": float(np.quantile(b_recall, .975)),
            })
            pooled_rows.append({
                "capacity": capacity, "model": model, "selected_n": int(sel.sum()),
                "selected_events": int(tp.sum()), "total_events": int(ev.sum()),
                "precision": obs_precision, "recall": obs_recall,
                "analysis_prevalence": float(d.y.mean()),
                "lift_over_analysis_prevalence": obs_precision / float(d.y.mean()),
            })
        comparisons = [("LightGBM", "Logistic regression (SGD)"),
                       ("LightGBM", "Logistic regression (LBFGS)")]
        for a, b in comparisons:
            if a not in boot_metrics or b not in boot_metrics:
                continue
            for metric in ("precision", "recall"):
                observed = observed_metrics[a][metric] - observed_metrics[b][metric]
                diff = boot_metrics[a][metric] - boot_metrics[b][metric]
                diff_rows.append({
                    "capacity": capacity, "comparison": f"{a} minus {b}", "metric": metric,
                    "n_experiments": n_exp, "difference": observed,
                    "bootstrap_95_low": float(np.quantile(diff, .025)),
                    "bootstrap_95_high": float(np.quantile(diff, .975)),
                    "bootstrap_unit": "experiment",
                })
    pd.DataFrame(model_rows).to_csv(OUT / "capacity_model_uncertainty.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(OUT / "capacity_difference_uncertainty.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(OUT / "capacity_curve.csv", index=False)
    return capexp


def shortlist_metrics(d, prediction_map, capacities=(.05, .10, .15, .20, .25, .30)):
    cohort = pd.read_csv(OUT / "experiment_cohort.csv")
    cohort["eligible_auroc"] = (cohort.n >= 50) & (cohort.events >= 10) & ((cohort.n - cohort.events) >= 10)
    eligible80 = set(cohort.loc[cohort.eligible_auroc, "experiment_id"])
    assert len(eligible80) == 80, f"Expected 80 AUROC-eligible experiments, found {len(eligible80)}"
    rows, per_exp = [], []
    pairs = [
        ("LightGBM seed 20260807", "LightGBM seed 20260808"),
        ("LightGBM", "Logistic regression (SGD)"),
        ("LightGBM", "Logistic regression (LBFGS)"),
    ]
    rng = np.random.default_rng(SEED + 4)
    for scope, exp_set in (("AUROC-eligible 80 experiments", eligible80),
                           ("All 88 experiments", set(cohort.experiment_id))):
        gs = [(e, g) for e, g in d.groupby("experiment_id", sort=False) if e in exp_set]
        min_n = min(len(g) for _, g in gs)
        for capacity in capacities:
            for a, b in pairs:
                if a not in prediction_map or b not in prediction_map:
                    continue
                overlaps, jaccards, shared_counts, selected_counts = [], [], [], []
                for e, g in gs:
                    ii = g.index.to_numpy()
                    k = max(1, int(math.ceil(capacity * len(ii))))
                    ia = set(ii[np.argsort(-prediction_map[a][ii], kind="mergesort")[:k]])
                    ib = set(ii[np.argsort(-prediction_map[b][ii], kind="mergesort")[:k]])
                    shared = len(ia & ib)
                    union = len(ia | ib)
                    overlap = shared / k
                    jaccard = shared / union if union else np.nan
                    overlaps.append(overlap); jaccards.append(jaccard)
                    shared_counts.append(shared); selected_counts.append(k)
                    per_exp.append({
                        "scope": scope, "capacity": capacity, "comparison": f"{a} vs {b}",
                        "experiment_id": e, "experiment_n": len(ii), "selected_n": k,
                        "shared_n": shared, "overlap": overlap, "jaccard": jaccard,
                    })
                x = np.asarray(overlaps, dtype=float)
                med, lo, hi = bootstrap_ci(x, "median", rng)
                rows.append({
                    "scope": scope, "capacity": capacity, "comparison": f"{a} vs {b}",
                    "n_experiments": len(gs), "minimum_experiment_size": min_n,
                    "minimum_selected_at_capacity": min(math.ceil(capacity * len(g)) for _, g in gs),
                    "median_overlap": med, "q1_overlap": float(np.quantile(x, .25)),
                    "q3_overlap": float(np.quantile(x, .75)),
                    "median_overlap_bootstrap_95_low": lo, "median_overlap_bootstrap_95_high": hi,
                    "size_weighted_overlap": float(np.sum(shared_counts) / np.sum(selected_counts)),
                    "median_jaccard": float(np.nanmedian(jaccards)),
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "shortlist_overlap_reference.csv", index=False)
    summary[summary.scope == "AUROC-eligible 80 experiments"].to_csv(
        OUT / "shortlist_overlap_summary.csv", index=False
    )
    per_exp_df = pd.DataFrame(per_exp)
    per_exp_df.to_csv(OUT / "shortlist_overlap_by_experiment.csv", index=False)

    # Paired contrast: how much lower is cross-model overlap than the within-LightGBM
    # seed-to-seed reference, using the same experiments at each capacity?
    contrast_rows = []
    for scope in ("AUROC-eligible 80 experiments", "All 88 experiments"):
        for capacity in capacities:
            z = per_exp_df[(per_exp_df.scope == scope) & (per_exp_df.capacity == capacity)].pivot(
                index="experiment_id", columns="comparison", values="overlap"
            )
            ref_col = "LightGBM seed 20260807 vs LightGBM seed 20260808"
            for comp in ("LightGBM vs Logistic regression (SGD)", "LightGBM vs Logistic regression (LBFGS)"):
                if ref_col not in z or comp not in z:
                    continue
                paired = z[[ref_col, comp]].dropna()
                delta = (paired[ref_col] - paired[comp]).to_numpy(dtype=float)
                med, lo, hi = bootstrap_ci(delta, "median", rng)
                mean, mean_lo, mean_hi = bootstrap_ci(delta, "mean", rng)
                contrast_rows.append({
                    "scope": scope, "capacity": capacity, "reference_minus_cross_model": comp,
                    "n_paired_experiments": len(delta), "median_overlap_difference": med,
                    "median_bootstrap_95_low": lo, "median_bootstrap_95_high": hi,
                    "mean_overlap_difference": mean, "mean_bootstrap_95_low": mean_lo,
                    "mean_bootstrap_95_high": mean_hi,
                })
    pd.DataFrame(contrast_rows).to_csv(OUT / "shortlist_overlap_reference_contrasts.csv", index=False)



def write_joint_purge_checksums():
    """Write SHA256 digests for the published joint-purge aggregate tables."""
    lines = []
    for name in JOINT_OUTPUT_NAMES:
        path = OUT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required joint-purge result table is missing: {path}")
        lines.append(f"{ap.sha256_file(path)}  {name}")
    JOINT_OUTPUT_CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_joint_purge_checksums():
    """Read the SHA256 list and require exactly the expected aggregate result files."""
    if not JOINT_OUTPUT_CHECKSUMS.is_file():
        raise FileNotFoundError("Aggregate result checksum file is missing")
    checksums = {}
    for line in JOINT_OUTPUT_CHECKSUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("Malformed SHA256 checksum entry")
        digest, name = parts
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError(f"Malformed SHA256 digest for {name}")
        if name in checksums:
            raise ValueError(f"Duplicate checksum entry for {name}")
        checksums[name] = digest.lower()
    if set(checksums) != set(JOINT_OUTPUT_NAMES):
        raise ValueError("Checksum file does not list exactly the required joint-purge tables")
    return checksums


def joint_purge_outputs_are_valid(d):
    """Check joint-purge aggregate table integrity and consistency with the cohort."""
    paths = {
        "by_experiment": OUT / "combined_purge_by_experiment.csv",
        "pooled": OUT / "combined_purge_performance.csv",
        "experiment_performance": OUT / "combined_purge_experiment_performance.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        return False, "one or more joint-purge result tables are missing"
    try:
        expected_hashes = read_joint_purge_checksums()
        for name in JOINT_OUTPUT_NAMES:
            path = OUT / name
            if expected_hashes[name] != ap.sha256_file(path):
                return False, f"joint-purge result table failed its SHA256 check: {name}"

        expected = d.groupby("experiment_id").agg(n=("y", "size"), events=("y", "sum"))
        expected_ids = set(expected.index)
        expected_n = int(expected.n.sum())
        expected_events = int(expected.events.sum())
        expected_eligible = int(((expected.n >= 50) & (expected.events >= 10) &
                                 ((expected.n - expected.events) >= 10)).sum())

        by_experiment = pd.read_csv(paths["by_experiment"])
        pooled = pd.read_csv(paths["pooled"])
        experiment_performance = pd.read_csv(paths["experiment_performance"])
        if len(by_experiment) != len(expected_ids) or set(by_experiment.experiment_id) != expected_ids:
            return False, "joint-purge experiment table does not cover the reconstructed cohort"
        if len(pooled) != 2 or set(pooled.model) != {"Logistic regression", "LightGBM"}:
            return False, "joint-purge pooled table has an unexpected model set"
        expected_validation = "Learner + class + teacher-purged LOEO"
        if set(pooled.validation) != {expected_validation}:
            return False, "joint-purge pooled table has an unexpected validation label"
        if not (pooled.n.astype(int) == expected_n).all():
            return False, "joint-purge pooled sample sizes do not match the reconstructed cohort"
        if not np.allclose(pooled.prevalence.to_numpy(dtype=float), expected_events / expected_n, atol=1e-12):
            return False, "joint-purge pooled prevalence does not match the reconstructed cohort"
        if len(experiment_performance) != 2 * len(expected_ids):
            return False, "joint-purge experiment-performance table has an unexpected row count"
        if set(experiment_performance.experiment_id) != expected_ids:
            return False, "joint-purge experiment-performance table has a different experiment set"
        if set(experiment_performance.model) != {"Logistic regression", "LightGBM"}:
            return False, "joint-purge experiment-performance table has an unexpected model set"
        if set(experiment_performance.validation) != {expected_validation}:
            return False, "joint-purge experiment-performance table has an unexpected validation label"
        counts = experiment_performance.groupby("experiment_id").model.nunique()
        if not (counts == 2).all():
            return False, "joint-purge table does not contain both models for every experiment"
        observed = experiment_performance.drop_duplicates("experiment_id").set_index("experiment_id")
        observed = observed.loc[expected.index]
        if not (observed.n.astype(int).to_numpy() == expected.n.astype(int).to_numpy()).all():
            return False, "joint-purge experiment sample sizes do not match the reconstructed cohort"
        if not (observed.events.astype(int).to_numpy() == expected.events.astype(int).to_numpy()).all():
            return False, "joint-purge event counts do not match the reconstructed cohort"
        expected_eligible_flag = (observed.n >= 50) & (observed.events >= 10) & ((observed.n - observed.events) >= 10)
        if not (observed.eligible_auroc.astype(bool).to_numpy() == expected_eligible_flag.to_numpy()).all():
            return False, "joint-purge AUROC eligibility flags do not match the reconstructed cohort"
        eligible_counts = experiment_performance[experiment_performance.eligible_auroc].groupby("model").experiment_id.nunique()
        if set(eligible_counts.index) != {"Logistic regression", "LightGBM"} or not (eligible_counts == expected_eligible).all():
            return False, "joint-purge experiment-performance table has unexpected AUROC eligibility counts"
        by_index = by_experiment.set_index("experiment_id").loc[expected.index]
        if not (by_index.test_n.astype(int).to_numpy() == expected.n.astype(int).to_numpy()).all():
            return False, "joint-purge test sample sizes do not match the reconstructed cohort"
        return True, "joint-purge aggregate tables passed SHA256, cohort, eligibility and model checks"
    except Exception as exc:
        return False, f"joint-purge aggregate validation failed: {exc}"



def main():
    ap.DIAGNOSTIC_STAGE = "additional_analyses.py"
    print("Checking raw archive and reconstructing evaluable cohort...", flush=True)
    full, d = load_dataset()
    predwork = load_prediction_workfile(d)
    purged = {
        "Logistic regression": predwork["learner_purged_loeo__logistic_regression"].to_numpy(),
        "LightGBM": predwork["learner_purged_loeo__lightgbm"].to_numpy(),
    }

    print("Estimating paired model-gap uncertainty...", flush=True)
    # Validate the supplied aggregate joint-purge tables before reuse. If they are
    # missing or inconsistent, reconstruct the predictors and refit the analysis.
    expperf = pd.read_csv(OUT / "experiment_level_performance.csv")
    combperf_path = OUT / "combined_purge_experiment_performance.csv"
    outputs_ok, outputs_message = joint_purge_outputs_are_valid(d)
    force_joint_refit = os.environ.get("TKL_RECOMPUTE_JOINT_PURGE", "0").strip().lower() in {"1", "true", "yes"}
    if force_joint_refit:
        print("A fresh joint-purge fit was explicitly requested.", flush=True)
    elif outputs_ok:
        print("Using validated joint-purge aggregate tables:", outputs_message, flush=True)
    else:
        print("Joint-purge aggregate tables did not pass validation:", outputs_message, flush=True)
    if force_joint_refit or not outputs_ok:
        print("Fitting joint learner/class/teacher-purged LOEO models...", flush=True)
        fit_data = d
        required_features = set(ap.FEATURE_SETS["Full transportable"])
        if not required_features.issubset(fit_data.columns):
            # When a validated prediction workfile is available, load_dataset() returns
            # only its keys and outcomes to avoid retaining a second copy of the feature
            # matrix. A validation failure or an explicit refit still needs the full predictors.
            print("Reconstructing the full feature table for joint-purge refitting...", flush=True)
            alogs, priors, slog, _ = ap.load_raw(ap.RAW)
            full_for_refit = ap.derive_dataset(alogs, priors, slog)
            fit_data = full_for_refit[full_for_refit.evaluable14].copy().reset_index(drop=True)
            fit_data["y"] = fit_data.y14.astype(int)
            pd.testing.assert_frame_equal(
                fit_data[KEY_COLUMNS].reset_index(drop=True),
                d[KEY_COLUMNS].reset_index(drop=True),
                check_dtype=False,
                check_names=True,
            )
        joint_purge(fit_data)
        outputs_ok, outputs_message = joint_purge_outputs_are_valid(d)
        if not outputs_ok:
            raise RuntimeError(f"Joint-purge outputs failed post-fit validation: {outputs_message}")
    combperf = pd.read_csv(combperf_path)
    paired_model_delta_table(expperf, combperf)
    paired_validation_model_contrasts(expperf, combperf)

    lbfgs = run_lbfgs_sensitivity(d, predwork)
    run_feature_uncertainty(d, predwork)

    prediction_map = {
        "Logistic regression (SGD)": purged["Logistic regression"],
        "Logistic regression (LBFGS)": lbfgs["loeo"],
        "LightGBM": purged["LightGBM"],
    }
    run_capacity_uncertainty(d, prediction_map)
    recalibration_predictions = {
        "Logistic regression": purged["Logistic regression"],
        "LightGBM": purged["LightGBM"],
        "Logistic regression (LBFGS)": lbfgs["loeo"],
    }
    run_local_prevalence_analysis(d, recalibration_predictions)

    print("Assessing shortlist stability...", flush=True)
    overlap_map = dict(prediction_map)
    overlap_map["LightGBM seed 20260807"] = purged["LightGBM"]
    overlap_map["LightGBM seed 20260808"] = predwork["learner_purged_loeo__lightgbm_seed_20260808"].to_numpy()
    shortlist_metrics(d, overlap_map)

    ap.write_convergence_diagnostics(append=True)
    print("Additional analyses complete. Aggregate outputs are in", OUT, flush=True)


if __name__ == "__main__":
    main()
