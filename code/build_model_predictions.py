#!/usr/bin/env python3
"""Construct the out-of-fold predictions required by the additional analyses.

Run after the main analysis pipeline and before additional_analyses.py.
Requires the original ASSISTments archive at TKL_RAW_ZIP or data/.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import analysis_pipeline as ap

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "analysis_work"
OUT = ROOT / "results"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

KEY_COLUMNS = ["experiment_id", "student_id", "class_id", "teacher_id", "start", "y"]


def prediction_content_sha256(frame, key_columns, prediction_columns):
    """Fingerprint row keys, outcomes and every generated probability column."""
    payload = frame[key_columns + prediction_columns].copy()
    hashed = pd.util.hash_pandas_object(payload, index=False).to_numpy(dtype="uint64").tobytes()
    return hashlib.sha256(hashed).hexdigest()


def model_configuration():
    """Return the fixed estimator and feature specification used for prediction generation."""
    estimators = {
        "sgd": ap.make_model("Logistic regression").get_params(deep=True),
        "lbfgs": ap.make_model("Logistic regression (LBFGS)").get_params(deep=True),
        "lightgbm": ap.make_model("LightGBM").get_params(deep=True),
        "lightgbm_alternate_seed": ap.make_model("LightGBM seed 20260808").get_params(deep=True),
    }
    payload = {
        "features": ap.FEATURE_SETS["Full transportable"],
        "feature_sets": {
            "No context": ap.FEATURE_SETS["No context"],
            "Complete learner history": ap.FEATURE_SETS["Complete learner history"],
        },
        "estimators": estimators,
        "seed": ap.SEED,
    }
    return payload




def write_predictor_comparison_tables(d, prediction_workfile):
    """Write pooled summaries for the learner-history and context-ablation fits."""
    feature_rows = []
    for feature_set, prediction in (
        ("Complete learner history", prediction_workfile["learner_purged_loeo__logistic_regression_history"].to_numpy()),
        ("Full transportable", prediction_workfile["learner_purged_loeo__logistic_regression"].to_numpy()),
    ):
        feature_rows.append({
            "validation": "Learner-purged LOEO", "model": "Logistic regression",
            "feature_set": feature_set, **ap.metrics(d.y.to_numpy(), prediction),
            "n": len(d), "prevalence": float(d.y.mean()),
        })
    pd.DataFrame(feature_rows).to_csv(OUT / "feature_blocks_loeo.csv", index=False)

    context_rows = []
    for feature_set, prediction in (
        ("Full transportable", prediction_workfile["learner_purged_loeo__lightgbm"].to_numpy()),
        ("No context", prediction_workfile["learner_purged_loeo__lightgbm_no_context"].to_numpy()),
    ):
        context_rows.append({
            "validation": "Learner-purged LOEO", "model": "LightGBM",
            "feature_set": feature_set, **ap.metrics(d.y.to_numpy(), prediction),
            "n": len(d), "prevalence": float(d.y.mean()),
        })
    pd.DataFrame(context_rows).to_csv(OUT / "lightgbm_context_loeo.csv", index=False)


def main():
    print("Verify raw archive and reconstruct cohort", flush=True)
    ap.verify_raw_archive(ap.RAW)
    alogs, priors, slog, _ = ap.load_raw(ap.RAW)
    full = ap.derive_dataset(alogs, priors, slog)
    d = full[full.evaluable14].copy().reset_index(drop=True)
    d["y"] = d.y14.astype(int)
    for col in ["experiment_id", "student_id", "class_id", "teacher_id"]:
        assert d[col].notna().all(), f"Missing grouping identifier: {col}"
    cohort = d.groupby("experiment_id").agg(
        n=("y", "size"), events=("y", "sum"), prevalence=("y", "mean"),
        learners=("student_id", "nunique"), classes=("class_id", "nunique"),
        teachers=("teacher_id", "nunique")
    ).reset_index()
    cohort.to_csv(OUT / "experiment_cohort.csv", index=False)
    print(f"Primary cohort: {len(d):,} records; {d.y.sum():,} events; {d.experiment_id.nunique()} experiments", flush=True)

    print("Fit row-random full-feature models (SGD, LBFGS, LightGBM)", flush=True)
    row_full = ap.run_kfold(
        d, "Row random", ap.FEATURE_SETS["Full transportable"],
        model_names=("Logistic regression", "Logistic regression (LBFGS)", "LightGBM")
    )
    print("Fit row-random LightGBM without context", flush=True)
    row_noctx = ap.run_kfold(
        d, "Row random", ap.FEATURE_SETS["No context"], model_names=("LightGBM",)
    )["LightGBM"]

    print("Fit learner-purged LOEO full-feature models (SGD, LBFGS, LightGBM, alternate seed)", flush=True)
    loeo_full, purge_stats = ap.run_loeo(
        d, "Learner-purged LOEO", ap.FEATURE_SETS["Full transportable"],
        model_names=("Logistic regression", "Logistic regression (LBFGS)",
                     "LightGBM", "LightGBM seed 20260808")
    )
    purge_stats.to_csv(OUT / "purge_by_experiment.csv", index=False)

    print("Fit learner-purged LOEO LightGBM without context", flush=True)
    loeo_noctx = ap.run_loeo(
        d, "Learner-purged LOEO", ap.FEATURE_SETS["No context"],
        model_names=("LightGBM",)
    )[0]["LightGBM"]

    print("Fit learner-purged LOEO logistic regression with learner-history predictors only", flush=True)
    loeo_history = ap.run_loeo(
        d, "Learner-purged LOEO", ap.FEATURE_SETS["Complete learner history"],
        model_names=("Logistic regression",)
    )[0]["Logistic regression"]

    pwork = d[["experiment_id", "student_id", "class_id", "teacher_id", "start", "y"]].copy()
    pwork["row_random__logistic_regression"] = row_full["Logistic regression"]
    pwork["row_random__logistic_regression_lbfgs"] = row_full["Logistic regression (LBFGS)"]
    pwork["row_random__lightgbm"] = row_full["LightGBM"]
    pwork["row_random__lightgbm_no_context"] = row_noctx
    pwork["learner_purged_loeo__logistic_regression"] = loeo_full["Logistic regression"]
    pwork["learner_purged_loeo__logistic_regression_lbfgs"] = loeo_full["Logistic regression (LBFGS)"]
    pwork["learner_purged_loeo__lightgbm"] = loeo_full["LightGBM"]
    pwork["learner_purged_loeo__lightgbm_seed_20260808"] = loeo_full["LightGBM seed 20260808"]
    pwork["learner_purged_loeo__lightgbm_no_context"] = loeo_noctx
    pwork["learner_purged_loeo__logistic_regression_history"] = loeo_history
    pwork.to_pickle(WORK / "predictions.pkl")
    prediction_columns = [c for c in pwork.columns if c not in KEY_COLUMNS]
    if pwork[KEY_COLUMNS].duplicated().any():
        raise ValueError("The prediction workfile contains duplicate record keys.")
    probabilities = pwork[prediction_columns].to_numpy(dtype=float)
    if not (pd.notna(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()):
        raise ValueError("The prediction workfile contains missing or out-of-range probabilities.")
    key_hash = prediction_content_sha256(pwork, KEY_COLUMNS, [])
    prediction_hash = prediction_content_sha256(pwork, KEY_COLUMNS, prediction_columns)
    builder_path = Path(__file__).resolve()
    pipeline_path = Path(ap.__file__).resolve()
    configuration_json = json.dumps(model_configuration(), sort_keys=True, default=str)
    configuration_hash = hashlib.sha256(configuration_json.encode("utf-8")).hexdigest()
    provenance = {
        "schema_version": 1,
        "generated_by": "build_model_predictions.py",
        "raw_sha256": ap.sha256_file(ap.RAW),
        "n_records": int(len(pwork)),
        "n_experiments": int(pwork.experiment_id.nunique()),
        "events": int(pwork.y.sum()),
        "key_sha256": key_hash,
        "prediction_sha256": prediction_hash,
        "key_columns": KEY_COLUMNS,
        "prediction_columns": prediction_columns,
        "model_configuration_sha256": configuration_hash,
        "analysis_pipeline_sha256": ap.sha256_file(pipeline_path),
        "prediction_builder_sha256": ap.sha256_file(builder_path),
        "pipeline_seed": int(ap.SEED),
    }
    (WORK / "prediction_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Saved {pwork.shape[1]-6} OOF prediction columns for {len(pwork):,} records to {WORK/'predictions.pkl'}", flush=True)

    # Pooled metrics used in the predictor-analysis supplementary tables.
    write_predictor_comparison_tables(d, pwork)


if __name__ == "__main__":
    main()
