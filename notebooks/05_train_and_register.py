# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Model training, tuning, evaluation, explainability, fairness, registration
# MAGIC
# MAGIC Flow: point-in-time training set (Feature Engineering client) → temporal split → baseline (logistic regression)
# MAGIC → LightGBM + Optuna HPO (nested MLflow runs) → threshold by care-team capacity → test metrics, calibration,
# MAGIC SHAP, fairness (Fairlearn) → log + register to **Unity Catalog** → alias `Challenger`.

# COMMAND ----------
# MAGIC %pip install optuna fairlearn shap lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

import json, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
import mlflow, optuna, shap
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from databricks.feature_engineering import FeatureEngineeringClient
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
                             precision_recall_curve, precision_score, confusion_matrix)
from sklearn.calibration import calibration_curve
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

from poc_lib.features import ID_COLS, LABEL, CATEGORICAL_ALL, PHI_DENYLIST, SENSITIVE_NOT_FOR_MODEL, get_feature_lookups
from poc_lib.metrics import precision_recall_at_k, expected_calibration_error, age_band

mlflow.autolog(disable=True)           # we log explicitly -> full control, no surprises
optuna.logging.set_verbosity(optuna.logging.WARNING)
fe = FeatureEngineeringClient()
client = MlflowClient()
SEED = 42

# COMMAND ----------
# MAGIC %md ## 1. Build the point-in-time training set
# MAGIC Two views of the same data: `ts_log` (exact model input, ids excluded - this object is what gets packaged with the model)
# MAGIC and `ts_full` (keeps ids so we can split by time and join audit attributes).

# COMMAND ----------

base = spark.table(T["training_base"])
lookups = get_feature_lookups(FQ)
ts_log = fe.create_training_set(df=base, feature_lookups=lookups, label=LABEL, exclude_columns=ID_COLS)
ts_full = fe.create_training_set(df=base, feature_lookups=lookups, label=LABEL, exclude_columns=[])
feature_cols = [c for c in ts_log.load_df().columns if c != LABEL]
assert not set(feature_cols) & set(PHI_DENYLIST), "PHI column leaked into model features"
assert not set(feature_cols) & set(SENSITIVE_NOT_FOR_MODEL), "sensitive attribute used as predictor"

pdf = ts_full.load_df().toPandas()                      # ~1M rows x ~35 cols; needs a >=32GB driver (or sample for dev)
aud = spark.table(T["silver_patients"]).select("patient_id", "race_ethnicity").toPandas()
pdf = pdf.merge(aud, on="patient_id", how="left")
pdf["admit_ts"] = pd.to_datetime(pdf["admit_ts"])
pdf["age_band"] = pdf["age_at_admit"].apply(age_band)
print(f"rows={len(pdf):,}  features={len(feature_cols)}")

cat_cols = [c for c in feature_cols if c in CATEGORICAL_ALL]
num_cols = [c for c in feature_cols if c not in CATEGORICAL_ALL]
for c in cat_cols:                       # must happen BEFORE slicing into train/val/test
    pdf[c] = pdf[c].astype(str)
for c in num_cols:
    pdf[c] = pdf[c].astype(float)

tr = pdf[pdf.admit_ts < TRAIN_END]
va = pdf[(pdf.admit_ts >= TRAIN_END) & (pdf.admit_ts < VAL_END)]
te = pdf[pdf.admit_ts >= VAL_END]
X_tr, y_tr, X_va, y_va, X_te, y_te = (tr[feature_cols], tr[LABEL].values, va[feature_cols], va[LABEL].values,
                                      te[feature_cols], te[LABEL].values)
print(f"train={len(tr):,} val={len(va):,} test={len(te):,}  prevalence tr/va/te = {y_tr.mean():.3f}/{y_va.mean():.3f}/{y_te.mean():.3f}")

def make_pre(scale: bool):
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(num), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
                             verbose_feature_names_out=False)

def metrics_for(y, p, prefix):
    k = precision_recall_at_k(y, p, 0.10)
    return {f"{prefix}_auroc": roc_auc_score(y, p), f"{prefix}_auprc": average_precision_score(y, p),
            f"{prefix}_brier": brier_score_loss(y, p), f"{prefix}_ece": expected_calibration_error(y, p),
            f"{prefix}_precision_top10": k["precision"], f"{prefix}_recall_top10": k["recall"],
            f"{prefix}_lift_top10": k["lift"], f"{prefix}_prevalence": float(np.mean(y))}

# COMMAND ----------
# MAGIC %md ## 2. Baseline: regularised logistic regression
# MAGIC Every ML project needs a simple, interpretable benchmark. If LightGBM can't beat it clearly, ship the simple model.

# COMMAND ----------

with mlflow.start_run(run_name="baseline_logreg"):
    pre_lr = make_pre(scale=True)
    lr = Pipeline([("pre", pre_lr), ("clf", LogisticRegression(max_iter=300, C=0.5, n_jobs=-1))]).fit(X_tr, y_tr)
    m = metrics_for(y_va, lr.predict_proba(X_va)[:, 1], "val")
    mlflow.log_metrics(m); mlflow.set_tags({"model_family": "logreg", "stage": "baseline"})
    print({k: round(v, 4) for k, v in m.items()})
baseline_auprc = m["val_auprc"]

# COMMAND ----------
# MAGIC %md ## 3. LightGBM + Optuna hyper-parameter search (each trial = nested MLflow run)
# MAGIC Objective = validation **AUPRC** (right metric for a ~15% positive class). Early stopping on the validation fold.
# MAGIC `pos_weight` > 1 trades calibration for recall - we check calibration afterwards.

# COMMAND ----------

pre = make_pre(scale=False)
Xtr_t = pre.fit_transform(X_tr); Xva_t = pre.transform(X_va); Xte_t = pre.transform(X_te)

N_TRIALS = 20   # raise to 50-100 for production tuning

def objective(trial):
    params = dict(
        n_estimators=2000, learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
        subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1.0, 2.0]),
        n_jobs=-1, random_state=SEED, verbose=-1)
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        clf = LGBMClassifier(**params)
        clf.fit(Xtr_t, y_tr, eval_set=[(Xva_t, y_va)], eval_metric="average_precision",
                callbacks=[early_stopping(50, verbose=False)])
        p = clf.predict_proba(Xva_t)[:, 1]
        ap, auc = average_precision_score(y_va, p), roc_auc_score(y_va, p)
        mlflow.log_params({k: v for k, v in params.items() if k not in ("n_jobs", "verbose")})
        mlflow.log_metrics({"val_auprc": ap, "val_auroc": auc, "best_iteration": clf.best_iteration_})
    return ap

with mlflow.start_run(run_name="lgbm_hpo_and_final") as parent:
    run_id = parent.info.run_id
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best = {k: v for k, v in study.best_params.items()}
    print("best val AUPRC:", round(study.best_value, 4), best)

    # ---- final fit with best params
    final = LGBMClassifier(n_estimators=2000, subsample_freq=1, n_jobs=-1, random_state=SEED, verbose=-1, **best)
    final.fit(Xtr_t, y_tr, eval_set=[(Xva_t, y_va)], eval_metric="average_precision",
              callbacks=[early_stopping(50, verbose=False)])
    pipe = Pipeline([("pre", pre), ("clf", final)])      # fitted components assembled into one deployable object

    p_va = pipe.predict_proba(X_va)[:, 1]
    p_te = pipe.predict_proba(X_te)[:, 1]

    # ---- operating threshold from care-team capacity: flag the top 10% as HIGH, next 20% as MEDIUM
    thr_high = float(np.quantile(p_va, 0.90))
    thr_med = float(np.quantile(p_va, 0.70))

    all_metrics = {**metrics_for(y_va, p_va, "val"), **metrics_for(y_te, p_te, "test")}
    all_metrics["test_auprc_gain_vs_baseline"] = all_metrics["test_auprc"] - baseline_auprc
    mlflow.log_params({**best, "n_trials": N_TRIALS, "best_iteration": final.best_iteration_,
                       "decision_threshold": thr_high, "medium_threshold": thr_med,
                       "train_end": TRAIN_END, "val_end": VAL_END, "n_features": len(feature_cols)})
    mlflow.log_metrics(all_metrics)
    print({k: round(v, 4) for k, v in all_metrics.items() if k.startswith("test")})

    # ---- plots: ROC, PR, calibration, lift
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    fpr, tpr, _ = roc_curve(y_te, p_te); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--"); ax[0].set_title(f"ROC AUC={all_metrics['test_auroc']:.3f}")
    pr, rc, _ = precision_recall_curve(y_te, p_te); ax[1].plot(rc, pr); ax[1].axhline(y_te.mean(), ls="--"); ax[1].set_title(f"PR AUC={all_metrics['test_auprc']:.3f}")
    fp, mp = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile"); ax[2].plot(mp, fp, marker="o"); ax[2].plot([0, 1], [0, 1], "--"); ax[2].set_title(f"Calibration ECE={all_metrics['test_ece']:.3f}")
    dec = pd.DataFrame({"y": y_te, "p": p_te}); dec["decile"] = pd.qcut(dec.p, 10, labels=False, duplicates="drop")
    ax[3].bar(dec.groupby("decile").y.mean().index, dec.groupby("decile").y.mean().values); ax[3].set_title("Observed readmit rate by risk decile")
    plt.tight_layout(); mlflow.log_figure(fig, "evaluation/curves.png"); plt.show()

    # ---- SHAP explainability (global + one local example)
    names = pre.get_feature_names_out()
    samp = pd.DataFrame(Xte_t[np.random.RandomState(0).choice(len(Xte_t), 4000, replace=False)], columns=names)
    sv = shap.TreeExplainer(final).shap_values(samp)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(); shap.summary_plot(sv, samp, show=False, max_display=20); plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "explainability/shap_summary.png"); plt.show()
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    mlflow.log_table(imp, "explainability/shap_importance.json"); display(imp.head(15))

    # ---- fairness audit (Fairlearn) on the test period at the operating threshold
    te2 = te.copy(); te2["pred"] = (p_te >= thr_high).astype(int)
    fair = {}
    for attr in ["gender", "race_ethnicity", "insurance_type", "age_band"]:
        mf = MetricFrame(metrics={"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate,
                                  "precision": lambda y, p: precision_score(y, p, zero_division=0)},
                         y_true=te2[LABEL], y_pred=te2["pred"], sensitive_features=te2[attr])
        d = mf.difference()
        fair[attr] = {"tpr_gap": float(d["tpr"]), "selection_rate_gap": float(d["selection_rate"]), "fpr_gap": float(d["fpr"])}
        mlflow.log_table(mf.by_group.reset_index(), f"fairness/by_{attr}.json")
        mlflow.log_metric(f"fair_tpr_gap_{attr}", fair[attr]["tpr_gap"])
    mlflow.log_dict(fair, "fairness/summary.json"); print(json.dumps(fair, indent=1))

    # ---- lineage + reproducibility tags
    delta_version = spark.sql(f"DESCRIBE HISTORY {T['training_base']} LIMIT 1").first()["version"]
    mlflow.log_input(mlflow.data.load_delta(table_name=T["training_base"], version=str(delta_version)), context="training")
    mlflow.set_tags({"git_sha": GIT_SHA, "env": ENV, "training_base_delta_version": str(delta_version),
                     "model_family": "lightgbm", "use_case": "30d_readmission", "pii": "none-in-features"})

    # ---- (A) model packaged WITH feature-lookup metadata -> batch scoring with point-in-time joins
    fe.log_model(model=pipe, artifact_path="model", flavor=mlflow.sklearn, training_set=ts_log,
                 registered_model_name=MODEL_NAME, pyfunc_predict_fn="predict_proba")

    # ---- (B) self-contained copy (request carries all features) -> low-latency real-time serving
    sig = infer_signature(X_tr.head(200), pipe.predict_proba(X_tr.head(200)))
    mlflow.sklearn.log_model(pipe, artifact_path="payload_model", signature=sig, input_example=X_tr.head(3),
                             registered_model_name=PAYLOAD_MODEL_NAME, pyfunc_predict_fn="predict_proba")

# COMMAND ----------
# MAGIC %md ## 4. Registry: description, tags, alias `Challenger` (promotion to `Champion` happens in notebook 06)

# COMMAND ----------

def version_for_run(name, rid):
    return max((v for v in client.search_model_versions(f"name='{name}'") if v.run_id == rid), key=lambda v: int(v.version))

for name in (MODEL_NAME, PAYLOAD_MODEL_NAME):
    mv = version_for_run(name, run_id)
    client.update_model_version(name, mv.version, description=(
        f"LightGBM 30-day readmission risk. Trained on encounters < {TRAIN_END}, validated < {VAL_END}. "
        f"Test AUROC={all_metrics['test_auroc']:.3f}, AUPRC={all_metrics['test_auprc']:.3f}, "
        f"precision@top10%={all_metrics['test_precision_top10']:.3f}. Not for diagnosis; decision-support only."))
    for k, v in {"decision_threshold": thr_high, "medium_threshold": thr_med, "validation_status": "PENDING",
                 "git_sha": GIT_SHA, "test_auprc": round(all_metrics["test_auprc"], 4),
                 "feature_columns": json.dumps(feature_cols)}.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    client.set_registered_model_alias(name, "Challenger", mv.version)
    print(f"{name} v{mv.version} -> alias Challenger")

dbutils.jobs.taskValues.set("run_id", run_id) if "dbutils" in globals() else None

# COMMAND ----------
# MAGIC %md ## 5. (Optional) AutoML benchmark
# MAGIC Quick sanity check that hand-tuned LightGBM isn't leaving large gains on the table. Generates editable notebooks.

# COMMAND ----------

# from databricks import automl
# summary = automl.classify(dataset=spark.table(T["training_base"]).drop(*ID_COLS), target_col=LABEL,
#                           primary_metric="log_loss", timeout_minutes=20)
# print(summary.best_trial.metrics)
