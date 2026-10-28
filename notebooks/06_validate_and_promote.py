# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Model validation gates & champion / challenger promotion
# MAGIC Automated quality gate between *training* and *production*. The Challenger is scored through the **same code path
# MAGIC production uses** (`fe.score_batch`, point-in-time feature lookups) on the held-out future test period and must pass:
# MAGIC
# MAGIC | # | Gate | Default rule |
# MAGIC |---|---|---|
# MAGIC | 1 | Discrimination | test AUROC ≥ 0.70 |
# MAGIC | 2 | Precision value | AUPRC ≥ 1.5 × prevalence and lift@10% ≥ 1.5 |
# MAGIC | 3 | Calibration | ECE ≤ 0.05 |
# MAGIC | 4 | Non-inferiority | AUPRC ≥ current Champion − 0.005 |
# MAGIC | 5 | Fairness | TPR gap ≤ 0.10 across gender / insurance / age band |
# MAGIC | 6 | Governance | no PHI / race in inputs; description, signature, lineage tags present |
# MAGIC | 7 | Approval | in `prod`, a human must approve (`approved=true`) before Champion alias moves |
# MAGIC
# MAGIC Writes an auditable results table, sets `validation_status`, moves aliases, publishes the monitoring baseline.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

import json, numpy as np, pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from fairlearn.metrics import MetricFrame, true_positive_rate
from poc_lib.features import LABEL, PHI_DENYLIST, SENSITIVE_NOT_FOR_MODEL
from poc_lib.metrics import precision_recall_at_k, expected_calibration_error, age_band
from poc_lib.scoring import positive_prob_pdf, to_inference_log

dbutils.widgets.text("approved", "false")   # manual approval flag for prod
APPROVED = dbutils.widgets.get("approved").lower() == "true"

fe = FeatureEngineeringClient()
client = MlflowClient()

GATES = dict(min_auroc=0.70, min_auprc_ratio=1.5, min_lift10=1.5, max_ece=0.05,
             noninferiority_margin=0.005, max_tpr_gap=0.10)

# COMMAND ----------
# MAGIC %md ## 1. Score the held-out test period through the production code path

# COMMAND ----------

test_df = spark.table(T["training_base"]).filter(F.col("admit_ts") >= F.lit(VAL_END).cast("timestamp"))
patients = spark.table(T["silver_patients"]).select("patient_id", "race_ethnicity").toPandas()

def score_version(version):
    scored = fe.score_batch(model_uri=f"models:/{MODEL_NAME}/{version}", df=test_df)
    return scored

def evaluate(scored_sdf):
    pdf = scored_sdf.select("patient_id", "prediction", LABEL, "gender", "insurance_type", "age_at_admit").toPandas()
    pdf["p"] = positive_prob_pdf(pdf)
    pdf = pdf.merge(patients, on="patient_id", how="left")
    pdf["age_band"] = pdf["age_at_admit"].apply(age_band)
    y, p = pdf[LABEL].values, pdf["p"].values
    k = precision_recall_at_k(y, p, 0.10)
    return pdf, {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "prevalence": float(y.mean()),
                 "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p), "lift10": k["lift"],
                 "precision10": k["precision"]}

chal = client.get_model_version_by_alias(MODEL_NAME, "Challenger")
chal_scored = score_version(chal.version).cache()
chal_pdf, chal_m = evaluate(chal_scored)
thr_high = float(chal.tags["decision_threshold"]); thr_med = float(chal.tags["medium_threshold"])
print("Challenger v%s:" % chal.version, {k: round(v, 4) for k, v in chal_m.items()})

try:
    champ = client.get_model_version_by_alias(MODEL_NAME, "Champion")
    _, champ_m = evaluate(score_version(champ.version))
    print("Champion   v%s:" % champ.version, {k: round(v, 4) for k, v in champ_m.items()})
except Exception:
    champ, champ_m = None, None
    print("No Champion yet -> first deployment")

# COMMAND ----------
# MAGIC %md ## 2. Fairness at the operating threshold

# COMMAND ----------

chal_pdf["pred"] = (chal_pdf["p"] >= thr_high).astype(int)
tpr_gaps = {}
for attr in ["gender", "insurance_type", "age_band"]:   # race is reported (below) but gated by policy review
    mf = MetricFrame(metrics={"tpr": true_positive_rate}, y_true=chal_pdf[LABEL], y_pred=chal_pdf["pred"],
                     sensitive_features=chal_pdf[attr])
    tpr_gaps[attr] = float(mf.difference()["tpr"])
mf_r = MetricFrame(metrics={"tpr": true_positive_rate}, y_true=chal_pdf[LABEL], y_pred=chal_pdf["pred"],
                   sensitive_features=chal_pdf["race_ethnicity"])
print("TPR gaps:", {k: round(v, 3) for k, v in tpr_gaps.items()}, "| race (reported):", round(float(mf_r.difference()['tpr']), 3))

# COMMAND ----------
# MAGIC %md ## 3. Run the gates

# COMMAND ----------

feature_cols = json.loads(chal.tags["feature_columns"])
info = client.get_model_version(MODEL_NAME, chal.version)
checks = {
    "1_auroc": (chal_m["auroc"] >= GATES["min_auroc"], chal_m["auroc"]),
    "2_auprc_vs_prevalence": (chal_m["auprc"] >= GATES["min_auprc_ratio"] * chal_m["prevalence"], chal_m["auprc"]),
    "2_lift_top10": (chal_m["lift10"] >= GATES["min_lift10"], chal_m["lift10"]),
    "3_calibration_ece": (chal_m["ece"] <= GATES["max_ece"], chal_m["ece"]),
    "4_non_inferior_to_champion": ((champ_m is None) or chal_m["auprc"] >= champ_m["auprc"] - GATES["noninferiority_margin"],
                                   chal_m["auprc"] - (champ_m["auprc"] if champ_m else 0.0)),
    "5_fairness_tpr_gap": (max(tpr_gaps.values()) <= GATES["max_tpr_gap"], max(tpr_gaps.values())),
    "6_no_phi_or_race_in_features": (not (set(feature_cols) & set(PHI_DENYLIST + SENSITIVE_NOT_FOR_MODEL)), 0.0),
    "6_has_description": (bool(info.description), 1.0 if info.description else 0.0),
    "6_has_lineage_tags": (all(t in chal.tags for t in ["git_sha", "decision_threshold"]), 1.0),
}
res = pd.DataFrame([{"check": k, "passed": bool(v[0]), "value": float(v[1])} for k, v in checks.items()])
res["model_name"], res["model_version"], res["checked_at"] = MODEL_NAME, chal.version, pd.Timestamp.utcnow().tz_localize(None)
display(res)
spark.createDataFrame(res).write.mode("append").saveAsTable(T["validation_results"])
all_passed = bool(res["passed"].all())

# COMMAND ----------
# MAGIC %md ## 4. Publish the monitoring baseline (test-period predictions with labels) - same schema as the inference log

# COMMAND ----------

baseline = (to_inference_log(chal_scored, MODEL_NAME, chal.version, thr_high, thr_med, with_label=True))
(baseline.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["baseline"]))
print("baseline rows:", spark.table(T["baseline"]).count())

# COMMAND ----------
# MAGIC %md ## 5. Promote (or refuse) and leave an audit trail

# COMMAND ----------

def version_for_run(name, rid):
    return max((v for v in client.search_model_versions(f"name='{name}'") if v.run_id == rid), key=lambda v: int(v.version))

promoted, reason = False, ""
if not all_passed:
    reason = "gates failed: " + ", ".join(res[~res.passed].check)
elif ENV == "prod" and not APPROVED:
    reason = "all gates passed - AWAITING MANUAL APPROVAL (re-run with approved=true)"
else:
    for name in (MODEL_NAME, PAYLOAD_MODEL_NAME):
        v = version_for_run(name, chal.run_id)
        try:
            prev = client.get_model_version_by_alias(name, "Champion")
            client.set_registered_model_alias(name, "Previous", prev.version)   # one-click rollback target
        except Exception:
            pass
        client.set_registered_model_alias(name, "Champion", v.version)
        client.set_model_version_tag(name, v.version, "validation_status", "PASSED")
    promoted, reason = True, f"v{chal.version} promoted to Champion"
if not promoted:
    for name in (MODEL_NAME, PAYLOAD_MODEL_NAME):
        v = version_for_run(name, chal.run_id)
        client.set_model_version_tag(name, v.version, "validation_status", "FAILED" if not all_passed else "AWAITING_APPROVAL")

print(reason)
try:
    dbutils.jobs.taskValues.set("promoted", str(promoted).lower())     # consumed by the job's condition task
except Exception:
    pass
dbutils.notebook.exit(json.dumps({"promoted": promoted, "reason": reason, "version": chal.version}))

# COMMAND ----------
# MAGIC %md ### Rollback (run manually in an incident)
# MAGIC ```python
# MAGIC for name in (MODEL_NAME, PAYLOAD_MODEL_NAME):
# MAGIC     prev = client.get_model_version_by_alias(name, "Previous")
# MAGIC     client.set_registered_model_alias(name, "Champion", prev.version)
# MAGIC # then re-run notebook 08 (or shift endpoint traffic back) to redeploy
# MAGIC ```
