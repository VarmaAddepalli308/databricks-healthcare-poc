# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · Production monitoring: data drift, model performance, data quality, operations, alerts
# MAGIC
# MAGIC | Layer | What | How |
# MAGIC |---|---|---|
# MAGIC | Data quality | quarantine rate by rule | `dq_run_metrics` (written by notebook 02) |
# MAGIC | Feature drift | PSI per feature vs training baseline | this notebook → `monitoring_feature_drift` |
# MAGIC | Prediction drift | risk-score distribution & HIGH-tier share | PSI on `risk_score` |
# MAGIC | Performance | AUROC/AUPRC/precision@10% **after labels arrive (+30d)** | label back-fill → `monitoring_performance` |
# MAGIC | Auto-profiling | Databricks **Lakehouse Monitoring** (InferenceLog) → profile + drift tables + dashboard | SDK |
# MAGIC | Ops | latency, errors, volume, cost | endpoint inference table, `system.billing.usage` |
# MAGIC | Alerts / action | warn at PSI>0.1, critical at >0.25 → retrain trigger (notebook 10) | task values + SQL alerts |
# MAGIC
# MAGIC Optional first step: **simulate drifted new data** (a new month with more emergency admissions, longer stays and worse
# MAGIC kidney function) to watch the monitoring fire.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

import json, numpy as np, pandas as pd
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score, average_precision_score
from poc_lib import synth
from poc_lib.metrics import psi, categorical_psi, precision_recall_at_k
from poc_lib.scoring import positive_prob_pdf

dbutils.widgets.dropdown("simulate_drift", "false", ["false", "true"])
dbutils.widgets.text("drift_strength", "1.0")

# COMMAND ----------
# MAGIC %md ## 0. (Optional) Simulate a drifted new month of data and run the pipeline end-to-end
# MAGIC Lands January-2025 data into the **same** landing folders; Auto Loader picks up only the new files.

# COMMAND ----------

if dbutils.widgets.get("simulate_drift") == "true":
    strength = float(dbutils.widgets.get("drift_strength"))
    n_new = max(20_000, N_ENCOUNTERS // 30)
    patients = spark.table(T["silver_patients"]).select("patient_id", F.make_date(F.col("birth_year"), F.lit(6), F.lit(15)).alias("dob"),
                                                         "chronic_conditions_count")
    jan25 = 1735689600   # 2025-01-01 UTC
    full_new = synth.generate_encounters(spark, n_new, patients, N_PATIENTS, start_epoch=jan25, span_days=31,
                                         seed=9000, id_offset=10_000_000, drift=strength)
    full_new.write.mode("overwrite").saveAsTable(f"{FQ}._tmp_new")
    e2, l2, y2 = synth.split_outputs(spark.table(f"{FQ}._tmp_new"))
    e2.write.mode("overwrite").option("header", "true").csv(f"{RAW}/encounters/batch_2025_01")
    l2.write.mode("overwrite").parquet(f"{RAW}/labs/batch_2025_01")
    y2.write.mode("overwrite").parquet(f"{RAW}/labels/batch_2025_01")
    spark.sql(f"DROP TABLE IF EXISTS {FQ}._tmp_new")
    # re-run the pipeline notebooks as a mini-DAG (in production this is the scoring job)
    for nb in ["./02_bronze_silver_gold", "./04_feature_engineering", "./07_batch_scoring"]:
        print("running", nb)
        dbutils.notebook.run(nb, 3600, {"catalog": CATALOG, "schema": SCHEMA, "env": ENV,
                                        "n_encounters": str(N_ENCOUNTERS), "n_patients": str(N_PATIENTS),
                                        "as_of_ts": "2025-04-15 00:00:00"})

# COMMAND ----------
# MAGIC %md ## 1. Feature & prediction drift (PSI) - baseline vs the most recent scoring window

# COMMAND ----------

base = spark.table(T["baseline"])
log = spark.table(T["inference_log"])
latest_ver = log.agg(F.max("model_version")).first()[0]
last_ts = log.agg(F.max("discharge_ts")).first()[0]
recent = log.filter((F.col("model_version") == latest_ver) & (F.col("discharge_ts") >= F.lit(last_ts) - F.expr("INTERVAL 30 DAYS")))

NUM = ["los_days", "num_procedures", "num_medications", "total_charges", "sodium", "creatinine", "hemoglobin",
       "prior_admits_90d", "prior_admits_365d", "prior_ed_admits_365d", "days_since_last_discharge",
       "age_at_admit", "chronic_conditions_count", "charges_per_day", "risk_score"]
CAT = ["admission_type", "discharge_disposition", "dx_group", "insurance_type", "gender"]

b_pdf = base.select(*NUM, *CAT).sample(0.2, seed=1).limit(200_000).toPandas()
r_pdf = recent.select(*NUM, *CAT).toPandas()
print(f"baseline={len(b_pdf):,} recent={len(r_pdf):,} (model v{latest_ver}, window ends {last_ts})")

rows = []
for c in NUM:
    rows.append((c, "numeric", psi(b_pdf[c], r_pdf[c])))
for c in CAT:
    rows.append((c, "categorical", categorical_psi(b_pdf[c], r_pdf[c])))
drift = pd.DataFrame(rows, columns=["feature", "type", "psi"])
drift["status"] = np.where(drift.psi > 0.25, "CRITICAL", np.where(drift.psi > 0.10, "WARNING", "OK"))
drift["model_version"], drift["window_end"], drift["computed_at"] = latest_ver, last_ts, pd.Timestamp.utcnow().tz_localize(None)
display(drift.sort_values("psi", ascending=False))
spark.createDataFrame(drift).write.mode("append").saveAsTable(T["feature_drift"])

n_crit, n_warn = int((drift.status == "CRITICAL").sum()), int((drift.status == "WARNING").sum())
print(f"CRITICAL={n_crit}  WARNING={n_warn}")

# COMMAND ----------
# MAGIC %md ## 2. Ground-truth back-fill & real performance
# MAGIC Labels become available 30 days after discharge. We MERGE matured labels into the inference log, then compute
# MAGIC performance **only on rows with labels** and compare with the offline test metrics (baseline table).

# COMMAND ----------

as_of = F.lit(AS_OF_TS).cast("timestamp")
(spark.sql(f"""
MERGE INTO {T['inference_log']} t
USING (SELECT encounter_id, readmit_30d FROM {T['silver_labels']} WHERE label_available_ts <= TIMESTAMP'{AS_OF_TS}') s
ON t.encounter_id = s.encounter_id AND t.readmit_30d IS NULL
WHEN MATCHED THEN UPDATE SET t.readmit_30d = s.readmit_30d
"""))

lab = (spark.table(T["inference_log"]).filter("readmit_30d IS NOT NULL")
       .select("model_version", "discharge_ts", "risk_score", "readmit_30d", "gender", "insurance_type").toPandas())
perf_rows = []
if len(lab) > 500:
    lab["week"] = pd.to_datetime(lab["discharge_ts"]).dt.to_period("W").dt.start_time
    for (ver, wk), g in lab.groupby(["model_version", "week"]):
        if g.readmit_30d.nunique() < 2 or len(g) < 200:
            continue
        k = precision_recall_at_k(g.readmit_30d.values, g.risk_score.values, 0.10)
        perf_rows.append((ver, wk, len(g), roc_auc_score(g.readmit_30d, g.risk_score),
                          average_precision_score(g.readmit_30d, g.risk_score), k["precision"], k["lift"]))
perf = pd.DataFrame(perf_rows, columns=["model_version", "week", "n_labeled", "auroc", "auprc", "precision_top10", "lift_top10"])
if len(perf):
    display(perf)
    spark.createDataFrame(perf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["perf_metrics"])
base_auroc = base.toPandas().pipe(lambda d: roc_auc_score(d.readmit_30d, d.risk_score))
perf_degraded = bool(len(perf) and (base_auroc - perf.auroc.tail(4).mean()) > 0.03)
print(f"baseline AUROC={base_auroc:.3f}  recent labelled mean={perf.auroc.tail(4).mean() if len(perf) else float('nan'):.3f}  degraded={perf_degraded}")

# COMMAND ----------
# MAGIC %md ## 3. Lakehouse Monitoring (InferenceLog profile) - managed profile/drift tables + auto-generated dashboard

# COMMAND ----------

def create_lakehouse_monitor():
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorCronSchedule)
    w = WorkspaceClient()
    user = spark.sql("select current_user()").first()[0]
    try:
        w.quality_monitors.get(T["inference_log"])
        w.quality_monitors.run_refresh(T["inference_log"])
        print("monitor exists -> refresh triggered")
    except Exception:
        w.quality_monitors.create(
            table_name=T["inference_log"], assets_dir=f"/Workspace/Users/{user}/lakehouse_monitoring/{SCHEMA}",
            output_schema_name=FQ, baseline_table_name=T["baseline"],
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                prediction_col="prediction_label", label_col="readmit_30d", timestamp_col="discharge_ts",
                model_id_col="model_version", granularities=["1 day", "1 week"]),
            schedule=MonitorCronSchedule(quartz_cron_expression="0 0 6 * * ?", timezone_id="UTC"))
        print("monitor created: profile_metrics / drift_metrics tables + dashboard are in", FQ)

try:
    create_lakehouse_monitor()
except Exception as e:      # SDK surface for monitoring is evolving - see docs if this fails
    print("Lakehouse Monitoring step skipped:", str(e)[:200])

# COMMAND ----------
# MAGIC %md ## 4. Alert queries (create as Databricks SQL Alerts → email/Slack/PagerDuty)
# MAGIC ```sql
# MAGIC -- A) any CRITICAL feature drift in the last day
# MAGIC SELECT count(*) AS critical_features FROM {FQ}.monitoring_feature_drift
# MAGIC WHERE status = 'CRITICAL' AND computed_at >= current_timestamp() - INTERVAL 1 DAY;      -- alert when > 0
# MAGIC -- B) quarantine rate spike (data feed broken)
# MAGIC SELECT max(failed_rows / total_rows) FROM {FQ}.dq_run_metrics WHERE run_ts >= current_date();   -- alert when > 0.05
# MAGIC -- C) cost watch for the endpoint
# MAGIC SELECT usage_date, sum(usage_quantity) dbus FROM system.billing.usage
# MAGIC WHERE usage_metadata.endpoint_name = 'readmission-risk-prod' GROUP BY 1 ORDER BY 1 DESC;
# MAGIC ```

# COMMAND ----------
# MAGIC %md ## 5. Decision output for the orchestrator

# COMMAND ----------

decision = {"critical_drift_features": n_crit, "warning_drift_features": n_warn, "performance_degraded": perf_degraded,
            "retrain_recommended": bool(n_crit >= 2 or perf_degraded)}
print(decision)
try:
    dbutils.jobs.taskValues.set("retrain_recommended", str(decision["retrain_recommended"]).lower())
except Exception:
    pass
dbutils.notebook.exit(json.dumps(decision))
