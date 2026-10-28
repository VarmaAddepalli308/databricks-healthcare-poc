# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 10 · Automated retraining policy
# MAGIC Retraining is triggered by **any** of:
# MAGIC 1. **Schedule** - monthly (job cron on the training job itself)
# MAGIC 2. **Data drift** - ≥ 2 features with PSI > 0.25 (from notebook 09)
# MAGIC 3. **Performance decay** - labelled AUROC dropped > 0.03 vs the offline baseline
# MAGIC 4. **Manual** - clinician/governance request (run the training job by hand)
# MAGIC
# MAGIC A retrain never auto-deploys: it creates a **Challenger**, which must pass notebook 06's gates (and human approval
# MAGIC in prod) before replacing the Champion. A cool-down avoids retrain storms.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
from datetime import datetime, timedelta
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

dbutils.widgets.text("monitoring_decision", "")          # JSON passed from the monitoring task / or read from the table
dbutils.widgets.text("training_job_name", f"[{ENV}] readmission_training_pipeline")
dbutils.widgets.text("cooldown_days", "14")

w = WorkspaceClient()
decision_raw = dbutils.widgets.get("monitoring_decision")
if decision_raw:
    decision = json.loads(decision_raw)
else:  # derive from tables if not passed in
    d = spark.table(T["feature_drift"]).filter(F.col("computed_at") >= F.current_timestamp() - F.expr("INTERVAL 2 DAYS"))
    decision = {"critical_drift_features": d.filter("status = 'CRITICAL'").count(), "performance_degraded": False}
    decision["retrain_recommended"] = decision["critical_drift_features"] >= 2
print("decision:", decision)

# COMMAND ----------

# MAGIC %md ## Cool-down + trigger

# COMMAND ----------

job = next((j for j in w.jobs.list(name=dbutils.widgets.get("training_job_name"))), None)
if job is None:
    raise RuntimeError("training job not found - deploy the bundle first (databricks bundle deploy)")

recent = [r for r in w.jobs.list_runs(job_id=job.job_id, limit=5)
          if r.start_time and datetime.utcfromtimestamp(r.start_time / 1000) > datetime.utcnow() - timedelta(days=int(dbutils.widgets.get("cooldown_days")))]

if not decision.get("retrain_recommended"):
    print("No retraining needed.")
elif recent:
    print(f"Retrain recommended but cool-down active (last run {recent[0].run_page_url}).")
else:
    run = w.jobs.run_now(job_id=job.job_id, job_parameters={"trigger_reason": "drift_or_performance", "catalog": CATALOG,
                                                          "schema": SCHEMA, "env": ENV})
    print("Triggered training job run:", run.run_id)