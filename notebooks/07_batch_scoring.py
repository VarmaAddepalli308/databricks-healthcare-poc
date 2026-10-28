# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Batch inference (daily discharge scoring)
# MAGIC Scores recently discharged patients with the **Champion** model using `fe.score_batch` - features are looked up
# MAGIC point-in-time from the feature table, the on-demand UDF is applied automatically. Results are merged
# MAGIC idempotently into the **inference log** (also the input for monitoring) and exposed to care managers through
# MAGIC a governed worklist view.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from poc_lib.features import build_encounter_features, pivot_labs
from poc_lib.scoring import to_inference_log, upsert

fe, client = FeatureEngineeringClient(), MlflowClient()
champ = client.get_model_version_by_alias(MODEL_NAME, "Champion")
thr_high, thr_med = float(champ.tags["decision_threshold"]), float(champ.tags["medium_threshold"])
print(f"Champion v{champ.version}  high>={thr_high:.4f}  medium>={thr_med:.4f}")

# COMMAND ----------
# MAGIC %md ## 1. Select encounters to score
# MAGIC "Today" in the simulation = latest discharge in silver. We score the last `window_days` and skip anything
# MAGIC already scored by this model version (idempotent re-runs).

# COMMAND ----------

dbutils.widgets.text("window_days", "45")
WINDOW = int(dbutils.widgets.get("window_days"))

enc = spark.table(T["silver_encounters"])
as_of = enc.agg(F.max("discharge_ts")).first()[0]
recent = enc.filter(F.col("discharge_ts") >= F.lit(as_of) - F.expr(f"INTERVAL {WINDOW} DAYS"))

if spark.catalog.tableExists(T["inference_log"]):
    done = (spark.table(T["inference_log"]).filter(F.col("model_version") == str(champ.version))
            .select("encounter_id").distinct())
    recent = recent.join(done, "encounter_id", "left_anti")

labs_pivot = pivot_labs(spark.table(T["silver_labs"]), enc)
to_score = (build_encounter_features(recent, labs_pivot)
            .select("encounter_id", "patient_id", "admit_ts", "discharge_ts",
                    *[c for c in spark.table(T["training_base"]).columns
                      if c not in ("encounter_id", "patient_id", "admit_ts", "discharge_ts", "readmit_30d")]))
n = to_score.count()
print(f"as_of={as_of}  rows to score: {n:,}")

# COMMAND ----------
# MAGIC %md ## 2. Score → normalise → merge

# COMMAND ----------

if n > 0:
    scored = fe.score_batch(model_uri=f"models:/{MODEL_NAME}@Champion", df=to_score)
    log = to_inference_log(scored, MODEL_NAME, champ.version, thr_high, thr_med)
    upsert(spark, log, T["inference_log"], keys=["encounter_id", "model_version"])
    print("merged into", T["inference_log"])
display(spark.table(T["inference_log"]).groupBy("risk_tier").count())

# COMMAND ----------
# MAGIC %md ## 3. Care-management worklist view (what the business actually consumes)
# MAGIC Joins the restricted PHI table - names are **masked** unless the caller is in `phi_admins`.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {FQ}.vw_care_mgmt_worklist AS
SELECT l.encounter_id, l.patient_id, p.first_name, p.last_name, p.phone,
       l.hospital_id, l.discharge_ts, l.risk_score, l.risk_tier, l.model_version
FROM {T['inference_log']} l
JOIN {T['silver_patients_phi']} p USING (patient_id)
WHERE l.risk_tier = 'HIGH'
  AND l.discharge_ts >= (SELECT max(discharge_ts) FROM {T['inference_log']}) - INTERVAL 7 DAYS
""")
for stmt in [f"GRANT SELECT ON VIEW {FQ}.vw_care_mgmt_worklist TO `care_mgmt_apps`"]:
    try:
        spark.sql(stmt)
    except Exception as e:
        print("SKIPPED:", str(e)[:120])
display(spark.sql(f"SELECT * FROM {FQ}.vw_care_mgmt_worklist ORDER BY risk_score DESC LIMIT 20"))

# COMMAND ----------
# MAGIC %md ## 4. Dashboard ideas (AI/BI dashboard on `gold_inference_log`)
# MAGIC * HIGH-risk volume by hospital and day · risk-score distribution vs baseline · outcomes by tier once labels land
# MAGIC * Precision@top-10% trend (from `monitoring_performance`) · PSI heatmap (from `monitoring_feature_drift`)
