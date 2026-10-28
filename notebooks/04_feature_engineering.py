# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Feature engineering & Feature Store (Unity Catalog)
# MAGIC
# MAGIC | Feature type | Where it lives | Why |
# MAGIC |---|---|---|
# MAGIC | Encounter-level (LOS, labs, admission type, flags) | computed from the request row (`build_encounter_features`) | known at discharge |
# MAGIC | Patient history (prior admits 30/90/180/365d, ED visits, days since last discharge ...) | **UC timeseries feature table** `patient_history_features` | point-in-time correct, reusable, governed, lineage-tracked |
# MAGIC | On-demand (`charges_per_day`) | **UC Python function** | computed at request time, identical in training and serving |
# MAGIC
# MAGIC Also covered: leakage test, feature validation, feature selection (correlation / mutual information),
# MAGIC training-base table, optional online-store publishing.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Install feature engineering library
# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()  # required for the newly installed package to be importable

# COMMAND ----------

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from poc_lib.features import (build_encounter_features, build_patient_history_features, pivot_labs,
                              ID_COLS, LABEL, ENC_NUMERIC, ENC_CATEGORICAL, HISTORY_FEATURES,
                              DEMOGRAPHIC_FEATURES, HISTORY_TABLE, ON_DEMAND_UDF)

fe = FeatureEngineeringClient()
enc = spark.table(T["silver_encounters"])
patients = spark.table(T["silver_patients"])
labs = spark.table(T["silver_labs"])

# COMMAND ----------

# MAGIC %md ## 1. Patient-history features (window aggregates, strictly *prior* encounters)

# COMMAND ----------

hist_df = build_patient_history_features(enc, patients).cache()
print("feature rows:", hist_df.count())
display(hist_df.orderBy("patient_id", "event_ts").limit(10))

# COMMAND ----------

# MAGIC %md ## 2. Leakage / point-in-time test (fail the pipeline if violated)
# MAGIC Re-compute `prior_admits_365d` with an independent self-join and compare. Also: the first-ever encounter of
# MAGIC every patient must have zero history.

# COMMAND ----------

smp = (hist_df.select(F.col("patient_id").alias("p"), F.col("event_ts").alias("t"),
                      F.col("prior_admits_365d").alias("got")).sample(0.001, seed=3).limit(500))
e = enc.select(F.col("patient_id").alias("ep"), F.col("admit_ts").alias("et"), "encounter_id")
chk = (smp.join(e, (F.col("p") == F.col("ep")) & (F.col("et") < F.col("t")) &
                   (F.col("et") >= F.col("t") - F.expr("INTERVAL 365 DAYS")), "left")
          .groupBy("p", "t", "got").agg(F.count("encounter_id").alias("expected")))
bad = chk.filter(F.col("got") != F.col("expected")).count()
assert bad == 0, f"Point-in-time leakage test failed for {bad} sampled rows"

first_enc_bad = (hist_df.withColumn("rn", F.row_number().over(
    __import__("pyspark.sql.window", fromlist=["Window"]).Window.partitionBy("patient_id").orderBy("event_ts")))
    .filter((F.col("rn") == 1) & (F.col("prior_total_admits") != 0)).count())
assert first_enc_bad == 0, "first encounter has non-zero history -> leakage"
print("PIT tests passed")

# COMMAND ----------

# MAGIC %md ## 3. Create / update the Unity Catalog feature table
# MAGIC * primary key `(patient_id, event_ts)` + `timeseries_columns` → enables `FeatureLookup(timestamp_lookup_key=...)` point-in-time joins
# MAGIC * `merge` mode makes re-runs idempotent and supports incremental refresh

# COMMAND ----------

ft_name = T["history_features"]
try:
    fe.get_table(name=ft_name)
    fe.write_table(name=ft_name, df=hist_df, mode="merge")
    print("merged into existing feature table")
except Exception:
    fe.create_table(
        name=ft_name, primary_keys=["patient_id", "event_ts"], timeseries_columns="event_ts", df=hist_df,
        description="Point-in-time patient utilisation history + demographics (de-identified). Grain: patient x admit time.",
        tags={"domain": "healthcare", "owner": "ml-platform", "phi": "false", "refresh": "daily"})
    print("created feature table")
spark.sql(f"ALTER TABLE {ft_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")  # required for online sync

# COMMAND ----------

# MAGIC %md ## 4. On-demand feature as a Unity Catalog Python function
# MAGIC Same code runs in training, batch scoring and real-time serving → no training/serving skew.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.{ON_DEMAND_UDF}(total_charges DOUBLE, los_days INT)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'On-demand feature: average billed charges per inpatient day'
AS $$
  if total_charges is None or los_days is None:
      return None
  return float(total_charges) / max(int(los_days), 1)
$$
""")
print(spark.sql(f"SELECT {FQ}.{ON_DEMAND_UDF}(10000.0, 4) AS v").first()["v"])

# COMMAND ----------

# MAGIC %md ## 5. Training base table = encounter features + label (feature-table values are joined later, point-in-time)

# COMMAND ----------

labs_pivot = pivot_labs(labs, enc)
labeled = spark.table(T["gold_labeled"])
base = build_encounter_features(labeled, labs_pivot)
base_cols = ID_COLS + ENC_NUMERIC + ENC_CATEGORICAL + [LABEL]
(base.select(*base_cols).write.mode("overwrite").option("overwriteSchema", "true")
     .clusterBy("admit_ts").saveAsTable(T["training_base"]))
spark.sql(f"COMMENT ON TABLE {T['training_base']} IS 'Labelled encounter-level features; history features joined via Feature Engineering client'")
print("training base rows:", spark.table(T["training_base"]).count())

# COMMAND ----------

# MAGIC %md ## 6. Feature validation (ranges, nulls, constants)

# COMMAND ----------

tb = spark.table(T["training_base"])
stats = tb.select(*[F.round(F.avg(F.col(c).isNull().cast("int")) * 100, 3).alias(f"{c}__null_pct") for c in ["sodium", "creatinine", "hemoglobin"]],
                  F.min("los_days").alias("los_min"), F.max("los_days").alias("los_max"),
                  F.min("total_charges").alias("charges_min")).first().asDict()
print(stats)
assert stats["los_min"] >= 1 and stats["charges_min"] >= 0
assert max(v for k, v in stats.items() if k.endswith("null_pct")) < 5, "lab null rate too high"

# COMMAND ----------

# MAGIC %md ## 7. Feature selection: correlation (redundancy) + mutual information (relevance)
# MAGIC Sampled to the driver. Decision rule: drop one of any pair with |ρ| > 0.95; keep features with MI > 0.

# COMMAND ----------

import pandas as pd, numpy as np
from sklearn.feature_selection import mutual_info_classif

j = (tb.select("patient_id", "admit_ts", LABEL, *ENC_NUMERIC)
       .join(hist_df.withColumnRenamed("event_ts", "admit_ts").drop("gender", "insurance_type"), ["patient_id", "admit_ts"])
       .sample(0.1, seed=11).toPandas())
num_cols = [c for c in ENC_NUMERIC + HISTORY_FEATURES + ["age_at_admit", "chronic_conditions_count"] if c in j.columns]
X = j[num_cols].astype(float)
corr = X.corr(method="spearman")
pairs = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(num_cols) for b in num_cols[i + 1:] if abs(corr.loc[a, b]) > 0.95]
print("highly correlated pairs (candidates to drop):", pairs)
mi = pd.Series(mutual_info_classif(X.fillna(X.median()), j[LABEL], random_state=0), index=num_cols).sort_values(ascending=False)
display(mi.reset_index().rename(columns={"index": "feature", 0: "mutual_information"}))

# COMMAND ----------

# MAGIC %md ## 8. (Optional) Online feature store for low-latency lookups
# MAGIC Needed only if the serving endpoint should *look up* history features by `patient_id`.
# MAGIC The online-store APIs have been evolving (Online Tables → Lakebase-based online feature store) - **verify against
# MAGIC the current Databricks docs before enabling**. This POC's default real-time path (notebook 08, Option B) sends the
# MAGIC full feature vector in the request, so this step is not required.

# COMMAND ----------

# Sketch (Online Tables via SDK; names/availability may differ in your workspace):
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
# w = WorkspaceClient()
# latest = f"{FQ}.patient_history_latest"      # one row per patient (latest event_ts) - online tables key on PK only
# spark.sql(f"CREATE OR REPLACE TABLE {latest} AS SELECT * EXCEPT (rn) FROM (SELECT *, row_number() OVER (PARTITION BY patient_id ORDER BY event_ts DESC) rn FROM {ft_name}) WHERE rn = 1")
# spark.sql(f"ALTER TABLE {latest} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# w.online_tables.create(table=OnlineTable(name=f"{latest}_online",
#       spec=OnlineTableSpec(primary_key_columns=["patient_id"], source_table_full_name=latest,
#                            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy())))
print("done")