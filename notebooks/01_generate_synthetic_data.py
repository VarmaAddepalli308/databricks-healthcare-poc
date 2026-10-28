# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Generate synthetic healthcare data (1,000,000 encounters)
# MAGIC **Use case:** predict whether a discharged inpatient will be **readmitted within 30 days**, so care
# MAGIC managers can prioritise follow-up calls / home-health visits for the highest-risk patients.
# MAGIC
# MAGIC | Feed | Rows (default) | Format | Notes |
# MAGIC |---|---|---|---|
# MAGIC | patients | 300k | CSV | contains fake PHI (name, SSN, DOB, phone, ZIP) |
# MAGIC | encounters | ~1.005M | CSV | includes injected dirty rows (nulls, negative LOS, dupes, bad enums) |
# MAGIC | labs | ~3M | Parquet | 3 discharge labs per encounter, some implausible values |
# MAGIC | labels | 1M | Parquet | arrive separately, **30 days after discharge** (ground-truth latency) |
# MAGIC
# MAGIC In production the label is derived from the *next* admission (`lead(admit_ts)` over patient) - here it is
# MAGIC simulated from a clinically-plausible risk model so the ML has real signal to learn.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F
from poc_lib import synth

# 1) patients -> materialise once so every later step sees identical values
patients = synth.generate_patients(spark, N_PATIENTS)
patients.write.mode("overwrite").format("delta").saveAsTable(f"{FQ}._tmp_patients")
patients = spark.table(f"{FQ}._tmp_patients")

# 2) wide encounter frame -> materialise (random columns are deterministic from here on)
full = synth.generate_encounters(spark, N_ENCOUNTERS, patients, N_PATIENTS, drift=0.0)
full.write.mode("overwrite").format("delta").saveAsTable(f"{FQ}._tmp_full")
full = spark.table(f"{FQ}._tmp_full")

# COMMAND ----------

enc, labs, labels = synth.split_outputs(full)
enc_raw = synth.inject_dirty_encounters(enc)
labs_raw = synth.inject_dirty_labs(labs)
pat_raw = synth.inject_dirty_patients(patients.withColumn("dob", F.col("dob")))

# Landing zone: initial full load. Later batches (notebook 09) land in sibling folders and are
# picked up incrementally by Auto Loader.
(enc_raw.repartition(8).write.mode("overwrite").option("header", "true").csv(f"{RAW}/encounters/initial"))
(pat_raw.repartition(2).write.mode("overwrite").option("header", "true").csv(f"{RAW}/patients/initial"))
(labs_raw.repartition(8).write.mode("overwrite").parquet(f"{RAW}/labs/initial"))
(labels.repartition(4).write.mode("overwrite").parquet(f"{RAW}/labels/initial"))

# COMMAND ----------

# MAGIC %md ### Sanity checks on the generated data

# COMMAND ----------

n_enc = full.count()
prev = full.agg(F.avg("readmit_30d")).first()[0]
print(f"encounters={n_enc:,}  patients={N_PATIENTS:,}  readmission prevalence={prev:.3%}")
assert n_enc == N_ENCOUNTERS
assert 0.08 < prev < 0.30, "prevalence outside expected band - tune intercept in synth.py"
display(full.groupBy("discharge_disposition").agg(F.count("*").alias("n"), F.avg("readmit_30d").alias("readmit_rate")))

# COMMAND ----------

# clean up temporary tables (keep them if you want to inspect the wide frame)
for t in ["_tmp_patients", "_tmp_full"]:
    spark.sql(f"DROP TABLE IF EXISTS {FQ}.{t}")
display(dbutils.fs.ls(f"{RAW}/encounters/initial")[:5])