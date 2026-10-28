# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Exploratory data analysis & data profiling
# MAGIC Goals: understand the target, find data problems *before* modelling, decide on split strategy and metrics,
# MAGIC and write down leakage risks. Figures are logged to MLflow so the analysis is reproducible/auditable.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pyspark.sql import functions as F

df = spark.table(T["gold_labeled"])
pat = spark.table(T["silver_patients"])
print(f"rows={df.count():,}  columns={len(df.columns)}")
display(df.limit(5))

# COMMAND ----------

# MAGIC %md ## 1. Profile (Databricks built-in) + null report

# COMMAND ----------

# DBTITLE 1,Cell 5
display(df.drop("_ingest_ts"))   # interactive profile: click the + / Profile option in the results view for histograms
nulls = df.select([F.round(F.avg(F.col(c).isNull().cast("int")) * 100, 3).alias(c) for c in df.columns])
display(nulls)

# COMMAND ----------

# MAGIC %md ## 2. Target: prevalence, class balance, trend over time

# COMMAND ----------

prev = df.agg(F.avg("readmit_30d")).first()[0]
print(f"Readmission prevalence: {prev:.2%}  -> moderate imbalance; use AUPRC / precision@k, not accuracy")
monthly = (df.groupBy(F.date_trunc("month", "discharge_ts").alias("month"))
             .agg(F.count("*").alias("n"), F.avg("readmit_30d").alias("rate")).orderBy("month").toPandas())

with mlflow.start_run(run_name="eda"):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(monthly["month"], monthly["rate"], marker="o"); ax.set_title("Monthly 30-day readmission rate")
    ax.set_ylim(0, monthly["rate"].max() * 1.3); plt.tight_layout()
    mlflow.log_figure(fig, "eda/monthly_readmit_rate.png"); plt.show()
    mlflow.log_metric("prevalence", prev)

# COMMAND ----------

# MAGIC %md ## 3. Segment analysis - where is risk concentrated?

# COMMAND ----------

for col in ["discharge_disposition", "admission_type", "dx_group"]:
    display(df.groupBy(col).agg(F.count("*").alias("n"), F.round(F.avg("readmit_30d"), 4).alias("readmit_rate"))
              .orderBy(F.desc("readmit_rate")))

by_los = (df.withColumn("los_bucket", F.when(F.col("los_days") <= 2, "1-2").when(F.col("los_days") <= 5, "3-5")
                                       .when(F.col("los_days") <= 10, "6-10").otherwise("11+"))
            .groupBy("los_bucket").agg(F.count("*").alias("n"), F.avg("readmit_30d").alias("rate")))
display(by_los)

# COMMAND ----------

# MAGIC %md ## 4. Patient-level structure - important for splitting
# MAGIC Many patients have several encounters. A *random* row split would put the same patient in train and test
# MAGIC (optimistic metrics). We therefore split **by time** (train < 2024, validate H1-2024, test H2-2024).

# COMMAND ----------

per_patient = df.groupBy("patient_id").count()
display(per_patient.groupBy(F.when(F.col("count") >= 10, "10+").otherwise(F.col("count").cast("string")).alias("encounters_per_patient"))
        .count().orderBy("encounters_per_patient"))

# COMMAND ----------

# MAGIC %md ## 5. Bias / fairness baseline - outcome rates by protected attributes
# MAGIC (Race/ethnicity is **not** a model input; we only audit with it.)

# COMMAND ----------

aud = df.join(pat, "patient_id")
for c in ["gender", "race_ethnicity", "insurance_type"]:
    display(aud.groupBy(c).agg(F.count("*").alias("n"), F.round(F.avg("readmit_30d"), 4).alias("readmit_rate")))

# COMMAND ----------

# MAGIC %md ## 6. Numeric relationships (sampled to the driver)

# COMMAND ----------

s = df.sample(0.05, seed=7).select("los_days", "num_procedures", "num_medications", "total_charges", "readmit_30d").toPandas()
corr = s.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
plt.colorbar(im); plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md ## 7. Findings → decisions (write these down, they go in the model card)
# MAGIC | Finding | Decision |
# MAGIC |---|---|
# MAGIC | ~15% positives | Use AUPRC + precision/recall@top-10%; no resampling needed, threshold chosen by care-team capacity |
# MAGIC | Multiple encounters per patient | Temporal split; point-in-time history features |
# MAGIC | Dirty rows (~1%) | Quarantine in silver (done in notebook 02) |
# MAGIC | AMA / SNF / emergency show high risk | Expect them in SHAP - clinically sensible |
# MAGIC | Labels mature 30d after discharge | Train only on matured labels; monitor with label delay |
# MAGIC | Race is a sensitive attribute | Exclude from features, audit fairness by race |