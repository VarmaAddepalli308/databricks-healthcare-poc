# Databricks notebook source
# MAGIC %md
# MAGIC # Optional · Same medallion flow as a **Lakeflow Spark Declarative Pipeline** (formerly Delta Live Tables)
# MAGIC Why: declarative dependencies, built-in **expectations** (data quality as code), auto-retries, lineage UI,
# MAGIC and incremental processing without hand-written checkpoint logic.
# MAGIC
# MAGIC Create a pipeline (Jobs & Pipelines → Create → ETL pipeline), add this file as source,
# MAGIC set `catalog`/`schema` and the config key `raw_path` = `/Volumes/<catalog>/<schema>/raw`.
# MAGIC `import dlt` is still supported; newer runtimes also expose `from pyspark import pipelines as dp` - check current docs.

# COMMAND ----------

import dlt
from pyspark.sql import functions as F

RAW = spark.conf.get("raw_path")


@dlt.table(name="bronze_encounters_dlt", comment="Raw encounters via Auto Loader")
def bronze_encounters_dlt():
    return (spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "csv").option("header", "true")
            .option("cloudFiles.inferColumnTypes", "true")
            .load(f"{RAW}/encounters")
            .withColumn("_ingest_ts", F.current_timestamp()))


@dlt.view(name="encounters_typed")
def encounters_typed():
    return (dlt.read_stream("bronze_encounters_dlt")
            .select(F.col("encounter_id").cast("long"), F.col("patient_id").cast("long"),
                    F.col("hospital_id").cast("int"), F.col("admit_ts").cast("timestamp"),
                    F.col("discharge_ts").cast("timestamp"), F.col("los_days").cast("int"),
                    F.upper("admission_type").alias("admission_type"),
                    F.upper("discharge_disposition").alias("discharge_disposition"),
                    "primary_dx_icd10", F.col("num_procedures").cast("int"),
                    F.col("num_medications").cast("int"), F.col("total_charges").cast("double"), "_ingest_ts"))


@dlt.table(name="silver_encounters_dlt", comment="Validated encounters (bad rows dropped + counted in event log)",
           cluster_by=["patient_id", "admit_ts"])
@dlt.expect_or_drop("patient_id_not_null", "patient_id IS NOT NULL")
@dlt.expect_or_drop("los_in_range", "los_days BETWEEN 1 AND 365")
@dlt.expect_or_drop("discharge_after_admit", "discharge_ts >= admit_ts")
@dlt.expect_or_drop("valid_admission_type", "admission_type IN ('EMERGENCY','URGENT','ELECTIVE')")
@dlt.expect("charges_non_negative", "total_charges >= 0")              # warn only
@dlt.expect_or_fail("encounter_id_not_null", "encounter_id IS NOT NULL")  # stop the pipeline
def silver_encounters_dlt():
    return dlt.read_stream("encounters_typed").dropDuplicates(["encounter_id"])
