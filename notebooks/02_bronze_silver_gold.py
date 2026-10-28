# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Medallion pipeline: ingestion → data quality → PHI protection → gold
# MAGIC
# MAGIC ```
# MAGIC landing volume ──Auto Loader──▶ bronze (raw, append-only, lineage cols)
# MAGIC                                 └──▶ silver (typed, deduped, DQ-validated)  ──▶ quarantine (bad rows + failed rule names)
# MAGIC                                       ├── silver_patients       (de-identified, used by ML)
# MAGIC                                       └── silver_patients_phi   (direct identifiers, masked, restricted)
# MAGIC                                 └──▶ gold_labeled_encounters (encounter + label known as of AS_OF)
# MAGIC ```
# MAGIC Covers: Auto Loader (incremental, schema evolution), idempotency, dedupe, DQ rules + quarantine + metrics,
# MAGIC HIPAA de-identification (Safe-Harbor style), column masks, tags, liquid clustering, OPTIMIZE/ANALYZE, CDF.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F, Window
from poc_lib import synth
from poc_lib.dq import (ENCOUNTER_RULES, LAB_RULES, PATIENT_RULES, LABEL_RULES,
                        split_by_rules, rule_failure_counts)

# COMMAND ----------
# MAGIC %md ## 1. Bronze - Auto Loader (incremental & idempotent)
# MAGIC `availableNow` trigger = process everything new, then stop (batch economics, streaming semantics).
# MAGIC Re-running never re-ingests the same file (checkpoint). New columns are picked up via schema evolution.

# COMMAND ----------

def ingest_bronze(name: str, fmt: str, extra: dict = None):
    r = (spark.readStream.format("cloudFiles")
         .option("cloudFiles.format", fmt)
         .option("cloudFiles.schemaLocation", f"{CHK}/{name}/schema")
         .option("cloudFiles.inferColumnTypes", "true")
         .option("cloudFiles.schemaEvolutionMode", "addNewColumns"))
    for k, v in (extra or {}).items():
        r = r.option(k, v)
    df = (r.load(f"{RAW}/{name}")
          .withColumn("_ingest_ts", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path")))
    (df.writeStream.option("checkpointLocation", f"{CHK}/{name}/checkpoint")
       .option("mergeSchema", "true").trigger(availableNow=True)
       .toTable(f"{FQ}.bronze_{name}").awaitTermination())
    print(f"bronze_{name}: {spark.table(f'{FQ}.bronze_{name}').count():,} rows")

ingest_bronze("encounters", "csv", {"header": "true"})
ingest_bronze("patients", "csv", {"header": "true"})
ingest_bronze("labs", "parquet")
ingest_bronze("labels", "parquet")

# COMMAND ----------
# MAGIC %md ## 2. Silver - typing, dedupe, DQ rules, quarantine

# COMMAND ----------

dq_frames = []

def write_silver(df, table, quarantine_table, rules, key_cols):
    """Cluster by access pattern (liquid clustering), enable CDF, overwrite = idempotent re-derivation."""
    valid, bad = split_by_rules(df, rules)
    total = df.count()
    (bad.withColumn("_quarantined_at", F.current_timestamp())
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(quarantine_table))
    qdf = spark.table(quarantine_table)
    dq_frames.append(rule_failure_counts(qdf, rules, table, total))
    (valid.write.mode("overwrite").option("overwriteSchema", "true")
          .clusterBy(*key_cols).saveAsTable(table))
    spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true, "
              f"delta.autoOptimize.optimizeWrite = true)")
    print(f"{table}: valid={spark.table(table).count():,}  quarantined={qdf.count():,}  of {total:,}")

# ---- encounters
b = spark.table(T["bronze_encounters"])
typed = b.select(
    F.col("encounter_id").cast("long"), F.col("patient_id").cast("long"), F.col("hospital_id").cast("int"),
    F.col("admit_ts").cast("timestamp"), F.col("discharge_ts").cast("timestamp"), F.col("los_days").cast("int"),
    F.upper(F.trim("admission_type")).alias("admission_type"),
    F.upper(F.trim("discharge_disposition")).alias("discharge_disposition"),
    "primary_dx_icd10", F.col("num_procedures").cast("int"), F.col("num_medications").cast("int"),
    F.col("total_charges").cast("double"), "_ingest_ts")
w = Window.partitionBy("encounter_id").orderBy(F.col("_ingest_ts").desc())
deduped = typed.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")   # keep latest copy

# reference dimension: ICD-10 -> diagnosis group
dx_ref = spark.createDataFrame(synth.DX, "primary_dx_icd10 string, dx_group string")
dx_ref.write.mode("overwrite").saveAsTable(f"{FQ}.ref_icd10_group")
enc_silver_in = (deduped.join(F.broadcast(dx_ref), "primary_dx_icd10", "left")
                 .withColumn("dx_group", F.coalesce("dx_group", F.lit("OTHER"))))
write_silver(enc_silver_in, T["silver_encounters"], T["quarantine_encounters"], ENCOUNTER_RULES,
             ["patient_id", "admit_ts"])

# ---- labs
bl = spark.table(T["bronze_labs"])
labs_in = (bl.select(F.col("encounter_id").cast("long"), F.upper("lab_name").alias("lab_name"),
                     F.col("lab_value").cast("double"), "unit", F.col("lab_ts").cast("timestamp"), "_ingest_ts")
           .withColumn("_rn", F.row_number().over(
               Window.partitionBy("encounter_id", "lab_name").orderBy(F.col("_ingest_ts").desc())))
           .filter("_rn = 1").drop("_rn"))
write_silver(labs_in, T["silver_labs"], T["quarantine_labs"], LAB_RULES, ["encounter_id"])

# ---- labels
lab_in = (spark.table(T["bronze_labels"]).select(
    F.col("encounter_id").cast("long"), F.col("readmit_30d").cast("int"),
    F.col("label_available_ts").cast("timestamp")).dropDuplicates(["encounter_id"]))
valid_l, bad_l = split_by_rules(lab_in, LABEL_RULES)
valid_l.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["silver_labels"])

# COMMAND ----------
# MAGIC %md ## 3. Patients - split into **de-identified** (for ML) and **PHI** (restricted) tables
# MAGIC HIPAA Safe-Harbor style: drop names/SSN/phone, keep only birth *year*, 3-digit ZIP; ages > 89 are capped later.
# MAGIC The ML team never needs the PHI table.

# COMMAND ----------

bp = spark.table(T["bronze_patients"])
pat_typed = bp.select(
    F.col("patient_id").cast("long"), F.col("dob").cast("date"), "gender", "race_ethnicity", "insurance_type",
    F.col("zip").cast("string").alias("zip"), "first_name", "last_name", F.col("ssn").cast("string").alias("ssn"),
    F.col("phone").cast("string").alias("phone"), F.col("chronic_conditions_count").cast("int"), "_ingest_ts"
).withColumn("_rn", F.row_number().over(Window.partitionBy("patient_id").orderBy(F.col("_ingest_ts").desc()))
             ).filter("_rn = 1").drop("_rn")
pat_valid, pat_bad = split_by_rules(pat_typed, PATIENT_RULES)
pat_bad.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["quarantine_patients"])

(pat_valid.select("patient_id", F.year("dob").alias("birth_year"), "gender", "race_ethnicity", "insurance_type",
                  F.substring("zip", 1, 3).alias("zip3"), "chronic_conditions_count")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["silver_patients"]))

(pat_valid.select("patient_id", "first_name", "last_name", "ssn", F.col("dob").cast("string").alias("dob"),
                  "phone", "zip")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T["silver_patients_phi"]))

# COMMAND ----------
# MAGIC %md ### 3b. Attach column masks, row filter, tags (Unity Catalog)

# COMMAND ----------

def try_sql(s):
    try:
        spark.sql(s)
    except Exception as e:
        print("SKIPPED:", s[:100], "->", str(e)[:150])

phi = T["silver_patients_phi"]
for col in ["first_name", "last_name", "dob", "phone", "zip"]:
    try_sql(f"ALTER TABLE {phi} ALTER COLUMN {col} SET MASK {FQ}.mask_phi")
try_sql(f"ALTER TABLE {phi} ALTER COLUMN ssn SET MASK {FQ}.mask_ssn")
try_sql(f"ALTER TABLE {phi} SET TAGS ('sensitivity' = 'phi', 'owner' = 'data-governance')")
for col, tag in [("ssn", "ssn"), ("first_name", "name"), ("last_name", "name"), ("dob", "dob"), ("phone", "phone")]:
    try_sql(f"ALTER TABLE {phi} ALTER COLUMN {col} SET TAGS ('pii' = '{tag}')")
try_sql(f"ALTER TABLE {T['silver_patients']} SET TAGS ('sensitivity' = 'deidentified')")
try_sql(f"ALTER TABLE {T['silver_encounters']} SET ROW FILTER {FQ}.hospital_row_filter ON (hospital_id)")
try_sql(f"COMMENT ON TABLE {T['silver_encounters']} IS 'Validated inpatient encounters (one row per encounter)'")
try_sql(f"COMMENT ON COLUMN {T['silver_encounters']}.los_days IS 'Length of stay in days (1-365)'")

# COMMAND ----------
# MAGIC %md ## 4. Gold - labelled encounters (only where the 30-day outcome window has closed)
# MAGIC Using un-matured labels would teach the model that recent patients "never come back" - a classic leakage bug.

# COMMAND ----------

gold = (spark.table(T["silver_encounters"])
        .join(spark.table(T["silver_labels"]), "encounter_id", "inner")
        .filter(F.col("label_available_ts") <= F.lit(AS_OF_TS).cast("timestamp")))
(gold.write.mode("overwrite").option("overwriteSchema", "true").clusterBy("patient_id", "admit_ts")
     .saveAsTable(T["gold_labeled"]))
print("gold_labeled_encounters:", spark.table(T["gold_labeled"]).count())

# COMMAND ----------
# MAGIC %md ## 5. Persist DQ metrics (trend them in a dashboard / alert on spikes) + table maintenance

# COMMAND ----------

from functools import reduce
dq_all = reduce(lambda a, b: a.unionByName(b), dq_frames)
dq_all.write.mode("append").saveAsTable(T["dq_metrics"])
display(dq_all.filter("failed_rows > 0"))

# fail the pipeline if the quarantine rate explodes (upstream feed broken)
enc_q = dq_all.filter("table_name LIKE '%silver_encounters'").agg(F.sum("failed_rows").alias("f"), F.max("total_rows").alias("t")).first()
assert enc_q["f"] / enc_q["t"] < 0.05, "DQ gate failed: >5% of encounters violate rules"

for t in [T["silver_encounters"], T["silver_labs"], T["gold_labeled"]]:
    spark.sql(f"OPTIMIZE {t}")
    spark.sql(f"ANALYZE TABLE {t} COMPUTE STATISTICS FOR ALL COLUMNS")
