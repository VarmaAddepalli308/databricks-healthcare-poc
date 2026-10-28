# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Shared configuration
# MAGIC Included by every notebook with `%run ./00_config`.
# MAGIC Values can be overridden by **job parameters** (same names as the widgets) - this is how
# MAGIC one codebase runs in `dev`, `staging` and `prod` with different catalogs.

# COMMAND ----------

import os, sys

dbutils.widgets.text("catalog", "health_poc")
dbutils.widgets.text("schema", "readmission")
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("n_encounters", "1000000")
dbutils.widgets.text("n_patients", "300000")
dbutils.widgets.text("git_sha", "local")
dbutils.widgets.text("as_of_ts", "2025-02-15 00:00:00")   # "today" in the simulated world

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENV = dbutils.widgets.get("env")
N_ENCOUNTERS = int(dbutils.widgets.get("n_encounters"))
N_PATIENTS = int(dbutils.widgets.get("n_patients"))
GIT_SHA = dbutils.widgets.get("git_sha")
AS_OF_TS = dbutils.widgets.get("as_of_ts")

FQ = f"{CATALOG}.{SCHEMA}"
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
CHK = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"

# Make the shared library importable (works for Repos, Git folders and Asset Bundles:
# the notebook's working directory is its own folder, src/ is its sibling).
_src = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if _src not in sys.path:
    sys.path.append(_src)

TABLES = {
    # bronze
    "bronze_encounters": f"{FQ}.bronze_encounters", "bronze_patients": f"{FQ}.bronze_patients",
    "bronze_labs": f"{FQ}.bronze_labs", "bronze_labels": f"{FQ}.bronze_labels",
    # silver
    "silver_encounters": f"{FQ}.silver_encounters", "silver_patients": f"{FQ}.silver_patients",
    "silver_patients_phi": f"{FQ}.silver_patients_phi", "silver_labs": f"{FQ}.silver_labs",
    "silver_labels": f"{FQ}.silver_labels",
    # quarantine + DQ
    "quarantine_encounters": f"{FQ}.quarantine_encounters", "quarantine_labs": f"{FQ}.quarantine_labs",
    "quarantine_patients": f"{FQ}.quarantine_patients", "dq_metrics": f"{FQ}.dq_run_metrics",
    # gold / ML
    "gold_labeled": f"{FQ}.gold_labeled_encounters", "training_base": f"{FQ}.gold_training_base",
    "history_features": f"{FQ}.patient_history_features",
    "inference_log": f"{FQ}.gold_inference_log", "baseline": f"{FQ}.gold_monitoring_baseline",
    "validation_results": f"{FQ}.model_validation_results", "feature_drift": f"{FQ}.monitoring_feature_drift",
    "perf_metrics": f"{FQ}.monitoring_performance",
}
T = TABLES

MODEL_NAME = f"{FQ}.readmission_model"                 # feature-store packaged model (batch scoring)
PAYLOAD_MODEL_NAME = f"{FQ}.readmission_model_payload"  # self-contained model (real-time serving)
ENDPOINT_NAME = f"readmission-risk-{ENV}"
EXPERIMENT = f"/Shared/{CATALOG}_readmission_{ENV}"

# temporal split (train on past, validate/test on future - never random split for clinical data)
TRAIN_END = "2024-01-01"
VAL_END = "2024-07-01"

import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

print(f"env={ENV} catalog={CATALOG} schema={SCHEMA} model={MODEL_NAME} endpoint={ENDPOINT_NAME}")
