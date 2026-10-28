"""Feature engineering logic - ONE implementation used by training, batch scoring and unit tests.

Two kinds of features:
  1. Encounter-level features   -> computed from the encounter row itself + discharge labs
                                   (available at the moment of discharge; passed in the request).
  2. Patient-history features   -> point-in-time-correct window aggregates over PRIOR encounters,
                                   stored in a Unity Catalog feature table (timeseries) and looked
                                   up at training / scoring time.

Leakage rule: a feature for an encounter may only use information known at ADMIT time (history)
or at DISCHARGE time (labs, LOS). Never anything after discharge.
"""
from pyspark.sql import DataFrame, Window, functions as F

LABEL = "readmit_30d"
ID_COLS = ["encounter_id", "patient_id", "admit_ts", "discharge_ts"]

ENC_NUMERIC = ["los_days", "num_procedures", "num_medications", "total_charges", "sodium", "creatinine",
               "hemoglobin", "is_emergency", "is_weekend_admit", "admit_month", "sodium_low",
               "anemia_flag", "aki_flag", "polypharmacy"]
ENC_CATEGORICAL = ["admission_type", "discharge_disposition", "dx_group", "hospital_id"]

HISTORY_FEATURES = ["prior_admits_30d", "prior_admits_90d", "prior_admits_180d", "prior_admits_365d",
                    "prior_ed_admits_365d", "prior_avg_los_365d", "prior_charges_365d",
                    "days_since_last_discharge", "prior_total_admits", "prior_distinct_dx"]
DEMOGRAPHIC_FEATURES = ["age_at_admit", "gender", "insurance_type", "chronic_conditions_count"]
CATEGORICAL_ALL = ENC_CATEGORICAL + ["gender", "insurance_type"]

# columns that must never become model inputs
PHI_DENYLIST = ["first_name", "last_name", "ssn", "dob", "phone", "zip", "address", "email", "mrn"]
SENSITIVE_NOT_FOR_MODEL = ["race_ethnicity"]  # used only to AUDIT fairness, never as a predictor

HISTORY_TABLE = "patient_history_features"
ON_DEMAND_UDF = "charges_per_day"


def pivot_labs(labs: DataFrame, enc: DataFrame) -> DataFrame:
    """Last lab value of each type drawn on/before discharge (no post-discharge leakage)."""
    l = (labs.join(enc.select("encounter_id", "discharge_ts"), "encounter_id")
             .filter(F.col("lab_ts") <= F.col("discharge_ts")))
    p = (l.groupBy("encounter_id")
          .pivot("lab_name", ["SODIUM", "CREATININE", "HEMOGLOBIN"])
          .agg(F.max_by("lab_value", "lab_ts")))
    return p.select("encounter_id", F.col("SODIUM").alias("sodium"),
                    F.col("CREATININE").alias("creatinine"), F.col("HEMOGLOBIN").alias("hemoglobin"))


def build_encounter_features(enc: DataFrame, labs_pivot: DataFrame) -> DataFrame:
    df = enc.join(labs_pivot, "encounter_id", "left")
    return (df
            .withColumn("hospital_id", F.col("hospital_id").cast("string"))
            .withColumn("is_emergency", (F.col("admission_type") == "EMERGENCY").cast("int"))
            .withColumn("is_weekend_admit", F.dayofweek("admit_ts").isin(1, 7).cast("int"))
            .withColumn("admit_month", F.month("admit_ts"))
            .withColumn("sodium_low", (F.col("sodium") < 135).cast("int"))
            .withColumn("anemia_flag", (F.col("hemoglobin") < 12).cast("int"))
            .withColumn("aki_flag", (F.col("creatinine") > 1.5).cast("int"))
            .withColumn("polypharmacy", (F.col("num_medications") >= 10).cast("int")))


def build_patient_history_features(enc: DataFrame, patients: DataFrame) -> DataFrame:
    """Window features as of each encounter's ADMIT time (strictly prior encounters only).

    Output grain: (patient_id, event_ts) where event_ts = admit_ts -> a timeseries feature table.
    """
    day = 86400
    base = enc.select(
        "encounter_id", "patient_id", "admit_ts", "discharge_ts", "los_days", "total_charges", "dx_group",
        (F.col("admission_type") == "EMERGENCY").cast("int").alias("is_emerg"))
    w = Window.partitionBy("patient_id").orderBy(F.col("admit_ts").cast("long"))

    def rng(days):  # [-days, -1] seconds -> excludes the current row
        return w.rangeBetween(-days * day, -1)

    w_all = w.rowsBetween(Window.unboundedPreceding, -1)
    f = (base
         .withColumn("prior_admits_30d", F.count("encounter_id").over(rng(30)))
         .withColumn("prior_admits_90d", F.count("encounter_id").over(rng(90)))
         .withColumn("prior_admits_180d", F.count("encounter_id").over(rng(180)))
         .withColumn("prior_admits_365d", F.count("encounter_id").over(rng(365)))
         .withColumn("prior_ed_admits_365d", F.coalesce(F.sum("is_emerg").over(rng(365)), F.lit(0)).cast("long"))
         .withColumn("prior_avg_los_365d", F.avg("los_days").over(rng(365)))
         .withColumn("prior_charges_365d", F.coalesce(F.sum("total_charges").over(rng(365)), F.lit(0.0)))
         .withColumn("days_since_last_discharge",
                     F.datediff(F.col("admit_ts"), F.lag("discharge_ts", 1).over(w)))
         .withColumn("prior_total_admits", F.count("encounter_id").over(w_all))
         .withColumn("prior_distinct_dx", F.size(F.collect_set("dx_group").over(w_all))))
    out = (f.join(patients.select("patient_id", "birth_year", "gender", "insurance_type", "chronic_conditions_count"),
                  "patient_id", "left")
            .withColumn("age_at_admit", F.least(F.lit(90), F.year("admit_ts") - F.col("birth_year")).cast("int"))
            .withColumnRenamed("admit_ts", "event_ts"))
    cols = ["patient_id", "event_ts"] + HISTORY_FEATURES + DEMOGRAPHIC_FEATURES
    return out.select(*cols).dropDuplicates(["patient_id", "event_ts"])


def get_feature_lookups(fq: str):
    """Feature lookups used identically at training, validation and batch scoring time."""
    from databricks.feature_engineering import FeatureLookup, FeatureFunction  # lazy: not needed in unit tests
    return [
        FeatureLookup(table_name=f"{fq}.{HISTORY_TABLE}", lookup_key="patient_id",
                      timestamp_lookup_key="admit_ts",
                      feature_names=HISTORY_FEATURES + DEMOGRAPHIC_FEATURES),
        FeatureFunction(udf_name=f"{fq}.{ON_DEMAND_UDF}",
                        input_bindings={"total_charges": "total_charges", "los_days": "los_days"},
                        output_name="charges_per_day"),
    ]
