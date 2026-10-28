"""Synthetic healthcare data generator for the 30-day readmission POC.

All data is fake. Nothing here is real PHI, but the shape (names, SSN, DOB, ZIP, phone)
deliberately mimics PHI so governance controls (masking, row filters) can be demonstrated.

NOTE on randomness: every random column is created ONCE with its own seed and materialised
as a column before being reused. Re-using a single rand() expression inside several
CASE WHEN branches gives skewed results in Spark, so we avoid that pattern.
"""
from pyspark.sql import DataFrame, SparkSession, functions as F

# (ICD-10 code, diagnosis group) - also used as the reference dimension table in silver
DX = [
    ("I50.9", "CHF"), ("J44.1", "COPD"), ("J18.9", "PNEUMONIA"), ("E11.9", "DIABETES"),
    ("N18.3", "CKD"), ("A41.9", "SEPSIS"), ("I10", "HTN"), ("K92.2", "GI_BLEED"),
    ("R07.9", "CHEST_PAIN"), ("M54.5", "OTHER"),
]
FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "Ravi", "Anita",
         "Wei", "Fatima", "Carlos", "Sofia", "Omar", "Priya", "Daniel", "Emma", "Arjun", "Lakshmi"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Reddy", "Sharma",
        "Chen", "Khan", "Lopez", "Martinez", "Nguyen", "Patel", "Singh", "Kim", "Rao", "Ali"]

EPOCH_2022_01_01 = 1640995200


def _pick(df: DataFrame, out: str, values, seed: int) -> DataFrame:
    """Uniform pick from a python list."""
    r = f"_r_{out}"
    arr = F.array(*[F.lit(v) for v in values])
    return (df.withColumn(r, F.rand(seed))
              .withColumn(out, F.element_at(arr, (F.floor(F.col(r) * len(values)) + 1).cast("int")))
              .drop(r))


def _pick_w(df: DataFrame, out: str, values, weights, seed: int) -> DataFrame:
    """Weighted pick; weights must sum to ~1."""
    r = f"_r_{out}"
    df = df.withColumn(r, F.rand(seed))
    cum, expr = 0.0, None
    for v, w in zip(values, weights):
        cum += w
        cond = F.col(r) < cum
        expr = F.when(cond, F.lit(v)) if expr is None else expr.when(cond, F.lit(v))
    return df.withColumn(out, expr.otherwise(F.lit(values[-1]))).drop(r)


def generate_patients(spark: SparkSession, n_patients: int, seed: int = 7) -> DataFrame:
    df = spark.range(n_patients).withColumnRenamed("id", "patient_id")
    df = df.withColumn("age_now", F.greatest(F.lit(21), F.least(F.lit(95), F.round(F.randn(seed) * 16 + 62))).cast("int"))
    df = df.withColumn("dob", F.expr("date_sub(DATE'2025-01-01', CAST(age_now * 365.25 AS INT))")).drop("age_now")
    df = _pick_w(df, "gender", ["F", "M", "U"], [0.51, 0.48, 0.01], seed + 1)
    df = _pick_w(df, "race_ethnicity", ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"], [0.58, 0.18, 0.15, 0.06, 0.03], seed + 2)
    df = _pick_w(df, "insurance_type", ["MEDICARE", "MEDICAID", "COMMERCIAL", "SELF_PAY"], [0.38, 0.20, 0.34, 0.08], seed + 3)
    df = df.withColumn("zip", F.lpad(F.floor(F.rand(seed + 4) * 99999).cast("string"), 5, "0"))
    df = _pick(df, "first_name", FIRST, seed + 5)
    df = _pick(df, "last_name", LAST, seed + 6)
    df = df.withColumn("ssn", F.concat(
        F.lpad(F.floor(F.rand(seed + 7) * 899 + 100).cast("string"), 3, "0"), F.lit("-"),
        F.lpad(F.floor(F.rand(seed + 8) * 99 + 1).cast("string"), 2, "0"), F.lit("-"),
        F.lpad(F.floor(F.rand(seed + 9) * 9999 + 1).cast("string"), 4, "0")))
    df = df.withColumn("phone", F.concat(F.lit("+1-555-"), F.lpad(F.floor(F.rand(seed + 12) * 9999).cast("string"), 4, "0")))
    # problem-list size: sicker patients have lower ids (and also more encounters, see below)
    df = df.withColumn("chronic_conditions_count", F.least(
        F.lit(8),
        F.floor(F.rand(seed + 10) * 2 + (1 - F.col("patient_id") / n_patients) * 2.5 + F.rand(seed + 11) * 1.5)).cast("int"))
    return df


def generate_encounters(spark: SparkSession, n_enc: int, patients: DataFrame, n_patients: int,
                        start_epoch: int = EPOCH_2022_01_01, span_days: int = 1095,
                        seed: int = 100, id_offset: int = 0, drift: float = 0.0) -> DataFrame:
    """Returns ONE wide dataframe (encounter + labs + label). Use split_outputs() afterwards.

    drift in [0, 1] shifts distributions (more emergency admissions, longer stays, worse
    kidney function) so we can demonstrate data drift detection and retraining.
    """
    pat = F.broadcast(patients.select("patient_id", "dob", "chronic_conditions_count"))
    df = spark.range(id_offset, id_offset + n_enc).withColumnRenamed("id", "encounter_id")
    # skew: low patient ids get many more encounters (frequent flyers)
    df = df.withColumn("patient_id", F.floor(F.pow(F.rand(seed + 1), F.lit(1.6)) * n_patients).cast("long"))
    df = df.withColumn("admit_ts", (F.lit(start_epoch) + F.floor(F.rand(seed + 2) * (span_days * 86400))).cast("timestamp"))
    df = df.withColumn("hospital_id", (F.floor(F.rand(seed + 3) * 50) + 1).cast("int"))
    p_em = min(0.55 + 0.15 * drift, 0.9)
    df = _pick_w(df, "admission_type", ["EMERGENCY", "URGENT", "ELECTIVE"], [p_em, 0.20, 1 - p_em - 0.20], seed + 4)
    df = df.withColumn("los_days", F.greatest(F.lit(1), F.round(F.exp(F.randn(seed + 5) * 0.6 + 1.2 + 0.2 * drift))).cast("int"))
    df = df.withColumn("discharge_ts", F.expr("admit_ts + make_interval(0, 0, 0, los_days, 0, 0, 0)"))
    df = _pick_w(df, "discharge_disposition", ["HOME", "HOME_HEALTH", "SNF", "REHAB", "AMA"],
                 [0.60, 0.15, 0.15, 0.06, 0.04], seed + 6)
    df = _pick(df, "primary_dx_icd10", [c for c, _ in DX], seed + 7)
    df = df.join(pat, "patient_id", "left")
    df = df.withColumn("num_procedures", F.floor(F.rand(seed + 8) * 5).cast("int"))
    df = df.withColumn("num_medications", (F.floor(F.rand(seed + 9) * 8) + F.col("chronic_conditions_count") * 2).cast("int"))
    df = df.withColumn("total_charges", F.round(
        F.col("los_days") * 2500 * F.exp(F.randn(seed + 10) * 0.3) + F.col("num_procedures") * 1800, 2))
    # discharge labs
    df = df.withColumn("sodium", F.round(F.randn(seed + 11) * 3 + 138, 1))
    df = df.withColumn("creatinine", F.round(F.greatest(
        F.lit(0.4), F.randn(seed + 12) * 0.4 + 0.9 + 0.12 * F.col("chronic_conditions_count") + 0.3 * drift), 2))
    df = df.withColumn("hemoglobin", F.round(F.randn(seed + 13) * 1.6 + 12.5 - 0.2 * F.col("chronic_conditions_count"), 1))
    df = df.withColumn("age", F.floor(F.datediff(F.col("admit_ts"), F.col("dob")) / 365.25))

    dx_effect = (F.when(F.col("primary_dx_icd10") == "I50.9", 0.5)
                 .when(F.col("primary_dx_icd10") == "J44.1", 0.4)
                 .when(F.col("primary_dx_icd10") == "A41.9", 0.3).otherwise(0.0))
    z = (F.lit(-3.6)
         + 0.025 * (F.col("age") - 62)
         + 0.22 * F.col("chronic_conditions_count")
         + 0.06 * F.col("los_days")
         + 0.30 * (F.col("los_days") > 7).cast("int")
         + 0.45 * (F.col("admission_type") == "EMERGENCY").cast("int")
         + 0.50 * (F.col("discharge_disposition") == "SNF").cast("int")
         + 0.30 * (F.col("discharge_disposition") == "HOME_HEALTH").cast("int")
         + 1.00 * (F.col("discharge_disposition") == "AMA").cast("int")
         + 0.50 * (F.col("creatinine") - 1.0)
         + 0.08 * F.greatest(F.lit(0.0), 135 - F.col("sodium"))
         + 0.12 * F.greatest(F.lit(0.0), 12 - F.col("hemoglobin"))
         + dx_effect
         + F.randn(seed + 14) * 0.3)
    df = df.withColumn("readmit_30d", (F.rand(seed + 15) < 1 / (1 + F.exp(-z))).cast("int"))
    return df.drop("dob", "age", "chronic_conditions_count")


def split_outputs(full: DataFrame):
    """Wide df -> three raw feeds like a hospital would send them (encounters, labs, labels)."""
    enc = full.select("encounter_id", "patient_id", "hospital_id", "admit_ts", "discharge_ts", "los_days",
                      "admission_type", "discharge_disposition", "primary_dx_icd10",
                      "num_procedures", "num_medications", "total_charges")
    labs = (full.select("encounter_id", "discharge_ts", "sodium", "creatinine", "hemoglobin")
            .withColumn("labs", F.array(
                F.struct(F.lit("SODIUM").alias("lab_name"), F.col("sodium").alias("lab_value"), F.lit("mmol/L").alias("unit")),
                F.struct(F.lit("CREATININE").alias("lab_name"), F.col("creatinine").alias("lab_value"), F.lit("mg/dL").alias("unit")),
                F.struct(F.lit("HEMOGLOBIN").alias("lab_name"), F.col("hemoglobin").alias("lab_value"), F.lit("g/dL").alias("unit"))))
            .select("encounter_id", "discharge_ts", F.explode("labs").alias("l"))
            .select("encounter_id", "l.lab_name", "l.lab_value", "l.unit",
                    (F.col("discharge_ts").cast("long") - (F.rand(991) * 86400).cast("long")).cast("timestamp").alias("lab_ts")))
    # label becomes known 30 days after discharge (ground-truth latency!)
    labels = full.select("encounter_id", "readmit_30d",
                         (F.col("discharge_ts") + F.expr("INTERVAL 30 DAYS")).alias("label_available_ts"))
    return enc, labs, labels


# ---------------------------------------------------------------- dirty data injection
def inject_dirty_encounters(enc: DataFrame, seed: int = 500) -> DataFrame:
    e = enc.withColumn("_a", F.rand(seed)).withColumn("_b", F.rand(seed + 1))
    e = e.withColumn("patient_id", F.when(F.col("_a") < 0.003, F.lit(None).cast("long")).otherwise(F.col("patient_id")))
    e = e.withColumn("los_days", F.when((F.col("_a") >= 0.003) & (F.col("_a") < 0.005), F.lit(-1)).otherwise(F.col("los_days")))
    e = e.withColumn("admission_type", F.when(F.col("_b") < 0.002, F.lit("UNKNOWN")).otherwise(F.col("admission_type")))
    e = e.drop("_a", "_b")
    return e.unionByName(e.sample(0.005, seed=seed + 2))  # ~0.5% duplicate rows


def inject_dirty_labs(labs: DataFrame, seed: int = 600) -> DataFrame:
    return (labs.withColumn("_r", F.rand(seed))
            .withColumn("lab_value", F.when(F.col("_r") < 0.002, F.lit(9999.0)).otherwise(F.col("lab_value")))
            .drop("_r"))


def inject_dirty_patients(p: DataFrame, seed: int = 700) -> DataFrame:
    return (p.withColumn("_r", F.rand(seed))
            .withColumn("dob", F.when(F.col("_r") < 0.001, F.lit("2030-01-01").cast("date")).otherwise(F.col("dob")))
            .drop("_r"))
