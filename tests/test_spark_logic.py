from pyspark.sql import functions as F
from poc_lib.dq import ENCOUNTER_RULES, split_by_rules
from poc_lib.features import build_patient_history_features, build_encounter_features


def _enc(spark):
    rows = [
        (1, 10, 1, "2024-01-01 10:00:00", "2024-01-03 10:00:00", 2, "EMERGENCY", "HOME", "I50.9", 1, 5, 5000.0, "CHF"),
        (2, 10, 1, "2024-01-20 10:00:00", "2024-01-22 10:00:00", 2, "ELECTIVE", "HOME", "I10", 0, 3, 3000.0, "HTN"),
        (3, 10, 1, "2024-06-01 10:00:00", "2024-06-04 10:00:00", 3, "EMERGENCY", "SNF", "I50.9", 2, 8, 9000.0, "CHF"),
    ]
    cols = ["encounter_id", "patient_id", "hospital_id", "admit_ts", "discharge_ts", "los_days", "admission_type",
            "discharge_disposition", "primary_dx_icd10", "num_procedures", "num_medications", "total_charges", "dx_group"]
    return (spark.createDataFrame(rows, cols)
            .withColumn("admit_ts", F.col("admit_ts").cast("timestamp")).withColumn("discharge_ts", F.col("discharge_ts").cast("timestamp")))


def test_history_features_are_point_in_time(spark):
    pats = spark.createDataFrame([(10, 1950, "F", "MEDICARE", 3)],
                                 "patient_id long, birth_year int, gender string, insurance_type string, chronic_conditions_count int")
    out = {r.event_ts.strftime("%Y-%m-%d"): r for r in build_patient_history_features(_enc(spark), pats).collect()}
    assert out["2024-01-01"].prior_admits_365d == 0 and out["2024-01-01"].prior_total_admits == 0   # no leakage
    assert out["2024-01-20"].prior_admits_30d == 1
    assert out["2024-06-01"].prior_admits_90d == 0 and out["2024-06-01"].prior_admits_180d == 2
    assert out["2024-06-01"].age_at_admit == 74


def test_dq_split_quarantines_bad_rows(spark):
    df = _enc(spark).withColumn("los_days", F.when(F.col("encounter_id") == 2, -1).otherwise(F.col("los_days")))
    valid, bad = split_by_rules(df, ENCOUNTER_RULES)
    assert valid.count() == 2 and bad.count() == 1
    assert bad.first()["_failed_rules"] == ["los_in_range"]


def test_encounter_features(spark):
    labs = spark.createDataFrame([(1, 134.0, 1.7, 11.0)], "encounter_id long, sodium double, creatinine double, hemoglobin double")
    r = build_encounter_features(_enc(spark), labs).filter("encounter_id = 1").first()
    assert (r.sodium_low, r.aki_flag, r.anemia_flag, r.is_emergency) == (1, 1, 1, 1)
