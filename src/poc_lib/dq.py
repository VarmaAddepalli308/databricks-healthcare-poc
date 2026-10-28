"""Data-quality rules (SQL expressions that must evaluate TRUE for a row to be valid).

Rows failing any rule go to a quarantine table with the list of failed rule names, so
data stewards can fix upstream issues and nothing is silently dropped.
"""
from typing import Dict, Tuple
from pyspark.sql import DataFrame, functions as F

ENCOUNTER_RULES: Dict[str, str] = {
    "encounter_id_not_null": "encounter_id IS NOT NULL",
    "patient_id_not_null": "patient_id IS NOT NULL",
    "los_in_range": "los_days BETWEEN 1 AND 365",
    "discharge_after_admit": "discharge_ts >= admit_ts",
    "admit_not_in_future": "admit_ts <= current_timestamp()",
    "valid_admission_type": "admission_type IN ('EMERGENCY','URGENT','ELECTIVE')",
    "valid_disposition": "discharge_disposition IN ('HOME','HOME_HEALTH','SNF','REHAB','AMA')",
    "charges_non_negative": "total_charges >= 0",
}

LAB_RULES: Dict[str, str] = {
    "encounter_id_not_null": "encounter_id IS NOT NULL",
    "lab_value_not_null": "lab_value IS NOT NULL",
    "plausible_range": (
        "(lab_name = 'SODIUM' AND lab_value BETWEEN 100 AND 180) OR "
        "(lab_name = 'CREATININE' AND lab_value BETWEEN 0.1 AND 20) OR "
        "(lab_name = 'HEMOGLOBIN' AND lab_value BETWEEN 3 AND 20)"),
}

PATIENT_RULES: Dict[str, str] = {
    "patient_id_not_null": "patient_id IS NOT NULL",
    "dob_not_in_future": "dob <= current_date()",
    "valid_gender": "gender IN ('F','M','U')",
    "valid_insurance": "insurance_type IN ('MEDICARE','MEDICAID','COMMERCIAL','SELF_PAY')",
}

LABEL_RULES: Dict[str, str] = {
    "encounter_id_not_null": "encounter_id IS NOT NULL",
    "label_binary": "readmit_30d IN (0, 1)",
}


def split_by_rules(df: DataFrame, rules: Dict[str, str]) -> Tuple[DataFrame, DataFrame]:
    """Return (valid_df, quarantine_df). Quarantine rows carry `_failed_rules` (array<string>)."""
    checks = [F.when(~F.coalesce(F.expr(expr), F.lit(False)), F.lit(name)) for name, expr in rules.items()]
    flagged = df.withColumn("_failed_rules", F.filter(F.array(*checks), lambda x: x.isNotNull()))
    valid = flagged.filter(F.size("_failed_rules") == 0).drop("_failed_rules")
    bad = flagged.filter(F.size("_failed_rules") > 0)
    return valid, bad


def rule_failure_counts(quarantine: DataFrame, rules: Dict[str, str], table: str, total_rows: int) -> DataFrame:
    """One row per rule: how many rows failed it. Append to a metrics table to trend DQ over time."""
    aggs = [F.sum(F.array_contains("_failed_rules", name).cast("int")).alias(name) for name in rules]
    row = quarantine.agg(*aggs).first()
    spark = quarantine.sparkSession
    data = [(table, name, int(row[name] or 0), int(total_rows)) for name in rules]
    return (spark.createDataFrame(data, "table_name string, rule_name string, failed_rows long, total_rows long")
            .withColumn("run_ts", F.current_timestamp()))
