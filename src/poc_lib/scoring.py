"""Spark helpers for scoring output normalisation and idempotent Delta writes."""
from typing import List
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


def upsert(spark: SparkSession, df: DataFrame, table: str, keys: List[str]) -> None:
    """Idempotent MERGE (create table on first run). Safe to re-run a failed job."""
    from delta.tables import DeltaTable
    if not spark.catalog.tableExists(table):
        df.write.format("delta").mode("overwrite").saveAsTable(table)
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        return
    cond = " AND ".join(f"t.{k} <=> s.{k}" for k in keys)
    (DeltaTable.forName(spark, table).alias("t").merge(df.alias("s"), cond)
        .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())


def to_inference_log(scored: DataFrame, model_name: str, version, thr_high: float, thr_med: float,
                     with_label: bool = False) -> DataFrame:
    """Normalise the output of fe.score_batch into the schema used for monitoring.

    fe.score_batch on a model logged with pyfunc_predict_fn="predict_proba" returns an
    array<double> [p0, p1]; if a plain float comes back we use it directly.
    """
    is_arr = isinstance(scored.schema["prediction"].dataType, T.ArrayType)
    p = F.col("prediction")[1] if is_arr else F.col("prediction").cast("double")
    out = (scored.withColumn("risk_score", p).drop("prediction")
           .withColumn("prediction_label", (F.col("risk_score") >= thr_high).cast("int"))
           .withColumn("risk_tier", F.when(F.col("risk_score") >= thr_high, "HIGH")
                                     .when(F.col("risk_score") >= thr_med, "MEDIUM").otherwise("LOW"))
           .withColumn("model_name", F.lit(model_name))
           .withColumn("model_version", F.lit(str(version)))
           .withColumn("scored_at", F.current_timestamp()))
    if not with_label and "readmit_30d" not in out.columns:
        out = out.withColumn("readmit_30d", F.lit(None).cast("int"))
    return out


def positive_prob_pdf(pdf, col: str = "prediction"):
    """pandas version of the above (array-or-scalar tolerant)."""
    import numpy as np
    first = pdf[col].iloc[0]
    if hasattr(first, "__len__") and not isinstance(first, str):
        return np.array([float(x[1]) for x in pdf[col]])
    return pdf[col].astype(float).to_numpy()
