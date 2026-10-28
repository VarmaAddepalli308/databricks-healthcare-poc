# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 08 · Real-time deployment on Mosaic AI Model Serving
# MAGIC Covers: endpoint create/update (idempotent) · inference table (request/response logging) · AI Gateway rate limits ·
# MAGIC permissions · smoke test · **load test (p50/p95/p99, errors)** · **A/B (champion vs challenger) traffic split** ·
# MAGIC rollback · parsing payload logs for monitoring · optional feature-lookup serving.
# MAGIC
# MAGIC **Option B (default)** serves the self-contained `*_payload` model: the caller sends the full feature vector
# MAGIC (the hospital's EHR integration / feature service computes it). Simple, fast, no online store needed.
# MAGIC **Option A** (end of notebook) serves the feature-store model and looks up history features online.
# MAGIC
# MAGIC > SDK class names move between `databricks-sdk` versions; if an import fails, `%pip install -U databricks-sdk` and check the serving docs.

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json, time, statistics, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (ServedEntityInput, TrafficConfig, Route, AiGatewayInferenceTableConfig,
                                            AiGatewayRateLimit, AiGatewayRateLimitKey, AiGatewayRateLimitRenewalPeriod,
                                            AiGatewayUsageTrackingConfig, ServingEndpointAccessControlRequest,
                                            ServingEndpointPermissionLevel)

w, client = WorkspaceClient(), MlflowClient()
champ_v = client.get_model_version_by_alias(PAYLOAD_MODEL_NAME, "Champion").version
print(f"Deploying {PAYLOAD_MODEL_NAME} v{champ_v} -> endpoint {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md ## 1. Create or update the endpoint (idempotent)
# MAGIC * `scale_to_zero` only outside prod (cold start would break a latency SLO)
# MAGIC * `min/max_provisioned_concurrency` give autoscaling headroom; size to expected QPS × latency

# COMMAND ----------

# DBTITLE 1,Create or update the endpoint
def entity(name, version, scale_to_zero):
    return ServedEntityInput(name=name, entity_name=PAYLOAD_MODEL_NAME, entity_version=str(version),
                             workload_size="Small", scale_to_zero_enabled=scale_to_zero,
                             environment_vars={"ENV": ENV})

served = [entity("champion", champ_v, scale_to_zero=(ENV != "prod"))]
traffic = TrafficConfig(routes=[Route(served_model_name="champion", traffic_percentage=100)])

try:
    w.serving_endpoints.get(ENDPOINT_NAME)
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT_NAME, served_entities=served, traffic_config=traffic)
    print("endpoint updated")
except Exception as e:
    if "not exist" in str(e).lower() or "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
        from databricks.sdk.service.serving import EndpointCoreConfigInput
        w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME,
                                            config=EndpointCoreConfigInput(name=ENDPOINT_NAME,
                                                                           served_entities=served, traffic_config=traffic),
                                            tags=[])
        print("endpoint created")
    else:
        raise

# COMMAND ----------

# MAGIC %md ## 2. AI Gateway: inference table, rate limits, usage tracking

# COMMAND ----------

# DBTITLE 1,AI Gateway: inference table, rate limits, usage tracking
try:
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT_NAME,
        inference_table_config=AiGatewayInferenceTableConfig(catalog_name=CATALOG, schema_name=SCHEMA,
                                                             table_name_prefix="readmission_endpoint", enabled=True),
        rate_limits=[AiGatewayRateLimit(calls=600, renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
                                        key=AiGatewayRateLimitKey.ENDPOINT)],
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))
    print("inference table -> %s.readmission_endpoint_payload" % FQ)
except Exception as e:
    if "inference table" in str(e).lower():
        # workspace does not support inference tables for this endpoint type - apply rate limits + usage tracking without logging
        w.serving_endpoints.put_ai_gateway(
            name=ENDPOINT_NAME,
            rate_limits=[AiGatewayRateLimit(calls=600, renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
                                            key=AiGatewayRateLimitKey.ENDPOINT)],
            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))
        print("SKIPPED inference table (not supported for this endpoint type in this workspace) - rate limits + usage tracking applied")
    else:
        raise

# COMMAND ----------

# MAGIC %md ## 3. Permissions - only the care-management app group may query; ML engineers manage

# COMMAND ----------

ep = w.serving_endpoints.get(ENDPOINT_NAME)
try:
    w.serving_endpoints.update_permissions(serving_endpoint_id=ep.id, access_control_list=[
        ServingEndpointAccessControlRequest(group_name="care_mgmt_apps", permission_level=ServingEndpointPermissionLevel.CAN_QUERY),
        ServingEndpointAccessControlRequest(group_name="ml_engineers", permission_level=ServingEndpointPermissionLevel.CAN_MANAGE)])
except Exception as e:
    print("SKIPPED permissions (group missing?):", str(e)[:150])

# COMMAND ----------

# MAGIC %md ## 4. Smoke test with real feature rows (signature comes from the registered model)

# COMMAND ----------

sig = mlflow.models.get_model_info(f"models:/{PAYLOAD_MODEL_NAME}@Champion").signature
input_cols = sig.inputs.input_names()
sample = (spark.table(T["baseline"]).select(*input_cols).limit(200).toPandas()
          .replace({np.nan: None}))
records = json.loads(sample.head(3).to_json(orient="records"))
resp = w.serving_endpoints.query(ENDPOINT_NAME, dataframe_records=records)
print("predictions [p(no readmit), p(readmit)]:", resp.predictions)
assert len(resp.predictions) == 3 and all(0 <= p[1] <= 1 for p in resp.predictions)

# COMMAND ----------

# MAGIC %md ## 5. Load test - SLO: p95 < 300 ms, error rate < 1%
# MAGIC Use a proper tool (Locust/k6) for sustained tests; this is a quick gate that runs in the deploy job.

# COMMAND ----------

# DBTITLE 1,Load test (SLO gate)
payloads = json.loads(sample.to_json(orient="records"))

def one_call(i):
    t0 = time.perf_counter()
    try:
        w.serving_endpoints.query(ENDPOINT_NAME, dataframe_records=[payloads[i % len(payloads)]])
        return (time.perf_counter() - t0) * 1000, 0
    except Exception:
        return (time.perf_counter() - t0) * 1000, 1

N, CONC = 300, 16
with ThreadPoolExecutor(CONC) as ex:
    out = list(ex.map(one_call, range(N)))
lat = [o[0] for o in out if o[1] == 0]; err = sum(o[1] for o in out) / N
q = lambda p: float(np.percentile(lat, p)) if lat else float("nan")
report = {"requests": N, "concurrency": CONC, "p50_ms": q(50), "p95_ms": q(95), "p99_ms": q(99), "error_rate": err}
print(report)
if ENV == "prod":
    assert err < 0.01 and q(95) < 300, f"SLO violated: {report}"
else:
    # dev endpoint is Small + scale-to-zero (sized down on purpose); the p95 SLO is a prod gate where compute is sized for it
    print("dev: SLO advisory only - hard p95 < 300 ms gate is enforced in prod (use Locust/k6 for sustained prod tests)")

# COMMAND ----------

# MAGIC %md ## 6. A/B test: champion 90% / challenger 10%
# MAGIC Run only when a *Challenger* exists that has passed offline validation but you want online evidence
# MAGIC (latency, score distribution, then outcomes after labels mature). Set `RUN_AB = True`.

# COMMAND ----------

RUN_AB = False
if RUN_AB:
    chal_v = client.get_model_version_by_alias(PAYLOAD_MODEL_NAME, "Challenger").version
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[entity("champion", champ_v, ENV != "prod"), entity("challenger", chal_v, ENV != "prod")],
        traffic_config=TrafficConfig(routes=[Route(served_model_name="champion", traffic_percentage=90),
                                             Route(served_model_name="challenger", traffic_percentage=10)]))
    print("A/B live. Compare by served_entity_id in the inference table; decide after label maturation (30 days).")
    # ROLLBACK / PROMOTE:  call update_config_and_wait again with a single entity at 100% traffic.

# COMMAND ----------

# MAGIC %md ## 7. Turn the inference table into monitoring-ready rows
# MAGIC The payload table stores raw JSON request/response. Latency/error KPIs come straight from its columns.

# COMMAND ----------

from pyspark.sql import functions as F
payload_tbl = f"{FQ}.readmission_endpoint_payload"
if spark.catalog.tableExists(payload_tbl):
    raw = spark.table(payload_tbl)
    ops = (raw.select(F.col("request_time"), F.col("status_code"), F.col("execution_duration_ms"),
                      F.from_json("response", "predictions array<array<double>>").alias("r"))
              .select("request_time", "status_code", "execution_duration_ms",
                      F.element_at(F.col("r.predictions"), 1)[1].alias("risk_score")))
    display(ops.groupBy(F.window("request_time", "1 hour")).agg(
        F.count("*").alias("requests"), F.avg("execution_duration_ms").alias("avg_ms"),
        F.expr("percentile_approx(execution_duration_ms, 0.95)").alias("p95_ms"),
        F.avg((F.col("status_code") != 200).cast("int")).alias("error_rate")).orderBy("window"))
else:
    print("Inference table not populated yet (it can take several minutes after the first requests).")

# COMMAND ----------

# MAGIC %md ## 8. Option A - feature-store model with online lookups (only if you published an online store, see notebook 04 §8)
# MAGIC The request would then contain only the keys + encounter-level fields; the endpoint fetches history features itself.
# MAGIC ```python
# MAGIC ServedEntityInput(name="champion_fs", entity_name=MODEL_NAME, entity_version=champ_fs_version,
# MAGIC                   workload_size="Small", scale_to_zero_enabled=False)
# MAGIC ```
# MAGIC Trade-off: fewer moving parts for callers vs. an extra online store to operate/pay for.
# MAGIC
# MAGIC ## 9. Client application
# MAGIC `app/app.py` is a minimal **Databricks App** (Streamlit) calling this endpoint with the app's service principal.