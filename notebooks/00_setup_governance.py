# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup: Unity Catalog, security & PHI governance
# MAGIC Covers: catalog/schema/volumes · RBAC grants · column masks · row filters · tagging ·
# MAGIC audit & lineage queries (HIPAA-style controls).
# MAGIC
# MAGIC > Run once per environment as a catalog owner / metastore admin.
# MAGIC > Account groups used below (create them in the account console / SCIM):
# MAGIC > `data_engineers`, `data_scientists`, `ml_engineers`, `phi_admins`, `care_mgmt_apps`

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
# MAGIC %md ## 1. Catalog, schema, volumes

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'Healthcare ML POC ({ENV})'")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ} COMMENT '30-day readmission risk: data, features, models'")
for v, c in [("raw", "Landing zone for hospital feeds"), ("checkpoints", "Auto Loader schemas/checkpoints"),
             ("artifacts", "Plots, reports, exports")]:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.{v} COMMENT '{c}'")

# COMMAND ----------
# MAGIC %md ## 2. Role-based access control (least privilege)
# MAGIC | Group | Can do |
# MAGIC |---|---|
# MAGIC | data_engineers | write bronze/silver, run pipelines |
# MAGIC | data_scientists | read silver/gold + features, write experiments/models |
# MAGIC | ml_engineers | create models, serving endpoints, jobs |
# MAGIC | phi_admins | see unmasked PHI |
# MAGIC | care_mgmt_apps | read predictions + query the endpoint only (no PHI tables) |

# COMMAND ----------

def safe_sql(stmt):
    try:
        spark.sql(stmt)
    except Exception as e:  # group may not exist in this workspace
        print("SKIPPED:", stmt[:90], "->", str(e)[:120])

safe_sql(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `data_engineers`")
safe_sql(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `data_scientists`")
safe_sql(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `ml_engineers`")
safe_sql(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `care_mgmt_apps`")
safe_sql(f"GRANT USE SCHEMA, SELECT, MODIFY, CREATE TABLE, CREATE VOLUME ON SCHEMA {FQ} TO `data_engineers`")
safe_sql(f"GRANT USE SCHEMA, SELECT, CREATE TABLE, CREATE MODEL, CREATE FUNCTION ON SCHEMA {FQ} TO `data_scientists`")
safe_sql(f"GRANT USE SCHEMA, SELECT, CREATE TABLE, CREATE MODEL, CREATE FUNCTION, EXECUTE ON SCHEMA {FQ} TO `ml_engineers`")
safe_sql(f"GRANT USE SCHEMA ON SCHEMA {FQ} TO `care_mgmt_apps`")
# care-management apps only get the predictions/worklist objects (granted in notebook 07)

# COMMAND ----------
# MAGIC %md ## 3. Column masks & row filters (created here, attached in notebook 02 once tables exist)
# MAGIC * **Column mask** - direct identifiers (name, SSN, DOB, phone, ZIP) are redacted unless the caller is in `phi_admins`.
# MAGIC * **Row filter** - hospital-level data isolation: a user only sees rows of hospitals they are entitled to.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.mask_phi(v STRING)
RETURNS STRING
COMMENT 'Redacts PHI for everyone except phi_admins'
RETURN CASE WHEN is_account_group_member('phi_admins') THEN v ELSE '*** REDACTED ***' END
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.mask_ssn(v STRING)
RETURNS STRING
RETURN CASE WHEN is_account_group_member('phi_admins') THEN v
            ELSE concat('***-**-', right(v, 4)) END
""")

# Entitlement table: which user/group may see which hospital (maintained by data governance team)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FQ}.ref_hospital_entitlements (group_name STRING, hospital_id INT)
COMMENT 'Row-level security entitlements'
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FQ}.hospital_row_filter(hospital_id INT)
RETURNS BOOLEAN
COMMENT 'phi_admins / ml_engineers see all hospitals, others only entitled ones'
RETURN is_account_group_member('phi_admins') OR is_account_group_member('ml_engineers')
       OR hospital_id IN (1, 2, 3)   -- POC stub; replace with entitlement-table lookup in production
""")

# COMMAND ----------
# MAGIC %md ## 4. Governance audit queries (system tables) - HIPAA "who accessed what"
# MAGIC Requires system tables to be enabled by an account admin.

# COMMAND ----------

AUDIT_SQL = f"""
SELECT event_time, user_identity.email AS user, action_name, request_params.full_name_arg AS object
FROM system.access.audit
WHERE service_name = 'unityCatalog'
  AND request_params.full_name_arg LIKE '{CATALOG}.{SCHEMA}.%'
  AND action_name IN ('getTable','commandSubmit','generateTemporaryTableCredential')
  AND event_date >= current_date() - INTERVAL 7 DAYS
ORDER BY event_time DESC LIMIT 200
"""
LINEAGE_SQL = f"""
SELECT source_table_full_name, target_table_full_name, entity_type, created_by, event_time
FROM system.access.table_lineage
WHERE target_table_full_name LIKE '{CATALOG}.{SCHEMA}.%' ORDER BY event_time DESC LIMIT 200
"""
# display(spark.sql(AUDIT_SQL)); display(spark.sql(LINEAGE_SQL))   # uncomment where system tables are enabled
print("Audit & lineage queries ready (see README -> Governance).")

# COMMAND ----------
# MAGIC %md ## 5. Account-level controls to enable for real PHI (outside this notebook)
# MAGIC * Sign a **BAA** with Databricks + your cloud; enable the **Compliance Security Profile (HIPAA)**
# MAGIC * Customer-managed keys for workspace storage / managed services; private link, no public IPs
# MAGIC * Secrets in a secret scope / vault, never in code; SCIM + SSO; service principals for jobs
# MAGIC * Data retention + legal-hold policy on Delta (`delta.logRetentionDuration`, VACUUM policy)
# MAGIC * De-identification (HIPAA Safe Harbor / Expert Determination) before using data outside the covered entity
