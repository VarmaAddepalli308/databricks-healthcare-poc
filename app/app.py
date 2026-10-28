"""Minimal Databricks App (Streamlit): care-manager what-if tool calling the serving endpoint.

The app runs as its own service principal (granted CAN_QUERY on the endpoint) -> no user tokens or secrets in code.
Deploy:  databricks apps create readmission-app  &&  databricks apps deploy readmission-app --source-code-path ./app
"""
import os
import streamlit as st
from databricks.sdk import WorkspaceClient

ENDPOINT = os.getenv("SERVING_ENDPOINT", "readmission-risk-dev")
w = WorkspaceClient()

st.title("30-day readmission risk (decision support)")
st.caption("Not a diagnostic tool. A clinician must review every recommendation.")

c1, c2 = st.columns(2)
with c1:
    los = st.number_input("Length of stay (days)", 1, 60, 5)
    admission = st.selectbox("Admission type", ["EMERGENCY", "URGENT", "ELECTIVE"])
    dispo = st.selectbox("Discharge disposition", ["HOME", "HOME_HEALTH", "SNF", "REHAB", "AMA"])
    dx = st.selectbox("Diagnosis group", ["CHF", "COPD", "PNEUMONIA", "DIABETES", "CKD", "SEPSIS", "HTN", "GI_BLEED", "CHEST_PAIN", "OTHER"])
    chronic = st.slider("Chronic conditions", 0, 8, 3)
with c2:
    creat = st.number_input("Creatinine (mg/dL)", 0.2, 10.0, 1.2)
    sodium = st.number_input("Sodium (mmol/L)", 110.0, 170.0, 138.0)
    hgb = st.number_input("Hemoglobin (g/dL)", 4.0, 20.0, 12.0)
    prior90 = st.number_input("Admissions in last 90 days", 0, 20, 1)
    age = st.number_input("Age", 18, 90, 70)

if st.button("Score"):
    # NOTE: the record must contain every column in the model signature (see notebook 08 -> sig.inputs.input_names()).
    # Remaining columns are filled with neutral defaults here for brevity.
    rec = {"los_days": los, "num_procedures": 1, "num_medications": 6 + 2 * chronic, "total_charges": los * 2500.0,
           "sodium": sodium, "creatinine": creat, "hemoglobin": hgb, "is_emergency": int(admission == "EMERGENCY"),
           "is_weekend_admit": 0, "admit_month": 6, "sodium_low": int(sodium < 135), "anemia_flag": int(hgb < 12),
           "aki_flag": int(creat > 1.5), "polypharmacy": int((6 + 2 * chronic) >= 10), "admission_type": admission,
           "discharge_disposition": dispo, "dx_group": dx, "hospital_id": "1", "prior_admits_30d": 0,
           "prior_admits_90d": prior90, "prior_admits_180d": prior90, "prior_admits_365d": prior90,
           "prior_ed_admits_365d": prior90, "prior_avg_los_365d": 4.0, "prior_charges_365d": prior90 * 10000.0,
           "days_since_last_discharge": 60, "prior_total_admits": prior90, "prior_distinct_dx": 1 if prior90 else 0,
           "age_at_admit": age, "gender": "F", "insurance_type": "MEDICARE", "chronic_conditions_count": chronic,
           "charges_per_day": 2500.0}
    resp = w.serving_endpoints.query(ENDPOINT, dataframe_records=[rec])
    p = resp.predictions[0][1]
    st.metric("Readmission probability", f"{p:.1%}")
    st.progress(min(p * 3, 1.0))
