
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CareInsight ML",
    page_icon="🧠",
    layout="centered"
)

st.title("CareInsight ML")
st.write(
    "An interactive ML research application exploring "
    "caregiver support patterns among adults with "
    "disabilities in India."
)

st.info(
    "Educational research only. "
    "Not intended for clinical or individual care decisions."
)

st.subheader("Example Survey Profile")

age = st.slider("Age", 18, 100, 51)

uses_transport = st.selectbox(
    "Uses public transport",
    options=[1.0, 0.0],
    format_func=lambda x: "Yes" if x == 1.0 else "No"
)

education = st.selectbox(
    "Education level (survey code)",
    options=[1.0],
    help="More survey categories will be added after validating the codebook."
)

st.caption(
    "Other model features are fixed to a synthetic example. "
    "Additional inputs will be enabled after survey-code validation."
)

if st.button("Predict", type="primary"):

    features = {
        "dis_seeing": 0.0,
        "dis_loss_sensation": 0.0,
        "dis_understanding": 0.0,
        "dis_hearing": 0.0,
        "uses_public_transport": uses_transport,
        "multiple_disability": 0.0,
        "principal_activity_status": 95.0,
        "dis_deformity": 0.0,
        "dis_anxiety_mood": 0.0,
        "relation_to_head": 1.0,
        "age": float(age),
        "education_level": education,
        "accesses_public_building": 0.0,
        "log_monthly_exp": 9.093918908958932
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=15
        )

        response.raise_for_status()
        result = response.json()

        st.success("Prediction completed!")

        st.metric(
            "Estimated Positive-Outcome Probability",
            f"{result['probability'] * 100:.2f}%"
        )

        st.write(
            "Predicted class:",
            result["predicted_class"]
        )

        st.caption(
            f"Model version: {result['model_version']}"
        )

    except requests.RequestException as error:
        st.error(f"Could not reach prediction API: {error}")
