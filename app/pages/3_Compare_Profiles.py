
import os
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Compare Profiles | CareInsight ML",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------
# API CONFIGURATION
# ---------------------------------------

API_URL = os.getenv("API_URL")

if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = "http://localhost:8000"

API_URL = API_URL.rstrip("/")


# ---------------------------------------
# BUILD A FICTIONAL SURVEY PROFILE
# ---------------------------------------

def build_features(age, transport):
    return {
        "dis_seeing": 0.0,
        "dis_loss_sensation": 0.0,
        "dis_understanding": 0.0,
        "dis_hearing": 0.0,
        "uses_public_transport": float(transport),
        "multiple_disability": 0.0,
        "principal_activity_status": 95.0,
        "dis_deformity": 0.0,
        "dis_anxiety_mood": 0.0,
        "relation_to_head": 1.0,
        "age": float(age),
        "education_level": 1.0,
        "accesses_public_building": 0.0,
        "log_monthly_exp": 9.093918908958932
    }


# ---------------------------------------
# CALL THE DEPLOYED ML MODEL
# ---------------------------------------

def get_prediction(features):
    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features},
        timeout=90
    )

    response.raise_for_status()

    result = response.json()

    probability = float(result["probability"])

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "API returned an invalid probability."
        )

    return probability


# ---------------------------------------
# PAGE INTRODUCTION
# ---------------------------------------

st.title("📈 Compare Two Survey Profiles")

st.write("""
Explore how CareInsight ML responds to differences
between two fictional survey profiles.

Adjust age and public transport use for each profile.
The remaining model inputs are kept identical.

The application sends both profiles to the deployed
Gradient Boosting model and compares its outputs.
""")

st.info(
    "This feature explores model behavior. "
    "It does not measure the causal effect of "
    "changing a person's characteristics."
)

st.divider()


# ---------------------------------------
# USER INPUTS
# ---------------------------------------

with st.form("comparison_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profile A")

        age_a = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            key="age_a"
        )

        transport_a = st.selectbox(
            "Uses public transport",
            options=[1.0, 0.0],
            format_func=lambda x: (
                "Yes" if x == 1.0 else "No"
            ),
            key="transport_a"
        )

    with col2:
        st.subheader("Profile B")

        age_b = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=65,
            key="age_b"
        )

        transport_b = st.selectbox(
            "Uses public transport",
            options=[0.0, 1.0],
            format_func=lambda x: (
                "Yes" if x == 1.0 else "No"
            ),
            key="transport_b"
        )

    submitted = st.form_submit_button(
        "Compare Predictions",
        type="primary"
    )


# ---------------------------------------
# GENERATE COMPARISON
# ---------------------------------------

if submitted:

    features_a = build_features(
        age_a,
        transport_a
    )

    features_b = build_features(
        age_b,
        transport_b
    )

    try:
        with st.spinner(
            "Generating predictions for both profiles..."
        ):

            probability_a = get_prediction(features_a)

            probability_b = get_prediction(features_b)

        st.success("Comparison completed!")

        st.divider()

        # ---------------------------------------
        # RESULTS
        # ---------------------------------------

        st.header("Prediction Comparison")

        result_a, result_b = st.columns(2)

        with result_a:
            st.metric(
                "Profile A",
                f"{probability_a:.2%}"
            )

            st.caption(
                f"Age: {age_a} | "
                f"Public transport: "
                f"{'Yes' if transport_a else 'No'}"
            )

        with result_b:
            st.metric(
                "Profile B",
                f"{probability_b:.2%}"
            )

            st.caption(
                f"Age: {age_b} | "
                f"Public transport: "
                f"{'Yes' if transport_b else 'No'}"
            )

        st.subheader("Visual Comparison")

        comparison = pd.DataFrame({
            "Profile": ["Profile A", "Profile B"],
            "Estimated Probability (%)": [
                probability_a * 100,
                probability_b * 100
            ]
        })

        st.bar_chart(
            comparison.set_index("Profile"),
            y_label="Estimated probability (%)",
            use_container_width=True
        )

        # ---------------------------------------
        # EXPLAIN THE DIFFERENCE
        # ---------------------------------------

        difference = (
            probability_b - probability_a
        ) * 100

        st.subheader("Understanding the Difference")

        if difference > 0:
            st.write(
                f"Profile B has a model-estimated "
                f"probability **{difference:.2f} "
                f"percentage points higher** than Profile A."
            )

        elif difference < 0:
            st.write(
                f"Profile B has a model-estimated "
                f"probability **{abs(difference):.2f} "
                f"percentage points lower** than Profile A."
            )

        else:
            st.write(
                "Both profiles have the same "
                "model-estimated probability."
            )

        st.info(
            "These differences describe the trained "
            "model's responses to the selected inputs. "
            "They do not establish causation, fairness, "
            "or actual individual outcomes."
        )

    except requests.Timeout:
        st.error(
            "The prediction API timed out. "
            "If the free backend was inactive, "
            "please try again."
        )

    except requests.RequestException as error:
        st.error(
            "Could not contact the prediction API."
        )
        st.caption(str(error))

    except (KeyError, TypeError, ValueError) as error:
        st.error(
            "The prediction API returned an "
            "unexpected result."
        )
        st.caption(str(error))


# ---------------------------------------
# RESEARCH LIMITATIONS
# ---------------------------------------

st.divider()

with st.expander("Important interpretation notes"):
    st.write("""
    - The model was trained using historical survey data.
    - Both profiles are fictional.
    - Only age and public transport use can be changed.
    - The other twelve model inputs remain fixed.
    - Differences are associations learned by the model,
      not causal effects.
    - Model probabilities have not been demonstrated
      to be calibrated individual risks.
    - The application must not be used to determine
      medical or personal care requirements.
    """)

st.caption(
    "CareInsight ML | Interactive Model Exploration"
)
