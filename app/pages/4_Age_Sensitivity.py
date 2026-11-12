
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Age Sensitivity | CareInsight",
    page_icon="📉",
    layout="wide"
)

# API configuration
API_URL = os.getenv("API_URL")

if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = "http://localhost:8000"

API_URL = API_URL.rstrip("/")


def build_features(age, transport):
    """Create the same fictional profile used elsewhere."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def calculate_age_sensitivity(transport):
    """Call FastAPI for a range of fictional ages."""

    ages = sorted(set(range(18, 101, 5)) | {51, 100})
    results = []

    with requests.Session() as session:
        for age in ages:
            response = session.post(
                f"{API_URL}/predict",
                json={
                    "features": build_features(age, transport)
                },
                timeout=90
            )

            response.raise_for_status()
            result = response.json()

            probability = float(result["probability"])

            if not 0 <= probability <= 1:
                raise ValueError("Invalid API probability")

            results.append({
                "Age": age,
                "Estimated Probability (%)": probability * 100
            })

    return pd.DataFrame(results)


# Page introduction
st.title("📉 Age Sensitivity Explorer")

st.write("""
This page explores how the trained Gradient Boosting
model responds when the age input changes.

All other survey characteristics are held fixed.

The chart shows model-estimated probabilities, not
observed caregiver-support rates across age groups.
""")

st.info(
    "This is a model sensitivity experiment, "
    "not a causal analysis or a clinical assessment."
)

st.divider()

# Controls
st.header("Configure the Experiment")

with st.form("age_sensitivity_form"):

    transport = st.selectbox(
        "Public transport use",
        options=[1.0, 0.0],
        format_func=lambda x: "Yes" if x == 1.0 else "No"
    )

    submitted = st.form_submit_button(
        "Generate Age Sensitivity Chart",
        type="primary"
    )


if submitted:
    try:
        with st.spinner(
            "Generating model predictions across ages..."
        ):
            df = calculate_age_sensitivity(transport)

        st.success("Sensitivity analysis completed!")

        st.header("Model Response Across Ages")

        st.line_chart(
            df.set_index("Age"),
            x_label="Age (years)",
            y_label="Estimated probability (%)",
            use_container_width=True
        )

        st.caption(
            "Each point is a separate inference request "
            "using the trained model. The chart connects "
            "evaluated ages for visualization."
        )

        # Summary metrics
        min_row = df.loc[
            df["Estimated Probability (%)"].idxmin()
        ]

        max_row = df.loc[
            df["Estimated Probability (%)"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Lowest Estimate",
                f"{min_row['Estimated Probability (%)']:.2f}%"
            )

        with col2:
            st.metric(
                "Highest Estimate",
                f"{max_row['Estimated Probability (%)']:.2f}%"
            )

        with col3:
            st.metric(
                "Ages Evaluated",
                len(df)
            )

        st.write(
            f"The lowest sampled estimate occurs at "
            f"age **{int(min_row['Age'])}**, and "
            f"the highest at age **{int(max_row['Age'])}**."
        )

        st.subheader("Detailed Results")

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="careinsight_age_sensitivity.csv",
            mime="text/csv"
        )

        st.warning(
            "The observed differences reflect model "
            "predictions on synthetic input combinations. "
            "They do not demonstrate that age causes "
            "caregiver-support outcomes to change. "
            "Some combinations may not be representative "
            "of real survey participants."
        )

    except requests.Timeout:
        st.error(
            "The API timed out. The free cloud backend "
            "may be waking up. Please try again."
        )

    except requests.RequestException as error:
        st.error("Unable to reach the prediction API.")
        st.caption(str(error))

    except (KeyError, TypeError, ValueError) as error:
        st.error("Invalid response from the model service.")
        st.caption(str(error))


st.divider()

with st.expander("How does this experiment work?"):
    st.markdown("""
    1. Create a fictional survey profile.
    2. Hold 13 of the 14 model inputs constant.
    3. Change age across selected values from 18 to 100.
    4. Request a prediction for each profile.
    5. Plot the returned probabilities.

    The experiment does not retrain the model.

    **Important:** The model's responses may reflect
    correlations and interactions learned from historical
    data. They should not be interpreted as causal
    relationships or calibrated individual risks.
    """)
