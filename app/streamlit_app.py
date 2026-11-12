
import os
import requests
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="CareInsight ML",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------
# 2. API CONFIGURATION
# ---------------------------------------

# Docker Compose supplies API_URL as an environment variable.
# Streamlit Cloud can supply it through app secrets.
API_URL = os.getenv("API_URL")

if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = "http://localhost:8000"

API_URL = API_URL.rstrip("/")


# ---------------------------------------
# 3. APPLICATION INTRODUCTION
# ---------------------------------------

st.title("🧠 CareInsight ML")

st.subheader(
    "Exploring caregiver support patterns "
    "through machine learning"
)

st.write(
    """
    **Welcome to CareInsight ML!**

    This application explores patterns associated with
    reported regular caregiver arrangements among adults
    with disabilities in India, using the Government of
    India's NSS 76th Round disability survey.

    Using a trained machine learning model, you can explore
    how selected survey characteristics affect the model's
    estimated probability of the positive caregiver-support
    survey outcome.
    """
)

st.info(
    "This is an educational research demonstration, "
    "not a clinical assessment or an eligibility tool."
)

st.markdown("### What can you explore?")

st.markdown(
    """
    - **Example predictions:** Adjust characteristics
      of a fictional survey profile.
    - **Machine learning inference:** See how a trained
      Gradient Boosting model processes the input.
    - **Model interpretation:** Understand what the
      estimated probability and predicted class mean.
    - **Research methodology:** Learn about the data,
      modeling process, and evaluation results.
    """
)

st.divider()


# ---------------------------------------
# 4. ABOUT THE MACHINE LEARNING MODEL
# ---------------------------------------

with st.expander("How does this machine learning model work?"):
    st.write(
        """
        The model was trained on data from India's
        NSS 76th Round disability survey.

        Four machine learning approaches were evaluated:

        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        - Stacking

        **Gradient Boosting** was selected for this
        demonstration because it achieved nearly the
        same test ROC-AUC as Stacking with a simpler
        inference pipeline.

        The saved model pipeline applies the same
        preprocessing transformations used during
        training before generating a prediction.

        **Model performance:**
        - Gradient Boosting test ROC-AUC: 0.7683
        - Stacking test ROC-AUC: 0.7694

        These results reflect historical held-out
        evaluation data, not validation on new visitors.
        """
    )


# ---------------------------------------
# 5. EXAMPLE SURVEY PROFILE
# ---------------------------------------

st.header("Explore an Example Survey Profile")

st.write(
    """
    Imagine a fictional adult whose characteristics
    are represented using variables from the survey.

    Adjust the available inputs and click
    **Generate Prediction**.

    The other model features remain fixed to
    predefined example values.

    Try changing one characteristic at a time
    and observing the resulting model estimate.
    """
)

age = st.slider(
    "Age of the example adult",
    min_value=18,
    max_value=100,
    value=51,
    help="Select an age between 18 and 100 years."
)

uses_transport = st.selectbox(
    "Uses public transport",
    options=[1.0, 0.0],
    format_func=lambda x: "Yes" if x == 1.0 else "No",
    help="Illustrative survey-coded public transport variable."
)

# The codebook meaning has not yet been verified.
# Keep this feature fixed until mappings are validated.
education = 1.0

st.caption(
    "Education level and other survey characteristics "
    "are fixed in this demonstration until their "
    "category mappings are fully validated."
)

with st.expander("View the fixed example characteristics"):
    st.write(
        """
        The prediction model expects 14 selected features.

        In this demonstration, you can change age and
        public transport use. The remaining characteristics
        are fixed for consistency.

        The fixed inputs include survey-coded disability
        indicators, principal activity status, relationship
        to the household head, education level,
        public-building access, and an engineered
        expenditure variable.

        These values represent an illustrative input
        combination, not a real survey participant.
        """
    )

st.divider()


# ---------------------------------------
# 6. BUILD MODEL INPUT
# ---------------------------------------

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


# ---------------------------------------
# 7. GENERATE PREDICTION
# ---------------------------------------

st.subheader("Generate a Model Prediction")

st.write(
    "Click the button below to send the example "
    "characteristics to the trained ML model."
)

if st.button("Generate Prediction", type="primary"):

    with st.spinner(
        "Contacting the prediction service..."
    ):

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=90
            )

            response.raise_for_status()

            result = response.json()

            probability = float(result["probability"])
            predicted_class = int(result["predicted_class"])

            # ---------------------------------------
            # 8. DISPLAY PREDICTION
            # ---------------------------------------

            st.success("Prediction completed!")

            st.subheader("Understanding the Prediction")

            st.metric(
                "Estimated Survey Outcome Probability",
                f"{probability:.2%}"
            )

            st.progress(
                max(0.0, min(probability, 1.0))
            )

            st.write(
                """
                This percentage is the model's estimated
                probability that the fictional input
                belongs to the positive class defined
                in the original survey analysis.

                It does not measure disability severity
                or determine whether someone needs care.
                """
            )

            # ---------------------------------------
            # 9. EXPLAIN THE PREDICTED CLASS
            # ---------------------------------------

            st.markdown("### How was the class determined?")

            if predicted_class == 1:
                st.info(
                    "The estimated probability is at or "
                    "above the 50% classification threshold, "
                    "so the model assigns class 1."
                )
            else:
                st.info(
                    "The estimated probability is below "
                    "the 50% classification threshold, "
                    "so the model assigns class 0."
                )

            st.write(
                f"**Predicted class:** {predicted_class}"
            )

            st.caption(
                "A classification threshold converts a "
                "probability into a binary model output. "
                "It is not a clinical decision threshold."
            )

            st.caption(
                "Model version: "
                f"{result.get('model_version', 'Not specified')}"
            )

            st.warning(
                "This estimate is based on a historical "
                "survey dataset and an imperfect statistical "
                "model. Do not use it for individual care "
                "decisions, diagnosis, or eligibility."
            )

        except requests.Timeout:
            st.error(
                "The prediction service took too long "
                "to respond. If the cloud backend was "
                "inactive, please try again."
            )

        except requests.RequestException as error:
            st.error(
                "The prediction service is currently "
                "unavailable. Please try again later."
            )
            st.caption(
                f"Technical details: {error}"
            )

        except (KeyError, TypeError, ValueError) as error:
            st.error(
                "The prediction service returned "
                "an unexpected response."
            )
            st.caption(
                f"Technical details: {error}"
            )


# ---------------------------------------
# 10. RESEARCH LIMITATIONS
# ---------------------------------------

st.divider()

st.header("Research Limitations")

st.write(
    """
    **Important considerations**

    - The model was developed using historical
      NSS 76th Round survey data.
    - Its predictions reflect patterns learned
      from the training dataset.
    - The model may not generalize to populations
      or circumstances outside that dataset.
    - Predictions are estimates, not verified
      individual outcomes.
    - Associations between survey characteristics
      and predicted outcomes do not establish
      cause-and-effect relationships.
    - The application uses fictional profiles
      and should not be used for medical decisions.

    **Research purpose:** To demonstrate applied
    machine learning, inference deployment, and
    responsible interpretation of model outputs.
    """
)


# ---------------------------------------
# 11. PROJECT LINKS
# ---------------------------------------

st.divider()

st.header("Explore the Engineering Project")

st.write(
    """
    CareInsight ML is built using:

    - Python and scikit-learn for machine learning
    - FastAPI for model serving
    - Streamlit for the user interface
    - Docker and Docker Compose for containerization
    - MLflow for experiment tracking
    - GitHub Actions for automated testing and CI
    - Render and Streamlit Community Cloud for hosting
    """
)

st.link_button(
    "View GitHub Repository",
    "https://github.com/macrosensor2022/careinsight-mlops"
)

st.caption(
    "CareInsight ML | Educational Machine Learning "
    "and MLOps Portfolio Project"
)

