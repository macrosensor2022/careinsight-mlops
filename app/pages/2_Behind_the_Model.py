
import streamlit as st

st.set_page_config(
    page_title="Behind the Model | CareInsight",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Behind the Model")

st.subheader("Research Objective")

st.write("""
CareInsight ML explores patterns associated with
reported regular caregiver arrangements among
adults with disabilities in India.

The project uses the NSS 76th Round disability survey
and evaluates several supervised learning approaches.
""")

st.subheader("Machine Learning Pipeline")

st.markdown("""
1. Survey data cleaning and preparation
2. Exploratory data analysis
3. Feature engineering and selection
4. Preprocessing
5. Model training and evaluation
6. Model selection
7. Model serialization using Joblib
8. Deployment through a prediction API
""")

st.subheader("Deployment Architecture")

st.code("""
User
  |
  v
Streamlit Cloud
  |
  v
FastAPI on Render
  |
  v
Scikit-learn Pipeline
  |
  v
Prediction Response
""", language="text")

st.subheader("MLOps Components")

st.markdown("""
- **Docker:** Containerizes the application.
- **Docker Compose:** Runs local services together.
- **MLflow:** Stores experiment metrics and artifacts.
- **GitHub Actions:** Runs automated API tests and builds.
- **Render:** Hosts the FastAPI backend.
- **Streamlit Cloud:** Hosts the public interface.
""")

st.subheader("Important Limitations")

st.warning("""
This application is an educational research
demonstration, not a clinical assessment.

The model uses historical survey data and has
not been validated for individual care decisions.
Predictions do not establish causation.
""")

st.link_button(
    "View GitHub Repository",
    "https://github.com/macrosensor2022/careinsight-mlops"
)
