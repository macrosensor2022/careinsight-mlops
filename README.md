
# CareInsight ML
### End-to-End Machine Learning & MLOps Application

[Live Application](https://careinsight-mlops.streamlit.app/) |
[FastAPI Documentation](https://careinsight-api.onrender.com/docs)

CareInsight ML is an end-to-end machine learning application
that explores patterns associated with reported regular
caregiver arrangements among adults with disabilities in India.

Built using the NSS 76th Round disability survey, the project
combines machine learning experimentation, API-based inference,
containerization, automated testing, and cloud deployment.

> **Research disclaimer:** This application is an educational
> research demonstration. It is not a medical, clinical, or
> individual care eligibility assessment tool.

---

## 1. Problem Statement

Understanding caregiver-support patterns among adults with
disabilities is an important research challenge.

CareInsight explores whether selected characteristics from
a historical disability survey can help predict a
survey-defined caregiver-support outcome.

The project focuses on both machine learning research and
deploying a usable, reproducible prediction service.

## 2. Live Application

**Try CareInsight:**

https://careinsight-mlops.streamlit.app/

The application provides:

- **Explore Predictions:** Generate a prediction for a
  fictional survey profile.
- **Model Performance:** Compare historical test-set
  results across four machine learning models.
- **Behind the Model:** Explore the ML workflow and
  MLOps architecture.
- **Compare Profiles:** Compare predictions for two
  fictional input profiles.
- **Age Sensitivity:** Explore model responses across
  different age inputs while holding other features fixed.

The application uses synthetic input combinations for
demonstration, not real visitor health records.

## 3. Dataset

**Source:** National Sample Survey (NSS), 76th Round,
India — disability-related survey data.

The model uses 14 selected features, including age,
survey-coded disability indicators, public transport use,
education level, household relationship, activity status,
and an expenditure-related engineered feature.

Survey category mappings should be verified against
the official documentation before extending the
interactive inputs.

## 4. Machine Learning Models

Four approaches were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Stacking

### Historical Test Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7628 | 0.7386 | 0.7992 | 0.6866 |
| Random Forest | 0.7678 | 0.7379 | 0.8063 | 0.6802 |
| Gradient Boosting | 0.7683 | 0.7344 | 0.8114 | 0.6707 |
| Stacking | 0.7694 | 0.7361 | 0.8101 | 0.6745 |

**Deployed model:** Gradient Boosting.

Stacking achieved the highest test ROC-AUC, but Gradient
Boosting achieved nearly identical discrimination with
a simpler inference pipeline.

These metrics reflect the original held-out evaluation.
They are not live production performance measurements.

## 5. System Architecture

```text
                 VISITOR
                    |
                    v
          Streamlit Cloud
          (User Interface)
                    |
               HTTPS API
                    |
                    v
            Render Web Service
                (FastAPI)
                    |
                    v
         Saved scikit-learn Pipeline
                    |
                    v
            Prediction Response


       DEVELOPMENT & MLOPS WORKFLOW

       Training / Evaluation
                    |
                    v
          MLflow Experiment Tracking
                    |
                    v
           Saved Model Artifacts
                    |
                    v
        FastAPI + Streamlit Application
                    |
                    v
             GitHub Repository
                    |
                    v
           GitHub Actions CI
          (Tests + Docker Builds)

       Docker Compose: local environment
```

MLflow is used for local experiment tracking. It is not
exposed as a public production monitoring service.

## 6. Technology Stack

| Component | Technology |
|---|---|
| Machine Learning | Python, scikit-learn |
| Model Serialization | Joblib |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker, Docker Compose |
| Automated Testing | pytest |
| Continuous Integration | GitHub Actions |
| Backend Hosting | Render |
| Frontend Hosting | Streamlit Community Cloud |

## 7. Run Locally

### Prerequisites

- Git
- Docker Desktop
- Docker Compose

Clone the repository:

```bash
git clone https://github.com/macrosensor2022/careinsight-mlops.git
cd careinsight-mlops
```

Build and start the local services:

```bash
docker compose up -d --build
```

Open:

- Streamlit: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- API health: http://localhost:8000/health
- MLflow: http://localhost:5000

Stop the services:

```bash
docker compose down
```

## 8. Automated Testing and CI

The repository contains API tests covering:

- API health and model availability
- Selected feature schema
- Invalid or incomplete feature inputs
- A reference inference result

The GitHub Actions workflow runs tests and builds
Docker images on configured repository events.

View workflow runs:

https://github.com/macrosensor2022/careinsight-mlops/actions

## 9. Prediction Interpretation

The API returns a model-estimated probability and
binary predicted class for the positive survey outcome.

A probability should not be interpreted as disability
severity, individual care requirements, or a medical
risk score.

Comparing synthetic profiles illustrates model behavior,
not causal relationships.

Model probabilities have not been established as
calibrated individual-level risk estimates.

## 10. Limitations

- Training data comes from a historical survey.
- Performance on contemporary populations is unknown.
- Model predictions are not causal findings.
- The user interface exposes only validated or
  explicitly fixed demonstration inputs.
- Synthetic profiles may contain combinations not
  representative of actual survey respondents.
- Historical test metrics do not establish clinical
  suitability, fairness, or deployment reliability.
- The application is not suitable for individual
  medical or eligibility decisions.

## 11. Future Improvements

- Validate and document the survey target definition.
- Expand category mappings using the original codebook.
- Add a detailed model card and evaluation methodology.
- Evaluate probability calibration and subgroup behavior.
- Improve input validation and public-demo safeguards.
- Document the dataset and training reproducibility.

---

**Project:** CareInsight ML

**Focus:** Applied Machine Learning, Backend Engineering,
MLOps, and Cloud Deployment
