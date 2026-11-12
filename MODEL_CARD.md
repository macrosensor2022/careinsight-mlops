# CareInsight ML — Model Card

## 1. Model Overview

**Model name:** CareInsight Gradient Boosting Classifier  
**Model version:** 1.0.1 (prediction API version)  
**Model type:** Histogram-based Gradient Boosting classifier  
**Framework:** scikit-learn  
**Task:** Binary classification  
**Dataset:** India's National Sample Survey (NSS), 76th Round, disability survey  
**Deployment:** FastAPI on Render, accessed through Streamlit Community Cloud

CareInsight ML is a research demonstration exploring patterns associated with survey-reported caregiver arrangements among adults with disabilities in India.

The model receives 14 selected survey-related features and returns a positive-class probability and a binary predicted class.

The deployed model is stored as a serialized scikit-learn pipeline.

## 2. Intended Use

The model is intended for:

- Educational machine learning demonstrations.
- Exploring how predictions respond to fictional survey profiles.
- Studying an end-to-end ML development and deployment workflow.
- Demonstrating FastAPI, Docker, MLflow, and cloud-based inference.

**The model is not intended for medical diagnosis, disability assessment, determining an individual's caregiver needs, or making decisions about access to support services.**

## 3. Dataset

The project uses disability-related data from India's NSS 76th Round survey, conducted during July–December 2018.

The locally reconstructed training dataset contains:

| Description | Count |
|---|---:|
| Adult records after age validation | 89,338 |
| Training records | 71,516 |
| Testing records | 17,822 |
| Selected input features | 14 |

Records are divided using household-grouped splitting so that members of the same household do not appear in both training and testing partitions.

These counts describe the reconstructed experimental training workflow. The precise equivalence of its split to the original deployed-model experiment has not been fully established.

## 4. Target Variable

The original research notebook constructs a binary target using:

```python
needs_care = (caregiver_code != 16).astype(int)
```

Under this rule:

- **Class 0:** Caregiver code equals `16`.
- **Class 1:** Caregiver code does not equal `16`.

The exact meaning of caregiver code `16` and the other survey categories still requires verification against the official NSS codebook.

Consequently, the target should be described as a **survey-code-derived caregiver-arrangement outcome**, rather than a validated measure of whether an individual medically requires caregiving.

## 5. Input Features

The deployed model uses 14 features:

| Feature | Description |
|---|---|
| `dis_seeing` | Survey-coded seeing-related disability indicator |
| `dis_loss_sensation` | Sensation-loss indicator |
| `dis_understanding` | Understanding-related disability indicator |
| `dis_hearing` | Hearing-related disability indicator |
| `uses_public_transport` | Public transport use indicator |
| `multiple_disability` | Engineered multiple-disability indicator |
| `principal_activity_status` | Survey-coded activity status |
| `dis_deformity` | Deformity-related indicator |
| `dis_anxiety_mood` | Anxiety-related indicator |
| `relation_to_head` | Relationship to household head |
| `age` | Age in years |
| `education_level` | Survey-coded education category |
| `accesses_public_building` | Public-building access indicator |
| `log_monthly_exp` | Log-transformed monthly consumer expenditure |

Nominal category values are converted to their training-aligned string representation before inference.

Exact categorical meanings require reference to the official survey documentation.

## 6. Model Development

The original research evaluated four modeling approaches:

- Logistic Regression
- Random Forest
- Histogram-based Gradient Boosting
- Stacking

The deployed model uses Histogram-based Gradient Boosting, selected for its relatively simple inference pipeline and performance close to the highest-scoring approach.

The saved scikit-learn pipeline includes preprocessing operations needed to transform model inputs.

## 7. Original Model Performance

Historical held-out evaluation results from the original notebook:

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7628 | 0.7386 | 0.7992 | 0.6866 |
| Random Forest | 0.7678 | 0.7379 | 0.8063 | 0.6802 |
| Gradient Boosting | 0.7683 | 0.7344 | 0.8114 | 0.6707 |
| Stacking | 0.7694 | 0.7361 | 0.8101 | 0.6745 |

The deployed Gradient Boosting model achieved a recorded historical test ROC-AUC of approximately **0.7683**.

These metrics are historical evaluation results, not live production monitoring measurements.

## 8. Independent Training Experiments

A separate Python-based workflow was developed to reconstruct data preparation, training, and evaluation from the original survey CSV files.

Two experimental models were trained:

| Metric | Baseline Gradient Boosting | Tuned Gradient Boosting |
|---|---:|---:|
| Accuracy | 0.6987 | 0.6985 |
| Precision | 0.8128 | 0.8135 |
| Recall | 0.6619 | 0.6606 |
| F1 | 0.7296 | 0.7291 |
| Test ROC-AUC | 0.7677 | 0.7686 |

The tuning experiment used `RandomizedSearchCV` with three household-grouped cross-validation folds and six sampled hyperparameter configurations.

**Best cross-validation ROC-AUC:** 0.7646.

These experiments are tracked in MLflow under `CareInsight-Reproducible-Training`.

The experimental models are separate from the model deployed in production.

## 9. Prediction Interpretation

The API returns:

- A probability assigned to the positive class.
- A predicted binary class.
- An API model-version identifier.

The classification threshold is 0.50.

For example, a probability of 0.5849 corresponds to a model-estimated positive-class probability of 58.49% for the reference fictional profile.

It does not imply a 58.49% level of disability severity or a clinically established probability of requiring care.

Probability calibration has not been independently established.

## 10. Training–Serving Consistency

During development, a mismatch was identified between nominal categorical feature representations used in training and inference.

The original preprocessing expected string-coded survey categories, while the initial API passed numeric representations.

This produced different probabilities for the same fictional profile.

The FastAPI inference code was updated to convert identified nominal features into their training-aligned representation.

Regression tests were updated to verify the corrected inference result.

This correction demonstrates the importance of validating feature preprocessing consistently across model training and production serving.

## 11. Limitations

### Historical Survey Data

The model was developed using data collected in 2018. Generalization to current populations has not been demonstrated.

### Target Definition

Official caregiver-code interpretation still requires verification. The binary target is a research construction, not a clinical label.

### Feature Selection

The independent training workflow reuses the 14 selected features from the original notebook. A full training-only feature-selection audit remains outstanding.

### Calibration

The model's predicted probabilities have not been verified as calibrated risks.

### Potential Bias

Comprehensive evaluation across population subgroups has not yet been completed.

### Synthetic Input Profiles

Interactive examples may contain feature combinations that are uncommon or unrealistic in the original survey population.

### Causality

Model predictions describe learned statistical relationships, not causal effects of age, disability characteristics, transport access, or other features.

## 12. Responsible Use

The model should not be used for:

- Clinical assessment.
- Determining caregiver eligibility.
- Disability severity evaluation.
- Decisions regarding access to support services.
- Automated decisions affecting real individuals.

The public application is restricted to educational model exploration.

## 13. Future Validation

Planned work includes:

- Verification of official survey-code meanings.
- Full training-only feature-selection validation.
- Reproduction of the exact original notebook split and tuning settings.
- Evaluation of probability calibration.
- Subgroup performance analysis.
- Dataset provenance and checksum documentation.
- Improved inference input validation.
- Stronger automated training and data-quality tests.

## 14. Supporting Documentation

- [Live CareInsight Application](https://careinsight-mlops.streamlit.app/)
- [Project README](README.md)
- [Training and Reproducibility Guide](TRAINING.md)
- [GitHub Actions](https://github.com/macrosensor2022/careinsight-mlops/actions)

---

**CareInsight ML — An educational machine learning and MLOps research demonstration.**