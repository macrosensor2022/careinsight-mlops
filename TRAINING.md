
# CareInsight ML — Reproducible Training Guide

## Overview

This guide explains how to recreate the CareInsight
experimental training workflow using the NSS 76th
Round disability survey.

The training pipeline covers:

1. Dataset verification
2. Data inspection
3. Survey-table merging
4. Target construction and grouped splitting
5. Feature engineering
6. Baseline model training
7. Hyperparameter tuning
8. MLflow experiment tracking

The experimental workflow does not overwrite the
model used by the deployed FastAPI application.

## 1. Requirements

- Python with project dependencies installed
- Access to the original NSS 76th Round survey files
- Docker and Docker Compose for the local MLflow server

Run commands from the repository root.

Note: Exact dependency versions should be pinned
and verified for full environment reproducibility.

## 2. Dataset Organization

Place the following files in `data/raw/`:

- Block_3_Level _2.csv
- Block 4 Level 3.csv
- Block 5 Level 4.csv
- Block_6_Level _12.csv
- Block 8 Level 14.csv
- Block 5_1 Level 5.csv

The data directory is excluded from Git.

Do not commit record-level survey data or
generated training datasets.

## 3. Verify and Inspect the Data

Run:

```powershell
python -m ml.validate_data
python -m ml.inspect_data
```

The validation checks whether all required files
are available.

The inspection reports dataset dimensions and
column information.

## 4. Prepare the Person-Level Dataset

Run:

```powershell
python -m ml.prepare_data
```

This merges the required NSS survey tables.

Expected output:

`data/processed/person_level.csv`

The adult population contains 89,340 records
before the later age-range validation.

## 5. Construct Target and Split Data

Run:

```powershell
python -m ml.create_split
```

This performs:

- Age validation
- Binary target construction
- Household-grouped train/test splitting
- Household-overlap verification

The original target rule is:

`needs_care = (caregiver_code != 16)`

The interpretation of caregiver codes must be
verified against the official survey codebook.

Expected cleaned records: 89,338

Observed training records: 71,516

Observed testing records: 17,822

The split is approximately 80/20.

## 6. Feature Engineering

Run:

```powershell
python -m ml.feature_engineering
```

The process includes:

- Binary survey-code transformations
- String encoding of nominal survey categories
- Log transformation of expenditure
- Multiple-disability indicator construction
- Selection of 14 previously identified features

Generated local files:

- X_train.pkl
- X_test.pkl
- y_train.pkl
- y_test.pkl

The current pipeline reuses the selected feature
list from the original experiment. It does not
reproduce the complete original feature-selection
procedure independently.

## 7. Train the Baseline Model

Run:

```powershell
python -m ml.train
```

Model: HistGradientBoostingClassifier

Observed held-out test metrics:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7677 |
| Accuracy | 0.6987 |
| Precision | 0.8128 |
| Recall | 0.6619 |
| F1 | 0.7296 |

Output directory:

`experiments/baseline_gb/`

## 8. Hyperparameter Tuning

Run:

```powershell
python -m ml.tune
```

The tuning procedure uses RandomizedSearchCV
with household-grouped cross-validation.

Current search settings:

- Six sampled parameter combinations
- Three cross-validation folds
- ROC-AUC as the optimization metric
- Random state 42

Observed best CV ROC-AUC: 0.7646

Observed held-out test ROC-AUC: 0.7686

Best recorded hyperparameters:

- learning_rate: 0.05
- max_iter: 150
- max_leaf_nodes: 31
- min_samples_leaf: 60
- l2_regularization: 0.0

Output directory:

`experiments/tuned_gb/`

These results come from a limited experimental
search, not an exact reproduction of the original
notebook's full tuning procedure.

## 9. Track Experiments in MLflow

Start the local MLflow service:

```powershell
docker compose up -d mlflow
```

Log the saved training results:

```powershell
python -m ml.log_training_runs
```

Open:

http://localhost:5000

Experiment:

CareInsight-Reproducible-Training

Runs:

- reproducible_baseline_gb
- reproducible_tuned_gb

Each execution of the logging script creates
additional MLflow runs.

## 10. Experiment vs Production Models

Experimental model artifacts:

- experiments/baseline_gb/model.joblib
- experiments/tuned_gb/model.joblib

Deployed model artifact:

- models/caregiver_model.joblib

Experimental training does not automatically
replace or redeploy the production model.

Model promotion requires separate validation,
regression testing, and deliberate deployment.

## 11. Reproducibility Limitations

The current workflow demonstrates repeatable
data preparation, training, and evaluation
on the available local survey files.

Additional verification is required for:

- Official caregiver-code interpretations
- Exact original notebook split equivalence
- Training-only feature selection
- Dataset provenance and checksums
- Exact dependency versions
- Probability calibration
- Evaluation across population subgroups

The predictions are intended for educational
model exploration, not individual care decisions.
