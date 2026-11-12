
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------
# 1. CONFIGURATION
# ---------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "experiments" / "baseline_gb"

RANDOM_STATE = 42

SELECTED = joblib.load(
    ROOT / "models" / "selected_features.joblib"
)

NUMERIC = [
    "age",
    "log_monthly_exp"
]

NOMINAL = [
    "principal_activity_status",
    "relation_to_head",
    "education_level"
]

BINARY = [
    feature
    for feature in SELECTED
    if feature not in NUMERIC + NOMINAL
]

# ---------------------------------------
# 2. LOAD PREPARED DATA
# ---------------------------------------

def load_data():
    X_train = pd.read_pickle(
        DATA_DIR / "X_train.pkl"
    )

    X_test = pd.read_pickle(
        DATA_DIR / "X_test.pkl"
    )

    y_train = pd.read_pickle(
        DATA_DIR / "y_train.pkl"
    )

    y_test = pd.read_pickle(
        DATA_DIR / "y_test.pkl"
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    assert list(X_train.columns) == list(SELECTED)
    assert list(X_test.columns) == list(SELECTED)

    return X_train, X_test, y_train, y_test


# ---------------------------------------
# 3. PREPROCESSING PIPELINE
# ---------------------------------------

def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(strategy="median")
        ),
        (
            "scale",
            StandardScaler()
        )
    ])

    nominal_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=50,
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            NUMERIC
        ),
        (
            "bin",
            "passthrough",
            BINARY
        ),
        (
            "nom",
            nominal_pipeline,
            NOMINAL
        )
    ])


# ---------------------------------------
# 4. BUILD GRADIENT BOOSTING MODEL
# ---------------------------------------

def build_model():

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", classifier)
    ])


# ---------------------------------------
# 5. EVALUATION
# ---------------------------------------

def evaluate(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        )
    }


# ---------------------------------------
# 6. TRAIN AND SAVE
# ---------------------------------------

def main():

    print("\nCareInsight ML - Experimental Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Features: {len(SELECTED)}")

    model = build_model()

    print("\nTraining Gradient Boosting...")

    model.fit(X_train, y_train)

    print("Training completed.")

    train_metrics = evaluate(
        model, X_train, y_train
    )

    test_metrics = evaluate(
        model, X_test, y_test
    )

    print("\nTRAINING METRICS")
    for name, value in train_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nTEST METRICS")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    gap = (
        train_metrics["roc_auc"]
        - test_metrics["roc_auc"]
    )

    print(f"\nTrain-Test ROC-AUC Gap: {gap:.4f}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = OUTPUT_DIR / "model.joblib"

    metrics_path = OUTPUT_DIR / "metrics.json"

    joblib.dump(
        model,
        model_path
    )

    with open(metrics_path, "w") as file:
        json.dump(
            {
                "train": train_metrics,
                "test": test_metrics,
                "roc_auc_gap": gap,
                "experiment": "baseline_gradient_boosting"
            },
            file,
            indent=4
        )

    print(f"\nSaved experimental model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

    print("\nProduction model was NOT modified.")


if __name__ == "__main__":
    main()
