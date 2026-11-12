
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold
)

from ml.train import (
    build_model,
    evaluate,
    load_data,
    ROOT
)

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "experiments" / "tuned_gb"


def main():
    print("\nCareInsight — Gradient Boosting Tuning")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    # These rows are in the same order used to create
    # X_train.pkl in the feature engineering script.
    train_records = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=["HHID"]
    )

    groups = train_records["HHID"]

    assert len(groups) == len(X_train)
    assert X_train.index.equals(y_train.index)

    # Keep entire households in one CV fold.
    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    param_distributions = {
        "model__learning_rate": [
            0.03, 0.05, 0.1
        ],
        "model__max_iter": [
            100, 150, 200
        ],
        "model__max_leaf_nodes": [
            15, 31
        ],
        "model__min_samples_leaf": [
            20, 40, 60
        ],
        "model__l2_regularization": [
            0.0, 0.1, 1.0
        ]
    }

    search = RandomizedSearchCV(
        estimator=build_model(),
        param_distributions=param_distributions,
        n_iter=6,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=2,
        verbose=1,
        refit=True,
        error_score="raise"
    )

    print("\nSearching hyperparameters...")

    search.fit(
        X_train,
        y_train,
        groups=groups
    )

    print("\nBest cross-validation ROC-AUC:")
    print(round(search.best_score_, 4))

    print("\nBest parameters:")
    for name, value in search.best_params_.items():
        print(f"{name}: {value}")

    best_model = search.best_estimator_

    # Held-out test evaluation is performed only
    # after the cross-validation search finishes.
    test_metrics = evaluate(
        best_model,
        X_test,
        y_test
    )

    print("\nHELD-OUT TEST METRICS")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        OUTPUT_DIR / "model.joblib"
    )

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "cv_roc_auc": float(search.best_score_),
                "best_parameters": search.best_params_,
                "test_metrics": test_metrics
            },
            file,
            indent=4
        )

    print("\nExperimental tuned model saved.")
    print("Production model was NOT modified.")


if __name__ == "__main__":
    main()
