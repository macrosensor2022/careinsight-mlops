
from pathlib import Path
import json
import os

import mlflow

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("CareInsight-Reproducible-Training")


def log_experiment(folder_name, run_name):
    experiment_dir = EXPERIMENTS / folder_name
    metrics_file = experiment_dir / "metrics.json"
    model_file = experiment_dir / "model.joblib"

    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)

    with open(metrics_file, encoding="utf-8") as file:
        results = json.load(file)

    with mlflow.start_run(run_name=run_name):

        mlflow.set_tag("project", "CareInsight ML")
        mlflow.set_tag("dataset", "NSS 76th Round")
        mlflow.set_tag(
            "experiment_type",
            "reproducible_training"
        )

        mlflow.log_param(
            "model_type",
            "HistGradientBoostingClassifier"
        )

        if folder_name == "baseline_gb":
            mlflow.set_tag("training_type", "baseline")

            mlflow.log_params({
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0
            })

            for metric, value in results["train"].items():
                mlflow.log_metric(
                    f"train_{metric}",
                    float(value)
                )

            for metric, value in results["test"].items():
                mlflow.log_metric(
                    f"test_{metric}",
                    float(value)
                )

            mlflow.log_metric(
                "roc_auc_gap",
                float(results["roc_auc_gap"])
            )

        elif folder_name == "tuned_gb":
            mlflow.set_tag(
                "training_type",
                "randomized_search"
            )

            mlflow.log_metric(
                "cv_roc_auc",
                float(results["cv_roc_auc"])
            )

            params = {
                key.replace("model__", ""): value
                for key, value in
                results["best_parameters"].items()
            }

            mlflow.log_params(params)

            for metric, value in results[
                "test_metrics"
            ].items():
                mlflow.log_metric(
                    f"test_{metric}",
                    float(value)
                )

        mlflow.log_artifact(str(metrics_file))

        if model_file.exists():
            mlflow.log_artifact(
                str(model_file),
                artifact_path="model_artifact"
            )

    print(f"[OK] Logged MLflow run: {run_name}")


def main():
    print("\nLogging CareInsight training experiments")

    log_experiment(
        "baseline_gb",
        "reproducible_baseline_gb"
    )

    log_experiment(
        "tuned_gb",
        "reproducible_tuned_gb"
    )

    print("\nBoth experiments logged successfully.")


if __name__ == "__main__":
    main()
