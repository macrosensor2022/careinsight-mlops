
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Load the original model comparison
csv_path = ROOT / "reports" / "model_comparison.csv"
results = pd.read_csv(csv_path, index_col="Model")

# Connect to your running MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("CareInsight-ML")

client = MlflowClient()

# Reuse your existing Gradient Boosting run
gb_run_id = "9977915fb2194357bde3d2e7fea76791"

metric_mapping = {
    "Test ROC-AUC": "test_roc_auc",
    "F1": "test_f1",
    "Precision": "test_precision",
    "Recall": "test_recall",
    "Train-Test Gap": "train_test_auc_gap"
}

for model_name, row in results.iterrows():

    if model_name == "Gradient Boosting":
        # Add metrics to the existing model run
        for csv_column, metric_name in metric_mapping.items():
            client.log_metric(
                gb_run_id,
                metric_name,
                float(row[csv_column])
            )

        client.set_tag(
            gb_run_id,
            "evaluation_source",
            "Original Colab notebook"
        )

    else:
        # Create a run for each additional model
        run_name = model_name.lower().replace(" ", "_") + "_historical"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("algorithm", model_name)

            for csv_column, metric_name in metric_mapping.items():
                mlflow.log_metric(
                    metric_name,
                    float(row[csv_column])
                )

            mlflow.set_tag(
                "run_type",
                "historical_metrics_import"
            )
            mlflow.set_tag(
                "evaluation_source",
                "Original Colab notebook"
            )

    print(f"Logged: {model_name}")

print("All model comparison metrics logged successfully!")
