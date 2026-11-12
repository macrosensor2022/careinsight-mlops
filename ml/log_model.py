
from pathlib import Path
import joblib
import mlflow

ROOT = Path(__file__).resolve().parents[1]

model_path = ROOT / "models" / "caregiver_model.joblib"
features_path = ROOT / "models" / "selected_features.joblib"

# Load the model and its feature list
model = joblib.load(model_path)
features = joblib.load(features_path)

# Connect to your Docker MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("CareInsight-ML")

# Create an experiment run
with mlflow.start_run(
    run_name="gradient_boosting_v1"
):
    mlflow.log_param("algorithm", "Gradient Boosting")
    mlflow.log_param("feature_count", len(features))
    mlflow.set_tag("source", "Colab Midterm Project")
    mlflow.set_tag("run_type", "existing_model_import")

    # Save model and feature artifacts
    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(features_path))

    print("Model logged to MLflow successfully!")
