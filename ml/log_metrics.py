
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Your run ID from the successful terminal output
run_id = "9977915fb2194357bde3d2e7fea76791"

# Historical evaluation result from the Colab notebook
client.log_metric(
    run_id,
    "test_roc_auc",
    0.7683
)

client.set_tag(
    run_id,
    "metric_source",
    "Original Colab notebook evaluation"
)

print("Historical model metric logged successfully!")
