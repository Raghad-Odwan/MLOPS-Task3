"""
Registers the model trained in Notebook 6 with MLflow: logs the run
(parameters, metrics) and registers the model itself in the MLflow
Model Registry, with a version and a stage. After this runs once, the
service loads the model from the registry instead of a local .joblib file.

Run this once (or whenever a new model is trained) from the project root:
    python register_model.py
"""

import mlflow
import mlflow.sklearn
import joblib

from src.config_loader import load_config, resolve_path

config = load_config()

# Store all MLflow tracking data (runs, params, metrics, model registry) in a
# local SQLite database file. In a real deployment this would point to a
# shared MLflow server/database the container can reach, not just this laptop.
mlflow.set_tracking_uri(f"sqlite:///{resolve_path('mlflow.db')}")
mlflow.set_experiment("late_delivery_prediction")

MODEL_NAME = "late_delivery_model"

# These match what Notebook 6 actually used and found (see results_summary.txt).
params = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 200,
    "max_depth": 15,
    "class_weight": "balanced",
    "random_state": 42,
}
metrics = {
    "val_f1": 0.2846,
    "test_f1": 0.2686,
    "test_precision": 0.2036,
    "test_recall": 0.3944,
    "test_roc_auc": 0.7024,
}

model = joblib.load(resolve_path(config["paths"]["model"]))

with mlflow.start_run(run_name="notebook6_random_forest") as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered as '{MODEL_NAME}', version info: {model_info.model_uri}")

# Mark the newest version of the model with the "staging" alias.
# (MLflow's newer API uses aliases instead of the older stage names.)
client = mlflow.tracking.MlflowClient()
latest_version = client.search_model_versions(f"name='{MODEL_NAME}'")[0].version
client.set_registered_model_alias(MODEL_NAME, "staging", latest_version)
print(f"Model version {latest_version} tagged with alias: staging")