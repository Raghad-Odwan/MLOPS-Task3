#!/bin/sh
# On first startup (or whenever mlflow.db is missing), register the model
# with MLflow using the exact same MLflow version installed in this
# container. This avoids any version mismatch between a locally-created
# mlflow.db and the MLflow package baked into the image.
set -e

if [ ! -f mlflow.db ]; then
    echo "No mlflow.db found in this container. Registering the model now..."
    python register_model.py
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
