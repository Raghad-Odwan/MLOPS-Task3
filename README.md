# Late Delivery Prediction Service

MLOps Training 2026/2027 - Task 3: From Notebooks to Production

This repository turns the training notebooks from Task 2 into a real inference
service: a FastAPI app, backed by a model loaded from the MLflow Model
Registry, containerized with Docker, tested with pytest, and built/tested
automatically on every push with GitHub Actions.

**Input:** a new order. **Output:** a late/on-time prediction with a
probability.

## Project structure

```
app/            FastAPI service: routes, request/response schemas
src/            The inference pipeline as Python modules (not one script):
                config loading, data validation, preprocessing, feature
                building, prediction, logging, monitoring
config/         config.yaml -- every path and parameter used by the
                pipeline, so nothing is hardcoded in the code
models/         The trained model and fitted transformers from Task 2
                (num_imputer, scaler, cat_imputer, encoder, final_model,
                feature_list.txt)
notebooks/      The six training notebooks from Task 2, kept for reference.
                Training does not run as part of this service.
tests/          Unit, data, model, and integration tests (pytest)
.github/workflows/ci.yml   CI pipeline: lint, format check, tests, Docker build
Dockerfile      Builds a small image for the API service only
docker-compose.yml   Brings up the database and the API with one command
entrypoint.sh   Registers the model with MLflow inside the container on
                first startup, then starts the API
register_model.py   Registers the model in models/ with MLflow (run once,
                or automatically inside the container -- see below)
requirements.txt        Runtime dependencies (pinned versions)
requirements-dev.txt    Adds pytest, black, flake8, dvc for development
```

## How to run everything from zero

Requirements: Docker Desktop installed and running.

```bash
git clone https://github.com/Raghad-Odwan/MLOPS-Task3.git
cd MLOPS-Task3
copy .env.example .env      # on Windows; use `cp` on Mac/Linux
docker compose up --build
```

This starts two containers:

- **db**: a PostgreSQL database (used by earlier tasks in this track; the
  API itself does not query it directly)
- **api**: the FastAPI service. On its first startup it has no MLflow
  tracking database yet, so `entrypoint.sh` automatically runs
  `register_model.py` to register the model in `models/` with MLflow,
  using the exact MLflow version installed in the container (this avoids
  version-mismatch errors between a locally created `mlflow.db` and the
  container). This adds a few seconds to the first startup only.

Once both containers are healthy, open:

```
http://localhost:8000/docs
```

to see the interactive API documentation and try the routes directly from
the browser.

## API routes

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/model-info` | GET | Which model/version is loaded |
| `/metrics` | GET | Request count, error rate, average latency, prediction distribution |
| `/predict` | POST | Predict late/on-time for a single order |
| `/predict-batch` | POST | Predict for a batch of orders |

Example request body for `/predict`:

```json
{
  "total_price": 120.50,
  "total_freight": 18.90,
  "n_items": 1,
  "total_payment_value": 139.40,
  "n_payment_installments": 2,
  "order_purchase_timestamp": "2018-05-10 14:23:00",
  "customer_state": "SP",
  "payment_type": "credit_card",
  "product_category_name_english": "housewares"
}
```

## Running locally without Docker (for development)

```bash
pip install -r requirements-dev.txt
python register_model.py        # registers the model locally, once
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running the tests

```bash
pip install -r requirements-dev.txt
python register_model.py        # tests need a registered model to load
pytest tests/ -v
```

This runs all 24 tests: unit tests for preprocessing, data validation
tests (including a data-leakage check), Great Expectations tests, model
tests, and end-to-end API integration tests. The same three commands run
automatically in CI on every push (see `.github/workflows/ci.yml`).

`test_pipeline_manually.py`, `test_logging_and_errors.py`, and
`test_great_expectations.py` in the project root are **not** part of the
automated suite. They are small manual smoke-test scripts used while
building this project to check the pipeline end-to-end by hand; the
automated tests that CI runs all live under `tests/`.

## Why the model is tracked directly in git (and how DVC is used)

The trained model and transformers (`models/*.joblib`) are tracked
directly in git, not only through DVC pointer files. This is a deliberate
choice: the task requires that "you can clone the repo on a clean machine
and start everything with one command." Since this project does not use a
cloud DVC remote (no S3/GCS credentials for a training exercise), a
DVC-only setup would leave a fresh clone with `.dvc` pointer files but no
way to fetch the actual model bytes. Tracking the model files in git
directly guarantees the Docker build and CI always have the real files.

DVC itself is still set up and used for real versioning/traceability, with
a **local folder as its remote** (`../dvc-storage`, outside the repo).
This lets you push/pull specific versions of the model artifacts and trace
any result back to the exact file version that produced it:

```bash
dvc remote list                 # shows the configured local remote
dvc push                        # push the current models/*.joblib to the remote
dvc pull                        # restore models/*.joblib from the remote
dvc add models/final_model.joblib   # re-run after retraining, to version a new model
```

Note: a local-folder remote only works on the machine where that folder
exists. For a real production setup this would point to S3, GCS, or
Azure Blob Storage instead -- the commands above would not change.

## Data versioning notes

The `data/` folder is intentionally empty in this repository (it only
exists as a placeholder in `.gitignore`). The actual training data
(`ml_table.csv`, `labeled_table.csv`, `train.csv`, `val.csv`, `test.csv`,
etc.) is produced by the notebooks in Task 2 and is not needed to run this
service, since the model and fitted transformers in `models/` already
encode everything needed for inference.

## Data validation

Incoming orders are checked in two layers before reaching the model:

1. **Basic checks** (`src/data_validation.py`): required fields present,
   correct types, no negative values.
2. **Great Expectations** (`src/ge_validation.py`): reasonable value
   ranges and allowed categories (e.g. a valid Brazilian state code, a
   known payment type).

Either layer failing returns a `422` response with a clear error message
instead of a server crash or a silently wrong prediction.

## Monitoring

See [MONITORING.md](MONITORING.md) for what is tracked (the `/metrics`
route and the prediction log) and what would trigger an alert in
production.