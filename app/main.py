"""
FastAPI service for late-delivery prediction.
Routes: health check, model info, single predict, batch predict, metrics.

Run locally with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
import logging

from src.logger_setup import setup_logging
from src.config_loader import load_config
from src.predict import LateDeliveryPredictor, PredictionError
from src.data_validation import ValidationError
from src.ge_validation import DataExpectationError
from src.monitoring import metrics, log_prediction

from app.schemas import (
    OrderRequest,
    PredictionResponse,
    BatchOrderRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
)

config = load_config()
setup_logging(config)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    description="Predicts whether an order will be delivered late, given order details.",
)

# The model and all fitted transformers are loaded once, when the service
# starts, not on every request.
predictor = LateDeliveryPredictor(config)
logger.info("FastAPI service started and predictor is ready.")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Simple check that the service is up and able to respond."""
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Return which model and version the service is currently using."""
    return ModelInfoResponse(
        model_name="late_delivery_model",
        model_alias="staging",
        model_version=predictor.model_version,
        api_version=config["api"]["version"],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def get_metrics():
    """
    Basic service metrics: request count, error rate, average latency,
    and the distribution of predictions (late vs on-time), so drift in
    predicted outcomes can be noticed over time.
    """
    return MetricsResponse(**metrics.snapshot())


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(order: OrderRequest):
    """Predict late/on-time for a single order."""
    try:
        order_dict = order.model_dump()
        result = predictor.predict(order_dict)
        metrics.record_success(result)
        log_prediction(order_dict, result)
        return PredictionResponse(**result)
    except ValidationError as e:
        metrics.record_error()
        raise HTTPException(status_code=422, detail=f"Invalid order: {e}")
    except DataExpectationError as e:
        metrics.record_error()
        raise HTTPException(status_code=422, detail=f"Order failed data validation: {e}")
    except PredictionError as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchOrderRequest):
    """Predict late/on-time for a batch of orders in one request."""
    results = []
    for order in batch.orders:
        try:
            order_dict = order.model_dump()
            result = predictor.predict(order_dict)
            metrics.record_success(result)
            log_prediction(order_dict, result)
            results.append(PredictionResponse(**result))
        except (ValidationError, DataExpectationError) as e:
            metrics.record_error()
            raise HTTPException(status_code=422, detail=f"Invalid order in batch: {e}")
        except PredictionError as e:
            metrics.record_error()
            raise HTTPException(status_code=500, detail=f"Prediction failed for an order in batch: {e}")
    return BatchPredictionResponse(predictions=results)
