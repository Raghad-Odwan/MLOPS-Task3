"""
Loads the trained model from the MLflow Model Registry (registered by
register_model.py) and produces predictions. The model is only loaded
here, never re-trained. Handles bad input and missing values without
crashing the service.
"""

import logging
import time
import mlflow.sklearn

from src.config_loader import load_config, resolve_path
from src.data_validation import validate_order, ValidationError
from src.ge_validation import validate_with_great_expectations, DataExpectationError
from src.preprocessing import order_to_dataframe, add_time_features
from src.feature_engineering import FeatureBuilder

logger = logging.getLogger(__name__)

MODEL_NAME = "late_delivery_model"
MODEL_ALIAS = "staging"


class PredictionError(Exception):
    """Raised when a prediction cannot be produced, for any reason other than bad input."""
    pass


class LateDeliveryPredictor:
    """End-to-end predictor: raw order in, prediction + probability out."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()

        # The service loads the model from the MLflow registry, not from a
        # local notebook folder, as required by the task.
        mlflow.set_tracking_uri(f"sqlite:///{resolve_path('mlflow.db')}")
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        self.model = mlflow.sklearn.load_model(model_uri)

        self.feature_builder = FeatureBuilder(self.config)
        self.model_version = self.config["api"]["version"]
        logger.info(f"Loaded model '{MODEL_NAME}' (alias={MODEL_ALIAS}) from MLflow registry.")

    def predict(self, order: dict) -> dict:
        """
        Take a raw order (dict), validate it (basic checks + Great Expectations),
        build features using the same fitted transformers as training, and
        return a prediction.

        Raises:
            ValidationError: basic field/type checks failed.
            DataExpectationError: Great Expectations checks failed (e.g. value
                out of a reasonable range, or an unknown category).
            PredictionError: something unexpected went wrong while building
                features or running the model.
        """
        start_time = time.time()

        validate_order(order)

        df = order_to_dataframe(order)
        validate_with_great_expectations(df)

        try:
            df = add_time_features(df)
            X = self.feature_builder.transform(df)

            prediction = int(self.model.predict(X)[0])
            probability = float(self.model.predict_proba(X)[0][1])
        except (ValidationError, DataExpectationError):
            raise
        except Exception as exc:
            logger.exception(f"Unexpected error while predicting for order={order}")
            raise PredictionError(f"Could not produce a prediction: {exc}") from exc

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "is_late": prediction,
            "late_probability": round(probability, 4),
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2),
        }

        logger.info(
            f"Prediction made | input={order} | output={result} | "
            f"latency_ms={result['latency_ms']}"
        )

        return result
