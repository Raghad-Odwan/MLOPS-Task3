"""
Loads the trained model (saved by Notebook 6) and produces predictions.
The model is only loaded here, never re-trained.
Handles bad input and missing values without crashing the service.
"""

import logging
import time
import joblib

from src.config_loader import load_config, resolve_path
from src.data_validation import validate_order, ValidationError
from src.ge_validation import validate_with_great_expectations, DataExpectationError
from src.preprocessing import order_to_dataframe, add_time_features
from src.feature_engineering import FeatureBuilder

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when a prediction cannot be produced, for any reason other than bad input."""
    pass


class LateDeliveryPredictor:
    """End-to-end predictor: raw order in, prediction + probability out."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        model_path = resolve_path(self.config["paths"]["model"])
        self.model = joblib.load(model_path)
        self.feature_builder = FeatureBuilder(self.config)
        self.model_version = self.config["api"]["version"]
        logger.info(f"Loaded model from {model_path} (version {self.model_version}).")

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

        # Layer 1: lightweight, code-based checks (missing fields, types, negatives).
        validate_order(order)

        # Layer 2: statistical / business rules via Great Expectations
        # (ranges, allowed categories). Both layers raise a clear, specific
        # error instead of letting bad data reach the model silently.
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
