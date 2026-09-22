"""
Lightweight monitoring: tracks basic service metrics in memory (request
count, latency, error rate) and appends every prediction to a CSV log,
so predictions can be evaluated later once the real delivery date is known.
"""

import csv
import logging
import threading
import time

from src.config_loader import resolve_path

logger = logging.getLogger(__name__)

PREDICTION_LOG_PATH = resolve_path("logs/prediction_log.csv")
PREDICTION_LOG_HEADER = [
    "timestamp",
    "order_purchase_timestamp",
    "customer_state",
    "payment_type",
    "product_category_name_english",
    "is_late",
    "late_probability",
    "model_version",
    "latency_ms",
]


class ServiceMetrics:
    """
    In-memory counters for the service. Thread-safe with a simple lock,
    since FastAPI/uvicorn can handle requests concurrently.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.late_predictions = 0
        self.on_time_predictions = 0
        self.start_time = time.time()

    def record_success(self, result: dict) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += result["latency_ms"]
            if result["is_late"] == 1:
                self.late_predictions += 1
            else:
                self.on_time_predictions += 1

    def record_error(self) -> None:
        with self._lock:
            self.request_count += 1
            self.error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = self.total_latency_ms / self.request_count if self.request_count else 0.0
            error_rate = self.error_count / self.request_count if self.request_count else 0.0
            late_ratio = (
                self.late_predictions / (self.late_predictions + self.on_time_predictions)
                if (self.late_predictions + self.on_time_predictions)
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": round(error_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "late_predictions": self.late_predictions,
                "on_time_predictions": self.on_time_predictions,
                "predicted_late_ratio": round(late_ratio, 4),
            }


# One shared instance for the whole running service.
metrics = ServiceMetrics()


def log_prediction(order: dict, result: dict) -> None:
    """
    Append one row to the prediction log CSV: the input order (key fields),
    the prediction, and metadata. This is what lets us later join predictions
    against the real delivery outcome once it is known, to check drift and
    real-world accuracy over time.
    """
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTION_LOG_PATH.exists()

    try:
        with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "order_purchase_timestamp": order.get("order_purchase_timestamp"),
                    "customer_state": order.get("customer_state"),
                    "payment_type": order.get("payment_type"),
                    "product_category_name_english": order.get("product_category_name_english"),
                    "is_late": result["is_late"],
                    "late_probability": result["late_probability"],
                    "model_version": result["model_version"],
                    "latency_ms": result["latency_ms"],
                }
            )
    except OSError:
        # Never let logging failures break a prediction response.
        logger.exception("Failed to write to the prediction log CSV.")
