"""
Quick manual check: run this once after copying the models/ folder,
to confirm the pipeline loads everything and produces a prediction
that matches what the notebooks would produce for a similar order.
"""

from src.logger_setup import setup_logging
from src.config_loader import load_config
from src.predict import LateDeliveryPredictor

config = load_config()
setup_logging(config)

sample_order = {
    "total_price": 120.50,
    "total_freight": 18.90,
    "n_items": 1,
    "total_payment_value": 139.40,
    "n_payment_installments": 2,
    "order_purchase_timestamp": "2018-05-10 14:23:00",
    "customer_state": "SP",
    "payment_type": "credit_card",
    "product_category_name_english": "housewares",
}

predictor = LateDeliveryPredictor(config)
result = predictor.predict(sample_order)
print("\nPrediction result:")
print(result)