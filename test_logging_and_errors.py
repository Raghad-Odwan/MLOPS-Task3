"""
Manual check for Task 3, item 3: logging and error handling.
Runs one good order (should succeed) and one bad order (should fail
gracefully, with a clear error, and the service should NOT crash).
"""

from src.logger_setup import setup_logging
from src.config_loader import load_config
from src.predict import LateDeliveryPredictor
from src.data_validation import ValidationError

config = load_config()
setup_logging(config)

predictor = LateDeliveryPredictor(config)

print("\n--- Test 1: a valid order ---")
good_order = {
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
result = predictor.predict(good_order)
print("Result:", result)

print("\n--- Test 2: a broken order (missing fields, negative price) ---")
bad_order = {
    "total_price": -50,          # invalid: negative
    "n_items": 1,
    "order_purchase_timestamp": "2018-05-10 14:23:00",
    "customer_state": "SP",
    # missing several required fields on purpose
}
try:
    result = predictor.predict(bad_order)
    print("Result:", result)
except ValidationError as e:
    print(f"Handled gracefully -- ValidationError: {e}")

print("\nService did not crash. Both cases handled.")
