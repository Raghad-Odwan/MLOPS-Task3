"""
Manual check for Task 3, item 4 (validation half): confirms the Great
Expectations layer specifically catches problems that the basic validation
layer (data_validation.py) does not, such as an unknown category or a
value outside a reasonable range that is still positive.
"""

from src.logger_setup import setup_logging
from src.config_loader import load_config
from src.predict import LateDeliveryPredictor
from src.data_validation import ValidationError
from src.ge_validation import DataExpectationError

config = load_config()
setup_logging(config)

predictor = LateDeliveryPredictor(config)

print("\n--- Test 1: a normal, valid order ---")
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
try:
    result = predictor.predict(good_order)
    print("Result:", result)
except (ValidationError, DataExpectationError) as e:
    print(f"Unexpectedly rejected: {e}")

print("\n--- Test 2: unknown state 'ZZ' (passes basic checks, fails Great Expectations) ---")
unknown_state_order = dict(good_order)
unknown_state_order["customer_state"] = "ZZ"
try:
    result = predictor.predict(unknown_state_order)
    print("Result:", result)
except ValidationError as e:
    print(f"Rejected by basic validation (unexpected here): {e}")
except DataExpectationError as e:
    print("Handled gracefully -- rejected by Great Expectations (unknown state).")

print("\n--- Test 3: 200 payment installments (positive, but out of a reasonable range) ---")
too_many_installments_order = dict(good_order)
too_many_installments_order["n_payment_installments"] = 200
try:
    result = predictor.predict(too_many_installments_order)
    print("Result:", result)
except ValidationError as e:
    print(f"Rejected by basic validation (unexpected here): {e}")
except DataExpectationError as e:
    print("Handled gracefully -- rejected by Great Expectations (installments out of range).")

print("\nService did not crash in any of the three cases.")