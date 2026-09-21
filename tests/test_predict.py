"""
Model tests: confirms the model and transformers load correctly, the
predictor produces the right output shape, and known inputs behave sensibly.
"""

import pytest
from src.predict import LateDeliveryPredictor
from src.data_validation import ValidationError
from src.ge_validation import DataExpectationError

GOOD_ORDER = {
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


@pytest.fixture(scope="module")
def predictor():
    """Load the predictor once for all tests in this file (loading the
    model and transformers is relatively slow, so we share one instance)."""
    return LateDeliveryPredictor()


def test_predictor_loads_without_error(predictor):
    assert predictor.model is not None
    assert predictor.feature_builder is not None


def test_predict_returns_expected_keys(predictor):
    result = predictor.predict(GOOD_ORDER)
    assert set(result.keys()) == {"is_late", "late_probability", "model_version", "latency_ms"}


def test_predict_is_late_is_binary(predictor):
    result = predictor.predict(GOOD_ORDER)
    assert result["is_late"] in (0, 1)


def test_predict_probability_in_valid_range(predictor):
    result = predictor.predict(GOOD_ORDER)
    assert 0.0 <= result["late_probability"] <= 1.0


def test_predict_rejects_invalid_order(predictor):
    bad_order = dict(GOOD_ORDER)
    del bad_order["total_price"]
    with pytest.raises(ValidationError):
        predictor.predict(bad_order)


def test_predict_rejects_order_failing_expectations(predictor):
    bad_order = dict(GOOD_ORDER)
    bad_order["customer_state"] = "ZZ"
    with pytest.raises(DataExpectationError):
        predictor.predict(bad_order)
