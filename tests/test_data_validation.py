"""
Data tests: schema checks, value ranges, missing fields, and a leakage
check (make sure fields that only exist after delivery are never required
as model input).
"""

import pytest
from src.data_validation import validate_order, ValidationError, REQUIRED_FIELDS

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


def test_valid_order_passes():
    # Should not raise
    validate_order(GOOD_ORDER)


def test_missing_field_is_rejected():
    order = dict(GOOD_ORDER)
    del order["total_price"]
    with pytest.raises(ValidationError):
        validate_order(order)


def test_negative_value_is_rejected():
    order = dict(GOOD_ORDER)
    order["total_freight"] = -5
    with pytest.raises(ValidationError):
        validate_order(order)


def test_wrong_type_is_rejected():
    order = dict(GOOD_ORDER)
    order["n_items"] = "one"  # should be numeric
    with pytest.raises(ValidationError):
        validate_order(order)


def test_no_leakage_fields_in_required_schema():
    """
    Data leakage check: fields that only exist after delivery
    (actual delivery date, review score) must never be part of what
    the service requires as input.
    """
    leak_fields = {
        "order_delivered_customer_date",
        "review_score",
        "order_delivered_carrier_date",
    }
    assert leak_fields.isdisjoint(set(REQUIRED_FIELDS))
