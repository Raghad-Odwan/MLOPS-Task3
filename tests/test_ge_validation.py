"""
Tests for the Great Expectations validation layer: ranges and allowed
categories that the basic validation layer does not check.
"""

import pytest
import pandas as pd
from src.ge_validation import validate_with_great_expectations, DataExpectationError

GOOD_ORDER_DF = pd.DataFrame(
    [
        {
            "total_price": 120.50,
            "total_freight": 18.90,
            "n_items": 1,
            "n_payment_installments": 2,
            "customer_state": "SP",
            "payment_type": "credit_card",
        }
    ]
)


def test_valid_order_passes_great_expectations():
    # Should not raise
    validate_with_great_expectations(GOOD_ORDER_DF)


def test_unknown_state_is_rejected():
    df = GOOD_ORDER_DF.copy()
    df["customer_state"] = "ZZ"
    with pytest.raises(DataExpectationError):
        validate_with_great_expectations(df)


def test_unknown_payment_type_is_rejected():
    df = GOOD_ORDER_DF.copy()
    df["payment_type"] = "bitcoin"
    with pytest.raises(DataExpectationError):
        validate_with_great_expectations(df)


def test_installments_out_of_range_is_rejected():
    df = GOOD_ORDER_DF.copy()
    df["n_payment_installments"] = 200
    with pytest.raises(DataExpectationError):
        validate_with_great_expectations(df)
