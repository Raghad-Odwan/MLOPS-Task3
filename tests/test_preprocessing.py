"""
Unit tests for src/preprocessing.py: building the raw DataFrame and
adding time-based features.
"""

import pandas as pd
from src.preprocessing import order_to_dataframe, add_time_features


def test_order_to_dataframe_creates_one_row():
    order = {"total_price": 100, "customer_state": "SP"}
    df = order_to_dataframe(order)
    assert len(df) == 1
    assert df.loc[0, "total_price"] == 100
    assert df.loc[0, "customer_state"] == "SP"


def test_add_time_features_extracts_correct_values():
    df = pd.DataFrame([{"order_purchase_timestamp": "2018-05-10 14:23:00"}])
    result = add_time_features(df)
    assert result.loc[0, "purchase_weekday"] == 3  # Thursday
    assert result.loc[0, "purchase_month"] == 5
    assert result.loc[0, "purchase_hour"] == 14


def test_add_time_features_does_not_mutate_original():
    df = pd.DataFrame([{"order_purchase_timestamp": "2018-05-10 14:23:00"}])
    add_time_features(df)
    # The original DataFrame should not have gained the new columns,
    # since add_time_features works on a copy.
    assert "purchase_weekday" not in df.columns
