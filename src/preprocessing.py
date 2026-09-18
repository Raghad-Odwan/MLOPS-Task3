"""
Turns a raw incoming order into a DataFrame with the time-based features
needed by the model, matching exactly what Notebook 5 did during training.
"""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add purchase_weekday, purchase_month, purchase_hour from the timestamp."""
    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    return df


def order_to_dataframe(order: dict) -> pd.DataFrame:
    """Convert a single order (as a dict, e.g. from an API request) into a one-row DataFrame."""
    return pd.DataFrame([order])
