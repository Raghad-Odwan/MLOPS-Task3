"""
Data validation using Great Expectations.
This runs a set of expectations on an incoming order (as a one-row DataFrame)
before it reaches the model, in addition to the lightweight checks in
data_validation.py. This layer focuses on statistical/business expectations:
column types, reasonable ranges, and allowed categories.
"""

import logging
import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_STATES = [
    "SP",
    "RJ",
    "MG",
    "RS",
    "PR",
    "SC",
    "BA",
    "DF",
    "ES",
    "GO",
    "PE",
    "CE",
    "PA",
    "MT",
    "MA",
    "MS",
    "PB",
    "PI",
    "RN",
    "AL",
    "SE",
    "TO",
    "RO",
    "AM",
    "AC",
    "AP",
    "RR",
]
ALLOWED_PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]


class DataExpectationError(Exception):
    """Raised when incoming data fails one or more Great Expectations checks."""

    pass


def _build_expectations():
    """The set of expectations applied to every incoming order."""
    return [
        gx.expectations.ExpectColumnValuesToBeBetween(column="total_price", min_value=0, max_value=100000),
        gx.expectations.ExpectColumnValuesToBeBetween(column="total_freight", min_value=0, max_value=10000),
        gx.expectations.ExpectColumnValuesToBeBetween(column="n_items", min_value=1, max_value=100),
        gx.expectations.ExpectColumnValuesToBeBetween(column="n_payment_installments", min_value=0, max_value=36),
        gx.expectations.ExpectColumnValuesToBeInSet(column="customer_state", value_set=ALLOWED_STATES),
        gx.expectations.ExpectColumnValuesToBeInSet(column="payment_type", value_set=ALLOWED_PAYMENT_TYPES),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="total_price"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_state"),
    ]


def validate_with_great_expectations(df: pd.DataFrame) -> None:
    """
    Run the expectation suite against an incoming order DataFrame.
    Raises DataExpectationError with the list of failed expectations if
    anything does not match, instead of letting bad data reach the model silently.
    """
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("incoming_orders")
    data_asset = data_source.add_dataframe_asset(name="orders_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    failures = []
    for expectation in _build_expectations():
        result = batch.validate(expectation)
        if not result.success:
            failures.append(
                {
                    "expectation": expectation.__class__.__name__,
                    "column": getattr(expectation, "column", None),
                    "result": result.result,
                }
            )

    if failures:
        logger.warning(f"Great Expectations validation failed: {failures}")
        raise DataExpectationError(f"Data failed validation: {failures}")

    logger.info("Order passed Great Expectations validation.")
