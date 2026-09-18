"""
Basic validation for an incoming order before it reaches the model.
This is a lightweight, code-based check. A more complete data validation
suite is added separately with Great Expectations.
"""

import logging

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "total_price",
    "total_freight",
    "n_items",
    "total_payment_value",
    "n_payment_installments",
    "order_purchase_timestamp",
    "customer_state",
    "payment_type",
    "product_category_name_english",
]


class ValidationError(Exception):
    """Raised when an incoming order fails basic validation."""
    pass


def validate_order(order: dict) -> None:
    """
    Check that an incoming order has the required fields and reasonable values.
    Raises ValidationError if something is wrong, so the API can return a clear
    error instead of crashing.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in order]
    if missing:
        logger.warning(f"Missing fields in incoming order: {missing}")
        raise ValidationError(f"Missing required fields: {missing}")

    numeric_fields = ["total_price", "total_freight", "n_items",
                       "total_payment_value", "n_payment_installments"]
    for field in numeric_fields:
        value = order[field]
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Field '{field}' must be numeric, got {type(value)}")
        if value < 0:
            raise ValidationError(f"Field '{field}' cannot be negative, got {value}")

    logger.info("Order passed basic validation.")
