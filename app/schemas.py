"""
Pydantic schemas for the API: define the exact shape of requests and
responses, so bad payloads are rejected clearly instead of reaching the model.
"""

from pydantic import BaseModel, Field
from typing import List


class OrderRequest(BaseModel):
    """A single incoming order to predict on."""
    total_price: float = Field(..., ge=0, description="Total price of items in the order")
    total_freight: float = Field(..., ge=0, description="Total freight/shipping value")
    n_items: int = Field(..., ge=1, description="Number of items in the order")
    total_payment_value: float = Field(..., ge=0, description="Total amount paid")
    n_payment_installments: int = Field(..., ge=0, description="Number of payment installments")
    order_purchase_timestamp: str = Field(..., description="Purchase timestamp, e.g. '2018-05-10 14:23:00'")
    customer_state: str = Field(..., description="Two-letter Brazilian state code, e.g. 'SP'")
    payment_type: str = Field(..., description="Payment type, e.g. 'credit_card'")
    product_category_name_english: str = Field(..., description="Product category in English")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """The result of a prediction for a single order."""
    is_late: int = Field(..., description="1 if predicted late, 0 if predicted on time")
    late_probability: float = Field(..., description="Predicted probability of being late")
    model_version: str = Field(..., description="Version of the API/model that produced this")
    latency_ms: float = Field(..., description="Time taken to produce this prediction, in ms")


class BatchOrderRequest(BaseModel):
    """A batch of orders to predict on in one request."""
    orders: List[OrderRequest]


class BatchPredictionResponse(BaseModel):
    """Results for a batch prediction request."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_alias: str
    model_version: str
    api_version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
