"""
Integration tests for the FastAPI routes, end to end: health check,
model info, single predict, and batch predict, using FastAPI's TestClient
(no real server process needed).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


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


def test_health_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_route(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_name" in body
    assert "model_version" in body


def test_predict_route_with_valid_order(client):
    response = client.post("/predict", json=GOOD_ORDER)
    assert response.status_code == 200
    body = response.json()
    assert body["is_late"] in (0, 1)
    assert 0.0 <= body["late_probability"] <= 1.0


def test_predict_route_rejects_bad_payload(client):
    bad_order = dict(GOOD_ORDER)
    del bad_order["total_price"]  # missing required field
    response = client.post("/predict", json=bad_order)
    assert response.status_code == 422  # FastAPI's own schema validation catches this


def test_predict_route_rejects_unknown_state(client):
    bad_order = dict(GOOD_ORDER)
    bad_order["customer_state"] = "ZZ"
    response = client.post("/predict", json=bad_order)
    assert response.status_code == 422  # caught by our Great Expectations layer


def test_predict_batch_route(client):
    response = client.post("/predict-batch", json={"orders": [GOOD_ORDER, GOOD_ORDER]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
