
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    customer = {
        "Gender": "Male",
        "Senior_Citizen": "No",
        "Tenure_Months": 12,
        "Contract_Type": "Month-to-month",
        "Internet_Service": "Fiber optic",
        "Payment_Method": "Electronic check",
        "Monthly_Charges_INR": 850,
        "Total_Charges_INR": 10200,
        "Tech_Support": "No",
        "Online_Security": "No",
        "Support_Tickets_6M": 5,
        "Avg_Data_Usage_GB": 75
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "Churn",
        "Not Churn"
    ]


def test_invalid_input():
    response = client.post("/predict", json={})
    assert response.status_code == 422
