
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from churn_utils import prepare_data

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer will churn",
    version="1.0"
)

# Load trained model
model = joblib.load("model.joblib")


# Customer input details
class Customer(BaseModel):
    Gender: str
    Senior_Citizen: str
    Tenure_Months: float
    Contract_Type: str
    Internet_Service: str
    Payment_Method: str
    Monthly_Charges_INR: float
    Total_Charges_INR: float
    Tech_Support: str
    Online_Security: str
    Support_Tickets_6M: float
    Avg_Data_Usage_GB: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }


@app.post("/predict")
def predict_churn(customer: Customer):

    # Convert customer details into DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Prepare input data
    data = prepare_data(
        data.assign(Customer_ID="NEW_CUSTOMER")
    )

    # Predict churn
    prediction = model.predict(data)[0]

    return {
        "prediction": prediction,
        "message": (
            "Customer is likely to churn"
            if prediction == "Churn"
            else "Customer is likely to stay"
        )
    }
