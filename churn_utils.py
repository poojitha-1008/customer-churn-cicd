
import pandas as pd

def prepare_data(df):
    # Remove customer ID because it is not useful for prediction
    df = df.drop(columns=["Customer_ID"])

    # Convert numeric columns into numbers
    numeric_columns = [
        "Tenure_Months",
        "Monthly_Charges_INR",
        "Total_Charges_INR",
        "Support_Tickets_6M",
        "Avg_Data_Usage_GB"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical columns
    categorical_columns = [
        "Gender",
        "Senior_Citizen",
        "Contract_Type",
        "Internet_Service",
        "Payment_Method",
        "Tech_Support",
        "Online_Security"
    ]

    for col in categorical_columns:
        df[col] = df[col].astype("object")
        df[col] = df[col].replace(["", " ", "NA", "N/A"], float("nan"))

    return df
