
import pandas as pd

# Load the Excel dataset
file_path = "data/MLOps_Q4_Telecom_Customer_Churn_Dataset.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Customer Churn Data"
)

print("\nFIRST FIVE RECORDS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

print("\nDATASET LOADED SUCCESSFULLY!")
