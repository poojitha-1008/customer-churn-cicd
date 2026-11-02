
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from churn_utils import prepare_data

# 1. Load dataset
file_path = "data/MLOps_Q4_Telecom_Customer_Churn_Dataset.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Customer Churn Data"
)

print("Dataset loaded successfully!")

# 2. Prepare dataset
df = prepare_data(df)

# 3. Separate input and output
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(str).str.strip()

# 4. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# 5. Preprocessing for numeric data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 9. Create machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

best_model = None
best_score = 0
best_name = ""

# 10. Train and evaluate models
for name, classifier in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Churn"
    )

    print("\nModel:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("F1 Score:", round(f1, 4))
    print(classification_report(y_test, predictions))

    if f1 > best_score:
        best_score = f1
        best_model = pipeline
        best_name = name

# 11. Save the best model
joblib.dump(best_model, "model.joblib")

print("\nBest Model:", best_name)
print("Best F1 Score:", round(best_score, 4))
print("Model saved successfully as model.joblib")
