from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/customers.csv")
REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn",
]

def validate_data(file_path=DATA_PATH):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Read CSV without interpreting "None" as NaN
    df = pd.read_csv(file_path, keep_default_na=False)
    if df.empty:
        raise ValueError("Dataset is empty")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # Replace empty strings with NaN to catch blank values cleanly
    df_clean = df.replace(r"^\s*$", np.nan, regex=True)

    # Check for missing values (supports both regex matches)
    if df_clean[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Found null values: Dataset contains missing values")

    # Ensure numeric columns are cast properly for range checks
    for col in ["tenure", "monthly_charges", "support_calls"]:
        df[col] = pd.to_numeric(df[col])

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1")

    if (df["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("Monthly charges cannot be negative")

    if (df["support_calls"] < 0).any():
        raise ValueError("Support calls cannot be negative")

    if not set(df["contract"].unique()).issubset(
        {"Monthly", "One Year", "Two Year"}
    ):
        raise ValueError("Unexpected contract type")

    if not set(df["internet_service"].unique()).issubset(
        {"DSL", "Fiber", "None"}
    ):
        raise ValueError("Unexpected internet service")

    if df["churn"].nunique() != 2:
        raise ValueError("Target must contain both churn classes")

    print(f"Data validation passed: {len(df)} rows")
    return df

if __name__ == "__main__":
    validate_data()