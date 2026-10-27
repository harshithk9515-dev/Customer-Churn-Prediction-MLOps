"""Train the customer churn prediction model."""
import argparse
import os
import json

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


def train_model(data_path, model_path, metrics_path, check_quality_gate=False):
    """Train the churn prediction model and log with MLflow."""
    # Validate data first
    from validation import validate_data
    validate_data(data_path)

    # Load dataset
    df = pd.read_csv(data_path, keep_default_na=False)

    # Define features and target
    y = df["churn"].values
    X = df.drop("churn", axis=1)

    # Train/test split (80/20, stratified, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define preprocessing pipeline
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "internet_service"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Start MLflow run
    mlflow.start_run()
    try:
        # Train the model
        mlflow.log_param("train_test_split", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log parameters
        mlflow.log_param("accuracy", accuracy)
        mlflow.log_param("roc_auc", auc)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", auc)

        # Log the model artifact
        mlflow.sklearn.log_model(clf, "model")

        # Save model to artifacts
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(value=clf, filename=model_path)

        # Save metrics to JSON
        metrics = {"accuracy": float(accuracy), "roc_auc": float(auc)}
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Training complete. Accuracy: {accuracy:.4f}, ROC-AUC: {auc:.4f}")

        # Quality gate check
        if check_quality_gate and auc < 0.60:
            raise RuntimeError(
                f"ROC-AUC {auc:.4f} is below the minimum threshold of 0.60"
            )

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--data-path", type=str, default="data/customers.csv", help="Path to input data"
    )
    parser.add_argument(
        "--model-path", type=str, default="artifacts/model.joblib", help="Path to save model"
    )
    parser.add_argument(
        "--metrics-path", type=str, default="artifacts/metrics.json", help="Path to save metrics"
    )
    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
        help="Fail if ROC-AUC is below 0.60",
    )
    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        check_quality_gate=args.check_quality_gate,
    )