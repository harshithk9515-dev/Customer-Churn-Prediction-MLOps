"""Unit and integration tests for the customer churn MLOps pipeline."""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import validate_data


class TestDataValidity:
    """Test that the generated data passes validation."""

    def test_data_validity(self):
        """Test that data/customers.csv passes all validation checks."""
        validate_data("data/customers.csv")


class TestValidationRejection:
    """Test that validation rejects invalid data."""

    def test_validation_rejects_missing_data(self, tmp_path):
        """Test that validate_data raises ValueError on missing values."""
        # Create a CSV with a missing value (empty field)
        invalid_path = os.path.join(str(tmp_path), "invalid.csv")
        with open(invalid_path, "w") as f:
            f.write("tenure,monthly_charges,support_calls,contract,internet_service,churn\n")
            f.write("1,,0,Monthly,DSL,0\n")  # empty monthly_charges
            f.write("3,50.0,1,One Year,Fiber,1\n")
            f.write("5,70.0,2,Two Year,None,0\n")
    
        with pytest.raises(ValueError, match="Found null values"):
            validate_data(invalid_path)


class TestArtifactsExistence:
    """Test that model artifacts exist after training."""

    def test_model_artifact_exists(self):
        """Test that artifacts/model.joblib exists."""
        assert os.path.exists("artifacts/model.joblib"), (
            "artifacts/model.joblib not found. "
            "Run train.py first to generate the model."
        )

    def test_metrics_artifact_exists(self):
        """Test that artifacts/metrics.json exists."""
        assert os.path.exists("artifacts/metrics.json"), (
            "artifacts/metrics.json not found. "
            "Run train.py first to generate the metrics."
        )


class TestModelPrediction:
    """Test model prediction output shape and probability properties."""

    def test_prediction_output_shape(self):
        """Test that model predictions have the correct shape."""
        import joblib
        from sklearn.metrics import roc_auc_score

        clf = joblib.load("artifacts/model.joblib")
        import pandas as pd

        df = pd.read_csv("data/customers.csv", keep_default_na=False)
        X = df.drop("churn", axis=1)
        y = df["churn"].values

        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)[:, 1]

        # Check prediction shape matches input
        assert len(predictions) == len(y), (
            f"Predictions length {len(predictions)} != "
            f"input length {len(y)}"
        )

        # Check probability shape matches
        assert len(probabilities) == len(y), (
            f"Probabilities length {len(probabilities)} != "
            f"input length {len(y)}"
        )

    def test_probability_sum(self):
        """Test that predicted probabilities are reasonable."""
        import joblib
        import pandas as pd

        clf = joblib.load("artifacts/model.joblib")
        df = pd.read_csv("data/customers.csv", keep_default_na=False)
        X = df.drop("churn", axis=1)

        probs = clf.predict_proba(X)[:, 1]

        # All probabilities should be in [0, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1), (
            "Probabilities should be in [0, 1]"
        )

        # Probability sum should be reasonable (not all 0 or all 1)
        prob_sum = probs.sum()
        assert 0 < prob_sum < len(probs), (
            f"Probability sum {prob_sum} is outside expected range"
        )


class TestModelQuality:
    """Test model quality metrics AUC bounds and minimum threshold."""

    def test_auc_valid_bounds(self):
        """Test that AUC is within valid bounds [0, 1]."""
        import json

        with open("artifacts/metrics.json", "r") as f:
            metrics = json.load(f)

        auc = metrics["roc_auc"]
        assert 0.0 <= auc <= 1.0, f"AUC {auc} is outside valid bounds [0, 1]"

    def test_auc_minimum_threshold(self):
        """Test that AUC meets the minimum threshold of 0.60."""
        import json

        with open("artifacts/metrics.json", "r") as f:
            metrics = json.load(f)

        auc = metrics["roc_auc"]
        min_threshold = 0.60
        assert auc >= min_threshold, (
            f"AUC {auc} is below minimum threshold {min_threshold}"
        )