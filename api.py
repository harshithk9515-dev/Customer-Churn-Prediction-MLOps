"""FastAPI service for customer churn prediction."""

import os
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import Literal

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager


# --- Pydantic model for input validation ---


class Customer(BaseModel):
    """Pydantic model for a customer request (features only, no target)."""

    tenure: float = Field(..., description="Tenure in months (1-72)")
    monthly_charges: float = Field(
        ..., description="Monthly charges (20-130)"
    )
    support_calls: int = Field(
        ..., description="Number of support calls, non-negative"
    )
    contract: Literal["Monthly", "One Year", "Two Year"] = Field(
        ..., description="Contract type"
    )
    internet_service: Literal["DSL", "Fiber", "None"] = Field(
        ..., description="Internet service type"
    )

    @field_validator("tenure")
    def tenure_range(cls, v):
        if v < 1 or v > 72:
            raise ValueError("tenure must be between 1 and 72")
        return v

    @field_validator("monthly_charges")
    def charges_range(cls, v):
        if v < 20 or v > 130:
            raise ValueError("monthly_charges must be between 20 and 130")
        return v

    @field_validator("support_calls")
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("support_calls must be non-negative")
        return v


# --- ML model loading via lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model on startup."""
    model_path = "artifacts/model.joblib"
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. Train the model first."
        )
    clf = joblib.load(model_path)
    app.state.model = clf
    yield


# --- FastAPI app initialization ---


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint returning service info."""
    return {
        "message": "Customer Churn Prediction MLOps API",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(customer: Customer):
    """Predict churn probability for a customer."""
    model = app.state.model

    # Convert input to DataFrame (matching training feature order)
    df = pd.DataFrame([customer.model_dump().values()],
                      columns=["tenure", "monthly_charges", "support_calls",
                               "contract", "internet_service"])

    # Get probability of churn class (class 1)
    proba = model.predict_proba(df)[:, 1][0]
    prediction = int(model.predict(df)[0])

    return {
        "churn_prediction": prediction,
        "churn_probability": float(proba),
        "churn_risk": "High" if proba > 0.5 else "Low",
    }