"""FastAPI adapter matching the three-input Streamlit prediction flow."""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "purchase_conversion_model.pkl"
DATA_PATH = BASE_DIR / "selected_features_dataset.csv"
TARGET_COLUMN = "Revenue"

app = FastAPI(title="ConversionIQ Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


class PredictionRequest(BaseModel):
    """The same three values exposed by the Streamlit form."""
    pages: int = Field(ge=0)
    duration: float = Field(ge=0)
    page_value: float = Field(ge=0)


class PredictionResponse(BaseModel):
    purchase_probability: float = Field(ge=0, le=1)
    predicted_class: str
    priority: Literal["low", "medium", "high"]


model = None
dataset = None
feature_names: list[str] = []
baseline_values: dict = {}
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    dataset = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"{DATA_PATH.name} must contain a {TARGET_COLUMN!r} column.")

    feature_frame = dataset.drop(columns=[TARGET_COLUMN])
    feature_names = feature_frame.columns.tolist()

    # Match the Streamlit code: numeric columns use the median; any remaining
    # non-numeric columns use their mode, falling back to 0 if no mode exists.
    baseline_values = feature_frame.median(numeric_only=True).to_dict()
    for feature in feature_names:
        if feature not in baseline_values:
            modes = feature_frame[feature].mode(dropna=True)
            baseline_values[feature] = modes.iloc[0] if not modes.empty else 0

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and list(model_feature_names) != feature_names:
        raise ValueError(
            "Model and selected_features_dataset.csv columns do not match in name/order."
        )
except Exception as exc:
    model = None
    load_error = str(exc)


@app.get("/health")
def health():
    if model is None or dataset is None:
        raise HTTPException(status_code=503, detail=f"Model/data unavailable: {load_error}")
    return {
        "status": "ok",
        "model_loaded": True,
        "model": MODEL_PATH.name,
        "feature_count": len(feature_names),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or dataset is None:
        raise HTTPException(status_code=503, detail=f"Model/data unavailable: {load_error}")

    # Start with the dataset-derived medians/modes, then replace the same three
    # features that the Streamlit UI overrides.
    input_values = dict(baseline_values)
    overrides = {
        "ProductRelated": request.pages,
        "ProductRelated_Duration": request.duration,
        "PageValues": request.page_value,
    }
    missing_overrides = [name for name in overrides if name not in input_values]
    if missing_overrides:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset is missing Streamlit input features: {missing_overrides}",
        )
    input_values.update(overrides)
    row = pd.DataFrame([[input_values[name] for name in feature_names]], columns=feature_names)

    try:
        predicted = model.predict(row)[0]
        if not hasattr(model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba().")
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        positive_index = next(
            (
                i for i, label in enumerate(classes)
                if str(label).strip().lower() in {"1", "true", "yes"}
            ),
            None,
        )
        if positive_index is None:
            raise ValueError(f"Could not identify purchase class in {classes!r}.")
        probability = float(probabilities[positive_index])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    priority = "high" if probability >= 0.70 else "medium" if probability >= 0.40 else "low"
    positive_prediction = str(predicted).strip().lower() in {"1", "true", "yes"}
    return PredictionResponse(
        purchase_probability=probability,
        predicted_class="purchase" if positive_prediction else "no_purchase",
        priority=priority,
    )
