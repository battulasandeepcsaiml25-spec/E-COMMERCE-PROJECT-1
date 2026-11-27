"""Vercel Serverless Function entry point for ConversionIQ FastAPI API."""
from pathlib import Path
from typing import Literal
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Base directories and asset paths
BASE_DIR = Path(__file__).resolve().parent

def find_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / filename,
        BASE_DIR.parent / "backend" / filename,
        Path.cwd() / "backend" / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return BASE_DIR / filename

def get_index_html() -> str:
    candidates = [
        BASE_DIR / "index.html",
        BASE_DIR.parent / "index.html",
        BASE_DIR.parent / "frontend" / "conversioniq.html",
        Path.cwd() / "index.html",
        Path.cwd() / "frontend" / "conversioniq.html",
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return "<h1>ConversionIQ Prediction Advisor</h1><p>index.html not found.</p>"

MODEL_PATH = find_file("purchase_conversion_model.pkl")
DATA_PATH = find_file("selected_features_dataset.csv")
TARGET_COLUMN = "Revenue"

FEATURE_NAMES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month_Mar",
    "Month_May",
    "Month_NOV",
    "Month_Nov",
    "VisitorType_New_Visitor",
    "VisitorType_Returning_Visitor",
]

DEFAULT_BASELINE_VALUES = {
    "Administrative": 1.0,
    "Administrative_Duration": 8.0,
    "Informational": 0.0,
    "Informational_Duration": 0.0,
    "ProductRelated": 18.0,
    "ProductRelated_Duration": 606.1,
    "BounceRates": 0.003030303,
    "ExitRates": 0.025,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month_Mar": 0.0,
    "Month_May": 0.0,
    "Month_NOV": 0.0,
    "Month_Nov": 0.0,
    "VisitorType_New_Visitor": 0.0,
    "VisitorType_Returning_Visitor": 1.0,
}

app = FastAPI(title="ConversionIQ Prediction API")

# Broad CORS for deployment environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    pages: int = Field(ge=0)
    duration: float = Field(ge=0)
    page_value: float = Field(ge=0)

class PredictionResponse(BaseModel):
    purchase_probability: float = Field(ge=0, le=1)
    predicted_class: str
    priority: Literal["low", "medium", "high"]

model = None
feature_names: list[str] = list(FEATURE_NAMES)
baseline_values: dict = dict(DEFAULT_BASELINE_VALUES)
load_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    if DATA_PATH.exists():
        try:
            dataset = pd.read_csv(DATA_PATH)
            if TARGET_COLUMN in dataset.columns:
                feature_frame = dataset.drop(columns=[TARGET_COLUMN])
                feature_names = feature_frame.columns.tolist()
                calculated_medians = feature_frame.median(numeric_only=True).to_dict()
                for feat in feature_names:
                    if feat in calculated_medians:
                        baseline_values[feat] = calculated_medians[feat]
                    else:
                        modes = feature_frame[feat].mode(dropna=True)
                        baseline_values[feat] = modes.iloc[0] if not modes.empty else 0
        except Exception:
            # Fall back to validated DEFAULT_BASELINE_VALUES if dataset read fails
            pass
except Exception as exc:
    model = None
    load_error = str(exc)

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
@app.get("/frontend/conversioniq.html", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=get_index_html())

@app.get("/health")
@app.get("/api/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {load_error}")
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(feature_names),
    }

@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {load_error}")

    input_values = dict(baseline_values)
    overrides = {
        "ProductRelated": request.pages,
        "ProductRelated_Duration": request.duration,
        "PageValues": request.page_value,
    }
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
