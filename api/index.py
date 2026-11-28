"""Vercel Serverless Function entry point for ConversionIQ FastAPI API."""
import math
from pathlib import Path
from typing import Literal
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

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

# Parameters extracted from the trained Pipeline(StandardScaler, LogisticRegression).
# Using exact formula computation keeps the Vercel function bundle ~15MB instead of 270MB+
# (preventing AWS Lambda/Vercel size limit errors) and enables instant cold starts.
SCALER_MEAN = [
    2.338550665718061, 80.67557968444304, 0.5028966358369753, 35.44457252446773,
    31.665921333468848, 1202.1309802170388, 0.021353204579632076, 0.04218055310448217,
    6.402395670885456, 0.06195751600772435, 0.12521597723345868, 0.23457668462242098,
    0.0458379916658197, 0.19402378290476674, 0.10620998068909442, 0.6971236914320561,
]

SCALER_SCALE = [
    3.330848397191776, 176.66476228801486, 1.2977212488306165, 144.74250397176414,
    43.77638955057694, 2014.1574786924289, 0.049499312218669135, 0.05069516875684017,
    28.540091296474266, 0.19953974336457214, 0.33096364797199196, 0.4237339538601718,
    0.2091336180193513, 0.3954472841885384, 0.30810618411696417, 0.4595021765739522,
]

MODEL_COEF = [
    0.08215223759293507, 0.03503877581647857, 0.09523317432465425, -0.051270638818320426,
    0.13610530819245603, 0.08433999170581823, -0.40424112329918543, -0.4045851720768666,
    3.1569429752255798, -0.1341135249595962, -0.11126340061359408, -0.18197104049293727,
    0.17309852492464498, 0.34041475317127184, 0.04106909578622636, -0.02813723484399387,
]

MODEL_INTERCEPT = -0.5534685942676714

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

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
@app.get("/frontend/conversioniq.html", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=get_index_html())

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(FEATURE_NAMES),
    }

@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_values = dict(DEFAULT_BASELINE_VALUES)
    overrides = {
        "ProductRelated": float(request.pages),
        "ProductRelated_Duration": float(request.duration),
        "PageValues": float(request.page_value),
    }
    input_values.update(overrides)

    feature_values = [float(input_values[name]) for name in FEATURE_NAMES]

    # Logistic Regression inference: z = intercept + sum(coef_i * (x_i - mean_i) / scale_i)
    z = MODEL_INTERCEPT
    for x, m, s, c in zip(feature_values, SCALER_MEAN, SCALER_SCALE, MODEL_COEF):
        z += c * ((x - m) / s)

    # Sigmoid function for probability
    probability = 1.0 / (1.0 + math.exp(-z))
    positive_prediction = probability >= 0.5

    priority = "high" if probability >= 0.70 else "medium" if probability >= 0.40 else "low"
    return PredictionResponse(
        purchase_probability=probability,
        predicted_class="purchase" if positive_prediction else "no_purchase",
        priority=priority,
    )
