"""
FastAPI application for real-time anomaly scoring.

Endpoints:
  POST /predict  -- score a user and get risk assessment
  GET  /alerts   -- top 20 most suspicious users
  GET  /health   -- health check
  GET  /stats    -- summary statistics
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    UserEventRequest, AnomalyResponse, FeatureContribution,
    AlertItem, AlertsResponse, HealthResponse, StatsResponse
)
from api.llm_alerts import generate_risk_summary, get_risk_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("forexguard")

# loaded at startup
models = {"isolation_forest": None, "scaler": None, "feature_cols": None, "loaded": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading ML models...")
    try:
        path = "models/saved/isolation_forest.pkl"
        if os.path.exists(path):
            bundle = joblib.load(path)
            models["isolation_forest"] = bundle["model"]
            models["scaler"] = bundle["scaler"]
            models["feature_cols"] = bundle["feature_cols"]
            models["loaded"] = True
            logger.info(f"Isolation Forest loaded ({len(bundle['feature_cols'])} features)")
        else:
            logger.warning(f"Model not found at {path}. Run training first.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ForexGuard API",
    description="Anomaly detection API for forex trading platforms",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def score_user(features_array):
    """Score with Isolation Forest. Returns (score, top_3_feature_indices)."""
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded. Run the pipeline first.")

    scaled = models["scaler"].transform(features_array.reshape(1, -1))
    raw_score = models["isolation_forest"].decision_function(scaled)[0]

    # sigmoid normalization to 0-1
    anomaly_score = float(np.clip(1 / (1 + np.exp(raw_score * 5)), 0, 1))

    # top 3 most unusual features (highest absolute z-score)
    top_3 = np.argsort(np.abs(scaled[0]))[-3:][::-1]
    return anomaly_score, top_3


@app.post("/predict", response_model=AnomalyResponse)
async def predict_anomaly(request: UserEventRequest):
    """Score a user's behavior and return risk assessment with explanation."""
    logger.info(f"Scoring user: {request.user_id}")

    try:
        feature_cols = models["feature_cols"]
        features = np.array([getattr(request, col) for col in feature_cols])

        anomaly_score, top_3_idx = score_user(features)
        risk_level = get_risk_level(anomaly_score)

        top_features = [
            FeatureContribution(feature=feature_cols[i], value=float(features[i]))
            for i in top_3_idx
        ]

        # generate LLM compliance alert for high-risk users
        llm_alert = ""
        if risk_level in ["HIGH", "CRITICAL"]:
            llm_alert = generate_risk_summary(
                user_id=request.user_id,
                anomaly_score=anomaly_score,
                top_features=[{"feature": f.feature, "value": f.value} for f in top_features],
                risk_level=risk_level
            )

        logger.info(f"  Score: {anomaly_score:.3f}, Risk: {risk_level}")
        return AnomalyResponse(
            user_id=request.user_id, anomaly_score=round(anomaly_score, 4),
            risk_level=risk_level, top_features=top_features,
            llm_alert=llm_alert, model_used="isolation_forest"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts():
    """Top 20 most suspicious users from the batch scores."""
    try:
        scores_path = "data/if_scores.csv"
        if not os.path.exists(scores_path):
            raise HTTPException(status_code=404, detail="No scores found. Run the pipeline first.")

        scores_df = pd.read_csv(scores_path)

        # merge with SHAP if available
        shap_path = "data/shap_explanations.csv"
        if os.path.exists(shap_path):
            shap_df = pd.read_csv(shap_path)
            scores_df = scores_df.merge(shap_df[["user_id", "feature_1", "value_1"]],
                                         on="user_id", how="left")
        else:
            scores_df["feature_1"] = ""
            scores_df["value_1"] = 0.0

        top_20 = scores_df.nlargest(20, "anomaly_score")
        alerts = [
            AlertItem(
                user_id=row["user_id"],
                anomaly_score=round(row["anomaly_score"], 4),
                risk_level=get_risk_level(row["anomaly_score"]),
                top_feature=str(row.get("feature_1", "")),
                top_feature_value=float(row.get("value_1", 0.0))
            )
            for _, row in top_20.iterrows()
        ]
        return AlertsResponse(total_alerts=len(alerts), alerts=alerts)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", models_loaded=models["loaded"])


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Risk level breakdown and model metrics."""
    try:
        scores_path = "data/if_scores.csv"
        if not os.path.exists(scores_path):
            raise HTTPException(status_code=404, detail="No scores found.")

        scores_df = pd.read_csv(scores_path)
        scores_df["risk_level"] = scores_df["anomaly_score"].apply(get_risk_level)
        risk_counts = scores_df["risk_level"].value_counts().to_dict()

        metrics = {}
        comp_path = "data/model_comparison.csv"
        if os.path.exists(comp_path):
            comp = pd.read_csv(comp_path)
            if_row = comp[comp["Model"] == "Isolation Forest"]
            if len(if_row) > 0:
                metrics = {
                    "precision": float(if_row["Precision"].iloc[0]),
                    "recall": float(if_row["Recall"].iloc[0]),
                    "f1_score": float(if_row["F1 Score"].iloc[0]),
                    "roc_auc": float(if_row["ROC-AUC"].iloc[0])
                }

        return StatsResponse(
            total_users_analyzed=len(scores_df),
            anomaly_counts=risk_counts,
            model_metrics=metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
