"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# --- Request ---

class UserEventRequest(BaseModel):
    """All feature values for a single user to score."""
    user_id: str = Field(..., description="Unique user identifier")

    # login
    login_count_24h: float = Field(0.0)
    unique_ips_24h: float = Field(0.0)
    unique_countries_7d: float = Field(0.0)
    avg_login_hour: float = Field(12.0)
    night_login_ratio: float = Field(0.0)
    failed_then_success_pattern: float = Field(0.0)

    # financial
    total_deposits_7d: float = Field(0.0)
    total_withdrawals_7d: float = Field(0.0)
    deposit_withdrawal_ratio: float = Field(0.0)
    avg_withdrawal_amount: float = Field(0.0)
    withdrawal_zscore: float = Field(0.0)
    structuring_score: float = Field(0.0)

    # trading
    avg_trade_volume: float = Field(0.0)
    trade_volume_zscore: float = Field(0.0)
    max_volume_spike_ratio: float = Field(0.0)
    instrument_diversity: float = Field(0.0)
    trade_frequency_per_hour: float = Field(0.0)

    # session
    avg_session_duration: float = Field(0.0)
    session_duration_zscore: float = Field(0.0)
    rapid_navigation_score: float = Field(0.0)

    # device/ip
    unique_devices_7d: float = Field(0.0)
    ip_switch_rate: float = Field(0.0)
    device_mismatch_score: float = Field(0.0)

    # pnl
    pnl_volatility: float = Field(0.0)
    consistent_profit_score: float = Field(0.0)


# --- Responses ---

class FeatureContribution(BaseModel):
    feature: str
    value: float


class AnomalyResponse(BaseModel):
    user_id: str
    anomaly_score: float
    risk_level: str
    top_features: List[FeatureContribution] = []
    llm_alert: str = ""
    model_used: str = "isolation_forest"


class AlertItem(BaseModel):
    user_id: str
    anomaly_score: float
    risk_level: str
    top_feature: str = ""
    top_feature_value: float = 0.0


class AlertsResponse(BaseModel):
    total_alerts: int
    alerts: List[AlertItem]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class StatsResponse(BaseModel):
    total_users_analyzed: int
    anomaly_counts: Dict[str, int] = {}
    model_metrics: Dict[str, float] = {}
