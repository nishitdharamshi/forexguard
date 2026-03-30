"""
Feature engineering pipeline.
Reads raw event data and computes per-user behavioral features
that surface suspicious trading patterns.

6 feature groups:
  - Login behavior (access patterns, geo, timing)
  - Financial patterns (deposits, withdrawals, structuring)
  - Trading behavior (volume, spikes, frequency)
  - Session characteristics (duration, bot-like patterns)
  - Device/IP signals (sharing, switching)
  - PnL analysis (volatility, consistency)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

ANOMALY_USER_IDS = set([f"ANOM-{str(i).zfill(4)}" for i in range(200)])


def load_raw_data(path="data/raw_events.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} events for {df['user_id'].nunique()} users")
    return df


def compute_login_features(df):
    """Login patterns -- catches compromised accounts, bots, geo-impossibility."""
    print("  login features...")
    login_events = df[df["event_type"] == "login"].copy()
    max_time = df["timestamp"].max()

    features = {}
    for user_id, group in df.groupby("user_id"):
        user_logins = login_events[login_events["user_id"] == user_id]

        recent_24h = user_logins[user_logins["timestamp"] >= max_time - pd.Timedelta(hours=24)]
        login_count_24h = len(recent_24h)
        unique_ips_24h = recent_24h["ip_address"].nunique() if len(recent_24h) > 0 else 0

        recent_7d = user_logins[user_logins["timestamp"] >= max_time - pd.Timedelta(days=7)]
        unique_countries_7d = recent_7d["country"].nunique() if len(recent_7d) > 0 else 0

        avg_login_hour = group["login_hour"].mean() if len(group) > 0 else 12
        total_events = len(group)
        night_events = len(group[(group["login_hour"] >= 0) & (group["login_hour"] <= 5)])
        night_login_ratio = night_events / max(total_events, 1)

        # proxy for failed-then-success: multiple logins within 5 min
        if len(user_logins) >= 2:
            login_diffs = user_logins["timestamp"].sort_values().diff().dt.total_seconds()
            failed_then_success = int((login_diffs < 300).sum())
        else:
            failed_then_success = 0

        features[user_id] = {
            "login_count_24h": login_count_24h,
            "unique_ips_24h": unique_ips_24h,
            "unique_countries_7d": unique_countries_7d,
            "avg_login_hour": round(avg_login_hour, 2),
            "night_login_ratio": round(night_login_ratio, 4),
            "failed_then_success_pattern": failed_then_success
        }
    return pd.DataFrame.from_dict(features, orient="index")


def compute_financial_features(df):
    """Financial patterns -- money laundering, structuring detection."""
    print("  financial features...")
    max_time = df["timestamp"].max()

    features = {}
    for user_id, group in df.groupby("user_id"):
        recent_7d = group[group["timestamp"] >= max_time - pd.Timedelta(days=7)]
        deposits = group[group["event_type"] == "deposit"]
        withdrawals = group[group["event_type"] == "withdrawal"]
        deposits_7d = recent_7d[recent_7d["event_type"] == "deposit"]
        withdrawals_7d = recent_7d[recent_7d["event_type"] == "withdrawal"]

        total_deposits_7d = deposits_7d["amount"].sum()
        total_withdrawals_7d = withdrawals_7d["amount"].sum()
        deposit_withdrawal_ratio = total_deposits_7d / max(total_withdrawals_7d, 1)
        avg_withdrawal_amount = withdrawals["amount"].mean() if len(withdrawals) > 0 else 0

        if len(withdrawals) >= 2:
            withdrawal_zscore = float(np.abs(stats.zscore(withdrawals["amount"])).max())
        else:
            withdrawal_zscore = 0.0

        # count deposits in the $500-$9,999 range (structuring indicator)
        structuring_deposits = deposits[
            (deposits["amount"] >= 500) & (deposits["amount"] <= 9999)
        ]

        features[user_id] = {
            "total_deposits_7d": round(total_deposits_7d, 2),
            "total_withdrawals_7d": round(total_withdrawals_7d, 2),
            "deposit_withdrawal_ratio": round(deposit_withdrawal_ratio, 4),
            "avg_withdrawal_amount": round(avg_withdrawal_amount, 2),
            "withdrawal_zscore": round(withdrawal_zscore, 4),
            "structuring_score": len(structuring_deposits)
        }
    return pd.DataFrame.from_dict(features, orient="index")


def compute_trading_features(df):
    """Trading patterns -- volume spikes, unusual frequency."""
    print("  trading features...")
    trade_events = df[df["event_type"].isin(["trade_open", "trade_close"])]

    features = {}
    for user_id, group in df.groupby("user_id"):
        user_trades = trade_events[trade_events["user_id"] == user_id]

        if len(user_trades) > 0:
            avg_trade_volume = user_trades["trade_volume"].mean()
            if len(user_trades) >= 2:
                trade_volume_zscore = float(np.abs(stats.zscore(user_trades["trade_volume"])).max())
            else:
                trade_volume_zscore = 0.0

            rolling_mean = user_trades["trade_volume"].expanding().mean()
            max_volume_spike_ratio = user_trades["trade_volume"].max() / max(rolling_mean.mean(), 1)
            instrument_diversity = user_trades["trade_instrument"].nunique()

            time_span_hours = max(
                (user_trades["timestamp"].max() - user_trades["timestamp"].min()).total_seconds() / 3600, 1
            )
            trade_frequency_per_hour = len(user_trades) / time_span_hours
        else:
            avg_trade_volume = 0.0
            trade_volume_zscore = 0.0
            max_volume_spike_ratio = 0.0
            instrument_diversity = 0
            trade_frequency_per_hour = 0.0

        features[user_id] = {
            "avg_trade_volume": round(avg_trade_volume, 2),
            "trade_volume_zscore": round(trade_volume_zscore, 4),
            "max_volume_spike_ratio": round(max_volume_spike_ratio, 4),
            "instrument_diversity": instrument_diversity,
            "trade_frequency_per_hour": round(trade_frequency_per_hour, 4)
        }
    return pd.DataFrame.from_dict(features, orient="index")


def compute_session_features(df):
    """Session patterns -- bot detection."""
    print("  session features...")
    features = {}
    for user_id, group in df.groupby("user_id"):
        avg_session = group["session_duration"].mean()
        if len(group) >= 2:
            session_zscore = float(np.abs(stats.zscore(group["session_duration"])).max())
        else:
            session_zscore = 0.0

        short_sessions = len(group[group["session_duration"] < 60])
        rapid_navigation_score = short_sessions / max(len(group), 1)

        features[user_id] = {
            "avg_session_duration": round(avg_session, 2),
            "session_duration_zscore": round(session_zscore, 4),
            "rapid_navigation_score": round(rapid_navigation_score, 4)
        }
    return pd.DataFrame.from_dict(features, orient="index")


def compute_device_ip_features(df):
    """Device/IP signals -- account sharing, compromised creds."""
    print("  device/IP features...")
    max_time = df["timestamp"].max()

    features = {}
    for user_id, group in df.groupby("user_id"):
        recent_7d = group[group["timestamp"] >= max_time - pd.Timedelta(days=7)]
        unique_devices_7d = recent_7d["device_id"].nunique() if len(recent_7d) > 0 else 0
        ip_switch_rate = group["ip_address"].nunique() / max(len(group), 1)

        device_ip_combos = group[["device_id", "ip_address"]].drop_duplicates()
        device_mismatch_score = len(device_ip_combos) / max(len(group), 1)

        features[user_id] = {
            "unique_devices_7d": unique_devices_7d,
            "ip_switch_rate": round(ip_switch_rate, 4),
            "device_mismatch_score": round(device_mismatch_score, 4)
        }
    return pd.DataFrame.from_dict(features, orient="index")


def compute_pnl_features(df):
    """PnL analysis -- wash trading, suspicious consistency."""
    print("  PnL features...")
    features = {}
    for user_id, group in df.groupby("user_id"):
        pnl_values = group[group["pnl"] != 0]["pnl"]

        if len(pnl_values) >= 2:
            pnl_volatility = pnl_values.std()
            consistent_profit_score = (pnl_values > 0).sum() / len(pnl_values)
        else:
            pnl_volatility = 0.0
            consistent_profit_score = 0.0

        features[user_id] = {
            "pnl_volatility": round(pnl_volatility, 2),
            "consistent_profit_score": round(consistent_profit_score, 4)
        }
    return pd.DataFrame.from_dict(features, orient="index")


def main():
    print("Running feature engineering pipeline...")
    df = load_raw_data()

    print("Computing features:")
    login_feats = compute_login_features(df)
    financial_feats = compute_financial_features(df)
    trading_feats = compute_trading_features(df)
    session_feats = compute_session_features(df)
    device_ip_feats = compute_device_ip_features(df)
    pnl_feats = compute_pnl_features(df)

    # merge all feature groups
    feature_df = login_feats \
        .join(financial_feats, how="outer") \
        .join(trading_feats, how="outer") \
        .join(session_feats, how="outer") \
        .join(device_ip_feats, how="outer") \
        .join(pnl_feats, how="outer")

    feature_df = feature_df.fillna(0)
    feature_df.index.name = "user_id"
    feature_df = feature_df.reset_index()

    # label anomalies
    feature_df["is_anomaly"] = feature_df["user_id"].apply(
        lambda x: 1 if x in ANOMALY_USER_IDS else 0
    )

    output_path = os.path.join("data", "features.csv")
    feature_df.to_csv(output_path, index=False)

    n_features = len(feature_df.columns) - 2  # exclude user_id and is_anomaly
    print(f"\nDone. {len(feature_df)} users, {n_features} features each.")
    print(f"  Anomaly: {feature_df['is_anomaly'].sum()}, Normal: {(feature_df['is_anomaly']==0).sum()}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
