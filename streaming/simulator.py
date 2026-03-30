"""
Streaming simulator using asyncio.
Reads events from raw_events.csv and processes them in batches of 100,
scoring each batch with the trained Isolation Forest model.

Trade-off: using asyncio queues instead of Kafka. Kafka would need
ZooKeeper + broker setup which is overkill for a demo. The pipeline
logic is the same -- swap asyncio.Queue for a Kafka topic in prod.
"""

import asyncio
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


def compute_batch_features(batch_df, feature_cols):
    """Simplified feature computation from a single batch."""
    features_list = []

    for user_id, user_events in batch_df.groupby("user_id"):
        feats = {
            "user_id": user_id,
            "login_count_24h": len(user_events[user_events["event_type"] == "login"]),
            "unique_ips_24h": user_events["ip_address"].nunique(),
            "unique_countries_7d": user_events["country"].nunique(),
            "avg_login_hour": user_events["login_hour"].mean(),
            "night_login_ratio": len(user_events[
                (user_events["login_hour"] >= 0) & (user_events["login_hour"] <= 5)
            ]) / max(len(user_events), 1),
            "failed_then_success_pattern": 0,
            "total_deposits_7d": user_events[user_events["event_type"] == "deposit"]["amount"].sum(),
            "total_withdrawals_7d": user_events[user_events["event_type"] == "withdrawal"]["amount"].sum(),
            "deposit_withdrawal_ratio": 0,
            "avg_withdrawal_amount": user_events[user_events["event_type"] == "withdrawal"]["amount"].mean()
                if len(user_events[user_events["event_type"] == "withdrawal"]) > 0 else 0,
            "withdrawal_zscore": 0,
            "structuring_score": len(user_events[
                (user_events["event_type"] == "deposit") &
                (user_events["amount"] >= 500) & (user_events["amount"] <= 9999)
            ]),
            "avg_trade_volume": user_events["trade_volume"].mean(),
            "trade_volume_zscore": 0,
            "max_volume_spike_ratio": user_events["trade_volume"].max() / max(user_events["trade_volume"].mean(), 1),
            "instrument_diversity": user_events["trade_instrument"].nunique(),
            "trade_frequency_per_hour": len(user_events[
                user_events["event_type"].isin(["trade_open", "trade_close"])
            ]),
            "avg_session_duration": user_events["session_duration"].mean(),
            "session_duration_zscore": 0,
            "rapid_navigation_score": len(user_events[user_events["session_duration"] < 60]) / max(len(user_events), 1),
            "unique_devices_7d": user_events["device_id"].nunique(),
            "ip_switch_rate": user_events["ip_address"].nunique() / max(len(user_events), 1),
            "device_mismatch_score": len(user_events[["device_id", "ip_address"]].drop_duplicates()) / max(len(user_events), 1),
            "pnl_volatility": user_events["pnl"].std() if len(user_events) > 1 else 0,
            "consistent_profit_score": (user_events["pnl"] > 0).sum() / max(len(user_events[user_events["pnl"] != 0]), 1)
        }
        feats["deposit_withdrawal_ratio"] = feats["total_deposits_7d"] / max(feats["total_withdrawals_7d"], 1)
        features_list.append(feats)

    if not features_list:
        return pd.DataFrame()
    return pd.DataFrame(features_list).fillna(0)


async def producer(queue, events_df, batch_size=100):
    total = len(events_df)
    batch_num = 0

    for start in range(0, total, batch_size):
        batch = events_df.iloc[start:start + batch_size].copy()
        await queue.put(batch)
        batch_num += 1
        await asyncio.sleep(2)

        if batch_num >= 10:
            break

    await queue.put(None)  # signal done


async def consumer(queue, model_bundle, feature_cols):
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    batch_count = 0
    total_alerts = 0

    while True:
        batch = await queue.get()
        if batch is None:
            break

        batch_count += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Batch {batch_count} -- {len(batch)} events, "
              f"{batch['user_id'].nunique()} users")

        batch_features = compute_batch_features(batch, feature_cols)
        if batch_features.empty:
            continue

        user_ids = batch_features["user_id"]
        X_batch = batch_features[feature_cols].fillna(0)
        X_scaled = scaler.transform(X_batch)
        raw_scores = model.decision_function(X_scaled)

        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            scores = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            scores = np.zeros_like(raw_scores)

        alerts = 0
        for i, (uid, score) in enumerate(zip(user_ids, scores)):
            if score > 0.6:
                top_idx = np.argmax(np.abs(X_scaled[i]))
                feat_name = feature_cols[top_idx]
                feat_val = X_batch.iloc[i][feat_name]
                print(f"  ALERT: {uid} -- Score: {score:.3f} -- Reason: {feat_name} = {feat_val:.2f}")
                alerts += 1
                total_alerts += 1

        if alerts == 0:
            print(f"  No alerts in this batch")
        print(f"  Processed {len(batch_features)} users, {alerts} alerts")

    print(f"\nDone. {batch_count} batches, {total_alerts} total alerts.")


async def run_pipeline():
    print("ForexGuard streaming simulator")
    print("(asyncio-based -- see README for Kafka trade-off)\n")

    events_df = pd.read_csv("data/raw_events.csv")
    print(f"Loaded {len(events_df):,} events")

    model_bundle = joblib.load("models/saved/isolation_forest.pkl")
    feature_cols = model_bundle["feature_cols"]
    print(f"Model loaded ({len(feature_cols)} features)")

    print("\nProcessing 10 batches of 100 events, 2s apart...")

    queue = asyncio.Queue(maxsize=5)
    await asyncio.gather(
        producer(queue, events_df, batch_size=100),
        consumer(queue, model_bundle, feature_cols)
    )


def main():
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
