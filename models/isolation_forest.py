"""
Isolation Forest and Local Outlier Factor training.
Two unsupervised anomaly detection models -- IF catches global outliers,
LOF catches local density-based anomalies.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def load_features(path="data/features.csv"):
    df = pd.read_csv(path)
    user_ids = df["user_id"]
    y_true = df["is_anomaly"]
    feature_cols = [c for c in df.columns if c not in ["user_id", "is_anomaly"]]
    X = df[feature_cols]
    print(f"Loaded {len(df)} users, {len(feature_cols)} features, {y_true.sum()} anomalies ({y_true.mean()*100:.1f}%)")
    return X, y_true, user_ids, feature_cols


def normalize_scores(scores):
    """Scale scores to 0-1 where 1 = most anomalous."""
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def evaluate(y_true, scores, name, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "model": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
    }
    return metrics


def train_isolation_forest(X, y_true, user_ids, feature_cols):
    print("\nTraining Isolation Forest...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination=0.2 because we know ~20% of users are anomalous
    model = IsolationForest(
        n_estimators=200,
        contamination=0.2,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # IF returns negative scores (more negative = more anomalous), so we flip
    raw_scores = model.decision_function(X_scaled)
    anomaly_scores = normalize_scores(-raw_scores)

    metrics = evaluate(y_true, anomaly_scores, "Isolation Forest")

    scores_df = pd.DataFrame({
        "user_id": user_ids, "anomaly_score": anomaly_scores, "is_anomaly": y_true
    })
    scores_df.to_csv("data/if_scores.csv", index=False)

    os.makedirs("models/saved", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols},
                "models/saved/isolation_forest.pkl")

    print(f"  Precision={metrics['precision']}, Recall={metrics['recall']}, "
          f"F1={metrics['f1_score']}, AUC={metrics['roc_auc']}")
    return metrics


def train_lof(X, y_true, user_ids, feature_cols):
    print("\nTraining Local Outlier Factor...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LocalOutlierFactor(n_neighbors=20, contamination=0.2, novelty=False, n_jobs=-1)
    model.fit_predict(X_scaled)

    raw_scores = model.negative_outlier_factor_
    anomaly_scores = normalize_scores(-raw_scores)

    metrics = evaluate(y_true, anomaly_scores, "LOF")

    scores_df = pd.DataFrame({
        "user_id": user_ids, "anomaly_score": anomaly_scores, "is_anomaly": y_true
    })
    scores_df.to_csv("data/lof_scores.csv", index=False)

    os.makedirs("models/saved", exist_ok=True)
    # save a novelty=True version so we can score new data later
    lof_novelty = LocalOutlierFactor(n_neighbors=20, contamination=0.2, novelty=True)
    lof_novelty.fit(X_scaled)
    joblib.dump({"model": lof_novelty, "scaler": scaler, "feature_cols": feature_cols},
                "models/saved/lof.pkl")

    print(f"  Precision={metrics['precision']}, Recall={metrics['recall']}, "
          f"F1={metrics['f1_score']}, AUC={metrics['roc_auc']}")
    return metrics


def main():
    print("Training anomaly detection models...")
    X, y_true, user_ids, feature_cols = load_features()

    if_metrics = train_isolation_forest(X, y_true, user_ids, feature_cols)
    lof_metrics = train_lof(X, y_true, user_ids, feature_cols)

    print(f"\n{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 62)
    for m in [if_metrics, lof_metrics]:
        print(f"{m['model']:<20} {m['precision']:>10} {m['recall']:>10} {m['f1_score']:>10} {m['roc_auc']:>10}")

    comparison_df = pd.DataFrame([if_metrics, lof_metrics])
    comparison_df.to_csv("data/if_lof_comparison.csv", index=False)
    print("\nSaved to data/if_lof_comparison.csv")


if __name__ == "__main__":
    main()
