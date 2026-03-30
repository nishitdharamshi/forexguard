"""
Model comparison and ensemble scoring.
Loads scores from all three models, averages them into an ensemble,
and picks the best performer.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate(y_true, scores, name, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    return {
        "Model": name,
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, scores), 4),
        "Avg Score (Anomaly)": round(scores[y_true == 1].mean(), 4),
        "Avg Score (Normal)": round(scores[y_true == 0].mean(), 4)
    }


def main():
    print("Comparing models...")

    if_df = pd.read_csv("data/if_scores.csv")
    lof_df = pd.read_csv("data/lof_scores.csv")
    lstm_df = pd.read_csv("data/lstm_scores.csv")

    # merge all scores by user_id
    merged = if_df[["user_id", "is_anomaly"]].copy()
    merged["if_score"] = if_df["anomaly_score"]
    merged["lof_score"] = lof_df.set_index("user_id").loc[merged["user_id"]]["anomaly_score"].values
    merged["lstm_score"] = lstm_df.set_index("user_id").loc[merged["user_id"]]["anomaly_score"].values

    # ensemble = simple average of all three
    merged["ensemble_score"] = (merged["if_score"] + merged["lof_score"] + merged["lstm_score"]) / 3

    y_true = merged["is_anomaly"]

    results = [
        evaluate(y_true, merged["if_score"], "Isolation Forest"),
        evaluate(y_true, merged["lof_score"], "Local Outlier Factor"),
        evaluate(y_true, merged["lstm_score"], "LSTM Autoencoder"),
        evaluate(y_true, merged["ensemble_score"], "Ensemble (Average)"),
    ]

    comparison_df = pd.DataFrame(results)
    print("\n" + comparison_df.to_string(index=False))

    best_idx = comparison_df["F1 Score"].idxmax()
    best = comparison_df.loc[best_idx]
    print(f"\nBest model: {best['Model']} (F1={best['F1 Score']}, AUC={best['ROC-AUC']})")

    comparison_df.to_csv("data/model_comparison.csv", index=False)
    merged.to_csv("data/all_scores.csv", index=False)
    print("Saved to data/model_comparison.csv and data/all_scores.csv")


if __name__ == "__main__":
    main()
