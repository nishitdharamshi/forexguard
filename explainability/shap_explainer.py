"""
SHAP explainability for the Isolation Forest model.
For each suspicious user (score > 0.6), extracts the top 3 features
that drove their anomaly score. This matters because compliance teams
need to know WHY a user was flagged, not just that they were.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("Running SHAP explainability analysis...")

    # load model and data
    bundle = joblib.load("models/saved/isolation_forest.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv("data/features.csv")
    user_ids = df["user_id"]
    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    scores_df = pd.read_csv("data/if_scores.csv")

    # compute SHAP values using TreeExplainer (optimized for tree models)
    print("Computing SHAP values (this takes a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    print(f"  shape: {shap_values.shape}")

    # for each suspicious user, find top 3 contributing features
    threshold = 0.6
    suspicious = scores_df[scores_df["anomaly_score"] > threshold]
    print(f"  {len(suspicious)} users above threshold {threshold}")

    explanations = []
    for idx in suspicious.index:
        user_id = user_ids.iloc[idx]
        score = scores_df.iloc[idx]["anomaly_score"]
        abs_shap = np.abs(shap_values[idx])
        top_3 = np.argsort(abs_shap)[-3:][::-1]

        row = {"user_id": user_id, "anomaly_score": round(score, 4)}
        for rank, feat_idx in enumerate(top_3, 1):
            row[f"feature_{rank}"] = feature_cols[feat_idx]
            row[f"value_{rank}"] = round(float(X.iloc[idx, feat_idx]), 4)
            row[f"shap_{rank}"] = round(float(abs_shap[feat_idx]), 4)
        explanations.append(row)

    explanations_df = pd.DataFrame(explanations)
    explanations_df.to_csv("data/shap_explanations.csv", index=False)

    # summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features=X_scaled, feature_names=feature_cols,
                      show=False, plot_size=(12, 8))
    plt.title("SHAP Feature Importance -- Isolation Forest")
    plt.tight_layout()
    plt.savefig("data/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # print a few examples
    print(f"\nTop 5 most suspicious users:")
    for _, r in explanations_df.nlargest(5, "anomaly_score").iterrows():
        print(f"  {r['user_id']} (score={r['anomaly_score']})")
        print(f"    1. {r['feature_1']} = {r['value_1']}")
        print(f"    2. {r['feature_2']} = {r['value_2']}")
        print(f"    3. {r['feature_3']} = {r['value_3']}")

    print(f"\nSaved {len(explanations_df)} explanations to data/shap_explanations.csv")
    print("Saved summary plot to data/shap_summary.png")


if __name__ == "__main__":
    main()
