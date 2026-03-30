"""
MLflow experiment tracking.
Logs all models with their params, metrics, and artifacts so
experiments are reproducible and comparable.
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        print(f"Connected to MLflow at {tracking_uri}")
    except Exception:
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)
        print(f"MLflow server not available, using local: {local_uri}")

    mlflow.set_experiment("ForexGuard-AnomalyDetection")


def _load_metrics(model_name):
    path = "data/model_comparison.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["Model"] == model_name]
    return row.iloc[0].to_dict() if len(row) > 0 else None


def _get_feature_count():
    path = "data/features.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=1)
        return len([c for c in df.columns if c not in ["user_id", "is_anomaly"]])
    return 0


def log_isolation_forest():
    print("Logging Isolation Forest...")
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.set_tag("model_type", "isolation_forest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("contamination", 0.2)
        mlflow.log_param("random_state", 42)

        metrics = _load_metrics("Isolation Forest")
        if metrics:
            mlflow.log_param("feature_count", _get_feature_count())
            mlflow.log_metric("precision", metrics["Precision"])
            mlflow.log_metric("recall", metrics["Recall"])
            mlflow.log_metric("f1_score", metrics["F1 Score"])
            mlflow.log_metric("roc_auc", metrics["ROC-AUC"])

        if os.path.exists("models/saved/isolation_forest.pkl"):
            mlflow.log_artifact("models/saved/isolation_forest.pkl")
    print("  done")


def log_lof():
    print("Logging LOF...")
    with mlflow.start_run(run_name="local_outlier_factor"):
        mlflow.set_tag("model_type", "lof")
        mlflow.log_param("n_neighbors", 20)
        mlflow.log_param("contamination", 0.2)

        metrics = _load_metrics("Local Outlier Factor")
        if metrics:
            mlflow.log_param("feature_count", _get_feature_count())
            mlflow.log_metric("precision", metrics["Precision"])
            mlflow.log_metric("recall", metrics["Recall"])
            mlflow.log_metric("f1_score", metrics["F1 Score"])
            mlflow.log_metric("roc_auc", metrics["ROC-AUC"])

        if os.path.exists("models/saved/lof.pkl"):
            mlflow.log_artifact("models/saved/lof.pkl")
    print("  done")


def log_lstm():
    print("Logging LSTM Autoencoder...")
    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.set_tag("model_type", "lstm_autoencoder")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "MSE")

        metrics = _load_metrics("LSTM Autoencoder")
        if metrics:
            mlflow.log_param("feature_count", _get_feature_count())
            mlflow.log_metric("precision", metrics["Precision"])
            mlflow.log_metric("recall", metrics["Recall"])
            mlflow.log_metric("f1_score", metrics["F1 Score"])
            mlflow.log_metric("roc_auc", metrics["ROC-AUC"])

        if os.path.exists("models/saved/lstm_autoencoder.pt"):
            mlflow.log_artifact("models/saved/lstm_autoencoder.pt")
        if os.path.exists("data/lstm_training_loss.png"):
            mlflow.log_artifact("data/lstm_training_loss.png")
    print("  done")


def log_ensemble():
    print("Logging Ensemble...")
    with mlflow.start_run(run_name="ensemble_average"):
        mlflow.set_tag("model_type", "ensemble")
        mlflow.log_param("method", "simple_average")
        mlflow.log_param("models_combined", "IF + LOF + LSTM")

        metrics = _load_metrics("Ensemble (Average)")
        if metrics:
            mlflow.log_metric("precision", metrics["Precision"])
            mlflow.log_metric("recall", metrics["Recall"])
            mlflow.log_metric("f1_score", metrics["F1 Score"])
            mlflow.log_metric("roc_auc", metrics["ROC-AUC"])
    print("  done")


def main():
    if not MLFLOW_AVAILABLE:
        print("MLflow not installed. Run: pip install mlflow")
        return

    print("Logging experiments to MLflow...")
    setup_mlflow()
    log_isolation_forest()
    log_lof()
    log_lstm()
    log_ensemble()
    print("\nAll models logged. View with: mlflow ui")


if __name__ == "__main__":
    main()
