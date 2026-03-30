"""
LSTM Autoencoder for anomaly detection.
Trained only on normal users -- the idea is that anomalous users
will have high reconstruction error because the model never learned
their patterns.

Note: we use seq_length=1 (aggregated features per user, not actual
time-series). With real event-level sequences this would be more powerful,
but this still works well as a reconstruction-based detector.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)
        return x, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, output_dim=None):
        super().__init__()
        self.lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, output_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_and_prepare_data(path="data/features.csv"):
    df = pd.read_csv(path)
    user_ids = df["user_id"]
    y_true = df["is_anomaly"]
    feature_cols = [c for c in df.columns if c not in ["user_id", "is_anomaly"]]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train only on normal users
    normal_mask = y_true == 0
    X_normal = X_scaled[normal_mask]

    print(f"Total: {len(df)}, Normal (training): {normal_mask.sum()}, Anomaly (targets): {(~normal_mask).sum()}")
    return X_scaled, X_normal, y_true, user_ids, scaler, feature_cols


def train_model(model, X_train, epochs=50, batch_size=64, lr=0.001):
    X_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_target in dataloader:
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} -- Loss: {avg_loss:.6f}")

    return losses


def compute_reconstruction_errors(model, X_all):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_all.reshape(X_all.shape[0], 1, X_all.shape[1]))
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=[1, 2])
    return errors.numpy()


def normalize_scores(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def main():
    print("Training LSTM Autoencoder...")
    X_all, X_normal, y_true, user_ids, scaler, feature_cols = load_and_prepare_data()

    input_dim = X_all.shape[1]
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=32, latent_dim=16)

    losses = train_model(model, X_normal, epochs=50, batch_size=64, lr=0.001)

    # plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.title("LSTM Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/lstm_training_loss.png", dpi=150)
    plt.close()

    # score all users
    print("Computing reconstruction errors...")
    errors = compute_reconstruction_errors(model, X_all)
    anomaly_scores = normalize_scores(errors)

    y_pred = (anomaly_scores >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, anomaly_scores)

    scores_df = pd.DataFrame({
        "user_id": user_ids, "anomaly_score": anomaly_scores,
        "reconstruction_error": errors, "is_anomaly": y_true
    })
    scores_df.to_csv("data/lstm_scores.csv", index=False)

    os.makedirs("models/saved", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim, "hidden_dim": 32, "latent_dim": 16,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols
    }, "models/saved/lstm_autoencoder.pt")

    print(f"\nResults:")
    print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
    print(f"  Avg error (normal):  {errors[y_true==0].mean():.6f}")
    print(f"  Avg error (anomaly): {errors[y_true==1].mean():.6f}")
    print("Saved to data/lstm_scores.csv and models/saved/lstm_autoencoder.pt")


if __name__ == "__main__":
    main()
