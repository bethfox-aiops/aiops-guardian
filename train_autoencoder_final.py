#!/usr/bin/env python3
"""
train_autoencoder_final.py

Train a simple dense autoencoder on the same 11-feature dataset used by the
current watchdog scripts, then save:
- autoencoder_model.keras
- autoencoder_scaler.pkl
- autoencoder_threshold.txt
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

DATA_FILE = "aiops_data/metrics.csv"
MODEL_FILE = "autoencoder_model.keras"
SCALER_FILE = "autoencoder_scaler.pkl"
THRESHOLD_FILE = "autoencoder_threshold.txt"

features = [
    "disk",
    "disk_free_gb",
    "disk_fill_rate_mb_min",
    "inode_pct",
    "cpu",
    "mem",
    "net_kbps",
    "disk_w_kbps",
    "gpu_util",
    "gpu_mem_mib",
    "gpu_temp_c",
]

if __name__ == "__main__":
    print("[INFO] Loading collected data...")
    df = pd.read_csv(DATA_FILE)

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    print(f"[INFO] Loaded {len(df)} samples.")
    print(f"[INFO] Training on columns: {features}")

    X = df[features].copy()

    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    print("[INFO] Building autoencoder model...")
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(input_dim, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse")

    print("[INFO] Training autoencoder...")
    history = model.fit(
        X_scaled,
        X_scaled,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    print("[INFO] Calculating reconstruction threshold...")
    recon = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - recon), axis=1)
    threshold = float(np.percentile(mse, 95))

    print(f"[INFO] Threshold set to 95th percentile MSE: {threshold:.6f}")

    print(f"[INFO] Saving model to: {MODEL_FILE}")
    model.save(MODEL_FILE)

    print(f"[INFO] Saving scaler to: {SCALER_FILE}")
    joblib.dump(scaler, SCALER_FILE)

    print(f"[INFO] Saving threshold to: {THRESHOLD_FILE}")
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))

    print("[INFO] Done.")
