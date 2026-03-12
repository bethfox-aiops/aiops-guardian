#!/usr/bin/env python3
"""
train_knn_final.py

Trains a multi-metric + GPU KNN anomaly detection model.

Inputs:
    aiops_data/metrics.csv

Outputs:
    knn_model.pkl
    scaler.pkl

Model:
    PyOD's KNN anomaly detector using 'largest' k-distance scoring
    (more stable than 'distance' method for datasets with uneven variance)

Metrics used:
    disk, cpu, mem,
    net_kbps, disk_w_kbps,
    gpu_util, gpu_mem_mib, gpu_temp_c
"""

import os
import pandas as pd
import joblib
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler

DATA_FILE = "aiops_data/metrics.csv"
MODEL_FILE = "knn_model.pkl"
SCALER_FILE = "scaler.pkl"

FEATURE_COLS = [
    "disk",
    "cpu",
    "mem",
    "net_kbps",
    "disk_w_kbps",
    "gpu_util",
    "gpu_mem_mib",
    "gpu_temp_c"
]

print("[INFO] Loading collected data...")

# Load and validate CSV
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"[ERROR] Cannot find {DATA_FILE}. "
        "Run aiops-watchdog-ml.py --train first to generate training data."
    )

df = pd.read_csv(DATA_FILE)

# Drop rows that have any missing values
df = df.dropna()
if df.empty:
    raise ValueError("[ERROR] Dataset is empty after dropping NaN rows.")

print(f"[INFO] Loaded {len(df)} samples.")
print("[INFO] Training on columns:", FEATURE_COLS)

# Extract features
X = df[FEATURE_COLS].values

# Scale features
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
print("[INFO] Training KNN anomaly detector...")
model = KNN(n_neighbors=5, method="largest")
model.fit(X_scaled)

# Save model + scaler
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print("\n[INFO] Training complete.")
print(f"[INFO] Saved model to: {MODEL_FILE}")
print(f"[INFO] Saved scaler to: {SCALER_FILE}")

# Print useful stats
labels = model.labels_
num_anomalies = (labels == 1).sum()
print(f"[INFO] Number of anomalies in training data: {num_anomalies}")
print("[INFO] Done.\n")
