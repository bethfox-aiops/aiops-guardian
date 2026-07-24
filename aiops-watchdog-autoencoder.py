#!/usr/bin/env python3
"""
aiops-watchdog-autoencoder.py

Real-time AIOps watchdog agent using a multi-metric + GPU AUTOENCODER anomaly detector.

- Loads:
    autoencoder_model.keras
    autoencoder_scaler.pkl

- Collects metrics every INTERVAL seconds:
    disk, cpu, mem,
    net_kbps, disk_w_kbps,
    gpu_util, gpu_mem_mib, gpu_temp_c

- Exposes Prometheus metrics on WATCHDOG_PORT (default: 8013):
    aiops_disk_usage_percent
    aiops_cpu_usage_percent
    aiops_mem_usage_percent
    aiops_net_kbps
    aiops_disk_write_kbps
    aiops_gpu_util_percent
    aiops_gpu_mem_mib
    aiops_gpu_temp_c

    aiops_anomaly_label        (0 = normal, 1 = anomaly)
    aiops_anomaly_score        (higher = more normal, lower = more anomalous)

    disk_anomaly_prediction    (legacy name, mirrors aiops_anomaly_label)

This is the live counterpart to:
    - aiops-watchdog-ml.py  (collector)
    - train_autoencoder_final.py    (trainer)

Shared collection/GPU/serving logic lives in watchdog_common.py; this file
only supplies the AUTOENCODER-specific model loading, feature-matrix
construction, and scoring (reconstruction error vs. threshold).
"""

import joblib
import pandas as pd
import tensorflow as tf
import numpy as np

import watchdog_common as common
from watchdog_common import WatchdogConfig

MODEL_FILE = "autoencoder_model.keras"
SCALER_FILE = "autoencoder_scaler.pkl"
THRESHOLD_FILE = "autoencoder_threshold.txt"
TRAINER = "train_autoencoder_final.py"

PORT = "8013"


def load_model():
    common.require_file(MODEL_FILE, TRAINER)
    common.require_file(SCALER_FILE, TRAINER)
    common.require_file(THRESHOLD_FILE, TRAINER)

    print(f"[INFO] Loading model from {MODEL_FILE}", flush=True)
    model = tf.keras.models.load_model(MODEL_FILE)

    print(f"[INFO] Loading scaler from {SCALER_FILE}", flush=True)
    scaler = joblib.load(SCALER_FILE)

    print(f"[INFO] Loading threshold from {THRESHOLD_FILE}", flush=True)
    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())

    return {"model": model, "scaler": scaler, "threshold": threshold}


def build_input(features, columns):
    return pd.DataFrame([features], columns=columns)


def score(state, X_scaled):
    recon = state["model"].predict(X_scaled, verbose=0)
    scores = np.mean(np.square(X_scaled - recon), axis=1)
    label = int(scores[0] > state["threshold"])
    return label, float(scores[0])


if __name__ == "__main__":
    common.run(WatchdogConfig(
        model_name="AUTOENCODER",
        default_port=PORT,
        load_model=load_model,
        build_input=build_input,
        score=score,
    ))
