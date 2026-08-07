#!/usr/bin/env python3
"""
aiops-watchdog-knn.py

Real-time AIOps watchdog agent using a multi-metric + GPU KNN anomaly detector.

- Loads:
    knn_model.pkl
    scaler.pkl

- Collects metrics every INTERVAL seconds:
    disk, cpu, mem,
    net_kbps, disk_w_kbps,
    gpu_util, gpu_mem_mib, gpu_temp_c

- Exposes Prometheus metrics on WATCHDOG_PORT (default: 8011):
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
    - train_knn_final.py    (trainer)

Shared collection/GPU/serving logic lives in watchdog_common.py; this file
only supplies the KNN-specific model loading, feature-matrix construction,
and scoring.
"""

import numpy as np
import joblib

import watchdog_common as common
from watchdog_common import WatchdogConfig

MODEL_FILE = "knn_model.pkl"
SCALER_FILE = "scaler.pkl"
TRAINER = "retrain_recent_knn.py"

PORT = "8011"


def load_model():
    common.require_file(MODEL_FILE, TRAINER)
    common.require_file(SCALER_FILE, TRAINER)

    print(f"[INFO] Loading model from {MODEL_FILE}", flush=True)
    model = joblib.load(MODEL_FILE)

    print(f"[INFO] Loading scaler from {SCALER_FILE}", flush=True)
    scaler = joblib.load(SCALER_FILE)

    return {"model": model, "scaler": scaler}


def build_input(features, columns):
    return np.array([features])


def score(state, X_scaled):
    # PyOD KNN: predict() -> label, decision_function() -> scores
    labels = state["model"].predict(X_scaled)  # 0 = normal, 1 = anomaly
    scores = state["model"].decision_function(X_scaled)
    return int(labels[0]), float(scores[0])


if __name__ == "__main__":
    common.run(WatchdogConfig(
        model_name="KNN",
        default_port=PORT,
        load_model=load_model,
        build_input=build_input,
        score=score,
    ))
