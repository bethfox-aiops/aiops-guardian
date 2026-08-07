#!/usr/bin/env python3
"""
aiops-watchdog-autoencoder.py

Real-time AIOps watchdog agent using a multi-metric + GPU AUTOENCODER anomaly detector.

- Loads:
    autoencoder_model.pkl
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

    aiops_anomaly_feature_reconstruction_error{feature="..."}
        Autoencoder-specific evidence, not part of the shared watchdog
        gauge set (KNN/iForest have no per-feature analogue): squared
        reconstruction error per input feature (in scaled/log1p-transformed
        space) at the last flagged anomaly, so an operator or the
        Behavioral Attestation pipeline can see *which* feature(s) drove
        the score instead of just the aggregate distance -- e.g.
        "gpu_temp_c: 8.2x normal" pointing straight at a driver outage,
        the same way Phase 1 attribution points at a process.

This is the live counterpart to:
    - aiops-watchdog-ml.py  (collector)
    - retrain_recent.py     (trainer)

Shared collection/GPU/serving logic lives in watchdog_common.py; this file
only supplies the AUTOENCODER-specific model loading, feature-matrix
construction, and scoring (reconstruction error vs. threshold).

Model (2026-08-07): PyOD's AutoEncoder (torch-backed) via joblib, same
library/serialization pattern as aiops-watchdog-knn.py -- not the earlier
hand-rolled Keras network. The threshold lives inside the pickled model
(set at fit time from contamination=0.01, see retrain_recent.py), not a
separate percentile computed here.
"""

import joblib
import numpy as np
import torch
from prometheus_client import Gauge

import watchdog_common as common
from watchdog_common import WatchdogConfig

MODEL_FILE = "autoencoder_model.pkl"
SCALER_FILE = "autoencoder_scaler.pkl"
TRAINER = "retrain_recent.py"

PORT = "8013"

feature_reconstruction_error = Gauge(
    "aiops_anomaly_feature_reconstruction_error",
    "Per-feature squared reconstruction error (scaled space) at the last flagged anomaly",
    ["feature"],
)


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


def _set_feature_evidence(model, X_scaled):
    """Reconstructs X_scaled through the model's underlying torch module
    directly (documented public attribute, not a private internal) to get
    per-feature squared error -- decision_function()/predict() only expose
    the aggregate row-wise distance PyOD itself uses for thresholding."""
    model.model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_scaled).float().to(model.device)
        recon = model.model(x).cpu().numpy()
    err = np.square(X_scaled - recon)[0]
    feature_reconstruction_error.clear()
    for name, e in zip(common.DATA_FEATURES, err):
        feature_reconstruction_error.labels(feature=name).set(float(e))


def score(state, X_scaled):
    # PyOD AutoEncoder: predict() -> label, decision_function() -> row-wise
    # reconstruction distance (higher = more anomalous, opposite sign
    # convention from KNN/iForest's decision_function but same as this
    # watchdog's score has always meant since the Keras version).
    labels = state["model"].predict(X_scaled)
    scores = state["model"].decision_function(X_scaled)
    label = int(labels[0])
    if label == 1:
        _set_feature_evidence(state["model"], X_scaled)
    return label, float(scores[0])


if __name__ == "__main__":
    common.run(WatchdogConfig(
        model_name="AUTOENCODER",
        default_port=PORT,
        load_model=load_model,
        build_input=build_input,
        score=score,
    ))
