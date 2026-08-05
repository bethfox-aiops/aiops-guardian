#!/usr/bin/env python3
"""
retrain_recent.py

Retrain the autoencoder on only the most recent N rows of metrics.csv
so the current system state is the baseline rather than a minority
in the full historical dataset.

Behavioral Attestation Phase 3: instrumented with OpenTelemetry so a
retrain run is a traceable workflow (load_data -> train_model -> save_model)
rather than just isolated log lines. Traces export to the local Tempo
instance.

Shared scaffolding (GPU init, self-attribution monitoring, verify_behavior)
lives in retrain_common.py; this file has the autoencoder-specific data
loading, keras model, and save logic.
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from otel_setup import get_tracer
from feature_transform import transform_bursty_features_df
from retrain_common import (
    DATA_FILE,
    FEATURES,
    archive_current_models,
    init_gpu_handle,
    monitor_self,
    report_self_attribution,
    run_verification,
)

_gpu_handle = init_gpu_handle()

tracer = get_tracer("aiops-retrain-autoencoder")

MODEL_FILE  = "autoencoder_model.keras"
SCALER_FILE = "autoencoder_scaler.pkl"
THRESHOLD_FILE = "autoencoder_threshold.txt"
RECENT_ROWS = 2000

if __name__ == "__main__":
    run_start_time = time.time()

    with tracer.start_as_current_span("retrain_autoencoder_run") as run_span:
        run_span.set_attribute("recent_rows_requested", RECENT_ROWS)
        trace_id = format(run_span.get_span_context().trace_id, "032x")
        print(f"[INFO] Trace ID: {trace_id}")

        with tracer.start_as_current_span("load_data") as span:
            df = pd.read_csv(DATA_FILE)
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            df = df.tail(RECENT_ROWS).reset_index(drop=True)
            print(f"[INFO] Training on most recent {len(df)} rows.")

            X = transform_bursty_features_df(df[FEATURES])
            print(f"[INFO] Feature stats (net_kbps/disk_w_kbps log1p-transformed):\n{X.describe().loc[['mean','std','min','max']].T.to_string()}")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            span.set_attribute("rows_loaded", len(df))

        with tracer.start_as_current_span("train_model") as span:
            input_dim = X_scaled.shape[1]
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(8, activation="relu"),
                layers.Dense(4, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(input_dim, activation="linear"),
            ])
            model.compile(optimizer="adam", loss="mse")

            split = int(len(X_scaled) * 0.9)
            X_train, X_val = X_scaled[:split], X_scaled[split:]

            with monitor_self(_gpu_handle) as stats:
                model.fit(X_train, X_train, epochs=50, batch_size=32,
                          validation_data=(X_val, X_val), verbose=1)

            evidence = report_self_attribution(span, stats)

            val_recon = model.predict(X_val, verbose=0)
            val_mse = np.mean(np.square(X_val - val_recon), axis=1)
            threshold = float(np.percentile(val_mse, 99))

            print(f"[INFO] Val MSE — mean: {val_mse.mean():.4f}, 99th pct (threshold): {threshold:.6f}")
            span.set_attribute("training_rows", split)
            span.set_attribute("validation_rows", len(X_val))
            span.set_attribute("val_mse_mean", float(val_mse.mean()))
            span.set_attribute("threshold", threshold)

        with tracer.start_as_current_span("save_model") as span:
            archive_dir = archive_current_models([MODEL_FILE, SCALER_FILE, THRESHOLD_FILE], "autoencoder")
            if archive_dir:
                print(f"[INFO] Archived previous model to {archive_dir}")
                span.set_attribute("archived_to", archive_dir)

            model.save(MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            with open(THRESHOLD_FILE, "w") as f:
                f.write(str(threshold))
            span.set_attribute("model_file", MODEL_FILE)
            span.set_attribute("scaler_file", SCALER_FILE)
            span.set_attribute("threshold_file", THRESHOLD_FILE)

        with tracer.start_as_current_span("verify_behavior") as span:
            run_verification(
                span, "retrain_autoencoder",
                output_files=(MODEL_FILE, SCALER_FILE, THRESHOLD_FILE),
                run_start_time=run_start_time,
                gpu_mib=stats["gpu_max_mib"],
                evidence=evidence,
                row_count=len(df),
            )

        print("[INFO] Done. Restart aiops-watchdog-autoencoder to load new model.")
