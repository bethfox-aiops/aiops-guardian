#!/usr/bin/env python3
"""
retrain_recent_iforest.py

Retrain the Isolation Forest anomaly detector on the most recent N rows of metrics.csv
so the current system state is the baseline.

Behavioral Attestation Phase 3: instrumented with OpenTelemetry so a
retrain run is a traceable workflow (load_data -> train_model -> save_model)
rather than just isolated log lines. Traces export to the local Tempo
instance.

Shared scaffolding (GPU init, self-attribution monitoring, verify_behavior)
lives in retrain_common.py; this file has the IForest-specific data
loading, model, and save logic.
"""

import time

import joblib
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

from otel_setup import get_tracer
from feature_transform import transform_bursty_features_df
from retrain_common import (
    DATA_FILE,
    FEATURES,
    RECENT_ROWS,
    RECENT_WINDOW_HOURS,
    archive_current_models,
    init_gpu_handle,
    monitor_self,
    report_self_attribution,
    run_verification,
    select_recent_window,
)

_gpu_handle = init_gpu_handle()

tracer = get_tracer("aiops-retrain-iforest")

MODEL_FILE  = "iforest_model.pkl"
SCALER_FILE = "iforest_scaler.pkl"

if __name__ == "__main__":
    run_start_time = time.time()

    with tracer.start_as_current_span("retrain_iforest_run") as run_span:
        run_span.set_attribute("recent_rows_requested", RECENT_ROWS)
        trace_id = format(run_span.get_span_context().trace_id, "032x")
        print(f"[INFO] Trace ID: {trace_id}")

        with tracer.start_as_current_span("load_data") as span:
            df = pd.read_csv(DATA_FILE).dropna()
            df = select_recent_window(df)
            print(f"[INFO] Training on {len(df)} rows spanning the last {RECENT_WINDOW_HOURS}h.")

            X = transform_bursty_features_df(df[FEATURES]).values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            span.set_attribute("rows_loaded", len(df))

        with tracer.start_as_current_span("train_model") as span:
            print("[INFO] Training Isolation Forest anomaly detector...")

            with monitor_self(_gpu_handle) as stats:
                model = IForest(n_estimators=200, contamination=0.05, random_state=42)
                model.fit(X_scaled)

            evidence = report_self_attribution(span, stats)

            num_anomalies = int((model.labels_ == 1).sum())
            print(f"[INFO] Anomalies flagged in training data: {num_anomalies} / {len(df)}")
            span.set_attribute("training_rows", len(df))
            span.set_attribute("anomalies_in_training_data", num_anomalies)

        with tracer.start_as_current_span("save_model") as span:
            archive_dir = archive_current_models([MODEL_FILE, SCALER_FILE], "iforest")
            if archive_dir:
                print(f"[INFO] Archived previous model to {archive_dir}")
                span.set_attribute("archived_to", archive_dir)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            print(f"[INFO] Saved {MODEL_FILE}, {SCALER_FILE}")
            span.set_attribute("model_file", MODEL_FILE)
            span.set_attribute("scaler_file", SCALER_FILE)

        with tracer.start_as_current_span("verify_behavior") as span:
            run_verification(
                span, "retrain_iforest",
                output_files=(MODEL_FILE, SCALER_FILE),
                run_start_time=run_start_time,
                gpu_mib=stats["gpu_max_mib"],
                evidence=evidence,
                row_count=len(df),
            )

        print("[INFO] Done. Restart aiops-watchdog-iforest to load new model.")
