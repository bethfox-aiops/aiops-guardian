#!/usr/bin/env python3
"""
retrain_recent_knn.py

Retrain the KNN anomaly detector on the most recent N rows of metrics.csv
so the current system state is the baseline.

Behavioral Attestation Phase 3: instrumented with OpenTelemetry so a
retrain run is a traceable workflow (load_data -> train_model -> save_model)
rather than just isolated log lines. Traces export to the local Tempo
instance.

Shared scaffolding (GPU init, self-attribution monitoring, verify_behavior)
lives in retrain_common.py; this file has the KNN-specific data loading,
model, and save logic.
"""

import time

import joblib
import pandas as pd
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler

from otel_setup import get_tracer
from grafana_annotate import post_annotation
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

tracer = get_tracer("aiops-retrain-knn")

MODEL_FILE = "knn_model.pkl"
SCALER_FILE = "scaler.pkl"
RECENT_ROWS = 2000

if __name__ == "__main__":
    run_start_time = time.time()

    with tracer.start_as_current_span("retrain_knn_run") as run_span:
        run_span.set_attribute("recent_rows_requested", RECENT_ROWS)
        trace_id = format(run_span.get_span_context().trace_id, "032x")
        print(f"[INFO] Trace ID: {trace_id}")

        with tracer.start_as_current_span("load_data") as span:
            df = pd.read_csv(DATA_FILE).dropna()
            df = df.tail(RECENT_ROWS).reset_index(drop=True)
            print(f"[INFO] Training on most recent {len(df)} rows.")

            X = df[FEATURES].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            span.set_attribute("rows_loaded", len(df))

        with tracer.start_as_current_span("train_model") as span:
            print("[INFO] Training KNN anomaly detector...")

            with monitor_self(_gpu_handle) as stats:
                model = KNN(n_neighbors=5, method="largest", contamination=0.05)
                model.fit(X_scaled)

            evidence = report_self_attribution(span, stats)

            num_anomalies = int((model.labels_ == 1).sum())
            print(f"[INFO] Anomalies flagged in training data: {num_anomalies} / {len(df)}")
            span.set_attribute("training_rows", len(df))
            span.set_attribute("anomalies_in_training_data", num_anomalies)

        with tracer.start_as_current_span("save_model") as span:
            archive_dir = archive_current_models([MODEL_FILE, SCALER_FILE], "knn")
            if archive_dir:
                print(f"[INFO] Archived previous model to {archive_dir}")
                span.set_attribute("archived_to", archive_dir)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            print(f"[INFO] Saved {MODEL_FILE}, {SCALER_FILE}")
            span.set_attribute("model_file", MODEL_FILE)
            span.set_attribute("scaler_file", SCALER_FILE)

        with tracer.start_as_current_span("verify_behavior") as span:
            def _annotate_failure(violations):
                post_annotation(
                    f"retrain_knn policy violation: {'; '.join(violations)}",
                    tags=["guardian", "policy-violation", "retrain_knn"],
                )

            run_verification(
                span, "retrain_knn",
                output_files=(MODEL_FILE, SCALER_FILE),
                run_start_time=run_start_time,
                gpu_mib=stats["gpu_max_mib"],
                evidence=evidence,
                row_count=len(df),
                on_fail=_annotate_failure,
            )

        print("[INFO] Done. Restart aiops-watchdog-knn to load new model.")
