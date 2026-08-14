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
loading, model, and save logic.

Model (2026-08-07): PyOD's AutoEncoder (torch-backed), not a hand-rolled
Keras network -- same library KNN/iForest already use, so the threshold is
set the same contamination-based way instead of a bespoke percentile-of-
validation-MSE computation. contamination=0.01 (not KNN/iForest's 0.05)
deliberately carries forward the 2026-08-03 lesson that this model's
normal-sample score variance is wide enough that a looser cutoff produces
constant flickering -- see retrain_common.py's RESUME_EXCLUSION_BUFFER_MINUTES
comment for the related resume-transient story. hidden_neuron_list=[8, 4]
matches the old architecture's capacity (8->4->8->10) rather than PyOD's
default [64, 32], which would be oversized for 10 features / 2000 rows.
"""

import time

import joblib
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler

from otel_setup import get_tracer
from feature_transform import transform_bursty_features_df
from retrain_common import (
    DATA_FILE,
    FEATURES,
    RECENT_ROWS,
    RESUME_EXCLUSION_BUFFER_MINUTES,
    archive_current_models,
    exclude_resume_transients,
    init_gpu_handle,
    monitor_self,
    report_self_attribution,
    run_verification,
    select_recent_window,
)

_gpu_handle = init_gpu_handle()

tracer = get_tracer("aiops-retrain-autoencoder")

MODEL_FILE  = "autoencoder_model.pkl"
SCALER_FILE = "autoencoder_scaler.pkl"
THRESHOLD_FILE = "autoencoder_threshold.txt"

if __name__ == "__main__":
    run_start_time = time.time()

    with tracer.start_as_current_span("retrain_autoencoder_run") as run_span:
        run_span.set_attribute("recent_rows_requested", RECENT_ROWS)
        trace_id = format(run_span.get_span_context().trace_id, "032x")
        print(f"[INFO] Trace ID: {trace_id}")

        with tracer.start_as_current_span("load_data") as span:
            df = pd.read_csv(DATA_FILE)

            # Wide-window selection (2026-08-14, see retrain_common.py) with
            # extra margin before filtering, so excluding resume-transient
            # rows still leaves enough to reach RECENT_ROWS afterward.
            df = select_recent_window(df, target_rows=RECENT_ROWS + 1000)
            rows_before_filter = len(df)
            df = exclude_resume_transients(df)
            excluded = rows_before_filter - len(df)
            if excluded:
                print(f"[INFO] Excluded {excluded} rows within {RESUME_EXCLUSION_BUFFER_MINUTES}min of a suspend/resume event.")

            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            df = df.tail(RECENT_ROWS).reset_index(drop=True)
            print(f"[INFO] Training on most recent {len(df)} rows.")

            X = transform_bursty_features_df(df[FEATURES])
            print(f"[INFO] Feature stats (net_kbps/disk_w_kbps log1p-transformed):\n{X.describe().loc[['mean','std','min','max']].T.to_string()}")

            scaler = StandardScaler()
            # .values: fit on a plain array, not a DataFrame -- the live
            # watchdog scores with plain numpy arrays (build_input() has no
            # column names to give it), and a scaler fit on a DataFrame
            # remembers feature names, so scoring later raises a spurious
            # "X does not have valid feature names" warning every cycle.
            X_scaled = scaler.fit_transform(X.values)
            span.set_attribute("rows_loaded", len(df))

        with tracer.start_as_current_span("train_model") as span:
            # preprocessing=False: scaling is handled externally (the fitted
            # StandardScaler above), same division of responsibility as
            # retrain_recent_knn.py -- avoids double-scaling.
            model = AutoEncoder(
                contamination=0.01,
                preprocessing=False,
                hidden_neuron_list=[8, 4],
                epoch_num=30,
                batch_size=32,
                verbose=0,
            )

            with monitor_self(_gpu_handle) as stats:
                model.fit(X_scaled)

            evidence = report_self_attribution(span, stats)

            num_anomalies = int((model.labels_ == 1).sum())
            print(f"[INFO] Anomalies flagged in training data: {num_anomalies} / {len(df)}")
            print(f"[INFO] decision_scores_ — mean: {model.decision_scores_.mean():.4f}, "
                  f"threshold (contamination=0.01): {model.threshold_:.6f}")
            span.set_attribute("training_rows", len(df))
            span.set_attribute("anomalies_in_training_data", num_anomalies)
            span.set_attribute("threshold", float(model.threshold_))

        with tracer.start_as_current_span("save_model") as span:
            archive_dir = archive_current_models([MODEL_FILE, SCALER_FILE, THRESHOLD_FILE], "autoencoder")
            if archive_dir:
                print(f"[INFO] Archived previous model to {archive_dir}")
                span.set_attribute("archived_to", archive_dir)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            # Not read by the live watchdog (the threshold lives inside the
            # pickled model itself, same as KNN/iForest) -- written purely
            # for human observability, same file name existing tooling
            # (behavioral_policy.py, guardian_common.py's integrity check)
            # already expects.
            with open(THRESHOLD_FILE, "w") as f:
                f.write(str(float(model.threshold_)))
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
