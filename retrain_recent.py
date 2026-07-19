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
"""

import os
import threading
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from otel_setup import get_tracer
from ebpf_trace import trace_suspect_process
from gpu_attribution import poll_max_gpu_usage
from behavioral_policy import verify

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
    nvmlInit()
    _gpu_handle = nvmlDeviceGetHandleByIndex(0)
except Exception:
    _gpu_handle = None

tracer = get_tracer("aiops-retrain-autoencoder")

DATA_FILE  = "aiops_data/metrics.csv"
MODEL_FILE = "autoencoder_model.keras"
SCALER_FILE = "autoencoder_scaler.pkl"
THRESHOLD_FILE = "autoencoder_threshold.txt"
RECENT_ROWS = 100000

features = [
    "disk", "disk_free_gb", "disk_fill_rate_mb_min", "inode_pct",
    "cpu", "mem", "net_kbps", "disk_w_kbps",
    "gpu_util", "gpu_mem_mib", "gpu_temp_c",
]

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

            X = df[features].copy()
            print(f"[INFO] Feature stats:\n{X.describe().loc[['mean','std','min','max']].T.to_string()}")

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

            # Behavioral Attestation: link this trace to Phase 1/2 evidence
            # by capturing process- and syscall-level evidence of this
            # training step itself, using the same tooling the watchdogs
            # use on suspect processes -- just pointed at our own PID. Note
            # the eBPF trace window is a fixed 3s (by design, see Phase 2),
            # so for a multi-epoch fit like this it's a representative
            # sample from the start of training, not full coverage.
            self_pid = os.getpid()
            self_proc = psutil.Process(self_pid)
            self_proc.cpu_percent(interval=None)  # prime

            ebpf_result = {}
            trace_thread = threading.Thread(
                target=lambda: ebpf_result.update(evidence=trace_suspect_process(self_pid))
            )
            trace_thread.start()

            # Unlike the fixed-duration eBPF trace (capped at 3s by design,
            # see Phase 2), GPU polling is cheap and unprivileged, so it
            # covers the *actual* training duration via stop_event rather
            # than guessing a fixed window -- this matters most here, since
            # a 50-epoch fit runs far longer than 3 seconds.
            gpu_stop = threading.Event()
            gpu_result = {}
            gpu_thread = threading.Thread(
                target=lambda: gpu_result.update(
                    max_mib=poll_max_gpu_usage(self_pid, _gpu_handle, duration=600, interval=0.5, stop_event=gpu_stop)
                )
            )
            gpu_thread.start()

            model.fit(X_train, X_train, epochs=50, batch_size=32,
                      validation_data=(X_val, X_val), verbose=1)

            gpu_stop.set()
            trace_thread.join(timeout=5)
            gpu_thread.join(timeout=5)
            span.set_attribute("process.cpu_percent", self_proc.cpu_percent(interval=None))
            span.set_attribute("process.mem_percent", self_proc.memory_percent())
            span.set_attribute("gpu.used_memory_mib", gpu_result.get("max_mib", 0.0))

            evidence = ebpf_result.get("evidence")
            if evidence:
                for syscall_type, count in evidence["counts"].items():
                    span.set_attribute(f"ebpf.syscall.{syscall_type.lower()}", count)
                if evidence["files_opened"]:
                    span.set_attribute("ebpf.files_opened", evidence["files_opened"])
                print(f"[INFO] eBPF evidence during training: {evidence['counts']}")

            val_recon = model.predict(X_val, verbose=0)
            val_mse = np.mean(np.square(X_val - val_recon), axis=1)
            threshold = float(np.percentile(val_mse, 95))

            print(f"[INFO] Val MSE — mean: {val_mse.mean():.4f}, 95th pct (threshold): {threshold:.6f}")
            span.set_attribute("training_rows", split)
            span.set_attribute("validation_rows", len(X_val))
            span.set_attribute("val_mse_mean", float(val_mse.mean()))
            span.set_attribute("threshold", threshold)

        with tracer.start_as_current_span("save_model") as span:
            model.save(MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            with open(THRESHOLD_FILE, "w") as f:
                f.write(str(threshold))
            span.set_attribute("model_file", MODEL_FILE)
            span.set_attribute("scaler_file", SCALER_FILE)
            span.set_attribute("threshold_file", THRESHOLD_FILE)

        with tracer.start_as_current_span("verify_behavior") as span:
            files_touched = [
                f for f in (MODEL_FILE, SCALER_FILE, THRESHOLD_FILE)
                if os.path.exists(f) and os.path.getmtime(f) >= run_start_time
            ]
            result = verify(
                "retrain_autoencoder",
                files_touched=files_touched,
                gpu_mib=gpu_result.get("max_mib", 0.0),
                network_connects=(evidence["counts"]["CONNECT"] if evidence else 0),
                row_count=len(df),
            )
            span.set_attribute("verification.passed", result["passed"])
            span.set_attribute("verification.violations", result["violations"])
            if result["passed"]:
                print("[VERIFY] PASS: this run matched its behavioral policy.")
            else:
                print("[VERIFY] FAIL: this run violated its behavioral policy:")
                for v in result["violations"]:
                    print(f"  - {v}")

        print("[INFO] Done. Restart aiops-watchdog-autoencoder to load new model.")
