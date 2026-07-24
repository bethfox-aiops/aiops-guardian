#!/usr/bin/env python3
"""
retrain_recent_knn.py

Retrain the KNN anomaly detector on the most recent N rows of metrics.csv
so the current system state is the baseline.

Behavioral Attestation Phase 3: instrumented with OpenTelemetry so a
retrain run is a traceable workflow (load_data -> train_model -> save_model)
rather than just isolated log lines. Traces export to the local Tempo
instance.
"""

import os
import threading
import time

import joblib
import pandas as pd
import psutil
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler

from otel_setup import get_tracer
from ebpf_trace import trace_suspect_process
from gpu_attribution import poll_max_gpu_usage
from behavioral_policy import verify
from grafana_annotate import post_annotation

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
    nvmlInit()
    _gpu_handle = nvmlDeviceGetHandleByIndex(0)
except Exception:
    _gpu_handle = None

tracer = get_tracer("aiops-retrain-knn")

DATA_FILE  = "aiops_data/metrics.csv"
MODEL_FILE = "knn_model.pkl"
SCALER_FILE = "scaler.pkl"
RECENT_ROWS = 2000

features = [
    "disk", "disk_free_gb", "disk_fill_rate_mb_min", "inode_pct",
    "cpu", "mem", "net_kbps", "disk_w_kbps",
    "gpu_util", "gpu_mem_mib", "gpu_temp_c",
]

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

            X = df[features].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            span.set_attribute("rows_loaded", len(df))

        with tracer.start_as_current_span("train_model") as span:
            print("[INFO] Training KNN anomaly detector...")

            # Behavioral Attestation: link this trace to Phase 1/2 evidence
            # by capturing process- and syscall-level evidence of this
            # training step itself, using the same tooling the watchdogs
            # use on suspect processes -- just pointed at our own PID.
            self_pid = os.getpid()
            self_proc = psutil.Process(self_pid)
            self_proc.cpu_percent(interval=None)  # prime

            ebpf_result = {}
            trace_thread = threading.Thread(
                target=lambda: ebpf_result.update(evidence=trace_suspect_process(self_pid))
            )
            trace_thread.start()

            gpu_stop = threading.Event()
            gpu_result = {}
            gpu_thread = threading.Thread(
                target=lambda: gpu_result.update(
                    max_mib=poll_max_gpu_usage(self_pid, _gpu_handle, duration=600, interval=0.5, stop_event=gpu_stop)
                )
            )
            gpu_thread.start()

            model = KNN(n_neighbors=5, method="largest", contamination=0.05)
            model.fit(X_scaled)

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

            num_anomalies = int((model.labels_ == 1).sum())
            print(f"[INFO] Anomalies flagged in training data: {num_anomalies} / {len(df)}")
            span.set_attribute("training_rows", len(df))
            span.set_attribute("anomalies_in_training_data", num_anomalies)

        with tracer.start_as_current_span("save_model") as span:
            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            print(f"[INFO] Saved {MODEL_FILE}, {SCALER_FILE}")
            span.set_attribute("model_file", MODEL_FILE)
            span.set_attribute("scaler_file", SCALER_FILE)

        with tracer.start_as_current_span("verify_behavior") as span:
            files_touched = [
                f for f in (MODEL_FILE, SCALER_FILE)
                if os.path.exists(f) and os.path.getmtime(f) >= run_start_time
            ]
            result = verify(
                "retrain_knn",
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
                post_annotation(
                    f"retrain_knn policy violation: {'; '.join(result['violations'])}",
                    tags=["guardian", "policy-violation", "retrain_knn"],
                )

        print("[INFO] Done. Restart aiops-watchdog-knn to load new model.")
