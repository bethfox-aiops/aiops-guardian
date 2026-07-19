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
"""

import os
import time
import sys
from datetime import datetime

import psutil
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np

from prometheus_client import Gauge, start_http_server

from process_attribution import get_top_processes
from ebpf_trace import trace_suspect_process
from gpu_attribution import get_gpu_usage_for_pid


def get_disk_extras(mountpoint: str, elapsed: float, prev_disk_used: int | None):
    """
    Returns: (disk_pct, disk_free_gb, disk_fill_rate_mb_min, inode_pct, new_prev_disk_used)
    """
    usage = psutil.disk_usage(mountpoint)
    disk_pct = usage.percent
    disk_free_gb = round(usage.free / (1024**3), 2)

    current_disk_used = usage.used
    disk_fill_rate_mb_min = 0.0
    if prev_disk_used is not None:
        delta_bytes = current_disk_used - prev_disk_used
        disk_fill_rate_mb_min = round(
            (delta_bytes / (1024**2)) * (60.0 / max(elapsed, 1e-6)),
            2
        )

    statvfs = os.statvfs(mountpoint)
    inode_pct = 0.0
    if statvfs.f_files > 0:
        inode_pct = round(
            ((statvfs.f_files - statvfs.f_ffree) / statvfs.f_files) * 100.0,
            2
        )

    return disk_pct, disk_free_gb, disk_fill_rate_mb_min, inode_pct, current_disk_used

# GPU / NVML support
try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetTemperature,
        NVML_TEMPERATURE_GPU,
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


# ---- Config ----

DATA_FEATURES = [
    "disk",
    "disk_free_gb",
    "disk_fill_rate_mb_min",
    "inode_pct",
    "cpu",
    "mem",
    "net_kbps",
    "disk_w_kbps",
    "gpu_util",
    "gpu_mem_mib",
    "gpu_temp_c",
]

MODEL_FILE = "autoencoder_model.keras"
SCALER_FILE = "autoencoder_scaler.pkl"

PORT = int(os.getenv("WATCHDOG_PORT", "8013"))
INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "5.0"))
GPU_INDEX = int(os.getenv("WATCHDOG_GPU_INDEX", "0"))


# ---- GPU Helpers ----

def init_gpu(gpu_index: int):
    if not NVML_AVAILABLE:
        print("[WARN] pynvml not installed. GPU metrics will be 0.", flush=True)
        return None

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        print(f"[INFO] Using GPU index {gpu_index} for watchdog metrics.", flush=True)
        return handle
    except Exception as e:
        print(f"[WARN] Failed to init NVML / GPU index {gpu_index}: {e}", flush=True)
        print("[WARN] GPU metrics will be recorded as 0.", flush=True)
        return None


def get_gpu_metrics(handle):
    """Return (util, mem_used_mib, temp_c). If no handle, return zeros."""
    if handle is None:
        return 0.0, 0.0, 0.0

    try:
        util = nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        mem_used_mib = mem_info.used / (1024 * 1024)
        temp_c = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        return float(util), float(mem_used_mib), float(temp_c)
    except Exception as e:
        print(f"[WARN] Error reading GPU metrics: {e}", flush=True)
        return 0.0, 0.0, 0.0


# ---- Prometheus Metric Definitions ----

aiops_disk_usage_percent = Gauge(
    "aiops_disk_usage_percent", "Disk usage percent of root filesystem"
)
aiops_disk_free_gb = Gauge(
    "aiops_disk_free_gb", "Free disk space on / in gigabytes"
)

aiops_disk_fill_rate_mb_min = Gauge(
    "aiops_disk_fill_rate_mb_min",
    "Disk usage fill rate on / in MB per minute (positive means filling)"
)

aiops_inode_usage_percent = Gauge(
    "aiops_inode_usage_percent",
    "Inode usage percent on /"
)

aiops_cpu_usage_percent = Gauge(
    "aiops_cpu_usage_percent", "CPU usage percent"
)
aiops_mem_usage_percent = Gauge(
    "aiops_mem_usage_percent", "Memory usage percent"
)
aiops_net_kbps = Gauge(
    "aiops_net_kbps", "Total network throughput (send+recv) in kB/s"
)
aiops_disk_write_kbps = Gauge(
    "aiops_disk_write_kbps", "Disk write throughput in kB/s"
)
aiops_gpu_util_percent = Gauge(
    "aiops_gpu_util_percent", "GPU utilization percent"
)
aiops_gpu_mem_mib = Gauge(
    "aiops_gpu_mem_mib", "GPU memory used in MiB"
)
aiops_gpu_temp_c = Gauge(
    "aiops_gpu_temp_c", "GPU temperature in Celsius"
)

aiops_anomaly_label = Gauge(
    "aiops_anomaly_label",
    "Anomaly label from AUTOENCODER (0=normal, 1=anomaly)",
)
aiops_anomaly_score = Gauge(
    "aiops_anomaly_score",
    "Anomaly score from AUTOENCODER decision_function (higher=more normal, lower=more anomalous)",
)

# Legacy metric for compatibility with existing dashboards/alerts
disk_anomaly_prediction = Gauge(
    "disk_anomaly_prediction",
    "Legacy anomaly metric; mirrors aiops_anomaly_label for compatibility.",
)

# Behavioral Attestation Phase 1: process-level attribution.
# Retains the last-flagged suspect process until the next anomaly, so
# operators can see "who caused the last incident" rather than resetting
# to empty every normal cycle.
aiops_anomaly_top_process_cpu_percent = Gauge(
    "aiops_anomaly_top_process_cpu_percent",
    "CPU percent of the top suspect process at the last flagged anomaly",
)
aiops_anomaly_top_process_mem_percent = Gauge(
    "aiops_anomaly_top_process_mem_percent",
    "Memory percent of the top suspect process at the last flagged anomaly",
)
aiops_anomaly_top_process_info = Gauge(
    "aiops_anomaly_top_process_info",
    "Info metric (always 1): labels identify the top suspect process at the last flagged anomaly",
    ["pid", "name"],
)

# Behavioral Attestation Phase 2: scoped eBPF trace of the suspect process.
# Only populated when a fresh trace actually runs (rate-limited, see
# ebpf_trace.py); retains the last trace's counts between runs.
aiops_anomaly_suspect_syscall_count = Gauge(
    "aiops_anomaly_suspect_syscall_count",
    "Syscall counts observed during the scoped eBPF trace of the last flagged suspect process. "
    "pid/name identify which process these counts belong to (may lag the current top-process "
    "suspect if that pid is still within the trace cooldown window).",
    ["syscall_type", "pid", "name"],
)

# Behavioral Attestation Phase 4: per-process GPU accounting for the
# suspect process. Retains the last flagged suspect's GPU usage between
# anomalies, same semantics as the other top_process_* gauges.
aiops_anomaly_top_process_gpu_mem_mib = Gauge(
    "aiops_anomaly_top_process_gpu_mem_mib",
    "GPU memory (MiB) used by the top suspect process at the last flagged anomaly",
)


def load_model_and_scaler():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Missing {MODEL_FILE}. Train model first with train_autoencoder_final.py", flush=True)
        sys.exit(1)
    if not os.path.exists(SCALER_FILE):
        print(f"[ERROR] Missing {SCALER_FILE}. Train model first with train_autoencoder_final.py", flush=True)
        sys.exit(1)
    if not os.path.exists("autoencoder_threshold.txt"):
        print("[ERROR] Missing autoencoder_threshold.txt. Train model first with train_autoencoder_final.py", flush=True)
        sys.exit(1)

    print(f"[INFO] Loading model from {MODEL_FILE}", flush=True)
    model = tf.keras.models.load_model(MODEL_FILE)

    print(f"[INFO] Loading scaler from {SCALER_FILE}", flush=True)
    scaler = joblib.load(SCALER_FILE)

    print("[INFO] Loading threshold from autoencoder_threshold.txt", flush=True)
    with open("autoencoder_threshold.txt", "r") as f:
        threshold = float(f.read().strip())

    return model, scaler, threshold

def main():
    print(f"[INFO] Starting AIOps Watchdog on port {PORT}", flush=True)
    print(f"[INFO] Interval: {INTERVAL} seconds", flush=True)
    print(f"[INFO] Feature order: {DATA_FEATURES}", flush=True)

    model, scaler, threshold = load_model_and_scaler()
    gpu_handle = init_gpu(GPU_INDEX)

    columns = DATA_FEATURES

    # Start Prometheus HTTP server
    start_http_server(PORT)
    print(f"[INFO] Prometheus metrics available on :{PORT}", flush=True)

    # Initialize baselines for network/disk IO
    net_prev = psutil.net_io_counters()
    disk_prev = psutil.disk_io_counters()
    t_prev = time.time()

    try:
        prev_disk_used = None
        while True:
            time.sleep(INTERVAL)

            t_now = time.time()
            elapsed = max(t_now - t_prev, 1e-6)

            disk_pct, disk_free_gb, disk_fill_rate_mb_min, inode_pct, prev_disk_used = get_disk_extras("/", elapsed, prev_disk_used)
            cpu_pct = psutil.cpu_percent(interval=None)
            mem_pct = psutil.virtual_memory().percent

            net_now = psutil.net_io_counters()
            disk_now = psutil.disk_io_counters()

            net_bytes = ((net_now.bytes_sent + net_now.bytes_recv) -
                         (net_prev.bytes_sent + net_prev.bytes_recv))
            net_kbps = (net_bytes / 1024.0) / elapsed

            disk_w_bytes = disk_now.write_bytes - disk_prev.write_bytes
            disk_w_kbps = (disk_w_bytes / 1024.0) / elapsed

            net_prev = net_now
            disk_prev = disk_now
            t_prev = t_now

            gpu_util, gpu_mem_mib, gpu_temp_c = get_gpu_metrics(gpu_handle)

            # Update Prometheus metrics for raw values
            aiops_disk_usage_percent.set(disk_pct)
            aiops_disk_free_gb.set(disk_free_gb)
            aiops_disk_fill_rate_mb_min.set(disk_fill_rate_mb_min)
            aiops_inode_usage_percent.set(inode_pct)
            aiops_cpu_usage_percent.set(cpu_pct)
            aiops_mem_usage_percent.set(mem_pct)
            aiops_net_kbps.set(net_kbps)
            aiops_disk_write_kbps.set(disk_w_kbps)
            aiops_gpu_util_percent.set(gpu_util)
            aiops_gpu_mem_mib.set(gpu_mem_mib)
            aiops_gpu_temp_c.set(gpu_temp_c)

            features = [
                disk_pct,
                disk_free_gb,
                disk_fill_rate_mb_min,
                inode_pct,
                cpu_pct,
                mem_pct,
                net_kbps,
                disk_w_kbps,
                gpu_util,
                gpu_mem_mib,
                gpu_temp_c,
            ]

            X = pd.DataFrame([features], columns=columns)
            X_scaled = scaler.transform(X)

            recon = model.predict(X_scaled, verbose=0)
            scores = np.mean(np.square(X_scaled - recon), axis=1)
            label = int(scores[0] > threshold)
            score = float(scores[0])

            aiops_anomaly_label.set(label)
            aiops_anomaly_score.set(score)

            # Legacy metric mirror
            disk_anomaly_prediction.set(label)

            # Behavioral Attestation Phase 1: attribute anomalies to a process
            if label == 1:
                top_procs = get_top_processes(n=5, by="cpu")
                if top_procs:
                    suspect = top_procs[0]
                    aiops_anomaly_top_process_info.clear()
                    aiops_anomaly_top_process_info.labels(
                        pid=str(suspect["pid"]), name=suspect["name"]
                    ).set(1)
                    aiops_anomaly_top_process_cpu_percent.set(suspect["cpu_percent"])
                    aiops_anomaly_top_process_mem_percent.set(suspect["mem_percent"])
                    print(
                        "    [ATTRIBUTION] top suspects: "
                        + ", ".join(
                            f"{p['name']}(pid={p['pid']}, cpu={p['cpu_percent']:.1f}%, mem={p['mem_percent']:.1f}%)"
                            for p in top_procs
                        ),
                        flush=True,
                    )

                    # Behavioral Attestation Phase 2: scoped eBPF trace
                    evidence = trace_suspect_process(suspect["pid"])
                    if evidence:
                        aiops_anomaly_suspect_syscall_count.clear()
                        for syscall_type, count in evidence["counts"].items():
                            aiops_anomaly_suspect_syscall_count.labels(
                                syscall_type=syscall_type, pid=str(suspect["pid"]), name=suspect["name"]
                            ).set(count)
                        print(
                            f"    [EBPF] pid={suspect['pid']} syscalls in 3s: {evidence['counts']}"
                            + (f" | files opened: {evidence['files_opened']}" if evidence["files_opened"] else ""),
                            flush=True,
                        )

                    # Behavioral Attestation Phase 4: per-process GPU accounting
                    gpu_mem = get_gpu_usage_for_pid(suspect["pid"], gpu_handle)
                    aiops_anomaly_top_process_gpu_mem_mib.set(gpu_mem)
                    if gpu_mem > 0:
                        print(f"    [GPU] pid={suspect['pid']} using {gpu_mem:.1f} MiB GPU memory", flush=True)

            ts = datetime.utcnow().isoformat()
            print(
                f"[{ts}] disk={disk_pct:.2f}% cpu={cpu_pct:.2f}% mem={mem_pct:.2f}% "
                f"net={net_kbps:.2f}kB/s disk_w={disk_w_kbps:.2f}kB/s "
                f"gpu_util={gpu_util:.2f}% gpu_mem={gpu_mem_mib:.2f}MiB gpu_temp={gpu_temp_c:.2f}C "
                f"| label={label} score={score:.4f}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Watchdog stopped via KeyboardInterrupt.", flush=True)
    finally:
        if NVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass
        print("[INFO] AIOps Watchdog shutting down.", flush=True)


if __name__ == "__main__":
    main()
