#!/usr/bin/env python3
"""
aiops-watchdog-ml.py

Training data collector for AIOps KNN model.

- Collects multi-metric system stats:
    disk, cpu, mem, net_kbps, disk_w_kbps
- Adds NVIDIA GPU stats via NVML:
    gpu_util, gpu_mem_mib, gpu_temp_c
- Appends rows to aiops_data/metrics.csv for offline training.
"""

import os
import time
import csv
import argparse
from datetime import datetime

import psutil

# GPU / NVML imports
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


DATA_DIR = "aiops_data"
DATA_FILE = os.path.join(DATA_DIR, "metrics.csv")

COLUMNS = [
    "timestamp",
    "disk",          # root filesystem usage %
    "disk_free_gb",        # free disk space in GB
    "disk_fill_rate_mb_min",  # disk growth rate MB/min
    "inode_pct",           # inode usage %
    "cpu",           # CPU usage %
    "mem",           # RAM usage %
    "net_kbps",      # total network throughput (send+recv) kB/s
    "disk_w_kbps",   # disk write throughput kB/s
    "gpu_util",      # GPU utilization %
    "gpu_mem_mib",   # GPU memory used MiB
    "gpu_temp_c",    # GPU temperature in Celsius
]


def init_gpu(gpu_index: int):
    """Initialize NVML and return a handle to the GPU, or None if unavailable."""
    if not NVML_AVAILABLE:
        print("[WARN] pynvml not installed. GPU metrics will be 0. "
              "Install with: pip3 install nvidia-ml-py3")
        return None

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        print(f"[INFO] Using GPU index {gpu_index} for metrics.")
        return handle
    except Exception as e:
        print(f"[WARN] Failed to initialize NVML / GPU index {gpu_index}: {e}")
        print("[WARN] GPU metrics will be recorded as 0.")
        return None


def get_gpu_metrics(handle):
    """Return (util, mem_used_mib, temp_c). If no handle, return zeros."""
    if handle is None:
        return 0.0, 0.0, 0.0

    try:
        util = nvmlDeviceGetUtilizationRates(handle).gpu  # percent
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        mem_used_mib = mem_info.used / (1024 * 1024)
        temp_c = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        return float(util), float(mem_used_mib), float(temp_c)
    except Exception as e:
        print(f"[WARN] Error reading GPU metrics: {e}")
        return 0.0, 0.0, 0.0


def ensure_data_file():
    """Ensure data directory and CSV file with header exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"[INFO] Creating new metrics file at {DATA_FILE}")
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
    else:
        print(f"[INFO] Appending to existing metrics file at {DATA_FILE}")


def collect_metrics_loop(interval: float, gpu_index: int):
    """Main training data collection loop."""
    ensure_data_file()

    # Initialize GPU
    gpu_handle = init_gpu(gpu_index)

    # Initialize net/disk IO baseline
    net_prev = psutil.net_io_counters()
    disk_prev = psutil.disk_io_counters()
    t_prev = time.time()
    prev_disk_used = None

    print(f"[INFO] Starting training data collection every {interval} seconds.")
    print("[INFO] Press Ctrl+C to stop when you have enough samples.\n")

    try:
        while True:
            time.sleep(interval)

            t_now = time.time()
            elapsed = max(t_now - t_prev, 1e-6)

            # System metrics
            usage = psutil.disk_usage("/")
            disk_pct = usage.percent
            disk_free_gb = usage.free / (1024**3)

            current_disk_used = usage.used
            disk_fill_rate_mb_min = 0.0
            if prev_disk_used is not None:
                delta_bytes = current_disk_used - prev_disk_used
                disk_fill_rate_mb_min = (delta_bytes / (1024**2)) * (60.0 / elapsed)

            prev_disk_used = current_disk_used

             # Guardrail: disk fill rate sanity
            if not (-5000.0 <= disk_fill_rate_mb_min <= 5000.0):
                print(f"[WARN] disk_fill_rate_mb_min out of range: {disk_fill_rate_mb_min:.2f} MB/min (skipping sample)")
                t_prev = t_now
                continue

            statvfs = os.statvfs("/")
            inode_pct = 0.0
            if statvfs.f_files > 0:
                inode_pct = ((statvfs.f_files - statvfs.f_ffree) / statvfs.f_files) * 100.0 

            cpu_pct = psutil.cpu_percent(interval=1)
            mem_pct = psutil.virtual_memory().percent

            net_now = psutil.net_io_counters()
            disk_now = psutil.disk_io_counters()

            net_bytes = ((net_now.bytes_sent + net_now.bytes_recv) -
                         (net_prev.bytes_sent + net_prev.bytes_recv))

            if net_bytes < 0:
                print(f"[WARN] net counter reset detected (net_bytes={net_bytes}); setting net_kbps=0 for this sample")
                net_kbps = 0.0
            else:
                net_kbps = (net_bytes / 1024.0) / elapsed

            disk_w_bytes = disk_now.write_bytes - disk_prev.write_bytes

            if disk_w_bytes < 0:
                print(f"[WARN] disk counter reset detected (disk_w_bytes={disk_w_bytes}); setting disk_w_kbps=0 for this sample")
                disk_w_kbps = 0.0
            else:

                disk_w_kbps = (disk_w_bytes / 1024.0) / elapsed

            # Guardrail: rate sanity
            if not (-1e6 <= net_kbps <= 1e6):
                print(f"[WARN] net_kbps out of range: {net_kbps:.2f} (skipping sample)")
                t_prev = t_now
                continue

            if not (-1e6 <= disk_w_kbps <= 1e6):
                print(f"[WARN] disk_w_kbps out of range: {disk_w_kbps:.2f} (skipping sample)")
                t_prev = t_now
                continue

            net_prev = net_now
            disk_prev = disk_now

            # Guardrail: CPU/MEM sanity
            if not (0.0 <= cpu_pct <= 100.0):
                print(f"[WARN] CPU out of range: {cpu_pct} (skipping sample)")
                t_prev = t_now
                continue

            if not (0.0 <= mem_pct <= 100.0):
                print(f"[WARN] MEM out of range: {mem_pct} (skipping sample)")
                t_prev = t_now
                continue

            t_prev = t_now

            # GPU metrics
            gpu_util, gpu_mem_mib, gpu_temp_c = get_gpu_metrics(gpu_handle)

            timestamp = datetime.utcnow().isoformat()

            row = [
                timestamp,
                round(disk_pct, 2),
                round(disk_free_gb, 2),
                round(disk_fill_rate_mb_min, 2),
                round(inode_pct, 2),
                round(cpu_pct, 2),
                round(mem_pct, 2),
                round(net_kbps, 2),
                round(disk_w_kbps, 2),
                round(gpu_util, 2),
                round(gpu_mem_mib, 2),
                round(gpu_temp_c, 2),   
            ]

            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

                print(
                    f"[DATA] {row[0]} | "
                    f"disk={row[1]}% free={row[2]}GB fill={row[3]}MB/min inode={row[4]}% | "
                    f"cpu={row[5]}% mem={row[6]}% | "
                    f"net={row[7]} kB/s disk_w={row[8]} kB/s | "
                    f"gpu_util={row[9]}% gpu_mem={row[10]} MiB gpu_temp={row[11]}C"
                )              
    except KeyboardInterrupt:
        print("\n[INFO] Stopped training data collection (Ctrl+C).")
    finally:
        if NVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass
        print(f"[INFO] Training data saved to {DATA_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="AIOps training data collector (multi-metric + GPU)."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run in training mode (collect metrics to CSV).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Sampling interval in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to monitor (default: 0).",
    )

    args = parser.parse_args()

    if not args.train:
        print(
            "[ERROR] This script currently only supports --train mode.\n"
            "Usage: python3 aiops-watchdog-ml.py --train [--interval 5.0] [--gpu-index 0]"
        )
        return

    collect_metrics_loop(interval=args.interval, gpu_index=args.gpu_index)


if __name__ == "__main__":
    main()
