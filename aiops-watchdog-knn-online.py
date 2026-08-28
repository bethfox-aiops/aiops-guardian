#!/usr/bin/env python3
"""
aiops-watchdog-knn-online.py

PILOT (2026-08-26) -- online/streaming alternative to aiops-watchdog-knn.py.

Runs PySAD's HalfSpaceTrees anomaly detector side-by-side with the existing
batch PyOD KNN watchdog, on the same feature vector, to test whether a model
that updates continuously (fit_partial every tick, no separate retrain step)
avoids the reboot-triggered / promtail-drift false positives that keep
requiring manual retrains of the batch models (see project_aiops memory).

Deliberately standalone rather than routed through watchdog_common.run():
that shared harness assumes a static pre-trained model + a fixed sklearn-style
scaler (state["scaler"].transform(X) is called unconditionally), which doesn't
fit the online paradigm here (no separate scaler; HalfSpaceTrees needs fixed
feature_mins/feature_maxes instead, and fit_partial/score_partial happen every
tick). Only the pure, read-only collection helpers are reused from
watchdog_common.py -- nothing about the existing three watchdogs is touched.

Requires the ISOLATED pilot venv, not the shared /opt/aiops-venv:
    /home/beth/aiops-agents/.venv-pilot/bin/python3 aiops-watchdog-knn-online.py

(pysad hard-pins pyod==3.5.2, which conflicts with the shared venv's pinned
pyod==2.0.5 that the three live watchdogs' pickled models were validated
against -- see project_aiops memory for why that pin matters. Keeping this in
its own venv avoids that conflict entirely rather than risking it.)

Exposes Prometheus metrics on WATCHDOG_PORT (default: 8019):
    aiops_online_anomaly_score    raw HalfSpaceTrees score (higher = more anomalous)
    aiops_online_anomaly_zscore   running z-score of that score (adaptive, no manual
                                   threshold retrain needed)
    aiops_online_anomaly_label    0 = normal, 1 = anomaly (|zscore| > ZSCORE_THRESHOLD)
    plus the same raw metric gauges as the other watchdogs (disk/cpu/mem/...)
    so this can be compared directly in Grafana.

Not yet wired into Prometheus scrape config or systemd -- run manually for
the pilot comparison period.
"""

import os
import time
from datetime import datetime

import numpy as np
import psutil
from prometheus_client import Gauge, start_http_server

import watchdog_common as common
from feature_transform import transform_bursty_features
from pysad.models import HalfSpaceTrees
from pysad.transform.postprocessing.running_postprocessors import RunningZScorePostprocessor

DATA_FEATURES = common.DATA_FEATURES

# Empirically derived from aiops_data/metrics.csv (125k rows, full history).
# Deliberately the *observed* operating range plus modest padding, NOT
# domain-theoretical bounds (e.g. 0-100 for disk%, 0-16384 for gpu_mem_mib):
# tested wide domain-max bounds first and they wasted almost all of
# HalfSpaceTrees' partitioning resolution on empty space the data never
# occupies (disk sits at 3.5-5% the whole time; gpu_mem tops out ~450MiB on
# this box), which produced a pathologically heavy-tailed, near-useless raw
# score distribution. net_kbps/disk_w_kbps bounds are in log1p space to
# match transform_bursty_features (log1p(500000) ~= 13.1, kept wide since
# promtail's real write bursts are a genuine, known, recurring phenomenon --
# see feature_transform.py's docstring -- not an artifact to tighten away).
FEATURE_MINS = [2.0, -5000.0, 0.4, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FEATURE_MAXES = [8.0, 5000.0, 1.0, 100.0, 100.0, 14.0, 14.0, 100.0, 2000.0, 90.0]

ZSCORE_WINDOW = 250  # ~20min of history at the default 5s interval

# HalfSpaceTrees' raw score (mass * 2^depth, summed across trees) is heavy-
# tailed/multiplicative by construction, not Gaussian -- feeding it straight
# into RunningZScorePostprocessor gave a ~2.9% false-positive rate on real
# historical data at |z|>3 (vs. the ~0.3% a 3-sigma cutoff implies for a
# roughly-Gaussian distribution) and couldn't reliably separate a clear
# synthetic anomaly from ordinary variance. log-transforming the score
# first (see below) fixed the shape; empirically p99.5 of |z| over 2700
# post-warmup historical samples was 5.54, so 5.5 targets a similar
# false-positive budget to that measurement rather than an arbitrary
# textbook cutoff -- same spirit as the autoencoder's 95th->99th percentile
# threshold fix (see project_aiops memory), re-derived for this model.
ZSCORE_THRESHOLD = 5.5

PORT = "8019"


def _make_gauges():
    g = {}
    g["disk_usage_percent"] = Gauge("aiops_disk_usage_percent", "Disk usage percent of root filesystem")
    g["cpu_usage_percent"] = Gauge("aiops_cpu_usage_percent", "CPU usage percent")
    g["mem_usage_percent"] = Gauge("aiops_mem_usage_percent", "Memory usage percent")
    g["net_kbps"] = Gauge("aiops_net_kbps", "Total network throughput (send+recv) in kB/s")
    g["disk_write_kbps"] = Gauge("aiops_disk_write_kbps", "Disk write throughput in kB/s")
    g["gpu_util_percent"] = Gauge("aiops_gpu_util_percent", "GPU utilization percent")
    g["gpu_mem_mib"] = Gauge("aiops_gpu_mem_mib", "GPU memory used in MiB")
    g["gpu_temp_c"] = Gauge("aiops_gpu_temp_c", "GPU temperature in Celsius")
    g["inode_usage_percent"] = Gauge("aiops_inode_usage_percent", "Inode usage percent on /")
    g["disk_fill_rate_mb_min"] = Gauge(
        "aiops_disk_fill_rate_mb_min",
        "Disk usage fill rate on / in MB per minute (positive means filling)",
    )

    g["anomaly_score"] = Gauge(
        "aiops_online_anomaly_score",
        "log(-raw HalfSpaceTrees score); lower = more anomalous (see ZSCORE_THRESHOLD "
        "comment in source for why this is log-transformed rather than raw)",
    )
    g["anomaly_zscore"] = Gauge(
        "aiops_online_anomaly_zscore",
        f"Running z-score ({ZSCORE_WINDOW}-sample window) of the log-transformed HalfSpaceTrees score",
    )
    g["anomaly_label"] = Gauge(
        "aiops_online_anomaly_label",
        f"Online KNN-alternative anomaly label (0=normal, 1=anomaly, |zscore|>{ZSCORE_THRESHOLD})",
    )
    return g


def main():
    port = int(os.getenv("WATCHDOG_PORT", PORT))
    interval = float(os.getenv("WATCHDOG_INTERVAL", "5.0"))
    gpu_index = int(os.getenv("WATCHDOG_GPU_INDEX", "0"))

    print(f"[INFO] Starting PILOT online watchdog (HalfSpaceTrees) on port {port}", flush=True)
    print(f"[INFO] Interval: {interval}s | feature order: {DATA_FEATURES}", flush=True)

    model = HalfSpaceTrees(feature_mins=FEATURE_MINS, feature_maxes=FEATURE_MAXES)
    zscorer = RunningZScorePostprocessor(window_size=ZSCORE_WINDOW)

    gpu_handle = common.init_gpu(gpu_index)
    g = _make_gauges()

    start_http_server(port)
    print(f"[INFO] Prometheus metrics available on :{port}", flush=True)

    net_prev = psutil.net_io_counters()
    disk_prev = psutil.disk_io_counters()
    t_prev = time.time()
    prev_disk_used = None
    n_samples = 0

    try:
        while True:
            time.sleep(interval)

            t_now = time.time()
            elapsed = max(t_now - t_prev, 1e-6)

            disk_pct, disk_free_gb, disk_fill_rate_mb_min, inode_pct, prev_disk_used = (
                common.get_disk_extras("/", elapsed, prev_disk_used)
            )
            cpu_pct = psutil.cpu_percent(interval=None)
            mem_pct = psutil.virtual_memory().percent

            net_now = psutil.net_io_counters()
            disk_now = psutil.disk_io_counters()

            net_bytes = (net_now.bytes_sent + net_now.bytes_recv) - (
                net_prev.bytes_sent + net_prev.bytes_recv
            )
            net_kbps = (net_bytes / 1024.0) / elapsed

            disk_w_bytes = disk_now.write_bytes - disk_prev.write_bytes
            disk_w_kbps = (disk_w_bytes / 1024.0) / elapsed

            net_prev = net_now
            disk_prev = disk_now
            t_prev = t_now

            gpu_util, gpu_mem_mib, gpu_temp_c = common.get_gpu_metrics(gpu_handle)

            g["disk_usage_percent"].set(disk_pct)
            g["disk_fill_rate_mb_min"].set(disk_fill_rate_mb_min)
            g["inode_usage_percent"].set(inode_pct)
            g["cpu_usage_percent"].set(cpu_pct)
            g["mem_usage_percent"].set(mem_pct)
            g["net_kbps"].set(net_kbps)
            g["disk_write_kbps"].set(disk_w_kbps)
            g["gpu_util_percent"].set(gpu_util)
            g["gpu_mem_mib"].set(gpu_mem_mib)
            g["gpu_temp_c"].set(gpu_temp_c)

            features = [
                disk_pct,
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
            features = transform_bursty_features(features, DATA_FEATURES)

            raw_score = model.fit_score_partial(features)
            n_samples += 1
            # log-transform before z-scoring -- see ZSCORE_THRESHOLD comment
            # above for why the raw score can't be z-scored directly.
            # raw_score = -sum(mass * 2^depth) is always <= 0, so -raw_score
            # is always >= 0; guard the one legitimate edge case (all-zero
            # mass on the very first sample, -raw_score == 0) since log(0)
            # is undefined.
            log_score = np.log(-raw_score) if raw_score < 0 else 0.0
            zscore = zscorer.fit_transform_partial(log_score)

            # Z-score is undefined (0.0 from pysad) until the running window
            # has enough samples for a real variance estimate -- don't flag
            # anomalies off a meaningless early z-score.
            label = 1 if (n_samples > ZSCORE_WINDOW and abs(zscore) > ZSCORE_THRESHOLD) else 0

            g["anomaly_score"].set(log_score)
            g["anomaly_zscore"].set(zscore)
            g["anomaly_label"].set(label)

            ts = datetime.utcnow().isoformat()
            print(
                f"[{ts}] disk={disk_pct:.2f}% cpu={cpu_pct:.2f}% mem={mem_pct:.2f}% "
                f"net={net_kbps:.2f}kB/s disk_w={disk_w_kbps:.2f}kB/s "
                f"gpu_util={gpu_util:.2f}% gpu_mem={gpu_mem_mib:.2f}MiB gpu_temp={gpu_temp_c:.2f}C "
                f"| raw_score={raw_score:.4f} zscore={zscore:.4f} label={label}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Pilot online watchdog stopped via KeyboardInterrupt.", flush=True)


if __name__ == "__main__":
    main()
