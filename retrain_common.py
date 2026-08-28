#!/usr/bin/env python3
"""
retrain_common.py

Shared scaffolding for retrain_recent{,_iforest,_knn}.py: the constants,
GPU init, self-attribution monitoring (Behavioral Attestation Phases 1/2/4
pointed at our own PID during training), and verify_behavior/verify()
wiring are identical across all three retrain scripts.

For KNN and IForest the *entire* run flow is identical too (they differ
only in the model constructor and a handful of names/labels), so it lives
here as run_simple_retrain() and each script is just a config block. The
autoencoder retrain (retrain_recent.py) keeps its own body -- its
load_data step (wide window + resume-transient filtering) and save_model
step (extra threshold file) differ meaningfully by model.
"""

import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import joblib
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler

from ebpf_trace import trace_suspect_process
from gpu_attribution import poll_max_gpu_usage
from otel_setup import get_tracer
from feature_transform import transform_bursty_features_df
from behavioral_policy import verify
from resume_detection import RESUME_EXCLUSION_BUFFER_MINUTES, get_resume_times

DATA_FILE = "aiops_data/metrics.csv"

# NIST AI RMF MEASURE 2.12 (energy/environmental cost): this host's actual
# Intel RAPL energy counters (/sys/class/powercap/intel-rapl:0/energy_uj)
# are root-only (Platypus side-channel mitigation, CVE-2020-8694/8695) -
# not worth adding new sudo scope just to read them, so this is a clearly-
# labeled ESTIMATE (linear interpolation between Intel's published base and
# turbo power for this CPU), not a real measurement. Source: Intel ARK
# product page for the i9-13950HX, base power 55W / max turbo power 157W,
# checked 2026-08-04 - update these if the host CPU ever changes.
_CPU_BASE_POWER_W = 55
_CPU_TURBO_POWER_W = 157
_CPU_LOGICAL_CORES = os.cpu_count() or 1


def estimate_energy_wh(cpu_percent, duration_s):
    """cpu_percent is psutil's convention (100% = one full core, so up to
    N*100% on an N-core box) - normalize to a 0-1 fraction of total logical
    capacity before interpolating between base and turbo power."""
    utilization = min(cpu_percent / (100 * _CPU_LOGICAL_CORES), 1.0)
    estimated_watts = _CPU_BASE_POWER_W + (_CPU_TURBO_POWER_W - _CPU_BASE_POWER_W) * utilization
    return estimated_watts * duration_s / 3600

FEATURES = [
    # disk_free_gb deliberately excluded (2026-08-05): its std within any
    # short training window is ~0.1 GB (disk usage barely moves minute to
    # minute), so normal real-world drift by score time produces z-scores
    # in the range of -6 to -7 on every sample -- structurally unstable for
    # a distance-based model, and redundant with disk_fill_rate_mb_min
    # (the rate of change), which is the actually anomaly-relevant signal.
    "disk", "disk_fill_rate_mb_min", "inode_pct",
    "cpu", "mem", "net_kbps", "disk_w_kbps",
    "gpu_util", "gpu_mem_mib", "gpu_temp_c",
]

# Shared by all three retrain scripts (2026-08-05) - was previously a
# separate local literal in each script, which is exactly how the
# retrain_recent_iforest.py RECENT_ROWS=100000 bug went unnoticed: three
# independent copies that happened to agree, with nothing to catch it when
# one drifted. One definition now; a script would have to explicitly
# override it to differ, not silently forget to update it.
RECENT_ROWS = 2000

# 2026-08-14 KNN drift fix: a plain df.tail(RECENT_ROWS) only spans ~3.3
# continuous hours at the ~6s sampling rate -- less than one wake/sleep
# cycle. mem swings ~40-60% over the course of a session (low right after
# a suspend/resume, climbing as apps/tabs accumulate), so whichever ~3.3h
# slice a retrain happens to catch bakes in an artificially tight std for
# that feature; real values from a different part of the cycle later score
# as 5+ sigma outliers even though they're completely normal. Confirmed via
# a real KNN retrain that landed in an afternoon-usage slice (mem std 1.75)
# going sustained-anomalous the next morning post-resume (mem down to
# 39.4%, z=-5.3 under that scaler). select_recent_window() below fixes this
# by choosing rows from a wide enough wall-clock span to see the full
# cycle, then downsampling -- density isn't what was missing, coverage was.
RECENT_WINDOW_HOURS = 48


def select_recent_window(df, window_hours=RECENT_WINDOW_HOURS, target_rows=RECENT_ROWS):
    """Select training rows spanning `window_hours` of wall-clock time
    (ending at the most recent row), then evenly downsample to roughly
    `target_rows` rows. `df` must still have its `timestamp` column. Unlike
    df.tail(target_rows), this guarantees the training data sees the full
    range a feature like `mem` covers across a wake/sleep cycle rather than
    whatever narrow slice happened to be active when the retrain ran --
    downsampling only thins density, it doesn't shrink the time span."""
    ts = pd.to_datetime(df["timestamp"])
    window = df[ts >= ts.max() - timedelta(hours=window_hours)]
    if len(window) > target_rows:
        step = len(window) // target_rows
        window = window.iloc[::step]
    return window.reset_index(drop=True)

def exclude_resume_transients(df, buffer_minutes=RESUME_EXCLUSION_BUFFER_MINUTES):
    """Drop rows within `buffer_minutes` after any real suspend/resume event
    found in `journalctl -k` (via resume_detection.get_resume_times), so a
    retrain window doesn't learn the resume transient itself as normal.
    `df` must still have its `timestamp` column (call this before dropping
    it). Best-effort: if journalctl can't be read for any reason, returns
    `df` unchanged rather than failing the whole retrain over a diagnostic
    side-check."""
    if "timestamp" not in df.columns or df.empty:
        return df

    ts = pd.to_datetime(df["timestamp"])
    resume_times = get_resume_times(ts.min())
    if not resume_times:
        return df

    mask = pd.Series(True, index=df.index)
    for rt in resume_times:
        window_end = rt + timedelta(minutes=buffer_minutes)
        mask &= ~((ts >= rt) & (ts <= window_end))

    return df[mask].reset_index(drop=True)


def init_gpu_handle():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
        nvmlInit()
        return nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


@contextmanager
def monitor_self(gpu_handle):
    """
    Behavioral Attestation: link this trace to Phase 1/2 evidence by
    capturing process- and syscall-level evidence of this training step
    itself, using the same tooling the watchdogs use on suspect processes
    -- just pointed at our own PID. Runs the caller's training code (inside
    the `with` block) while ebpf + GPU monitoring threads are active, then
    yields a stats dict with the results once they've stopped.

    Note the eBPF trace window is a fixed 3s (by design, see Phase 2), so
    for a multi-epoch fit it's a representative sample from the start of
    training, not full coverage. GPU polling is cheap and unprivileged, so
    it covers the actual training duration via stop_event instead.
    """
    self_pid = os.getpid()
    self_proc = psutil.Process(self_pid)
    self_proc.cpu_percent(interval=None)  # prime
    start_time = time.time()

    ebpf_result = {}
    trace_thread = threading.Thread(
        target=lambda: ebpf_result.update(evidence=trace_suspect_process(self_pid))
    )
    trace_thread.start()

    gpu_stop = threading.Event()
    gpu_result = {}
    gpu_thread = threading.Thread(
        target=lambda: gpu_result.update(
            max_mib=poll_max_gpu_usage(self_pid, gpu_handle, duration=600, interval=0.5, stop_event=gpu_stop)
        )
    )
    gpu_thread.start()

    stats = {}
    try:
        yield stats
    finally:
        gpu_stop.set()
        trace_thread.join(timeout=5)
        gpu_thread.join(timeout=5)
        stats["cpu_percent"] = self_proc.cpu_percent(interval=None)
        stats["mem_percent"] = self_proc.memory_percent()
        stats["gpu_max_mib"] = gpu_result.get("max_mib", 0.0)
        stats["evidence"] = ebpf_result.get("evidence")
        stats["duration_s"] = time.time() - start_time


def archive_current_models(model_files, archive_prefix):
    """NIST AI RMF GOVERN 1.7 (model decommissioning/phase-out): before a
    retrain overwrites the live model files, snapshot whatever's currently
    live into old/<prefix>_<date>/. Turns the one-off manual backup done by
    hand during the 2026-07-17 defect-demo into an automatic, repeatable
    practice. Cheap - these files are tens of KB - so no retention/pruning
    logic; every retrain gets its own dated snapshot. Returns the archive
    dir, or None if there was nothing live yet to archive (first-ever run)."""
    existing = [f for f in model_files if os.path.exists(f)]
    if not existing:
        return None
    dest_dir = os.path.join("old", f"{archive_prefix}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    os.makedirs(dest_dir, exist_ok=True)
    for f in existing:
        shutil.copy2(f, dest_dir)
    return dest_dir


def report_self_attribution(span, stats):
    """Sets the process/GPU span attributes from monitor_self()'s stats and
    prints+attributes the eBPF evidence, if any fired. Returns the evidence
    (or None) for the caller to pass on to run_verification()."""
    span.set_attribute("process.cpu_percent", stats["cpu_percent"])
    span.set_attribute("process.mem_percent", stats["mem_percent"])
    span.set_attribute("gpu.used_memory_mib", stats["gpu_max_mib"])
    span.set_attribute("training.duration_s", stats["duration_s"])
    span.set_attribute(
        "energy.estimated_wh",
        estimate_energy_wh(stats["cpu_percent"], stats["duration_s"]),
    )
    span.set_attribute("energy.estimation_method", "cpu_percent-interpolated, not RAPL-measured")

    evidence = stats["evidence"]
    if evidence:
        for syscall_type, count in evidence["counts"].items():
            span.set_attribute(f"ebpf.syscall.{syscall_type.lower()}", count)
        if evidence["files_opened"]:
            span.set_attribute("ebpf.files_opened", evidence["files_opened"])
        print(f"[INFO] eBPF evidence during training: {evidence['counts']}")
    return evidence


def run_verification(span, policy_name, output_files, run_start_time, gpu_mib, evidence, row_count, on_fail=None):
    """Runs behavioral_policy.verify() for this retrain run, sets the
    verification span attributes, prints PASS/FAIL, and (if the run failed
    and on_fail is given) calls on_fail(violations) -- e.g. to post a
    Grafana annotation."""
    files_touched = [
        f for f in output_files
        if os.path.exists(f) and os.path.getmtime(f) >= run_start_time
    ]
    result = verify(
        policy_name,
        files_touched=files_touched,
        gpu_mib=gpu_mib,
        network_connects=(evidence["counts"]["CONNECT"] if evidence else 0),
        row_count=row_count,
    )
    span.set_attribute("verification.passed", result["passed"])
    span.set_attribute("verification.violations", result["violations"])
    if result["passed"]:
        print("[VERIFY] PASS: this run matched its behavioral policy.")
    else:
        print("[VERIFY] FAIL: this run violated its behavioral policy:")
        for v in result["violations"]:
            print(f"  - {v}")
        if on_fail:
            on_fail(result["violations"])
    return result


def run_simple_retrain(*, name, detector_label, model_factory, model_file,
                       scaler_file, archive_prefix, service_name, on_fail=None):
    """Full retrain run for the distance/tree models (KNN, IForest), whose
    load_data/train_model/save_model/verify_behavior bodies are identical
    bar the model constructor. `name` is the short model slug used in the
    tracer service name, span names, and policy name ("knn"/"iforest");
    `detector_label` is the human name in the training log line;
    `model_factory` is a zero-arg callable returning a fresh unfitted
    model (called inside the self-attribution monitor, same as the
    inline versions did); `on_fail` is forwarded to run_verification().

    The autoencoder retrain does NOT use this -- see module docstring."""
    tracer = get_tracer(f"aiops-retrain-{name}")
    gpu_handle = init_gpu_handle()
    run_start_time = time.time()

    with tracer.start_as_current_span(f"retrain_{name}_run") as run_span:
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
            print(f"[INFO] Training {detector_label} anomaly detector...")

            with monitor_self(gpu_handle) as stats:
                model = model_factory()
                model.fit(X_scaled)

            evidence = report_self_attribution(span, stats)

            num_anomalies = int((model.labels_ == 1).sum())
            print(f"[INFO] Anomalies flagged in training data: {num_anomalies} / {len(df)}")
            span.set_attribute("training_rows", len(df))
            span.set_attribute("anomalies_in_training_data", num_anomalies)

        with tracer.start_as_current_span("save_model") as span:
            archive_dir = archive_current_models([model_file, scaler_file], archive_prefix)
            if archive_dir:
                print(f"[INFO] Archived previous model to {archive_dir}")
                span.set_attribute("archived_to", archive_dir)

            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            print(f"[INFO] Saved {model_file}, {scaler_file}")
            span.set_attribute("model_file", model_file)
            span.set_attribute("scaler_file", scaler_file)

        with tracer.start_as_current_span("verify_behavior") as span:
            run_verification(
                span, f"retrain_{name}",
                output_files=(model_file, scaler_file),
                run_start_time=run_start_time,
                gpu_mib=stats["gpu_max_mib"],
                evidence=evidence,
                row_count=len(df),
                on_fail=on_fail,
            )

    print(f"[INFO] Done. Restart {service_name} to load new model.")
