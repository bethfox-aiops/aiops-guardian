#!/usr/bin/env python3
"""
retrain_common.py

Shared scaffolding for retrain_recent{,_iforest,_knn}.py: the constants,
GPU init, self-attribution monitoring (Behavioral Attestation Phases 1/2/4
pointed at our own PID during training), and verify_behavior/verify()
wiring are identical across all three retrain scripts. Each script keeps
its own load_data/train_model/save_model span bodies, since those differ
meaningfully by model (the autoencoder's keras architecture and
validation-split threshold derivation in particular).
"""

import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import psutil

from ebpf_trace import trace_suspect_process
from gpu_attribution import poll_max_gpu_usage
from behavioral_policy import verify

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
