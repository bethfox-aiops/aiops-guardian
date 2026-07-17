#!/usr/bin/env python3
"""
gpu_attribution.py

Behavioral Attestation Phase 4: per-process GPU accounting, extending the
existing system-wide GPU util/mem/temp metrics to "which process is using
how much GPU," via NVML (the same library the watchdogs already use for
aggregate GPU stats).
"""

import time

try:
    from pynvml import (
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetGraphicsRunningProcesses,
        NVMLError,
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def get_gpu_processes(handle):
    """
    Return per-process GPU memory usage as a list of dicts:
    [{"pid": int, "used_memory_mib": float, "type": "compute"|"graphics"}, ...]

    Returns [] if NVML is unavailable, no handle, or the query fails (some
    drivers restrict per-process memory querying to root).
    """
    if not NVML_AVAILABLE or handle is None:
        return []

    processes = []
    try:
        for p in nvmlDeviceGetComputeRunningProcesses(handle):
            mem = (p.usedGpuMemory / (1024 * 1024)) if p.usedGpuMemory else 0.0
            processes.append({"pid": p.pid, "used_memory_mib": mem, "type": "compute"})
    except NVMLError:
        pass

    try:
        for p in nvmlDeviceGetGraphicsRunningProcesses(handle):
            mem = (p.usedGpuMemory / (1024 * 1024)) if p.usedGpuMemory else 0.0
            processes.append({"pid": p.pid, "used_memory_mib": mem, "type": "graphics"})
    except NVMLError:
        pass

    return processes


def get_gpu_usage_for_pid(pid, handle):
    """Return GPU memory (MiB) used by a specific pid, or 0.0 if not found."""
    for p in get_gpu_processes(handle):
        if p["pid"] == pid:
            return p["used_memory_mib"]
    return 0.0


def poll_max_gpu_usage(pid, handle, duration=3.0, interval=0.5, stop_event=None):
    """
    Poll GPU memory usage for `pid`, returning the maximum observed value
    in MiB. Meant to run in a background thread alongside a training call
    -- usage can be transient, so a single snapshot could easily land on a
    zero moment and miss real activity.

    If `stop_event` is given, polling continues until the event is set
    (checked once per `interval`) or `duration` elapses as a safety cap,
    whichever comes first -- letting a caller cover the exact span of a
    training call (of unknown length) rather than a fixed guess. Without a
    stop_event, polls for the fixed `duration` only.
    """
    if not NVML_AVAILABLE or handle is None:
        return 0.0

    max_usage = 0.0
    deadline = time.time() + duration
    while time.time() < deadline:
        max_usage = max(max_usage, get_gpu_usage_for_pid(pid, handle))
        if stop_event is not None:
            if stop_event.wait(interval):
                break
        else:
            time.sleep(interval)
    return max_usage
