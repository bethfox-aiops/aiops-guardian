#!/usr/bin/env python3
"""
process_attribution.py

Behavioral Attestation Phase 1: process-level attribution.

Shared by the anomaly watchdogs. Tracks per-process CPU/memory across calls
(psutil's cpu_percent needs a prior call per process to report a meaningful
delta rather than 0) and reports the top-N processes by resource usage, so a
watchdog can attach "who was responsible" evidence to an anomaly.
"""

import psutil

_proc_cache = {}  # pid -> psutil.Process


def _refresh_cache():
    """Track newly seen processes and drop ones that have exited."""
    live_pids = set()
    for p in psutil.process_iter(["pid"]):
        pid = p.info["pid"]
        live_pids.add(pid)
        if pid not in _proc_cache:
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)  # prime the internal timer
                _proc_cache[pid] = proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    for pid in list(_proc_cache.keys()):
        if pid not in live_pids:
            del _proc_cache[pid]


def get_top_processes(n=5, by="cpu"):
    """
    Return the top-N processes by resource usage:
    [{"pid": int, "name": str, "cpu_percent": float, "mem_percent": float}, ...]

    Call this once per watchdog loop iteration so cpu_percent deltas stay
    meaningful (they're measured since this function's previous call per pid).
    """
    _refresh_cache()

    snapshot = []
    for pid, proc in list(_proc_cache.items()):
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_percent()
            name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        snapshot.append({"pid": pid, "name": name, "cpu_percent": cpu, "mem_percent": mem})

    key = "cpu_percent" if by == "cpu" else "mem_percent"
    snapshot.sort(key=lambda d: d[key], reverse=True)
    return snapshot[:n]
