#!/usr/bin/env python3
"""
ebpf_trace.py

Behavioral Attestation Phase 2: scoped eBPF tracing of a single suspect
process, invoked when Phase 1 (process_attribution.py) has already
identified a suspect PID for a flagged anomaly.

Runs trace_suspect.sh via passwordless sudo (see /etc/sudoers.d/aiops-trace)
since bpftrace requires actual root and cannot run with capabilities alone.
Rate-limited per PID so a long-sustained anomaly doesn't spawn a fresh
privileged trace on every watchdog tick.
"""

import subprocess
import time

TRACE_SCRIPT = "/home/beth/aiops-agents/trace_suspect.sh"
SUDO = "/usr/bin/sudo"
COOLDOWN_SECONDS = 60

_last_traced = {}  # pid -> unix timestamp of last trace


def trace_suspect_process(pid, timeout=5):
    """
    Run a scoped eBPF trace against `pid` and return a summary evidence dict:
    {"counts": {"OPEN": n, "EXEC": n, "CONNECT": n, "WRITE": n}, "files_opened": [...]}

    Returns None if skipped (still in cooldown) or if the trace failed for
    any reason (process exited, sudoers misconfigured, bpftrace error, etc).
    """
    now = time.time()
    last = _last_traced.get(pid)
    if last is not None and (now - last) < COOLDOWN_SECONDS:
        return None
    _last_traced[pid] = now

    try:
        result = subprocess.run(
            [SUDO, "-n", TRACE_SCRIPT, str(pid)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # trace_suspect.sh runs bpftrace under `timeout 3`, which exits 124 on
    # the expected timeout — that's success, not failure, for our purposes.
    if result.returncode not in (0, 124):
        return None

    counts = {"OPEN": 0, "EXEC": 0, "CONNECT": 0, "WRITE": 0}
    files_opened = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind in counts:
            counts[kind] += 1
        if kind == "OPEN" and len(parts) >= 3:
            files_opened.add(parts[2])

    return {
        "counts": counts,
        "files_opened": sorted(files_opened)[:10],
    }
