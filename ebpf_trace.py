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

import json
import os
import subprocess
import time

TRACE_SCRIPT = "/home/beth/aiops-agents/trace_suspect.sh"
SUDO = "/usr/bin/sudo"
SEQUENCE_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aiops_data", "syscall_sequences.jsonl"
)
COOLDOWN_SECONDS = 60

# Ticket file proving *this* code path is the one requesting a trace, not an
# arbitrary "sudo trace_suspect.sh <any pid>" invocation from elsewhere on the
# box. The sudoers NOPASSWD rule for trace_suspect.sh has no way to scope
# *which* PID is legitimate to trace (a bare wildcard), and PID ownership
# can't be used to scope it either -- promtail, Guardian's own most-traced
# real suspect, runs as root, so "only trace beth-owned PIDs" would break the
# feature this exists for. Writing the intended PID here immediately before
# the sudo call, and having trace_suspect.sh require a fresh, matching ticket,
# closes that gap without narrowing which processes can legitimately be
# traced. Lives next to this script rather than /run/user/<uid> because
# systemd system services (User=beth, not a login session) don't get
# XDG_RUNTIME_DIR -- this path exists regardless of login-session state.
TICKET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".trace_ticket")

_last_traced = {}  # pid -> unix timestamp of last trace


def _write_ticket(pid):
    fd = os.open(TICKET_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"{pid} {time.time()}\n")


def trace_suspect_process(pid, timeout=5):
    """
    Run a scoped eBPF trace against `pid` and return a summary evidence dict:
    {"counts": {"OPEN": n, "EXEC": n, "CONNECT": n, "WRITE": n}, "files_opened": [...],
     "sequence": ["OPEN", "WRITE", "WRITE", ...]}

    `sequence` preserves syscall-type order as it actually happened during
    the trace window (bpftrace's own printf-per-event output is already
    ordered; this just keeps that order instead of collapsing straight to
    counts) -- collected for future syscall-sequence modeling
    (Agent Behavioral Attribution backlog item, see ROADMAP.md), not
    consumed by anything yet.

    Returns None if skipped (still in cooldown) or if the trace failed for
    any reason (process exited, sudoers misconfigured, bpftrace error, etc).
    """
    now = time.time()
    last = _last_traced.get(pid)
    if last is not None and (now - last) < COOLDOWN_SECONDS:
        return None
    _last_traced[pid] = now

    try:
        _write_ticket(pid)
        result = subprocess.run(
            [SUDO, "-n", TRACE_SCRIPT, str(pid)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    # trace_suspect.sh runs bpftrace under `timeout 3`, which exits 124 on
    # the expected timeout — that's success, not failure, for our purposes.
    if result.returncode not in (0, 124):
        return None

    counts = {"OPEN": 0, "EXEC": 0, "CONNECT": 0, "WRITE": 0}
    files_opened = set()
    sequence = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind in counts:
            counts[kind] += 1
            sequence.append(kind)
        if kind == "OPEN" and len(parts) >= 3:
            files_opened.add(parts[2])

    result_dict = {
        "counts": counts,
        "files_opened": sorted(files_opened)[:10],
        "sequence": sequence,
    }

    # Best-effort corpus collection for future syscall-sequence modeling
    # (Agent Behavioral Attribution backlog item, see ROADMAP.md) -- append
    # every real trace as one JSON line. Never let a logging failure break
    # the actual attribution feature this function exists for.
    if sequence:
        try:
            os.makedirs(os.path.dirname(SEQUENCE_LOG), exist_ok=True)
            with open(SEQUENCE_LOG, "a") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "pid": pid,
                    "sequence": sequence,
                    "counts": counts,
                }) + "\n")
        except OSError:
            pass

    return result_dict
