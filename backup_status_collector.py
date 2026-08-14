#!/usr/bin/env python3
"""
backup_status_collector.py

Textfile-collector script for the two backups on this host (deja-dup,
user-level; root-backup-to-usb.sh, root-level) -- same pattern as
guardian_disk_health.ps1 uses on the Windows side to fill a gap
node_exporter's default collectors don't cover: neither backup exposes
any Prometheus-native metrics on its own.

Two different levels of fidelity, deliberately:
- root-backup-to-usb.sh invokes `duplicity` directly (now with --progress,
  added alongside this script) and logs to journal via its systemd unit,
  so a real percent-complete is parseable from its own progress line.
- deja-dup is a compiled GNOME app with no CLI progress hook available
  from outside it and no verbose journal logging -- running/elapsed/
  last-success is the honest ceiling here, not a percentage that would
  just be a guess dressed up as data.

Run periodically via backup-status-collector.timer (30s -- backups are
bursty, time-sensitive events, unlike the 15min disk_watchdog.py cadence).
"""

import re
import subprocess
from datetime import datetime, timezone

OUTPUT_FILE = "/var/lib/node_exporter/textfile_collector/aiops_backup_status.prom"


def _run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _iso_to_epoch(iso_str):
    iso_str = iso_str.strip().strip("'")
    if not iso_str:
        return None
    try:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _process_running(exact_name):
    out = _run(["pgrep", "-x", exact_name])
    pids = [p for p in out.split() if p.isdigit()]
    return pids[0] if pids else None


def _elapsed_seconds(pid):
    out = _run(["ps", "-o", "etimes=", "-p", pid]).strip()
    return float(out) if out.isdigit() else None


RESULT_RUNNING, RESULT_FAILED, RESULT_SUCCESS = 2, 0, 1


def collect_deja_dup():
    pid = _process_running("deja-dup")
    running = 1 if pid else 0
    elapsed = _elapsed_seconds(pid) if pid else None
    last_backup_raw = _run(["gsettings", "get", "org.gnome.DejaDup", "last-backup"])
    last_success = _iso_to_epoch(last_backup_raw)

    if running:
        result = RESULT_RUNNING
    else:
        # No systemd unit to ask (deja-dup is a plain background app, not a
        # service) -- but last-backup (set only on genuine success) vs.
        # last-run (set whenever an attempt starts) is an equally reliable
        # signal already exposed by deja-dup itself: if the most recent
        # attempt succeeded, last-backup catches up to (or matches) it; if
        # it failed/got killed partway, last-backup stays behind at
        # whatever the previous successful run set it to. Confirmed against
        # a real case: today's run got stuck and was killed, but its actual
        # data transfer had already completed and set last-backup before
        # the stuck phase -- correctly reads as success, not a false
        # failure, because the transfer genuinely did succeed.
        last_run_raw = _run(["gsettings", "get", "org.gnome.DejaDup", "last-run"])
        last_run = _iso_to_epoch(last_run_raw)
        if last_run is not None and last_success is not None and last_success >= last_run:
            result = RESULT_SUCCESS
        elif last_success is not None:
            result = RESULT_FAILED
        else:
            result = None  # never run at all -- no opinion either way

    return running, elapsed, last_success, result


def collect_root_backup():
    pid = _process_running("duplicity")
    running = 1 if pid else 0

    percent = None
    if running:
        # Latest --progress line from the current run only, not a stale one
        # from a prior run sitting earlier in the same journal window.
        log = _run(
            ["journalctl", "-u", "root-backup-to-usb.service", "--no-pager",
             "-n", "500", "--output=cat"],
            timeout=10,
        )
        matches = re.findall(r"(\d+)%\s+ETA", log)
        if matches:
            percent = float(matches[-1])

    last_success = None
    log = _run(
        ["journalctl", "-u", "root-backup-to-usb.service", "--no-pager",
         "-n", "2000", "--grep", "root-backup-to-usb: done",
         "--output=short-iso"],
        timeout=10,
    )
    lines = [l for l in log.splitlines() if l.strip() and not l.startswith("-- Boot")]
    if lines:
        m = re.match(r"^(\S+)", lines[-1])
        if m:
            try:
                # strptime's %z accepts both "-0700" (journalctl's format)
                # and "-07:00" -- fromisoformat only accepts the latter on
                # Python <3.11, and silently produced 0 here until caught.
                dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S%z")
                last_success = dt.timestamp()
            except ValueError:
                pass

    if running:
        result = RESULT_RUNNING
    else:
        # Type=oneshot systemd unit, so systemd itself already tracks the
        # last invocation's real exit status -- a cleaner signal than
        # anything log-parsing could infer, and free (no script changes).
        systemd_result = _run(
            ["systemctl", "show", "root-backup-to-usb.service", "-p", "Result", "--value"]
        ).strip()
        if systemd_result == "":
            result = None  # never run at all
        elif systemd_result == "success":
            result = RESULT_SUCCESS
        else:
            result = RESULT_FAILED

    return running, percent, last_success, result


def render(deja_dup, root_backup):
    dd_running, dd_elapsed, dd_last, dd_result = deja_dup
    rb_running, rb_percent, rb_last, rb_result = root_backup

    lines = [
        "# HELP aiops_backup_running 1 if this backup is currently running, 0 otherwise",
        "# TYPE aiops_backup_running gauge",
        f'aiops_backup_running{{backup="deja_dup"}} {dd_running}',
        f'aiops_backup_running{{backup="root_usb"}} {rb_running}',
        "# HELP aiops_backup_elapsed_seconds How long the current run has been active (only meaningful while running)",
        "# TYPE aiops_backup_elapsed_seconds gauge",
        f'aiops_backup_elapsed_seconds{{backup="deja_dup"}} {dd_elapsed if dd_elapsed is not None else 0}',
        "# HELP aiops_backup_percent_complete duplicity's own reported percent complete (root_usb only -- deja-dup exposes no progress hook)",
        "# TYPE aiops_backup_percent_complete gauge",
        f'aiops_backup_percent_complete{{backup="root_usb"}} {rb_percent if rb_percent is not None else -1}',
        "# HELP aiops_backup_last_success_timestamp Unix timestamp of the last successful completion",
        "# TYPE aiops_backup_last_success_timestamp gauge",
        f'aiops_backup_last_success_timestamp{{backup="deja_dup"}} {dd_last if dd_last is not None else 0}',
        f'aiops_backup_last_success_timestamp{{backup="root_usb"}} {rb_last if rb_last is not None else 0}',
        "# HELP aiops_backup_last_result 2=currently running, 1=last attempt succeeded, 0=last attempt failed, absent=never run",
        "# TYPE aiops_backup_last_result gauge",
    ]
    if dd_result is not None:
        lines.append(f'aiops_backup_last_result{{backup="deja_dup"}} {dd_result}')
    if rb_result is not None:
        lines.append(f'aiops_backup_last_result{{backup="root_usb"}} {rb_result}')
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    content = render(collect_deja_dup(), collect_root_backup())
    tmp_path = OUTPUT_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
    import os
    os.replace(tmp_path, OUTPUT_FILE)  # atomic, so node_exporter never reads a half-written file
    print(content)
