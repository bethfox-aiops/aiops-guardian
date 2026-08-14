#!/usr/bin/env python3
"""
resume_detection.py

Shared suspend/resume detection via `journalctl -k`. Used by both the
retrain scripts (exclude the transient from *training* data,
retrain_common.py's exclude_resume_transients) and the live watchdogs
(watchdog_common.py, suppress alerts during the transient) -- one
definition of "what counts as a resume event" instead of two copies that
can silently drift apart the way RECENT_ROWS once did.
"""

import re
import subprocess
from datetime import datetime, timedelta

# Confirmed twice (2026-08-06, 2026-08-07) via real retrains: a training
# window that still contains the actual resume transient (disk catch-up
# burst, CPU blip, GPU cold-start warming from ~27C to steady-state) gets
# baked into the model as "normal," inflating the autoencoder's threshold
# roughly 10-30x. 15/30/45-minute buffers were tried against real data;
# 45 minutes was the shortest buffer that consistently (not just on a lucky
# run) brought the threshold back down close to the historical healthy
# baseline. Reused as-is (2026-08-14) to also suppress live watchdog alerts
# during the same transient window, not just training -- same underlying
# transient, same proven duration.
RESUME_EXCLUSION_BUFFER_MINUTES = 45


def get_resume_times(since):
    """Every 'PM: suspend exit' kernel log timestamp since `since` (a
    datetime). Best-effort: if journalctl can't be read for any reason,
    returns an empty list rather than raising -- a diagnostic side-check
    should never take down the caller."""
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--utc", "--since", since.strftime("%Y-%m-%d %H:%M:%S"), "--no-pager"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    resume_times = []
    for line in out.splitlines():
        if "PM: suspend exit" in line:
            m = re.match(r"^(\w+ \d+ \d+:\d+:\d+)", line)
            if m:
                try:
                    resume_times.append(
                        datetime.strptime(m.group(1), "%b %d %H:%M:%S").replace(year=since.year)
                    )
                except ValueError:
                    continue
    return resume_times


def seconds_since_last_resume(lookback_hours=6):
    """For live use: how many seconds ago did the system last resume from
    suspend, looking back up to `lookback_hours`? Returns None if no resume
    event was found in that window (i.e. not currently in a post-resume
    grace period). get_resume_times() calls journalctl with --utc, so both
    the `since` cutoff and the returned timestamps must stay in UTC -- mixing
    in local time here would silently introduce a tz-offset-sized error."""
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    resume_times = get_resume_times(since)
    if not resume_times:
        return None
    return (datetime.utcnow() - max(resume_times)).total_seconds()
