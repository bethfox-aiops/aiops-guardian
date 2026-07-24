#!/usr/bin/env python3
"""
aiops-watchdog-logs.py

Guardian's log-anomaly watchdog. Like aiops-watchdog-windows.py, this
doesn't collect anything locally -- it queries an existing data source
(Loki, via journald logs shipped by Promtail's `journal` scrape job) for
each Guardian service and re-exposes the result as Prometheus gauges.

Checks two things per unit, both chosen from what was actually observed in
this codebase's logs rather than guessed at:
    error_count   lines matching "[ERROR]" or a raw Python traceback in the
                  lookback window. Guardian's own code uses "[ERROR]"
                  narrowly and only for genuine failure conditions (see
                  watchdog_common.require_file(), aiops-watchdog-ml.py,
                  train_knn_final.py) -- confirmed silent on a healthy
                  system. Deliberately NOT matching on generic
                  case-insensitive "error"/"warn" substrings: that matched
                  the sklearn "UserWarning: X has feature names..." noise
                  on effectively every cycle during design (same class of
                  false-positive as the Windows watchdog's first service
                  check draft -- see CRITICAL_SERVICES there).
    silent        1 if the unit has logged nothing at all in the lookback
                  window while systemd reports it active -- catches a hung
                  process that's still "active (running)" but stuck, which
                  no Prometheus gauge would show. Only evaluated for units
                  systemd reports active, so an intentionally-stopped
                  service doesn't get flagged.

Exposes Prometheus metrics on WATCHDOG_PORT (default: 8017):
    aiops_logs_active{unit}
    aiops_logs_query_ok{unit}
    aiops_logs_error_count{unit}
    aiops_logs_silent{unit}
    aiops_logs_health_score{unit}
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from prometheus_client import Gauge, start_http_server

LOKI_URL = "http://127.0.0.1:3100"
LOOKBACK = "5m"
LOOKBACK_SECONDS = 5 * 60

GUARDIAN_UNITS = [
    "aiops-guardian-health.service",
    "aiops-watchdog-knn.service",
    "aiops-watchdog-iforest.service",
    "aiops-watchdog-autoencoder.service",
    "aiops-watchdog-ml.service",
    "aiops-watchdog-windows.service",
]

# Deliberately narrow -- see module docstring for why this isn't a broader
# case-insensitive "error|warn" match. Backtick-quoted: LogQL double-quoted
# strings go through Go-style string unescaping before regex compilation,
# where "\[" isn't a valid escape ("invalid char escape") -- backticks are
# raw strings, so the backslashes reach the regex engine unchanged.
ERROR_PATTERN = r"`\[ERROR\]|Traceback \(most recent call last\)`"

PORT = int(os.getenv("WATCHDOG_PORT", "8017"))
INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "60.0"))

g_active       = Gauge("aiops_logs_active",       "1 if systemd reports this unit active", ["unit"])
g_query_ok     = Gauge("aiops_logs_query_ok",     "1 if the Loki queries for this unit succeeded this cycle; 0 if Loki itself was unreachable/erroring -- distinct from error_count/silent, which reflect the unit's own logs, not Loki's health", ["unit"])
g_error_count  = Gauge("aiops_logs_error_count",  f"Count of [ERROR]/traceback lines in the last {LOOKBACK}", ["unit"])
g_silent       = Gauge("aiops_logs_silent",       f"1 if the unit has logged nothing in the last {LOOKBACK} while active", ["unit"])
g_health_score = Gauge("aiops_logs_health_score", "Log health score per unit (0-100)", ["unit"])


def loki_instant_query(expr):
    """Returns the float value of a scalar/single-series LogQL instant
    query, or None if there's no result or the query failed."""
    url = f"{LOKI_URL}/loki/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[WARN] Loki query failed: {e}", flush=True)
        return None
    result = data.get("data", {}).get("result", [])
    return float(result[0]["value"][1]) if result else 0.0


def is_active(unit):
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def check_unit(unit):
    active = is_active(unit)
    g_active.labels(unit=unit).set(int(active))

    if not active:
        # Not this watchdog's job to flag a service that isn't supposed
        # to be running right now -- just stop reporting on it.
        g_query_ok.labels(unit=unit).set(1)
        g_error_count.labels(unit=unit).set(0)
        g_silent.labels(unit=unit).set(0)
        g_health_score.labels(unit=unit).set(100)
        print(f"[INFO] {unit}: not active, skipping log checks", flush=True)
        return

    error_count = loki_instant_query(
        f'count_over_time({{job="systemd-journal", unit="{unit}"}} |~ {ERROR_PATTERN} [{LOOKBACK}])'
    )
    total_lines = loki_instant_query(
        f'count_over_time({{job="systemd-journal", unit="{unit}"}}[{LOOKBACK}])'
    )

    if error_count is None or total_lines is None:
        # Loki itself is unreachable/erroring -- we genuinely don't know
        # this unit's log health right now. Don't guess healthy or
        # unhealthy (that's exactly the "or 0.0" bug this replaced: it
        # silently reported error_count=0 on query failure, indistinguishable
        # from a verified zero); leave health_score at its last known value
        # and surface the uncertainty via query_ok instead.
        g_query_ok.labels(unit=unit).set(0)
        print(f"[WARN] {unit}: Loki query failed, log health unknown this cycle", flush=True)
        return

    g_query_ok.labels(unit=unit).set(1)
    silent = total_lines == 0

    score = 100
    if error_count > 0:
        score -= 50
    if silent:
        score -= 50

    g_error_count.labels(unit=unit).set(error_count)
    g_silent.labels(unit=unit).set(int(silent))
    g_health_score.labels(unit=unit).set(max(score, 0))

    print(
        f"[INFO] {unit}: score={score} error_count={int(error_count)} silent={int(silent)}",
        flush=True,
    )


def main():
    print(f"[INFO] Starting log watchdog on port {PORT}", flush=True)
    print(f"[INFO] Interval: {INTERVAL} seconds, lookback: {LOOKBACK}", flush=True)
    print(f"[INFO] Watching units: {GUARDIAN_UNITS}", flush=True)
    start_http_server(PORT)
    print(f"[INFO] Prometheus metrics available on :{PORT}", flush=True)

    while True:
        for unit in GUARDIAN_UNITS:
            check_unit(unit)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
