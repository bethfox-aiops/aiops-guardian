#!/usr/bin/env python3
"""
diagnose_anomaly.py

Behavioral Attestation: automated first-pass diagnosis for a sustained
watchdog anomaly. Codifies the manual investigation playbook developed
across a week of real incidents (reboot-triggered drift, suspend/resume-
triggered drift, the promtail-write-rate pattern) into a deterministic
script, instead of a human re-deriving the same checks by hand every time
an alert fires.

Scope note (2026-08-05): this is the diagnostic-logic half only. It does
not run automatically when an alert fires (no webhook/trigger wiring yet)
and does not itself take any action (no automatic retrain) -- it produces
a report for a human to read and act on. Triggering this from Alertmanager
and gating any recommended action behind an approval step (matching
aiops-approval.py's existing pattern) are deliberately separate, not-yet-
built next steps.

Usage: python3 diagnose_anomaly.py knn|iforest|autoencoder
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from process_attribution import get_top_processes

PROM_URL = "http://127.0.0.1:9090"

WATCHDOGS = {
    "knn":         {"port": 8011, "service": "aiops-watchdog-knn.service",         "job": "aiops-watchdog-knn"},
    "iforest":     {"port": 8012, "service": "aiops-watchdog-iforest.service",     "job": "aiops-watchdog-iforest"},
    "autoencoder": {"port": 8013, "service": "aiops-watchdog-autoencoder.service", "job": "aiops-watchdog-autoencoder"},
}

# Ground-truth thresholds -- deliberately loose. This isn't re-deciding
# whether the HOST is healthy (guardian_security.py/aiops-guardian-health.py
# already do that); it's a quick sanity check for "does this look like a
# real incident" before trusting a model's anomaly call.
#
# swap_used_pct is deliberately NOT a health gate, only reported for
# context: found live while testing this script (2026-08-05) that a
# multi-day uptime with only 2GB total swap normally accumulates cold
# pages up into the 80%+ range with zero real memory pressure (mem_used
# was a completely healthy 49% at the same moment) -- swap fullness alone
# is not a reliable unhealthy signal at this scale.
LOAD1_UNHEALTHY = 4.0
MEM_UNHEALTHY_PCT = 90.0


def _run(cmd, timeout=15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


def prom_query_range(expr, start_ts, end_ts, step="15s"):
    """Returns [(timestamp, value), ...] for a single-series PromQL range query."""
    params = {"query": expr, "start": start_ts, "end": end_ts, "step": step}
    url = f"{PROM_URL}/api/v1/query_range?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[WARN] Prometheus range query failed: {e}", flush=True)
        return []
    result = data.get("data", {}).get("result", [])
    if not result:
        return []
    return [(float(t), float(v)) for t, v in result[0]["values"]]


def check_current_state(port):
    """Reads label/score directly from the watchdog's own /metrics -- the
    same source of truth the live dashboards use, not a Prometheus scrape
    that could lag by one interval."""
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None, None
    label, score = None, None
    for line in text.splitlines():
        if line.startswith("aiops_anomaly_label "):
            label = int(float(line.split()[1]))
        elif line.startswith("aiops_anomaly_score "):
            score = float(line.split()[1])
    return label, score


def check_ground_truth():
    """Real system health, independent of what any model thinks. Reuses
    process_attribution.py's cache -- its cpu_percent needs two calls with
    a gap to be meaningful (first call only primes the internal timer), so
    call it once here, sleep briefly, then take the real snapshot."""
    load1, _, _ = __import__("os").getloadavg()

    import psutil
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    get_top_processes(n=5, by="cpu")  # prime
    time.sleep(1.0)
    top = get_top_processes(n=5, by="cpu")

    healthy = (
        load1 < LOAD1_UNHEALTHY
        and mem.percent < MEM_UNHEALTHY_PCT
    )
    return {
        "load1": load1,
        "mem_used_pct": mem.percent,
        "mem_available_pct": 100 - mem.percent,
        "swap_used_pct": swap.percent,
        "top_processes": top,
        "healthy": healthy,
    }


def check_suspend_or_reboot_correlation(lookback_minutes=30):
    """Returns (timestamp_str, kind) for the most recent suspend/resume or
    reboot within the lookback window, or (None, None). Deliberately a
    loose "did one of these happen recently" check, not a precise
    onset-time correlation -- matches how this has been diagnosed by hand
    all week; a tighter correlation is a reasonable future refinement."""
    out = _run(["journalctl", "-k", "--since", f"-{lookback_minutes}min", "--no-pager"], timeout=10)
    for line in out.splitlines():
        if "PM: suspend exit" in line:
            return " ".join(line.split()[:3]), "suspend/resume"

    uptime_out = _run(["uptime", "-s"], timeout=5).strip()
    if uptime_out:
        try:
            boot_time = datetime.strptime(uptime_out, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - boot_time < timedelta(minutes=lookback_minutes):
                return uptime_out, "reboot"
        except ValueError:
            pass
    return None, None


def check_self_resolution(job, window_minutes=10):
    """Returns (pct_anomalous, is_declining) over the recent window, or
    (None, None) if Prometheus has no data for it. is_declining compares
    the anomaly rate in the second half of the window against the first
    half -- a rough but effective "is this settling on its own" signal."""
    now = time.time()
    points = prom_query_range(f'aiops_anomaly_label{{job="{job}"}}', now - window_minutes * 60, now, step="15s")
    if not points:
        return None, None
    values = [v for _, v in points]
    pct_anomalous = sum(values) / len(values) * 100
    half = len(values) // 2
    if half == 0:
        return pct_anomalous, False
    first_half_rate = sum(values[:half]) / half
    second_half_rate = sum(values[half:]) / (len(values) - half)
    return pct_anomalous, (second_half_rate < first_half_rate)


def diagnose(model_name):
    if model_name not in WATCHDOGS:
        print(f"[ERROR] Unknown model '{model_name}'. Choose from: {', '.join(WATCHDOGS)}")
        sys.exit(1)

    cfg = WATCHDOGS[model_name]
    print(f"=== Diagnosing {model_name} ({cfg['service']}) ===\n")

    label, score = check_current_state(cfg["port"])
    if label is None:
        print("[ERROR] Could not reach the watchdog's /metrics endpoint.")
        sys.exit(1)
    print(f"Current state: label={label} score={score:.4f}")

    if label == 0:
        print("\nNot currently anomalous. If this ran because an alert fired, the")
        print("issue may already have resolved by the time this ran -- still worth")
        print("checking the trend below rather than assuming it's fully over.")

    print("\nChecking ground truth (takes ~1s for a real CPU sample)...")
    gt = check_ground_truth()
    print(f"Ground truth: load1={gt['load1']:.2f} mem_used={gt['mem_used_pct']:.1f}% "
          f"swap_used={gt['swap_used_pct']:.1f}%")
    print("Top processes: " + ", ".join(
        f"{p['name']}(pid={p['pid']}, cpu={p['cpu_percent']:.1f}%)" for p in gt["top_processes"]
    ))
    print(f"Ground truth healthy: {gt['healthy']}")

    corr_time, corr_type = check_suspend_or_reboot_correlation()
    if corr_type:
        print(f"\nCorrelation found: {corr_type} at {corr_time} (within the last 30 min)")
    else:
        print("\nNo suspend/resume or reboot correlation in the last 30 minutes.")

    pct_anom, declining = check_self_resolution(cfg["job"])
    if pct_anom is not None:
        trend = "declining (self-resolving)" if declining else "not declining"
        print(f"\nLast 10 min: {pct_anom:.0f}% anomalous, trend: {trend}")
    else:
        print("\nCould not retrieve recent history from Prometheus.")

    print("\n--- Verdict ---")
    if not gt["healthy"]:
        print("GROUND TRUTH DEGRADED. This may be a real incident, not a model artifact.")
        print("Recommendation: investigate directly, do not just retrain.")
    elif label == 0 and pct_anom is not None and pct_anom < 20:
        print("All clear. Not currently anomalous, low recent anomaly rate, ground truth healthy.")
        print("Recommendation: no action needed.")
    elif corr_type and pct_anom is not None and declining:
        print(f"Matches the known {corr_type} drift pattern and is already self-resolving.")
        print("Recommendation: no action needed, keep monitoring.")
    elif corr_type:
        print(f"Matches the known {corr_type} drift pattern, not yet resolving.")
        print("Recommendation: retrain on current post-event data.")
    elif pct_anom is not None and pct_anom > 80 and not declining:
        print("Sustained anomaly, no reboot/suspend correlation, ground truth healthy.")
        print("Recommendation: check the top-suspect process above for a known pattern "
              "(e.g. promtail); retrain if this persists.")
    else:
        print("No known pattern matched confidently.")
        print("Recommendation: needs human investigation.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 diagnose_anomaly.py knn|iforest|autoencoder")
        sys.exit(1)
    diagnose(sys.argv[1])
