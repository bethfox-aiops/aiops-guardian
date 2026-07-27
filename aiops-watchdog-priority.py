#!/usr/bin/env python3
"""
aiops-watchdog-priority.py

Guardian's cross-source warning triage. Pulls together every "something's
wrong" signal already exposed as a Prometheus gauge -- Guardian's own
health/security/AI-risk/Windows/log-anomaly checks, and the older,
looser all_check.service (/opt/aiops/all_check.py, outside this repo) --
and ranks them so the most important ones surface first, rather than
just listing raw warning counts.

The key design problem this solves: raw warning-line counting is
dominated by chronic noise. all_check.py's `check_service_status("ssh")`
prints a "Service ssh is not running" warning every ~15s forever, because
sshd was never installed on this host -- not a real problem, just a
constant false alarm. A naive "count warnings" view would be swamped by
this single source.

Instead: every check here already has a backing Prometheus gauge, so
priority is computed from whether that gauge just changed state
(`changes(gauge[6h])`), not how often it's been printed. A gauge that's
been stuck at the same "bad" value for 6h+ (chronic, like the SSH check)
gets no novelty bonus; one that just flipped gets the full bonus. Combined
with a per-tier base weight (currently-firing Prometheus ALERTS, already
vetted by a 30-minute sustain requirement, always rank highest; Guardian's
own disciplined checks next; all_check.py's looser, un-debounced checks
last), this naturally surfaces "something just broke" over "something's
always been slightly off" without needing a hand-maintained exclusion list.

Exposes Prometheus metrics on WATCHDOG_PORT (default: 8018):
    aiops_priority_score{check, detail, tier}
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from prometheus_client import Gauge, start_http_server

PROM_URL = "http://127.0.0.1:9090"
NOVELTY_LOOKBACK = "6h"

PORT = int(os.getenv("WATCHDOG_PORT", "8018"))
INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "60.0"))

# (check name, tier, base weight, PromQL selector that returns ONLY
# currently-bad series). Each selector's own labels (plus __name__) become
# the "detail" string for that series once it fires.
TIER_ALERTS = "prometheus_alert"
TIER_GUARDIAN = "guardian"
TIER_ALL_CHECK = "all_check"

BASE_WEIGHT = {
    TIER_ALERTS: 100,    # already vetted: 30-min sustain + severity label
    TIER_GUARDIAN: 60,   # this repo's own disciplined checks
    TIER_ALL_CHECK: 30,  # older, un-debounced, at least one known chronic false positive (ssh)
}

# (tier, name, selector, comparator) -- kept separate rather than one
# combined PromQL string because `offset` must be applied directly after
# the bare selector, before any comparison (`sel offset 6h == 0`, not
# `(sel == 0) offset 6h` -- the latter is a PromQL parse error), and
# novelty_bonus() needs to build that offset query itself.
CHECKS = [
    # -- Guardian: health --
    (TIER_GUARDIAN, "health_cpu",       'aiops_health_cpu_ok',      '== 0'),
    (TIER_GUARDIAN, "health_mem",       'aiops_health_mem_ok',      '== 0'),
    (TIER_GUARDIAN, "health_disk",      'aiops_health_disk_ok',     '== 0'),
    (TIER_GUARDIAN, "health_inode",     'aiops_health_inode_ok',    '== 0'),
    (TIER_GUARDIAN, "health_service",   'aiops_health_service_ok',  '== 0'),
    # -- Guardian: security --
    (TIER_GUARDIAN, "security_issue",   'aiops_security_issue_code', '> 0'),
    # -- Guardian: AI risk --
    (TIER_GUARDIAN, "ai_watchdog_external", 'ai_watchdog_port_external_access', '== 1'),
    (TIER_GUARDIAN, "ai_exposed_keys",      'ai_exposed_api_keys',              '> 0'),
    (TIER_GUARDIAN, "ai_llm_connections",   'ai_outbound_llm_connections',      '> 0'),
    (TIER_GUARDIAN, "ai_shadow_models",     'ai_shadow_model_count',            '> 0'),
    (TIER_GUARDIAN, "ai_training_changed",  'ai_training_data_hash_changed',    '== 1'),
    (TIER_GUARDIAN, "ai_model_drift",       'ai_model_file_age_drift',          '== 1'),
    (TIER_GUARDIAN, "ai_gpu_spike",         'ai_gpu_spike_no_known_workload',   '== 1'),
    # -- Guardian: Windows hosts --
    (TIER_GUARDIAN, "windows_health",   'aiops_windows_health_score', '< 100'),
    # -- Guardian: log anomalies --
    (TIER_GUARDIAN, "logs_error",       'aiops_logs_error_count', '> 0'),
    (TIER_GUARDIAN, "logs_silent",      'aiops_logs_silent',      '== 1'),
    # -- all_check.service (older, looser checks; see module docstring) --
    (TIER_ALL_CHECK, "allcheck_cpu",        'aiops_anomaly_cpu{job="all_check_agent"}',            '== 1'),
    (TIER_ALL_CHECK, "allcheck_memory",     'aiops_anomaly_memory{job="all_check_agent"}',         '== 1'),
    (TIER_ALL_CHECK, "allcheck_disk",       'aiops_anomaly_disk_usage{job="all_check_agent"}',     '== 1'),
    (TIER_ALL_CHECK, "allcheck_process",    'aiops_anomaly_critical_process{job="all_check_agent"}', '== 1'),
    (TIER_ALL_CHECK, "allcheck_load",       'aiops_anomaly_load{job="all_check_agent"}',           '== 1'),
    (TIER_ALL_CHECK, "allcheck_network",    'network_latency_issue{job="all_check_agent"}',        '== 1'),
    (TIER_ALL_CHECK, "allcheck_service",    'aiops_service_status{job="all_check_agent"}',         '== 0'),
]

DETAIL_LABELS = ["instance", "unit", "service", "mountpoint", "process", "exported_instance"]

g_priority = Gauge(
    "aiops_priority_score",
    "Cross-source warning priority (0-120ish): tier base weight + novelty bonus "
    "for gauges that changed state recently vs. chronic/unchanging bad state",
    ["check", "detail", "tier"],
)


def prom_query(expr):
    """Returns a list of (labels_dict, float_value) for a PromQL instant query."""
    url = f"{PROM_URL}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[WARN] Prometheus query failed: {e}", flush=True)
        return []
    result = data.get("data", {}).get("result", [])
    return [(r["metric"], float(r["value"][1])) for r in result]


def detail_string(labels):
    parts = [f"{k}={labels[k]}" for k in DETAIL_LABELS if k in labels]
    return ",".join(parts) if parts else "-"


def novelty_bonus(selector, comparator, target_labels):
    """20 if this exact series was healthy (or didn't exist) at
    NOVELTY_LOOKBACK ago and is bad now -- a genuinely new problem. 0 if it
    was already bad back then too (chronic, e.g. the all_check.py ssh
    check, which has been "not running" for as long as this metric has
    existed). 10 (neutral) if there's no data at all that far back to
    compare against -- e.g. right after a Prometheus restart, before the
    head block has been running long enough; don't guess either way when
    we genuinely can't tell.

    changes(metric[window]) was tried first and rejected: it can't tell
    "flipped once and has stayed bad since" apart from "was already bad
    before the window even started" -- both produce a low change count,
    so it scored the chronic ssh case at the *maximum* bonus instead of
    zero. offset must be applied directly after the bare selector, before
    the comparison (`sel offset 6h == 0`) -- `(sel == 0) offset 6h` is a
    PromQL parse error.
    """
    target = {k: v for k, v in target_labels.items() if k != "__name__"}

    bad_then = [
        {k: v for k, v in labels.items() if k != "__name__"}
        for labels, _ in prom_query(f"{selector} offset {NOVELTY_LOOKBACK} {comparator}")
    ]
    if target in bad_then:
        return 0  # confirmed bad back then too -- chronic

    any_data_then = prom_query(f"{selector} offset {NOVELTY_LOOKBACK}")
    if any({k: v for k, v in labels.items() if k != "__name__"} == target for labels, _ in any_data_then):
        return 20  # confirmed healthy back then, bad now -- genuinely new

    return 10  # no data that far back for this series -- don't guess


def check_firing_alerts():
    for labels, _ in prom_query('ALERTS{alertstate="firing"}'):
        name = labels.get("alertname", "unknown_alert")
        detail = detail_string(labels) or labels.get("severity", "-")
        g_priority.labels(check=name, detail=detail, tier=TIER_ALERTS).set(BASE_WEIGHT[TIER_ALERTS])
        print(f"[INFO] priority={BASE_WEIGHT[TIER_ALERTS]} tier={TIER_ALERTS} check={name} detail={detail}", flush=True)


def check_gauges():
    for tier, name, selector, comparator in CHECKS:
        for labels, _ in prom_query(f"{selector} {comparator}"):
            bonus = novelty_bonus(selector, comparator, labels)
            score = BASE_WEIGHT[tier] + bonus
            detail = detail_string(labels)
            g_priority.labels(check=name, detail=detail, tier=tier).set(score)
            print(f"[INFO] priority={score} tier={tier} check={name} detail={detail} novelty_bonus={bonus}", flush=True)


def main():
    print(f"[INFO] Starting priority-triage watchdog on port {PORT}", flush=True)
    print(f"[INFO] Interval: {INTERVAL} seconds, novelty lookback: {NOVELTY_LOOKBACK}", flush=True)
    start_http_server(PORT)
    print(f"[INFO] Prometheus metrics available on :{PORT}", flush=True)

    while True:
        g_priority.clear()  # stale (now-healthy) checks shouldn't linger as old series
        check_firing_alerts()
        check_gauges()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
