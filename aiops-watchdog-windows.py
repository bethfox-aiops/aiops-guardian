#!/usr/bin/env python3
"""
aiops-watchdog-windows.py

Guardian's Windows health watchdog. Unlike the Linux watchdogs/guardian
services (which collect metrics locally via psutil), Windows hosts are
already instrumented by windows_exporter and scraped by Prometheus under
job="windows-node" -- this script's only job is to query that existing
data, apply the same kind of threshold-based health scoring
aiops-guardian-health.py does for this Linux host, and re-expose the
result as its own Prometheus gauges (one series per Windows instance).

- Discovers Windows instances from `up{job="windows-node"}` every cycle,
  so a newly added or removed host picks up automatically.
- For each reachable instance, checks:
    cpu_ok      busy% < 85          (windows_cpu_time_total)
    mem_ok      used% < 85          (windows_memory_available_bytes /
                                      windows_memory_physical_total_bytes)
    disk_ok     C: used% < 90       (windows_logical_disk_{free,size}_bytes)
    service_ok  no crashed auto-start services
                                    (windows_service_state / _start_mode)
  and rolls them into aiops_windows_health_score (100, -25 per failed
  check), mirroring compute_health()'s 100/-20-per-check pattern.
- Unreachable instances (up == 0) get health_score=0 and every _ok gauge
  cleared to 0 rather than silently dropping out of the metrics -- an
  unreachable host is not a passing check.

Exposes Prometheus metrics on WATCHDOG_PORT (default: 8016):
    aiops_windows_health_up{instance}
    aiops_windows_health_cpu_ok{instance}
    aiops_windows_health_mem_ok{instance}
    aiops_windows_health_disk_ok{instance}
    aiops_windows_health_service_ok{instance}
    aiops_windows_health_score{instance}
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from prometheus_client import Gauge, start_http_server

PROM_URL = "http://127.0.0.1:9090"
JOB = "windows-node"
DISK_VOLUME = "C:"

PORT = int(os.getenv("WATCHDOG_PORT", "8016"))
INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "30.0"))

g_up          = Gauge("aiops_windows_health_up",         "1 if the windows_exporter target is reachable", ["instance"])
g_cpu_ok      = Gauge("aiops_windows_health_cpu_ok",      "CPU health: 1=healthy (<85% busy), 0=unhealthy", ["instance"])
g_mem_ok      = Gauge("aiops_windows_health_mem_ok",      "Memory health: 1=healthy (<85% used), 0=unhealthy", ["instance"])
g_disk_ok     = Gauge("aiops_windows_health_disk_ok",     f"Disk health on {DISK_VOLUME}: 1=healthy (<90% used), 0=unhealthy", ["instance"])
g_service_ok  = Gauge("aiops_windows_health_service_ok",  "Service health: 1=healthy (no crashed auto-start services), 0=unhealthy", ["instance"])
g_health_score = Gauge("aiops_windows_health_score",      "Overall Windows host health score (0-100)", ["instance"])


def prom_query_vector(expr):
    """Returns a list of (labels_dict, float_value) for a PromQL instant query."""
    url = f"{PROM_URL}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[WARN] Prometheus query failed: {e}", flush=True)
        return []
    result = data.get("data", {}).get("result", [])
    return [(r["metric"], float(r["value"][1])) for r in result]


def prom_query_by_instance(expr):
    """Convenience wrapper for queries expected to return one series per
    instance: {instance: value}."""
    return {labels.get("instance"): value for labels, value in prom_query_vector(expr)}


def discover_instances():
    """Returns {instance: is_up_bool} for every configured windows-node target."""
    return {
        instance: (value == 1)
        for instance, value in prom_query_by_instance(f'up{{job="{JOB}"}}').items()
    }


def check_cpu(instance):
    busy = prom_query_by_instance(
        f'100 - (avg by (instance) (rate(windows_cpu_time_total{{job="{JOB}",instance="{instance}",mode="idle"}}[5m])) * 100)'
    )
    return busy.get(instance, 100.0) < 85


def check_mem(instance):
    avail = prom_query_by_instance(f'windows_memory_available_bytes{{job="{JOB}",instance="{instance}"}}')
    total = prom_query_by_instance(f'windows_memory_physical_total_bytes{{job="{JOB}",instance="{instance}"}}')
    a, t = avail.get(instance), total.get(instance)
    if not a or not t:
        return True  # can't evaluate -- don't flag unhealthy on missing data
    used_pct = 100 * (1 - a / t)
    return used_pct < 85


def check_disk(instance):
    free = prom_query_by_instance(f'windows_logical_disk_free_bytes{{job="{JOB}",instance="{instance}",volume="{DISK_VOLUME}"}}')
    size = prom_query_by_instance(f'windows_logical_disk_size_bytes{{job="{JOB}",instance="{instance}",volume="{DISK_VOLUME}"}}')
    f, s = free.get(instance), size.get(instance)
    if not f or not s:
        return True
    used_pct = 100 * (1 - f / s)
    return used_pct < 90



# Deliberately NOT "any service with start_mode=auto that isn't running":
# Windows has many services (Google Updater, AppXSvc, MapsBroker, sppsvc,
# edgeupdate, gpsvc, ...) registered start_mode=auto but using
# trigger-start semantics -- idle/stopped is their normal resting state,
# not a crash. windows_exporter doesn't expose that distinction, so
# instead check a small allowlist of services that are genuinely always
# running on a healthy Windows host.
CRITICAL_SERVICES = ["RpcSs", "EventLog", "Dnscache", "LanmanWorkstation"]


def check_services(instance):
    """Healthy if all CRITICAL_SERVICES are running."""
    names = "|".join(CRITICAL_SERVICES)
    running = prom_query_by_instance(
        f'count by (instance) (windows_service_state{{job="{JOB}",instance="{instance}",'
        f'name=~"{names}",state="running"}} == 1)'
    )
    return running.get(instance, 0) == len(CRITICAL_SERVICES)


def compute_health(instance):
    cpu_ok = check_cpu(instance)
    mem_ok = check_mem(instance)
    disk_ok = check_disk(instance)
    service_ok = check_services(instance)

    score = 100
    for ok in (cpu_ok, mem_ok, disk_ok, service_ok):
        if not ok:
            score -= 25

    g_cpu_ok.labels(instance=instance).set(int(cpu_ok))
    g_mem_ok.labels(instance=instance).set(int(mem_ok))
    g_disk_ok.labels(instance=instance).set(int(disk_ok))
    g_service_ok.labels(instance=instance).set(int(service_ok))
    g_health_score.labels(instance=instance).set(max(score, 0))

    print(
        f"[INFO] {instance}: score={score} cpu_ok={int(cpu_ok)} mem_ok={int(mem_ok)} "
        f"disk_ok={int(disk_ok)} service_ok={int(service_ok)}",
        flush=True,
    )


def mark_unreachable(instance):
    g_cpu_ok.labels(instance=instance).set(0)
    g_mem_ok.labels(instance=instance).set(0)
    g_disk_ok.labels(instance=instance).set(0)
    g_service_ok.labels(instance=instance).set(0)
    g_health_score.labels(instance=instance).set(0)
    print(f"[WARN] {instance}: unreachable (up=0), health_score=0", flush=True)


def main():
    print(f"[INFO] Starting Windows watchdog on port {PORT}", flush=True)
    print(f"[INFO] Interval: {INTERVAL} seconds", flush=True)
    start_http_server(PORT)
    print(f"[INFO] Prometheus metrics available on :{PORT}", flush=True)

    while True:
        instances = discover_instances()
        if not instances:
            print(f'[WARN] No targets found for job="{JOB}" -- is Prometheus scraping windows-node?', flush=True)

        for instance, up in instances.items():
            g_up.labels(instance=instance).set(int(up))
            if up:
                compute_health(instance)
            else:
                mark_unreachable(instance)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
