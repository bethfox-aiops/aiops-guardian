#!/usr/bin/env python3
"""
guardian_health.py

Guardian's health engine: CPU/mem/disk/inode/service checks, aggregated
into the aiops_health_* gauges and aiops_health_score.
"""

import os
import shutil
import subprocess

import psutil
from prometheus_client import Gauge

# ─── Health gauges ───────────────────────────────────────────────────────────
cpu_ok        = Gauge("aiops_health_cpu_ok",      "CPU health: 1=healthy, 0=unhealthy")
mem_ok        = Gauge("aiops_health_mem_ok",      "Memory health: 1=healthy, 0=unhealthy")
disk_ok       = Gauge("aiops_health_disk_ok",     "Disk health: 1=healthy, 0=unhealthy")
inode_ok      = Gauge("aiops_health_inode_ok",    "Inode health: 1=healthy, 0=unhealthy")
service_ok    = Gauge("aiops_health_service_ok",  "Service health aggregate: 1=healthy, 0=unhealthy")
health_score  = Gauge("aiops_health_score",       "Overall system health score (0-100)")


# ════════════════════════════════════════════════════════════════════════════
# Existing health checks
# ════════════════════════════════════════════════════════════════════════════

def check_service_active(service_name: str) -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service_name], check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def get_inode_usage_percent(path: str = "/") -> float:
    stats = os.statvfs(path)
    if stats.f_files <= 0:
        return 0.0
    return ((stats.f_files - stats.f_ffree) / stats.f_files) * 100.0


def compute_health():
    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory().percent
    disk = shutil.disk_usage("/").used / shutil.disk_usage("/").total * 100.0
    inode = get_inode_usage_percent("/")

    cpu_state   = 1 if cpu   < 85 else 0
    mem_state   = 1 if mem   < 85 else 0
    disk_state  = 1 if disk  < 90 else 0
    inode_state = 1 if inode < 90 else 0

    services = ["prometheus", "grafana-server", "loki"]
    service_state = 1 if all(check_service_active(s) for s in services) else 0

    score = 100
    if cpu_state == 0:
        score -= 20
    if mem_state == 0:
        score -= 20
    if disk_state == 0:
        score -= 20
    if inode_state == 0:
        score -= 20
    if service_state == 0:
        score -= 20

    cpu_ok.set(cpu_state)
    mem_ok.set(mem_state)
    disk_ok.set(disk_state)
    inode_ok.set(inode_state)
    service_ok.set(service_state)
    health_score.set(max(score, 0))
