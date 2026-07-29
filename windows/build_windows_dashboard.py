#!/usr/bin/env python3
"""
build_windows_dashboard.py

Build script for the "Guardian Metrics - Cross Platform - Windows" Grafana
dashboard (uid ad8xfcb). Not part of the running system -- a build tool, run
once (or re-run to update), same pattern as talks/build_flagship_dashboard.py.
Reads the same .grafana_token.

This dashboard had no tracked JSON or builder script at all before this file
-- it was built directly in the Grafana UI at some point and never version
controlled. Fetches the current live dashboard first (so this script's own
history starts from what's actually deployed, not a guess) rather than
hand-reconstructing it from scratch.
"""

import json
import os
import urllib.request

GRAFANA_URL = "http://127.0.0.1:3000"
DASHBOARD_UID = "ad8xfcb"
PROM_DS = {"type": "prometheus", "uid": "depci0ud1uscga"}
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".grafana_token")

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()


def _get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def gauge_panel(title, expr, x, y, w=8, h=4, unit="short", thresholds=None, max_val=100):
    return {
        "type": "gauge",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": PROM_DS,
        "targets": [{"expr": expr, "datasource": PROM_DS, "refId": "A"}],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": 0,
                "max": max_val,
                "thresholds": thresholds or {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 60}, {"color": "red", "value": 85}],
                },
            },
            "overrides": [],
        },
    }


def stat_panel(title, expr, x, y, w=8, h=4, unit="short", thresholds=None):
    return {
        "type": "stat",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": PROM_DS,
        "targets": [{"expr": expr, "datasource": PROM_DS, "refId": "A"}],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": thresholds or {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            },
            "overrides": [],
        },
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "textMode": "auto"},
    }


existing = _get(f"{GRAFANA_URL}/api/dashboards/uid/{DASHBOARD_UID}")
dashboard = existing["dashboard"]
panels = dashboard["panels"]

# New row, added 2026-07-29 in response to the DESKTOP-0AJUKU3 drive-failure
# investigation -- disk-health data from windows/guardian_disk_health.ps1
# (via windows_exporter's textfile collector) had no panels anywhere before
# this. Continues the existing 2-column, w=8/h=4 layout; last existing row
# ends at y=12, so this starts at y=16.
new_panels = [
    gauge_panel(
        "Windows Disk Temperature (°C)", "windows_disk_reliability_temperature_celsius",
        x=0, y=16, unit="celsius", max_val=90,
        thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 60}, {"color": "red", "value": 75}]},
    ),
    gauge_panel(
        "Windows Disk Wear %", "windows_disk_reliability_wear_percent",
        x=8, y=16, unit="percent", max_val=100,
        thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 50}, {"color": "red", "value": 80}]},
    ),
    {
        "type": "table",
        "title": "Windows Disk Reliability Events (24h)",
        "gridPos": {"h": 4, "w": 8, "x": 0, "y": 20},
        "datasource": PROM_DS,
        "targets": [{"expr": "windows_disk_event_count", "datasource": PROM_DS, "refId": "A", "instant": True, "format": "table"}],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "transformations": [{
            "id": "organize",
            "options": {
                "excludeByName": {"Time": True, "instance": True, "job": True, "__name__": True},
                "indexByName": {"event_id": 0, "Value": 1},
                "renameByName": {"Value": "Count (24h)", "event_id": "Event ID (7=bad block, 11=controller err, 15=not ready, 51=delayed write, 153=IO error)"},
            },
        }],
        "options": {"sortBy": [{"desc": True, "displayName": "Count (24h)"}]},
    },
    stat_panel(
        "Windows Disk Health Collector Freshness", "time() - windows_disk_health_collector_last_run_timestamp_seconds",
        x=8, y=20, unit="s",
        thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 1200}, {"color": "red", "value": 2400}]},
    ),
]

dashboard["panels"] = panels + new_panels

payload = {"dashboard": dashboard, "overwrite": True}

req = urllib.request.Request(
    f"{GRAFANA_URL}/api/dashboards/db",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    result = json.loads(resp.read())
    print(json.dumps(result, indent=2))
    if result.get("url"):
        print(f"\nDashboard URL: {GRAFANA_URL}{result['url']}")
