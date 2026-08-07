#!/usr/bin/env python3
"""
build_flagship_dashboard.py

One-off script that constructs and creates the Guardian flagship dashboard
via Grafana's HTTP API. Not part of the running system -- a build tool,
run once (or re-run to update). Reads the same .grafana_token used by
grafana_annotate.py.
"""

import json
import os
import urllib.request

GRAFANA_URL = "http://127.0.0.1:3000"
PROM_DS = {"type": "prometheus", "uid": "depci0ud1uscga"}
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".grafana_token")

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()


def row(title, y):
    return {"type": "row", "title": title, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "collapsed": False}


def stat_panel(title, expr, x, y, w=6, h=6, thresholds=None, unit="short"):
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


def gauge_panel(title, expr, x, y, w=6, h=6):
    return {
        "type": "gauge",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": PROM_DS,
        "targets": [{"expr": expr, "datasource": PROM_DS, "refId": "A"}],
        "fieldConfig": {
            "defaults": {
                "unit": "short",
                "min": 0,
                "max": 100,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 60},
                        {"color": "green", "value": 85},
                    ],
                },
            },
            "overrides": [],
        },
    }


# ── Row 1: Guardian Status at a Glance ──────────────────────────────────────
panels = [row("Guardian Status at a Glance", 0)]
panels.append({
    "type": "stat",
    "title": "Guardian Status",
    "description": (
        "Worst of the three headline scores: min(Health, Security, AI Risk). "
        "<50 = Critical, 50-79 = Needs Attention, >=80 = Healthy. "
        "Check which of the three gauges is lowest to see what's driving this."
    ),
    "gridPos": {"h": 6, "w": 6, "x": 0, "y": 1},
    "datasource": PROM_DS,
    "targets": [{
        # min(a, b, c) isn't valid PromQL -- min() is an aggregation
        # operator, not a variadic scalar function. And a plain `or` union
        # of the three metrics doesn't work either: they share the same
        # instance/job labels, so Prometheus's `or` treats them as
        # duplicates and drops all but the first. label_replace() gives
        # each one a distinguishing label first, so all three survive the
        # union and min() aggregates across all three correctly.
        "expr": (
            'min('
            'label_replace(aiops_health_score, "metric", "health", "", "") or '
            'label_replace(aiops_security_score, "metric", "security", "", "") or '
            'label_replace(ai_risk_score, "metric", "ai_risk", "", "")'
            ')'
        ),
        "datasource": PROM_DS,
        "refId": "A",
    }],
    "fieldConfig": {
        "defaults": {
            "unit": "short",
            "thresholds": {
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 50},
                    {"color": "green", "value": 80},
                ],
            },
            "mappings": [
                {"type": "range", "options": {"from": 0, "to": 49.999, "result": {"text": "Critical", "color": "red"}}},
                {"type": "range", "options": {"from": 50, "to": 79.999, "result": {"text": "Needs Attention", "color": "yellow"}}},
                {"type": "range", "options": {"from": 80, "to": 100, "result": {"text": "Healthy", "color": "green"}}},
            ],
        },
        "overrides": [],
    },
    "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "textMode": "value"},
})
panels.append(gauge_panel("Health Score", "aiops_health_score", x=6, y=1))
panels.append(gauge_panel("Security Score", "aiops_security_score", x=12, y=1))
panels.append(gauge_panel("AI Risk Score", "ai_risk_score", x=18, y=1))

# ── Row 2: Priority Warnings ─────────────────────────────────────────────────
# Cross-source triage from aiops-watchdog-priority.py: currently-firing
# Prometheus ALERTS, Guardian's own health/security/AI-risk/Windows/log
# checks, and all_check.service, ranked by tier + whether each check just
# changed state vs. has been chronically bad -- see CLAUDE.md's
# aiops-watchdog-priority.py section for why raw warning-line counting
# doesn't work here (all_check.py's ssh check alone would swamp any such
# view). Sorted descending so the most important warning is always the
# top row; empty when nothing's currently flagged.
panels.append(row("Priority Warnings", 7))
panels.append({
    "type": "table",
    "title": "Top Priority Warnings (all sources, ranked)",
    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
    "datasource": PROM_DS,
    "targets": [{"expr": "aiops_priority_score", "datasource": PROM_DS, "refId": "A", "instant": True, "format": "table"}],
    "fieldConfig": {"defaults": {}, "overrides": []},
    "transformations": [{
        "id": "organize",
        "options": {
            "excludeByName": {"Time": True, "instance": True, "job": True, "__name__": True},
            "indexByName": {"tier": 0, "check": 1, "detail": 2, "Value": 3},
            "renameByName": {"Value": "Priority"},
        },
    }],
    "options": {"sortBy": [{"desc": True, "displayName": "Priority"}]},
})

# ── Row 3: Recent Events ─────────────────────────────────────────────────────
panels.append(row("Recent Events", 16))
panels.append({
    "type": "table",
    "title": "Currently Firing Alerts",
    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 17},
    "datasource": PROM_DS,
    "targets": [{"expr": 'ALERTS{alertstate="firing"}', "datasource": PROM_DS, "refId": "A", "instant": True, "format": "table"}],
    "fieldConfig": {"defaults": {}, "overrides": []},
})
panels.append({
    "type": "annolist",
    "title": "Guardian Events (Annotations)",
    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 17},
    "options": {"tags": ["guardian"], "limit": 20, "showUser": False, "showTime": True, "showTags": True},
})

# ── Row 4: Anomaly Detection: Model Agreement ────────────────────────────────
panels.append(row("Anomaly Detection: Model Agreement", 25))
model_thresholds = {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}
panels.append(stat_panel("KNN", 'aiops_anomaly_label{job="aiops-watchdog-knn"}', x=0, y=26, w=8, h=4, thresholds=model_thresholds))
panels.append(stat_panel("Isolation Forest", 'aiops_anomaly_label{job="aiops-watchdog-iforest"}', x=8, y=26, w=8, h=4, thresholds=model_thresholds))
panels.append(stat_panel("Autoencoder", 'aiops_anomaly_label{job="aiops-watchdog-autoencoder"}', x=16, y=26, w=8, h=4, thresholds=model_thresholds))
panels.append({
    "type": "timeseries",
    "title": "Model Agreement Over Time",
    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 30},
    "datasource": PROM_DS,
    "targets": [
        {"expr": 'aiops_anomaly_label{job="aiops-watchdog-knn"}', "datasource": PROM_DS, "refId": "A", "legendFormat": "KNN"},
        {"expr": 'aiops_anomaly_label{job="aiops-watchdog-iforest"}', "datasource": PROM_DS, "refId": "B", "legendFormat": "Isolation Forest"},
        {"expr": 'aiops_anomaly_label{job="aiops-watchdog-autoencoder"}', "datasource": PROM_DS, "refId": "C", "legendFormat": "Autoencoder"},
    ],
    "fieldConfig": {"defaults": {"unit": "short", "min": 0, "max": 1}, "overrides": []},
})

# ── Row 5: Security Detail ───────────────────────────────────────────────────
panels.append(row("Security Detail", 38))
panels.append(stat_panel("UFW Enabled", "aiops_security_ufw_enabled", x=0, y=39, w=5, h=5,
                          thresholds={"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}))
panels.append(stat_panel("Open Ports", "aiops_security_open_ports_count", x=5, y=39, w=5, h=5))
panels.append(stat_panel("Failed Logins (recent)", "aiops_security_failed_logins_recent", x=10, y=39, w=5, h=5))
panels.append(stat_panel("Root SSH Enabled", "aiops_security_root_ssh_enabled", x=15, y=39, w=4, h=5,
                          thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}))
panels.append(stat_panel("Updates Pending", "aiops_security_updates_pending", x=19, y=39, w=5, h=5))

# ── Row 6: Go Deeper ──────────────────────────────────────────────────────────
panels.append(row("Go Deeper", 44))
panels.append({
    "type": "text",
    "title": "Component Dashboards",
    "gridPos": {"h": 6, "w": 24, "x": 0, "y": 45},
    "options": {
        "mode": "markdown",
        "content": (
            "For deeper investigation, see the component dashboards: "
            "[AI Anomaly Detection](/d/ad6lc4j/ai-anomaly-detection) | "
            "[Node Exporter Full](/d/rYdddlPWk/node-exporter-full) | "
            "[System Metrics + Linked Logs](/d/metrics-logs/system-metrics-2b-linked-logs) | "
            "[Loki Log Dashboard](/d/loki-logs/loki-log-dashboard)"
        ),
    },
})

dashboard = {
    "dashboard": {
        "id": None,
        "uid": "guardian-flagship",  # fixed UID so re-running this script updates in place, not a new dashboard
        "title": "Guardian: Flagship Overview",
        "tags": ["guardian", "flagship"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 0,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "annotations": {
            "list": [{
                "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                "enable": True,
                "iconColor": "red",
                "name": "Guardian Events",
                "tags": ["guardian"],
                "type": "tags",
            }]
        },
        "panels": panels,
    },
    "overwrite": True,
}

req = urllib.request.Request(
    f"{GRAFANA_URL}/api/dashboards/db",
    data=json.dumps(dashboard).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    result = json.loads(resp.read())
    print(json.dumps(result, indent=2))
    if result.get("url"):
        print(f"\nDashboard URL: {GRAFANA_URL}{result['url']}")
