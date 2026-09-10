#!/usr/bin/env python3
"""
fix_dashboard_edge_dedup.py

One-off fix for the "Guardian Metrics - Cross Platform - Windows" Grafana
dashboard (uid ad8xfcb): every panel's PromQL query was written against raw
windows_exporter metric names with no edge_site filter. That was invisible
while only one host (DESKTOP-0AJUKU3) was forwarded through the Pi edge
collector (EDGE_ARCHITECTURE.md M1) -- each panel just showed one duplicate
value per metric, easy to miss. Once a second host (DESKTOP-503POVP) was
also put on the edge path (2026-09-10), every panel started visibly showing
2x the expected number of series per host: the direct Core scrape
(edge_site="") and the Pi-forwarded copy (edge_site="guardian-proto-1") of
the exact same underlying metric.

Same dedup principle aiops-watchdog-windows.py's discover_instances()
already applies (filtering to edge_site="" so the same physical host isn't
double-counted) -- this script applies the equivalent fix to every panel's
query on this dashboard, not just the watchdog's own internal gauges.

Fetches the live dashboard first (same pattern as build_windows_dashboard.py)
rather than hand-editing a stale local copy.
"""

import json
import os
import re
import urllib.request

GRAFANA_URL = "http://127.0.0.1:3000"
DASHBOARD_UID = "ad8xfcb"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".grafana_token")

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()


def _get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def add_edge_site_filter(expr):
    """Insert edge_site="" into every metric selector in a PromQL expression.
    Metrics with an existing {...} label selector get edge_site="" appended
    inside it; bare metric names (no selector at all) get one added."""

    def replace_with_braces(m):
        name, inner = m.group(1), m.group(2)
        if "edge_site" in inner:
            return m.group(0)
        sep = "," if inner.strip() else ""
        return f'{name}{{{inner}{sep}edge_site=""}}'

    # Metric name followed by an existing {...} selector.
    expr = re.sub(r'\b([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}', replace_with_braces, expr)

    # Bare metric names with no selector at all (not already handled above,
    # and not a PromQL function/keyword like avg, rate, by, sum, time).
    keywords = {"avg", "by", "rate", "sum", "count", "time", "instance"}

    def replace_bare(m):
        name = m.group(1)
        if name in keywords or name.startswith("windows_") is False:
            return name
        return f'{name}{{edge_site=""}}'

    expr = re.sub(r'\b(windows_[a-zA-Z0-9_:]*)\b(?!\{)', replace_bare, expr)
    return expr


existing = _get(f"{GRAFANA_URL}/api/dashboards/uid/{DASHBOARD_UID}")
dashboard = existing["dashboard"]

changed = []
for panel in dashboard["panels"]:
    for target in panel.get("targets", []):
        expr = target.get("expr")
        if not expr:
            continue
        new_expr = add_edge_site_filter(expr)
        if new_expr != expr:
            changed.append((panel["title"], expr, new_expr))
            target["expr"] = new_expr

print(f"Updating {len(changed)} panel queries:")
for title, old, new in changed:
    print(f"  {title!r}:\n    old: {old}\n    new: {new}")

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
