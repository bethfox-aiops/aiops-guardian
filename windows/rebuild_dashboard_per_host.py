#!/usr/bin/env python3
"""
rebuild_dashboard_per_host.py

Restructures the "Guardian Metrics - Cross Platform - Windows" dashboard
(uid ad8xfcb) from "one panel per metric type, both hosts shown together"
into "one column per host, all that host's metrics grouped together" --
prompted by a ChatGPT-generated mockup image (2026-09-10) showing that
layout. Each panel's exact fieldConfig/options/thresholds are cloned from
the live dashboard (fetched first, not hand-guessed) so formatting doesn't
drift from what's already tuned; only the PromQL gets an instance="..."
filter added and the gridPos gets repositioned into that host's column.

One real change beyond a layout reshuffle: the mockup showed a Network
Throughput "(In)"/"(Out)" split, but that was the image generator
mislabeling the *existing* single combined-throughput panel's two
per-host values as one host's in/out split -- not a real measurement.
This script builds a genuine in/out split using
windows_net_bytes_received_total / windows_net_bytes_sent_total, which
windows_exporter actually exposes separately.

Grafana has no native "bordered box grouping multiple panels" primitive
(a Row panel is a full-width divider, not a column container), so the
per-host grouping is approximated with a colored HTML header panel per
column plus positional grouping (left column / right column), not an
actual border running around every panel in that host's block.

Fetches the live dashboard first (same pattern as build_windows_dashboard.py)
so this starts from what's actually deployed, not a guess.
"""

import copy
import json
import os
import urllib.request

GRAFANA_URL = "http://127.0.0.1:3000"
DASHBOARD_UID = "ad8xfcb"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".grafana_token")

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()

HOSTS = [
    {"instance": "DESKTOP-0AJUKU3:9182", "short": "DESKTOP-0AJUKU3", "color": "#3B82F6", "x": 0},
    {"instance": "DESKTOP-503POVP:9182", "short": "DESKTOP-503POVP", "color": "#10B981", "x": 12},
]


def _get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def find_panel(panels, title):
    for p in panels:
        if p["title"] == title:
            return p
    raise KeyError(title)


def cloned(panel, title, x, y, w, h, instance, extra_filter=""):
    """Deep-copy an existing panel, point its query(ies) at one instance,
    and reposition it. extra_filter is inserted alongside instance= for
    panels whose expr needs another label added too."""
    p = copy.deepcopy(panel)
    p.pop("id", None)
    p["title"] = title
    p["gridPos"] = {"h": h, "w": w, "x": x, "y": y}
    for t in p["targets"]:
        t["expr"] = t["expr"].replace(
            'edge_site=""', f'edge_site="",instance="{instance}"{extra_filter}'
        )
    return p


def header_panel(host, x):
    return {
        "type": "text",
        "title": "",
        "gridPos": {"h": 2, "w": 12, "x": x, "y": 0},
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {
            "mode": "html",
            "content": (
                f'<div style="border:2px solid {host["color"]}; border-radius:6px; '
                f'padding:6px 16px; height:100%; box-sizing:border-box; display:flex; '
                f'align-items:center;">'
                f'<h2 style="margin:0; color:{host["color"]};">{host["short"]}</h2>'
                f"</div>"
            ),
        },
    }


def network_inout_panel(base, x, y, w, h, instance):
    p = copy.deepcopy(base)
    p.pop("id", None)
    p["title"] = "Network Throughput"
    p["gridPos"] = {"h": h, "w": w, "x": x, "y": y}
    template_target = p["targets"][0]
    in_target = copy.deepcopy(template_target)
    in_target["expr"] = f'sum by (instance) (rate(windows_net_bytes_received_total{{edge_site="",instance="{instance}"}}[5m]))'
    in_target["legendFormat"] = "In"
    in_target["refId"] = "A"
    out_target = copy.deepcopy(template_target)
    out_target["expr"] = f'sum by (instance) (rate(windows_net_bytes_sent_total{{edge_site="",instance="{instance}"}}[5m]))'
    out_target["legendFormat"] = "Out"
    out_target["refId"] = "B"
    p["targets"] = [in_target, out_target]
    return p


existing = _get(f"{GRAFANA_URL}/api/dashboards/uid/{DASHBOARD_UID}")
dashboard = existing["dashboard"]
old_panels = dashboard["panels"]

templates = {
    "cpu": find_panel(old_panels, "Windows CPU"),
    "memory": find_panel(old_panels, "Windows Free Memory"),
    "disk_space": find_panel(old_panels, "Windows Free Disk Space (C:)"),
    "disk_busy": find_panel(old_panels, "Windows Physical Disk Busy %"),
    "services": find_panel(old_panels, "Windows Services Not Running"),
    "processes": find_panel(old_panels, "Windows Process Count"),
    "network": find_panel(old_panels, "Windows Network Throughput"),
    "freshness": find_panel(old_panels, "Windows Disk Health Collector Freshness"),
    "os_info": find_panel(old_panels, "Windows OS Info"),
    "disk_temp": find_panel(old_panels, "Windows Disk Temperature (°C)"),
    "disk_wear": find_panel(old_panels, "Windows Disk Wear %"),
    "reliability": find_panel(old_panels, "Windows Disk Reliability Events (24h)"),
}

# CPU Usage looks better as a gauge (matching Memory/Disk Space/Disk Busy%'s
# ring style) than the flat stat number it is today -- same visual family
# as every other percentage panel in this dashboard.
cpu_gauge = copy.deepcopy(templates["disk_busy"])
cpu_gauge["targets"] = copy.deepcopy(templates["cpu"]["targets"])

new_panels = []
for host in HOSTS:
    x, inst = host["x"], host["instance"]
    new_panels.append(header_panel(host, x))

    row1_y = 2
    new_panels.append(cloned(cpu_gauge, "CPU Usage", x, row1_y, 4, 4, inst))
    new_panels.append(cloned(templates["memory"], "Memory Free", x + 4, row1_y, 4, 4, inst))
    new_panels.append(cloned(templates["disk_space"], "C: Free Space", x + 8, row1_y, 4, 4, inst))

    row2_y = row1_y + 4
    new_panels.append(cloned(templates["disk_busy"], "Disk Busy %", x, row2_y, 4, 4, inst))
    new_panels.append(cloned(templates["services"], "Services Not Running", x + 4, row2_y, 4, 4, inst))
    new_panels.append(cloned(templates["processes"], "Process Count", x + 8, row2_y, 4, 4, inst))

    row3_y = row2_y + 4
    new_panels.append(network_inout_panel(templates["network"], x, row3_y, 6, 4, inst))
    new_panels.append(cloned(templates["freshness"], "Disk Health Collector Freshness", x + 6, row3_y, 6, 4, inst))

    row4_y = row3_y + 4
    new_panels.append(cloned(templates["os_info"], "OS Info", x, row4_y, 4, 4, inst))
    new_panels.append(cloned(templates["disk_temp"], "Disk Temperature (°C)", x + 4, row4_y, 4, 4, inst))
    new_panels.append(cloned(templates["disk_wear"], "Disk Wear %", x + 8, row4_y, 4, 4, inst))

    row5_y = row4_y + 4
    new_panels.append(cloned(templates["reliability"], "Disk Reliability Events (24h)", x, row5_y, 12, 6, inst))

dashboard["panels"] = new_panels

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
