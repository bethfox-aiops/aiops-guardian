#!/usr/bin/env python3
"""
generate_report.py

Guardian's automated interpretation layer: pulls live state from
Prometheus and Grafana, compares it against the previous run's snapshot,
runs a small rule-based "playbook" engine to interpret what the numbers
mean, and renders a markdown report -- the report_state.json / rule-based
version of what previously required an AI chat session to synthesize by
hand each time (see full_system_report_2026-07-13.md for the manually
written precedent this reuses the structure of).

Manual for now: run `python3 generate_report.py`. Not wired to a systemd
timer yet -- get the content right first.
"""

import datetime
import json
import os
import urllib.error
import urllib.parse
import urllib.request

PROM_URL = "http://127.0.0.1:9090"
GRAFANA_URL = "http://127.0.0.1:3000"
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(REPO_DIR, "report_state.json")
TOKEN_FILE = os.path.join(REPO_DIR, ".grafana_token")

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


def _fmt(v):
    if v is None:
        return "?"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"


def prom_query(expr):
    url = f"{PROM_URL}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    result = data.get("data", {}).get("result", [])
    return float(result[0]["value"][1]) if result else None


def gather_firing_alerts():
    url = f"{PROM_URL}/api/v1/query?" + urllib.parse.urlencode({"query": 'ALERTS{alertstate="firing"}'})
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return []
    return [r["metric"].get("alertname") for r in data.get("data", {}).get("result", [])]


def gather_annotations(tag="guardian", limit=10):
    if not os.path.exists(TOKEN_FILE):
        return []
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    req = urllib.request.Request(
        f"{GRAFANA_URL}/api/annotations?tags={tag}&limit={limit}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return []


def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(metrics):
    with open(STATE_FILE, "w") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(), "metrics": metrics}, f, indent=2)


METRIC_QUERIES = {
    "health_score": "aiops_health_score",
    "security_score": "aiops_security_score",
    "ai_risk_score": "ai_risk_score",
    "guardian_status": "aiops_guardian_status",
    "knn_label": 'aiops_anomaly_label{job="aiops-watchdog-knn"}',
    "iforest_label": 'aiops_anomaly_label{job="aiops-watchdog-iforest"}',
    "autoencoder_label": 'aiops_anomaly_label{job="aiops-watchdog-autoencoder"}',
    "ufw_enabled": "aiops_security_ufw_enabled",
    "open_ports": "aiops_security_open_ports_count",
    "failed_logins": "aiops_security_failed_logins_recent",
    "root_ssh": "aiops_security_root_ssh_enabled",
    "updates_pending": "aiops_security_updates_pending",
    "watchdog_external": "ai_watchdog_port_external_access",
    "ai_tools_detected": "ai_tools_detected",
    "ai_processes_running": "ai_processes_running",
}


def gather_metrics():
    return {key: prom_query(expr) for key, expr in METRIC_QUERIES.items()}


def build_findings(current, previous, firing_alerts):
    """The rule-based 'playbook' engine: (condition, severity, template)
    triples, evaluated against current + previous state."""
    findings = []

    scores = [s for s in (current.get("health_score"), current.get("security_score"), current.get("ai_risk_score")) if s is not None]
    if scores:
        composite = min(scores)
        if composite < 50:
            findings.append(("Critical", f"Overall Guardian score is {_fmt(composite)} -- Critical (weakest of health/security/AI-risk)."))
        elif composite < 80:
            findings.append(("Medium", f"Overall Guardian score is {_fmt(composite)} -- Needs Attention (weakest of health/security/AI-risk)."))

    labels = {
        "KNN": current.get("knn_label"),
        "Isolation Forest": current.get("iforest_label"),
        "Autoencoder": current.get("autoencoder_label"),
    }
    present = {k: v for k, v in labels.items() if v is not None}
    if present and len(set(present.values())) > 1:
        anomalous = [k for k, v in present.items() if v == 1]
        normal = [k for k, v in present.items() if v == 0]
        findings.append((
            "High",
            f"Anomaly models disagree: {', '.join(anomalous)} anomalous, {', '.join(normal)} normal -- "
            "possible model-specific drift, not necessarily a real system anomaly.",
        ))

    for alert in firing_alerts:
        findings.append(("High", f"Alert firing: {alert}"))

    if current.get("ufw_enabled") == 0:
        findings.append(("Critical", "UFW firewall is disabled."))
    if current.get("root_ssh") == 1:
        findings.append(("Critical", "Root SSH login is enabled."))
    if current.get("watchdog_external") == 1:
        findings.append(("High", "Watchdog ports may be externally reachable (raw bind check, verify ufw rules)."))

    open_ports = current.get("open_ports") or 0
    if open_ports > 50:
        findings.append(("Medium", f"{_fmt(open_ports)} open ports -- elevated."))
    elif open_ports > 25:
        findings.append(("Low", f"{_fmt(open_ports)} open ports -- somewhat elevated."))

    if (current.get("updates_pending") or 0) > 0:
        findings.append(("Low", f"{_fmt(current['updates_pending'])} pending security update(s)."))

    if (current.get("failed_logins") or 0) > 20:
        findings.append(("Medium", f"{_fmt(current['failed_logins'])} recent failed login(s)."))

    prev_metrics = previous.get("metrics", {})
    for key, label in [("health_score", "Health"), ("security_score", "Security"), ("ai_risk_score", "AI Risk")]:
        cur = current.get(key)
        prev = prev_metrics.get(key)
        if cur is not None and prev is not None and abs(cur - prev) >= 5:
            direction = "up" if cur > prev else "down"
            findings.append(("Info", f"{label} score {direction} from {_fmt(prev)} to {_fmt(cur)} since last report."))

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f[0], 5))
    return findings


def render_report(current, previous, firing_alerts, annotations, findings):
    now = datetime.datetime.now()
    top = findings[:3]
    if top:
        summary = " ".join(f"**{sev}:** {text}" for sev, text in top)
    else:
        summary = "No findings above baseline -- Guardian reports a clean bill of health this cycle."

    lines = []
    lines.append("# Guardian Automated Report")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M %Z') or now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Generated by:** generate_report.py (rule-based, not an AI chat session)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Scores")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Health Score | {_fmt(current.get('health_score'))} |")
    lines.append(f"| Security Score | {_fmt(current.get('security_score'))} |")
    lines.append(f"| AI Risk Score | {_fmt(current.get('ai_risk_score'))} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Anomaly Detection: Model Comparison")
    lines.append("")
    lines.append("| Model | Label |")
    lines.append("|---|---|")
    for name, key in [("KNN", "knn_label"), ("Isolation Forest", "iforest_label"), ("Autoencoder", "autoencoder_label")]:
        val = current.get(key)
        state = "Anomaly" if val == 1 else ("Normal" if val == 0 else "No data")
        lines.append(f"| {name} | {state} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Recent Events")
    lines.append("")
    if annotations:
        for a in annotations:
            ts = datetime.datetime.fromtimestamp(a["time"] / 1000).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- **{ts}** — {a['text']}")
    else:
        lines.append("No recent annotated events.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if findings:
        lines.append("| Severity | Finding |")
        lines.append("|---|---|")
        for sev, text in findings:
            lines.append(f"| {sev} | {text} |")
    else:
        lines.append("No findings this cycle.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated from live Prometheus metrics (localhost:9090) and Grafana annotations (localhost:3000).*")

    return "\n".join(lines)


if __name__ == "__main__":
    previous = load_previous_state()
    current = gather_metrics()
    firing_alerts = gather_firing_alerts()
    annotations = gather_annotations()
    findings = build_findings(current, previous, firing_alerts)

    report_md = render_report(current, previous, firing_alerts, annotations, findings)

    out_path = os.path.join(REPO_DIR, f"full_system_report_{datetime.date.today().isoformat()}.md")
    with open(out_path, "w") as f:
        f.write(report_md)

    save_state(current)

    print(f"Report written to {out_path}")
    print(f"({len(findings)} findings)")
