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

Runs daily via a user-level systemd timer
(~/.config/systemd/user/guardian-daily-report.timer, 07:00) -- can still
be run manually any time with `python3 generate_report.py`.
"""

import datetime
import json
import os
import time
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


def prom_query_all(expr):
    """Like prom_query, but returns every series (with labels), not just the
    first result's value -- needed for per-instance checks (multiple Windows
    hosts, multiple edge collectors)."""
    url = f"{PROM_URL}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return []
    return data.get("data", {}).get("result", [])


def gather_windows_health():
    """Per-instance score from aiops-watchdog-windows.py, keyed by the
    Windows host's exported_instance label (e.g. 'DESKTOP-0AJUKU3:9182')."""
    out = {}
    for r in prom_query_all("aiops_windows_health_score"):
        instance = r["metric"].get("exported_instance", "?")
        out[instance] = float(r["value"][1])
    return out


# How stale a Guardian textfile-collector timestamp can get before it's
# treated as a finding. Both collectors run every 15 minutes via a
# SYSTEM/Highest scheduled task (confirmed 2026-08-28) -- 30 minutes gives
# margin for one missed run without being noisy.
COLLECTOR_STALE_SECONDS = 1800

COLLECTOR_METRICS = {
    "windows_process_attribution_collector_last_run_timestamp_seconds": "process-attribution",
    "windows_disk_health_collector_last_run_timestamp_seconds": "disk-health",
}


def gather_collector_staleness():
    """Age (seconds) of each Guardian textfile-collector's last successful
    run, per Windows instance. A large age means the collector script /
    scheduled task on that host has stopped running, even if windows_exporter
    itself is fine (this is the gap aiops-watchdog-windows.py doesn't cover --
    it checks cpu/mem/disk/service, not these Guardian-specific collectors)."""
    now = time.time()
    out = []
    for metric, label in COLLECTOR_METRICS.items():
        for r in prom_query_all(metric):
            instance = r["metric"].get("instance", "?")
            ts = float(r["value"][1])
            out.append((label, instance, now - ts))
    return out


# Windows hosts to report on individually, mapped to their direct-scrape
# Prometheus instance label. Deliberately the direct-scrape series, not the
# edge_site-labeled duplicate via the Pi -- same physical machine, showing
# both would look like two separate hosts.
WINDOWS_MACHINES = {
    "DESKTOP-0AJUKU3": "DESKTOP-0AJUKU3:9182",
    "DESKTOP-503POVP": "DESKTOP-503POVP:9182",
}

# Every metric Guardian itself computes and exports from Core (health,
# security, AI-risk, anomaly labels/scores) -- excludes raw runtime/exporter
# internals (go_*, process_*, prometheus_*) which aren't something Guardian
# chose to collect, just artifacts of the libraries it's built on.
CORE_METRIC_REGEX = "aiops_.*|ai_[a-z].*"


def gather_core_snapshot():
    """Every Guardian-authored metric for Core itself, as (display_name, value).
    Disambiguates same-named series generically off every label besides
    __name__/instance -- not just job -- since some metrics (e.g.
    aiops_windows_health_score) vary by exported_instance instead, and
    others by a mount/check-style label. Hardcoding just 'job' silently
    collapsed those into identical-looking duplicate rows."""
    rows = []
    for r in prom_query_all('{__name__=~"' + CORE_METRIC_REGEX + '"}'):
        labels = r["metric"]
        name = labels.get("__name__", "?")
        val = float(r["value"][1])
        job = labels.get("job", "")
        extras = []
        if job and job != "aiops-guardian-health":
            extras.append(job.replace("aiops-watchdog-", "").replace("aiops-", ""))
        for k, v in sorted(labels.items()):
            if k in ("__name__", "instance", "job"):
                continue
            extras.append(f"{k}={v}")
        if extras:
            name += " [" + ", ".join(extras) + "]"
        rows.append((name, val))
    rows.sort(key=lambda x: x[0])
    return rows


def _first_value(expr):
    r = prom_query_all(expr)
    return float(r[0]["value"][1]) if r else None


def gather_windows_snapshot(instance):
    """Curated per-host snapshot for one Windows machine: Guardian's own
    per-instance health checks, computed resource-usage percentages, textfile
    -collector freshness, and top-5 process attribution. Deliberately skips
    windows_exporter's ~100 low-level internal families (per-core interrupt/
    DPC/cstate counters, service_info's hundreds of rows, etc.) -- those are
    exporter internals, not something Guardian intentionally curates."""
    rows = []

    for check in ["score", "cpu_ok", "mem_ok", "disk_ok", "service_ok", "up"]:
        metric = "aiops_windows_health_score" if check == "score" else f"aiops_windows_health_{check}"
        v = _first_value(f'{metric}{{exported_instance="{instance}"}}')
        if v is not None:
            rows.append((metric, v))

    mem_avail = _first_value(f'windows_memory_available_bytes{{instance="{instance}"}}')
    mem_total = _first_value(f'windows_memory_physical_total_bytes{{instance="{instance}"}}')
    if mem_avail is not None and mem_total:
        rows.append(("memory_used_percent", 100 * (1 - mem_avail / mem_total)))

    disk_free = _first_value(f'windows_logical_disk_free_bytes{{instance="{instance}",volume="C:"}}')
    disk_size = _first_value(f'windows_logical_disk_size_bytes{{instance="{instance}",volume="C:"}}')
    if disk_free is not None and disk_size:
        rows.append(("disk_C_used_percent", 100 * (1 - disk_free / disk_size)))

    for metric, label in COLLECTOR_METRICS.items():
        v = _first_value(f'{metric}{{instance="{instance}"}}')
        if v is not None:
            rows.append((f"{label}_collector_age_minutes", round((time.time() - v) / 60, 1)))

    for metric_name in ["windows_top_process_cpu_percent", "windows_top_process_mem_percent"]:
        series = prom_query_all(f'{metric_name}{{instance="{instance}"}}')
        for r in sorted(series, key=lambda r: -float(r["value"][1]))[:5]:
            proc = r["metric"].get("name", "?")
            rows.append((f"{metric_name}[{proc}]", float(r["value"][1])))

    return rows


def gather_machine_snapshots():
    core = gather_core_snapshot()
    windows = {name: gather_windows_snapshot(instance) for name, instance in WINDOWS_MACHINES.items()}
    return core, windows


def render_machines_section(core_snapshot, windows_snapshots):
    lines = []
    lines.append("## Machines")
    lines.append("")
    lines.append(
        "Every metric Guardian collects, listed per machine. This is a "
        "reference snapshot, not an interpretation -- see Findings above "
        "for what actually needs attention."
    )
    lines.append("")
    lines.append("### beth-ikins (Guardian Core)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for name, val in core_snapshot:
        lines.append(f"| {name} | {_fmt(val)} |")
    lines.append("")
    lines.append("### guardian-proto-1 (Edge Collector Pi)")
    lines.append("")
    lines.append(
        "No self-telemetry collected for this host yet -- it currently "
        "only relays windows_exporter data scraped from the Windows hosts "
        "below (visible on Core via their `edge_site=\"guardian-proto-1\"`-"
        "labeled series). A node_exporter deployment for the Pi itself was "
        "scoped out of the original M1 milestone and hasn't been added since."
    )
    lines.append("")
    for machine in WINDOWS_MACHINES:
        lines.append(f"### {machine}")
        lines.append("")
        rows = windows_snapshots.get(machine, [])
        if rows:
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for name, val in rows:
                lines.append(f"| {name} | {_fmt(val)} |")
        else:
            lines.append("No data (host unreachable or not yet scraped).")
        lines.append("")
    return "\n".join(lines)


# Prometheus alert name -> the watchdog job it corresponds to, so a firing
# alert can be paired with that job's own attribution evidence below.
ALERT_JOB_MAP = {
    "KNNWatchdogSustainedAnomaly": "aiops-watchdog-knn",
    "IsolationForestWatchdogSustainedAnomaly": "aiops-watchdog-iforest",
    "AutoencoderWatchdogSustainedAnomaly": "aiops-watchdog-autoencoder",
}

MODEL_JOB_MAP = {
    "KNN": "aiops-watchdog-knn",
    "Isolation Forest": "aiops-watchdog-iforest",
    "Autoencoder": "aiops-watchdog-autoencoder",
}


def gather_anomaly_attribution():
    """Guardian's own Behavioral Attestation evidence, per watchdog job: the
    top-CPU process at anomaly time (Phase 1, process_attribution.py) and the
    eBPF syscall counts traced against it (Phase 2, trace_suspect.sh). This is
    the 'who/what caused it' data that already gets computed on every anomaly
    but, before this, only showed up buried in the raw Machines snapshot --
    not attached to the finding that actually needs it."""
    attribution = {}
    for r in prom_query_all("aiops_anomaly_top_process_info"):
        job = r["metric"].get("job", "")
        attribution[job] = {"name": r["metric"].get("name", "?"), "pid": r["metric"].get("pid", "?"), "syscalls": {}}

    for r in prom_query_all("aiops_anomaly_suspect_syscall_count"):
        job = r["metric"].get("job", "")
        stype = r["metric"].get("syscall_type", "?")
        count = float(r["value"][1])
        if job in attribution and count > 0:
            attribution[job]["syscalls"][stype] = count

    return attribution


def _describe_attribution(attribution, job):
    info = attribution.get(job)
    if not info:
        return ""
    desc = f" Attributed to `{info['name']}` (pid {info['pid']})"
    if info["syscalls"]:
        parts = ", ".join(f"{v:.0f} {k}" for k, v in sorted(info["syscalls"].items(), key=lambda x: -x[1]))
        desc += f" -- {parts} observed in the eBPF trace."
    else:
        desc += " -- no suspicious syscalls observed in the eBPF trace."
    return desc


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


def build_findings(current, previous, firing_alerts, windows_health, collector_staleness, attribution):
    """The rule-based 'playbook' engine: (condition, severity, template)
    triples, evaluated against current + previous state."""
    findings = []

    for instance, score in windows_health.items():
        if score < 50:
            findings.append(("Critical", f"Windows host {instance} health score is {_fmt(score)} -- Critical."))
        elif score < 80:
            findings.append(("Medium", f"Windows host {instance} health score is {_fmt(score)} -- Needs Attention."))

    for label, instance, age in collector_staleness:
        if age > COLLECTOR_STALE_SECONDS:
            hours = age / 3600
            findings.append((
                "Medium",
                f"{instance}: {label} collector data is stale ({hours:.1f}h old) -- "
                "check the scheduled task/script on that host.",
            ))

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
        attrib_text = "".join(
            f" {name}:{_describe_attribution(attribution, MODEL_JOB_MAP[name])}" for name in anomalous if MODEL_JOB_MAP.get(name)
        )
        findings.append((
            "High",
            f"Anomaly models disagree: {', '.join(anomalous)} anomalous, {', '.join(normal)} normal -- "
            "possible model-specific drift, not necessarily a real system anomaly."
            + attrib_text,
        ))

    for alert in firing_alerts:
        job = ALERT_JOB_MAP.get(alert)
        findings.append(("High", f"Alert firing: {alert}." + (_describe_attribution(attribution, job) if job else "")))

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


def render_report(current, previous, firing_alerts, annotations, findings, windows_health, core_snapshot, windows_snapshots):
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
    lines.append("## Windows Hosts")
    lines.append("")
    if windows_health:
        lines.append("| Instance | Health Score |")
        lines.append("|---|---|")
        for instance, score in sorted(windows_health.items()):
            lines.append(f"| {instance} | {_fmt(score)} |")
    else:
        lines.append("No Windows hosts reporting.")
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
    lines.append("")
    lines.append(render_machines_section(core_snapshot, windows_snapshots))
    lines.append("---")
    lines.append("*Generated from live Prometheus metrics (localhost:9090) and Grafana annotations (localhost:3000).*")

    return "\n".join(lines)


if __name__ == "__main__":
    previous = load_previous_state()
    current = gather_metrics()
    firing_alerts = gather_firing_alerts()
    annotations = gather_annotations()
    windows_health = gather_windows_health()
    collector_staleness = gather_collector_staleness()
    attribution = gather_anomaly_attribution()
    findings = build_findings(current, previous, firing_alerts, windows_health, collector_staleness, attribution)
    core_snapshot, windows_snapshots = gather_machine_snapshots()

    report_md = render_report(current, previous, firing_alerts, annotations, findings, windows_health, core_snapshot, windows_snapshots)

    out_path = os.path.join(REPO_DIR, f"full_system_report_{datetime.date.today().isoformat()}.md")
    with open(out_path, "w") as f:
        f.write(report_md)

    save_state(current)

    print(f"Report written to {out_path}")
    print(f"({len(findings)} findings)")
