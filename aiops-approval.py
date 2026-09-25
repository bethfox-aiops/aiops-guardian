#!/opt/aiops-venv/bin/python3
"""
aiops-approval.py

Guardian's Governance Engine "Human Approval" control plane: a small Flask
panel where a human explicitly approves or denies the privileged actions
Guardian's automation is allowed to take. Runs as User=beth
(aiops-approval.service, port 8020) -- not root -- and every action is
logged to LOGFILE.

Brought into version control here for the first time (2026-09-24) -- it
previously lived only at /usr/local/bin/aiops-approval.py, root-owned,
untracked. Install a change with:
    sudo cp aiops-approval.py /usr/local/bin/aiops-approval.py
    sudo systemctl restart aiops-approval

Model retrain recommendations (added 2026-09-24): the two service-restart
sections below (Prometheus/Loki) predate this and are unconditional --
the same two buttons render regardless of system state. The retrain
section is different on purpose: it only offers Approve/Deny when
diagnose_anomaly.build_verdict() actually recommends a retrain for that
model, so this panel is closer to the "Guardian recommends an action, a
human approves it" pattern than the static buttons above it. It reuses
diagnose_anomaly.py's verdict logic directly (not a re-implementation) so
the CLI diagnostic and this panel can't silently disagree about when a
retrain is warranted.

Deliberately NOT done here: restarting the corresponding watchdog service
after a successful retrain. That needs sudo this service does not have
(see CLAUDE.md -- aiops-watchdog-{knn,iforest,autoencoder} restarts are not
in beth's passwordless sudoers list), so it stays a separate manual step,
same as the retrain scripts' existing behavior (retrain_recent*.py never
restart the service that will load the new model, on purpose).
"""
from flask import Flask, request, render_template_string
import os
import sys
import subprocess
import datetime
import psutil

PORT = int(os.environ.get("PORT", 8020))
APPROVE_TOKEN = os.environ.get("APPROVE_TOKEN", "changeme")
LOGFILE = "/var/log/aiops-approvals.log"

REPO_DIR = "/home/beth/aiops-agents"
sys.path.insert(0, REPO_DIR)
import diagnose_anomaly  # noqa: E402  (needs sys.path set first)

RETRAIN_SCRIPTS = {
    "knn": "retrain_recent_knn.py",
    "iforest": "retrain_recent_iforest.py",
    "autoencoder": "retrain_recent.py",
}
RESTART_SERVICE = {
    "knn": "aiops-watchdog-knn",
    "iforest": "aiops-watchdog-iforest",
    "autoencoder": "aiops-watchdog-autoencoder",
}

app = Flask(__name__)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>AIOps Approval Control Plane</title>
<style>
body { font-family: Arial; margin: 40px; }
.box {
    border: 1px solid #ccc;
    padding: 14px;
    margin-bottom: 20px;
    border-radius: 8px;
    width: 420px;
}
.box.wide { width: 640px; }
.btn { padding: 10px 18px; font-size: 14px; margin-right: 10px; }
.good { color: green; }
.bad { color: red; }
.model-row { border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px; }
.model-row:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }
.verdict-text { font-size: 13px; color: #555; }
.hint { font-size: 12px; color: #888; margin-top: 10px; }
.logbox {
    white-space: pre-wrap;
    background: #f0f0f0;
    padding: 10px;
    border-radius: 8px;
    max-height: 200px;
    overflow-y: scroll;
}
</style>
</head>
<body>

<h1>AIOps Approval Control Plane</h1>

<div class="box">
    <h3>System Status</h3>
    <p><b>Uptime:</b> {{uptime}}</p>
    <p><b>CPU Usage:</b> {{cpu}}%</p>
    <p><b>Memory Usage:</b> {{mem}}%</p>
</div>

<div class="box">
    <h3>Prometheus</h3>
    <p>Status: <span class="{{prom_color}}">{{prom_status}}</span></p>
    <p>Last restart: {{prom_last}}</p>

    <form method="POST" action="/approve-prometheus">
        <input type="hidden" name="token" value="{{token}}">
        <button class="btn" type="submit">Approve Restart</button>
    </form>

    <form method="POST" action="/deny-prometheus">
        <input type="hidden" name="token" value="{{token}}">
        <button class="btn" type="submit">Deny</button>
    </form>
</div>

<div class="box">
    <h3>Loki</h3>
    <p>Status: <span class="{{loki_color}}">{{loki_status}}</span></p>
    <p>Last restart: {{loki_last}}</p>

    <form method="POST" action="/approve-loki">
        <input type="hidden" name="token" value="{{token}}">
        <button class="btn" type="submit">Approve Restart</button>
    </form>

    <form method="POST" action="/deny-loki">
        <input type="hidden" name="token" value="{{token}}">
        <button class="btn" type="submit">Deny</button>
    </form>
</div>

<div class="box">
    <h3>Batch Operations</h3>
    <form method="POST" action="/approve-both">
        <input type="hidden" name="token" value="{{token}}">
        <button class="btn" type="submit">Restart Both</button>
    </form>
</div>

<div class="box wide">
    <h3>Model Retrain Recommendations</h3>
    {% for m in models %}
    <div class="model-row">
        <p><b>{{m.label}}</b> &mdash; <span class="{{m.color}}">{{m.status_text}}</span></p>
        <p class="verdict-text">{{m.verdict}}</p>
        {% if m.recommend_retrain %}
        <form method="POST" action="/approve-retrain" style="display:inline;">
            <input type="hidden" name="token" value="{{token}}">
            <input type="hidden" name="model" value="{{m.id}}">
            <button class="btn" type="submit">Approve Retrain</button>
        </form>
        <form method="POST" action="/deny-retrain" style="display:inline;">
            <input type="hidden" name="model" value="{{m.id}}">
            <button class="btn" type="submit">Deny</button>
        </form>
        {% endif %}
    </div>
    {% endfor %}
    <p class="hint">Approving a retrain writes a new model file but deliberately does not restart
        the watchdog service that loads it -- that needs a separate manual step:
        <code>sudo systemctl restart aiops-watchdog-&lt;model&gt;</code>.</p>
</div>

<div class="box">
    <h3>Recent Activity Log</h3>
    <div class="logbox">{{log}}</div>
</div>

</body>
</html>
"""


def get_status(service):
    result = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True)
    status = result.stdout.strip()
    return ("Running", "good") if status == "active" else ("Not Running", "bad")


def get_last_restart(key):
    if not os.path.exists(LOGFILE):
        return "None recorded"

    last = "None recorded"
    with open(LOGFILE) as f:
        for line in f:
            if key in line:
                last = line.strip()
    return last


def read_log():
    if not os.path.exists(LOGFILE):
        return "No log entries."
    with open(LOGFILE) as f:
        lines = f.readlines()
    return "".join(lines[-12:])  # last 12 entries


def get_uptime():
    with open("/proc/uptime") as f:
        seconds = float(f.readline().split()[0])
    return str(datetime.timedelta(seconds=int(seconds)))


def get_model_rows():
    """One row per watchdog for the template, built from
    diagnose_anomaly.build_verdict(). Ground truth and the suspend/reboot
    correlation check are each computed once here and shared across all
    three models (see build_verdict()'s docstring) -- doing them per-model
    would triple a ~1s CPU sample and a journalctl scan for no benefit."""
    rows = []
    try:
        gt = diagnose_anomaly.check_ground_truth()
        corr = diagnose_anomaly.check_suspend_or_reboot_correlation()
    except Exception as e:
        return [{
            "id": mid, "label": mid.upper(), "status_text": "Diagnostics unavailable",
            "color": "bad", "verdict": str(e), "recommend_retrain": False,
        } for mid in diagnose_anomaly.WATCHDOGS]

    for mid in diagnose_anomaly.WATCHDOGS:
        try:
            v = diagnose_anomaly.build_verdict(mid, gt=gt, corr=corr)
        except Exception as e:
            rows.append({
                "id": mid, "label": mid.upper(), "status_text": "Diagnostics unavailable",
                "color": "bad", "verdict": str(e), "recommend_retrain": False,
            })
            continue
        if not v["reachable"]:
            rows.append({
                "id": mid, "label": mid.upper(), "status_text": "Unreachable",
                "color": "bad", "verdict": v["verdict"], "recommend_retrain": False,
            })
            continue
        rows.append({
            "id": mid,
            "label": mid.upper(),
            "status_text": "Anomaly" if v["label"] == 1 else "Normal",
            "color": "bad" if v["recommend_retrain"] else "good",
            "verdict": v["verdict"],
            "recommend_retrain": v["recommend_retrain"],
        })
    return rows


@app.route("/")
def home():
    prom_status, prom_color = get_status("prometheus")
    loki_status, loki_color = get_status("loki")

    return render_template_string(
        PAGE_TEMPLATE,
        uptime=get_uptime(),
        cpu=psutil.cpu_percent(),
        mem=psutil.virtual_memory().percent,
        prom_status=prom_status,
        prom_color=prom_color,
        loki_status=loki_status,
        loki_color=loki_color,
        prom_last=get_last_restart("PROMETHEUS"),
        loki_last=get_last_restart("LOKI"),
        models=get_model_rows(),
        log=read_log(),
        token=APPROVE_TOKEN
    )


def log_action(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a") as f:
        f.write(f"{timestamp}: {message}\n")


@app.route("/approve-prometheus", methods=["POST"])
def approve_prometheus():
    if request.form.get("token") != APPROVE_TOKEN:
        return "Invalid token", 403
    subprocess.call(["systemctl", "restart", "prometheus"])
    log_action("PROMETHEUS restarted")
    return home()


@app.route("/deny-prometheus", methods=["POST"])
def deny_prometheus():
    log_action("PROMETHEUS restart denied")
    return home()


@app.route("/approve-loki", methods=["POST"])
def approve_loki():
    if request.form.get("token") != APPROVE_TOKEN:
        return "Invalid token", 403
    subprocess.call(["systemctl", "restart", "loki"])
    log_action("LOKI restarted")
    return home()


@app.route("/deny-loki", methods=["POST"])
def deny_loki():
    log_action("LOKI restart denied")
    return home()


@app.route("/approve-both", methods=["POST"])
def approve_both():
    if request.form.get("token") != APPROVE_TOKEN:
        return "Invalid token", 403
    subprocess.call(["systemctl", "restart", "prometheus"])
    subprocess.call(["systemctl", "restart", "loki"])
    log_action("BOTH SERVICES restarted")
    return home()


@app.route("/approve-retrain", methods=["POST"])
def approve_retrain():
    if request.form.get("token") != APPROVE_TOKEN:
        return "Invalid token", 403
    model = request.form.get("model", "")
    script = RETRAIN_SCRIPTS.get(model)
    if not script:
        return "Unknown model", 400
    try:
        result = subprocess.run(
            ["/opt/aiops-venv/bin/python3", script],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=300,
        )
        ok = result.returncode == 0
        tail_lines = (result.stdout + result.stderr).strip().splitlines()
        tail = " | ".join(tail_lines[-3:]) if tail_lines else "(no output)"
        service = RESTART_SERVICE.get(model, "?")
        log_action(
            f"{model.upper()} RETRAIN {'SUCCEEDED' if ok else 'FAILED'} -- {tail}"
            + (f" -- restart manually: sudo systemctl restart {service}" if ok else "")
        )
    except subprocess.TimeoutExpired:
        log_action(f"{model.upper()} RETRAIN TIMED OUT after 300s")
    except Exception as e:
        log_action(f"{model.upper()} RETRAIN ERROR: {e}")
    return home()


@app.route("/deny-retrain", methods=["POST"])
def deny_retrain():
    model = request.form.get("model", "?")
    log_action(f"{model.upper()} retrain denied")
    return home()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
