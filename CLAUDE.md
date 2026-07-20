# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Guardian: an AI Systems Intelligence Platform (see `VISION.md`). Its north-star
question is "can I explain and prove what this system actually did?" — not just
detect anomalies, but attribute cause and verify behavior against expectation.
It's organized around five engines (Observability, ML, Security, Governance,
Behavioral Attestation); `ROADMAP.md` has the phased build plan for the newest
one, Behavioral Attestation. Read `VISION.md` and `ROADMAP.md` before proposing
new features — the scope is deliberately narrow (performance attribution for AI
workloads on Linux, not general-purpose observability), and there's a stated
litmus test for whether new work belongs here.

## Running & operating

There's no build step — scripts run directly against a venv that lives outside
this repo at `/opt/aiops-venv` (no `requirements.txt`/`pyproject.toml` is
tracked here; check `/opt/aiops-venv/bin/pip list` if unsure whether a package
is available before assuming it isn't).

In production, everything runs as systemd services (`User=beth`,
`WorkingDirectory=/home/beth/aiops-agents`, all `Restart=always`/`on-failure`):

| Service | Script | Port |
|---|---|---|
| `aiops-guardian-health` | `aiops-guardian-health.py` | 8014 |
| `aiops-watchdog-knn` | `aiops-watchdog-knn.py` | 8011 |
| `aiops-watchdog-iforest` | `aiops-watchdog-iforest.py` | 8012 |
| `aiops-watchdog-autoencoder` | `aiops-watchdog-autoencoder.py` | 8013 |
| `aiops-watchdog-ml` | `aiops-watchdog-ml.py` | — (collector, no HTTP) |

Restarting any of these needs `sudo` and is **not** in this user's passwordless
sudoers list (only `ufw`, `trace_suspect.sh`, and restarting
`prometheus`/`loki` are passwordless) — restarts have to be run interactively
by the user, not scripted.

Every watchdog/health script reads its port from an env var with the port
above as default (`WATCHDOG_PORT` for the three watchdogs,
`GUARDIAN_HEALTH_PORT` for guardian-health), so you can smoke-test a change
without touching the live service:
```
WATCHDOG_PORT=18011 /opt/aiops-venv/bin/python aiops-watchdog-knn.py
```

**A second, independent copy of the KNN watchdog also runs in microk8s**
(namespace `aiops`, deployment `aiops-watchdog-knn`) — a long-lived pod
unrelated to the systemd service. A `python aiops-watchdog-knn.py` process
running as root with no venv in its path is that pod's container, not a stray
leftover; check `microk8s kubectl get pods -A` before assuming it's junk and
killing it (killing it just makes Kubernetes restart it).

## Linting

`ruff` isn't installed by default — `pip install --user ruff`, then
`~/.local/bin/ruff check .`. There's no lint config file in the repo (no
`ruff.toml`/`pyproject.toml`), so it runs with ruff's defaults. There is no
automated test suite in this repo.

## Architecture

**Data flow:** `aiops-watchdog-ml.py` collects system telemetry (disk, CPU,
mem, net, disk I/O, GPU util/mem/temp, inode — every ~5s via `psutil`/NVML)
into `aiops_data/metrics.csv` → the three anomaly watchdogs load a model
trained on that data and score live metrics against it → all exporters push
Prometheus gauges → Grafana dashboards.

**The three anomaly watchdogs** (`aiops-watchdog-{knn,iforest,autoencoder}.py`)
are structurally parallel: load model + scaler on startup, loop every
`INTERVAL` seconds collecting metrics, run inference, export
`aiops_anomaly_label`/`aiops_anomaly_score` plus the underlying metric gauges,
and on an anomaly, pull in `process_attribution.py`/`ebpf_trace.py`/
`gpu_attribution.py` to attach "who was responsible" evidence.

**`aiops-guardian-health.py`** computes three independent scores exposed as
gauges — health (`aiops_health_*`: cpu/mem/disk/inode/service checks),
security (`aiops_security_*`, computed in `compute_security()`), and AI risk
(`ai_*`, computed in `calculate_ai_risk_score()`). `sudo ufw status` output is
cached for 25s (`_get_ufw_status_cached()`) since several checks per cycle used
to shell out to it separately.

**Security posture note:** the watchdog ports (8011–8014) are bound to
`0.0.0.0` at the socket level, not `127.0.0.1` — ufw's explicit
`DENY`-from-Anywhere rules are what actually keep them non-external, not the
bind address. `check_watchdog_port_external_access()` /
`_ufw_denies_port_externally()` check those ufw rules directly (v4 and v6)
rather than only testing raw socket reachability, so don't "fix" a reported
exposure by just changing the bind address without also checking ufw.

**Behavioral Attestation modules** (shared by watchdogs and retrain scripts,
one file per roadmap phase):
- `process_attribution.py` (Phase 1) — top-N process attribution by CPU/mem
- `ebpf_trace.py` + `trace_suspect.sh` (Phase 2) — scoped `bpftrace` via
  passwordless sudo, rate-limited per PID
- `otel_setup.py` (Phase 3) — shared OpenTelemetry tracer, exports to a local
  Tempo instance (port 4317, localhost-only via ufw)
- `gpu_attribution.py` (Phase 4) — per-process GPU accounting via NVML
- `behavioral_policy.py` (Phase 5) — declarative `POLICIES` dict; checks a
  completed workflow run's evidence against expected behavior
- `release_record.py` (Phase 6) — ties a git commit to Phases 1–5 evidence
  into one structured release record

**`retrain_recent*.py`** (one per model) retrain against the tail of
`aiops_data/metrics.csv`, instrumented with OTel spans and
`behavioral_policy.verify()`. They don't restart the corresponding service —
that's a separate manual step. **Known inconsistency:** `RECENT_ROWS` is `2000`
in `retrain_recent_knn.py` but still `100000` in `retrain_recent.py`
(autoencoder) and `retrain_recent_iforest.py` — check which you're touching
before assuming the row count is sane.

## Other notes

- `full_system_report_*.md` / `system_report_*.md` are point-in-time snapshot
  reports and are gitignored — don't expect them to be tracked or complete.
- `disk_watchdog.py`/`.service` is legacy and currently inactive; superseded
  by the health/security checks in `aiops-guardian-health.py`.
