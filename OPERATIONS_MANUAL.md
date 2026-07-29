# Guardian AIOps System — Operations and Architecture Manual

Design, Implementation, and Operation of a Local AIOps Platform

**Author:** Beth Fox
**Originally written:** April 2026 (as a Word document, `AIOps_Field_Manual_Deploying_MicroK8s.doc`)
**This revision:** July 2026 — full rewrite to match current system state, converted to Markdown and moved into the repo (same tracked-doc pattern as `VISION.md`/`ROADMAP.md`/`CLAUDE.md`)

## Table of Contents

- Chapter 1 — System Configuration (Baseline)
- Chapter 2 — System Overview
- Chapter 3 — Core Applications and Services
- Chapter 4 — Custom Scripts and Intelligence Layer
- Chapter 5 — Behavioral Attestation Engine
- Chapter 6 — Operations and Runbook
- Chapter 7 — Security, Governance, and Risk Management
- Chapter 8 — Alerting and Incident Response
- Chapter 9 — Scalability and Future Architecture
- Chapter 10 — Known Gaps and Roadmap

---

## Chapter 1 — System Configuration (Baseline)

### 1.1 Purpose

This chapter documents the baseline configuration of the system running Guardian — a reference for setup/recovery, configuration consistency, and rebuild scenarios.

### 1.2 Hardware Overview

- **System Type:** Laptop (Primary AIOps Lab)
- **Model:** Dell Precision 7780
- **CPU:** Intel i9 (multi-core, hyperthreaded)
- **GPU:** NVIDIA RTX 5000 Ada Generation Laptop GPU, 16 GB VRAM, driver 535.309.01
- **Memory:** ~16 GB
- **Storage:** 1.9 TB internal SSD (root filesystem; ~82 GB used as of this revision) + external 1 TB backup drive

### 1.3 Operating System

- **OS:** Ubuntu 22.04.5 LTS ("jammy")
- **Kernel:** 6.8.0-134-generic (HWE kernel — this version number moves with routine `apt upgrade`s; treat the exact build as a snapshot, not a pinned requirement)

Verification:
```
lsb_release -a
uname -r
```

### 1.4 Base System Setup

```
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl vim python3 python3-pip net-tools htop
```

### 1.5 Directory Structure

| Path | Purpose |
|---|---|
| `/home/beth/aiops-agents` | Python scripts, agents, this manual — git repo, remote `git@github.com:bethfox-aiops/aiops-guardian.git` (public) |
| `/opt/aiops-venv` | Python virtual environment (all Guardian scripts run against this, not system Python) |
| `/etc/` | System and service configs (prometheus, grafana, loki, promtail, tempo, alertmanager) |
| `/var/log/` | System logs |
| External 1 TB drive | Déjà Dup backup storage |

### 1.6 Python Environment

```
source /opt/aiops-venv/bin/activate
```

Key packages: `psutil`, `prometheus_client`, `pandas`, `scikit-learn`, `pyod`, `flask`, `opentelemetry-api`/`-sdk`/`-exporter-otlp-proto-grpc`, `pynvml`, `pytest`. There's no tracked `requirements.txt`/`pyproject.toml` in the repo — check `/opt/aiops-venv/bin/pip list` before assuming a package is or isn't available.

### 1.7 Service Management (systemd)

```
sudo systemctl start|stop|restart|status <service>
sudo systemctl enable <service>
```

Restarting any `aiops-*` service requires `sudo` and is **not** in this user's passwordless sudoers list (the only passwordless entries are `ufw`, `trace_suspect.sh`, and restarting `prometheus`/`loki`) — restarts must be run interactively by a human, not scripted/automated.

### 1.8 Networking

IP assignment is DHCP. DHCP changes have previously broken things that assumed a stable address (see Loki's `instance_addr` bug, Chapter 3.4, and Kubernetes certificate risk, Chapter 9).

### 1.9 Backup Configuration

Déjà Dup, versioned/incremental backups (duplicity underneath) to an external 1 TB USB drive.

### 1.10 Monitoring/Observability Stack Installation Locations

| Component | Location | Port(s) |
|---|---|---|
| Prometheus | `/etc/prometheus/` | 9090 |
| Grafana | `/etc/grafana/` | 3000 |
| Loki | `/etc/loki/` | — (queried via Grafana) |
| Promtail | `/etc/promtail/` | — |
| Tempo | `/etc/tempo/` | 3200 (HTTP), 4317 (OTLP gRPC), 4318 (OTLP HTTP) — all localhost-only |
| Alertmanager | `/etc/alertmanager/` | 9093 |

**New since the original manual:** Tempo (distributed tracing backend) and Alertmanager (alert routing) — neither existed in April. See Chapters 5 and 8.

### 1.11 Linting and Testing

**New since the original manual — did not exist in April.** `ruff` (installed via `pip install --user ruff`) runs with defaults (no `ruff.toml`/`pyproject.toml` in the repo). A `pytest` suite exists (`test_behavioral_policy.py`, `test_guardian_health.py`) and runs in GitHub Actions on every push/PR (`.github/workflows/tests.yml`). This closes what used to be a real gap — see Chapter 10.

### 1.12 Notes and Observations

- Environment is designed for local AIOps experimentation, but has grown a genuine test suite, CI, and alerting pipeline since April — closer to "small production system" than "experiment" now.
- Supports ML-based monitoring, observability tooling, Behavioral Attestation, and Agentic AI expansion.

---

## Chapter 2 — System Overview

### 2.1 Purpose of the System

The Guardian AIOps System is a locally deployed platform that collects real-time system/security telemetry, detects abnormal behavior with machine learning, evaluates health/risk/security posture, attributes *why* anomalies happen (not just that they happened), and verifies runtime behavior against declared expectations — with dashboards, logs, and alerting throughout.

### 2.2 System Philosophy

The original operational loop still holds:

```
Observe → Detect → Decide → Act → Log
```

But as of this revision, Guardian's organizing question (see `VISION.md`) has sharpened to:

> **Can I explain and prove what this system actually did?**

That's a stronger claim than "detect anomalies" — it's the difference between "something looks wrong" and "here is the process, the syscalls, the GPU usage, and the policy verification proving what happened and why."

### 2.3 Core Components

**Monitoring & Metrics:** Prometheus, Python exporters (8 watchdog/health services, see Ch. 3)
**Visualization:** Grafana — 9 dashboards total; see Chapter 3.3 for the full inventory (4 Guardian-specific, 4 supporting/imported, plus the standard bundled Loki board)
**Logging & Tracing:** Loki + Promtail (logs), **Tempo** (distributed traces — new)
**Machine Learning:** KNN (primary), Isolation Forest, Autoencoder — **all three now live as systemd services** (the autoencoder was experimental/not-running-as-a-service in the April manual; that's since resolved)
**Behavioral Attestation (new engine, entirely absent from the April manual):** process attribution, eBPF syscall tracing, OpenTelemetry traces, GPU-per-process accounting, declarative policy verification, release provenance records — see Chapter 5
**Alerting (new):** Prometheus alerting rules + Alertmanager routing to Slack — see Chapter 8
**Service Management:** systemd
**Container/Scaling Layer:** MicroK8s (currently **not running** — see Chapter 9 for current state vs. the April manual's aspirational framing)

### 2.4 System Architecture — Five Engines

The April manual described a 3-layer model (collection → analysis → visualization). The system has since grown a formal five-engine architecture (`VISION.md`):

```
Guardian
├── Observability Engine     — metrics, logs, dashboards
├── ML Engine                — KNN, Isolation Forest, Autoencoder
├── Security Engine          — file integrity, user monitoring, SSH, AI security
├── Governance Engine        — guardrails (v1 live), human approval, policy
└── Behavioral Attestation   — trace collection, eBPF, OpenTelemetry, GPU tracing,
    Engine (new, 2026-07)      process attribution, root cause, workflow verification
```

Guardian's litmus test for whether a new feature belongs at all: does it collect evidence, explain behavior, attribute responsibility, or increase trust in autonomous systems? If not, it belongs in a different project.

### 2.5 Key Metrics and Signals

**System:** CPU %, memory %, disk usage/free/fill-rate, network throughput, GPU util/mem/temp, inode %
**Health:** per-component checks, aggregate `aiops_health_score`, `aiops_guardian_status`
**Security:** UFW state, pending updates, root SSH, failed logins, plus ~40 additional gauges added 2026-07-17 (file integrity, user monitoring, AI security — see Chapter 7)
**ML:** anomaly label (0/1), anomaly score, per model
**Behavioral Attestation (new):** top suspect process + CPU/mem, syscall counts (open/exec/connect/write) during a scoped eBPF trace, GPU MiB used, policy pass/fail + violations
**Windows hosts (new):** per-instance health via `windows_exporter` + Prometheus (no local agent — see Chapter 4)
**Cross-source priority (new):** a single `aiops_priority_score` ranking every currently-bad signal (Prometheus alerts, Guardian's own checks, the legacy `all_check.service`) by tier + novelty

### 2.6 Data Flow Summary

```
Watchdogs (psutil/NVML/exporters)
        ↓
Prometheus  ──────────────→  Alertmanager → Slack
        ↓
Grafana dashboards
        ↓
Loki (logs) ←→ Tempo (traces, linked via tracesToLogsV2)
```

### 2.7 Current Capabilities

Everything from the April manual, plus: process/syscall/GPU-level anomaly attribution, distributed tracing of ML training runs, declarative behavioral policy verification, AI-managed release provenance records, real Slack alerting, a Windows-fleet health watchdog, a log-anomaly watchdog, cross-source priority triage, an automated test suite + CI, and a rule-based report generator (`generate_report.py`) that interprets live Prometheus/Grafana state without needing an ad-hoc AI chat session each time.

### 2.8 Future Direction

See Chapter 9 (Scalability) and Chapter 10 (Roadmap) — near-term direction has shifted from "MicroK8s multi-node scaling" (April's framing) toward two things: (1) closing operational-hygiene gaps (Chapter 10), and (2) a Raspberry-Pi-based edge-collector architecture for future consulting deployments (`EDGE_ARCHITECTURE.md`), not yet started (no hardware purchased).

### 2.9 Summary

Guardian has grown from a single-machine ML-monitoring experiment into a system that observes, explains, attributes, verifies, and now alerts on its own behavior — while remaining honest, in its own roadmap, about where operational maturity still lags the architecture's ambition.

---

## Chapter 3 — Core Applications and Services

### 3.1 Purpose

Operational reference for every service Guardian depends on: role, config location, port, management commands, verification.

### 3.2 Prometheus

- Config: `/etc/prometheus/prometheus.yml`; alert rules: `/etc/prometheus/rules/aiops-alerts.yml` (new)
- Port: 9090
- `sudo systemctl {start|stop|restart|status} prometheus` (restart is passwordless for this user)
- Verify: `curl http://localhost:9090`, targets at `/targets`, alerts at `/alerts`
- **Reload vs. restart (new, important):** `--web.enable-lifecycle` is enabled — for scrape-config-only changes, use `curl -X POST http://localhost:9090/-/reload` instead of restarting. A full restart risks losing whatever portion of the in-memory head block/WAL hasn't yet compacted to disk (empirically only ~30-45 min reliably survives a restart if the most recent compaction is old).

### 3.3 Grafana

- Location: `/etc/grafana/`; Port: 3000
- `sudo systemctl {start|stop|restart|status} grafana-server`
- Verify: `http://localhost:3000`
- Push updates via the Grafana HTTP API using a **service account token** (`guardian-dashboard-deploy`, Editor role, stored in gitignored `.grafana_token`) — admin-password basic auth returns 401 at the API level even though UI login works. Always include the dashboard's `uid` in the POST body — omitting it creates a duplicate dashboard instead of updating, even with `overwrite: true`.

#### 3.3.1 Dashboard Inventory (9 total, audited 2026-07-27)

**Important naming note:** there is no single "the flagship dashboard" — that phrase was ambiguous in earlier revisions of this manual. There are two distinct, actively-maintained Guardian dashboards plus a genuinely separate one literally titled "Flagship." Disambiguated below.

**Guardian-specific dashboards (4):**

| Title | UID | Version | What it's for |
|---|---|---|---|
| **Guardian: Flagship Overview** | `guardian-flagship` | — | The newest one (built 2026-07-27 by `talks/build_flagship_dashboard.py`, a build script — not hand-edited JSON, not part of the running system, re-run to update). 6 rows / ~17 panels: Guardian Status at a Glance (health/security/AI-risk scores), Priority Warnings (the `aiops-watchdog-priority` triage table), Recent Events (firing alerts + a Grafana-annotations list fed by `grafana_annotate.py`), Model Agreement Over Time (all 3 ML watchdogs side by side), Security Detail (UFW/ports/logins/root-SSH/updates), and a "Go Deeper" panel linking to the component dashboards below. Intended as the single screen for a talk/demo/portfolio walkthrough. |
| **AI Anomaly Detection** | `ad6lc4j` | 199 | The original, most comprehensive dashboard — by far the highest version number in the system, meaning it's the most iterated-on and likely what's actually used day-to-day. 5 collapsed rows, ~65 panels: Anomaly Detection (KNN score/timeseries), **Security Posture** (~35 panels — the full detail behind Chapter 7.4's ~40 security gauges: `/etc/passwd`/`/etc/shadow`/`/etc/sudoers` integrity, SUID binaries, zombie processes, SSH keys, cron jobs, world-writable files, and more), AI Detection (AI risk score and every `guardian_ai_risk.py` factor), System Status (CPU/mem/disk/GPU/inode), and **System Guardrails** (Allowed/Blocked/Approved/Denied/Invalid counts — see the Guardrails note below, this is the panel that led to discovering `~/aiops-guardrail-lab`). |
| **AIOps Autoencoder Anomaly Detection** | `4b70bb0f-5cca-4f0c-9a01-43b1001d0042` | 11 | The per-model Behavioral Attestation dashboard built up through Phases 1–4 (Chapter 5): reconstruction error/anomaly score, suspect-process tables, CPU%/Mem% by watchdog, GPU metrics, syscall evidence. 16 panels, tracked in-repo as `grafana_autoencoder_dashboard.json` (verified in sync with the live dashboard as of this revision — both 16 panels). |
| **Guardian Metrics - Cross Platform - Windows** | `ad8xfcb` | 18 | Backs `aiops-watchdog-windows.py` (Chapter 4.9): Windows Free Memory, CPU, Free Disk Space (C:), Physical Disk Busy %, Services Not Running, Process Count, OS Info, Network Throughput — one panel per Windows host metric surfaced from `windows_exporter` via Prometheus. |

**Supporting/reference dashboards (4) — one line each, not bespoke Guardian architecture:**

| Title | UID | Notes |
|---|---|---|
| System Metrics + Linked Logs | `metrics-logs` | Custom-built: CPU/Mem/Disk timeseries alongside a systemd-journal log panel — the practical Loki↔Prometheus correlation view, linked from the Flagship's "Go Deeper" panel. |
| Loki Log Dashboard | `loki-logs` | Custom-built: live logs, error rate, top log sources — general-purpose log browsing, also linked from "Go Deeper." |
| Node Exporter Full | `rYdddlPWk` | Standard imported community dashboard (the well-known Node Exporter Full board), not authored for this project. |
| Kubernetes Cluster (Prometheus) | `4XuMd2Iiz` | Standard imported community dashboard — its own description states it was "taken from" a public repo. Only relevant if MicroK8s (Chapter 9) is running. |
| LOKI | `dac964ae-8439-4b4a-b2fa-a108bd2f41dc` | Loki's own bundled default dashboard. |

**Clutter, closed 2026-07-27:** two empty dashboards both literally titled "New dashboard" (`bda60430-8ab0-403c-aed3-42ba053cadca`, `ad4x54q`) were found with no custom content — leftover stubs from UI experimentation. Deleted via the Grafana API same day; confirmed gone via `/api/search`.

#### 3.3.2 Guardrails Discovery (found while auditing this manual, 2026-07-27)

The "System Guardrails" row on the **AI Anomaly Detection** dashboard (Allowed/Blocked/Approved/Denied/Invalid counts) is backed by real, live Prometheus metrics (`guardrail_allow_total`, `guardrail_block_total`, etc., job `guardrail_lab`, port 8015) — tracing this back led to `~/aiops-guardrail-lab/`, a small **untracked** (no git repo) directory containing `guardrail_v1.py` → `guardrail_v3.py` (an incremental learning exercise) and `guardrail_exporter.py` (the version now running as `guardrail-exporter.service`, since 2026-07-19). It classifies actions from a guardrail-decision log against simulated `prod-sim`/`backup-sim` customer data into allow/block/approve/deny/invalid and a low/medium/high/critical risk tier, exporting counts as Prometheus gauges.

This is Guardrails' genuine first version (see `VISION.md`'s Governance Engine note) — real, running, and visible on a dashboard — but built independently of `behavioral_policy.py`'s Phase 5 verification, against simulated rather than real Guardian data, and living outside this repo entirely. Treat "Guardrails" in Chapter 2.4's engine table as **v1 live, not yet integrated** rather than either "not started" or "done" — see `ROADMAP.md`'s Phase 7 for the concrete next step (merging this exporter's classification model with `behavioral_policy.py` into one coherent, framework-mapped Guardrails component).

### 3.4 Loki

- Config: `/etc/loki/config.yml`
- `sudo systemctl {start|stop} loki` (restart is passwordless for this user)
- **Fixed 2026-07-24:** `common.instance_addr` used to auto-detect from the LAN interface and went stale on every DHCP change, breaking internal ring/ingester/scheduler calls with "no route to host" — intermittent since 2026-02-13 (602k+ error lines, 328k+ dropped entries). Fixed by pinning `instance_addr: 127.0.0.1` (single-node Loki never needs to reach itself over the LAN).

### 3.5 Promtail

- Config: `/etc/promtail/config.yml`
- `sudo systemctl {start|stop} promtail`
- Two scrape jobs: `varlogs` (original, covers `/var/log/*.log` only) and **`journal`** (added 2026-07-24, required for Guardian's own systemd services to be queryable in Loki with a `unit` label — `varlogs` alone never covered systemd's own journal output).

### 3.6 Tempo (new — did not exist in April)

- Package `tempo` 3.0.2 (apt.grafana.com), config `/etc/tempo/config.yml`
- Ports: 3200 (HTTP), 4317 (OTLP gRPC — note: **not** the 9095 default, which collides with Loki's own internal gRPC port), 4318 (OTLP HTTP) — all bound and firewalled to localhost only
- `sudo systemctl {start|stop|restart|status} tempo`
- Grafana datasource `tempo` (uid `efsbsavb9ff9cd`), linked to the Loki datasource via `tracesToLogsV2`
- Direct trace-by-ID lookup (`/api/traces/{id}`) works immediately after ingestion; the TraceQL search API (`/api/search?q=...`) requires explicit `start`/`end` params and can lag ingestion by seconds to minutes — don't treat "not found in search" as "didn't ingest."

### 3.7 Alertmanager (new — did not exist in April)

- Manual tarball install, `/usr/local/bin/alertmanager` v0.33.1, config `/etc/alertmanager/alertmanager.yml`
- Port: 9093; `prometheus.yml`'s `alerting.alertmanagers` points here
- `alertmanager.service`, enabled at boot
- Routes to a Slack Incoming Webhook — see Chapter 8 for full detail

### 3.8 Watchdog Agents (Custom Python Services)

All watchdogs read their HTTP port from an env var (default matches the table below), so any can be smoke-tested standalone without touching the live service:
```
WATCHDOG_PORT=18011 /opt/aiops-venv/bin/python aiops-watchdog-knn.py
```

| Service | Script | Port | Purpose |
|---|---|---|---|
| `aiops-watchdog-knn` | `aiops-watchdog-knn.py` | 8011 | Primary ML anomaly detection (KNN) |
| `aiops-watchdog-iforest` | `aiops-watchdog-iforest.py` | 8012 | Secondary ML anomaly detection (Isolation Forest) |
| `aiops-watchdog-autoencoder` | `aiops-watchdog-autoencoder.py` | 8013 | Neural-net anomaly detection — **now a live service**, was experimental-only in April |
| `aiops-watchdog-ml` | `aiops-watchdog-ml.py` | — (collector) | Telemetry → `aiops_data/metrics.csv` every ~5s |
| `aiops-guardian-health` | `aiops-guardian-health.py` | 8014 | Health + Security + AI-risk scoring |
| `aiops-watchdog-windows` | `aiops-watchdog-windows.py` | 8016 | **New.** Threshold health checks over existing `windows_exporter`/Prometheus data — no local agent on Windows hosts |
| `aiops-watchdog-logs` | `aiops-watchdog-logs.py` | 8017 | **New.** Log-anomaly checks per Guardian systemd unit via Loki |
| `aiops-watchdog-priority` | `aiops-watchdog-priority.py` | 8018 | **New.** Cross-source warning triage — ranks ALERTS/Guardian checks/legacy `all_check.service` by tier + novelty |
| `aiops-approval` | `/usr/local/bin/aiops-approval.py` | 8020 | Governance Engine's Human Approval dashboard |
| `disk_watchdog` | `disk_watchdog.py` | — (timer, oneshot) | Runs every 15 min via `disk_watchdog.timer`; `inactive (dead)` between runs is normal, not abandoned |
| `guardrail-exporter` | `~/aiops-guardrail-lab/scripts/guardrail_exporter.py` (**outside this repo**, untracked) | 8015 | Governance Engine's Guardrails, v1 — see Chapter 3.3.2 |

The three ML watchdogs (`knn`/`iforest`/`autoencoder`) now share a common implementation, `watchdog_common.py` — each script only supplies model-specific load/build-input/score logic; metric collection, GPU handling, and Behavioral Attestation attribution are identical across all three and live in the shared module. `aiops-guardian-health.py` was similarly split into `guardian_health.py`/`guardian_security.py`/`guardian_ai_risk.py` (per-engine) with shared state in `guardian_common.py`.

**A second, independent copy of the KNN watchdog can also run in MicroK8s** (namespace `aiops`, deployment `aiops-watchdog-knn`) when MicroK8s is up — a long-lived pod unrelated to the systemd service, pointed at the same local files. As of this revision **MicroK8s is not currently running** (`microk8s status` reports it stopped) — see Chapter 9.

### 3.9 Ports and Endpoints Summary

| Component | Port | Bind | Externally reachable? |
|---|---|---|---|
| Prometheus | 9090 | 0.0.0.0 | Yes (UFW ALLOW) |
| Grafana | 3000 | 0.0.0.0 | Yes (UFW ALLOW) |
| SSH | 22 | 0.0.0.0 | Yes (UFW ALLOW) |
| KNN watchdog | 8011 | 0.0.0.0 | No (UFW ALLOW-127.0.0.1 + DENY-Anywhere) |
| iForest watchdog | 8012 | 0.0.0.0 | No (same pattern) |
| Autoencoder watchdog | 8013 | 0.0.0.0 | No (same pattern) |
| Guardian Health | 8014 | 0.0.0.0 | No (same pattern) |
| Windows watchdog | 8016 | 0.0.0.0 | No (UFW ALLOW-127.0.0.1 + DENY-Anywhere, added 2026-07-27 to close a gap found writing this manual) |
| Logs watchdog | 8017 | 0.0.0.0 | No (same fix) |
| Priority watchdog | 8018 | 0.0.0.0 | No (same fix) |
| Approval dashboard | 8020 | 0.0.0.0 | Yes (explicit UFW ALLOW-Anywhere, intentional) |
| Tempo (HTTP/gRPC/HTTP) | 3200/4317/4318 | 127.0.0.1 | No |
| Alertmanager | 9093 | — | (not checked directly; internal to Prometheus↔Alertmanager) |

**Closed 2026-07-27:** the 8016/8017/8018 gap identified while writing this revision was fixed the same day — matching UFW rules added (ALLOW-127.0.0.1 + DENY-Anywhere, v4 and v6) for all three ports. See Chapter 7.

### 3.10 Summary

The stack has grown from 6 services (April) to 14+, adding a tracing backend, an alert router, three cross-cutting watchdogs, and a governance/approval dashboard — while the underlying five-service core (Prometheus/Grafana/Loki/Promtail + the original 3 ML watchdogs) is unchanged in shape, just refactored for shared code.

---

## Chapter 4 — Custom Scripts and Intelligence Layer

### 4.1 Purpose

Documents the Python scripts forming Guardian's intelligence layer: data collection, ML anomaly detection, health/security evaluation, and Prometheus export.

### 4.2 Architecture Overview

Every watchdog/health script: collect → validate/normalize → analyze (ML or rule-based) → export via `/metrics` → repeat. They're intelligent Prometheus exporters, not one-shot scripts.

### 4.3 Directory Structure

`/home/beth/aiops-agents/` contains watchdog scripts, the Guardian health engine (now split across `guardian_common.py`/`guardian_health.py`/`guardian_security.py`/`guardian_ai_risk.py`), ML models (`.pkl`/`.keras`, gitignored), training/retrain scripts, Behavioral Attestation modules (Chapter 5), `generate_report.py`, the test suite, and this manual plus `VISION.md`/`ROADMAP.md`/`EDGE_ARCHITECTURE.md`.

### 4.4 Feature Set (`DATA_FEATURES`, shared across all three ML models)

```python
DATA_FEATURES = [
    "disk", "disk_free_gb", "disk_fill_rate_mb_min", "inode_pct",
    "cpu", "mem", "net_kbps", "disk_w_kbps",
    "gpu_util", "gpu_mem_mib", "gpu_temp_c",
]
```

Unchanged since April — captures current state, behavior over time, hidden failure conditions, and hardware utilization.

### 4.5 Data Collection and Guardrails

`aiops-watchdog-ml.py` collects real system data into `aiops_data/metrics.csv` every ~5s, with guardrails rejecting impossible values (e.g. CPU > 1000%), filtering corrupted sensor readings, and calculating disk fill-rate (MB/min) for abnormal-growth detection.

### 4.6 KNN Watchdog — Primary Detection Engine

Port 8011. Loads `knn_model.pkl`/`scaler.pkl`, scores the same `DATA_FEATURES`, exports `aiops_anomaly_label`/`aiops_anomaly_score`/`disk_anomaly_prediction`. **This is the model that has drifted into a sustained-anomaly false-positive state four separate times** (2026-07-13, -16, -17, -27), each traced via Behavioral Attestation evidence (Chapter 5) to Promtail's write-syscall pattern sitting near the edge of what a 2000-row training snapshot captures as normal — see Chapter 10 for the open structural question this raises.

### 4.7 Isolation Forest Watchdog

Port 8012, `iforest_model.pkl`. Cross-checks KNN's anomaly calls using the same feature set.

### 4.8 Autoencoder Watchdog

Port 8013, `autoencoder_model.keras` + `autoencoder_threshold.txt`. Reconstruction error as the anomaly signal. **Status update: this is now a live systemd service** (`aiops-watchdog-autoencoder.service`) — the April manual's "functional, not currently running as systemd service" note is stale. Still TensorFlow CPU-only (`Error loading CUDA libraries` in every training log) despite a capable GPU sitting idle — unaddressed, separate future work if GPU-accelerated training is ever wanted.

### 4.9 Windows Watchdog (new, not in April manual)

Port 8016. Architecturally different from the other three: there's no local psutil-based agent on Windows hosts. Instead, hosts are already scraped by Prometheus (`job="windows-node"`, targets are hostnames like `DESKTOP-*:9182`, via `windows_exporter`), and this script queries that existing Prometheus data (`prom_query_vector()`), applies the same threshold-based per-check pattern as `compute_health()`, and re-exposes as `aiops_windows_health_*{instance}`. Deliberately checks a small allowlist of always-on services (`CRITICAL_SERVICES`) rather than "any `start_mode=auto` service not running" — many Windows services report `auto` start mode but use trigger-start semantics (idle/stopped is their normal resting state).

### 4.10 Logs Watchdog (new, not in April manual)

Port 8017. Same external-data-source pattern as the Windows watchdog, but over Loki. Requires Promtail's `journal` scrape job (Chapter 3.5). Per Guardian systemd unit, checks `error_count` (lines matching `[ERROR]` or a raw Python traceback in the last 5m — deliberately narrower than a case-insensitive `error|warn` match, which matched benign sklearn `UserWarning`s on effectively every cycle during design) and `silent` (no log lines at all in 5m while systemd reports the unit active — catches a hung-but-technically-running process). A separate `aiops_logs_query_ok` gauge distinguishes "Loki failed to answer" from "logs are genuinely clean."

### 4.11 Priority Watchdog (new, not in April manual)

Port 8018. Pulls together currently-firing Prometheus `ALERTS`, Guardian's own health/security/AI-risk/Windows/log gauges, and the older `all_check.service` (port 8000, three job names scraping the same target) into one `aiops_priority_score{check, detail, tier}` gauge. Priority = per-tier base weight (ALERTS highest, already 30-min-sustain-vetted; Guardian's own checks next; `all_check.service` lowest — it's un-debounced and has one confirmed permanent false positive, `check_service_status("ssh")` firing forever since sshd was never installed on this host) plus a novelty bonus comparing each check's current state against the same query `offset 6h` (0 = chronic, 20 = genuinely new, 10 = no history yet to compare against — expected to self-resolve as history accumulates).

### 4.12 Guardian Health, Security, and AI-Risk Engines

Now split into three modules (`guardian_health.py`, `guardian_security.py`, `guardian_ai_risk.py`, shared state in `guardian_common.py`), wired together by `aiops-guardian-health.py`. Port 8014.

- **Health** (`compute_health()`): CPU/mem/disk/inode thresholds, aggregate service health
- **Security** (`compute_security()`): grew substantially since April — ~40 new gauges added 2026-07-17 covering file integrity, user monitoring, SSH, and AI-specific security checks (full detail in Chapter 7), on top of the original UFW/updates/root-SSH/failed-logins set
- **AI Risk** (`calculate_ai_risk_score()`): AI process/GPU-spike/API-key-exposure/model-drift/shadow-model risk signals

Output metrics now include `aiops_health_score`, `aiops_security_score`, `aiops_guardian_status`, `aiops_security_issue_code`, `aiops_security_recommendation`, `ai_risk_score`, and the ~40 newer security gauges (Chapter 7).

### 4.13 Training and Retrain Pipeline

**Original training scripts** (train from scratch on full history): `train_knn_final.py`, `train_iforest.py`, `train_autoencoder_final.py`.

**Retrain scripts** (new since April — train on a recent tail of `metrics.csv`, avoiding old-state bias): `retrain_recent_knn.py`, `retrain_recent.py` (autoencoder), `retrain_recent_iforest.py`, sharing common logic in `retrain_common.py`. **Known, currently-live inconsistency:** `RECENT_ROWS = 2000` in `retrain_recent_knn.py` (fixed after a 2026-07-13 incident where the default of 100000 rows pulled in the entire, stale Mar–Jul history) but still `100000` in `retrain_recent.py` and `retrain_recent_iforest.py` — fix the same way (drop to ~2000) before rerunning either of those as-is.

None of the retrain scripts restart the corresponding service automatically — that's a deliberate separate manual step (requires sudo, not passwordless).

Every retrain run is now also instrumented with OpenTelemetry traces and Behavioral Attestation evidence — see Chapter 5.

**`grafana_annotate.py` (undocumented until this revision):** posts a Grafana annotation — a labeled marker on dashboard timelines — when something worth flagging happens, e.g. a Phase 5 behavioral-policy verification failure (Chapter 5.5). Deliberately fails soft: if Grafana is unreachable or `.grafana_token` isn't configured, it prints a warning and moves on rather than breaking the retrain run that called it — annotating a dashboard is never worth crashing a retrain over. Uses the same Grafana service-account token as `talks/build_flagship_dashboard.py` (Chapter 4.17), not the admin login password, so it's independently revocable. Currently wired into `retrain_recent_knn.py` only — `retrain_recent.py` (autoencoder) and `retrain_recent_iforest.py` don't call it yet. Annotations tagged `guardian` surface in the Flagship dashboard's "Guardian Events (Annotations)" panel (Chapter 3.3.1).

### 4.14 Report Generation (new, not in April manual)

`generate_report.py` — pulls live state from Prometheus and Grafana, diffs it against the previous run's snapshot (`report_state.json`), runs a small rule-based "playbook" engine to interpret what the numbers mean (severity-ranked: Critical/High/Medium/Low/Info), and renders a markdown report. This replaces what used to require an ad-hoc AI chat session each time to synthesize (`full_system_report_*.md` files were hand-written precedents). Manual invocation only for now (`python3 generate_report.py`), not yet wired to a systemd timer.

### 4.15 Testing (new, not in April manual)

`test_behavioral_policy.py` (Phase 5 verification logic) and `test_guardian_health.py` (including `_get_ufw_status_cached()` caching behavior and `_score_security_base`) — first test coverage this repo has ever had, run via `pytest` and wired into GitHub Actions (`.github/workflows/tests.yml`) on every push/PR.

### 4.16 Prometheus Integration

All scripts expose `/metrics`, scraped by Prometheus, visualized in Grafana — unchanged pattern from April, just more services doing it.

### 4.17 Talks, Demos, and Dashboard Build Tooling (`talks/`, undocumented until this revision)

A `talks/` directory holds portfolio/presentation materials — not part of the running system, but real, git-tracked repo content:

- **`build_flagship_dashboard.py`** — the actual source of the "Guardian: Flagship Overview" dashboard (Chapter 3.3.1). A one-off build script, not a service: constructs the full dashboard JSON in Python (helper functions for row/stat/gauge panels) and POSTs it to Grafana's API using the same `.grafana_token` as `grafana_annotate.py`. Uses a fixed `uid` (`guardian-flagship`) so re-running it updates the dashboard in place rather than creating duplicates — same gotcha as the general Grafana push pattern (Chapter 3.3). Re-run this script (not hand-edit JSON) whenever the Flagship dashboard needs a new panel — see the 2026-07-27 "Add Priority Warnings panel" commit for the working example.
- **`defect-demo-recording-script.md`** — a ~4-5 minute recording script for a portfolio/talk video, "Catching a Bad AI-Driven Change." Walks through reproducing the real 2026-07-17 Phase 5 defect-catch (Chapter 5.6) on camera, including a safety-first model-file backup step before recording and framing language tying it to Guardian's north-star question (`VISION.md`). Not hypothetical — explicitly scripted to reproduce something that already happened for real, not a staged scenario.

### 4.18 Summary

The intelligence layer has grown from 3 ML watchdogs + 1 health script into 8 watchdog/health services plus a shared-code refactor, a report generator, a real test suite, and a set of portfolio/demo build tools — while the core collect→analyze→export loop from April is structurally unchanged.

---

## Chapter 5 — Behavioral Attestation Engine

**This entire chapter documents capability that did not exist in the April manual.** It is Guardian's fifth engine (`VISION.md`) and answers the question the rest of the system couldn't: not just *"is something wrong,"* but *"what caused it, and did it match what was expected?"*

### 5.1 Phase 1 — Process-Level Attribution

Shared module `process_attribution.py` — top-N processes by CPU/mem via `psutil` (cached `Process` objects for meaningful `cpu_percent()` deltas across the ~5s loop). All three ML watchdogs snapshot the top 5 suspect processes whenever `label==1`, exposing `aiops_anomaly_top_process_cpu_percent`, `_mem_percent`, `_info{pid,name}` (an info gauge, cleared each anomaly to bound cardinality, intentionally retaining the last-flagged suspect between anomalies rather than resetting to nothing).

**Real catches to date:** repeated `promtail` write-burst flags (the KNN drift pattern — benign but recurring); `prometheus-node-exporter-apt.service`'s 15-minute apt-metrics timer showing up as a `dpkg` CPU spike (legitimate, recurring, good future candidate for a Phase 5 "known expected behavior" exception).

### 5.2 Phase 2 — eBPF / System-Activity Correlation

Architecture: `trace_suspect.sh` (root-run wrapper, validated numeric PID arg, fixed 3-second `bpftrace` trace of `openat`/`execve`/`connect`/`write` tracepoints scoped to that PID) + `ebpf_trace.py` (`trace_suspect_process(pid)`, invoked via `sudo -n`, 60-second per-PID cooldown so a sustained anomaly doesn't spawn a fresh privileged trace every tick).

**Key design constraint discovered:** `bpftrace` has a hardcoded `euid==0` check and refuses to run under `CAP_BPF`+`CAP_PERFMON` alone — ruled out a least-privilege capabilities approach. Settled on a narrow `NOPASSWD` sudoers rule (`/etc/sudoers.d/aiops-trace`, restricted to `trace_suspect.sh *`), matching the existing pattern for `ufw`/`systemctl restart` entries.

**Wildcard-sudo gap found and closed (2026-07-29):** the `trace_suspect.sh *` sudoers rule scopes *which script* can run as root, but not *which PID* it can be pointed at — confirmed concretely (not just theoretically) that `beth` could invoke `sudo -n trace_suspect.sh 1` and get a real, successful root-level eBPF trace of `systemd` (PID 1, root's own most-privileged process), with zero connection to anything Guardian's own detection logic flagged. No shell/command injection was ever possible (the real caller passes an argv list, never a shell string, and the script's numeric-PID check runs before the PID reaches the bpftrace program) — the actual gap was "any PID on the system, by any local process, with no rate-limit or audit trail outside the Python-side cooldown." PID-ownership restriction was considered and rejected: `promtail`, this feature's own most-traced real suspect, runs as root, so "only trace beth-owned PIDs" would have broken the exact use case that validated this feature. Fixed instead with a **ticket mechanism**: `ebpf_trace.py` writes the intended PID + timestamp to `.trace_ticket` (mode 0600, next to the repo rather than `/run/user/<uid>` since systemd system services don't get `XDG_RUNTIME_DIR`) immediately before calling `sudo`; `trace_suspect.sh` now refuses to run unless a ticket exists, names the exact PID requested, and is under 2 seconds old. Verified in all four directions: no ticket → refused, wrong-PID ticket → refused, stale ticket → refused, fresh matching ticket → traces successfully. This is `ROADMAP.md`'s Phase 7 item 2 ("govern the tooling engineers hand the agent") closed against its own concrete example. **Live services need a restart to pick this up** — `aiops-watchdog-{knn,iforest,autoencoder}.service` already have the old `ebpf_trace.py` loaded in memory (Chapter 1.7, needs interactive sudo).

Exposes `aiops_anomaly_suspect_syscall_count{syscall_type,pid,name}`. Real findings: `promtail` doing ~90-120 single-byte `write()` calls/sec (inefficient I/O); the `dpkg`/apt-timer spike upgraded from correlation to hard proof (29,686 `open()` calls in one 3-second window).

### 5.3 Phase 3 — AI Workflow Traceability

New infra: Grafana **Tempo** (Chapter 3.6). Shared helper `otel_setup.py` (`get_tracer()`, uses `SimpleSpanProcessor` deliberately — not batched — since these are short one-shot scripts where a batch processor could exit before flushing). All three `retrain_recent*.py` scripts get a root span (`retrain_<model>_run`) plus `load_data`/`train_model`/`save_model` child spans with meaningful attributes (row counts, anomaly counts, thresholds, file paths). Each retrain run prints its own trace ID at start (`[INFO] Trace ID: <hex>`) — permanent feature for jumping straight to that run's trace.

**Cross-phase linking:** the `train_model` span also captures Phase 1/2-style evidence of *its own* process (reusing `ebpf_trace.py`/`psutil` pointed at `os.getpid()` instead of a watchdog-detected suspect) — `process.cpu_percent`, `process.mem_percent`, `ebpf.syscall.*`, `ebpf.files_opened`.

### 5.4 Phase 4 — GPU Activity Correlation

Shared module `gpu_attribution.py` (`get_gpu_processes()`, `get_gpu_usage_for_pid()`, `poll_max_gpu_usage()`) via `pynvml` — matches `nvidia-smi` exactly (verified directly). Wired into watchdogs (`aiops_anomaly_top_process_gpu_mem_mib`) and retrain scripts (`gpu.used_memory_mib` span attribute, polled via a `threading.Event` for the actual training duration rather than a fixed window — matters most for the autoencoder's multi-epoch fit).

### 5.5 Phase 5 — Runtime Behavioral Verification

`behavioral_policy.py` — a `POLICIES` dict (one entry per workflow: `retrain_knn`/`retrain_iforest`/`retrain_autoencoder`) specifying expected files written, max GPU MiB, max outbound `connect()` calls (0 for all three — doubles as a lightweight exfiltration tripwire), min/max row-count bounds. `verify(workflow_name, **evidence)` returns `{"passed": bool, "violations": [...]}`.

Results attach to the same OTel trace as a `verify_behavior` span rather than a new Prometheus metric (a one-shot script exits before the next scrape). "Files touched" is checked via mtime against run-start time, not the eBPF trace's `files_opened` (the 3s eBPF window runs during training, before `save_model` — it structurally can't see the model-file writes).

**Verified both directions:** a real KNN retrain passes cleanly (`verification.passed=True`); a direct call with deliberately bad evidence (missing file, 512 MiB GPU, 3 connects, `row_count=50`) caught all four violations with specific messages.

Now covered by `test_behavioral_policy.py` (Chapter 4.15).

### 5.6 Phase 6 — AI-Managed Release Traceability

`release_record.py` — `get_build_provenance()` reads git (commit, subject, author, date, branch, files-changed stat), `record_release()` combines that with a workflow's trace ID and Phase 5 verification result into a JSON artifact under `releases/`. Also usable as a CLI.

**Proved both directions on real (not fabricated) data**, both committed and pushed:
- `releases/guardian-behavioral-attestation-2026-07-17.json` — genuine record for commit `548fdd6`, `passed: true`
- `releases/guardian-defect-demo-2026-07-17.json` — deliberately dropped KNN's `RECENT_ROWS` to 20 (below the Phase 5 policy's `min_rows: 100`), ran it for real, correctly caught (`"row_count 20 below policy minimum 100"`). Local/uncommitted-code-change only, by design — model files backed up/restored, live service never touched, code change reverted via `git checkout --`.

`releases/` is treated as a permanent, growing git-tracked ledger — expect one JSON file per future real release.

### 5.7 Summary

All 6 originally-scoped roadmap phases have concrete, verified implementations, each proven against real system behavior rather than synthetic tests alone. A proposed **Phase 7** ("govern the agent, not just the output") exists only as a scoped problem statement so far — see Chapter 10.

---

## Chapter 6 — Operations and Runbook

### 6.1 Purpose

Day-to-day operations guide: startup/shutdown, service management, patch/reboot, verification, troubleshooting.

### 6.2 Core Operational Principles

Unchanged from April: manage services via systemd, verify monitoring after any change, keep system state observable, validate changes against metrics and dashboards.

### 6.3 Service Inventory (current, supersedes the April table)

| Service | Description |
|---|---|
| prometheus | Metrics collection |
| grafana-server | Visualization |
| loki | Log aggregation |
| promtail | Log shipping |
| tempo | Trace storage (new) |
| alertmanager | Alert routing (new) |
| aiops-watchdog-knn | Primary anomaly detection |
| aiops-watchdog-iforest | Secondary anomaly detection |
| aiops-watchdog-autoencoder | Neural-net anomaly detection (now live) |
| aiops-watchdog-ml | Telemetry collector |
| aiops-guardian-health | Health/Security/AI-risk evaluation |
| aiops-watchdog-windows | Windows fleet health (new) |
| aiops-watchdog-logs | Log-anomaly checks (new) |
| aiops-watchdog-priority | Cross-source priority triage (new) |
| aiops-approval | Human approval dashboard |
| disk_watchdog.timer | 15-min disk check (timer, not continuous) |

### 6.4 Start All Services

```
# Core observability stack
sudo systemctl start prometheus grafana-server loki promtail tempo alertmanager

# Guardian services
sudo systemctl start aiops-watchdog-knn aiops-watchdog-iforest aiops-watchdog-autoencoder
sudo systemctl start aiops-watchdog-ml aiops-guardian-health
sudo systemctl start aiops-watchdog-windows aiops-watchdog-logs aiops-watchdog-priority
sudo systemctl start aiops-approval
```

Host reboots restart all of these automatically and cleanly (`Restart=always`/`on-failure`, verified during a real reboot with no metrics-data gap).

### 6.5 Verify System Status

```
sudo systemctl status <service>
curl http://localhost:8011/metrics   # KNN
curl http://localhost:8012/metrics   # iForest
curl http://localhost:8013/metrics   # Autoencoder
curl http://localhost:8014/metrics   # Guardian Health
curl http://localhost:8016/metrics   # Windows
curl http://localhost:8017/metrics   # Logs
curl http://localhost:8018/metrics   # Priority
curl http://localhost:9090/targets   # all should show UP
curl http://localhost:9090/api/v1/alerts   # current alert state
```
Grafana: `http://localhost:3000` — verify no missing panels / "No Data" errors.

### 6.6 Safe Shutdown Procedure

```
# Guardian services first
sudo systemctl stop aiops-approval aiops-watchdog-priority aiops-watchdog-logs aiops-watchdog-windows
sudo systemctl stop aiops-guardian-health aiops-watchdog-ml
sudo systemctl stop aiops-watchdog-autoencoder aiops-watchdog-iforest aiops-watchdog-knn

# Then the observability stack
sudo systemctl stop alertmanager tempo promtail loki grafana-server prometheus

# Optional
microk8s stop   # only if it was running
```

### 6.7 Patch and Reboot Procedure

```
sudo apt update && sudo apt upgrade -y
sudo reboot
# post-reboot:
sudo systemctl status aiops-guardian-health
curl http://localhost:8014/metrics
```

### 6.8 Troubleshooting Guide

| Issue | Check |
|---|---|
| Service not running | `sudo systemctl status <service>` |
| Service fails to start | `sudo journalctl -u <service> -n 50 --no-pager` |
| Port not responding | `ss -tlnp \| grep <port>` |
| Prometheus target DOWN | service running? correct port? script errors? |
| No data in Grafana | Prometheus running? query correct? datasource connected? |
| Alert not reaching Slack | `curl localhost:9090/api/v1/alertmanagers` shows Alertmanager connected? check `/etc/alertmanager/alertmanager.yml` webhook URL for stray characters (see Chapter 8's real incident) |
| Trace not found in Tempo search | try direct `/api/traces/{id}` lookup instead — search can lag ingestion by seconds to minutes |
| Stray k8s pod | `microk8s kubectl get pods -A` before assuming it's abandoned — a long-running KNN pod is expected when MicroK8s is up |

### 6.9 Common Failure Scenarios

Unchanged categories from April (port conflicts, script exits immediately, missing service unit, DHCP breaking Kubernetes certs) — plus, new: **KNN sustained-anomaly drift** is now a recognized recurring pattern (4 occurrences as of this revision), not a fresh incident each time — see Chapter 10 before re-diagnosing from scratch.

### 6.10 Operational Checks

**Daily:** Grafana dashboards, Prometheus targets, Guardian Health metrics, Prometheus `/alerts` for anything firing.
**Weekly:** Loki logs review, disk growth trends, security metrics, whether KNN has drifted again (retrain if score is climbing away from its ~0.1-0.14 baseline).

### 6.11 Recovery Checklist

1. Verify OS running → 2. Start Prometheus → 3. Start Grafana → 4. Start Loki/Promtail/Tempo → 5. Start Alertmanager → 6. Start Guardian services → 7. Verify metrics endpoints → 8. Check dashboards → 9. Confirm Alertmanager shows connected to Prometheus.

### 6.12 Summary

The runbook now covers roughly 2.5x the services it did in April, but the operating principles (systemd-managed, verify-after-change, always-observable) haven't changed — only the inventory has grown.

---

## Chapter 7 — Security, Governance, and Risk Management

### 7.1 Purpose

Security controls, governance practices, risk management — updated to reflect the substantially larger Security Engine built 2026-07-17, and to be honest about a gap found while writing this revision.

### 7.2 Security Philosophy

Unchanged: visibility first, least disruption, actionable insights, continuous evaluation.

### 7.3 Current Security Posture (verified live, this revision)

UFW is **active** (default deny incoming/routed, allow outgoing) — this supersedes the April manual's example of "Firewall: Disabled," which was illustrative/stale even at the time of writing this revision. Current rules:

```
22, 3000, 9090, 8020        ALLOW IN  Anywhere        (SSH, Grafana, Prometheus, Approval dashboard)
8011-8014                   ALLOW IN  127.0.0.1  +  DENY IN  Anywhere   (watchdog ports)
3200, 4317, 4318            ALLOW IN  127.0.0.1  +  DENY IN  Anywhere   (Tempo)
```

**Gap found while writing this revision, closed same day (2026-07-27):** ports **8016, 8017, 8018** (Windows/Logs/Priority watchdogs) were bound to `0.0.0.0` with no corresponding UFW rule at all — unlike 8011-8014, which follow the established ALLOW-127.0.0.1 + DENY-Anywhere pattern. These three ports were added earlier this session without extending that pattern to them. Matching rules (v4 and v6) have since been added for all three, verified via `ufw status verbose`.

The bind-address-vs-firewall distinction from April still holds: watchdog ports bind to `0.0.0.0` at the socket level; UFW's explicit DENY rules are what actually keep them non-external, not the bind address — `check_watchdog_port_external_access()`/`_ufw_denies_port_externally()` (in `guardian_ai_risk.py` now) check ufw rules directly (both v4 and v6) rather than only testing raw reachability.

### 7.4 Security Metrics (grew substantially, 2026-07-17)

Original set: `aiops_security_score`, `_issue_code`, `_recommendation`, `_ufw_enabled`, `_root_ssh_enabled`, `_updates_pending`, `_failed_logins_recent`.

**~40 new gauges added 2026-07-17**, implementing the Security Engine's remaining named categories from `VISION.md` (File Integrity, User Monitoring, SSH, AI Security) — full detail is in `guardian_security.py`/`guardian_ai_risk.py` rather than duplicated here, since the module is the source of truth and this manual would otherwise drift out of sync again.

### 7.5 Security/Recommendation Issue Mapping

Unchanged from April (codes 0-4: none / firewall disabled / updates pending / SSH risk / multiple issues; recommendations 0-3: none / enable firewall / apply updates / secure SSH).

### 7.6 AI Risk Awareness

`guardian_ai_risk.py`'s `AI_RISK_SCORE` and related gauges (`AI_API_KEYS_PRESENT`, `AI_EXPOSED_KEYS`, `AI_GPU_SPIKE`, `AI_LLM_CONNECTIONS`, `AI_MODEL_AGE_DRIFT`, `AI_PROCESSES_RUNNING`, `AI_SHADOW_MODELS`) — this category barely existed as a concept in the April manual ("AI and Automation Risk Awareness" was a short forward-looking section); it's now a concrete, scored engine.

### 7.7 Governance Practices

The Human Approval piece of the Governance Engine (`aiops-approval.service`, port 8020) already exists and predates this revision. Guardrails now also has a real v1 (`guardrail-exporter.service`, Chapter 3.3.2) — found during this revision's audit, running since 2026-07-19 but built outside this repo against simulated data, independent of `behavioral_policy.py`'s per-workflow rules (Chapter 5.5). The two are not yet integrated: one classifies logged actions into allow/block/approve/deny tiers, the other verifies workflow evidence against declarative policies — merging them into one coherent, framework-mapped Guardrails component is `ROADMAP.md`'s Phase 7 next step, not a from-scratch build.

### 7.8 Authentication Monitoring, Patch Management, SSH Security

Unchanged commands/approach from April (`journalctl` for failed logins, `apt update && apt list --upgradable`, `grep PermitRootLogin`).

### 7.9 Operational Security Checks

Same daily/weekly cadence as April, plus: review Prometheus `/alerts` and confirm Alertmanager delivery is still working (Chapter 8).

### 7.10 Summary

Security moved from "four checks and a score" (April) to a ~44-gauge engine plus a live firewall posture plus a dedicated AI-risk scoring layer — genuine growth, with one known, low-severity firewall-coverage gap (8016-8018) surfaced by writing this revision rather than by an incident.

---

## Chapter 8 — Alerting and Incident Response

**This entire chapter is new — no alerting infrastructure existed at all when the April manual was written** (`rule_files: []`, no Alertmanager).

### 8.1 Prometheus Alerting Rules

`/etc/prometheus/rules/aiops-alerts.yml`, referenced from `prometheus.yml`'s `rule_files`. Three rules, one per ML watchdog, all `for: 30m` on `aiops_anomaly_label{job="..."} == 1`, severity `warning`: `KNNWatchdogSustainedAnomaly`, `IForestWatchdogSustainedAnomaly`, `AutoencoderWatchdogSustainedAnomaly`. The 30-minute sustain window is why the Priority watchdog (Chapter 4.11) treats ALERTS-sourced signals as already-vetted, higher-tier than raw Guardian checks.

**`WindowsHostUnreachable` (added 2026-07-29):** `up{job="windows-node"} == 0` for `5m`, severity `warning`. Added after `DESKTOP-0AJUKU3` crashed and this had zero alerting coverage — Guardian's priority watchdog had already flagged it (Chapter 4.11) but nothing pushed it to Slack, so it was only found by asking. The 5-minute window is deliberate: a normal reboot's downtime (observed the same morning: ~45 seconds) should self-resolve without paging, but a second, unresolved outage 19 minutes later should not go unnoticed the way it did. This rule only detects unreachability — it can't distinguish a real OS crash from a network/firewall/service issue, since Guardian has no local agent on Windows hosts, only remote `windows_exporter` scraping (Chapter 4.9). For the actual cause, check the host directly: Event Viewer's System log, Event ID 41 (unexpected shutdown) or 1001 (BugCheck with stop code).

### 8.2 Alertmanager

v0.33.1, manual tarball install (same pattern as Prometheus itself), `/etc/alertmanager/alertmanager.yml`, `alertmanager.service` (enabled). `prometheus.yml`'s `alerting.alertmanagers` points at `localhost:9093`; confirm connection via `curl localhost:9090/api/v1/alertmanagers`.

Routes to a Slack Incoming Webhook. **Real incident during setup:** a stray trailing single-quote in the webhook URL (no matching opening quote) made every notify attempt fail, with Slack redirecting to its docs 404 page — manual `curl`/`urllib` tests kept misleadingly succeeding because the diagnostic script's own `.strip("'")` silently removed the bad character before testing. Fixed and verified end-to-end (a real `KNNWatchdogSustainedAnomaly` alert confirmed landing in Slack with full annotation text).

### 8.3 Current Scope and Known Limits

- Only Slack is wired up — no email/PagerDuty-style escalation. Deliberately deferred (user chose not to add another external account yet).
- No alert-fatigue tuning yet — grouping/inhibition rules in `alertmanager.yml` are still the initial pass.
- Alerting covers the three ML watchdogs' sustained-anomaly case and, as of 2026-07-29, Windows host unreachability. Guardian Health/Security/AI-risk scores and the Logs watchdog still don't have their own Prometheus alerting rules (they do feed into the Priority watchdog's triage score, Chapter 4.11, but that's a dashboard signal, not a push notification).

### 8.4 Incident Response Reference

For the recurring KNN sustained-anomaly case specifically: see Chapter 10 before treating it as a fresh incident — check `curl localhost:8011/metrics` for the current score, and if it's genuinely climbing (not just past its 30-min sustain threshold once), retrain per Chapter 4.13 and restart the service (Chapter 1.7 — needs interactive sudo).

### 8.5 Summary

Alerting went from nonexistent to a working Prometheus→Alertmanager→Slack pipeline, verified against a real alert, in one session — the main remaining gap is breadth (more rule coverage, a second notification channel) rather than whether the pipeline itself works.

---

## Chapter 9 — Scalability and Future Architecture

### 9.1 Purpose

How Guardian could evolve from single-machine to multi-site/multi-node — updated to reflect what MicroK8s actually demonstrated versus the April manual's more aspirational framing, plus the new near-term direction (edge consulting deployment) that has since taken priority over "scale this box up."

### 9.2 Current Architecture (Single Node)

Unchanged shape from April: local watchdog agents, local Prometheus/Grafana/Loki, now also local Tempo/Alertmanager. **Status update: MicroK8s is currently not running** (`microk8s status` reports stopped) — treat any reference to a live KNN pod (Chapter 3.8) as conditional on MicroK8s being started, not a permanent parallel deployment.

### 9.3 What MicroK8s Has Actually Demonstrated (honest accounting, was missing from April manual)

MicroK8s was installed to test whether these scripts could run at enterprise scale — worth being clear-eyed about what a single-node deployment does and doesn't prove:

- It proves the scripts **can be containerized/orchestrated**. It does **not** exercise multi-node scheduling, cross-machine network policy, or resource contention under real distributed load — that needs more than one node to observe at all.
- **Bigger blocker:** the watchdogs assume local-filesystem, singleton state (`metrics.csv`, `.pkl`/`.keras` model files, appending writes). Scaling `aiops-watchdog-knn` to multiple replicas today would have them fighting over the same local files, not actually scaling horizontally.
- Current state, when running: one k8s-deployed KNN replica runs in parallel with the systemd instance, both pointed at the same local files — a reasonable "learned the orchestration primitives" milestone, not a scale test.

**Path to an actual enterprise-scale test, in order:** (1) make the watchdogs stateless first — move `metrics.csv` off local disk (real DB or Prometheus remote-write), move model artifacts to shared storage read-only by replicas; (2) get real node boundaries cheaply (`microk8s add-node` across a few cloud VMs — `kind` with multiple worker containers tests scheduling logic but doesn't substitute for real node isolation); (3) test things that actually demonstrate "enterprise-ready" — N replicas evenly loaded, a killed node rescheduling automatically, a `NetworkPolicy` actually isolating a namespace.

### 9.4 New Near-Term Direction: Edge Deployment for Consulting (not in April manual)

Full detail lives in `EDGE_ARCHITECTURE.md` (added 2026-07-27) — summarized here since it now supersedes "scale this laptop with MicroK8s" as the more concrete next step:

- **One Raspberry Pi per customer site** acts as a "Guardian Edge Collector" — scrapes exporters (`node_exporter`/`windows_exporter`), receives evidence from a small **Guardian endpoint agent** on each monitored server, does first-pass anomaly detection, buffers, and forwards to "Guardian Core" (this laptop/Dell, running today's full stack).
- The endpoint agent is deliberately tiny (read-only checks, hash evidence, push to the Pi) and will eventually need **both a Windows and a Linux/Unix version** — stated requirement, no implementation decisions made yet.
- Push-based data flow is a requirement, not a preference — a consultant's laptop typically can't reach into a locked-down customer network; Promtail's existing push-to-Loki model is the template.
- Multi-tenant labeling (a `site`/`customer` label on every series) is a from-day-one design constraint if this is ever built for real, to avoid data collision across customers in the shared central Loki/Prometheus.
- **Status: planning only.** No Raspberry Pi purchased, no agent code written. Milestone-based rollout is explicitly planned to start slow: prove Pi→Core forwarding on two existing exporters first, before writing any endpoint-agent code.

### 9.5 Federation and Aggregation (unchanged from April, still future)

Prometheus Federation / Thanos for larger scale — not started, no change since April.

### 9.6 Design Principles for Scaling

Unchanged from April: start simple, scale incrementally, validate at each step, maintain observability throughout. The edge-deployment plan (9.4) follows this same philosophy explicitly (its own milestone plan is "prove one thing, then find the next question," not "design the whole platform first").

### 9.7 Summary

The scaling story has bifurcated since April: the original MicroK8s/multi-node vertical-scaling thread is honestly assessed as early-stage and currently paused (not running), while a new, more concrete horizontal-deployment thread (edge collectors for a future consulting business) has emerged and is the more likely next real build — though it too is still pre-hardware, planning-only.

---

## Chapter 10 — Known Gaps and Roadmap

This chapter did not exist in the April manual in this form — it consolidates the honest, self-critical gap analysis that's accumulated in `ROADMAP.md` since, so a reader gets the same unvarnished picture without needing to cross-reference the roadmap file separately.

### 10.1 Closed Since April (genuine progress, not just new features)

- **No test suite → real test suite + CI.** `pytest` coverage for `behavioral_policy.py` and `guardian_health.py`, running in GitHub Actions on every push/PR.
- **No alerting reaching a human → working Slack pipeline.** Chapter 8.
- **60 unnoticed `ruff` lint errors → fixed**, including a real logic bug (`_ufw_denies_port_externally()` missing IPv6-only DENY rules).
- **Autoencoder "not running as a service" → live systemd service.**
- **No rule-based interpretation of live state → `generate_report.py`.**
- **8016/8017/8018 had no UFW coverage → fixed same-day.** Found while writing this revision, closed by adding the same ALLOW-127.0.0.1 + DENY-Anywhere pattern already used for 8011-8014.
- **2 empty "New dashboard" stubs → deleted.** Found during the dashboard-inventory audit, removed via the Grafana API same day.
- **`trace_suspect.sh`'s unrestricted wildcard sudo argument → fixed (2026-07-29).** Confirmed exploitable (not just theoretical) as a root-level eBPF trace of PID 1, unrelated to any Guardian detection. Closed with a ticket mechanism — see Chapter 5.2 for full detail. Live watchdog services still need a restart to load the fix (Chapter 1.7).
- **No alert for Windows host unreachability → `WindowsHostUnreachable` rule added (2026-07-29).** Found the gap directly: `DESKTOP-0AJUKU3` crashed and Guardian's priority watchdog flagged it, but nothing pushed it to Slack — see Chapter 8.1.

### 10.2 Still Open

- **`RECENT_ROWS` inconsistency** across `retrain_recent*.py` scripts (Chapter 4.13) — nothing currently checks sibling scripts stay consistent; this is a process gap, not just a one-time bug.
- **KNN sustained-anomaly drift, 4th recurrence as of this revision** (2026-07-13, -16, -17, -27) — same promtail-write-rate signature each time. Retraining has worked as a stopgap every time, but if it recurs a 5th time, the right fix is probably structural (a training window that better captures promtail's write-burst variability, or handling that feature's sensitivity differently) rather than "retrain again."
- **No secrets manager** — config lives in plaintext systemd env drop-ins.
- **No staging environment** — this box is prod, dev, and workstation simultaneously; every change is tested live.
- **No documented rollback procedure** beyond `git checkout --` for a single file; `Restart=always` is a crash-loop safety net, not an operations strategy.
- **Alerting breadth** — Guardian Health/Security/AI-risk and the Logs watchdog still have no Prometheus alert rules of their own; only Slack is wired up (Chapter 8.3).
- **Documentation-vs-reality drift, found via a full audit this revision (2026-07-27):** three real implementations existed with zero documentation anywhere (a second Grafana dashboard, `grafana_annotate.py`, the `talks/` build tooling — all now fixed in this manual), a dashboard-naming collision in this manual's own earlier draft (now disambiguated, Chapter 3.3), `README.md`/`Futurework.md` were describing an April-or-earlier snapshot against a ~30-script repo (retired/rewritten, see below), and Guardrails turned out to have a real v1 that VISION.md and this manual both incorrectly described as "not yet formalized" (now corrected). Worth periodically re-running this kind of audit (grep every script against every doc, check live Grafana/Prometheus state against what docs claim) rather than assuming docs stay accurate as the system grows.

### 10.3 Proposed, Not Yet Scoped: Phase 7 — Govern the Agent, Not Just the Output

From `ROADMAP.md` (proposed 2026-07-20): Phase 6 proves what an AI-driven change *did*; this would prove the agent making changes was *governed while doing it* — the distinction a real audit actually cares about. Four concrete, currently-missing pieces, in rough priority order:

1. **A scoped identity for the agent**, not the developer's own full, unscoped session (today, Claude Code operates with the same sudo/git/filesystem reach as the human).
2. **Govern the tooling engineers hand the agent, not just the agent's identity** (clarified 2026-07-27 — the actual motivating idea behind `~/aiops-guardrail-lab`, Chapter 3.3.2). Distinct from item 1: a systems engineer can cause an AI agent to do things it shouldn't *by accident*, through overly broad scripts/wrappers/sudoers rules, no bad intent required on either side. **First concrete instance found and closed (2026-07-29):** `trace_suspect.sh`'s passwordless sudo accepted an unrestricted wildcard PID argument — confirmed exploitable as a real root-level eBPF trace of PID 1 (Chapter 5.2), fixed with a ticket mechanism rather than PID-ownership restriction (promtail, the feature's real validating use case, runs as root). Audit every other script an agent is expected to invoke for the same class of gap — over-broad arguments or missing input validation — as its own ongoing checklist, not a one-time fix.
3. **Map `behavioral_policy.py`'s `POLICIES` to a real framework** (ISO 42001 and/or NIST AI RMF) instead of an ad-hoc dict.
4. **Genuinely tamper-evident logging** — today's evidence trail (journal, OTel/Tempo, `release_record.py` JSON) is observable but not cryptographically signed or append-only. Hash-chaining `releases/*.json` (each record includes the previous one's hash) is the simplest starting point before reaching for anything heavier.

Not fully designed yet — a real gap worth tracking, not a plan ready to execute.

### 10.4 If This Were Gated for a Real Pilot

Minimum bar before touching real customer infrastructure (per the 2026-07-20 operational-readiness review): a broader test suite than exists today, sockets bound correctly rather than relying on the firewall alone (closing 10.2's gap), and at least one documented "if this breaks, do X" runbook per service beyond what Chapter 6 already provides.

### 10.5 Summary

Guardian's architecture has consistently outpaced its own operational hygiene since day one — this isn't new to this revision, it's a pattern the project's own roadmap has tracked and partially closed each time it's been reviewed. The honest position: genuinely impressive design work (Behavioral Attestation, working end-to-end alerting, real provenance records) sitting on top of a system that would not yet pass a strict production-readiness gate — and that gap, not new features, is where the highest-leverage next work probably is.
