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
| `aiops-watchdog-windows` | `aiops-watchdog-windows.py` | 8016 |
| `aiops-watchdog-logs` | `aiops-watchdog-logs.py` | 8017 |
| `aiops-watchdog-priority` | `aiops-watchdog-priority.py` | 8018 |

Restarting any of these needs `sudo` and is **not** in this user's passwordless
sudoers list (only `ufw_guard.sh` — a wrapper around `ufw`, not `ufw` directly,
see "Other notes" below — `trace_suspect.sh`, and restarting
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

**`aiops-watchdog-windows.py`** is architecturally different from the other
watchdogs: Windows hosts aren't collected locally (there's no psutil-on-Windows
agent), they're already instrumented by `windows_exporter` and scraped by
Prometheus under `job="windows-node"` (see `prometheus.yml`; targets are
`DESKTOP-*:9182` hostnames, not IPs). This script queries that existing
Prometheus data via its HTTP API (`prom_query_vector()`), applies
threshold-based health checks per discovered instance (cpu/mem/disk/service,
mirroring `compute_health()`'s 100-minus-N-per-failed-check pattern), and
re-exposes the result as `aiops_windows_health_*{instance}` gauges. The
service check deliberately checks a small allowlist of always-on services
(`CRITICAL_SERVICES`) rather than "any `start_mode=auto` service that isn't
running" — many Windows services (Google Updater, AppXSvc, MapsBroker,
`sppsvc`, ...) report `start_mode=auto` but use trigger-start semantics, so
idle/stopped is their normal resting state, not a fault.

**`aiops-watchdog-logs.py`** is a log-anomaly watchdog over Loki, same
external-data-source pattern as the Windows watchdog. It requires Promtail's
`journal` scrape job (added 2026-07-24 alongside this watchdog — the older
`varlogs` job only covers `/var/log/*log`, not systemd service output) so
Guardian's own services are actually queryable in Loki with a `unit` label.
Per Guardian systemd unit, it checks `error_count` (lines matching
`[ERROR]` or a raw Python traceback in the last 5m — deliberately not a
broader case-insensitive `error|warn` match, which during design matched
the benign sklearn `UserWarning` on effectively every cycle) and `silent`
(no log lines at all in 5m while systemd reports the unit active, catching
a hung-but-technically-running process). A separate `aiops_logs_query_ok`
gauge distinguishes "Loki itself failed to answer this cycle" from "the
unit's logs are clean" — an earlier draft used `query_result or 0.0`,
which silently reported 0 errors on a failed query, indistinguishable from
a verified zero. LogQL detail: the error-match pattern is backtick-quoted
(`` `\[ERROR\]|...` ``) because double-quoted LogQL strings go through
Go-style string unescaping before regex compilation, where `\[` isn't a
valid string escape.

**Loki reliability note:** Loki's `common.instance_addr` was previously
auto-detected from the LAN interface at startup, which went stale on every
DHCP address change and broke internal ring/ingester/scheduler calls with
"no route to host" until the next restart — live intermittently since
2026-02-13 (602k+ error log lines, 328k+ dropped log entries via
Promtail's `promtail_dropped_entries_total{reason="ingester_error"}`)
before being caught and fixed on 2026-07-24 by pinning
`instance_addr: 127.0.0.1` in `/etc/loki/config.yml` (outside this repo).
Single-node Loki never needs to reach itself over the LAN, so loopback is
the correct fix, not a workaround.

**`aiops-watchdog-priority.py`** cross-source warning triage: pulls
together currently-firing Prometheus `ALERTS`, Guardian's own
health/security/AI-risk/Windows/log gauges, and the older, looser
`all_check.service` (`/opt/aiops/all_check.py`, outside this repo, port
8000 under three job names — `all_check_agent`/`disk_anomaly`/
`disk_anomaly_ml` all scraping the same target) into one
`aiops_priority_score{check, detail, tier}` gauge. Priority = a per-tier
base weight (ALERTS highest, already vetted by a 30-min sustain
requirement; Guardian's own checks next; all_check.service lowest, since
it's un-debounced and has at least one confirmed permanent false positive
— `check_service_status("ssh")` fires every ~15s forever because sshd was
never installed on this host) plus a novelty bonus from comparing each
check's current bad/healthy state against the same query `offset 6h`: 0 if
it was already bad back then too (chronic, deprioritized without any
hand-maintained exclusion list), 20 if it was healthy then and bad now
(genuinely new), 10 (neutral) if there's no data that far back to compare
against yet. That third case is the common one for a while after any
Prometheus restart — see the reliability note below — and is expected to
self-resolve into real 0/20 verdicts as history accumulates, not a bug.

An earlier version used `changes(gauge[6h])` instead of the offset
comparison, on the theory that a low change count meant "hasn't moved,
therefore stable." That's wrong: `changes()` can't distinguish "flipped
once and has stayed bad since" from "was already bad before the window
even started" — both produce zero recorded changes within the window. It
scored the chronic ssh case at the *maximum* possible bonus instead of
zero. Also note PromQL requires `offset` directly after the bare selector,
before any comparison (`sel offset 6h == 0`) — `(sel == 0) offset 6h` is a
parse error, which is why `CHECKS` stores selector and comparator
separately rather than as one combined PromQL string.

**Prometheus reload vs. restart:** `--web.enable-lifecycle` is enabled on
this Prometheus, so scrape-config-only changes should go through
`curl -X POST http://localhost:9090/-/reload`, not
`systemctl restart prometheus`. Discovered why this matters while
debugging the priority watchdog on 2026-07-27: the newest *compacted*
(on-disk, durable) TSDB block was found to cover only up to 2026-07-24
11:00 — a ~69-hour gap versus "now" — meaning days of recent history
existed only in the in-memory head block/WAL, not on disk. A `systemctl
restart` (done three times that day, once per new watchdog's scrape job)
risks losing however much of that uncompacted window doesn't survive WAL
replay; empirically only ~30-45 minutes reliably came back after the most
recent one. A reload avoids the restart entirely for config-only changes.

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

**`retrain_recent*.py`** (one per model) retrain against a recent window of
`aiops_data/metrics.csv`, instrumented with OTel spans and
`behavioral_policy.verify()`. They don't restart the corresponding service —
that's a separate manual step. `RECENT_ROWS` (2000) and `RECENT_WINDOW_HOURS`
(48) are single shared constants in `retrain_common.py` — all three scripts
import them, none defines a local copy (this closed a class of drift bug where
`retrain_recent_iforest.py` silently kept a `100000` default long after the
others were fixed). The KNN and IForest runs are now *identical* bar the model
constructor, so the whole flow lives in `retrain_common.run_simple_retrain()`
and each script is just a config block; `retrain_recent.py` (autoencoder) keeps
its own body — its window selection (resume-transient filtering) and save step
(threshold file) genuinely differ.

## Other notes

- **`ufw_guard.sh` (added 2026-07-30):** the passwordless sudoers entry for
  `ufw` used to point at the raw `/usr/sbin/ufw` binary with no argument
  restriction at all (`(ALL) NOPASSWD: /usr/sbin/ufw`) — anything running as
  `beth`, including an agent session, could passwordlessly run `ufw disable`
  or `ufw --force reset`, removing the DENY rules several watchdog ports rely
  on as their only defense. The sudoers entry now points at
  `ufw_guard.sh` instead, which blocks `disable`/`reset`/`default` and passes
  everything else (status, allow, deny, delete) straight through to real
  `ufw` — same governance pattern as `trace_suspect.sh`'s ticket mechanism
  (sudoers scopes which script runs as root, the script scopes what it
  actually does). `guardian_common.py`'s `_get_ufw_status_cached()` calls the
  wrapper now, not raw `ufw` — if you ever see `sudo ufw ...` in new code,
  route it through `ufw_guard.sh` instead or it'll fail passwordlessly.
- `full_system_report_*.md` / `system_report_*.md` are point-in-time snapshot
  reports and are gitignored — don't expect them to be tracked or complete.
- `disk_watchdog.py` runs via `disk_watchdog.timer` (every 15 min, oneshot),
  not continuously — `systemctl status` showing `inactive (dead)` between runs
  is normal, not a sign it's abandoned.
- `/etc/prometheus/rules/aiops-alerts.yml` defines sustained-anomaly alerts
  for all three watchdogs and is loaded/evaluating in Prometheus, but
  Alertmanager isn't running — firing alerts don't reach a human yet. Check
  `curl localhost:9090/api/v1/rules` for current alert state before assuming
  nothing is instrumented here.
- `releases/*.json` (written by `release_record.py`, Phase 6) already has a
  real example of the "AI-managed release traceability" demo from
  `VISION.md` succeeding — see `guardian-defect-demo-2026-07-17.json`.
