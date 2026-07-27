# AIOps Guardian

A locally deployed AIOps platform: real-time system/security telemetry, ML-based
anomaly detection, health/security/AI-risk scoring, and a Behavioral Attestation
engine that attributes *why* anomalies happen and verifies runtime behavior
against declared expectations — with dashboards, logs, tracing, and Slack
alerting throughout.

## Where to look

This README is intentionally short — it's a map, not the manual. Detail lives in:

- **[`OPERATIONS_MANUAL.md`](OPERATIONS_MANUAL.md)** — the full operations and architecture reference: every service, script, dashboard, runbook procedure, and known gap. Start here for anything operational.
- **[`CLAUDE.md`](CLAUDE.md)** — the terse, code-first version of the above, written for an AI coding assistant working in this repo.
- **[`VISION.md`](VISION.md)** — the long-term architectural direction (the five-engine model, the north-star question this project is organized around).
- **[`ROADMAP.md`](ROADMAP.md)** — the phased, mutable execution plan for the Behavioral Attestation engine, plus honest gap-tracking notes.
- **[`EDGE_ARCHITECTURE.md`](EDGE_ARCHITECTURE.md)** — future-state plan for deploying Guardian to customer sites (planning only, not built).

## Stack (high level)

Prometheus + Grafana (metrics/dashboards) · Loki + Promtail (logs) · Tempo (traces) · Alertmanager → Slack (alerts) · Python (watchdogs, ML models, Behavioral Attestation modules) · systemd (service management)

Three ML models (KNN, Isolation Forest, Autoencoder) run as live watchdog services alongside health/security/AI-risk scoring, Windows-fleet monitoring, log-anomaly detection, and cross-source priority triage — see `OPERATIONS_MANUAL.md` Chapter 3 for the full service inventory.
