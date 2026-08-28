# Guardian TODO

Running list of open work items. Unlike `ROADMAP.md` (Behavioral Attestation
phases) or `EDGE_ARCHITECTURE.md` (Pi/edge milestones), this is a flat,
general backlog — anything goes here regardless of which part of the project
it belongs to.

- [ ] **Core pipeline self-monitoring (transport-health layer).** Scoped
  2026-08-28, not started. See VISION.md's "Pipeline health model" —
  endpoint health → edge health → **transport health** → ingestion health →
  analysis health. This item covers transport health only; endpoint and
  ingestion health remain separate, un-scoped gaps.
  Prompted by a real incident: the Pi's `remote_write` to Core was silently
  broken for ~12 days (stale IP after a DHCP change), caught by accident.
  Plan:
  1. Add an `EdgePipelineStale` alert rule to
     `/etc/prometheus/rules/aiops-alerts.yml` —
     `absent_over_time(up{edge_site="guardian-proto-1"}[30m])`.
  2. Install Alertmanager (systemd service matching Prometheus's pattern),
     wire it into `prometheus.yml`, firewall-scope port 9093 like the
     watchdog ports.
  Existing gap this also fixes: Core already has sustained-anomaly alert
  rules for the three watchdogs, but Alertmanager isn't running, so none of
  them currently reach anyone.
  **Blocked on:** which notification channel? (email needs an SMTP relay;
  ntfy.sh/Pushover are simpler, no account needed for ntfy.sh; or just
  Alertmanager's own web UI, checked manually.) Decide before building step 2.
